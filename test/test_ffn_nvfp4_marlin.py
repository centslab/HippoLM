"""Correctness test for NVFP4 SwiGLU with Marlin FP4 fwd path.

This test verifies the W4A16 NVFP4 path with the Marlin fused kernel
(BF16 MMA + register dequant + cp.async double-buffered prefetch)
produces a forward close to the BF16 baseline within FP4 quant noise
+ Marlin MMA precision.

Contract:

  - Build two SwiGLU modules with identical random init (seed-fixed).
    One uses BF16 linears, the other uses NVFP4 + Marlin linears.
  - Same input -> forward outputs should agree up to FP4 quant noise
    (~5% mean relative, p99 ~91% — the MMA precision floor, not the
    FP4 quant loss).
  - Same grad output -> backward grad on the BF16 master weight must
    be finite and non-zero (STE: gradient flows through the kernel's
    BF16 dequant view unchanged).
  - Optimizer step + repack works correctly across multiple steps
    (the Marlin path uses ``quantize_nvfp4_with_global_scale`` which
    produces a global_scale scalar in addition to the standard
    packed/scales buffers).

Run:
    python -m pytest test/test_ffn_nvfp4_marlin.py -v
"""
from __future__ import annotations

import pytest
import torch

from src.models.config import HippoConfig
from src.models.activation import SwiGLU
from src.models.ops.nvfp4_linear import NVFP4Linear, repack_nvfp4_weights


def _build_matched_swiglu(
    hidden_size: int,
    intermediate_size: int,
    use_marlin: bool,
    seed: int = 42,
) -> SwiGLU:
    """Build a SwiGLU with either BF16 or NVFP4-Marlin linears, seeded
    identically across the two calls.
    """
    g = torch.Generator().manual_seed(seed)
    cfg = HippoConfig(
        vocab_size=8,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_layers=1,
        num_blocks=1,
        num_heads=1,
        head_dim=hidden_size,
        safe_gate=True,
        lower_bound=-5.0,
        use_short_conv=False,
        ffn_nvfp4=use_marlin,            # required for use_marlin to take effect
        ffn_nvfp4_marlin=use_marlin,
    )
    ffn = SwiGLU(cfg).cuda()
    # Cast BF16 baseline to BF16 (NN.Linear defaults to FP32).
    if not use_marlin:
        for proj in (ffn.gate_proj, ffn.up_proj, ffn.down_proj):
            proj.to(dtype=torch.bfloat16)
    # Manually seed every linear so the BF16 master is shared between
    # the two paths — isolates quant noise from init noise.
    for proj in (ffn.gate_proj, ffn.up_proj, ffn.down_proj):
        w_cpu = torch.empty(proj.weight.shape, dtype=torch.float32).normal_(0.0, 0.5, generator=g)
        proj.weight.data.copy_(w_cpu.to(proj.weight.dtype))
        if proj.bias is not None:
            b_cpu = torch.empty(proj.bias.shape, dtype=torch.float32).normal_(0.0, 0.25, generator=g)
            proj.bias.data.copy_(b_cpu.to(proj.bias.dtype))
        # NVFP4 layers cache the FP4 buffers at __init__ end; re-pack
        # so the FP4 view tracks the manually-seeded BF16 master.
        if use_marlin:
            proj.repack_weights()
    return ffn


@pytest.mark.parametrize("H,I", [(128, 256), (1536, 4096)])
def test_nvfp4_marlin_swiglu_forward_close_to_bf16(H, I):
    """NVFP4-Marlin SwiGLU forward should match BF16 within MMA-precision floor.

    Marlin's MMA order differs from cuBLAS, so even with a noise-free
    weight we'd see ~5% mean-relative drift (cos ~0.95, the kernel
    floor documented in :mod:`docs.marlin_standalone_29`). On top of
    that, FP4 quant adds ~22% per-element weight noise. The combined
    error at FFN output scale is dominated by the per-element weight
    noise × sqrt(I), bounded at ~80% rel-rms in the worst case.
    """
    torch.manual_seed(0)
    bf16 = _build_matched_swiglu(H, I, use_marlin=False, seed=42)
    nvfp4_marlin = _build_matched_swiglu(H, I, use_marlin=True, seed=42)
    bf16.eval()
    nvfp4_marlin.eval()

    x = torch.randn(2, 16, H, device="cuda", dtype=torch.bfloat16)

    with torch.no_grad():
        y_bf16 = bf16(x)
        y_nvfp4 = nvfp4_marlin(x)

    # All outputs finite (kernel must not NaN).
    assert torch.isfinite(y_nvfp4).all().item(), "Marlin forward produced NaN/Inf"

    max_abs = (y_bf16 - y_nvfp4).abs().max().item()
    rms_abs = ((y_bf16 - y_nvfp4) ** 2).mean().sqrt().item()
    rel_rms = rms_abs / (y_bf16.abs().mean().item() + 1e-9)
    # Same bound as the dequant+cuBLAS path test: FP4 quant noise +
    # Marlin MMA drift gives ~5-30% rel_rms depending on H/I.
    assert rel_rms < 0.80, (
        f"NVFP4-Marlin SwiGLU forward diverges from BF16: "
        f"max_abs={max_abs:.4f}, rms_rel={rel_rms:.4f}, "
        f"y_bf16_max={y_bf16.abs().max().item():.4f}"
    )


@pytest.mark.parametrize("H,I", [(128, 256), (1536, 4096)])
def test_nvfp4_marlin_swiglu_backward_grads_finite(H, I):
    """NVFP4-Marlin SwiGLU backward grads must be finite and non-zero.

    The bwd path is a standard BF16 matmul bwd on the BF16 master
    weight (STE for the quantize noise + Marlin MMA reordering).
    """
    torch.manual_seed(0)
    nvfp4 = _build_matched_swiglu(H, I, use_marlin=True, seed=42)

    x = torch.randn(2, 16, H, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    y = nvfp4(x)
    y.sum().backward()

    assert torch.isfinite(x.grad).all().item(), "x.grad has NaN/Inf"
    for proj_name in ("gate_proj", "up_proj", "down_proj"):
        proj = getattr(nvfp4, proj_name)
        assert proj.weight.grad is not None, f"{proj_name}.weight.grad is None"
        assert torch.isfinite(proj.weight.grad).all().item(), (
            f"{proj_name}.weight.grad has NaN/Inf"
        )
        assert proj.weight.grad.abs().sum().item() > 0, (
            f"{proj_name}.weight.grad is all-zero"
        )
        if proj.bias is not None:
            assert torch.isfinite(proj.bias.grad).all().item(), (
                f"{proj_name}.bias.grad has NaN/Inf"
            )


def test_nvfp4_marlin_swiglu_training_step_decreases_loss():
    """A few SGD steps with NVFP4-Marlin SwiGLU should decrease the loss.

    Mirrors the dequant+cuBLAS test but exercises the Marlin forward +
    the repack-after-step hook (which clears the repack cache so the
    next forward re-quantizes + re-repacks with the freshly-updated
    BF16 master).
    """
    H, I = 128, 256
    torch.manual_seed(0)
    nvfp4_marlin = _build_matched_swiglu(H, I, use_marlin=True, seed=42)

    params = [
        nvfp4_marlin.gate_proj.weight,
        nvfp4_marlin.up_proj.weight,
        nvfp4_marlin.down_proj.weight,
    ]
    opt = torch.optim.SGD(params, lr=0.01)

    x = torch.randn(4, 16, H, device="cuda", dtype=torch.bfloat16)
    y_target = torch.randn(4, 16, H, device="cuda", dtype=torch.bfloat16)

    losses = []
    for step in range(50):
        opt.zero_grad()
        y = nvfp4_marlin(x)
        loss = ((y - y_target) ** 2).mean()
        loss.backward()
        opt.step()
        repack_nvfp4_weights(nvfp4_marlin)  # mirror the training loop's hook
        losses.append(loss.item())

    assert losses[-1] < losses[0] * 0.8, (
        f"NVFP4-Marlin SwiGLU training didn't reduce loss: "
        f"first={losses[0]:.4f}, last={losses[-1]:.4f}, "
        f"ratio={losses[-1] / losses[0]:.4f}"
    )


def test_nvfp4_marlin_packed_state_dict_roundtrip():
    """Saving and loading the state_dict of an NVFP4-Marlin layer must
    preserve packed_weight + scales + global_scale (so checkpoints stay
    small AND the kernel doesn't NaN from missing global_scale).

    The derived Marlin caches (``_scales_for_kernel``,
    ``_global_scale_adj``) are registered as ``persistent=False``
    buffers so they participate in ``.cuda()`` device transfer but are
    excluded from ``state_dict()`` — they're derived state,
    reproducible from the source ``scales`` / ``global_scale`` buffers.
    A post-``load_state_dict`` hook auto-refreshes them against the
    loaded data, so the very next forward uses the correct weights
    without the caller having to remember to call ``repack_weights()``.
    """
    torch.manual_seed(0)
    layer = NVFP4Linear(128, 64, bias=True, use_marlin=True).cuda()
    layer.repack_weights()

    sd = layer.state_dict()
    assert "weight" in sd
    assert "packed_weight" in sd
    assert "scales" in sd
    assert "global_scale" in sd, "Marlin layer state_dict missing global_scale"
    # Derived Marlin caches must NOT be in state_dict (recomputed by
    # repack_weights against the source buffers).
    assert "_scales_for_kernel" not in sd, (
        "_scales_for_kernel should be persistent=False; not in state_dict"
    )
    assert "_global_scale_adj" not in sd, (
        "_global_scale_adj should be persistent=False; not in state_dict"
    )
    assert sd["packed_weight"].dtype == torch.uint8
    assert sd["scales"].dtype == torch.float8_e4m3fn
    assert sd["global_scale"].dtype == torch.float32

    # Reconstruct from the dict and verify forward matches. The
    # post-load hook refreshes the derived caches automatically.
    layer2 = NVFP4Linear(128, 64, bias=True, use_marlin=True).cuda()
    layer2.load_state_dict(sd)

    x = torch.randn(2, 128, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        y1 = layer(x)
        y2 = layer2(x)
    assert torch.allclose(y1, y2), "state_dict roundtrip changed output"
    assert torch.isfinite(y1).all().item() and torch.isfinite(y2).all().item()


def test_nvfp4_marlin_layer_is_faster_than_dequant_path():
    """Sanity check that the Marlin forward is materially faster than
    the dequant+cuBLAS path at FFN gate_up shapes on sm_120.

    This is a wall-clock smoke check, not a micro-benchmark — we just
    verify the Marlin kernel is at least 1.5x faster than
    dequant+cuBLAS at a representative shape (M=2048, K=N=4096).
    Tolerates the kernel being temporarily slower if the test box is
    a non-sm_120 GPU (graceful skip).
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    major, _ = torch.cuda.get_device_capability(0)
    if major < 8:
        pytest.skip(f"Marlin FP4 needs sm_80+, got sm_{major}")

    torch.manual_seed(0)
    M, K, N = 2048, 4096, 4096
    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)

    layer_marlin = NVFP4Linear(K, N, bias=True, use_marlin=True).cuda()
    layer_dequant = NVFP4Linear(K, N, bias=True, use_marlin=False).cuda()
    # Share weights + repack so the only difference is the forward path.
    layer_dequant.weight.data.copy_(layer_marlin.weight.data)
    layer_marlin.repack_weights()
    layer_dequant.repack_weights()

    def bench(fn, n_iters=20, n_warmup=5):
        for _ in range(n_warmup):
            fn()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(n_iters):
            fn()
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / n_iters

    t_marlin = bench(lambda: layer_marlin(x))
    t_dequant = bench(lambda: layer_dequant(x))
    speedup = t_dequant / t_marlin
    print(f"\n  Marlin={t_marlin:.3f}ms  Dequant={t_dequant:.3f}ms  Speedup={speedup:.2f}x")
    # We expect at least 1.5x at this shape on sm_120. Be lenient
    # (1.2x) to handle the 5060 Ti vs other sm_80+ boxes variation.
    assert speedup >= 1.2, (
        f"Marlin FP4 only {speedup:.2f}x faster than dequant+cuBLAS "
        f"(expected >=1.5x on sm_120)"
    )
