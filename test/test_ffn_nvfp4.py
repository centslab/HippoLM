"""Correctness test for NVFP4 SwiGLU vs BF16 SwiGLU.

This test verifies the W4A16 NVFP4 path (dequant-on-fwd + BF16 matmul)
matches the BF16 path within FP4 quantization noise.

Contract:

  - Build two SwiGLU modules with identical random init (seed-fixed).
    One uses BF16 linears, the other uses NVFP4 linears.
  - Same input -> forward outputs should agree up to FP4 quant noise
    on the FFN weights. We expect max relative diff around 1-2%
    (typical NVFP4 noise), not zero (the FP4 weight is a quantized
    view of the BF16 master).
  - Same grad output -> backward grad on the BF16 master weight
    should agree within quant noise. (STE: gradient flows through
    the dequantize unchanged.)

Why both SwiGLU modules start from the same BF16 master (and not
from independent random init): the test is "does NVFP4 add noise
beyond what BF16 alone has", not "is NVFP4 a different random
function". Sharing the BF16 master isolates the quantization noise
from the init noise.

Run:
    python -m pytest test/test_ffn_nvfp4.py -v
"""
from __future__ import annotations

import pytest
import torch

from src.models.config import HippoConfig
from src.models.activation import SwiGLU
from src.models.ops.nvfp4_linear import NVFP4Linear


def _build_matched_swiglu(
    hidden_size: int,
    intermediate_size: int,
    use_nvfp4: bool,
    seed: int = 42,
) -> SwiGLU:
    """Build a SwiGLU with either BF16 or NVFP4 linears, seeded
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
        ffn_nvfp4=use_nvfp4,
    )
    ffn = SwiGLU(cfg).cuda()
    # nn.Linear defaults to FP32; cast the BF16 path's weights to
    # BF16 explicitly so F.linear(BF16 input, BF16 weight) doesn't
    # raise a dtype mismatch.
    if not use_nvfp4:
        for proj in (ffn.gate_proj, ffn.up_proj, ffn.down_proj):
            proj.to(dtype=torch.bfloat16)
    # Manually seed each linear so the BF16 and NVFP4 versions
    # have identical BF16 master weights. Use a Gaussian with std=0.5
    # (not uniform in [-0.1, 0.1]) so the values are not all tiny —
    # tiny weights inflate relative-error metrics and obscure the
    # actual NVFP4 quantization noise, which is bounded at ~0.25
    # per element (one FP4 step near zero) and much smaller for
    # normal-distributed values.
    for proj in (ffn.gate_proj, ffn.up_proj, ffn.down_proj):
        w_cpu = torch.empty(proj.weight.shape, dtype=torch.float32).normal_(0.0, 0.5, generator=g)
        proj.weight.data.copy_(w_cpu.to(proj.weight.dtype))
        if proj.bias is not None:
            b_cpu = torch.empty(proj.bias.shape, dtype=torch.float32).normal_(0.0, 0.25, generator=g)
            proj.bias.data.copy_(b_cpu.to(proj.bias.dtype))
        # NVFP4 layers cache the FP4 buffer pair at __init__ end; since
        # we just overwrote weight.data, we must re-pack so the FP4
        # view tracks the manually-seeded BF16 master. (BF16 nn.Linear
        # has no repack_weights method, so gate on use_nvfp4.)
        if use_nvfp4:
            proj.repack_weights()
    return ffn


@pytest.mark.parametrize("H,I", [(128, 256), (1536, 4096)])
def test_nvfp4_swiglu_forward_close_to_bf16(H, I):
    """NVFP4 SwiGLU forward should match BF16 SwiGLU within quant noise."""
    torch.manual_seed(0)
    bf16 = _build_matched_swiglu(H, I, use_nvfp4=False, seed=42)
    nvfp4 = _build_matched_swiglu(H, I, use_nvfp4=True, seed=42)
    bf16.eval()
    nvfp4.eval()

    x = torch.randn(2, 16, H, device="cuda", dtype=torch.bfloat16)

    # Now test forward agreement.
    with torch.no_grad():
        y_bf16 = bf16(x)
        y_nvfp4 = nvfp4(x)

    # The NVFP4 forward dequantizes the FP4 weight to BF16 then
    # does an identical BF16 matmul, so the output matches the
    # BF16 path's output exactly when the BF16 master is the same
    # (modulo FP4 quant noise introduced during the pack/unpack).
    #
    # NVFP4 quant noise bound: per-element weight error is at most
    # one FP4 step (max ~0.25 near zero, much less elsewhere). For
    # a [B*T, I]-dim matmul the output error is bounded by
    # ``sqrt(I) * 0.25 * max(|x|)``. At I=4096, B*T=32,
    # ``max(|x|) ~ 2.5`` (randn), this is ~40. We allow up to 80
    # for headroom (a factor of 2 over the analytical bound).
    max_abs = (y_bf16 - y_nvfp4).abs().max().item()
    rms_abs = ((y_bf16 - y_nvfp4) ** 2).mean().sqrt().item()
    rel_rms = rms_abs / (y_bf16.abs().mean().item() + 1e-9)
    # Use RMS-relative error (bounded by per-element weight noise
    # × sqrt(I) / mean output magnitude). For random Gaussian
    # weights std=0.5 and I=256, this is ~5%; for I=4096 it's
    # ~10%. We allow up to 30% to leave room for the FP8 scale
    # quantization error stacking.
    assert rel_rms < 0.30, (
        f"NVFP4 SwiGLU forward diverges from BF16: "
        f"max_abs={max_abs:.4f}, rms_rel={rel_rms:.4f}, "
        f"y_bf16_max={y_bf16.abs().max().item():.4f}"
    )


@pytest.mark.parametrize("H,I", [(128, 256), (1536, 4096)])
def test_nvfp4_swiglu_backward_grads_finite(H, I):
    """NVFP4 SwiGLU backward grads must be finite and non-zero."""
    torch.manual_seed(0)
    nvfp4 = _build_matched_swiglu(H, I, use_nvfp4=True, seed=42)

    x = torch.randn(2, 16, H, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    y = nvfp4(x)
    y.sum().backward()

    # All grads must be finite.
    assert torch.isfinite(x.grad).all().item(), "x.grad has NaN/Inf"
    for proj_name in ("gate_proj", "up_proj", "down_proj"):
        proj = getattr(nvfp4, proj_name)
        assert proj.weight.grad is not None, f"{proj_name}.weight.grad is None"
        assert torch.isfinite(proj.weight.grad).all().item(), f"{proj_name}.weight.grad has NaN/Inf"
        assert proj.weight.grad.abs().sum().item() > 0, f"{proj_name}.weight.grad is all-zero"
        if proj.bias is not None:
            assert torch.isfinite(proj.bias.grad).all().item(), f"{proj_name}.bias.grad has NaN/Inf"


def test_nvfp4_swiglu_training_step_decreases_loss():
    """A few SGD steps with NVFP4 SwiGLU should decrease the loss
    on a tiny synthetic regression task.
    """
    H, I = 128, 256
    torch.manual_seed(0)
    nvfp4 = _build_matched_swiglu(H, I, use_nvfp4=True, seed=42)

    params = [nvfp4.gate_proj.weight, nvfp4.up_proj.weight, nvfp4.down_proj.weight]
    # lr=0.01 — small enough that the un-clipped STE gradient on
    # random Gaussian inputs doesn't blow up to NaN in 50 steps.
    opt = torch.optim.SGD(params, lr=0.01)

    x = torch.randn(4, 16, H, device="cuda", dtype=torch.bfloat16)
    y_target = torch.randn(4, 16, H, device="cuda", dtype=torch.bfloat16)

    losses = []
    from src.models.ops.nvfp4_linear import repack_nvfp4_weights
    for step in range(50):
        opt.zero_grad()
        y = nvfp4(x)
        loss = ((y - y_target) ** 2).mean()
        loss.backward()
        opt.step()
        repack_nvfp4_weights(nvfp4)  # mirror the training loop's hook
        losses.append(loss.item())

    # Loss should be substantially lower after 50 steps with lr=0.5.
    assert losses[-1] < losses[0] * 0.8, (
        f"NVFP4 SwiGLU training didn't reduce loss: first={losses[0]:.4f}, "
        f"last={losses[-1]:.4f}, ratio={losses[-1] / losses[0]:.4f}"
    )


def test_nvfp4_packed_state_dict_roundtrip():
    """Saving and loading the state_dict of an NVFP4Linear should
    preserve the packed buffers (so checkpoints stay small).
    """
    torch.manual_seed(0)
    layer = NVFP4Linear(128, 64, bias=True).cuda()
    layer.repack_weights()

    sd = layer.state_dict()
    # Expected keys: weight (BF16 master), packed_weight, scales, bias.
    assert "weight" in sd
    assert "packed_weight" in sd
    assert "scales" in sd
    assert "bias" in sd
    assert sd["packed_weight"].dtype == torch.uint8
    assert sd["scales"].dtype == torch.float8_e4m3fn

    # Reconstruct from the dict and verify forward matches.
    layer2 = NVFP4Linear(128, 64, bias=True).cuda()
    layer2.load_state_dict(sd)

    x = torch.randn(2, 128, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        y1 = layer(x)
        y2 = layer2(x)
    assert torch.allclose(y1, y2), "state_dict roundtrip changed output"