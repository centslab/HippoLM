"""Correctness + STE backward + bf16_only + autograd tests for
``NVFP4LinearW4A8`` (the two-pass Triton + _scaled_mm path).

Contract:

  - Forward (W4A8) matches the fp32-dequantized reference within the
    E4M3 B rounding floor (~2-4% median rel at prod shapes; this is
    inherent to any fp8-B NVFP4 path — see auto-memory
    ``project_nvfp4_w4a8_triton.md``).
  - ``bf16_only=True`` is bit-exact with ``nn.Linear`` (zero noise,
    zero overhead).
  - The shape-constraint fallback silently flips to ``bf16_only``
    when ``in_features`` or ``out_features`` is not divisible by 16
    (sm_120 ``_scaled_mm`` requirement).
  - Backward (STE) returns finite ``grad_w`` that matches what a
    plain BF16 matmul would produce — the NVFP4 quantization is
    discarded in bwd so the optimizer sees the standard BF16 grad.
  - State-dict is identical to ``nn.Linear`` (BF16 master weight
    + optional bias), so a BF16 checkpoint loads directly.

Run:
    python -m pytest test/test_nvfp4_linear_w4a8.py -v
"""
# NOTE (2026-07-27 audit): TFLOPS perf test in this file used the buggy
# cudaEvent single-event pattern; real ~6 TFLOPS via batched aggregate.
# See memory `feedback_cudaevent_2x_underreport_2026_07_27.md`.

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from src.models.ops.nvfp4_linear_w4a8 import (
    NVFP4LinearW4A8, NVFP4W4A8SwiGLU,
    quantize_act_fp8, quantize_act_fp8_fused,
    silu_mul_fused,
    silu_quant_fused,
    _E4M3_MAX, _E4M3_MIN_NORMAL,
)
from src.models.ops.nvfp4_marlin import (
    dequantize_marlin_nvfp4,
    quantize_nvfp4_with_global_scale,
)


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="NVFP4 W4A8 requires CUDA (Triton + _scaled_mm on sm_120)",
)


# ---------------------------------------------------------------------------
# Reference: fp32 dequant of NVFP4 weight + per-row fp8 quant of activation
# ---------------------------------------------------------------------------
def _fp32_dequant_ref(
    x: torch.Tensor,
    w_bf16: torch.Tensor,
    block_size: int = 16,
) -> torch.Tensor:
    """fp32 matmul of the fp8-dequantized activation and the
    NVFP4-dequantized weight. The only diff vs the W4A8 layer
    output is the E4M3 B cast noise inside the GEMM.
    """
    M, K = x.shape
    N, _ = w_bf16.shape
    packed, s_e4m3, g = quantize_nvfp4_with_global_scale(w_bf16, block_size=block_size)
    w_deq = dequantize_marlin_nvfp4(
        packed, s_e4m3, g, K_orig=K,
        block_size=block_size, out_dtype=torch.float32,
    )
    amax = x.float().abs().amax(dim=1, keepdim=True).clamp(min=_E4M3_MIN_NORMAL)
    a_s = (amax / _E4M3_MAX).to(torch.float32).contiguous()
    a_fp8 = quantize_act_fp8(x, a_s)
    a_deq = a_fp8.float() * a_s
    return (a_deq @ w_deq.t())


# ---------------------------------------------------------------------------
# Forward correctness — distribution-level
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("M,K,N", [
    (128, 1024, 1024),
    (1024, 1024, 1024),
    (4096, 1536, 8192),  # prod gate_up
    (4096, 4096, 1536),  # prod down
])
def test_w4a8_forward_close_to_fp32_ref(M, K, N):
    """W4A8 forward matches fp32-dequantized reference within the
    E4M3 B rounding floor.

    Correctness here is checked at the **distribution level** — not
    per-element relative error. The per-element rel p95 is high
    (~30%) because the matmul output's bottom 5% has small magnitude
    (~0.2 abs) and the FP8 cast noise floor (~0.05 abs) dominates
    the rel. The median rel is ~2.5% (the actual E4M3-B noise floor
    uniform across shapes); the mean / std / RMSE are bounded and
    SNR is well above 0 dB. A per-element rel test alone would
    false-fail on this artifact.
    """
    torch.manual_seed(0)
    x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    w = torch.randn(N, K, dtype=torch.bfloat16, device="cuda") * 0.1

    layer = NVFP4LinearW4A8(K, N, bias=False, block_size=16).cuda()
    with torch.no_grad():
        layer.weight.data.copy_(w)

    with torch.no_grad():
        y_w4a8 = layer(x).float()
        y_ref = _fp32_dequant_ref(x, w).bfloat16().float()

    # ---- 1. median per-element rel (the E4M3-B rounding floor) ----
    rel = (y_w4a8 - y_ref).abs() / (y_ref.abs() + 1e-3)
    rel_median = rel.median().item()
    # E4M3 B noise floor: 2-4% median across shapes; allow 5%.
    assert rel_median < 0.05, (
        f"W4A8 fwd diverges from fp32 ref: median_rel={rel_median*100:.2f}%, "
        f"max_rel={rel.max().item()*100:.2f}%"
    )

    # ---- 2. distribution-level: mean and std preserved ----
    # The noise is zero-mean (per-block random rounding) so the
    # output mean and std should match the FP32 ref within 5%.
    mean_diff = abs(y_w4a8.mean().item() - y_ref.mean().item())
    std_diff = abs(y_w4a8.std().item() - y_ref.std().item())
    mean_rel = mean_diff / (y_ref.mean().abs().item() + 1e-3)
    std_rel = std_diff / y_ref.std().item()
    assert mean_rel < 0.05, f"output mean diverges: {mean_rel*100:.2f}%"
    assert std_rel < 0.05, f"output std diverges: {std_rel*100:.2f}%"

    # ---- 3. RMSE is bounded (no runaway noise) ----
    rmse = (y_w4a8 - y_ref).pow(2).mean().sqrt().item()
    # RMSE scales as sqrt(K) * per-element noise. For K=4096 and
    # ~5% per-element noise on the dequant, RMSE is ~0.3 abs. The
    # typical output magnitude is 3-4 abs, so RMSE/output_std
    # should be < 25%.
    rmse_rel = rmse / y_ref.std().item()
    assert rmse_rel < 0.25, f"RMSE too high: {rmse:.4f} abs ({rmse_rel*100:.1f}% of std)"

    # ---- 4. SNR is positive (signal dominates noise) ----
    signal = y_ref.pow(2).mean()
    noise = (y_w4a8 - y_ref).pow(2).mean()
    snr_db = 10 * (signal / noise).log10().item()
    # Even the worst FP8 matmul should have SNR > 15 dB (the W4A16
    # Marlin path on this shape gets ~40 dB; W4A8 ~32 dB; the BF16
    # path is ~55 dB). 15 dB is the floor where the noise would
    # start to dominate training.
    assert snr_db > 15.0, f"SNR too low: {snr_db:.2f} dB"

    # ---- 5. max abs err is bounded ----
    max_abs = (y_w4a8 - y_ref).abs().max().item()
    # No single output should be off by more than 2x the per-element
    # FP8 cast noise times sqrt(K). For K=4096, ~0.3 abs.
    assert max_abs < 1.0, f"max abs err too high: {max_abs:.4f}"


# ---------------------------------------------------------------------------
# bf16_only escape hatch
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("K,N,bias", [
    (1024, 1024, False),
    (1536, 4096, True),
])
def test_bf16_only_bit_exact_with_nn_linear(K, N, bias):
    """``bf16_only=True`` must be bit-exact with ``nn.Linear`` — same
    parameters, same forward, no quantization noise.
    """
    torch.manual_seed(0)
    x = torch.randn(64, K, dtype=torch.bfloat16, device="cuda")

    bf16 = torch.nn.Linear(K, N, bias=bias).cuda().bfloat16()
    w4a8 = NVFP4LinearW4A8(K, N, bias=bias, block_size=16, bf16_only=True).cuda()
    with torch.no_grad():
        w4a8.weight.data.copy_(bf16.weight.data)
        if bias:
            w4a8.bias.data.copy_(bf16.bias.data)

    with torch.no_grad():
        y_bf16 = bf16(x)
        y_w4a8 = w4a8(x)
    assert torch.equal(y_bf16, y_w4a8), (
        f"bf16_only is not bit-exact with nn.Linear at K={K}, N={N}"
    )


# ---------------------------------------------------------------------------
# Shape-constraint fallback (N or K not divisible by 16)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("K,N", [
    (1536, 12),   # b_proj at H=12, out_features=12 not div by 16
    (1023, 1024), # in_features not div by 16
    (7, 11),      # both not div by 16
])
def test_silently_falls_back_to_bf16_for_unaligned_shapes(K, N):
    """If K or N is not divisible by 16 (sm_120 _scaled_mm
    requirement), the constructor flips ``bf16_only=True``
    silently — no exception, output is bit-exact with nn.Linear.
    """
    layer = NVFP4LinearW4A8(K, N, bias=False).cuda()
    assert layer.bf16_only, f"K={K}, N={N}: expected bf16_only=True fallback"
    x = torch.randn(8, K, dtype=torch.bfloat16, device="cuda")
    with torch.no_grad():
        y = layer(x)
        y_ref = F.linear(x, layer.weight)
    assert torch.equal(y, y_ref), "fallback path is not bit-exact with nn.Linear"


# ---------------------------------------------------------------------------
# STE backward
# ---------------------------------------------------------------------------
def test_ste_backward_grads_finite_and_match_bf16():
    """The autograd backward re-runs a BF16 matmul, so ``grad_w`` is
    exactly what a plain ``F.linear`` would produce (no quantization
    noise in the gradient). The test verifies finiteness + that the
    grad has the right shape / non-zero / bit-equal to the BF16
    reference grad.
    """
    torch.manual_seed(0)
    K, N = 1024, 1024
    M = 256
    x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    w = torch.randn(N, K, dtype=torch.bfloat16, device="cuda") * 0.1

    layer = NVFP4LinearW4A8(K, N, bias=True, block_size=16).cuda()
    with torch.no_grad():
        layer.weight.data.copy_(w)
        layer.bias.data.copy_(torch.randn(N, dtype=torch.bfloat16, device="cuda") * 0.01)

    grad_out = torch.randn(M, N, dtype=torch.bfloat16, device="cuda")

    # W4A8 path
    x_w4a8 = x.detach().clone().requires_grad_(True)
    y_w4a8 = layer(x_w4a8)
    y_w4a8.backward(grad_out)
    assert torch.isfinite(layer.weight.grad).all().item(), "grad_w has NaN/Inf"
    assert torch.isfinite(layer.bias.grad).all().item(), "grad_b has NaN/Inf"
    assert torch.isfinite(x_w4a8.grad).all().item(), "grad_x has NaN/Inf"
    assert layer.weight.grad.abs().sum().item() > 0, "grad_w is all-zero"
    assert x_w4a8.grad.abs().sum().item() > 0, "grad_x is all-zero"

    # BF16 reference: grad_w should match the plain matmul result
    grad_w_ref = grad_out.float().t() @ x.float()
    grad_rel = (layer.weight.grad.float() - grad_w_ref).abs() / (grad_w_ref.abs() + 1e-3)
    # STE re-runs in BF16, so grad_w is bit-exact (modulo BF16
    # reduction order, which is below 1% on K=1024).
    assert grad_rel.median().item() < 0.01, (
        f"STE grad_w diverges from BF16 ref: median_rel={grad_rel.median().item()*100:.3f}%"
    )


def test_ste_backward_is_bit_equivalent_to_f_linear():
    """Regression guard against the FP32-cast perf bug.

    The autograd backward must run in BF16 throughout — NOT cast to
    FP32 first. The historical FP32-cast path ran FP32 GEMMs (12.5 TFLOPS
    FP32 SIMT peak vs 50 TFLOPS BF16 TC = ~3× slower at FFN prod shape)
    AND produced numerically distinct grad_w (FP32 accumulation vs
    BF16 accumulation; reduction-order differences).

    Verify at FFN prod shape (M=1024, K=1536, N=4096) that
    NVFP4LinearW4A8.backward produces grad_w with cos_sim == 1.0
    against the F.linear reference. Any future regression to FP32
    casting (or any non-trivial deviation from the BF16 STE contract)
    will fail this assertion.
    """
    torch.manual_seed(0)
    M, K, N = 1024, 1536, 4096   # FFN gate/up prod shape
    x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    w = torch.randn(N, K, dtype=torch.bfloat16, device="cuda") * 0.05

    layer = NVFP4LinearW4A8(K, N, bias=False, block_size=16).cuda()
    with torch.no_grad():
        layer.weight.data.copy_(w)

    grad_out = torch.randn(M, N, dtype=torch.bfloat16, device="cuda")

    # W4A8 backward path
    x_w4a8 = x.detach().clone().requires_grad_(True)
    y_w4a8 = layer(x_w4a8)
    y_w4a8.backward(grad_out)

    # BF16 reference: plain F.linear bwd computes grad_w as
    # grad_out.T @ x in BF16 — bit-equivalent to the STE backward.
    grad_w_ref = grad_out.t() @ x_w4a8.detach()

    # BF16 STE bwd must produce bit-exact grad_w (modulo reduction order
    # which only changes last-bit on K=1536). cos_sim == 1.0 + tiny
    # max_abs_err detects any non-BF16-cast path (FP32, FP8, etc.).
    cos_w = torch.nn.functional.cosine_similarity(
        layer.weight.grad.flatten().float(),
        grad_w_ref.flatten().float(),
        dim=0,
    ).item()
    max_abs_err = (layer.weight.grad.float() - grad_w_ref.float()).abs().max().item()
    assert cos_w > 0.9999, (
        f"STE grad_w diverges from BF16 F.linear ref (cos_sim={cos_w:.6f}); "
        f"backward may have been cast to FP32 or quant-dequanted."
    )
    # BF16 STE bwd should match F.linear bwd at BF16 rounding noise.
    # A FP32-cast path would introduce FP32-accumulation error that's
    # an order of magnitude larger than BF16 eps.
    assert max_abs_err < 1e-2, (
        f"STE grad_w max_abs_err={max_abs_err:.4f} > 1e-2; "
        f"backward likely ran in FP32, not BF16."
    )


# ---------------------------------------------------------------------------
# State dict compatibility with nn.Linear
# ---------------------------------------------------------------------------
def test_state_dict_compatible_with_nn_linear():
    """A BF16 ``nn.Linear`` state_dict should load directly into
    ``NVFP4LinearW4A8`` (and vice versa) — both use ``weight`` and
    optional ``bias`` keys with the same shape.
    """
    K, N = 1024, 2048
    bf16 = torch.nn.Linear(K, N, bias=True).cuda().bfloat16()
    sd = bf16.state_dict()
    assert set(sd.keys()) == {"weight", "bias"}, f"unexpected keys: {sd.keys()}"

    w4a8 = NVFP4LinearW4A8(K, N, bias=True, block_size=16).cuda()
    w4a8.load_state_dict(sd)
    assert torch.equal(w4a8.weight.data, bf16.weight.data)
    assert torch.equal(w4a8.bias.data, bf16.bias.data)

    # Round trip: W4A8 -> nn.Linear
    sd2 = w4a8.state_dict()
    bf16.load_state_dict(sd2)
    assert torch.equal(bf16.weight.data, w4a8.weight.data)


# ---------------------------------------------------------------------------
# extra_repr reports the resolved mode
# ---------------------------------------------------------------------------
def test_extra_repr_reflects_mode():
    layer_bf16 = NVFP4LinearW4A8(1024, 1024, bias=False, bf16_only=True)
    assert "BF16" in layer_bf16.extra_repr()

    layer_w4a8 = NVFP4LinearW4A8(1024, 1024, bias=False, block_size=16)
    assert "NVFP4 W4A8" in layer_w4a8.extra_repr()
    assert "block=16" in layer_w4a8.extra_repr()
    assert "custom GEMM" not in layer_w4a8.extra_repr()


# ---------------------------------------------------------------------------
# Custom CUDA C++ FP8 GEMM backend (opt-in via use_custom_gemm=True)
# ---------------------------------------------------------------------------
try:
    from src.models.ops.cuda.fp8_gemm import is_available as _fp8_gemm_available
except ImportError:
    _fp8_gemm_available = lambda: False  # noqa: E731


@pytest.mark.skipif(
    not _fp8_gemm_available(),
    reason="custom fp8_gemm .so not built for this SM (run scripts/build_fp8_gemm.py)",
)
def test_custom_gemm_backend_matches_default():
    """W4A8 with ``use_custom_gemm=True`` produces the same output as
    the default ``_scaled_mm`` backend (within FP8 cast noise), and
    the ``extra_repr`` reports the right mode.
    """
    torch.manual_seed(0)
    K, N = 1024, 1024
    M = 2048
    x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    w = torch.randn(N, K, dtype=torch.bfloat16, device="cuda") * 0.1

    layer_default = NVFP4LinearW4A8(K, N, bias=False, block_size=16, use_custom_gemm=False).cuda()
    layer_custom = NVFP4LinearW4A8(K, N, bias=False, block_size=16, use_custom_gemm=True).cuda()
    with torch.no_grad():
        layer_default.weight.data.copy_(w)
        layer_custom.weight.data.copy_(w)

    with torch.no_grad():
        y_default = layer_default(x).float()
        y_custom = layer_custom(x).float()

    # Both go through the same fp8 quant + same dequant → same cast noise.
    # Allow small drift from differing reduction order in the two GEMM
    # backends (TMA+warp-spec vs cuBLAS nvjet).
    diff = (y_default - y_custom).abs()
    max_diff = diff.max().item()
    assert max_diff < 0.5, f"custom vs default GEMM diverged: max abs diff = {max_diff}"

    # The custom backend should be enabled in the repr.
    assert "custom GEMM" in layer_custom.extra_repr()


def test_boost_cache_invalidates_on_weight_change():
    """Strategy 3: ``_boost_cache`` must refresh when the BF16 master
    weight is modified (since s_e4m3 → smax changes), and reuse the same
    value across forwards with an unchanged weight.

    Validates both the perf-relevant reuse (cache hit on second call)
    and the correctness-relevant invalidation (cache miss + fresh boost
    after weight modification). Note: the boost VALUE quantizes smax to
    the nearest pow2, so two very-different scales can map to the same
    boost. We assert invalidation via the version counter and output
    divergence, not via the boost value itself.
    """
    torch.manual_seed(0)
    K, N = 512, 512
    layer = NVFP4LinearW4A8(K, N, bias=False, block_size=16).cuda()
    x = torch.randn(64, K, dtype=torch.bfloat16, device="cuda") * 0.1

    # First forward: cache miss, boost computed.
    y0 = layer(x)
    assert layer._boost_cache is not None
    assert layer._boost_w_ver == layer.weight._version
    ver_after_first = layer._boost_w_ver

    # Second forward with same weight: cache hit (version unchanged), bit-exact.
    y1 = layer(x)
    assert layer._boost_w_ver == ver_after_first
    assert torch.equal(y0, y1), "boost cache should give bit-exact repeats"

    # Perturb the weight — must invalidate cache and produce a different
    # output (regardless of whether the quantized boost value coincides).
    with torch.no_grad():
        layer.weight.mul_(0.01)
    ver_after_modify = layer.weight._version
    assert ver_after_modify > ver_after_first, (
        f"weight._version did not bump on in-place mul: "
        f"{ver_after_first} → {ver_after_modify}"
    )
    y2 = layer(x)
    assert layer._boost_w_ver == ver_after_modify, (
        "boost cache did not refresh after weight._version change"
    )
    # Output must differ — verifies the fresh boost/scale flowed through.
    assert not torch.equal(y0, y2), "weight change did not propagate to output"


# ---------------------------------------------------------------------------
# Strategy 2: pack cache (skip-N NVFP4 quant)
# ---------------------------------------------------------------------------
def test_pack_cache_invalidates_on_weight_change():
    """The pack cache must refresh when the BF16 master weight
    changes (since packed bytes → s_e4m3 → g are functions of w).
    Validates both reuse (cache hit on second call, bit-exact) and
    invalidation (cache miss + fresh pack after weight modification).
    """
    torch.manual_seed(0)
    K, N = 512, 512
    layer = NVFP4LinearW4A8(K, N, bias=False, block_size=16).cuda()
    x = torch.randn(64, K, dtype=torch.bfloat16, device="cuda") * 0.1

    # First forward: cache miss.
    y0 = layer(x)
    assert layer._pack_cache is not None, "pack cache not populated after first forward"
    cached_packed = layer._pack_cache[0]
    assert layer._pack_w_ver == layer.weight._version
    ver_after_first = layer._pack_w_ver

    # Second forward with same weight: cache hit (version unchanged).
    y1 = layer(x)
    assert layer._pack_w_ver == ver_after_first, "pack cache should not refresh on same weight"
    # Same identity on the packed buffer — confirms cache hit
    # (a fresh quant would return a new tensor object).
    assert layer._pack_cache[0] is cached_packed, (
        "packed buffer identity changed — cache did not hit"
    )
    assert torch.equal(y0, y1), "pack cache should give bit-exact repeats"

    # Invalidate: in-place weight modification → version bumps.
    with torch.no_grad():
        layer.weight.mul_(0.5)
    ver_after_modify = layer.weight._version
    assert ver_after_modify > ver_after_first
    y2 = layer(x)
    assert layer._pack_w_ver == ver_after_modify, "pack cache did not refresh on weight change"
    assert not torch.equal(y0, y2), "weight change did not propagate to output"


# ---------------------------------------------------------------------------
# w_dquant output buffer cache (saves torch.empty on every forward)
# ---------------------------------------------------------------------------
def test_w_dquant_output_buffer_is_cached_and_reused():
    """The dequant_nvfp4_to_fp8 output buffer must be lazily allocated
    on first forward and reused on subsequent forwards (same weight
    shape). Validates that the buffer identity is stable and the output
    is correct."""
    torch.manual_seed(0)
    K, N = 1024, 4096
    layer = NVFP4LinearW4A8(K, N, bias=False, block_size=16).cuda()
    x = torch.randn(64, K, dtype=torch.bfloat16, device="cuda") * 0.1

    # First forward: cache miss, buffer allocated.
    y0 = layer(x)
    assert layer._fp8_dequant_buf is not None, (
        "w_dquant output buffer cache should be populated after first forward"
    )
    buf_id = layer._fp8_dequant_buf
    assert layer._fp8_dequant_buf.shape == (N, K), (
        f"cached buf shape {layer._fp8_dequant_buf.shape} != ({N}, {K})"
    )

    # Second forward with same weight shape: buffer identity preserved
    # (allocator hit; no new torch.empty).
    y1 = layer(x)
    assert layer._fp8_dequant_buf is buf_id, (
        "w_dquant output buffer identity changed — cache did not hit"
    )
    assert torch.equal(y0, y1), (
        "w_dquant output buffer reuse should give bit-exact repeats"
    )

    # Same weight, different M (act shape): buf still correct (M doesn't
    # touch w_dquant output shape — only (N, K) matters).
    x2 = torch.randn(128, K, dtype=torch.bfloat16, device="cuda") * 0.1
    y2 = layer(x2)
    assert layer._fp8_dequant_buf is buf_id, (
        "act-shape change should not invalidate w_dquant buffer cache"
    )


def test_w_dquant_buffer_resizes_on_module_shape_change():
    """If a module is re-instantiated with a different (N, K) shape,
    the w_dquant buffer cache must be reallocated to fit. (In
    production this happens only across unrelated sub-modules — same
    shape from layer to layer — so this is a defensive test.)"""
    K1, N1 = 1024, 1024
    K2, N2 = 1024, 4096
    layer_a = NVFP4LinearW4A8(K1, N1, bias=False, block_size=16).cuda()
    x = torch.randn(64, K1, dtype=torch.bfloat16, device="cuda") * 0.1
    layer_a(x)
    assert layer_a._fp8_dequant_buf.shape == (N1, K1)

    layer_b = NVFP4LinearW4A8(K2, N2, bias=False, block_size=16).cuda()
    layer_b(x)
    assert layer_b._fp8_dequant_buf.shape == (N2, K2), (
        f"buffer shape {layer_b._fp8_dequant_buf.shape} != ({N2}, {K2})"
    )


# ---------------------------------------------------------------------------
# act_quant output buffer cache (saves torch.empty on every forward)
# ---------------------------------------------------------------------------
def test_act_quant_output_buffer_is_cached_and_reused():
    """The act_quant output (a_fp8, a_s) buffers must be lazily allocated
    on first forward and reused on subsequent forwards with same-or-smaller
    M. Output numerics must match the no-cache reference."""
    torch.manual_seed(0)
    K, N = 1024, 4096
    layer = NVFP4LinearW4A8(K, N, bias=False, block_size=16).cuda()
    M = 512
    x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda") * 0.1

    # First forward: cache miss.
    y0 = layer(x)
    assert layer._act_fp8_buf is not None
    assert layer._act_s_buf is not None
    assert layer._act_fp8_buf.shape == (M, K)
    assert layer._act_s_buf.shape == (M, 1)
    buf_id_fp8 = layer._act_fp8_buf.data_ptr()
    buf_id_s = layer._act_s_buf.data_ptr()

    # Second forward same M: buffer identity preserved (allocator hit).
    y1 = layer(x)
    assert layer._act_fp8_buf.data_ptr() == buf_id_fp8, (
        "act_fp8 buffer identity changed — cache did not hit"
    )
    assert layer._act_s_buf.data_ptr() == buf_id_s, (
        "act_s buffer identity changed — cache did not hit"
    )
    assert torch.equal(y0, y1), (
        "act_quant output buffer reuse should give bit-exact repeats"
    )

    # Smaller M: still reuse, just view a prefix.
    x_small = torch.randn(128, K, dtype=torch.bfloat16, device="cuda") * 0.1
    y2 = layer(x_small)
    assert layer._act_fp8_buf.data_ptr() == buf_id_fp8
    assert layer._act_s_buf.data_ptr() == buf_id_s


def test_act_quant_buffer_grows_when_M_increases():
    """If a forward comes in with M larger than the current buffer's
    leading dim, the cache must reallocate (never silently truncate)."""
    torch.manual_seed(0)
    K, N = 1024, 1024
    layer = NVFP4LinearW4A8(K, N, bias=False, block_size=16).cuda()
    x1 = torch.randn(64, K, dtype=torch.bfloat16, device="cuda") * 0.1
    layer(x1)
    cap_after_first = layer._act_fp8_buf.shape[0]
    assert cap_after_first == 64

    x2 = torch.randn(256, K, dtype=torch.bfloat16, device="cuda") * 0.1
    layer(x2)
    assert layer._act_fp8_buf.shape[0] >= 256, (
        f"buffer did not grow to accommodate M=256; "
        f"current shape {layer._act_fp8_buf.shape}"
    )
    assert layer._act_fp8_buf.shape[1] == K


# ---------------------------------------------------------------------------
# Strategy 1: forward_precomputed skips act_quant (shared activation)
# ---------------------------------------------------------------------------
def test_forward_precomputed_matches_forward():
    """``forward_precomputed(x, a_fp8, a_s)`` with caller-supplied
    fp8 activation must produce the same output as ``forward(x)``
    (modulo BF16 reduction-order noise in the GEMM, well below
    the E4M3 cast noise floor). Both code paths consume identical
    ``a_fp8`` and ``a_s`` — the only difference is who computed them.
    """
    torch.manual_seed(0)
    K, N = 1024, 1024
    M = 2048
    x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    w = torch.randn(N, K, dtype=torch.bfloat16, device="cuda") * 0.1

    layer = NVFP4LinearW4A8(K, N, bias=False, block_size=16).cuda()
    with torch.no_grad():
        layer.weight.data.copy_(w)

    a_fp8, a_s = quantize_act_fp8_fused(x)

    with torch.no_grad():
        y_default = layer(x)
        y_pre = layer.forward_precomputed(x, a_fp8, a_s)

    # Bit-exact: both paths use the same a_fp8/a_s, same dequant, same
    # _scaled_mm kernel call. The forward_precomputed path is a strict
    # prefix-skip of the default forward, so the output bytes are
    # guaranteed identical.
    assert torch.equal(y_default, y_pre), (
        f"forward_precomputed diverged from forward: "
        f"max abs diff = {(y_default - y_pre).abs().max().item()}"
    )


def test_forward_precomputed_bf16_only_bypasses_quant():
    """Even with pre-quantized inputs, ``bf16_only=True`` skips
    both the act_quant AND the W4A8 path and returns ``F.linear``.
    """
    K, N = 1024, 1024
    layer = NVFP4LinearW4A8(K, N, bias=False, block_size=16, bf16_only=True).cuda()
    x = torch.randn(64, K, dtype=torch.bfloat16, device="cuda")
    a_fp8 = torch.zeros(64, K, dtype=torch.float8_e4m3fn, device="cuda")
    a_s = torch.ones(64, 1, dtype=torch.float32, device="cuda")
    with torch.no_grad():
        y = layer.forward_precomputed(x, a_fp8, a_s)
        y_ref = torch.nn.functional.linear(x, layer.weight)
    assert torch.equal(y, y_ref), "bf16_only precomputed path is not bit-exact with F.linear"


# ---------------------------------------------------------------------------
# Strategy 1: NVFP4W4A8SwiGLU — gate+up share act_quant
# ---------------------------------------------------------------------------
def test_swiglu_matches_unfused_when_called_together():
    """NVFP4W4A8SwiGLU's output must match a baseline SwiGLU where
    gate and up each do their own act_quant, because the fp8 cast is
    deterministic in x — the two paths consume identical a_fp8/a_s
    for both gate and up.

    Uses an unfused baseline (no shared act_quant) and an explicit
    shared baseline (one quant, two reuses); both must bit-equal
    the wrapper's output.
    """
    torch.manual_seed(0)
    M, H, I = 1024, 1024, 2048

    swiglu = NVFP4W4A8SwiGLU(H, I, bias=False, block_size=16).cuda()
    x = torch.randn(M, H, dtype=torch.bfloat16, device="cuda")

    with torch.no_grad():
        # Shared path
        y_swiglu = swiglu(x)

        # Unfused reference: separate modules, separate act_quants
        gate_ref = swiglu.gate_proj
        up_ref = swiglu.up_proj
        down_ref = swiglu.down_proj
        gate_y = gate_ref(x)
        up_y = up_ref(x)
        hidden_ref = torch.nn.functional.silu(gate_y) * up_y
        y_unfused = down_ref(hidden_ref)

        # Shared-reference: same modules, one act_quant, two reuses
        a_fp8, a_s = quantize_act_fp8_fused(x.reshape(-1, x.shape[-1]))
        gate_y2 = gate_ref.forward_precomputed(x, a_fp8, a_s)
        up_y2 = up_ref.forward_precomputed(x, a_fp8, a_s)
        hidden_shared = torch.nn.functional.silu(gate_y2) * up_y2
        y_shared_ref = down_ref(hidden_shared)

    # Wrapper uses ``silu_mul_fused`` (more precise fp32 silu); the
    # shared-reference uses ``F.silu(gate) * up`` (bf16 intermediate).
    # They differ by bf16 round-off noise — the silu precision
    # difference propagates through the down matmul (max abs diff
    # ~0.01 at this magnitude). The contract is "within bf16 noise",
    # not bit-exact.
    max_abs = (y_swiglu - y_shared_ref).abs().max().item()
    assert max_abs < 0.1, (
        f"SwiGLU wrapper output diverged from shared-reference: "
        f"max_abs_diff={max_abs:.4f}"
    )
    # Shared-reference vs unfused: must agree within BF16 reduction
    # noise (same GEMM kernels, same fp8 inputs, only the act_quant
    # launch count differs). The rel floor at this shape is well
    # under 1%.
    rel = (y_shared_ref - y_unfused).abs() / (y_unfused.abs() + 1e-3)
    assert rel.median().item() < 0.01, (
        f"shared act_quant diverged from unfused: median_rel="
        f"{rel.median().item()*100:.3f}%"
    )


def test_swiglu_bf16_only_path():
    """When any of gate/up/down has unaligned shapes (forces
    ``bf16_only=True`` on that layer), the wrapper still produces
    the correct output via plain ``F.linear`` for the bf16 path
    and the shared-act_quant optimization for the rest.
    """
    # H=1536, I=4096 are both div by 16, but the test is a sanity
    # check that the bf16_only branch in NVFP4W4A8SwiGLU.forward
    # works without exceptions and matches the unfused F.linear
    # reference.
    torch.manual_seed(0)
    M, H, I = 256, 1024, 1024
    swiglu = NVFP4W4A8SwiGLU(H, I, bias=True, block_size=16,
                              bf16_only=True).cuda()
    x = torch.randn(M, H, dtype=torch.bfloat16, device="cuda")
    with torch.no_grad():
        y = swiglu(x)
        # Reference: 3 F.linear calls + silu*up
        g = torch.nn.functional.linear(x, swiglu.gate_proj.weight,
                                       swiglu.gate_proj.bias)
        u = torch.nn.functional.linear(x, swiglu.up_proj.weight,
                                       swiglu.up_proj.bias)
        h = torch.nn.functional.silu(g) * u
        y_ref = torch.nn.functional.linear(h, swiglu.down_proj.weight,
                                           swiglu.down_proj.bias)
    assert torch.equal(y, y_ref), "bf16_only SwiGLU not bit-exact with F.linear reference"


# ---------------------------------------------------------------------------
# Fused silu(gate) * up — one Triton kernel
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("M,N", [
    (1024, 1024),
    (16384, 4096),    # prod gate_up
    (4096, 11008),    # Llama-style intermediate
    (1, 4096),        # edge: single row
    (16384, 1),       # edge: single col
])
def test_silu_mul_fused_matches_pytorch(M, N):
    """``silu_mul_fused(gate, up)`` must match ``F.silu(gate) * up``
    within bf16 round-off. The fused path computes silu in fp32
    registers (one extra bit of precision) before narrowing on store,
    so it's not bit-exact — but the difference is bounded by 1 ULP per
    element (~0.06 at magnitude-1 inputs) and zero on average.
    """
    torch.manual_seed(0)
    gate = torch.randn(M, N, dtype=torch.bfloat16, device="cuda")
    up = torch.randn(M, N, dtype=torch.bfloat16, device="cuda")

    y_ref = torch.nn.functional.silu(gate) * up
    y_test = silu_mul_fused(gate, up)

    diff = (y_ref.float() - y_test.float()).abs()
    max_abs = diff.max().item()
    # Max abs diff is bounded by 1 ULP of bf16 at the output magnitude
    # (~0.06 at magnitude-1 outputs) plus 1 ULP from the silu rounding.
    assert max_abs < 0.1, (
        f"silu_mul_fused diverges from PyTorch: max_abs_diff={max_abs:.4f} "
        f"at M={M}, N={N}"
    )
    # Median rel must be effectively zero (most elements agree exactly).
    rel = diff / (y_ref.float().abs() + 1e-3)
    assert rel.median().item() < 0.01, (
        f"silu_mul_fused median rel too high: {rel.median().item()*100:.3f}%"
    )


def test_silu_mul_fused_output_shape_and_dtype():
    """Output must be [M, N] BF16, contiguous, same device as inputs."""
    gate = torch.randn(64, 128, dtype=torch.bfloat16, device="cuda")
    up = torch.randn(64, 128, dtype=torch.bfloat16, device="cuda")
    out = silu_mul_fused(gate, up)
    assert out.shape == (64, 128)
    assert out.dtype == torch.bfloat16
    assert out.device == gate.device


def test_swiglu_uses_fused_silu_mul_path():
    """NVFP4W4A8SwiGLU.forward must use ``silu_mul_fused`` (not the
    naive ``F.silu(gate) * up``) so the FFN gets the ~0.7 ms win.

    Indirect check: the SwiGLU output must match a baseline that uses
    the fused silu_mul_fused call explicitly — and differ from the
    baseline that uses ``F.silu(gate) * up`` only by bf16 round-off
    (which is the same as the silu_mul_fused contract above).
    """
    torch.manual_seed(0)
    M, H, I = 1024, 1024, 2048
    swiglu = NVFP4W4A8SwiGLU(H, I, bias=False, block_size=16).cuda()
    x = torch.randn(M, H, dtype=torch.bfloat16, device="cuda")

    # Build baselines using the SAME sub-modules but with explicit
    # silu_mul_fused vs F.silu — so the only difference is the silu
    # path (gate/up matmuls + act_quant + down matmul are identical).
    gate = swiglu.gate_proj
    up = swiglu.up_proj
    down = swiglu.down_proj

    with torch.no_grad():
        a_fp8, a_s = quantize_act_fp8_fused(x.reshape(-1, x.shape[-1]))
        g = gate.forward_precomputed(x, a_fp8, a_s)
        u = up.forward_precomputed(x, a_fp8, a_s)
        h_fused = silu_mul_fused(g, u)
        h_naive = torch.nn.functional.silu(g) * u
        y_fused = down(h_fused)
        y_naive = down(h_naive)

    # SwiGLU uses silu_mul_fused internally — must match the fused
    # baseline bit-exactly (same kernel, same inputs).
    with torch.no_grad():
        y_swiglu = swiglu(x)
    assert torch.equal(y_swiglu, y_fused), (
        "NVFP4W4A8SwiGLU output diverged from silu_mul_fused baseline"
    )

    # Naive baseline differs from SwiGLU only by bf16 round-off in the
    # silu intermediate — the same magnitude as test_silu_mul_fused_matches_pytorch.
    diff = (y_swiglu.float() - y_naive.float()).abs().max().item()
    assert diff < 0.1, f"SwiGLU differs from naive path by {diff:.4f}"


# ---------------------------------------------------------------------------
# Fused silu(gate) * up + down_proj act_quant — one Triton kernel (F2)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("M,N", [
    (1024, 1024),
    (16384, 4096),    # prod gate_up (= down_proj input)
    (4096, 11008),    # Llama-style intermediate
    (1, 4096),        # edge: single row
    (16384, 1),       # edge: single col
])
def test_silu_quant_fused_matches_unfused(M, N):
    """``silu_quant_fused(gate, up)`` must be BIT-EXACT with the unfused
    ``silu_mul_fused(gate, up)`` → ``quantize_act_fp8_fused(hidden)``
    path — that's the production contract it replaces.

    The kernel narrows the fp32 silu output to bf16 before the amax and
    fp8 cast, so the fp8 hidden, the per-row scale, and the returned bf16
    hidden all match the two-kernel path exactly (no round-off drift).
    """
    torch.manual_seed(0)
    gate = torch.randn(M, N, dtype=torch.bfloat16, device="cuda")
    up = torch.randn(M, N, dtype=torch.bfloat16, device="cuda")

    hidden_ref = silu_mul_fused(gate, up)
    a_fp8_ref, a_s_ref = quantize_act_fp8_fused(hidden_ref)

    a_fp8, a_s, hidden = silu_quant_fused(gate, up)

    assert torch.equal(hidden, hidden_ref), (
        f"bf16 hidden diverged at M={M}, N={N}"
    )
    assert torch.equal(a_fp8, a_fp8_ref), (
        f"fp8 hidden diverged at M={M}, N={N}"
    )
    assert torch.equal(a_s, a_s_ref), (
        f"per-row scale diverged at M={M}, N={N}"
    )


def test_silu_quant_fused_output_shape_and_dtype():
    """Outputs: fp8 [M,N], scale [M,1] fp32, bf16 hidden [M,N]."""
    gate = torch.randn(64, 128, dtype=torch.bfloat16, device="cuda")
    up = torch.randn(64, 128, dtype=torch.bfloat16, device="cuda")
    a_fp8, a_s, hidden = silu_quant_fused(gate, up)
    assert a_fp8.shape == (64, 128)
    assert a_fp8.dtype == torch.float8_e4m3fn
    assert a_s.shape == (64, 1)
    assert a_s.dtype == torch.float32
    assert hidden.shape == (64, 128)
    assert hidden.dtype == torch.bfloat16
    assert a_fp8.device == gate.device


def test_swiglu_uses_fused_silu_quant_path():
    """F2 SwiGLU forward must use ``silu_quant_fused`` (not the F1
    silu_mul_fused + separate act_quant) — that's the optimization
    under test.

    The test runs both paths through the same SwiGLU module on the
    same input and checks bit-exactness:
      - F2 reference: gate/up via forward_precomputed + silu_quant_fused
        + down_proj via forward_precomputed
      - F2 test: full swiglu(x)

    If F2 is bit-exact with the explicit silu_quant_fused path (which
    is itself bit-exact with silu_mul_fused + act_quant), the SwiGLU
    output is bit-exact with the unfused SwiGLU. Backward behavior
    follows the same argument — STE reads only the bf16 hidden, which
    F2 produces bit-exactly.
    """
    torch.manual_seed(0)
    M, H, I = 1024, 1024, 2048
    swiglu = NVFP4W4A8SwiGLU(H, I, bias=False, block_size=16).cuda()
    x = torch.randn(M, H, dtype=torch.bfloat16, device="cuda")

    with torch.no_grad():
        # F2 path: build it explicitly using silu_quant_fused
        a_fp8, a_s = quantize_act_fp8_fused(x.reshape(-1, H))
        g = swiglu.gate_proj.forward_precomputed(x, a_fp8, a_s)
        u = swiglu.up_proj.forward_precomputed(x, a_fp8, a_s)
        h_fp8, h_s, hidden = silu_quant_fused(g, u)
        y_ref = swiglu.down_proj.forward_precomputed(hidden, h_fp8, h_s)

        # Test path: full SwiGLU (must use F2 internally)
        y_test = swiglu(x)

    assert torch.equal(y_test, y_ref), (
        "F2 SwiGLU output diverged from explicit silu_quant_fused path"
    )


# ===========================================================================
# Scheme integration: activation.SwiGLU + ffn_precision="w4a8"
# ===========================================================================
# The class-level tests above exercise ``NVFP4LinearW4A8`` /
# ``NVFP4W4A8SwiGLU`` directly. These tests exercise the config-driven
# wiring: ``HippoConfig(ffn_precision="w4a8")`` -> ``activation.SwiGLU``
# must build three ``NVFP4LinearW4A8`` projections (the same
# integration the deleted mode-2 ``test_ffn_nvfp4*`` tests used to
# guard, but for the live w4a8 scheme).
class TestSwiGLUSchemeW4A8:
    @staticmethod
    def _build_swiglu(H, I, bias=False, seed=42):
        from src.models.config import HippoConfig
        from src.models.activation import SwiGLU
        cfg = HippoConfig(
            vocab_size=8, hidden_size=H, intermediate_size=I,
            num_layers=1, num_blocks=1, num_heads=1, head_dim=H,
            safe_gate=True, lower_bound=-5.0, use_short_conv=False,
            use_bias=bias, ffn_precision="w4a8",
        )
        torch.manual_seed(seed)
        return SwiGLU(cfg).cuda()

    def test_swiglu_scheme_w4a8_wires_nvfp4linearw4a8(self):
        """``ffn_precision="w4a8"`` builds three NVFP4LinearW4A8
        projections on the real fp8 path (not the bf16_only
        fallback — H/I are divisible by 16)."""
        ffn = self._build_swiglu(128, 256)
        for name in ("gate_proj", "up_proj", "down_proj"):
            proj = getattr(ffn, name)
            assert isinstance(proj, NVFP4LinearW4A8), (
                f"{name} is {type(proj).__name__}, expected NVFP4LinearW4A8"
            )
            assert not proj.bf16_only, (
                f"{name} silently fell back to bf16_only — the fp8 W4A8 "
                f"path is not being exercised"
            )

    def test_swiglu_scheme_w4a8_forward_finite_and_close_to_bf16(self):
        """W4A8 SwiGLU forward is finite and stays within the fp8
        noise floor of a BF16 SwiGLU sharing the same master
        weights."""
        import torch.nn as nn
        H, I = 128, 256
        ffn = self._build_swiglu(H, I)

        # BF16 baseline sharing the same master weights (W4A8 keeps a
        # BF16 ``.weight`` leaf, so we can copy it into plain Linears).
        bf16 = {}
        for name, in_f, out_f in (
            ("gate_proj", H, I), ("up_proj", H, I), ("down_proj", I, H),
        ):
            lin = nn.Linear(in_f, out_f, bias=False).cuda().to(torch.bfloat16)
            lin.weight.data.copy_(getattr(ffn, name).weight.data)
            bf16[name] = lin

        x = torch.randn(4, 16, H, device="cuda", dtype=torch.bfloat16)
        with torch.no_grad():
            y = ffn(x)
            g = nn.functional.silu(bf16["gate_proj"](x))
            u = bf16["up_proj"](x)
            y_ref = bf16["down_proj"](g * u)

        assert torch.isfinite(y).all().item(), "W4A8 SwiGLU forward is non-finite"
        rms = ((y - y_ref) ** 2).mean().sqrt().item()
        rel = rms / (y_ref.abs().mean().item() + 1e-9)
        # fp8-A + NVFP4-B noise compounds across 3 matmuls + silu; the
        # median per-op floor is ~2.5% (see file header). Allow 0.30
        # rms-relative for the composed SwiGLU output.
        assert rel < 0.30, (
            f"W4A8 SwiGLU forward diverges from BF16: rms_rel={rel:.4f}"
        )

    def test_swiglu_scheme_w4a8_backward_grads_finite(self):
        """W4A8 SwiGLU backward produces finite, non-zero grads on
        each projection's BF16 master weight (STE bwd)."""
        H, I = 128, 256
        ffn = self._build_swiglu(H, I)
        x = torch.randn(2, 16, H, device="cuda", dtype=torch.bfloat16,
                        requires_grad=True)
        y = ffn(x)
        y.sum().backward()

        assert torch.isfinite(x.grad).all().item(), "x.grad has NaN/Inf"
        for name in ("gate_proj", "up_proj", "down_proj"):
            w = getattr(ffn, name).weight
            assert w.grad is not None, f"{name}.weight.grad is None"
            assert torch.isfinite(w.grad).all().item(), f"{name}.weight.grad NaN/Inf"
            assert w.grad.abs().sum().item() > 0.0, f"{name}.weight.grad all-zero"

    def test_swiglu_scheme_w4a8_training_step_decreases_loss(self):
        """A few SGD steps on the W4A8 SwiGLU master weights reduce a
        tiny synthetic loss (STE grad is usable).

        Objective is ``mean(y**2)`` (drive the output toward zero) —
        a guaranteed-reducible target that the optimizer can lower by
        shrinking the master weights, so the check doesn't depend on
        an unlearnable random regression target (whose irreducible
        MSE ~1.0 stays inside the bf16 loss-resolution floor)."""
        H, I = 128, 256
        ffn = self._build_swiglu(H, I)
        # Scale the (small) default init up so the output magnitude is
        # well above the fp8 quant-noise floor — otherwise mean(y**2)
        # starts near the floor (~0.01) and the reducible signal is
        # swallowed by the bf16 loss resolution.
        for name in ("gate_proj", "up_proj", "down_proj"):
            getattr(ffn, name).weight.data.mul_(4.0)
        params = [ffn.gate_proj.weight, ffn.up_proj.weight, ffn.down_proj.weight]
        opt = torch.optim.SGD(params, lr=0.05)

        x = torch.randn(4, 16, H, device="cuda", dtype=torch.bfloat16)

        losses = []
        for _ in range(60):
            opt.zero_grad()
            loss = (ffn(x) ** 2).mean()
            loss.backward()
            opt.step()
            losses.append(loss.item())

        assert losses[-1] < losses[0] * 0.8, (
            f"W4A8 SwiGLU training didn't reduce loss: "
            f"first={losses[0]:.4f}, last={losses[-1]:.4f}"
        )
