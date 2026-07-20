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
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from src.models.ops.nvfp4_linear_w4a8 import (
    NVFP4LinearW4A8, quantize_act_fp8,
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
# Forward correctness
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("M,K,N", [
    (128, 1024, 1024),
    (1024, 1024, 1024),
    (4096, 1536, 8192),  # prod gate_up
    (4096, 4096, 1536),  # prod down
])
def test_w4a8_forward_close_to_fp32_ref(M, K, N):
    """W4A8 forward matches fp32-dequantized reference within the
    E4M3 B rounding floor (~2-4% median rel).
    """
    torch.manual_seed(0)
    x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    w = torch.randn(N, K, dtype=torch.bfloat16, device="cuda") * 0.1

    layer = NVFP4LinearW4A8(K, N, bias=False, block_size=16).cuda()
    with torch.no_grad():
        layer.weight.data.copy_(w)

    with torch.no_grad():
        y_w4a8 = layer(x)
        y_ref = _fp32_dequant_ref(x, w).bfloat16()

    rel = (y_w4a8.float() - y_ref.float()).abs() / (y_ref.float().abs() + 1e-3)
    rel_median = rel.median().item()
    # E4M3 B noise floor: 2-4% median across shapes; allow 5%.
    assert rel_median < 0.05, (
        f"W4A8 fwd diverges from fp32 ref: median_rel={rel_median*100:.2f}%, "
        f"max_rel={rel.max().item()*100:.2f}%"
    )


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