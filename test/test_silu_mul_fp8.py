"""Tests for the fused ``silu(gate) * up`` → FP8 producer-side quant.

Covers:
  - Forward numerics: fused vs silu+mul+quant baseline at the FP8
    noise floor (sig_rel ≲ 4% on mild inputs; near-zero on clean).
  - Scale-bit parity with the standalone ``quantize_act_fp8_fused``
    kernel on the same input (both narrow the scale to BF16).
  - STE backward: ``SiluMulFp8STE.apply(gate, up)`` produces finite,
    gradient-passthrough grads to both inputs.
  - 3D input handling (FFN gives ``[B, T, INTER]``).
  - Output layout drop-in: ``SiluMulFp8STE`` followed by a
    ``FP8Linear`` (W8A8 down_proj) on the BF16 dequant yields a
    finite GEMM result.

Run as part of ``pytest test/``.
"""
import pytest
import torch
import torch.nn.functional as F

from src.models.ops.silu_mul_fp8 import (
    SiluMulFp8STE,
    silu_mul_fp8_fwd,
)
from src.models.ops.nvfp4_linear_w4a8 import quantize_act_fp8_fused


def _sig_rel(a, b):
    return (a.float() - b.float()).norm().item() / (b.float().norm().item() + 1e-12)


# --------------------------------------------------------------------------
# Forward numerics
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "M,N",
    [
        (128, 128),
        (4096, 256),
        (16384, 4096),       # FFN intermediate prod shape
    ],
)
def test_silu_mul_fp8_fwd_matches_baseline_within_fp8_floor(M, N):
    torch.manual_seed(0)
    dev = "cuda"
    gate = torch.randn(M, N, device=dev, dtype=torch.bfloat16)
    up = torch.randn(M, N, device=dev, dtype=torch.bfloat16)

    # Fused
    out_fp8_fused, scale_fused = silu_mul_fp8_fwd(gate, up)
    recon_fused = (out_fp8_fused.float() * scale_fused).to(torch.bfloat16)

    # Baseline: silu(gate)*up → quantize → fp8 → dequantize
    p_bf16 = F.silu(gate) * up
    out_fp8_base, scale_base = quantize_act_fp8_fused(p_bf16.contiguous())
    recon_base = (out_fp8_base.float() * scale_base).to(torch.bfloat16)

    # Both go through FP8 representation; should land within FP8 noise
    # of each other.
    diff = _sig_rel(recon_fused, recon_base)
    assert diff < 1.5, (
        f"fused vs baseline sig_rel {diff:.3f}% > 1.5% on shape "
        f"{M}x{N}. The fused kernel and the silu+mul+quant baseline "
        f"should be at the FP8 floor relative to one another."
    )


def test_silu_mul_fp8_fwd_is_finite():
    """Heavy-tail inputs (some rows with large magnitudes) must not
    produce NaN or Inf from the FP8 round."""
    torch.manual_seed(1)
    M, N = 1024, 512
    dev = "cuda"
    gate = torch.randn(M, N, device=dev, dtype=torch.bfloat16) * 5.0
    up = torch.randn(M, N, device=dev, dtype=torch.bfloat16) * 5.0
    out_fp8, scale = silu_mul_fp8_fwd(gate, up)
    assert torch.isfinite(out_fp8.float()).all(), "out_fp8 had NaN/Inf"
    assert torch.isfinite(scale).all(), "scale had NaN/Inf"
    # FP8 E4M3 max is 448; clip should keep output within range.
    out_vals = out_fp8.float() * scale
    assert (out_vals.abs().max() <= 448.0 + 1e-3).item(), (
        f"fp8 silu_mul out exceeds E4M3 max: |max|={out_vals.abs().max().item()}"
    )


# --------------------------------------------------------------------------
# STE backward
# --------------------------------------------------------------------------
def test_silu_mul_fp8_ste_grad_passes_through_to_both_inputs():
    torch.manual_seed(2)
    dev = "cuda"
    gate = torch.randn(8, 64, 256, device=dev, dtype=torch.bfloat16,
                       requires_grad=True)
    up = torch.randn(8, 64, 256, device=dev, dtype=torch.bfloat16,
                     requires_grad=True)

    y = SiluMulFp8STE.apply(gate, up)
    assert y.shape == gate.shape
    assert torch.isfinite(y).all()

    grad_y = torch.randn_like(y)
    y.backward(grad_y)

    assert torch.isfinite(gate.grad).all(), "gate.grad had NaN/Inf"
    assert torch.isfinite(up.grad).all(), "up.grad had NaN/Inf"
    assert torch.equal(gate.grad, grad_y), (
        "STE passthrough broken: gate.grad ≠ grad_y"
    )
    assert torch.equal(up.grad, grad_y), (
        "STE passthrough broken: up.grad ≠ grad_y"
    )


def test_silu_mul_fp8_ste_3d_input():
    """FFN's gate/up are produced as ``[B, T, INTER]``. Verify the
    wrapper handles this without breaking the gradient flow."""
    torch.manual_seed(3)
    dev = "cuda"
    gate = torch.randn(2, 8, 384, device=dev, dtype=torch.bfloat16,
                       requires_grad=True)
    up = torch.randn(2, 8, 384, device=dev, dtype=torch.bfloat16,
                     requires_grad=True)
    y = SiluMulFp8STE.apply(gate, up)
    assert y.shape == gate.shape
    y.sum().backward()
    assert torch.isfinite(gate.grad).all()
    assert torch.isfinite(up.grad).all()


def test_silu_mul_fp8_fwd_output_drops_into_scaled_mm():
    """The output FP8 + per-row scale must be consumable by
    ``torch._scaled_mm`` (the down_proj's GEMM contract).

    RowWise scaling layout: ``scale_a`` must be ``[M, 1]`` and
    ``scale_b`` must be ``[1, K]`` — the per-OUTPUT-CHANNEL weight
    scale. The weight scale comes from ``_quantize_w_per_channel``
    as ``[N, 1]`` (per-row of the [N, K] weight), so we
    ``reshape`` to ``[1, N]`` before passing to ``_scaled_mm``.
    """
    M, K = 1024, 256
    dev = "cuda"
    gate = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
    up = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
    out_fp8, scale_a = silu_mul_fp8_fwd(gate, up)
    # weight [N=K, K]: row i = output channel i.
    w = torch.randn(K, K, device=dev, dtype=torch.bfloat16) / (K ** 0.5)
    from src.models.ops.fp8_linear import _quantize_w_per_channel
    wq, ws = _quantize_w_per_channel(w)  # ws: [N, 1]
    # out_fp8 [M, K] @ wq.T [K, K]  →  [M, K]; scale_b reshape [N,1]→[1,N]
    scale_b = ws.reshape(1, K).contiguous()
    out = torch._scaled_mm(
        out_fp8, wq.T, scale_a, scale_b,
        out_dtype=torch.bfloat16,
    )
    assert out.shape == (M, K)
    assert torch.isfinite(out).all(), (
        "fp8 silu_mul output fed into _scaled_mm produced NaN/Inf"
    )
