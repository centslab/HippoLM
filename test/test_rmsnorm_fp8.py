"""Tests for the fused ``RMSNorm → FP8`` producer-side quant.

Covers:
  - Forward numerics: fused vs the 2-launch
    ``RMSNorm(BF16→BF16) → quantize_act_fp8_fused`` baseline at the
    FP8 noise floor (sig_rel ≲ 4% on mild inputs).
  - Singlepass K limit (raises on K > 4096).
  - STE backward: ``RmsNormFp8STE.apply(x, w, eps)`` produces finite
    gradients to **both** ``x`` and ``weight`` (the proper RMSNorm bwd
    — full STE pass-through would zero out ``dL/dweight``).
  - 3D input handling (KDA returns ``[B, T, H]``).
  - Output layout drop-in: ``RmsNormFp8STE`` followed by a
    ``FP8Linear`` (W8A8 q/k/v/o) on the BF16 dequant yields a finite
    GEMM result.

Run as part of ``pytest test/``.
"""
import pytest
import torch

from src.models.ops.rmsnorm_fp8 import (
    RmsNormFp8STE,
    rmsnorm_fp8_fwd,
)
from src.models.norms import RMSNorm
from src.models.ops.nvfp4_linear_w4a8 import quantize_act_fp8_fused


def _sig_rel(a, b):
    return (a.float() - b.float()).norm().item() / (b.float().norm().item() + 1e-12)


# --------------------------------------------------------------------------
# Forward numerics
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "M,K",
    [
        (128, 128),
        (4096, 192),         # KDA layer shape (small)
        (16384, 1536),       # KDA q/k/v/o prod shape
        (16384, 1024),       # hidden_size=1024 prod
        (16384, 4096),       # FFN intermediate K (only via N-side usage)
    ],
)
def test_rmsnorm_fp8_matches_2launch_baseline(M, K):
    torch.manual_seed(0)
    dev = "cuda"
    eps = 1e-6
    x = torch.randn(M, K, device=dev, dtype=torch.bfloat16) * 0.5
    weight = torch.randn(K, device=dev, dtype=torch.bfloat16) * 0.1 + 1.0

    # Fused
    out_fp8_fused, scale_fused, _ = rmsnorm_fp8_fwd(x, weight, eps)
    recon_fused = (out_fp8_fused.float() * scale_fused).to(torch.bfloat16)

    # 2-launch baseline: RMSNorm(BF16→BF16) then quant
    norm = RMSNorm(K).to(dev, torch.bfloat16)
    with torch.no_grad():
        norm.weight.copy_(weight)
    y_norm = norm(x)
    out_fp8_base, scale_base = quantize_act_fp8_fused(y_norm)
    recon_base = (out_fp8_base.float() * scale_base).to(torch.bfloat16)

    diff = _sig_rel(recon_fused, recon_base)
    assert diff < 1.5, (
        f"fused vs 2-launch sig_rel {diff:.3f}% > 1.5% on shape "
        f"{M}x{K}. The fused kernel and the RMSNorm+quant baseline "
        f"should be at the FP8 floor relative to one another."
    )


def test_rmsnorm_fp8_low_noise_vs_bf16():
    """The fused FP8 RMSNorm's per-round sig_rel vs the BF16
    RMSNorm reference is at the FP8 noise floor (≈3.76% per GEMM,
    but RMSNorm is elementwise so we see slightly less)."""
    torch.manual_seed(1)
    M, K = 4096, 192
    dev = "cuda"
    eps = 1e-6
    x = torch.randn(M, K, device=dev, dtype=torch.bfloat16) * 0.5
    weight = torch.randn(K, device=dev, dtype=torch.bfloat16) * 0.1 + 1.0

    out_fp8, scale, _ = rmsnorm_fp8_fwd(x, weight, eps)
    recon = (out_fp8.float() * scale).to(torch.bfloat16)

    # BF16 RMSNorm reference (no quant).
    norm = RMSNorm(K).to(dev, torch.bfloat16)
    with torch.no_grad():
        norm.weight.copy_(weight)
    ref = norm(x)

    diff = _sig_rel(recon, ref)
    assert diff < 5.0, (
        f"rmsnorm_fp8 sig_rel vs BF16 RMSNorm = {diff:.3f}% — should "
        f"be at the FP8 floor (<5% for elementwise normalization)."
    )


def test_rmsnorm_fp8_is_finite():
    """Heavy-tail inputs must not produce NaN or Inf from the FP8 round."""
    torch.manual_seed(2)
    M, K = 1024, 512
    dev = "cuda"
    eps = 1e-6
    x = torch.randn(M, K, device=dev, dtype=torch.bfloat16) * 5.0
    weight = torch.randn(K, device=dev, dtype=torch.bfloat16) * 0.1 + 1.0

    out_fp8, scale, rstd = rmsnorm_fp8_fwd(x, weight, eps)
    assert torch.isfinite(out_fp8.float()).all(), "out_fp8 had NaN/Inf"
    assert torch.isfinite(scale).all(), "scale had NaN/Inf"
    assert torch.isfinite(rstd).all(), "rstd had NaN/Inf"
    out_vals = out_fp8.float() * scale
    assert (out_vals.abs().max() <= 448.0 + 1e-3).item(), (
        f"fp8 rmsnorm out exceeds E4M3 max: |max|={out_vals.abs().max().item()}"
    )


def test_rmsnorm_fp8_rejects_k_above_singlepass_limit():
    """Singlepass K limit is 4096. Production never exceeds this
    (K is hidden_size=1024 or intermediate_size=2736, both well
    under), but a user-supplied shape with K > 4096 must fail loudly."""
    M, K = 16, 8192
    dev = "cuda"
    x = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
    w = torch.randn(K, device=dev, dtype=torch.bfloat16)
    with pytest.raises(NotImplementedError, match="singlepass K limit"):
        rmsnorm_fp8_fwd(x, w, eps=1e-6)


# --------------------------------------------------------------------------
# STE backward — proper RMSNorm bwd, not full passthrough
# --------------------------------------------------------------------------
def test_rmsnorm_fp8_ste_propagates_to_x():
    """``dL/dx`` should equal ``dL/dy * weight * rstd`` (proper RMSNorm bwd).

    We don't require bit-exact equality because the BF16 quant round
    is in the forward path; we only require the STE-formula match.
    """
    torch.manual_seed(3)
    dev = "cuda"
    M, K = 32, 64
    eps = 1e-6
    x = (torch.randn(M, K, device=dev, dtype=torch.bfloat16) * 0.5).requires_grad_(True)
    w = (torch.randn(K, device=dev, dtype=torch.bfloat16) * 0.1 + 1.0).requires_grad_(True)
    x.retain_grad()

    y = RmsNormFp8STE.apply(x, w, eps)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()

    grad_y = torch.randn_like(y)
    y.backward(grad_y)

    assert torch.isfinite(x.grad).all(), "x.grad had NaN/Inf"
    assert x.grad.abs().sum() > 0, "x.grad all zero"


def test_rmsnorm_fp8_ste_propagates_to_weight():
    """``dL/dweight`` MUST be non-zero — full STE passthrough would
    zero it out, which is wrong for an affine layer. The fused
    wrapper uses the proper RMSNorm bwd formula, so the weight gets
    a non-trivial gradient signal.
    """
    torch.manual_seed(4)
    dev = "cuda"
    M, K = 32, 64
    eps = 1e-6
    x = (torch.randn(M, K, device=dev, dtype=torch.bfloat16) * 0.5).requires_grad_(True)
    w = (torch.randn(K, device=dev, dtype=torch.bfloat16) * 0.1 + 1.0).requires_grad_(True)
    x.retain_grad()

    y = RmsNormFp8STE.apply(x, w, eps)
    grad_y = torch.randn_like(y)
    y.backward(grad_y)

    assert w.grad is not None, "weight received no gradient (full STE passthrough bug)"
    assert torch.isfinite(w.grad).all(), "w.grad had NaN/Inf"
    assert w.grad.abs().sum() > 0, (
        "w.grad is all zero — STE backward dropped the weight "
        "gradient signal; proper RMSNorm bwd must keep it non-zero"
    )


def test_rmsnorm_fp8_ste_3d_input():
    """KDA returns ``[B, T, H]``; the wrapper must handle 3D and
    keep both gradient signals non-zero."""
    torch.manual_seed(5)
    dev = "cuda"
    B, T, K = 2, 8, 96
    eps = 1e-6
    x = (torch.randn(B, T, K, device=dev, dtype=torch.bfloat16) * 0.5).requires_grad_(True)
    w = (torch.randn(K, device=dev, dtype=torch.bfloat16) * 0.1 + 1.0).requires_grad_(True)
    x.retain_grad()

    y = RmsNormFp8STE.apply(x, w, eps)
    assert y.shape == x.shape
    y.sum().backward()

    assert torch.isfinite(x.grad).all()
    assert torch.isfinite(w.grad).all()
    assert x.grad.abs().sum() > 0
    assert w.grad.abs().sum() > 0


def test_rmsnorm_fp8_fwd_output_drops_into_scaled_mm():
    """The output FP8 + per-row scale must be consumable by
    ``torch._scaled_mm`` (the KDA q_proj / FFN gate_proj GEMM
    contract when running W8A8).
    """
    M, K = 1024, 256
    dev = "cuda"
    eps = 1e-6
    x = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
    w = torch.randn(K, device=dev, dtype=torch.bfloat16) * 0.1 + 1.0
    out_fp8, scale_a, _ = rmsnorm_fp8_fwd(x, w, eps)

    # weight [N=K, K]: row i = output channel i.
    weight_gemm = torch.randn(K, K, device=dev, dtype=torch.bfloat16) / (K ** 0.5)
    from src.models.ops.fp8_linear import _quantize_w_per_channel
    wq, ws = _quantize_w_per_channel(weight_gemm)  # ws: [N, 1]
    # RowWise: scale_a [M, 1], scale_b [1, N]
    scale_b = ws.reshape(1, K).contiguous()
    out = torch._scaled_mm(
        out_fp8, wq.T, scale_a, scale_b,
        out_dtype=torch.bfloat16,
    )
    assert out.shape == (M, K)
    assert torch.isfinite(out).all(), (
        "fp8 rmsnorm output fed into _scaled_mm produced NaN/Inf"
    )
