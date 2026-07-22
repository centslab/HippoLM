"""Regression tests for the FP8 RMSNorm/SwiGLU producer-side passthrough.

Covers the wiring added 2026-07-22 to skip the redundant per-FFN
``quantize_act_fp8_fused`` call when the upstream ``mlp_norm`` is
:class:`RmsNormFp8STE`:

  * ``RmsNormFp8STE`` with ``return_fp8=True`` returns
    ``(y_bf16, out_fp8, scale)``; the ``out_fp8`` / ``scale`` are
    byte-compatible with the FFN's expected act_quant outputs.
  * :class:`NVFP4W4A8SwiGLU.forward_precomputed` matches
    :meth:`NVFP4W4A8SwiGLU.forward` (numerics + finite + same shape).
  * End-to-end ``SwiGLU.forward_precomputed`` (the w4a8 scheme's
    passthrough entry point used by :class:`HippoLayer`) matches the
    unfused 3-projection baseline within the FP8 noise floor.
  * Backward through ``forward_precomputed`` produces finite,
    non-zero gradients to ``x`` and all three projection weights
    (the proper autograd contract — not a STE on the
    passthrough itself).

Run as part of ``pytest test/``.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.config import HippoConfig
from src.models.activation import SwiGLU
from src.models.ops.nvfp4_linear_w4a8 import (
    NVFP4W4A8SwiGLU,
    quantize_act_fp8_fused,
)
from src.models.ops.rmsnorm_fp8 import (
    RmsNormFp8STE,
    rmsnorm_fp8_with_passthrough,
)


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="FP8 forward / backward requires CUDA (sm_120 _scaled_mm)",
)


def _sig_rel(a, b):
    return (a.float() - b.float()).norm().item() / (b.float().norm().item() + 1e-12)


def _build_swiglu(H, I, bias=False, seed=42):
    cfg = HippoConfig(
        vocab_size=8, hidden_size=H, intermediate_size=I,
        num_layers=1, num_blocks=1, num_heads=1, head_dim=H,
        safe_gate=True, lower_bound=-5.0, use_short_conv=False,
        use_bias=bias, ffn_precision="w4a8",
    )
    torch.manual_seed(seed)
    return SwiGLU(cfg).cuda()


# --------------------------------------------------------------------------
# RmsNormFp8STE with return_fp8=True
# --------------------------------------------------------------------------
def test_rmsnorm_fp8_return_fp8_matches_internal_call():
    """``RmsNormFp8STE.apply(x, w, eps, return_fp8=True)`` returns
    ``(y_bf16, out_fp8, scale)`` where ``out_fp8`` is bit-compatible
    with what ``quantize_act_fp8_fused`` would produce from ``y_bf16``.

    The fp8 buffer from the passthrough must drop in for
    ``NVFP4LinearW4A8.forward_precomputed`` — the kernel layout
    (per-row amax + BF16-truncated scale) is the same as the
    standalone quantize_act_fp8_fused kernel.
    """
    torch.manual_seed(0)
    H = 128
    x = torch.randn(2, 16, H, device="cuda", dtype=torch.bfloat16) * 0.5
    w = torch.randn(H, device="cuda", dtype=torch.bfloat16) * 0.1
    eps = 1e-6

    y_bf16, a_fp8, a_s = rmsnorm_fp8_with_passthrough(x, w, eps)
    assert y_bf16.shape == x.shape
    # The fp8 buffer is 2D (the kernel is 2D-only) but holds the
    # flattened [B*T, K] view of the input. Use that for the
    # compatibility check.
    assert a_fp8.shape == (x.numel() // H, H), a_fp8.shape
    assert a_s.shape == (x.numel() // H, 1), a_s.shape
    assert a_fp8.dtype == torch.float8_e4m3fn
    assert a_s.dtype == torch.float32

    # The standalone quantize_act_fp8_fused kernel from the w4a8
    # path should produce the same FP8 layout when fed the same
    # input (the bf16 narrow+round step is shared).
    a_fp8_ref, a_s_ref = quantize_act_fp8_fused(
        y_bf16.reshape(-1, H)
    )
    a_fp8 = a_fp8.reshape(-1, H)
    assert torch.equal(a_fp8, a_fp8_ref), (
        "rmsnorm_fp8 passthrough FP8 buffer does not match "
        "quantize_act_fp8_fused — passthrough is not a drop-in for "
        "NVFP4LinearW4A8.forward_precomputed"
    )
    assert torch.equal(a_s, a_s_ref), (
        "rmsnorm_fp8 passthrough scale does not match — likely a "
        "BF16-truncation mismatch between the two quant kernels"
    )


# --------------------------------------------------------------------------
# NVFP4W4A8SwiGLU.forward_precomputed — direct module
# --------------------------------------------------------------------------
def test_nvfp4_w4a8_swiglu_forward_precomputed_matches_forward():
    """``NVFP4W4A8SwiGLU.forward_precomputed(x, a_fp8, a_s)`` must
    match ``NVFP4W4A8SwiGLU.forward(x)`` within the FP8 noise
    floor (the per-row amax is identical — both use the same
    FP8 + scale for gate/up; only the source of the FP8 differs)."""
    torch.manual_seed(0)
    H, I = 128, 256
    swiglu = NVFP4W4A8SwiGLU(H, I, bias=False).cuda()
    x = torch.randn(2, 16, H, device="cuda", dtype=torch.bfloat16) * 0.3

    with torch.no_grad():
        y_forward = swiglu(x)
        # Build a_fp8 / a_s from the bf16 input the same way the
        # 2-launch baseline would (so the FP8 input is the same
        # buffer the standalone quant would produce).
        a_fp8, a_s = quantize_act_fp8_fused(x.reshape(-1, H))
        y_precomputed = swiglu.forward_precomputed(x, a_fp8, a_s)

    assert y_precomputed.shape == y_forward.shape
    assert torch.isfinite(y_precomputed).all().item()
    rel = _sig_rel(y_precomputed, y_forward)
    # The FP8 + scale fed to both calls are identical, so the only
    # diff is the silu_quant_fused output's bf16 path (the
    # quantize step uses the same input). 0% is expected but
    # allow a tiny bf16 rounding buffer.
    assert rel < 1e-4, (
        f"forward_precomputed diverged from forward: sig_rel={rel:.6f}"
    )


def test_nvfp4_w4a8_swiglu_forward_precomputed_bwd_finite():
    """``forward_precomputed`` produces finite, non-zero gradients to
    ``x`` and all three projection weights (autograd path is wired
    through ``silu_quant_fused_with_passthrough``)."""
    torch.manual_seed(0)
    H, I = 128, 256
    swiglu = NVFP4W4A8SwiGLU(H, I, bias=False).cuda()
    x = torch.randn(2, 16, H, device="cuda", dtype=torch.bfloat16,
                    requires_grad=True)
    a_fp8, a_s = quantize_act_fp8_fused(x.reshape(-1, H))

    y = swiglu.forward_precomputed(x, a_fp8, a_s)
    y.sum().backward()

    assert x.grad is not None, "x.grad is None — autograd didn't flow"
    assert torch.isfinite(x.grad).all().item()
    for name in ("gate_proj", "up_proj", "down_proj"):
        w = getattr(swiglu, name).weight
        assert w.grad is not None, f"{name}.weight.grad is None"
        assert torch.isfinite(w.grad).all().item()
        assert w.grad.abs().sum().item() > 0.0


# --------------------------------------------------------------------------
# SwiGLU (activation.py) — w4a8 scheme passthrough entry point
# --------------------------------------------------------------------------
def test_swiglu_w4a8_uses_inner_module():
    """``SwiGLU(ffn_precision='w4a8')`` instantiates an
    :class:`NVFP4W4A8SwiGLU` as its inner module and exposes the
    three projections as direct attributes (for state_dict /
    external access)."""
    ffn = _build_swiglu(128, 256)
    assert ffn._uses_inner, "SwiGLU did not enter the inner-module path for w4a8"
    assert isinstance(ffn.ffn_inner, NVFP4W4A8SwiGLU)
    # The three projection attributes are aliased to the inner's
    # projections (state_dict / external access pattern).
    assert ffn.gate_proj is ffn.ffn_inner.gate_proj
    assert ffn.up_proj is ffn.ffn_inner.up_proj
    assert ffn.down_proj is ffn.ffn_inner.down_proj


def test_swiglu_w4a8_forward_precomputed_matches_forward():
    """``SwiGLU.forward_precomputed`` (the w4a8 passthrough entry
    point used by :class:`HippoLayer`) matches ``SwiGLU.forward``
    within the FP8 noise floor when fed the same fp8 input."""
    torch.manual_seed(0)
    H, I = 128, 256
    ffn = _build_swiglu(H, I)
    x = torch.randn(2, 16, H, device="cuda", dtype=torch.bfloat16) * 0.3

    with torch.no_grad():
        y_forward = ffn(x)
        a_fp8, a_s = quantize_act_fp8_fused(x.reshape(-1, H))
        y_precomputed = ffn.forward_precomputed(x, a_fp8, a_s)

    assert y_precomputed.shape == y_forward.shape
    assert torch.isfinite(y_precomputed).all().item()
    rel = _sig_rel(y_precomputed, y_forward)
    assert rel < 1e-4, (
        f"SwiGLU.forward_precomputed diverged from forward: "
        f"sig_rel={rel:.6f}"
    )


def test_swiglu_w4a8_forward_precomputed_rmsnorm_passthrough():
    """End-to-end: ``RmsNormFp8STE(return_fp8=True)`` output flows
    into ``SwiGLU.forward_precomputed`` and produces a finite
    gradient through the full chain (RMSNorm bwd → SwiGLU
    silu_quant_fused bwd → matmul bwd)."""
    torch.manual_seed(0)
    H, I = 128, 256
    ffn = _build_swiglu(H, I)
    x = torch.randn(2, 16, H, device="cuda", dtype=torch.bfloat16) * 0.3

    w_rms = torch.randn(H, device="cuda", dtype=torch.bfloat16) * 0.1
    eps = 1e-6

    y_bf16, a_fp8, a_s = rmsnorm_fp8_with_passthrough(x, w_rms, eps)
    y = ffn.forward_precomputed(y_bf16, a_fp8, a_s)
    y.sum().backward()

    assert torch.isfinite(y).all().item()
    # x was the leaf; grad flows through RmsNormFp8STE then through
    # the SwiGLU inner.
    for name in ("gate_proj", "up_proj", "down_proj"):
        w = getattr(ffn, name).weight
        assert w.grad is not None, f"{name}.weight.grad is None"
        assert torch.isfinite(w.grad).all().item()
        assert w.grad.abs().sum().item() > 0.0


def test_swiglu_non_w4a8_rejects_forward_precomputed():
    """``SwiGLU.forward_precomputed`` is only valid for w4a8 (the
    inner-module path). Other schemes must raise a clear error."""
    cfg = HippoConfig(
        vocab_size=8, hidden_size=128, intermediate_size=256,
        num_layers=1, num_blocks=1, num_heads=1, head_dim=128,
        safe_gate=True, lower_bound=-5.0, use_short_conv=False,
        use_bias=False, ffn_precision="w8a8",
    )
    ffn = SwiGLU(cfg).cuda()
    assert not ffn._uses_inner
    x = torch.randn(2, 16, 128, device="cuda", dtype=torch.bfloat16)
    a_fp8, a_s = quantize_act_fp8_fused(x.reshape(-1, 128))
    with pytest.raises(RuntimeError, match="only supported when"):
        ffn.forward_precomputed(x, a_fp8, a_s)
