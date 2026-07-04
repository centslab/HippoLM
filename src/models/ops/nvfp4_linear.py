"""W4A16 NVFP4 Linear (FFN weight stored in NVFP4 packed format, BF16 activation).

This module exposes a drop-in replacement for ``F.linear`` whose weight
**storage** is NVFP4 packed (uint8 + FP8-e4m3fn microblock scales). The
matmul itself runs in BF16 (dequantize-on-fwd); see the FP4-matmul
note below for why.

State
-----
The BF16 master weight is ``self.weight`` (an ``nn.Parameter``). It is
the tensor the optimizer updates each step. The packed FP4 buffers
``self.packed_weight`` / ``self.scales`` are derived from it; the
caller calls ``repack_weights()`` AFTER each optimizer step to keep
them in sync.

Why both? The BF16 master is what the optimizer / autograd expects
(muon / adamw, CPU offload, etc.). The packed buffers are the "save
format" — checkpoints write them in NVFP4 form (~3.5x smaller), and
they let the fwd path use the latest quantized weights without having
to re-quantize every forward.

FP4-matmul note
---------------
PyTorch's ``torch._scaled_mm`` NVFP4 path (``scale_block_size=16``)
requires BOTH A and B to be FP4-packed (W4A4 only). A pure BF16 x FP4
mixed-precision GEMM is not exposed in 2.9.1. We therefore dequantize
to BF16 before the matmul, keeping the autograd contract identical
to a plain Linear. Swapping in a real FP4 GEMM is a follow-up that
touches only ``_NVFP4Matmul.forward``.

Autograd design (STE for the quantize round-trip)
-------------------------------------------------
The forward dequantizes ``packed_weight`` / ``scales`` to BF16 and
calls ``F.linear``. The dequantized BF16 is conceptually a quantized
view of ``self.weight``; gradient should therefore flow into
``self.weight`` (the optimizer's view). We achieve this by passing
``self.weight`` as a graph anchor through the custom Function — the
forward ignores it, the backward computes the standard
``grad_out.T @ x`` matmul and returns it as the gradient on
``self.weight``. This is the standard straight-through estimator
(STE) for fake-quantization.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.ops.nvfp4 import dequantize_nvfp4


class _NVFP4Matmul(torch.autograd.Function):
    """BF16 act @ NVFP4-packed weight (dequantized to BF16 in forward).

    Inputs (only ``x`` and ``w_master`` need gradients):
        x            : [..., K] in BF16 (activation, requires_grad)
        w_master     : [N, K] in BF16 (the BF16 master weight, requires_grad).
                       Used as a graph anchor; the forward does NOT consume it.
                       The backward returns the standard matmul grad on it so
                       the optimizer can update it. STE for the quantize noise.
        packed_w     : [N, K_padded // 2] uint8 (FP4-packed, no grad)
        scales_w     : [N, K_padded // 16] fp8_e4m3fn (no grad)
        bias         : [N] or None
        K_orig       : int (the un-padded K)
        block_size   : int (NVFP4 microblock size, fixed at 16)
    """

    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        x: torch.Tensor,
        w_master: torch.Tensor,
        packed_w: torch.Tensor,
        scales_w: torch.Tensor,
        bias: torch.Tensor | None,
        K_orig: int,
        block_size: int,
    ) -> torch.Tensor:
        # Dequantize FP4 -> BF16 for the matmul (the actual work).
        w_bf16 = dequantize_nvfp4(
            packed_w, scales_w, K_orig,
            block_size=block_size,
            out_dtype=x.dtype,
        )
        # Save (x, w_bf16) for backward. w_bf16 is the dequantized view;
        # its grad in backward is the conceptual grad on w_master (STE).
        ctx.save_for_backward(x, w_bf16)
        ctx.has_bias = bias is not None
        return F.linear(x, w_bf16, bias)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):  # type: ignore[override]
        x, w_bf16 = ctx.saved_tensors
        # Standard BF16 matmul backward (we can't use F.linear's backward
        # because w_bf16 isn't a leaf that the autograd graph knows about
        # as a parameter — we dequantized it from packed/scales).
        K = w_bf16.shape[1]
        N = w_bf16.shape[0]
        grad_out_2d = grad_out.reshape(-1, N)
        x_2d = x.reshape(-1, K)
        # grad_x = grad_out @ W  (broadcasts over leading dims).
        grad_x = grad_out @ w_bf16
        # grad_w = grad_out.T @ x  -> shape [N, K]. This is what we want
        # to flow into ``self.weight`` (the BF16 master).
        grad_w = (grad_out_2d.t().to(x_2d.dtype)) @ x_2d
        grad_bias = grad_out_2d.sum(dim=0) if ctx.has_bias else None
        return grad_x, grad_w, None, None, grad_bias, None, None


class _RmsNormNvfp4Matmul(torch.autograd.Function):
    """Fused RMSNorm + NVFP4 BF16-activation matmul.

    Drops one full ``x`` save from the unfused chain
    (``rms_norm`` saves x via res_out; ``_NVFP4Matmul`` saves y_norm
    = the matmul input). The fused function saves ``x`` once and
    *recomputes* ``y_norm`` in backward from ``(x, w_norm, rstd)``,
    avoiding the second activation save.

    Forward:  y = rms_norm(x); out = NVFP4_dequant(W) @ y.T
    Backward: Recompute y from x; NVFP4 bwd → d_y, d_w_master;
              RMSNorm bwd → d_x, d_w_norm.

    Saved tensors: x, w_bf16, w_norm, rstd  (no y!). Compared to the
    unfused chain (rms_norm saves x; NVFP4 saves y_norm = matmul
    input): saves one full ``[T, hidden]`` per call (≈ 48 MiB at
    prod shape for a single layer).

    Why not also drop x for rms bwd? ``x`` is needed for the
    RMSNorm backward dx computation. Could be eliminated only via
    fusing the previous sublayer's residual add into this Function
    (out of scope; would require graph traversal).

    Why keep w_bf16? The dequantized BF16 weight is needed for the
    NVFP4 backward (``grad_y = grad_out @ w_bf16``). Both paths
    save it; net is the same.
    """

    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        x: torch.Tensor,
        w_norm: torch.Tensor,
        w_master: torch.Tensor,
        packed_w: torch.Tensor,
        scales_w: torch.Tensor,
        bias: torch.Tensor | None,
        K_orig: int,
        block_size: int,
        eps: float,
    ) -> torch.Tensor:
        # Fused RMSNorm forward via fla Triton kernel. The kernel
        # works on 2D [T, D] input; res_out is x (no residual case).
        from src.models.ops._vendored.fla.modules.layernorm import layer_norm_fwd
        x_shape = x.shape
        x_2d = x.reshape(-1, x_shape[-1])
        y_2d, _mean, rstd, res_out_2d = layer_norm_fwd(
            x_2d, w_norm, None, eps=eps, is_rms_norm=True,
        )
        y = y_2d.reshape(x_shape)
        # res_out is x_2d for RMSNorm w/o residual (res_out=None → return x).
        # The fla fwd returns res_out=x when no residual — confirm:
        assert res_out_2d is x_2d, "expected res_out=x for RMSNorm w/o residual"

        # NVFP4 dequant + matmul (same path as _NVFP4Matmul).
        w_bf16 = dequantize_nvfp4(
            packed_w, scales_w, K_orig,
            block_size=block_size,
            out_dtype=y.dtype,
        )
        out = F.linear(y, w_bf16, bias)

        # Save for backward: x (rms bwd), w_bf16 (nvfp4 dx),
        # w_norm + rstd (rms dweight + rms dx). Do NOT save y —
        # recompute it in backward from (x, w_norm, rstd).
        ctx.save_for_backward(x, w_bf16, w_norm, rstd)
        ctx.has_bias = bias is not None
        ctx.eps = eps
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):  # type: ignore[override]
        x, w_bf16, w_norm, rstd = ctx.saved_tensors
        K = w_bf16.shape[1]
        N = w_bf16.shape[0]
        grad_out_2d = grad_out.reshape(-1, N)
        x_shape = x.shape

        # Recompute y_norm from saved (x, w_norm, rstd). y = x * rstd * w_norm
        # (this is what fla layer_norm_fwd computes for RMSNorm).
        x_2d = x.reshape(-1, K)
        y_2d = (x_2d.to(w_norm.dtype) * rstd.unsqueeze(-1) * w_norm).to(x_2d.dtype)
        y = y_2d.reshape(x_shape)

        # NVFP4 bwd (same as _NVFP4Matmul.backward but on y instead of x).
        # grad_y = grad_out @ W
        # grad_w = grad_out.T @ y
        d_y = grad_out @ w_bf16
        d_w_master = (grad_out_2d.t().to(y_2d.dtype)) @ y_2d
        d_bias = grad_out_2d.sum(dim=0) if ctx.has_bias else None

        # RMSNorm bwd on (x, d_y, w_norm, rstd) → (d_x, d_w_norm).
        from src.models.ops._vendored.fla.modules.layernorm import layer_norm_bwd
        d_y_2d = d_y.reshape(-1, K)
        d_x_2d, d_w_norm_2d, _db, _dresidual_in = layer_norm_bwd(
            d_y_2d,
            x_2d,
            w_norm,
            None,  # bias
            None, rstd,  # mean=None, rstd
            None,  # dresidual
            False,  # has_residual
            True,  # is_rms_norm
        )
        d_x = d_x_2d.reshape(x_shape)
        d_w_norm = d_w_norm_2d.reshape(w_norm.shape) if d_w_norm_2d is not None else None

        return (
            d_x, d_w_norm, d_w_master,
            None, None,  # packed_w, scales_w (no grad)
            d_bias,
            None, None, None,  # K_orig, block_size, eps (no grad)
        )


class NVFP4Linear(nn.Module):
    """``nn.Linear``-equivalent module that stores its weight in NVFP4.

    See module docstring for the autograd / state design.

    State dict keys:
        weight                 : the BF16 master weight (shape [N, K])
        packed_weight          : uint8 buffer [N, K // 2]   (FP4)
        scales                 : fp8_e4m3fn buffer [N, K // 16]
        bias                   : present iff ``bias=True``
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        device=None,
        dtype=None,
        block_size: int = 16,
    ) -> None:
        super().__init__()
        assert block_size == 16, (
            f"NVFP4 spec requires block_size=16, got {block_size}"
        )
        self.in_features = in_features
        self.out_features = out_features
        self.block_size = block_size

        # BF16 master weight — the parameter the optimizer updates.
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, device=device, dtype=dtype or torch.bfloat16)
        )
        # NVFP4-packed view + scales (buffers, not parameters; recomputed
        # by ``repack_weights`` after each optimizer step).
        K_padded = in_features + (-in_features % block_size)
        self.register_buffer(
            "packed_weight",
            torch.zeros(out_features, K_padded // 2, device=device, dtype=torch.uint8),
        )
        self.register_buffer(
            "scales",
            torch.zeros(out_features, K_padded // block_size, device=device, dtype=torch.float8_e4m3fn),
        )

        if bias:
            self.bias = nn.Parameter(
                torch.empty(out_features, device=device, dtype=dtype or torch.bfloat16)
            )
        else:
            self.register_parameter("bias", None)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
        # Populate the FP4 buffers from the freshly-initialized BF16
        # master so the layer is fully ready on construction — no
        # lazy-repack branch in the forward.
        self.repack_weights()

    @torch.no_grad()
    def repack_weights(self) -> None:
        """Re-quantize the BF16 master weight into the NVFP4 buffer pair.

        Call AFTER each optimizer step (NOT after each forward). This
        keeps the packed buffers in sync with the (now-updated) BF16
        master, so the next forward sees the latest quantized weights.
        """
        from src.models.ops.nvfp4 import quantize_nvfp4
        packed, scales = quantize_nvfp4(self.weight.data, block_size=self.block_size)
        self.packed_weight.copy_(packed)
        self.scales.copy_(scales)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _NVFP4Matmul.apply(
            x, self.weight,
            self.packed_weight, self.scales,
            self.bias,
            self.in_features, self.block_size,
        )

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, block_size={self.block_size}, "
            f"weight_storage=NVFP4(E2M1+FP8-e4m3fn 1x16)"
        )


def repack_nvfp4_weights(model: nn.Module) -> int:
    """Walk ``model`` and call ``repack_weights()`` on every NVFP4 module.

    Looks for the NVFP4 modules shipped with this project:

      - :class:`NVFP4Linear`          (single-GPU SwiGLU path)
      - :class:`NVFP4ColumnParallelLinear`  (TP column-parallel SwiGLU)
      - :class:`NVFP4RowParallelLinear`     (TP row-parallel SwiGLU)

    All three expose a ``repack_weights()`` method that re-quantizes
    the BF16 master weight into the FP4 packed buffer pair. The
    training loop calls this once per optimizer step (after the
    optimizer has updated the BF16 master) to keep the FP4 buffers
    in sync with the master.

    Returns the number of NVFP4 modules repacked. ``0`` is a valid
    return value when NVFP4 is not enabled (the function is a no-op).
    """
    count = 0
    # Local imports to avoid a circular dependency at module-load time
    # (this module is imported by both activation.py and tp_model.swiglu,
    # which themselves are imported here at runtime, not at top level).
    from src.models.ops.nvfp4_tp import (
        NVFP4ColumnParallelLinear,
        NVFP4RowParallelLinear,
    )
    target_types = (NVFP4Linear, NVFP4ColumnParallelLinear, NVFP4RowParallelLinear)
    for module in model.modules():
        if isinstance(module, target_types):
            module.repack_weights()
            count += 1
    return count