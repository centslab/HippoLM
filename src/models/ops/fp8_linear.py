"""W8A8 FP8 Linear (E4M3) — drop-in ``nn.Linear`` replacement.

The forward uses ``torch._scaled_mm`` with E4M3 on both sides:

    x (BF16, [..., K])
        -> per-row E4M3 quant   (scale_a shape [M, 1], FP32)
    w (BF16, [N, K])
        -> per-row (per-output-channel) E4M3 quant   (scale_b shape [1, N], FP32)
    y = x_q @ w_q.T  via _scaled_mm(scale_a, scale_b) -> BF16

The autograd design mirrors ``src/models.ops.nvfp4_linear``'s
mode-2 pattern: the BF16 master ``weight`` is the leaf parameter
the optimizer updates; the FP8 quantized view is computed on
the fly in forward and discarded; the backward is STE — it
re-runs BF16 matmul against the BF16 weight (the optimizer sees
the standard ``grad_w``).

State dict keys (BF16 leaf present):

    weight : the BF16 master weight (shape [N, K])
    bias   : present iff ``bias=True``

This is intentionally identical to ``nn.Linear``'s state_dict
shape so a BF16 checkpoint loads directly — no conversion.

Why per-row-act + per-channel-weight (the COAT / transformer-engine
convention) rather than per-tensor:

- per-tensor E4M3 hits 3.7-3.8% sig_rel on Gaussian-distributed
  post-RMSNorm activations; per-row + per-channel is in the same
  ballpark on this synthetic distribution but absorbs row-wise
  outlier rows cleanly (the row's amax becomes the FP32 scale, no
  row gets starved). On real (non-Gaussian, post-silu) data the
  outlier benefit will be larger.
- per-row + per-channel is the only row-wise scaling
  configuration ``torch._scaled_mm`` supports for E4M3 on
  sm_120.

Why E4M3 only (no E5M2 path): sm_120 cuBLAS does NOT support
E5M2 x E5M2 matmul. The "E5M2 for grads" NVIDIA recommendation
would force an E5M2 grad tensor, but the only GEMM shape
available is E4M3 x E5M2 per-tensor (not row-wise). See
``test_kda_w8a8_bwd.py`` for the empirical sweep; the +2pp
E5M2 noise benefit is not worth the hardware constraint.

Autograd contract (STE for the quantize round-trip):

    forward : quantize(x) @ quantize(w).T  via _scaled_mm -> BF16
    backward: grad_x = grad_out @ w_bf16    (BF16 matmul)
              grad_w = grad_out.T @ x_bf16  (BF16 matmul)
              grad_b = grad_out.sum(dim=0)  (if bias present)

The "STE" idea: the forward's quantization noise is a
deterministic function of x and w. The standard PyTorch
backward would propagate through the rounding op as a step
function (zero grad almost everywhere) which is the wrong
training signal. We bypass this by re-running the backward
matmul against the BF16 leaves; the gradient flows as if
the forward had been BF16. This is the same scheme NVFP4
mode-2 uses and is what production FP8 training (NVlabs
COAT, TransformerEngine) does.

bf16_only escape hatch
----------------------
``bf16_only=True`` skips quantization and falls back to plain
BF16 GEMM. Useful for A/B tests against the FP8 path without
re-instantiating the model. Same numerics as ``nn.Linear``.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# FP8 E4M3 hardware constants (sm_120 / Blackwell consumer tensor cores).
_E4M3_MAX = 448.0
_E4M3_MIN_NORMAL = 6.103515625e-05


class _FP8E4M3Matmul(torch.autograd.Function):
    """BF16 leaf + per-row-act / per-channel-weight FP8 GEMM, STE backward.

    Forward: quantize ``x`` (per-row) and ``w`` (per-output-channel)
    to E4M3 in FP32, call ``torch._scaled_mm`` to compute the GEMM
    in FP32 then cast to BF16. Add bias if present.

    Backward: STE — re-run the matmul backward against the BF16
    leaves. ``grad_w`` matches what a plain ``F.linear`` would
    produce, so the optimizer update is the standard BF16 one.

    Inputs to forward (all BF16):

        x       : [..., K]   (activation; requires_grad)
        w       : [N, K]     (BF16 master weight; requires_grad)
        bias    : [N] or None
    """

    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        x: torch.Tensor,
        w: torch.Tensor,
        bias: torch.Tensor | None,
    ) -> torch.Tensor:
        # Quantize to FP32 (so we can use FP32 division for the scale).
        x_fp32 = x.to(torch.float32)
        w_fp32 = w.to(torch.float32)

        # Per-row activation scale (M, 1). amax over the K axis (last
        # dim) gives one scale per row. ``reshape(-1, K)`` flattens the
        # leading dims so we always operate on 2D.
        x_2d = x_fp32.reshape(-1, x_fp32.shape[-1])
        amax_x = x_2d.abs().amax(dim=1, keepdim=True).clamp(min=_E4M3_MIN_NORMAL)
        scale_x = (amax_x / _E4M3_MAX).contiguous()            # (M, 1) FP32

        # Per-output-channel weight scale. amax over K gives one
        # scale per output row (channel). We compute the *input* to
        # the quantization as ``amax_w / _E4M3_MAX`` (shape [N, 1])
        # so the FP8 values land in [-448, 448]; the *output* scale
        # for ``_scaled_mm`` is the same value transposed to [1, N].
        amax_w = w_fp32.abs().amax(dim=1, keepdim=True).clamp(min=_E4M3_MIN_NORMAL)
        scale_w_input = (amax_w / _E4M3_MAX)                   # [N, 1] FP32
        scale_w = scale_w_input.T.contiguous()                  # (1, N) FP32

        # Quantize: divide by scale, clamp, cast to FP8.
        x_q = (x_2d / scale_x).clamp(-_E4M3_MAX, _E4M3_MAX).to(torch.float8_e4m3fn)
        w_q = (w_fp32 / scale_w_input).clamp(-_E4M3_MAX, _E4M3_MAX).to(torch.float8_e4m3fn)

        # ``torch._scaled_mm`` computes ``x_q @ w_q.T`` with FP32
        # accum. The b operand is the *transposed* weight (column-
        # major view of the row-major [N, K] weight).
        out_2d = torch._scaled_mm(
            x_q, w_q.T,
            scale_x, scale_w,
            out_dtype=torch.bfloat16,
        )                                                        # (M, N)

        # Add bias (broadcasts over the leading dims of the input).
        if bias is not None:
            out_2d = out_2d + bias

        # Reshape back to the input's leading shape.
        out = out_2d.reshape(*x.shape[:-1], w.shape[0])

        # Save the BF16 leaves for the STE backward (no FP8 tensors
        # saved — we recompute everything from x, w).
        ctx.save_for_backward(x, w)
        ctx.has_bias = bias is not None
        ctx.input_shape = x.shape
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):  # type: ignore[override]
        x, w = ctx.saved_tensors
        K = w.shape[1]
        N = w.shape[0]
        # Flatten leading dims so the matmul shape contract holds.
        grad_out_2d = grad_out.reshape(-1, N).to(torch.float32)
        x_2d = x.reshape(-1, K).to(torch.float32)
        w_2d = w.to(torch.float32)
        # grad_x = grad_out @ w  (broadcast over leading dims).
        grad_x_2d = grad_out_2d @ w_2d
        # grad_w = grad_out.T @ x  -> [N, K]. This is the gradient
        # the optimizer will see (matched to the BF16 leaf).
        grad_w = grad_out_2d.t() @ x_2d
        # Reshape grad_x back to the input shape.
        grad_x = grad_x_2d.reshape(*ctx.input_shape)
        if ctx.has_bias:
            grad_bias = grad_out_2d.sum(dim=0)
        else:
            grad_bias = None
        # Cast back to BF16 to match the leaf dtypes (PyTorch auto-casts
        # if the leaf is BF16, but be explicit so the autograd trace
        # is clean).
        return grad_x.to(x.dtype), grad_w.to(w.dtype), grad_bias


class FP8Linear(nn.Module):
    """``nn.Linear``-equivalent module that runs its GEMM in FP8 E4M3.

    Storage: BF16 master ``weight`` (the optimizer's view) + optional
    ``bias``. The FP8 quantization is computed on the fly in forward
    and discarded.

    Args:
        in_features  : K
        out_features : N
        bias         : whether to include a bias parameter
        device       : passed to the parameter allocation
        dtype        : parameter dtype (BF16 by default)
        bf16_only    : escape hatch — skip quantization, use plain
                       BF16 GEMM (matches ``nn.Linear`` exactly)

    Shape constraint
    ----------------
    ``torch._scaled_mm`` requires both dimensions of each operand
    to be divisible by 16 on sm_120. If either ``in_features`` or
    ``out_features`` is not a multiple of 16, the constructor
    silently flips into ``bf16_only=True`` mode (the resulting
    module is bit-exact with a plain ``nn.Linear`` — see
    :meth:`extra_repr` for the resolved mode). This is the
    fallback that lets the production rollout of ``kda_fp8``
    transparently skip ``b_proj`` (which has ``out_features=12``
    at H=12, not divisible by 16) without raising.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        device=None,
        dtype=None,
        bf16_only: bool = False,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # ``torch._scaled_mm`` requires both dims divisible by 16.
        # Fall back to BF16 silently when the shape doesn't satisfy
        # this — production's ``b_proj`` (1536 -> 12) hits this.
        if not bf16_only and (in_features % 16 != 0 or out_features % 16 != 0):
            bf16_only = True
        self.bf16_only = bf16_only

        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, device=device, dtype=dtype or torch.bfloat16),
        )
        if bias:
            self.bias = nn.Parameter(
                torch.empty(out_features, device=device, dtype=dtype or torch.bfloat16),
            )
        else:
            self.register_parameter("bias", None)

        self._init_weights()

    def _init_weights(self) -> None:
        # Kaiming-uniform matches nn.Linear's default reset (kaiming_uniform_
        # with a=sqrt(5) is the canonical "Linear init" — same as nn.Linear).
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            # ``nn.Linear`` init: bias ~ Uniform(-1/sqrt(in_features), 1/sqrt(in_features)).
            fan_in = self.in_features
            bound = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 0.0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.bf16_only:
            return F.linear(x, self.weight, self.bias)
        return _FP8E4M3Matmul.apply(x, self.weight, self.bias)

    def extra_repr(self) -> str:
        storage = "BF16(bf16_only)" if self.bf16_only else "FP8-E4M3 (W8A8, per-row + per-channel)"
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, weight_storage={storage}"
        )