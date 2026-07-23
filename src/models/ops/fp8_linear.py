"""W8A8 FP8 Linear (E4M3) — drop-in ``nn.Linear`` replacement.

The forward uses ``torch._scaled_mm`` with E4M3 on both sides:

    x (BF16, [..., K])
        -> per-row E4M3 quant   (scale_a shape [M, 1], FP32)
    w (BF16, [N, K])
        -> per-row (per-output-channel) E4M3 quant   (scale_b shape [1, N], FP32)
    y = x_q @ w_q.T  via _scaled_mm(scale_a, scale_b) -> BF16

The autograd design mirrors ``src/models.ops.nvfp4_linear``'s
mode-2 pattern: the BF16 master ``weight`` is the leaf parameter
the optimizer updates; the FP8 quantized view is computed on the
fly in forward and discarded; the backward is STE — it
re-runs BF16 matmul against the BF16 weight (the optimizer sees
the standard ``grad_w``).

Two backward paths
==================

The constructor takes ``fp8_bwd: bool = True`` (default since
2026-07-21):

- ``fp8_bwd=True`` (default) — FP8 E4M3 backward via two additional
  ``_scaled_mm`` calls (one for ``dx``, one for ``dW``). Same
  scheme as the FFN W4A8 autograd contract (gradients are
  themselves FP8-quantized, not STE). The ``_scaled_mm`` CUTLASS
  path on sm_120 requires B to be col-major (``stride(0) == 1``).
  Both bwd GEMMs need a col-major B operand, so each requires
  re-quantizing the input in a transposed layout. The three
  transposed quants (w, grad_out, x) are computed by
  ``quantize_act_fp8_fused_transposed`` — a single Triton kernel
  that fuses per-col amax + FP8 cast + transposed write in one
  pass. At KDA prod shape the fused path is ~2.95x faster than
  the explicit ``.T.contiguous() + quantize`` baseline (~335 us
  vs ~990 us per transposed-quant call). The full FP8 bwd path
  is ~1.7-1.9x faster than the BF16 STE bwd path at q_proj shape
  (M=16384 K=N=1536) and ~1.74x faster than the now-fixed
  FP32-cast BF16 bwd. The numerical gap vs the BF16 reference is
  ~2-3% sig_rel on grad_x and ~1% on grad_w. Recommended for
  large M (>= 8k) — see ``test_kda_fp8.py`` for the regression
  coverage.

- ``fp8_bwd=False`` — BF16 STE backward. Re-runs BF16 matmul
  against the BF16 leaves. ``grad_w`` is bit-equivalent to a plain
  ``F.linear`` reference at the BF16 rounding-noise floor
  (cos_sim 1.0). Used by tests / A/B comparisons and the FFN
  W4A8 NVFP4 path (which keeps BF16 STE bwd even though the
  forward is W4A8). Note: the historical FP32-cast BF16 bwd was
  a perf bug (auto-memory ``project_fp8_bwd_fp32_bug.md``) — that
  path is now fixed; the bwd stays in BF16 throughout.

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

Autograd contract (STE for the BF16-bwd path):

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

For the ``fp8_bwd=True`` path the backward contract is:

    grad_x = quantize(grad_out) @ quantize(w.T).T   via _scaled_mm
    grad_w = quantize(grad_out.T) @ quantize(x.T).T via _scaled_mm
    grad_b = grad_out.sum(dim=0)                   (BF16 reduction)

The two transposes (w -> [K,N], x -> [K,M]) are needed because
``torch._scaled_mm`` on sm_120 requires B in col-major layout
(stride(0) == 1). Re-quantizing in transposed layout gives B in
col-major form (via the .T view). Same noise floor as the
feasibility sweep (~2-3% sig_rel on grad_x, ~1% on grad_w).

bf16_only escape hatch
----------------------
``bf16_only=True`` skips quantization and falls back to plain
BF16 GEMM. Useful for A/B tests against the FP8 path without
re-instantiating the model. Same numerics as ``nn.Linear``.
"""
from __future__ import annotations

import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# FP8 E4M3 hardware constants (sm_120 / Blackwell consumer tensor cores).
_E4M3_MAX = 448.0
_E4M3_MIN_NORMAL = 6.103515625e-05


def _quantize_w_per_channel(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-output-channel E4M3 quant for w [N, K].

    Returns (w_q, scale_w) where w_q is [N, K] FP8 and scale_w is
    [1, N] FP32. Used by the forward path; the BF16 leaf weight is
    quantized on-the-fly each call (no caching — see
    ``project_fp8_linear_quant_fusion.md`` for the followup to
    cache w_q on the module).

    Per-output-channel == per-row of the [N, K] weight over K, which
    is exactly what ``quantize_act_fp8_fused`` (the per-row fused
    Triton kernel) computes. We delegate to it rather than the
    PyTorch broadcast path (``w.to(fp32)`` materialize + amax +
    divide + cast): at N=K=1536 the fused kernel is ~2.5x faster
    (56 us vs 140 us) and numerically equivalent (the fused kernel
    truncates the FP32 scale to BF16, matching the activation-quant
    convention already used in the forward — reconstruct sig_rel
    2.646% broadcast vs 2.660% fused, i.e. at the FP8 noise floor).
    The bwd path already uses the fused (transposed) kernel for w;
    this makes the fwd consistent. See
    ``project_fp8_perf_root_cause.md``.
    """
    from src.models.ops.nvfp4_linear_w4a8 import quantize_act_fp8_fused
    # w is a contiguous [N, K] nn.Parameter; per-row scale is [N, 1].
    w_q, scale_w_col = quantize_act_fp8_fused(w)            # [N, K] FP8, [N, 1] FP32
    scale_w = scale_w_col.T.contiguous()                    # [1, N] FP32
    return w_q, scale_w


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
        # Activation quant: use the Triton fused kernel from
        # ``nvfp4_linear_w4a8.quantize_act_fp8_fused``. The original
        # PyTorch broadcast path (x.to(fp32) → amax → divide → cast)
        # cost ~2.65 ms per call at KDA prod shape (M=16384 K=1536);
        # the fused kernel is 8.2× faster (~0.32 ms) because the amax
        # stays in registers across the two passes and the FP32
        # intermediate is never materialized. See auto-memory
        # ``project_fp8_linear_quant_fusion.md``.
        #
        # The kernel truncates the FP32 scale to BF16 precision (matches
        # the FFN W4A8 path's existing convention) — gives 0.62% med_rel
        # shift vs the prior PyTorch FP32-scale path, well under the FP8
        # ~3.7% noise floor. Tests in ``test/test_kda_fp8.py`` still pass.
        from src.models.ops.nvfp4_linear_w4a8 import quantize_act_fp8_fused
        x_2d = x.reshape(-1, x.shape[-1])
        x_q, scale_x = quantize_act_fp8_fused(x_2d)

        w_q, scale_w = _quantize_w_per_channel(w)

        # Custom CUDA C++ FP8 GEMM via the runtime dispatcher
        # (per-arch `.so` discovery + per-(M, K, N) micro-bench at
        # first call). At prod M=1024 this picks BM64/BN64 over the
        # CUTLASS `_scaled_mm` path and saves ~13-29% on the GEMM.
        # Falls back to ``torch._scaled_mm`` automatically if no
        # prebuilt .so matches the current arch (e.g. on a fresh env
        # without ``scripts/build_fp8_gemm.py``). See
        # ``docs/fp8_gemm_kernel_pipeline.md`` and auto-memory
        # ``project_fp8_dispatch_by_m.md``.
        from src.models.ops.cuda.fp8_gemm_dispatch import fp8_gemm_auto_dispatch
        out_2d = fp8_gemm_auto_dispatch(x_q, w_q, scale_x, scale_w)

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
        # KEEP BF16 throughout — do NOT cast to float32 here. The
        # historical FP32-cast (auto-memory
        # project_fp8_bwd_fp32_bug.md) was a perf bug: FP32 GEMM is
        # ~1.87x slower than BF16 GEMM on sm_120, and the cast costs
        # ~1 ms per call. BF16 STE is bit-equivalent to a plain
        # ``F.linear`` backward at the BF16 rounding-noise floor
        # (cos_sim 1.0, max abs diff at BF16 eps).
        grad_out_2d = grad_out.reshape(-1, N)
        x_2d = x.reshape(-1, K)
        w_2d = w
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
        return grad_x, grad_w, grad_bias


class _FP8E4M3MatmulFP8Bwd(torch.autograd.Function):
    """BF16 leaf + FP8 forward AND backward via ``torch._scaled_mm``.

    Forward: same per-row-act / per-channel-weight E4M3 GEMM as
    :class:`_FP8E4M3Matmul` (identical numerics — the FP8 forward
    is unchanged).

    Backward: two additional ``_scaled_mm`` calls.

        dx = grad_out @ w     via _scaled_mm
        dW = grad_out.T @ x   via _scaled_mm
        db = grad_out.sum(0)  (BF16 reduction, unchanged)

    The ``_scaled_mm`` CUTLASS path on sm_120 requires B in
    col-major layout (``stride(0) == 1``). Both bwd GEMMs need a
    col-major B operand, so each requires re-quantizing the
    input in a *transposed* layout:

        dx path: B = w_q_T.T (col-major [N, K])  → quantize w in [K, N]
        dW path: B = x_q_T.T (col-major [M, K])  → quantize x in [K, M]

    Each transposed quant is one physical transpose copy (BF16
    [N, K] or [M, K]) + one Triton fused quantize kernel.
    Re-quantizing on-the-fly in bwd is cheap (~0.3 ms each) and
    avoids adding FP8 tensors to ``saved_tensors`` (which would
    inflate the autograd graph memory).

    Numerical noise vs the BF16 reference at the KDA feasibility
    shapes (per ``project_fp8_kda_feasibility.md``):
        grad_x sig_rel ~3.3-3.7%, cos_sim 0.9993-0.9994
        grad_w sig_rel ~3.4-3.8%, cos_sim 0.9993-0.9994
    """

    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        x: torch.Tensor,
        w: torch.Tensor,
        bias: torch.Tensor | None,
    ) -> torch.Tensor:
        from src.models.ops.nvfp4_linear_w4a8 import quantize_act_fp8_fused
        x_2d = x.reshape(-1, x.shape[-1])
        x_q, scale_x = quantize_act_fp8_fused(x_2d)
        w_q, scale_w = _quantize_w_per_channel(w)
        # Dispatcher swap (see ``_FP8E4M3Matmul.forward`` comment).
        from src.models.ops.cuda.fp8_gemm_dispatch import fp8_gemm_auto_dispatch
        out_2d = fp8_gemm_auto_dispatch(x_q, w_q, scale_x, scale_w)
        if bias is not None:
            out_2d = out_2d + bias
        out = out_2d.reshape(*x.shape[:-1], w.shape[0])
        ctx.save_for_backward(x, w)
        ctx.has_bias = bias is not None
        ctx.input_shape = x.shape
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):  # type: ignore[override]
        from src.models.ops.nvfp4_linear_w4a8 import (
            quantize_act_fp8_fused,
            quantize_act_fp8_fused_transposed,
        )
        x, w = ctx.saved_tensors
        K = w.shape[1]
        N = w.shape[0]
        grad_out_2d = grad_out.reshape(-1, N)                # [M, N]
        M = grad_out_2d.shape[0]

        # ---- dx = grad_out @ w (FP8 _scaled_mm) ---- #
        # A = grad_out_q [M, N] row-major (per-row scale over N) — natural,
        # no transpose needed; the per-row kernel handles it.
        go_q_dx, go_scale_dx = quantize_act_fp8_fused(grad_out_2d)   # [M, N], scale [M, 1]

        # B = col-major [N, K] — re-quantize w in transposed [K, N] layout
        # via the fused transposed quant kernel. The w nn.Parameter is
        # already contiguous, so we pass it directly.
        w_T_q, w_T_scale = quantize_act_fp8_fused_transposed(w)     # [K, N] FP8, scale [K, 1]
        # w_T_q.T has shape [N, K] and strides (1, K) — col-major [N, K] ✓
        # scale_b is per-col of B [N, K] col-major = per-row of w_T_q [K, N] over N
        # = w_T_scale [K, 1]. The .T is a view (no copy) so w_T_scale_T is
        # free.
        w_T_scale_T = w_T_scale.T.contiguous()               # [1, K]

        grad_x_2d = torch._scaled_mm(
            go_q_dx, w_T_q.T,
            go_scale_dx, w_T_scale_T,
            out_dtype=torch.bfloat16,
        )                                                      # [M, K]

        # ---- dW = grad_out.T @ x (FP8 _scaled_mm) ---- #
        # A = grad_out_T_q [N, M] row-major (per-row scale over M).
        # B = col-major [M, K] — re-quantize x in transposed [K, M] layout.
        go_T_q, go_T_scale = quantize_act_fp8_fused_transposed(grad_out_2d)  # [N, M] FP8, scale [N, 1]

        x_2d = x.reshape(-1, K).contiguous()                  # [M, K] BF16 (contiguous)
        x_T_q, x_T_scale = quantize_act_fp8_fused_transposed(x_2d)         # [K, M] FP8, scale [K, 1]
        # x_T_q.T has shape [M, K] and strides (1, M) — col-major [M, K] ✓
        x_T_scale_T = x_T_scale.T.contiguous()                # [1, K]

        grad_w = torch._scaled_mm(
            go_T_q, x_T_q.T,
            go_T_scale, x_T_scale_T,
            out_dtype=torch.bfloat16,
        )                                                      # [N, K]

        # Reshape + bias grad
        grad_x = grad_x_2d.reshape(*ctx.input_shape)
        grad_w_out = grad_w.to(w.dtype)
        grad_bias = grad_out_2d.sum(dim=0) if ctx.has_bias else None
        return grad_x, grad_w_out, grad_bias


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
        fp8_bwd      : FP8 E4M3 backward via two ``_scaled_mm`` calls
                       (per-row-act + per-channel-weight). Same
                       scheme as the FFN W4A8 autograd contract
                       (gradients are themselves FP8-quantized, not
                       STE). Default ``True`` since 2026-07-21: the
                       prior BF16-STE default was using an FP32-cast
                       bug (fixed in the same commit — see
                       ``_FP8E4M3Matmul.backward``). At q_proj shape
                       (M=16384 K=N=1536) the FP8 bwd is ~1.74x faster
                       than the old FP32-cast BF16 bwd and ~equal to
                       the BF16 STE bwd; at small M (eval / tests) it
                       can be slightly slower because of the four
                       quantize passes + three transpose copies. Set
                       to ``False`` to force BF16 STE backward (useful
                       for A/B tests against the BF16 reference).

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
        fp8_bwd: bool = True,
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
        self.fp8_bwd = fp8_bwd

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

        # 2026-07-23: surfaced for debug — verify which FP8 path
        # is actually wired for a given layer (and whether the
        # ``bf16_only`` shape-div-16 silent fallback fired).
        resolved = (
            "BF16(bf16_only)" if self.bf16_only
            else ("FP8-W8A8 (FP8 bwd)" if self.fp8_bwd else "FP8-W8A8 (BF16 STE bwd)")
        )
        logger.info(
            f"FP8Linear: in={in_features} out={out_features}"
            f" bias={bias} dtype={self.weight.dtype}"
            f" device={self.weight.device}"
            f" → {resolved}"
        )

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
        if self.fp8_bwd:
            return _FP8E4M3MatmulFP8Bwd.apply(x, self.weight, self.bias)
        return _FP8E4M3Matmul.apply(x, self.weight, self.bias)

    def extra_repr(self) -> str:
        if self.bf16_only:
            storage = "BF16(bf16_only)"
        elif self.fp8_bwd:
            storage = "FP8-E4M3 (W8A8, per-row + per-channel, FP8 bwd)"
        else:
            storage = "FP8-E4M3 (W8A8, per-row + per-channel, BF16 STE bwd)"
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, weight_storage={storage}"
        )