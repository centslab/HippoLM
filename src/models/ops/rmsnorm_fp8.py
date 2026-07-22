"""Fused RMSNorm → FP8 — producer-side quant for KDA / FFN GEMMs.

Every transformer layer's ``attn_norm`` and ``mlp_norm`` precedes a
FP8 GEMM (KDA q/k/v/o or FFN gate/up). Without fusion, the path is:

    [1] RMSNorm   (BF16 → BF16, fused fla kernel)
    [2] quantize  (BF16 → FP8 + per-row scale)

Two kernel launches, two HBM round-trips for the intermediate BF16
normalized buffer. The fused single-pass kernel merges them into one
launch: each program owns a ``BLOCK_M × BLOCK_K`` tile, computes
``y = x * rstd * weight`` in registers (FP32, narrowed to BF16 to match
the standalone pipeline's rounding), computes the per-row amax on the
result, requantizes to FP8, and writes the FP8 output plus the new
per-row scale. The rstd is also written out so the backward can use
the standard RMSNorm gradient formula.

Numerics: matches the 2-launch baseline up to ``<1%`` sig_rel (FP8
quant is deterministic; the only delta is the per-row amax precision,
which is BF16-truncated both here and in the existing ``_quantize_
act_fp8_fused_singlepass_kernel``).

Backward: STE on the FP8 round, but the **proper** RMSNorm backward
for ``dL/dx`` and ``dL/dweight``. This matters because the weight
parameter must keep receiving a non-zero gradient signal — pure
STE passthrough (the recipe used in :mod:`fp8_residual` and
:mod:`silu_mul_fp8`) would zero out ``dL/dweight`` for the layer,
which is wrong. The formula is:

    y_pre_weight = x * rstd                                    (BF16 in HBM, FP32 in compute)
    dL/dweight  = sum_rows(dL/dy * y_pre_weight)               (proper RMSNorm bwd)
    dL/dx       = dL/dy * weight * rstd                        (proper RMSNorm bwd)

The FP8 round is treated as identity (STE); the BF16 RMSNorm is the
"differentiable surrogate" the gradient flows through. Numerically
identical to the unfused path's backward within BF16 rounding.

The kernel mirrors ``_quantize_act_fp8_fused_singlepass_kernel`` in
``src/models/ops/nvfp4_linear_w4a8.py`` so the output is **drop-in
compatible** with subsequent ``_scaled_mm`` calls — same
``(scale_norm, fp8_norm)`` layout the KDA / FFN linears consume.
"""
import torch
import triton
import triton.language as tl


# FP8 E4M3 representation bounds (matches the per-row quant kernel in
# nvfp4_linear_w4a8.py: scale truncated to BF16, then back to FP32,
# so that rounding noise from the BF16 master copies of the
# production linears lands on equal footing in both quant kernels).
_E4M3_MAX = 448.0
_E4M3_MIN_NORMAL = 6.103515625e-05
_SINGLEPASS_BLOCK_M = 8
# Singlepass requires the row tile to fit in registers; 4096 was set
# in the probe (matches the prod FFN intermediate K=4096 and the KDA
# head_dim*num_heads=1536 — both common shapes).
_SINGLEPASS_MAX_K = 4096


@triton.jit
def _rmsnorm_fp8_kernel(
    X_ptr,             # [M, K] BF16 input
    W_ptr,             # [K] BF16 weight (RMSNorm affine)
    S_ptr,             # [M, 1] FP32 per-row output scale (FP8 quant)
    R_ptr,             # [M] FP32 per-row rstd (saved for backward)
    Y_ptr,             # [M, K] float8_e4m3fn — output
    M, K,
    stride_xm, stride_xk,
    stride_w,
    stride_sm,
    stride_rm,
    stride_ym, stride_yk,
    eps,
    E4M3_MAX: tl.constexpr,
    E4M3_MIN_NORMAL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Single-pass fused ``RMSNorm → FP8``.

    Each program owns a ``BLOCK_M × BLOCK_K`` tile. Loads ``x`` and
    ``weight`` once into registers, computes the per-row rstd, applies
    the affine weight, narrows to BF16 (matching the standalone 2-launch
    pipeline), computes the per-row amax, requantizes to FP8 + scale,
    and writes out.
    """
    pid_m = tl.program_id(0)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_K)
    mask_m = offs_m[:, None] < M
    mask_k = offs_k[None, :] < K

    # Single load of the full row tile + weight into registers.
    x_ptrs = X_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
    x = tl.load(
        x_ptrs,
        mask=mask_m & mask_k,
        other=tl.cast(0.0, tl.bfloat16),
    ).to(tl.float32)

    w = tl.load(W_ptr + offs_k * stride_w, mask=offs_k < K, other=0.0).to(tl.float32)

    # Per-row sum of squares → rstd (matching the fla fused kernel's
    # formula: rstd = (sum_sq / K + eps)^(-1/2), eps applied in FP32).
    x_sq = tl.where(mask_k, x * x, 0.0)
    sum_sq = tl.sum(x_sq, axis=1)                # [BLOCK_M]
    rstd = 1.0 / tl.sqrt(sum_sq / K + eps)      # [BLOCK_M]

    # Write rstd for backward (one FP32 per row).
    tl.store(R_ptr + offs_m * stride_rm, rstd, mask=offs_m < M)

    # Apply RMSNorm: y = x * rstd * w  in FP32, then narrow to BF16
    # (mirrors the standalone 2-launch pipeline's rounding so the
    # FP8 quant scale computation matches).
    y = x * rstd[:, None] * w[None, :]
    y_bf16 = y.to(tl.bfloat16).to(tl.float32)

    # Per-row amax across K.
    y_abs = tl.where(mask_k, tl.abs(y_bf16), 0.0)
    amax = tl.max(y_abs, axis=1)
    amax = tl.maximum(amax, E4M3_MIN_NORMAL)

    # BF16 truncate the scale — matches the existing
    # ``_quantize_act_fp8_fused_singlepass_kernel``.
    scale = (amax / E4M3_MAX).to(tl.bfloat16).to(tl.float32)
    inv = (E4M3_MAX / amax).to(tl.bfloat16).to(tl.float32)
    tl.store(S_ptr + offs_m * stride_sm, scale, mask=offs_m < M)

    # Quantize from the same resident tile (no HBM re-read).
    yq = tl.minimum(
        tl.maximum(y_bf16 * inv[:, None], -E4M3_MAX),
        E4M3_MAX,
    ).to(tl.float8e4nv)

    y_ptrs = Y_ptr + offs_m[:, None] * stride_ym + offs_k[None, :] * stride_yk
    tl.store(y_ptrs, yq, mask=mask_m & mask_k)


def rmsnorm_fp8_fwd(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    out: torch.Tensor | None = None,
    scale: torch.Tensor | None = None,
    rstd_out: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Single-pass fused RMSNorm → FP8 (forward only).

    Args:
        x: ``[M, K]`` BF16 input.
        weight: ``[K]`` BF16 RMSNorm affine weight.
        eps: epsilon for the rstd denominator.
        out: optional pre-allocated output ``[M, K]`` FP8 buffer.
        scale: optional pre-allocated ``[M, 1]`` FP32 output scale.
        rstd_out: optional pre-allocated ``[M]`` FP32 rstd buffer
            (saved for the autograd backward).

    Returns:
        ``(fp8_out, scale_out, rstd_out)`` — drop-in for the next
        GEMM's act_quant contract.

    Notes:
        ``K`` must be ≤ 4096 (singlepass max). Production K is
        hidden_size=1024 and intermediate_size=2736 — both inside
        the budget.
    """
    M, K = x.shape
    assert weight.shape == (K,)
    assert x.dtype == torch.bfloat16
    assert weight.dtype == torch.bfloat16
    if out is None:
        out = torch.empty(M, K, dtype=torch.float8_e4m3fn, device=x.device)
    if scale is None:
        scale = torch.empty(M, 1, dtype=torch.float32, device=x.device)
    if rstd_out is None:
        rstd_out = torch.empty(M, dtype=torch.float32, device=x.device)

    BLOCK_K = triton.next_power_of_2(K)
    if BLOCK_K > _SINGLEPASS_MAX_K:
        raise NotImplementedError(
            f"rmsnorm_fp8 singlepass K limit ({_SINGLEPASS_MAX_K}); "
            f"got K={K}. Production uses K ≤ 4096 — if you really "
            f"need larger, add a chunked-K loop in the kernel."
        )

    num_warps = 4 if K <= 256 else 8
    grid = (triton.cdiv(M, _SINGLEPASS_BLOCK_M),)
    _rmsnorm_fp8_kernel[grid](
        x, weight, scale, rstd_out, out,
        M, K,
        x.stride(0), x.stride(1),
        weight.stride(0),
        scale.stride(0),
        rstd_out.stride(0),
        out.stride(0), out.stride(1),
        eps=eps,
        E4M3_MAX=_E4M3_MAX,
        E4M3_MIN_NORMAL=_E4M3_MIN_NORMAL,
        BLOCK_M=_SINGLEPASS_BLOCK_M,
        BLOCK_K=BLOCK_K,
        num_warps=num_warps,
    )
    return out, scale, rstd_out


class RmsNormFp8STE(torch.autograd.Function):
    """Fused RMSNorm → FP8 with STE backward.

    Forward: ``RMSNorm(x, w) → quant → fp8 → dequant (BF16)``.
    Backward (STE on the FP8 round, proper RMSNorm bwd on the math):
        ``y_pre_weight = x * rstd`` (re-derived; ``x`` and ``rstd``
        are saved as BF16 / FP32).
        ``dL/dweight = sum_rows(dL/dy * y_pre_weight)``
        ``dL/dx      = dL/dy * weight * rstd``

    The FP8 round is treated as identity (no gradient flowing through
    the requantization); the BF16 RMSNorm is the "differentiable
    surrogate" the gradient flows through. Numerically identical to
    the unfused path's backward within BF16 rounding.

    When constructed via :func:`rmsnorm_fp8_with_passthrough`, returns
    a tuple ``(y_bf16, out_fp8, scale)`` instead of just ``y_bf16`` —
    the caller can route ``out_fp8 / scale`` directly into a
    ``forward_precomputed`` consumer (FFN/KDA linear) to skip the
    redundant downstream ``quantize_act_fp8_fused``. The autograd
    graph still flows through ``y_bf16``, so the backward is unchanged.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor, eps: float,
                return_fp8: bool = False):
        orig_shape = x.shape
        K = orig_shape[-1]
        x_2d = x.reshape(-1, K).contiguous()
        out_fp8, scale, rstd = rmsnorm_fp8_fwd(x_2d, weight, eps)
        # Dequantize for the downstream BF16 consumer.
        y_2d = (out_fp8.float() * scale).to(torch.bfloat16)
        y = y_2d.reshape(orig_shape)
        # Save for backward. ``x`` is BF16 [M, K]; ``weight`` is BF16
        # [K]; ``rstd`` is FP32 [M]. Together these let us re-derive
        # y_pre_weight = x * rstd without re-running the kernel.
        ctx.save_for_backward(x_2d, weight, rstd)
        ctx.eps = eps
        ctx.orig_shape = orig_shape
        if return_fp8:
            # Tuple return: (bf16_for_autograd, fp8_for_precomputed,
            # scale_for_precomputed). Only ``y`` is part of the
            # autograd graph; ``out_fp8`` / ``scale`` are auxiliary
            # tensors the caller routes to ``forward_precomputed``
            # consumers. No gradient flows to them — the FP8 round
            # is STE (treated as identity in backward).
            return y, out_fp8, scale
        return y

    @staticmethod
    def backward(ctx, grad_y: torch.Tensor, grad_fp8=None, grad_scale=None):
        """Backward for the (y_bf16,) or (y_bf16, out_fp8, scale) tuple.

        The optional ``grad_fp8`` / ``grad_scale`` are the upstream
        gradients on the FP8 + scale auxiliary outputs (only present
        when ``return_fp8=True`` was used). Both are None — no
        gradient flows through the FP8 round (STE convention).
        """
        x_2d, weight, rstd = ctx.saved_tensors
        K = x_2d.shape[-1]
        grad_y_2d = grad_y.reshape(-1, K).contiguous()

        x_fp32 = x_2d.float()
        w_fp32 = weight.float()
        rstd_col = rstd[:, None]               # [M, 1]

        # y_pre_weight = x * rstd  (the BF16-normalized value before
        # the affine weight). Used for both dL/dx and dL/dweight.
        y_pre_weight = x_fp32 * rstd_col

        # dL/dweight = sum_rows(dL/dy * y_pre_weight), shape [K],
        # narrowed back to BF16 to match the input weight dtype.
        d_weight = (grad_y_2d.float() * y_pre_weight).sum(dim=0).to(weight.dtype)

        # dL/dx = dL/dy * weight * rstd
        d_x = (grad_y_2d.float() * w_fp32[None, :] * rstd_col).to(x_2d.dtype)
        d_x = d_x.reshape(ctx.orig_shape)

        # 4 grads returned (x, weight, eps, return_fp8); the last
        # two are non-tensor knobs.
        return d_x, d_weight, None, None


def rmsnorm_fp8_with_passthrough(
    x: torch.Tensor, weight: torch.Tensor, eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply :class:`RmsNormFp8STE` and return ``(y_bf16, out_fp8, scale)``.

    Convenience wrapper around ``RmsNormFp8STE.apply(x, w, eps, return_fp8=True)``.
    The returned ``out_fp8`` / ``scale`` can be passed directly to a
    :func:`forward_precomputed` consumer (FFN / KDA linear) to skip
    the redundant downstream ``quantize_act_fp8_fused`` call —
    saving ~105 us per FFN at prod shape (1024, 1536).

    The ``y_bf16`` is what autograd tracks; the FP8 round is STE.
    """
    return RmsNormFp8STE.apply(x, weight, eps, True)
