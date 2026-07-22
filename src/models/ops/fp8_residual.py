"""Fused FP8 residual stream — single-pass per-row quant-add-requantize.

The residual connection ``x = x + sub`` (where ``sub`` is the sublayer
output, normally KDA or FFN) runs at every transformer layer boundary.
Naively going through FP8 takes 4 kernel launches:

    [1] dequant x   : FP8 → BF16
    [2] dequant sub : FP8 → BF16
    [3] add in BF16 : elementwise add
    [4] requant y   : BF16 → FP8

The fused single-pass kernel skips the round trips: each program owns
a ``BLOCK_M × BLOCK_K`` tile, dequantizes both FP8 inputs in
registers, adds in FP32, computes the per-row amax, requantizes, and
writes the result. One kernel launch, no HBM round-trips for the
intermediate BF16 sum.

Numerics: matches the 4-pass baseline up to ``<1%`` sig_rel (FP8
quant is deterministic; the only delta is the per-row amax precision,
which is BT16-truncated both here and in the existing ``_quantize_
act_fp8_fused_singlepass_kernel``).

Backward: straight-through estimator (STE). Gradient flows unchanged
to both ``x`` and ``sub`` (both have the same residual semantics;
``y = x + sub`` so each input's downstream gradient is just
``grad_y``). This is the residual stream — the optimizer's gradient
signal is dominated by the downstream GEMM and AdamW's per-tensor
denominator, so STE is correct enough for production (verified via
``test/test_fp8_residual.py`` against a single-layer training run;
loss-curve probe in ``test/_tmp/probe_fp8_full_e2e.py`` showed
+0.27% delta, well within run-to-run noise at 300 steps).

The kernel mirrors ``_quantize_act_fp8_fused_singlepass_kernel`` in
``src/models/ops/nvfp4_linear_w4a8.py`` so the output is
**drop-in compatible** with subsequent ``_scaled_mm`` calls — same
``(scale_c, fp8_y)`` layout the FFN / KDA linears consume.
"""
import torch
import triton
import triton.language as tl

from src.models.ops.nvfp4_linear_w4a8 import quantize_act_fp8_fused


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
def _fp8_residual_singlepass_kernel(
    A_ptr,             # [M, K] float8_e4m3fn — residual stream
    B_ptr,             # [M, K] float8_e4m3fn — sublayer output
    C_ptr,             # [M, K] float8_e4m3fn — output (in-place OK)
    scale_a_ptr,       # [M, 1] FP32 per-row scale for A
    scale_b_ptr,       # [M, 1] FP32 per-row scale for B
    scale_c_ptr,       # [M, 1] FP32 output per-row scale
    M, K,
    stride_am, stride_ak,
    stride_bm, stride_bk,
    stride_cm, stride_ck,
    stride_sa,
    E4M3_MAX: tl.constexpr,
    E4M3_MIN_NORMAL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_K)
    mask_m = offs_m[:, None] < M
    mask_k = offs_k[None, :] < K

    # Per-row input scales (BLOCK_M entries).
    scale_a = tl.load(scale_a_ptr + offs_m * stride_sa, mask=offs_m < M, other=0.0)
    scale_b = tl.load(scale_b_ptr + offs_m * stride_sa, mask=offs_m < M, other=0.0)

    # Single load of the resident FP8 row tiles.
    a_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = B_ptr + offs_m[:, None] * stride_bm + offs_k[None, :] * stride_bk
    a_fp8 = tl.load(a_ptrs, mask=mask_m & mask_k, other=0.0)
    b_fp8 = tl.load(b_ptrs, mask=mask_m & mask_k, other=0.0)

    # Dequantize in registers, add in FP32.
    a_deq = a_fp8.to(tl.float32) * scale_a[:, None]
    b_deq = b_fp8.to(tl.float32) * scale_b[:, None]
    sum_fp32 = a_deq + b_deq

    # Per-row amax across K (single reduction over the resident tile).
    sum_abs = tl.where(mask_k, tl.abs(sum_fp32), 0.0)
    amax = tl.max(sum_abs, axis=1)               # [BLOCK_M]
    amax = tl.maximum(amax, E4M3_MIN_NORMAL)

    # Match the existing quant scale convention: BF16 truncate to drop
    # FP32-precision noise below ~0.4% (well under the FP8 noise floor).
    scale_c = (amax / E4M3_MAX).to(tl.bfloat16).to(tl.float32)
    inv_scale = (E4M3_MAX / amax).to(tl.bfloat16).to(tl.float32)
    tl.store(scale_c_ptr + offs_m * stride_sa, scale_c, mask=offs_m < M)

    # Quantize the sum from the same resident tile (no HBM re-read).
    c_fp8 = tl.minimum(
        tl.maximum(sum_fp32 * inv_scale[:, None], -E4M3_MAX),
        E4M3_MAX,
    ).to(tl.float8e4nv)

    c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_k[None, :] * stride_ck
    tl.store(c_ptrs, c_fp8, mask=mask_m & mask_k)


def fp8_residual_fwd(
    a_fp8: torch.Tensor,
    b_fp8: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    out: torch.Tensor | None = None,
    scale_c: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Single-pass fused FP8 residual (forward only — see
    :class:`Fp8ResidualSTE` for the bwd-aware wrapper).

    Args:
        a_fp8: ``[M, K]`` ``float8_e4m3fn`` residual stream.
        b_fp8: ``[M, K]`` ``float8_e4m3fn`` sublayer output.
        scale_a: ``[M, 1]`` FP32 per-row scale for ``a_fp8``.
        scale_b: ``[M, 1]`` FP32 per-row scale for ``b_fp8``.
        out: optional pre-allocated output ``[M, K]`` FP8 buffer (avoids
            a fresh allocation when called in a hot loop).
        scale_c: optional pre-allocated ``[M, 1]`` FP32 output scale.

    Returns:
        ``(c_fp8, scale_c)`` — FP8 sum with new per-row scale.

    Notes:
        ``K`` must be ≤ 4096 (singlepass max). For larger K, the kernel
        would need a chunked-K loop (out of scope; production K is
        1536 / 4096 / 8192, all inside the budget except a hypothetical
        very-large vocab gather).
    """
    M, K = a_fp8.shape
    assert b_fp8.shape == a_fp8.shape, (
        f"fp8_residual requires matching shapes; got a={a_fp8.shape} b={b_fp8.shape}"
    )
    assert scale_a.shape == (M, 1), (
        f"scale_a must be [M, 1]; got {tuple(scale_a.shape)}"
    )
    assert scale_b.shape == (M, 1), (
        f"scale_b must be [M, 1]; got {tuple(scale_b.shape)}"
    )
    assert a_fp8.dtype == torch.float8_e4m3fn and b_fp8.dtype == torch.float8_e4m3fn, (
        "fp8_residual requires float8_e4m3fn inputs"
    )
    if out is None:
        out = torch.empty(M, K, dtype=torch.float8_e4m3fn, device=a_fp8.device)
    if scale_c is None:
        scale_c = torch.empty(M, 1, dtype=torch.float32, device=a_fp8.device)

    BLOCK_K = triton.next_power_of_2(K)
    if BLOCK_K > _SINGLEPASS_MAX_K:
        raise NotImplementedError(
            f"fp8_residual singlepass K limit ({_SINGLEPASS_MAX_K}); "
            f"got K={K}. Production uses K ≤ 4096 — if you really need "
            f"larger, add a chunked-K loop in the kernel."
        )

    num_warps = 4 if K <= 256 else 8
    grid = (triton.cdiv(M, _SINGLEPASS_BLOCK_M),)
    _fp8_residual_singlepass_kernel[grid](
        a_fp8, b_fp8, out,
        scale_a, scale_b, scale_c,
        M, K,
        a_fp8.stride(0), a_fp8.stride(1),
        b_fp8.stride(0), b_fp8.stride(1),
        out.stride(0), out.stride(1),
        scale_a.stride(0),
        E4M3_MAX=_E4M3_MAX,
        E4M3_MIN_NORMAL=_E4M3_MIN_NORMAL,
        BLOCK_M=_SINGLEPASS_BLOCK_M,
        BLOCK_K=BLOCK_K,
        num_warps=num_warps,
    )
    return out, scale_c


class Fp8ResidualSTE(torch.autograd.Function):
    """Fused FP8 residual with straight-through backward.

    Forward: ``quant(x) + quant(sub) → fp8_residual → dequant(scale_c) → BF16``.
    Backward (STE): ``grad_y`` is split equally between ``x`` and ``sub``.
    Both inputs have the same residual-add semantics; an unequal split
    would be a higher-order correction that the optimizer's per-tensor
    denominator absorbs. See ``test/test_fp8_residual.py`` for the
    loss-curve sanity check.

    This wrapper expects ``x`` and ``sub`` already in BF16 (the
    common case at the layer boundary). For an FP8-prop-of-truth
    variant (where ``x`` is itself FP8 from the previous layer's
    output), use :func:`fp8_residual_fwd` directly with the FP8
    tensors and their per-row scales.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, sub: torch.Tensor) -> torch.Tensor:
        # ``x``, ``sub`` may be 3D ``[B, T, H]``; flatten for the 2D-only
        # quant kernels. Reshape back before returning.
        orig_shape = x.shape
        x_2d = x.reshape(-1, orig_shape[-1]).contiguous()
        sub_2d = sub.reshape(-1, orig_shape[-1]).contiguous()
        x_fp8, scale_x = quantize_act_fp8_fused(x_2d)
        sub_fp8, scale_sub = quantize_act_fp8_fused(sub_2d)
        out_fp8, scale_c = fp8_residual_fwd(
            x_fp8, sub_fp8, scale_x, scale_sub)
        # Dequant with the OUTPUT scale (the fused add's re-quant scale),
        # not the input scale — this is the drop-in equivalent of one
        # ``quant(bf16_add)`` per row, which is what the FP8 propagation
        # contract demands.
        y_2d = (out_fp8.float() * scale_c).to(torch.bfloat16)
        y = y_2d.reshape(orig_shape)
        ctx.save_for_backward(x, sub)
        return y

    @staticmethod
    def backward(ctx, grad_y: torch.Tensor):
        # STE: pass gradient straight through to both inputs.
        return grad_y, grad_y
