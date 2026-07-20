"""W4A8 NVFP4 Linear — two-pass forward (Triton dequant → fp8 + GEMM).

Forward path
-------------
Activation (FP8 E4M3, per-token):
    x (BF16, [..., K]) → per-row scale a_s [M] FP32 + FP8 E4M3 values a_fp8 [M, K]

Weight (NVFP4 E2M1, per-block scale + global):
    w (BF16, [N, K]) → quantize_nvfp4_with_global_scale → packed + s_e4m3 + g

Two-pass forward:
    pass1: dequant_nvfp4_fp8(packed, s_e4m3, g) → b_fp8 [N, K] float8_e4m3fn
    pass2: GEMM(a_fp8, b_fp8, scale_a=a_s, scale_b=g/boost) → [M, N] BF16

The pass-2 GEMM has two backends:
  - ``torch._scaled_mm`` (default, routes to cuBLAS nvjet) — 96-100 TFLOPS
    on sm_120 at prod FFN shapes (~30% of 400 TFLOPS fp8 spec).
  - ``fp8_gemm_scaled`` from :mod:`src.models.ops.cuda.fp8_gemm` (opt-in
    via ``use_custom_gemm=True``) — custom CUDA C++ TMA + warp
    specialization kernel. Measured 1.01-1.04x cuBLAS at prod shapes
    on sm_120. Requires ``scripts/build_fp8_gemm.py`` to have been
    run first (the .so is loaded lazily; if missing, falls back to
    ``_scaled_mm`` with a warning).

The 448-range fix (auto-boost): s_e4m3 spans 64..448 so e2m1*s products
can exceed ±448 (E4M3 clamp bound), causing ~49% mean_rel. Fix: pow2
boost C = 2^floor(log2(448/(6*smax))) applied to b before fp8 cast,
folded into the epilogue as g/C.

Autograd (STE — same as fp8_linear.FP8E4M3Matmul)
-----------------------------------------------------
The forward's NVFP4→fp8 quantization is discarded in bwd.
Backward re-runs:
    grad_x = grad_out @ w_bf16      (BF16 matmul)
    grad_w = grad_out.T @ x_bf16    (BF16 matmul)
The BF16 master weight is the optimizer leaf; packed/scales are
derived buffers recomputed from it each forward.

bf16_only escape hatch
----------------------
``bf16_only=True`` skips the NVFP4 path entirely and calls
``F.linear``. Bit-exact with ``nn.Linear``. Useful for shapes that
don't benefit from W4A8 or as a correctness reference.

Why NVFP4 → fp8 (not direct BF16 dequant)
------------------------------------------
Dequantizing NVFP4 → BF16 writes [N, K] BF16 = 50 MB at gate_up.
Dequantizing NVFP4 → FP8 writes [N, K] FP8  = 12 MB.
_fp8 GEMM runs at 96-100 TFLOPS on sm_120 (custom kernel) — 2-3x
faster than dequant→BF16+BF16 GEMM (which caps at ~50 TFLOPS,
the 5060 Ti BF16 chip peak).

Noise floor (inherent to NVFP4 E4M3 B rounding)
----------------------------------------------
Rounding e2m1*s to E4M3 before the GEMM costs ~2.1% mean_rel vs fp32 ref.
This is uniform across all shapes and applies to ANY fp8-B NVFP4 path,
including the CUDA Marlin W4A8 route. MXFP4 (E8M0 pow2 scales) avoids it
but requires different quantization; NVFP4 E2M1 is what we have.

State dict: BF16 master only (same as fp8_linear.FP8Linear).
No conversion needed to load a BF16 checkpoint.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

from src.models.ops.nvfp4_marlin import (
    _process_scales_for_marlin,
    _repack_for_marlin,
    quantize_nvfp4_with_global_scale,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_BLOCK_SIZE = 16          # NVFP4 block size (K dimension)
_E4M3_MAX = 448.0
_E4M3_MIN_NORMAL = 6.103515625e-05


# ---------------------------------------------------------------------------
# Auto-boost: per-tensor pow2 scale so e2m1*s*C stays in E4M3 range
# ---------------------------------------------------------------------------
def _auto_boost(scales_e4m3: torch.Tensor) -> float:
    """Pow2 boost C so max|e2m1 * s_e4m3 * C| ≤ _E4M3_MAX.

    max|e2m1| = 6, so the safe bound is C ≤ 448 / (6 * smax).
    Flooring to the nearest power of 2 costs nothing in precision
    (floating point is scale-invariant until subnormals).
    """
    smax = scales_e4m3.float().abs().max().item()
    if smax <= 0:
        return 1.0
    return 2.0 ** math.floor(math.log2(_E4M3_MAX / (6.0 * smax)))


# ---------------------------------------------------------------------------
# Triton kernel: per-row activation quant (BF16 → fp8 E4M3)
# ---------------------------------------------------------------------------
@triton.jit
def _quantize_act_fp8_kernel(
    X_ptr,           # [M, K] BF16
    X_scale_ptr,     # [M] FP32 per-row amax/E4M3_MAX
    X_out_ptr,       # [M, K] float8_e4m3fn
    M, K,
    stride_xm, stride_xk,
    stride_sm,
    stride_om, stride_ok,
    E4M3_MAX: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Quantize BF16 [M, K] -> fp8 E4M3 with pre-computed per-row scales.

    scale[m] = amax(|x[m, :|) / E4M3_MAX  (computed in Python before this
    kernel). Output is x_q = clamp(x / scale, +/-E4M3_MAX). Since
    ``scale`` already encodes the ``/E4M3_MAX`` factor, no extra
    multiply is needed inside the kernel — the equivalent Python
    expression is ``(x / amax * E4M3_MAX).clamp(...)``.

    Kept for tests + small-shape paths. The production forward uses
    :func:`quantize_act_fp8_fused` (which folds the per-row amax
    reduction into the same kernel — saves one Python amax launch
    + one scale-to-fp32 cast per forward, ~0.5 ms at prod shape).
    """
    pid_m = tl.program_id(0)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_K)

    scale = tl.load(
        X_scale_ptr + offs_m * stride_sm,
        mask=offs_m < M,
        other=1.0,
    ).to(tl.float32)

    x_ptrs = X_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
    o_ptrs = X_out_ptr + offs_m[:, None] * stride_om + offs_k[None, :] * stride_ok

    for k0 in range(tl.cdiv(K, BLOCK_K)):
        k_base = k0 * BLOCK_K
        mask_m = offs_m[:, None] < M
        mask_k = (k_base + offs_k)[None, :] < K
        x = tl.load(
            x_ptrs + k_base * stride_xk,
            mask=mask_m & mask_k,
            other=tl.cast(0.0, tl.bfloat16),
        ).to(tl.float32)
        x_q = tl.minimum(
            tl.maximum(x / scale[:, None], -E4M3_MAX),
            E4M3_MAX,
        ).to(tl.float8e4nv)
        tl.store(
            o_ptrs + k_base * stride_ok,
            x_q,
            mask=mask_m & mask_k,
        )


# ---------------------------------------------------------------------------
# Triton kernel: fused per-row amax + quantize (BF16 → fp8 E4M3)
# ---------------------------------------------------------------------------
@triton.jit
def _quantize_act_fp8_fused_kernel(
    X_ptr,           # [M, K] BF16 (input)
    X_scale_out_ptr, # [M, 1] FP32  (output: per-row amax/E4M3_MAX)
    X_out_ptr,       # [M, K] float8_e4m3fn (output)
    M, K,
    stride_xm, stride_xk,
    stride_sm,
    stride_om, stride_ok,
    E4M3_MAX: tl.constexpr,
    E4M3_MIN_NORMAL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Fused per-row amax reduction + quantize in one kernel.

    One program owns ``BLOCK_M`` rows. In pass 1 we walk all K and
    reduce per-row ``amax``; in pass 2 we walk all K again and write
    the fp8 quantization using the just-computed scale. The amax
    stays in registers between the two passes (the Triton compiler
    keeps the live values across the for-loop boundary since it can
    see they're loop-invariant w.r.t. ``k0``).

    This replaces two separate Python launches (``amax`` + the
    scale-to-fp32 cast) and a separate quant kernel with one
    fused launch — saves ~0.5 ms per forward at the prod
    gate_up shape (sm_120).

    Note on dtype: ``X_ptr`` is treated as BF16 (post-RMSNorm FFN
    input). The masked load uses ``other=0.0`` (BF16), then upcasts
    to FP32 for the amax + quantize math. If the model ever feeds
    FP32 in, change ``other`` and drop the upcast — the arithmetic
    is identical in FP32.
    """
    pid_m = tl.program_id(0)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_K)

    x_ptrs = X_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
    mask_m = offs_m[:, None] < M

    # ---- pass 1: per-row amax across all K ----
    amax = tl.zeros((BLOCK_M,), dtype=tl.float32)
    for k0 in range(tl.cdiv(K, BLOCK_K)):
        k_base = k0 * BLOCK_K
        mask_k = (k_base + offs_k)[None, :] < K
        x = tl.load(
            x_ptrs + k_base * stride_xk,
            mask=mask_m & mask_k,
            other=tl.cast(0.0, tl.bfloat16),
        ).to(tl.float32)
        # Per-row max over BLOCK_K. Mask out-of-bounds K so they
        # don't poison the amax (loaded as 0.0, so abs(0) = 0 anyway,
        # but be defensive for negative values).
        x_abs = tl.where(mask_k, tl.abs(x), 0.0)
        chunk_max = tl.max(x_abs, axis=1)
        amax = tl.maximum(amax, chunk_max)

    # Floor to FP32 E4M3 min normal so we don't divide by subnormal
    # near zero (would amplify noise). This matches the Python
    # ``amax.clamp(min=_E4M3_MIN_NORMAL)`` step.
    amax = tl.maximum(amax, E4M3_MIN_NORMAL)
    # PyTorch's reference path does the divide in BF16 then casts
    # to FP32 (``(amax_bf16 / 448.0).to(fp32)``), which truncates
    # the scale to ~7-bit mantissa precision. The Triton kernel
    # would otherwise compute the divide in full FP32 (~23-bit),
    # giving a ~2% per-row scale shift that shows up as a 2%
    # median rel in the matmul output (vs the PyTorch amax). Round
    # to BF16 first to match.
    scale_bf16 = (amax / E4M3_MAX).to(tl.bfloat16)
    scale = scale_bf16.to(tl.float32)  # [BLOCK_M] FP32
    inv_scale = (E4M3_MAX / amax).to(tl.bfloat16).to(tl.float32)

    # Write scale to global memory for the GEMM
    tl.store(X_scale_out_ptr + offs_m * stride_sm, scale, mask=offs_m < M)

    # ---- pass 2: quantize using the just-computed scale ----
    o_ptrs = X_out_ptr + offs_m[:, None] * stride_om + offs_k[None, :] * stride_ok
    for k0 in range(tl.cdiv(K, BLOCK_K)):
        k_base = k0 * BLOCK_K
        mask_k = (k_base + offs_k)[None, :] < K
        x = tl.load(
            x_ptrs + k_base * stride_xk,
            mask=mask_m & mask_k,
            other=tl.cast(0.0, tl.bfloat16),
        ).to(tl.float32)
        # x_q = clamp(x / scale, +/-E4M3_MAX) = clamp(x * inv_scale, +/-E4M3_MAX).
        # Multiplying by the pre-computed reciprocal is one fewer division
        # per element than dividing by scale.
        x_q = tl.minimum(
            tl.maximum(x * inv_scale[:, None], -E4M3_MAX),
            E4M3_MAX,
        ).to(tl.float8e4nv)
        tl.store(
            o_ptrs + k_base * stride_ok,
            x_q,
            mask=mask_m & mask_k,
        )


def quantize_act_fp8(
    x: torch.Tensor,
    scale: torch.Tensor,
    out: torch.Tensor | None = None,
    block_m: int = 128,
    block_k: int = 128,
) -> torch.Tensor:
    """Per-row BF16 -> fp8 E4M3 quantization using pre-computed scales.

    Args:
        x     : [M, K] BF16 input
        scale : [M] FP32 per-row scales (amax / E4M3_MAX)
        out   : optional [M, K] fp8_e4m3fn output buffer
        block_m, block_k : Triton tile dims (default 128/128)

    Returns:
        [M, K] float8_e4m3fn tensor (caller-supplied or freshly allocated)
    """
    M, K = x.shape
    if out is None:
        out = torch.empty(M, K, dtype=torch.float8_e4m3fn, device=x.device)
    grid = (triton.cdiv(M, block_m),)
    _quantize_act_fp8_kernel[grid](
        x,
        scale,
        out,
        M, K,
        x.stride(0), x.stride(1),
        scale.stride(0),
        out.stride(0), out.stride(1),
        E4M3_MAX=448.0,
        BLOCK_M=block_m,
        BLOCK_K=block_k,
        num_warps=4,
    )
    return out


def quantize_act_fp8_fused(
    x: torch.Tensor,
    out: torch.Tensor | None = None,
    scale: torch.Tensor | None = None,
    block_m: int = 64,
    block_k: int = 128,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused per-row amax + BF16 -> fp8 E4M3 quantize in one kernel.

    The production forward's hot path. Combines the per-row ``amax``
    reduction, the ``amax / E4M3_MAX`` scale computation, and the
    fp8 cast into a single Triton launch — saves the Python
    ``amax + .clamp + /E4M3_MAX + .to(fp32)`` chain (~0.34 ms at
    prod M=16k) and a separate quantize-kernel launch (~0.20 ms).

    Args:
        x     : [M, K] BF16 input (post-RMSNorm FFN input)
        out   : optional [M, K] fp8_e4m3fn output buffer
        scale : optional [M, 1] FP32 per-row scale output buffer
                (shape must match what ``torch._scaled_mm`` expects)
        block_m, block_k : Triton tile dims (default 64 / 128)

    Returns:
        (a_fp8, a_s) — ``a_fp8`` is [M, K] float8_e4m3fn, ``a_s`` is
        [M, 1] FP32 row scales.
    """
    M, K = x.shape
    if out is None:
        out = torch.empty(M, K, dtype=torch.float8_e4m3fn, device=x.device)
    if scale is None:
        scale = torch.empty(M, 1, dtype=torch.float32, device=x.device)
    grid = (triton.cdiv(M, block_m),)
    _quantize_act_fp8_fused_kernel[grid](
        x,
        scale,
        out,
        M, K,
        x.stride(0), x.stride(1),
        scale.stride(0),
        out.stride(0), out.stride(1),
        E4M3_MAX=448.0,
        E4M3_MIN_NORMAL=_E4M3_MIN_NORMAL,
        BLOCK_M=block_m,
        BLOCK_K=block_k,
        num_warps=4,
    )
    return out, scale


# ---------------------------------------------------------------------------
# Triton kernel: NVFP4 → fp8 E4M3
# ---------------------------------------------------------------------------
@triton.jit
def _lk(x: tl.int32) -> tl.float32:
    return tl.where(
        x == 0, 0.0, tl.where(
            x == 1, 0.5, tl.where(
                x == 2, 1.0, tl.where(
                    x == 3, 1.5, tl.where(
                        x == 4, 2.0, tl.where(
                            x == 5, 3.0, tl.where(x == 6, 4.0, 6.0)
                        )
                    )
                )
            )
        ),
    )


@triton.jit
def _dequant_nvfp4_to_fp8_kernel(
    B_packed_ptr,    # [N, K//2] uint8
    B_scales_ptr,     # [N, K//16] float8_e4m3fn
    B_out_ptr,        # [N, K] float8_e4m3fn
    N, K,
    stride_bn, stride_bk,
    stride_sn, stride_sk,
    stride_on, stride_ok,
    E4M3_MAX: tl.constexpr,
    BOOST: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    k_base = pid_k * BLOCK_K

    # Load packed bytes [N, BLOCK_K//2]
    byte_k = k_base // 2 + tl.arange(0, BLOCK_K // 2)
    b = tl.load(
        B_packed_ptr + offs_n[:, None] * stride_bn + byte_k[None, :] * stride_bk,
        mask=(offs_n[:, None] < N) & (byte_k[None, :] < (K // 2)),
        other=tl.cast(0, tl.uint8),
    ).to(tl.uint8)
    lo = (b & 0xF).to(tl.int32)
    hi = ((b >> 4) & 0xF).to(tl.int32)
    b_lo = _lk(lo & 0x7) * (1.0 - 2.0 * ((lo >> 3) & 1).to(tl.float32))
    b_hi = _lk(hi & 0x7) * (1.0 - 2.0 * ((hi >> 3) & 1).to(tl.float32))
    b_full = tl.reshape(tl.join(b_lo, b_hi), (BLOCK_N, BLOCK_K))

    # Load scales [N, BLOCK_K//16] broadcast to [N, BLOCK_K]
    s_k = k_base // 16 + tl.arange(0, BLOCK_K // 16)
    s = tl.load(
        B_scales_ptr + offs_n[:, None] * stride_sn + s_k[None, :] * stride_sk,
        mask=(offs_n[:, None] < N) & (s_k[None, :] < (K // 16)),
        other=tl.cast(0.0, tl.float8e4nv),
    ).to(tl.float32)
    s_full = tl.reshape(
        tl.broadcast_to(s[:, :, None], (BLOCK_N, BLOCK_K // 16, 16)),
        (BLOCK_N, BLOCK_K),
    )

    # Scale, boost, clamp, cast
    b32 = b_full * s_full * BOOST
    b32 = tl.minimum(tl.maximum(b32, -E4M3_MAX), E4M3_MAX)
    offs_k_full = k_base + tl.arange(0, BLOCK_K)
    tl.store(
        B_out_ptr + offs_n[:, None] * stride_on + offs_k_full[None, :] * stride_ok,
        b32.to(tl.float8e4nv),
        mask=(offs_n[:, None] < N) & (offs_k_full[None, :] < K),
    )


def dequant_nvfp4_to_fp8(
    b_packed: torch.Tensor,
    b_scales: torch.Tensor,
    boost: float,
    out: torch.Tensor | None = None,
    block_n: int = 64,
    block_k: int = 128,
) -> torch.Tensor:
    """Dequantize NVFP4 → fp8 E4M3, returns [N, K] fp8 tensor.

    Args:
        b_packed: [N, K//2] uint8 NVFP4 packed
        b_scales: [N, K//16] float8_e4m3fn per-block scales
        boost: pow2 scale so max|e2m1*s*boost| ≤ 448
        out: optional output buffer
        block_n, block_k: Triton block dims

    Returns:
        [N, K] float8_e4m3fn tensor (row-major, same stride as standard [N,K])
    """
    N, KH = b_packed.shape
    K = KH * 2
    if out is None:
        out = torch.empty(N, K, dtype=torch.float8_e4m3fn, device=b_packed.device)
    grid = (triton.cdiv(N, block_n), triton.cdiv(K, block_k))
    _dequant_nvfp4_to_fp8_kernel[grid](
        b_packed,
        b_scales,
        out,
        N, K,
        b_packed.stride(0), b_packed.stride(1),
        b_scales.stride(0), b_scales.stride(1),
        out.stride(0), out.stride(1),
        E4M3_MAX=448.0,
        BOOST=boost,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=4,
    )
    return out


# ---------------------------------------------------------------------------
# Autograd function
# ---------------------------------------------------------------------------
class _NVFP4W4A8Matmul(torch.autograd.Function):
    """Two-pass NVFP4 W4A8 matmul with STE backward.

    Forward:
        1. Quantize activation to per-token FP8 E4M3 + scale
        2. Dequantize weight (packed NVFP4 → fp8)
        3. _scaled_mm(a_fp8, b_fp8.T, scale_a=a_s, scale_b=g/boost)

    Backward (STE):
        grad_x = grad_out @ w_bf16    (BF16 matmul)
        grad_w = grad_out.T @ x_bf16   (BF16 matmul)
    """

    @staticmethod
    def forward(ctx, x, w_bf16, bias, block_size, use_custom_gemm):
        K = x.shape[-1]
        K = w_bf16.shape[1]
        N = w_bf16.shape[0]

        # Quantize activation: per-token FP8 E4M3.
        # amax in Python (cheap reduction), quantize in Triton (fused load
        # -> quantize -> fp8 store — no per-element dispatch).
        x_2d = x.reshape(-1, K)
        # Quantize activation: fused per-row amax + fp8 cast in one
        # Triton launch (replaces the prior Python amax + separate
        # quantize kernel — saves ~0.5 ms per forward at prod shape).
        a_fp8, a_s = quantize_act_fp8_fused(x_2d)

        # Quantize weight: NVFP4 packed + scales + global
        packed, s_e4m3, g = quantize_nvfp4_with_global_scale(
            w_bf16, block_size=block_size,
        )
        boost = _auto_boost(s_e4m3)

        # Dequant weight: NVFP4 → fp8 [N, K]
        b_fp8 = dequant_nvfp4_to_fp8(packed, s_e4m3, boost)

        # Pass-2 GEMM: pick backend
        g_adj = g.float() / boost
        if use_custom_gemm:
            from src.models.ops.cuda.fp8_gemm import (
                fp8_gemm_scaled, is_available as _fp8_gemm_available,
            )
            if not _fp8_gemm_available():
                # .so not built for this arch — fall back to _scaled_mm
                # (no warning; the caller opted in knowing the .so
                # may or may not be present).
                use_custom_gemm = False

        if use_custom_gemm:
            # Custom kernel: a_s is [M, 1], scale_b is a scalar
            # broadcast to [1, N] (all entries = g_adj).
            scale_b = torch.full((1, N), g_adj.item(), dtype=torch.float32, device=x.device)
            out_2d = fp8_gemm_scaled(a_fp8, b_fp8, a_s, scale_b)
        else:
            # cuBLAS nvjet via _scaled_mm (B is [K, N] col-major view).
            scale_b = torch.full((1, N), g_adj.item(), dtype=torch.float32, device=x.device)
            out_2d = torch._scaled_mm(
                a_fp8,
                b_fp8.t(),
                scale_a=a_s,
                scale_b=scale_b,
                out_dtype=torch.bfloat16,
            )

        if bias is not None:
            out_2d = out_2d + bias

        ctx.save_for_backward(x, w_bf16)
        ctx.has_bias = bias is not None
        ctx.block_size = block_size
        ctx.shape = x.shape
        return out_2d.reshape(*x.shape[:-1], N)

    @staticmethod
    def backward(ctx, grad_out):
        x, w = ctx.saved_tensors
        K = w.shape[1]
        N = w.shape[0]

        grad_out_2d = grad_out.reshape(-1, N).to(torch.float32)
        x_2d = x.reshape(-1, K).to(torch.float32)
        w_2d = w.to(torch.float32)

        # STE backward: re-run in BF16
        grad_x_2d = grad_out_2d @ w_2d           # [M, K]
        grad_w = grad_out_2d.t() @ x_2d          # [N, K]
        grad_bias = grad_out_2d.sum(dim=0) if ctx.has_bias else None

        grad_x = grad_x_2d.reshape(*ctx.shape)
        # 5 grads: x, w, bias, block_size, use_custom_gemm (last two are None — non-differentiable knobs)
        return grad_x.to(x.dtype), grad_w.to(w.dtype), grad_bias, None, None


# ---------------------------------------------------------------------------
# Module
# ---------------------------------------------------------------------------
class NVFP4LinearW4A8(nn.Module):
    """NVFP4 W4A8 drop-in for ``nn.Linear`` (FFN weight).

    Storage: BF16 master ``weight`` (the optimizer leaf) + derived
    NVFP4 packed buffers (recomputed from the master each forward).

    State dict keys (BF16 master only — identical to ``nn.Linear``):

        weight : [N, K] BF16
        bias   : [N] BF16 (present iff ``bias=True``)

    Autograd: STE — backward re-runs BF16 matmul, so the optimizer
    update is the standard BF16 one. The NVFP4 quantization is
    discarded in backward.

    Args:
        in_features  : K
        out_features : N
        bias         : include bias parameter
        block_size   : NVFP4 block size along K (default 16)
        device       : parameter device
        dtype        : parameter dtype (BF16 by default)
        bf16_only    : escape hatch — skip NVFP4 path, use ``F.linear``

    Shape constraint:
        ``torch._scaled_mm`` requires both K and N divisible by 16.
        When they are not, the constructor silently sets ``bf16_only=True``
        (bit-exact with ``nn.Linear``). The ``extra_repr`` reports the
        resolved mode.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        block_size: int = 16,
        device=None,
        dtype=None,
        bf16_only: bool = False,
        use_custom_gemm: bool = False,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.block_size = block_size
        self.use_custom_gemm = use_custom_gemm

        # _scaled_mm requires [N, K] divisible by 16 on sm_120.
        # b_proj (1536 → 12) hits this, so we fall back silently.
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
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            bound = 1.0 / math.sqrt(self.in_features)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.bf16_only:
            return F.linear(x, self.weight, self.bias)
        return _NVFP4W4A8Matmul.apply(
            x, self.weight, self.bias, self.block_size, self.use_custom_gemm,
        )

    def extra_repr(self) -> str:
        if self.bf16_only:
            mode = "BF16 (bf16_only=True)"
        elif self.use_custom_gemm:
            mode = f"NVFP4 W4A8 + custom GEMM (block={self.block_size})"
        else:
            mode = f"NVFP4 W4A8 (block={self.block_size})"
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"bias={self.bias is not None}, "
            f"mode={mode}"
        )
