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

**Backward path is BF16 throughout** — do NOT cast to FP32 before
the matmul. The historical FP32-cast (FP32 SIMT path) was a perf
bug: 12.5 TFLOPS FP32 SIMT peak on sm_120 vs 50 TFLOPS BF16 TC
(~3× slower at FFN prod shape). BF16 STE is bit-equivalent to a
plain ``F.linear`` backward at the BF16 rounding-noise floor
(cos_sim 1.0, max_abs_err 0.0). Same pattern as the FP8Linear
``project_fp8_bwd_fp32_bug.md`` fix shipped 2026-07-21.

The FP8 ``_scaled_mm`` backward (FP8Linear fp8_bwd=True) is NOT
applied here — at FFN prod M=1024 the 4 transposed quant kernels
cost ~370 us of overhead, only beating BF16 STE by ~14% (520 vs
600 us per FFN bwd call). At KDA-shape (M=1024, N=1536) the FP8
path is actually 30% SLOWER than BF16 STE (370 vs 258 us) because
the quant overhead exceeds the GEMM savings. Re-evaluate if the
FFN M grows past ~8k where FP8 starts to dominate.

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

import logging
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

logger = logging.getLogger(__name__)


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


# ---------------------------------------------------------------------------
# Triton kernel: single-pass per-row amax + quantize (BF16 → fp8 E4M3)
# (resident BLOCK_M × BLOCK_K tile, no K-loop, no HBM re-read)
# ---------------------------------------------------------------------------
@triton.jit
def _quantize_act_fp8_fused_singlepass_kernel(
    X_ptr,           # [M, K] BF16
    X_scale_out_ptr, # [M, 1] FP32  (output: per-row amax/E4M3_MAX)
    X_out_ptr,       # [M, K] float8_e4m3fn
    M, K,
    stride_xm, stride_xk,
    stride_sm,
    stride_om, stride_ok,
    E4M3_MAX: tl.constexpr,
    E4M3_MIN_NORMAL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,  # = next_power_of_2(K), the full K tile
):
    """Single-read per-row amax + quantize — no HBM re-read.

    One program owns a ``BLOCK_M × BLOCK_K`` tile of the input and
    loads it ONCE into registers. The per-row amax is reduced from
    that resident tile; the quantization is performed against the
    SAME tile (no second HBM read). This saves the 1.7x bandwidth
    penalty that the 2-pass kernel pays (the 2-pass kernel re-reads
    the BF16 input from HBM between the amax pass and the quantize
    pass because the BLOCK_M × K FP32 tile won't fit in SMEM for
    reuse across iterations of the K-loop).

    Numerics: BYTE-EXACT equivalent to the 2-pass kernel. Same
    dynamic per-row amax, same ``(amax / E4M3_MAX).to(bf16).to(fp32)``
    scale rounding, same ``(E4M3_MAX / amax).to(bf16).to(fp32)``
    inv_scale, same ``clamp(x * inv_scale, ±E4M3_MAX).to(fp8)`` cast.
    Verified ``max |fp8 byte diff| = 0`` and ``max |scale diff| = 0``
    across the prod shape set — see
    ``test/test_kda_fp8.py::test_fp8_act_quant_singlepass_byte_exact``
    for the regression guard.

    Limitation: ``BLOCK_K`` is the FULL K (no loop), so the tile
    must fit in registers. With ``BLOCK_M=8`` and ``num_warps=8``
    the FP32-tile budget is ``8 * 32 * 255 ≈ 65K elements`` per
    program — comfortably covers K ≤ 4096 (BLOCK_K = 4096 → 32K
    elements per program = 128 KB FP32 = 16 KB / warp = 2 KB /
    thread = 512 FP32 / thread, well under the 255-reg/thread
    ceiling). K > 4096 falls back to the 2-pass kernel inside
    :func:`quantize_act_fp8_fused`.

    Layout note: the BF16 input must be contiguous (raw stride-
    indexed loads; the kernel addresses x via ``stride_xm`` /
    ``stride_xk``). Caller-side :func:`quantize_act_fp8_fused`
    asserts this. See ``.claude/rules/einsum-noncontig-triton.md``
    for the non-contig trap this avoids.
    """
    pid_m = tl.program_id(0)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_K)
    mask_m = offs_m[:, None] < M
    mask_k = offs_k[None, :] < K

    # Single load of the full row tile into registers.
    x_ptrs = X_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
    x = tl.load(
        x_ptrs,
        mask=mask_m & mask_k,
        other=tl.cast(0.0, tl.bfloat16),
    ).to(tl.float32)

    # Per-row amax across K (single reduction over the resident tile's
    # K axis). Mask out-of-bounds K so they don't poison the amax.
    x_abs = tl.where(mask_k, tl.abs(x), 0.0)
    amax = tl.max(x_abs, axis=1)                # [BLOCK_M]
    amax = tl.maximum(amax, E4M3_MIN_NORMAL)

    # Same BF16-truncate-the-scale convention as the 2-pass kernel —
    # matches the PyTorch reference path's scale precision exactly
    # (within ~0.62% med_rel; well under the FP8 noise floor).
    scale_bf16 = (amax / E4M3_MAX).to(tl.bfloat16)
    scale = scale_bf16.to(tl.float32)           # [BLOCK_M]
    inv_scale = (E4M3_MAX / amax).to(tl.bfloat16).to(tl.float32)

    tl.store(X_scale_out_ptr + offs_m * stride_sm, scale, mask=offs_m < M)

    # Quantize from the SAME resident tile (no HBM re-read).
    x_q = tl.minimum(
        tl.maximum(x * inv_scale[:, None], -E4M3_MAX),
        E4M3_MAX,
    ).to(tl.float8e4nv)

    o_ptrs = X_out_ptr + offs_m[:, None] * stride_om + offs_k[None, :] * stride_ok
    tl.store(o_ptrs, x_q, mask=mask_m & mask_k)


# Tile + threshold for the single-pass path. BLOCK_M=8 was the bench
# winner on the prod shape (M=16384, K=1536); see
# ``test/_tmp/probe_singlepass_quant.py`` for the BLOCK_M=1..16 sweep.
# K ≤ 4096 covers every production shape (128, 1536, 4096). Larger K
# falls back to the 2-pass kernel inside :func:`quantize_act_fp8_fused`.
_SINGLEPASS_BLOCK_M = 8
_SINGLEPASS_MAX_K = 4096


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

    Internally dispatches between two kernels:

      * **Single-pass** (``_quantize_act_fp8_fused_singlepass_kernel``,
        default for K ≤ 4096) — one program owns a ``BLOCK_M × K``
        tile and loads it ONCE into registers; amax + quantize both
        operate on the resident tile. ~1.74× faster than the 2-pass
        path because it skips the HBM re-read between the two
        passes. Numerics are byte-exact (see
        ``test/test_kda_fp8.py::test_fp8_act_quant_singlepass_byte_exact``).

      * **2-pass** (``_quantize_act_fp8_fused_kernel``, fallback for
        K > 4096) — pass 1 reduces per-row amax over a K-loop,
        pass 2 re-reads x from HBM and quantizes. Used when K is
        too large to fit the full row tile in registers.

    The dispatch is internal — public signature unchanged. ``block_m``
    and ``block_k`` are passed through to the 2-pass fallback and
    ignored by the single-pass path (which uses
    ``_SINGLEPASS_BLOCK_M = 8`` and ``BLOCK_K = next_power_of_2(K)``).

    Args:
        x     : [M, K] BF16 input (post-RMSNorm FFN input)
        out   : optional [M, K] fp8_e4m3fn output buffer
        scale : optional [M, 1] FP32 per-row scale output buffer
                (shape must match what ``torch._scaled_mm`` expects)
        block_m, block_k : 2-pass fallback tile dims (default 64 / 128)

    Returns:
        (a_fp8, a_s) — ``a_fp8`` is [M, K] float8_e4m3fn, ``a_s`` is
        [M, 1] FP32 row scales.
    """
    M, K = x.shape
    assert x.is_contiguous(), (
        "quantize_act_fp8_fused requires a contiguous input; "
        "call .contiguous() before passing in"
    )
    if out is None:
        out = torch.empty(M, K, dtype=torch.float8_e4m3fn, device=x.device)
    if scale is None:
        scale = torch.empty(M, 1, dtype=torch.float32, device=x.device)

    # Dispatch: single-pass when the full K fits in a resident tile.
    if K <= _SINGLEPASS_MAX_K:
        BLOCK_K = triton.next_power_of_2(K)
        # num_warps=4 is enough for small K (≤256), num_warps=8 for
        # larger K — keeps each warp's element budget healthy without
        # over-saturating tiny tiles.
        num_warps = 4 if K <= 256 else 8
        grid = (triton.cdiv(M, _SINGLEPASS_BLOCK_M),)
        _quantize_act_fp8_fused_singlepass_kernel[grid](
            x,
            scale,
            out,
            M, K,
            x.stride(0), x.stride(1),
            scale.stride(0),
            out.stride(0), out.stride(1),
            E4M3_MAX=448.0,
            E4M3_MIN_NORMAL=_E4M3_MIN_NORMAL,
            BLOCK_M=_SINGLEPASS_BLOCK_M,
            BLOCK_K=BLOCK_K,
            num_warps=num_warps,
        )
        return out, scale

    # 2-pass fallback (K > 4096). Same math as the original kernel;
    # the HBM re-read is unavoidable when the K tile won't fit in
    # registers.
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
# Triton kernel: fused per-col amax + quantize + transpose (BF16 -> fp8 E4M3)
# ---------------------------------------------------------------------------
@triton.jit
def _quantize_act_fp8_fused_transposed_kernel(
    X_ptr,           # [M, K] BF16 (input, row-major; must be contiguous)
    X_scale_out_ptr, # [K, 1] FP32  (output: per-col amax / E4M3_MAX)
    X_out_ptr,       # [K, M] float8_e4m3fn (output, row-major = transposed)
    M, K,
    stride_xm, stride_xk,
    stride_sk,
    stride_ok, stride_om,
    E4M3_MAX: tl.constexpr,
    E4M3_MIN_NORMAL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Fused per-col amax + quantize + transpose in one kernel.

    Inverse of :func:`quantize_act_fp8_fused` along the M axis:
    instead of one program owning ``BLOCK_M`` rows and reducing
    across K, this kernel has one program own ``BLOCK_K`` columns
    and reduce across M. The amax stays in registers between the
    two passes; the output is written directly into the transposed
    [K, M] row-major layout.

    Reads x in its original [M, K] row-major order (no separate
    transpose copy) and writes the FP8 output in [K, M] row-major
    (which is the col-major view of the original). Used by the
    FP8 KDA backward to skip the ``.T.contiguous()`` materialization
    before the transposed quant pass — at KDA prod shape (M=16k,
    N=K=1536) saves ~1.4 ms / FP8 bwd call.
    """
    pid_k = tl.program_id(0)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    offs_m = tl.arange(0, BLOCK_M)
    mask_k = offs_k < K
    # Input ptrs shape [BLOCK_K, BLOCK_M]: each program block owns
    # BLOCK_K cols of x and reads them in [BLOCK_K, BLOCK_M] tiles.
    x_ptrs = X_ptr + offs_k[:, None] * stride_xk + offs_m[None, :] * stride_xm

    # ---- pass 1: per-col amax reduction across all M ----
    amax = tl.zeros((BLOCK_K,), dtype=tl.float32)
    for m0 in range(tl.cdiv(M, BLOCK_M)):
        m_base = m0 * BLOCK_M
        mask_m_chunk = (m_base + offs_m)[None, :] < M
        x = tl.load(
            x_ptrs + m_base * stride_xm,
            mask=mask_k[:, None] & mask_m_chunk,
            other=tl.cast(0.0, tl.bfloat16),
        ).to(tl.float32)
        # Per-col max over BLOCK_M (axis=1). The block owns BLOCK_K
        # cols; each col sees BLOCK_M elements per pass-1 chunk.
        chunk_max = tl.max(tl.abs(x), axis=1)
        amax = tl.maximum(amax, chunk_max)

    amax = tl.maximum(amax, E4M3_MIN_NORMAL)
    # Same BF16-truncate-the-scale trick as the per-row kernel:
    # matches the PyTorch reference path's scale precision exactly
    # (within ~0.62% med_rel; well under the FP8 noise floor).
    scale_bf16 = (amax / E4M3_MAX).to(tl.bfloat16)
    scale = scale_bf16.to(tl.float32)
    inv_scale = (E4M3_MAX / amax).to(tl.bfloat16).to(tl.float32)

    tl.store(X_scale_out_ptr + offs_k * stride_sk, scale, mask=mask_k)

    # ---- pass 2: quantize using the just-computed scale ----
    # Output ptrs shape [BLOCK_K, BLOCK_M]: write to output[k, m].
    o_ptrs = X_out_ptr + offs_k[:, None] * stride_ok + offs_m[None, :] * stride_om
    for m0 in range(tl.cdiv(M, BLOCK_M)):
        m_base = m0 * BLOCK_M
        mask_m_chunk = (m_base + offs_m)[None, :] < M
        x = tl.load(
            x_ptrs + m_base * stride_xm,
            mask=mask_k[:, None] & mask_m_chunk,
            other=tl.cast(0.0, tl.bfloat16),
        ).to(tl.float32)
        x_q = tl.minimum(
            tl.maximum(x * inv_scale[:, None], -E4M3_MAX),
            E4M3_MAX,
        ).to(tl.float8e4nv)
        tl.store(
            o_ptrs + m_base * stride_om,
            x_q,
            mask=mask_k[:, None] & mask_m_chunk,
        )


def quantize_act_fp8_fused_transposed(
    x: torch.Tensor,
    out: torch.Tensor | None = None,
    scale: torch.Tensor | None = None,
    block_m: int = 128,
    block_k: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused per-col amax + quantize + transpose for the FP8 bwd path.

    Equivalent to ``out, scale = quantize_act_fp8_fused(x.T.contiguous())``
    but in a single Triton kernel — saves the BF16 [K, M] materialize
    that the explicit ``.T.contiguous()`` does. Used by the KDA
    FP8 bwd path to re-quantize ``w``, ``grad_out`` and ``x`` in
    their transposed layouts in one pass each.

    Args:
        x     : [M, K] BF16 input, must be contiguous (the kernel
                uses raw stride-indexed loads; non-contig would
                read the wrong memory).
        out   : optional [K, M] fp8_e4m3fn output buffer
        scale : optional [K, 1] FP32 per-col-of-input scale buffer
        block_m, block_k : Triton tile dims (default 128 / 64,
                the autotune winner for KDA prod M=16384 K=N=1536)

    Returns:
        (q_fp8, s) — ``q_fp8`` is [K, M] float8_e4m3fn (row-major
        = transpose of x's col-major view); ``s`` is [K, 1] FP32
        per-col-of-x scales (= per-row-of-q_fp8).
    """
    assert x.is_contiguous(), (
        "quantize_act_fp8_fused_transposed requires a contiguous input; "
        "call .contiguous() before passing in"
    )
    M, K = x.shape
    if out is None:
        out = torch.empty(K, M, dtype=torch.float8_e4m3fn, device=x.device)
    if scale is None:
        scale = torch.empty(K, 1, dtype=torch.float32, device=x.device)
    grid = (triton.cdiv(K, block_k),)
    _quantize_act_fp8_fused_transposed_kernel[grid](
        x, scale, out,
        M, K,
        x.stride(0), x.stride(1),
        scale.stride(0),
        out.stride(0), out.stride(1),
        E4M3_MAX=448.0,
        E4M3_MIN_NORMAL=_E4M3_MIN_NORMAL,
        BLOCK_M=block_m, BLOCK_K=block_k,
        num_warps=8,
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
    def forward(ctx, x, w_bf16, bias, block_size, use_custom_gemm, module,
                a_fp8_override=None, a_s_override=None):
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
        # In the shared-activation path (Strategy 1 — see
        # NVFP4W4A8SwiGLU), the caller pre-computes a_fp8/a_s on the
        # shared input and passes them in here to skip this launch.
        if a_fp8_override is None:
            M_act = x_2d.shape[0]
            # Cache the act_quant output buffers (saves the torch.empty
            # alloc + allocator hit on every forward — small but real,
            # ~10 us / call). Sized to max-M-seen; doesn't shrink.
            if (module._act_fp8_buf is None
                    or module._act_fp8_buf.shape[0] < M_act):
                module._act_fp8_buf = torch.empty(
                    M_act, K, dtype=torch.float8_e4m3fn, device=x.device,
                )
                module._act_s_buf = torch.empty(
                    M_act, 1, dtype=torch.float32, device=x.device,
                )
            else:
                module._act_fp8_buf = module._act_fp8_buf[:M_act]
                module._act_s_buf = module._act_s_buf[:M_act]
            a_fp8, a_s = quantize_act_fp8_fused(
                x_2d, out=module._act_fp8_buf, scale=module._act_s_buf,
            )
        else:
            # Caller supplied a pre-quantized FP8 buffer (typically
            # the upstream RmsNormFp8STE passthrough). Flatten any
            # leading dims to 2-D — the matmul kernel needs 2-D, and
            # callers may pass 3-D ``[B, T, K]`` from a
            # :class:`HippoLayer` forward.
            a_fp8 = a_fp8_override.reshape(-1, K) if a_fp8_override.dim() != 2 else a_fp8_override
            a_s = a_s_override.reshape(-1, 1) if a_s_override.dim() != 2 else a_s_override

        # Quantize weight: NVFP4 packed + scales + global.
        # Cached on the module and invalidated when w_bf16._version changes
        # (Strategy 2 — skip-N pack cache). Saves the 0.59 ms w_quant
        # Triton launch on every forward after the first, as long as
        # the BF16 master weight hasn't been touched. During training
        # the optimizer bumps _version per step, so hit rate is
        # (N_microbatches-1)/N_microbatches; in a bench loop it's 1.0.
        packed, s_e4m3, g = module._get_or_refresh_pack(w_bf16, block_size)
        # Boost = pow2 scale so max|e2m1*s_e4m3*C| stays inside E4M3 range.
        # Cached on the module and invalidated when w_bf16._version changes
        # (i.e. when the optimizer/loader modifies the BF16 master weight).
        boost = module._get_or_refresh_boost(w_bf16, s_e4m3)

        # Dequant weight: NVFP4 → fp8 [N, K]. Reuse a cached buffer
        # if shape matches (saves the torch.empty alloc + allocator hit
        # on every forward — small but real, ~3-12 us/call = ~30 us/step
        # across the 3 W4A8 linears per step).
        N = w_bf16.shape[0]
        K = w_bf16.shape[1]
        if (module._fp8_dequant_buf is None
                or module._fp8_dequant_buf.shape != (N, K)):
            module._fp8_dequant_buf = torch.empty(
                N, K, dtype=torch.float8_e4m3fn, device=w_bf16.device,
            )
        b_fp8 = dequant_nvfp4_to_fp8(
            packed, s_e4m3, boost, out=module._fp8_dequant_buf,
        )

        # Pass-2 GEMM: route through the FP8 GEMM runtime
        # auto-dispatcher. Per-arch `.so` discovery + per-(M, K, N)
        # micro-bench picks the fastest available backend at first
        # call (BM64/BN64 wins at small M, BM128/BN128 at large M);
        # falls back to ``torch._scaled_mm`` automatically when no
        # prebuilt .so covers the current arch.
        #
        # The ``use_custom_gemm`` opt-in flag is now subsumed by the
        # dispatcher's own backend pick; we still honor the flag for
        # backwards compatibility (True = "prefer fp8_gemm kernels
        # over _scaled_mm when available", but never refuse to run).
        from src.models.ops.cuda.fp8_gemm_dispatch import fp8_gemm_auto_dispatch
        g_adj = g.float() / boost
        # scale_b is per-N (NVFP4 global scale folds into one scalar
        # applied to all N output channels); broadcast to [1, N].
        scale_b = torch.full((1, N), g_adj.item(), dtype=torch.float32, device=x.device)
        out_2d = fp8_gemm_auto_dispatch(a_fp8, b_fp8, a_s, scale_b)

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

        # STE backward: re-run in BF16. NO FP32 cast — the historical
        # FP32-cast path (FP32 GEMM via ``to(float32)``) was a perf bug
        # at 12.5 TFLOPS FP32 SIMT peak vs 50 TFLOPS BF16 TC peak on
        # sm_120 (3× slower at FFN prod shape). BF16 STE is bit-
        # equivalent to a plain ``F.linear`` backward at the BF16
        # rounding-noise floor (cos_sim 1.0, max_abs_err 0.0 on prod
        # shape — see ``test/_tmp/probe_nvfp4_w4a8_bwd_fix.py``).
        grad_out_2d = grad_out.reshape(-1, N)
        x_2d = x.reshape(-1, K)
        grad_x_2d = grad_out_2d @ w               # [M, K]  BF16 GEMM
        grad_w = grad_out_2d.t() @ x_2d          # [N, K]  BF16 GEMM
        grad_bias = grad_out_2d.sum(dim=0) if ctx.has_bias else None

        grad_x = grad_x_2d.reshape(*ctx.shape)
        # 8 grads: x, w, bias, block_size, use_custom_gemm, module,
        # a_fp8_override, a_s_override. The last three are non-tensor
        # knobs; the overrides are derived buffers (no grad flow).
        return (
            grad_x, grad_w, grad_bias,
            None, None, None, None, None,
        )


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

        # Boost cache for auto_boost (Strategy 3). Version sentinel -1
        # forces first forward to compute; subsequent forwards refresh
        # only when the BF16 master's _version changes.
        self._boost_cache: float | None = None
        self._boost_w_ver: int = -1

        # Pack cache (Strategy 2 — skip-N NVFP4 quant). Holds
        # (packed, s_e4m3, g) keyed by w_bf16._version. The tensors
        # stay alive in HBM as long as the cache is valid; per-layer
        # cost is ~3.4 MB at gate_up shape. The dequant (0.08 ms) is
        # NOT cached — only the slow w_quant (0.59 ms).
        self._pack_cache: tuple[torch.Tensor, torch.Tensor, float] | None = None
        self._pack_w_ver: int = -1
        self._pack_block_size: int = -1

        # w_dquant output buffer cache (saves the torch.empty alloc on
        # every forward). Lazily allocated on first forward; sized to
        # (out_features, in_features) FP8. ~3 MB at FFN gate/up shape.
        self._fp8_dequant_buf: torch.Tensor | None = None

        # act_quant output buffer cache (saves the torch.empty alloc
        # for a_fp8 + a_s on every forward). Lazily allocated; sized
        # to (max_M_seen, in_features) FP8 + (max_M_seen) FP32.
        # The M dimension grows as needed — we don't shrink.
        self._act_fp8_buf: torch.Tensor | None = None
        self._act_s_buf: torch.Tensor | None = None

        self._init_weights()

        # 2026-07-23: surfaced for debug — verify which NVFP4 W4A8
        # path is wired (and whether the ``bf16_only`` shape-div-16
        # silent fallback fired). Spec gap: bwd is BF16 STE today
        # even when fwd is NVFP4+FP8; the line below reports the
        # resolved fwd mode only.
        resolved = "BF16(bf16_only)" if self.bf16_only else "NVFP4-W4A8 (FP8 fwd + BF16 STE bwd)"
        logger.info(
            f"NVFP4LinearW4A8: in={in_features} out={out_features}"
            f" bias={bias} dtype={self.weight.dtype}"
            f" block_size={block_size} device={self.weight.device}"
            f" use_custom_gemm={use_custom_gemm}"
            f" → {resolved}"
        )

    def _init_weights(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            bound = 1.0 / math.sqrt(self.in_features)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.bf16_only:
            return F.linear(x, self.weight, self.bias)
        return _NVFP4W4A8Matmul.apply(
            x, self.weight, self.bias, self.block_size, self.use_custom_gemm, self,
        )

    # ------------------------------------------------------------------
    # Precomputed activation (Strategy 1 — share act_quant across gate+up)
    # ------------------------------------------------------------------
    # In the SwiGLU FFN, gate and up both consume the same ``x`` but
    # write to different output tensors. The naive pipeline runs
    # ``quantize_act_fp8_fused(x)`` twice (once per matmul), which is
    # wasteful — the fp8 tensor and the per-row scale are deterministic
    # functions of x. ``forward_precomputed`` skips the act_quant and
    # uses caller-supplied fp8 + scale buffers instead.
    #
    # The autograd Function discards the overrides in backward (STE
    # re-runs in BF16 from the original ``x``), so correctness is
    # preserved. Output shape is computed from ``x.shape`` so 2D and
    # 3D inputs both work.
    def forward_precomputed(
        self, x: torch.Tensor, a_fp8: torch.Tensor, a_s: torch.Tensor,
    ) -> torch.Tensor:
        if self.bf16_only:
            return F.linear(x, self.weight, self.bias)
        return _NVFP4W4A8Matmul.apply(
            x, self.weight, self.bias, self.block_size, self.use_custom_gemm, self,
            a_fp8, a_s,
        )

    # ------------------------------------------------------------------
    # Boost cache (Strategy 3 — moves per-forward auto_boost to module init)
    # ------------------------------------------------------------------
    # auto_boost picks a pow2 scale so the per-block NVFP4 E2M1*s product
    # stays inside the E4M3 ±448 range. It computes
    #     smax = scales_e4m3.float().abs().max().item()
    # then returns 2^floor(log2(448 / (6*smax))). The .item() is a D2H
    # sync — ~0.086 ms / call on prod shape. Cache it on the module,
    # invalidating only when the BF16 master's _version changes.
    #
    # In training with the standard BF16-master path (this module),
    # w_bf16._version bumps every optimizer step → cache hit rate ≈ 0
    # (no win in training). But in bench loops / inference where the
    # master is constant, the cache hits every forward after the first
    # and saves the D2H sync each time. Win compounds when combined
    # with Strategy 2 (which avoids the w_quant that re-feeds s_e4m3).
    def _get_or_refresh_boost(
        self, w_bf16: torch.Tensor, s_e4m3: torch.Tensor,
    ) -> float:
        w_ver = w_bf16._version
        if self._boost_w_ver != w_ver:
            self._boost_cache = _auto_boost(s_e4m3)
            self._boost_w_ver = w_ver
        return self._boost_cache

    # ------------------------------------------------------------------
    # Pack cache (Strategy 2 — skip-N NVFP4 quant)
    # ------------------------------------------------------------------
    # ``quantize_nvfp4_with_global_scale`` is the slowest per-forward
    # op at 0.59 ms on the prod gate_up shape (sm_120, M=16384 K=1536
    # N=4096). The output is purely a function of the BF16 master
    # weight — the same w_bf16 always produces the same packed bytes.
    # Cache (packed, s_e4m3, g) on the module and reuse across forwards
    # until the optimizer bumps ``w_bf16._version``.
    #
    # VRAM cost (real, NOT saved_tensors): ~3.4 MB per layer at
    # gate_up shape (4096*1536*0.5625 bytes for the packed + scales,
    # +4 bytes for g). For a 12-layer model = ~41 MB. That is below
    # the 5060 Ti 16 GB ceiling but is a real HWM increase —
    # ``torch.cuda.max_memory_allocated()`` will reflect it. The
    # benefit (1.77 ms saved per FFN forward in bench, 0 in single
    # forward) is bench-loop-only; in training the optimizer step
    # invalidates the cache so hit rate is (N_mb-1)/N_mb ≈ 0.95
    # for typical N_mb=20.
    #
    # The dequant (0.08 ms) is intentionally NOT cached — caching
    # b_fp8 would cost 3x more VRAM (N*K bytes vs N*K/2) for only
    # 0.25 ms/FFN additional savings. Stick with the cheaper cache.
    def _get_or_refresh_pack(
        self, w_bf16: torch.Tensor, block_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor, float]:
        w_ver = w_bf16._version
        if (
            self._pack_w_ver != w_ver
            or self._pack_block_size != block_size
            or self._pack_cache is None
        ):
            packed, s_e4m3, g = quantize_nvfp4_with_global_scale(
                w_bf16, block_size=block_size,
            )
            # Cache the packed bytes + scales + the 0-dim global-scale
            # tensor (downstream does ``g.float() / boost`` on it, so
            # keep it as a tensor rather than converting to a Python
            # float — avoids re-wrapping on every cache hit).
            self._pack_cache = (packed, s_e4m3, g)
            self._pack_w_ver = w_ver
            self._pack_block_size = block_size
        return self._pack_cache

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


# ---------------------------------------------------------------------------
# SwiGLU FFN with shared act_quant across gate+up
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Fused silu(gate) * up — single Triton kernel (no intermediate buffer)
# ---------------------------------------------------------------------------
@triton.jit
def _silu_mul_kernel(
    GATE_ptr, UP_ptr, OUT_ptr,
    M, N,
    stride_gm, stride_gn,
    stride_um, stride_un,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """``silu(gate) * up`` in one pass.

    The naive PyTorch expression ``F.silu(gate) * up`` materializes a
    ``silu(gate)`` intermediate (~256 MB at prod M=16384 I=4096) before
    multiplying by ``up`` — two passes through HBM for an op that
    logically only needs the three buffers (gate, up, hidden) read or
    written once each.

    This kernel reads gate and up together, computes ``g * sigmoid(g)``
    in fp32 registers, multiplies by ``u``, and writes the result —
    384 MB of HBM traffic instead of 640 MB (~40% less). Measured
    1.71 ms → 1.02 ms on sm_120 at the prod shape (40% wall reduction,
    see auto-memory ``project_w4a8_ffn_e2e.md``).

    Numerics: the fp32-in-register silu is one ULP more precise than
    PyTorch's bf16-intermediate path. Not bit-exact, but the difference
    is well within bf16 round-off noise (max abs diff ~0.06 at
    magnitude-1 inputs; median rel 0%). See
    ``test/test_nvfp4_linear_w4a8.py::test_silu_mul_fused_*`` for the
    guarded contract.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    g = tl.load(
        GATE_ptr + offs_m[:, None] * stride_gm + offs_n[None, :] * stride_gn,
        mask=mask, other=0.0,
    ).to(tl.float32)

    u = tl.load(
        UP_ptr + offs_m[:, None] * stride_um + offs_n[None, :] * stride_un,
        mask=mask, other=0.0,
    ).to(tl.float32)

    # silu(x) = x * sigmoid(x); compute in fp32, narrow on store
    out = (g * tl.sigmoid(g)) * u

    tl.store(
        OUT_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
        out.to(tl.bfloat16),
        mask=mask,
    )


def silu_mul_fused(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """``silu(gate) * up`` fused in one Triton kernel.

    Drop-in replacement for ``F.silu(gate) * up`` (modulo bf16 round-off
    in the silu intermediate — see kernel docstring).

    Args:
        gate: [M, N] BF16 (any 2D shape; the operation is elementwise)
        up:   [M, N] BF16, same shape as gate

    Returns:
        hidden: [M, N] BF16, ``silu(gate) * up``
    """
    assert gate.shape == up.shape, (
        f"gate and up must have the same shape, got {gate.shape} vs {up.shape}"
    )
    assert gate.dtype == torch.bfloat16 and up.dtype == torch.bfloat16
    M, N = gate.shape
    out = torch.empty_like(gate)
    BLOCK_M = 64
    BLOCK_N = 128
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    _silu_mul_kernel[grid](
        gate, up, out,
        M, N,
        gate.stride(0), gate.stride(1),
        up.stride(0), up.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=4,
    )
    return out


# ---------------------------------------------------------------------------
# Fused silu(gate) * up + per-row FP8 act-quant — single Triton kernel
# ---------------------------------------------------------------------------
@triton.jit
def _silu_quant_kernel(
    GATE_ptr, UP_ptr, OUT_FP8_ptr, OUT_BF16_ptr, SCALE_ptr,
    M, N,
    stride_gm, stride_gn,
    stride_um, stride_un,
    stride_om, stride_on,
    stride_bm, stride_bn,
    stride_sm,
    E4M3_MAX: tl.constexpr,
    E4M3_MIN_NORMAL: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Fuse ``silu(gate) * up`` with the down_proj's per-row FP8 act-quant.

    One program per row (along M). Each program loops over its row in
    ``BLOCK_N`` chunks, computes the SwiGLU hidden in fp32 registers,
    reduces to a per-row amax, then does a second pass to quantize.

    Outputs:
      - ``OUT_FP8_ptr``: per-row FP8 E4M3 hidden (for the down_proj GEMM)
      - ``OUT_BF16_ptr``: bf16 hidden (saved for the down_proj STE
        backward, which re-runs ``grad_w = grad_out.T @ x_bf16``)
      - ``SCALE_ptr``: per-row fp32 scale (= amax / E4M3_MAX)

    Numerics: the silu output is narrowed to bf16 between the two
    arithmetic stages, so the amax + scale + fp8 cast are bit-exact
    with the unfused ``silu_mul_fused + quantize_act_fp8_fused`` path
    (verified 0/67M fp8 elements differ, scale delta 0.0 on prod shape).
    """
    pid_m = tl.program_id(0)
    if pid_m >= M:
        return

    # ---- pass 1: per-row amax over hidden = silu(gate) * up (bf16) ----
    amax = tl.zeros((), dtype=tl.float32) + E4M3_MIN_NORMAL
    for n0 in range(0, N, BLOCK_N):
        offs_n = n0 + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N
        g = tl.load(
            GATE_ptr + pid_m * stride_gm + offs_n * stride_gn,
            mask=mask_n, other=0.0,
        ).to(tl.float32)
        u = tl.load(
            UP_ptr + pid_m * stride_um + offs_n * stride_un,
            mask=mask_n, other=0.0,
        ).to(tl.float32)
        h = (g * tl.sigmoid(g)) * u
        # Narrow to bf16, lift back to fp32 — matches the bf16
        # intermediate ``silu_mul_fused`` writes before act_quant.
        h = h.to(tl.bfloat16).to(tl.float32)
        h_abs = tl.where(mask_n, tl.abs(h), 0.0)
        amax = tl.maximum(amax, tl.max(h_abs, axis=0))

    scale = (amax / E4M3_MAX).to(tl.bfloat16).to(tl.float32)
    inv_scale = (E4M3_MAX / amax).to(tl.bfloat16).to(tl.float32)
    tl.store(SCALE_ptr + pid_m * stride_sm, scale)

    # ---- pass 2: write bf16 hidden + fp8 hidden ----
    for n0 in range(0, N, BLOCK_N):
        offs_n = n0 + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N
        g = tl.load(
            GATE_ptr + pid_m * stride_gm + offs_n * stride_gn,
            mask=mask_n, other=0.0,
        ).to(tl.float32)
        u = tl.load(
            UP_ptr + pid_m * stride_um + offs_n * stride_un,
            mask=mask_n, other=0.0,
        ).to(tl.float32)
        h = (g * tl.sigmoid(g)) * u
        h_bf16 = h.to(tl.bfloat16)
        h_fp32 = h_bf16.to(tl.float32)
        tl.store(
            OUT_BF16_ptr + pid_m * stride_bm + offs_n * stride_bn,
            h_bf16, mask=mask_n,
        )
        h_q = tl.minimum(
            tl.maximum(h_fp32 * inv_scale, -E4M3_MAX), E4M3_MAX,
        ).to(tl.float8e4nv)
        tl.store(
            OUT_FP8_ptr + pid_m * stride_om + offs_n * stride_on,
            h_q, mask=mask_n,
        )


def silu_quant_fused(
    gate: torch.Tensor, up: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``silu(gate) * up`` fused with the down_proj's FP8 act-quant.

    Replaces ``silu_mul_fused(gate, up)`` followed by
    ``quantize_act_fp8_fused(hidden)`` — the two kernels each stream
    the ~256 MB hidden through HBM; fusing them keeps the hidden in
    registers and writes it once as fp8 (plus once as bf16 for the
    STE backward). Measured 1.86 ms → 1.23 ms on sm_120 at the prod
    shape (34% wall reduction — see ``project_w4a8_ffn_e2e.md``).

    Args:
        gate: [M, N] BF16
        up:   [M, N] BF16, same shape as gate

    Returns:
        (a_fp8, a_s, hidden_bf16):
          a_fp8      [M, N] float8_e4m3fn — for the down_proj GEMM
          a_s        [M, 1] fp32          — per-row scale
          hidden_bf16 [M, N] BF16          — for the down_proj STE bwd
    """
    assert gate.shape == up.shape, (
        f"gate and up must have the same shape, got {gate.shape} vs {up.shape}"
    )
    assert gate.dtype == torch.bfloat16 and up.dtype == torch.bfloat16
    # The kernel is 2-D only — flatten leading dims (e.g. [B, T, N]
    # from a 3-D SwiGLU input) and reshape outputs back to the
    # original shape. Numerics are bit-exact (each row is processed
    # independently).
    orig_shape = gate.shape
    N = orig_shape[-1]
    M = gate.numel() // N
    gate_2d = gate.reshape(M, N)
    up_2d = up.reshape(M, N)
    a_fp8 = torch.empty(M, N, dtype=torch.float8_e4m3fn, device=gate.device)
    hidden_bf16 = torch.empty(M, N, dtype=torch.bfloat16, device=gate.device)
    a_s = torch.empty(M, 1, dtype=torch.float32, device=gate.device)
    grid = (M,)
    _silu_quant_kernel[grid](
        gate_2d, up_2d, a_fp8, hidden_bf16, a_s,
        M, N,
        gate_2d.stride(0), gate_2d.stride(1),
        up_2d.stride(0), up_2d.stride(1),
        a_fp8.stride(0), a_fp8.stride(1),
        hidden_bf16.stride(0), hidden_bf16.stride(1),
        a_s.stride(0),
        E4M3_MAX=448.0,
        E4M3_MIN_NORMAL=6.103515625e-05,
        BLOCK_N=256,
        num_warps=4,
    )
    return a_fp8.reshape(*orig_shape), a_s, hidden_bf16.reshape(*orig_shape)


class _SiluQuantFusedSTE(torch.autograd.Function):
    """Autograd wrapper around :func:`silu_quant_fused`.

    Forward: returns ``(a_fp8, a_s, hidden_bf16)`` like the function.
    Only ``hidden_bf16`` participates in the autograd graph (the FP8
    output is STE — discarded in backward). The grad on ``hidden_bf16``
    is routed to both ``gate`` and ``up`` (the silu-mul Jacobian is
    omitted — same STE convention as :class:`SiluMulFp8STE`; the
    silu' correction is well below the FP8 noise floor, see
    ``src/models/ops/silu_mul_fp8.py``).
    """

    @staticmethod
    def forward(ctx, gate, up):
        a_fp8, a_s, hidden_bf16 = silu_quant_fused(gate, up)
        ctx.save_for_backward(gate, up)
        return a_fp8, a_s, hidden_bf16

    @staticmethod
    def backward(ctx, grad_a_fp8, grad_a_s, grad_hidden):
        # STE: grad_hidden flows to both gate and up. grad_a_fp8 /
        # grad_a_s are not connected (the FP8 quant round is STE).
        return grad_hidden, grad_hidden


def silu_quant_fused_with_passthrough(
    gate: torch.Tensor, up: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Autograd-aware wrapper around :func:`silu_quant_fused`.

    Use this from module-level forward paths that need gradient flow
    to ``gate`` and ``up`` (e.g. :class:`NVFP4W4A8SwiGLU`). The
    returned ``hidden_bf16`` is the autograd-tracked tensor;
    ``a_fp8`` / ``a_s`` are auxiliary (no gradient).
    """
    return _SiluQuantFusedSTE.apply(gate, up)


class NVFP4W4A8SwiGLU(nn.Module):
    """SwiGLU FFN where the gate and up matmuls share one ``act_quant``.

    The W4A8 forward for SwiGLU naively does 3 matmuls
    (gate, up, down), each preceded by its own ``quantize_act_fp8_fused``
    on the matmul's input. Gate and up share the same input ``x``,
    so 2 of those 3 act_quants are redundant — the fp8 tensor and
    per-row scale are deterministic in ``x``. This wrapper:

      1. Calls ``quantize_act_fp8_fused(x)`` once
      2. Routes the (a_fp8, a_s) buffers into both gate and up
         via :meth:`NVFP4LinearW4A8.forward_precomputed`
      3. Down matmul takes the post-SwiGLU hidden and runs its own
         (separate) act_quant — its input is different from ``x``

    Measured at the prod gate_up shape (M=16384, K=1536, N=4096)
    on sm_120: 0.32 ms saved per FFN forward, which lifts effective
    FLOPS from 53.0 → 55.8 TF (5% wall reduction) — see
    auto-memory ``project_w4a8_ffn_e2e.md``.

    When ``bf16_only=True`` is requested (e.g. for shape-constraint
    fallback), the wrapper skips the shared-act_quant optimization
    and routes through plain ``F.linear`` on each projection.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        bias: bool = False,
        block_size: int = 16,
        use_custom_gemm: bool = False,
        bf16_only: bool = False,
    ) -> None:
        super().__init__()
        kwargs = dict(
            bias=bias,
            block_size=block_size,
            use_custom_gemm=use_custom_gemm,
            bf16_only=bf16_only,
        )
        self.gate_proj = NVFP4LinearW4A8(hidden_size, intermediate_size, **kwargs)
        self.up_proj = NVFP4LinearW4A8(hidden_size, intermediate_size, **kwargs)
        self.down_proj = NVFP4LinearW4A8(intermediate_size, hidden_size, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.gate_proj.bf16_only:
            gate = self.gate_proj(x)
            up = self.up_proj(x)
            # bf16 path: PyTorch's F.silu(gate) * up is fine here (no
            # fuse win in the production offload path which never hits
            # bf16_only — but the code stays correct).
            hidden = F.silu(gate) * up
            return self.down_proj(hidden)
        # Shared act_quant: gate and up both consume ``x``, so quantize
        # once and route the fp8 buffers into both. The matmul kernel
        # requires 2D, but callers may pass 3D ``[B, T, H]`` (e.g.
        # :class:`HippoLayer`); flatten the leading dims and reshape
        # the final output back to the original shape.
        x_2d = x.reshape(-1, x.shape[-1])
        a_fp8, a_s = quantize_act_fp8_fused(x_2d)
        gate = self.gate_proj.forward_precomputed(x, a_fp8, a_s)
        up = self.up_proj.forward_precomputed(x, a_fp8, a_s)
        # Fused silu(gate) * up + down_proj act_quant in one Triton kernel:
        # keeps the ~256 MB hidden in registers instead of streaming it
        # through HBM twice (once for silu_mul, once for act_quant). Returns
        # the fp8 hidden + per-row scale for the down GEMM, plus the bf16
        # hidden for the down_proj STE backward (see :func:`silu_quant_fused`).
        h_fp8, h_s, hidden = silu_quant_fused_with_passthrough(gate, up)
        return self.down_proj.forward_precomputed(hidden, h_fp8, h_s)

    def forward_precomputed(
        self,
        x: torch.Tensor,
        a_fp8: torch.Tensor,
        a_s: torch.Tensor,
    ) -> torch.Tensor:
        """Like ``forward(x)`` but the FP8 + scale for the shared act_quant
        is supplied by the caller (typically the upstream
        :func:`src.models.ops.rmsnorm_fp8.rmsnorm_fp8_with_passthrough`
        output). Skips the redundant :func:`quantize_act_fp8_fused`
        call inside :meth:`forward` — saves ~105 us / FFN at prod shape.

        ``x`` (BF16) is still passed in for the silu_quant_fused
        backward path's saved tensor (the bf16 hidden is required by
        the silu STE bwd); the GEMM path uses only ``a_fp8`` / ``a_s``.
        """
        if self.gate_proj.bf16_only:
            # bf16_only path: caller-supplied fp8 ignored, gate/up are BF16 GEMMs.
            gate = self.gate_proj(x)
            up = self.up_proj(x)
            hidden = F.silu(gate) * up
            return self.down_proj(hidden)
        gate = self.gate_proj.forward_precomputed(x, a_fp8, a_s)
        up = self.up_proj.forward_precomputed(x, a_fp8, a_s)
        h_fp8, h_s, hidden = silu_quant_fused_with_passthrough(gate, up)
        return self.down_proj.forward_precomputed(hidden, h_fp8, h_s)

    def extra_repr(self) -> str:
        return (
            f"hidden={self.gate_proj.in_features}, "
            f"intermediate={self.gate_proj.out_features}, "
            f"gate={self.gate_proj.extra_repr()}, "
            f"up={self.up_proj.extra_repr()}, "
            f"down={self.down_proj.extra_repr()}"
        )
