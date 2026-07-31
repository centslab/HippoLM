"""W4A8 GEMM — Python entry point.

Two implementations, selected by the `native` flag:

1. **Two-pass (default, robust)**:
   - `dequant_nvfp4_to_fp8` from `src.models.ops.nvfp4_linear_w4a8`: NVFP4
     packed → FP8 E4M3 (per-microblock E4M3 scale + global scale folded in,
     clamped to ±448).
   - The R10 hybrid GEMM (`hybrid_a1x128_twb_f32`, F32 acc): FP8 A (per-row
     1×128 BF16 scale) @ FP8 B (tensorwise scale = 1, since B is already
     dequantized) → BF16 out.

2. **Native (experimental)**: `src/models/ops/cuda/fp8_gemm_build/sources/
   w4a8_gemm.cuh` — NVFP4 dequant fused into the GEMM producer. Correct for
   single-K-tile shapes (K ≤ 128) but has an unresolved async-proxy smem
   visibility race in the producer for multi-K-tile pipelines (nondeterministic
   NaN in the second stage's dequant output). See MEMORY.md "W4A8 native —
   plan" and the 2026-07-31 debug log. Kept as experimental until the race
   is fixed; the two-pass path is the production default.

Args:
    A            : [M, K] float8_e4m3fn (pre-quantized activation, 1×128 BF16 scale)
    B_packed     : [N, K/2] uint8 NVFP4 packed (low nibble = even K, high = odd K)
    B_scales     : [N, K/16] float8_e4m3fn E4M3 microblock scales
    B_global     : [1] FP32 scalar global scale
    a_block_scale: [M, K/128] bfloat16 per-row 1×128 scale
    b_block_scale: [K/128] bfloat16 (unused; API parity)
    native       : use the experimental fused-dequant kernel (single-K-tile only)
    out          : optional [M, N] bfloat16 output

Returns: [M, N] bfloat16.
"""
from __future__ import annotations

import ctypes
from pathlib import Path
from typing import Optional

import torch

from src.models.ops.nvfp4_linear_w4a8 import dequant_nvfp4_to_fp8


_LIB_DIR = Path(__file__).resolve().parent / "lib"
_sm_tag_cache: Optional[str] = None


def _sm_tag() -> str:
    global _sm_tag_cache
    if _sm_tag_cache is None:
        major, minor = torch.cuda.get_device_capability()
        _sm_tag_cache = f"sm_{major}{minor}"
    return _sm_tag_cache


# ---------------------------------------------------------------------------
# Native kernel binding (experimental — single-K-tile only)
# ---------------------------------------------------------------------------
_NATIVE_LIB: Optional[ctypes.CDLL] = None


def _load_native() -> ctypes.CDLL:
    global _NATIVE_LIB
    if _NATIVE_LIB is not None:
        return _NATIVE_LIB
    sm = _sm_tag()
    so_path = _LIB_DIR / f"w4a8_gemm_{sm}.so"
    if not so_path.exists():
        raise RuntimeError(
            f"No prebuilt w4a8_gemm .so for {sm} at {so_path}. "
            "Run `python scripts/build_w4a8_gemm.py` to build it."
        )
    _NATIVE_LIB = ctypes.CDLL(str(so_path), mode=ctypes.RTLD_LOCAL)
    _NATIVE_LIB.w4a8_gemm_run.restype = None
    _NATIVE_LIB.w4a8_gemm_run.argtypes = [
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p,
    ]
    return _NATIVE_LIB


def _run_native(
    A, B_packed, B_scales, B_global, a_block_scale, b_block_scale, out,
):
    lib = _load_native()
    M, K = A.shape
    N, _ = B_packed.shape
    stream = torch.cuda.current_stream().cuda_stream
    lib.w4a8_gemm_run(
        M, N, K,
        A.data_ptr(), B_packed.data_ptr(), B_scales.data_ptr(), B_global.data_ptr(),
        0, a_block_scale.data_ptr(), b_block_scale.data_ptr(),
        out.data_ptr(), 0, 0, stream,
    )
    return out


# ---------------------------------------------------------------------------
# R10 hybrid GEMM binding (two-pass path — production default)
# ---------------------------------------------------------------------------
_HYBRID_LIB: Optional[ctypes.CDLL] = None


def _load_hybrid() -> ctypes.CDLL:
    global _HYBRID_LIB
    if _HYBRID_LIB is not None:
        return _HYBRID_LIB
    sm = _sm_tag()
    so_path = _LIB_DIR / (
        f"fp8_blockwise_gemm_{sm}_bm128_bn128_bk128_s3_cwg2_wm32_wn64_ds_"
        "blockwise_hybrid_a1x128_twb_f32_bf16.so"
    )
    if not so_path.exists():
        raise RuntimeError(
            f"No hybrid_a1x128_twb_f32 .so for {sm}. Run "
            "`python scripts/build_fp8_blockwise_gemm.py --variant "
            "hybrid_a1x128_twb_f32`."
        )
    _HYBRID_LIB = ctypes.CDLL(str(so_path), mode=ctypes.RTLD_LOCAL)
    _HYBRID_LIB.hybrid_a1x128_twb_f32_bf16_gemm_run.restype = None
    _HYBRID_LIB.hybrid_a1x128_twb_f32_bf16_gemm_run.argtypes = [
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ]
    return _HYBRID_LIB


def _run_hybrid(A, B_fp8, a_block_scale, out):
    """R10 hybrid GEMM: FP8 A (1×128) @ FP8 B (tensorwise scale 1) → BF16."""
    lib = _load_hybrid()
    M, K = A.shape
    N, _ = B_fp8.shape
    b_s = torch.ones(K // 128, dtype=torch.bfloat16, device=A.device)
    stream = torch.cuda.current_stream().cuda_stream
    lib.hybrid_a1x128_twb_f32_bf16_gemm_run(
        M, N, K,
        A.data_ptr(), B_fp8.data_ptr(), out.data_ptr(),
        a_block_scale.data_ptr(), b_s.data_ptr(), 0, stream,
    )
    return out


def is_available() -> bool:
    if not torch.cuda.is_available():
        return False
    sm = _sm_tag()
    return (
        _LIB_DIR
        / f"fp8_blockwise_gemm_{sm}_bm128_bn128_bk128_s3_cwg2_wm32_wn64_ds_"
          "blockwise_hybrid_a1x128_twb_f32_bf16.so"
    ).exists()


def w4a8_gemm(
    A: torch.Tensor,
    B_packed: torch.Tensor,
    B_scales: torch.Tensor,
    B_global: torch.Tensor,
    a_block_scale: torch.Tensor,
    b_block_scale: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    a_scale_row: Optional[torch.Tensor] = None,
    native: bool = False,
) -> torch.Tensor:
    """Run W4A8 GEMM (default: two-pass dequant + R10 hybrid F32-acc GEMM).

    Args:
        A            : [M, K] float8_e4m3fn
        B_packed     : [N, K/2] uint8 NVFP4 packed
        B_scales     : [N, K/16] float8_e4m3fn
        B_global     : [1] FP32 scalar
        a_block_scale: [M, K/128] bfloat16 per-row 1×128
        b_block_scale: [K/128] bfloat16 (unused)
        out          : optional [M, N] bfloat16
        a_scale_row  : optional [M] FP32 (unused; API parity)
        native       : use the experimental fused-dequant kernel
    """
    assert A.dtype == torch.float8_e4m3fn and A.is_cuda and A.is_contiguous()
    assert B_packed.dtype == torch.uint8 and B_packed.is_cuda and B_packed.is_contiguous()
    assert B_scales.dtype == torch.float8_e4m3fn and B_scales.is_cuda and B_scales.is_contiguous()
    assert B_global.dtype == torch.float32 and B_global.is_cuda
    assert a_block_scale.dtype == torch.bfloat16 and a_block_scale.is_cuda and a_block_scale.is_contiguous()

    M, K = A.shape
    N, K2 = B_packed.shape
    assert K == K2 * 2, f"K mismatch: A has {K}, B_packed has K/2={K2}"
    assert M % 128 == 0 and N % 128 == 0 and K % 128 == 0, (
        f"M/N/K must be multiples of 128, got M={M} N={N} K={K}")
    assert B_scales.shape == (N, K // 16)
    assert a_block_scale.shape == (M, K // 128)

    if out is None:
        out = torch.empty(M, N, dtype=torch.bfloat16, device=A.device)
    else:
        assert out.shape == (M, N) and out.dtype == torch.bfloat16 and out.is_contiguous()

    if native:
        # Experimental fused-dequant kernel. Single-K-tile only (K ≤ 128);
        # multi-K-tile has an unresolved producer async-proxy smem race.
        assert K <= 128, (
            "native path is single-K-tile only (K <= 128); the multi-K-tile "
            "producer dequant has a race. Use native=False (two-pass).")
        return _run_native(A, B_packed, B_scales, B_global,
                           a_block_scale, b_block_scale, out)

    # Two-pass: NVFP4 → FP8, then R10 hybrid GEMM (F32 acc, overflow-safe).
    b_fp8 = torch.empty(N, K, dtype=torch.float8_e4m3fn, device=A.device)
    dequant_nvfp4_to_fp8(B_packed, B_scales, B_global.item(), out=b_fp8)
    return _run_hybrid(A, b_fp8, a_block_scale, out)