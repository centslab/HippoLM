"""FP8 E4M3 GEMM via hand-written Round 3 kernel (sm_120).

Round 3 = warp-specialized structure (CUTLASS-style):
  - 1 producer WG (4 warps): warp 0 lane 0 issues TMA for A and B
  - 2 consumer WGs (4 warps each, 8 warps total): 4x2 grid of
    32x64 sub-tiles, MMA + per-MmaTile scale + accumulate
  - NUM_STAGES=3 (deeper pipeline so producer can stay ahead)
  - DIRECT_STORE=true (skip Y_out smem buffer, frees BM*BN*2 bytes)
  - Swizzled tile rasterization (SWIZZLE_WIDTH=4) for L2 locality

Same per-MmaTile (1x1) FP32 scales as R0 / R2 / CUTLASS 87a.

Inputs:
  A   : [M, K] float8_e4m3fn  row-major
  B   : [N, K] float8_e4m3fn  row-major  (project convention)
  SFA : [M/128, K/128] FP32   (per-MmaTile scale; row-major shape)
  SFB : [N/128, K/128] FP32
  D   : [M, N] bfloat16        row-major

CUTLASS MN-major layout: the flat memory is laid out as
``data[m_tile + k_tile * (M/128)]`` (M is the fastest dim). We accept
the natural Python row-major shape and transpose on the fly -- one
HBM-to-HBM copy per scale tensor, negligible vs the GEMM.

This is a research path for closing the 96->188 TF gap on sm_120
(see auto-memory `project_fp8_gemm_bottleneck.md`). Built via
``scripts/build_fp8_hand_gemm.py``.
"""
from __future__ import annotations

import ctypes
from pathlib import Path
from typing import Optional

import torch

_LIB_DIR = Path(__file__).resolve().parent / "lib"
_LIB: Optional[ctypes.CDLL] = None


def _sm_tag() -> str:
    major, minor = torch.cuda.get_device_capability()
    return f"sm_{major}{minor}"


def _load() -> ctypes.CDLL:
    global _LIB
    if _LIB is not None:
        return _LIB
    sm = _sm_tag()
    so_path = _LIB_DIR / f"fp8_hand_r3_{sm}.so"
    if not so_path.exists():
        raise RuntimeError(
            f"No hand-written FP8 GEMM R3 for {sm} at {so_path}; run "
            "scripts/build_fp8_hand_gemm.py"
        )
    lib = ctypes.CDLL(str(so_path), mode=ctypes.RTLD_LOCAL)
    lib.fp8_hand_r3_gemm_run.restype = ctypes.c_int
    lib.fp8_hand_r3_gemm_run.argtypes = [
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p,
    ]
    _LIB = lib
    return lib


def is_available() -> bool:
    if not torch.cuda.is_available():
        return False
    sm = _sm_tag()
    return (_LIB_DIR / f"fp8_hand_r3_{sm}.so").exists()


def fp8_gemm_hand_r3(
    A: torch.Tensor,
    B_nk: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Per-MmaTile FP8 GEMM, R3 (warp-specialized). See module docstring."""
    assert A.dtype == torch.float8_e4m3fn and A.is_cuda and A.is_contiguous()
    assert B_nk.dtype == torch.float8_e4m3fn and B_nk.is_cuda and B_nk.is_contiguous()
    assert a_scale.dtype == torch.float32 and a_scale.is_cuda and a_scale.is_contiguous()
    assert b_scale.dtype == torch.float32 and b_scale.is_cuda and b_scale.is_contiguous()
    M, K = A.shape
    N, K2 = B_nk.shape
    assert K == K2 and K % 128 == 0 and N % 128 == 0 and M % 128 == 0
    assert a_scale.shape == (M // 128, K // 128), (
        f"a_scale must be [M/128, K/128] per-MmaTile, got {tuple(a_scale.shape)} "
        f"for M={M}, K={K}"
    )
    assert b_scale.shape == (N // 128, K // 128), (
        f"b_scale must be [N/128, K/128] per-MmaTile, got {tuple(b_scale.shape)} "
        f"for N={N}, K={K}"
    )
    # CUTLASS MN-major layout is K-major in memory:
    #   data[m_tile + k_tile * (M/128)]
    # Python row-major (M/128, K/128) is M-major, K-minor -- transpose.
    a_scale_dev = a_scale.t().contiguous()  # [K/128, M/128] row-major
    b_scale_dev = b_scale.t().contiguous()  # [K/128, N/128] row-major
    if out is None:
        out = torch.empty((M, N), dtype=torch.bfloat16, device=A.device)
    else:
        assert out.shape == (M, N) and out.dtype == torch.bfloat16 and out.is_contiguous()
    stream = torch.cuda.current_stream().cuda_stream
    rc = _load().fp8_hand_r3_gemm_run(
        M, N, K,
        A.data_ptr(), B_nk.data_ptr(),
        a_scale_dev.data_ptr(), b_scale_dev.data_ptr(),
        out.data_ptr(), stream,
    )
    if rc != 0:
        raise RuntimeError(f"fp8_hand_r3_gemm_run failed (rc={rc})")
    return out