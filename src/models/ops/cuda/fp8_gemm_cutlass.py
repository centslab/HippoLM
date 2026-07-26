"""FP8 E4M3 GEMM via project-local CUTLASS-based kernel (sm_120).

Mirrors /hy-tmp/cutlass/examples/87_blackwell_geforce_gemm_blockwise/87a_*:
  - TileShape_MNK = 128x128x128
  - Per-MmaTile FP32 scales (one scale per 128x128x128 block — i.e., a
    single scale value is applied to all 128x128 elements within a
    single MmaTile, NOT per-row × per-128-K-block)
  - Hardware-supported sm_120 warp-specialized blockwise path
  - B is [K, N] row-major (CUTLASS "TN" layout); the kernel sees
    B(n, k) = data[k*N + n] in memory.
  - Scale memory layout: MN-major. The flat data is laid out as
    `data[m_tile + k_tile * (M/128)]` (M is the fastest dim). The
    wrapper accepts the natural Python shape `(M/128, K/128)` row-major
    and transposes on the fly to CUTLASS's expected K-major layout
    (one HBM-to-HBM copy per scale — negligible vs GEMM cost).

For the KDA path (K = 128 head dim), per-MmaTile scaling is coarser
than the existing per-row TensorWise scalar. The win is the
hardware-supported blockwise MMA path, not scale granularity.

Built per-arch via :file:`scripts/build_fp8_cutlass_gemm.py`.
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
    so_path = _LIB_DIR / f"fp8_gemm_cutlass_{sm}.so"
    if not so_path.exists():
        raise RuntimeError(
            f"No CUTLASS FP8 GEMM for {sm} at {so_path}; run "
            "scripts/build_fp8_cutlass_gemm.py"
        )
    lib = ctypes.CDLL(str(so_path), mode=ctypes.RTLD_LOCAL)
    # cutlass_fp8_blockwise_gemm_run(
    #     int M, int N, int K,
    #     const void* A, const void* B,           # A=[M,K] row-major, B=[K,N] row-major
    #     const void* SFA, const void* SFB,       # [M,K/128] and [N,K/128] FP32
    #     void* D,                                # [M,N] BF16
    #     cudaStream_t stream)
    lib.cutlass_fp8_blockwise_gemm_run.restype = ctypes.c_int
    lib.cutlass_fp8_blockwise_gemm_run.argtypes = [
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
    return (_LIB_DIR / f"fp8_gemm_cutlass_{sm}.so").exists()


def fp8_blockwise_gemm(
    A: torch.Tensor,
    B_nk: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """FP8 E4M3 GEMM with per-MmaTile (128x128x128) FP32 blockwise scales.

    A is [M, K] FP8 row-major.  B_nk is [N, K] FP8 row-major
    (project convention — matches :func:`fp8_gemm_scaled`). The kernel
    computes Y = A @ B.T internally with CUTLASS TN layout (B viewed
    as [N, K] column-major).  Scales are FP32 with shape [M/128, K/128]
    and [N/128, K/128] (one scale per MmaTile).  Output is BF16 [M, N]
    row-major.

    The CUTLASS inner scale layout is MN-major: the flat data is laid
    out as `data[m_tile + k_tile * (M/128)]` (M is the fastest dim). We
    accept the natural Python row-major (M/128, K/128) and transpose on
    the fly — one HBM-to-HBM copy per scale, negligible vs the GEMM.
    """
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
    # CUTLASS MN-major: stride (1, M/128) over (M/128, K/128).
    # Python row-major is (M/128*K/128) strides → M-major, K-minor.
    # Transpose to get K-major layout that matches CUTLASS.
    a_scale_dev = a_scale.t().contiguous()  # [K/128, M/128] row-major
    b_scale_dev = b_scale.t().contiguous()  # [K/128, N/128] row-major
    if out is None:
        out = torch.empty((M, N), dtype=torch.bfloat16, device=A.device)
    else:
        assert out.shape == (M, N) and out.dtype == torch.bfloat16 and out.is_contiguous()
    stream = torch.cuda.current_stream().cuda_stream
    rc = _load().cutlass_fp8_blockwise_gemm_run(
        M, N, K,
        A.data_ptr(), B_nk.data_ptr(),
        a_scale_dev.data_ptr(), b_scale_dev.data_ptr(),
        out.data_ptr(), stream,
    )
    if rc != 0:
        raise RuntimeError(f"cutlass_fp8_blockwise_gemm_run failed (rc={rc})")
    return out
