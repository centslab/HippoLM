"""FP8 E4M3 GEMM with R10 production strategy: 2D 64×128 BF16 block scales,
BLOCK_ACCUM, ACC_RAW_F16 (F16 in-block accumulator, sm_120 2x rate).

Used as the perf baseline for the hybrid (FSFP8) GEMM probe — both
share the same BM/BN/BK/NUM_STAGES/CWG/WM/WN config (128/128/128/3/2/32/64)
and DIRECT_STORE, but differ in A-scale layout (2D vs 1D per-row).

Layout:
  A : [M, K] float8_e4m3fn
  B : [N, K] float8_e4m3fn
  a_block_scale : [M/64, K/128] bfloat16   (2D 64×128, same as b_block_scale)
  b_block_scale : [N/64, K/128] bfloat16
"""
from __future__ import annotations

import ctypes
from pathlib import Path
from typing import Optional

import torch


_LIB_DIR = Path(__file__).resolve().parent / "lib"
_LIB: Optional[ctypes.CDLL] = None


def _load() -> ctypes.CDLL:
    global _LIB
    if _LIB is not None:
        return _LIB
    major, minor = torch.cuda.get_device_capability()
    sm = f"sm_{major}{minor}"
    so_path = _LIB_DIR / (
        f"fp8_blockwise_gemm_{sm}_"
        "bm128_bn128_bk128_s3_cwg2_wm32_wn64_ds_blockwise_accum_bsk128_f16.so"
    )
    if not so_path.exists():
        raise RuntimeError(f"No R10 FP8 GEMM .so for {sm} at {so_path}.")
    _LIB = ctypes.CDLL(str(so_path), mode=ctypes.RTLD_LOCAL)
    _LIB.blockwise_accum_bsk128_f16_gemm_run.restype = None
    _LIB.blockwise_accum_bsk128_f16_gemm_run.argtypes = [
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ]
    return _LIB


def fp8_r10_gemm(
    A: torch.Tensor,
    B: torch.Tensor,
    a_block_scale_2d: torch.Tensor,
    b_block_scale_2d: torch.Tensor,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    lib = _load()
    assert A.dtype == torch.float8_e4m3fn and A.is_cuda and A.is_contiguous()
    assert B.dtype == torch.float8_e4m3fn and B.is_cuda and B.is_contiguous()
    M, K = A.shape
    N, K2 = B.shape
    assert K == K2 and K % 128 == 0 and M % 128 == 0 and N % 128 == 0
    assert a_block_scale_2d.shape == (M // 64, K // 128)
    assert b_block_scale_2d.shape == (N // 64, K // 128)
    assert a_block_scale_2d.dtype == torch.bfloat16 and a_block_scale_2d.is_contiguous()
    assert b_block_scale_2d.dtype == torch.bfloat16 and b_block_scale_2d.is_contiguous()
    if out is None:
        out = torch.empty(M, N, dtype=torch.bfloat16, device=A.device)
    else:
        assert out.shape == (M, N) and out.dtype == torch.bfloat16 and out.is_contiguous()
    stream = torch.cuda.current_stream().cuda_stream
    lib.blockwise_accum_bsk128_f16_gemm_run(
        M, N, K,
        A.data_ptr(), B.data_ptr(), out.data_ptr(),
        a_block_scale_2d.data_ptr(), b_block_scale_2d.data_ptr(),
        0, stream,
    )
    return out