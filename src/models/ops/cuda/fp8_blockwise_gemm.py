"""FP8 E4M3 GEMM with BF16 per-1x32 K-block scales."""
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
    candidates = sorted(_LIB_DIR.glob(f"fp8_blockwise_gemm_{sm}_*.so"))
    if not candidates:
        raise RuntimeError(
            f"No blockwise FP8 GEMM for {sm}; run "
            "scripts/build_fp8_blockwise_gemm.py"
        )
    preferred = _LIB_DIR / (
        f"fp8_blockwise_gemm_{sm}_"
        "bm64_bn128_bk128_s3_cwg1_wm32_wn64_ds_blockwise.so"
    )
    path = preferred if preferred.exists() else candidates[0]
    _LIB = ctypes.CDLL(str(path), mode=ctypes.RTLD_LOCAL)
    _LIB.blockwise_gemm_run.restype = None
    _LIB.blockwise_gemm_run.argtypes = [
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ]
    return _LIB


def is_available() -> bool:
    if not torch.cuda.is_available():
        return False
    major, minor = torch.cuda.get_device_capability()
    sm = f"sm_{major}{minor}"
    return any(_LIB_DIR.glob(f"fp8_blockwise_gemm_{sm}_*.so"))


def fp8_blockwise_gemm(
    A: torch.Tensor,
    B: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute ``(A * a_scale) @ (B * b_scale).T`` with 1x32 scales.

    A is ``[M,K]`` and B is ``[N,K]`` float8 E4M3. Scales are BF16
    ``[M,K//32]`` and ``[N,K//32]`` respectively. The scale is applied
    to each 32-wide K partial before accumulation.
    """
    assert A.dtype == torch.float8_e4m3fn and A.is_cuda and A.is_contiguous()
    assert B.dtype == torch.float8_e4m3fn and B.is_cuda and B.is_contiguous()
    assert a_scale.dtype == torch.bfloat16 and a_scale.is_cuda and a_scale.is_contiguous()
    assert b_scale.dtype == torch.bfloat16 and b_scale.is_cuda and b_scale.is_contiguous()
    M, K = A.shape
    N, K2 = B.shape
    assert K == K2 and K % 32 == 0
    assert a_scale.shape == (M, K // 32)
    assert b_scale.shape == (N, K // 32)
    if out is None:
        out = torch.empty((M, N), dtype=torch.bfloat16, device=A.device)
    else:
        assert out.shape == (M, N) and out.dtype == torch.bfloat16 and out.is_contiguous()
    stream = torch.cuda.current_stream().cuda_stream
    _load().blockwise_gemm_run(
        M, N, K,
        A.data_ptr(), B.data_ptr(), out.data_ptr(),
        a_scale.data_ptr(), b_scale.data_ptr(), 0, stream,
    )
    return out
