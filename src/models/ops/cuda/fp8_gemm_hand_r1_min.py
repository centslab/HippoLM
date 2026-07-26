"""R1-min wrapper (R0 + NUM_STAGES=3 only)."""
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
    so_path = _LIB_DIR / f"fp8_hand_r1min_{sm}.so"
    if not so_path.exists():
        raise RuntimeError(
            f"No R1-min .so at {so_path}; run scripts/build_fp8_hand_gemm.py"
        )
    lib = ctypes.CDLL(str(so_path), mode=ctypes.RTLD_LOCAL)
    lib.fp8_hand_r1min_gemm_run.restype = ctypes.c_int
    lib.fp8_hand_r1min_gemm_run.argtypes = [
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
    return (_LIB_DIR / f"fp8_hand_r1min_{sm}.so").exists()


def fp8_gemm_hand_r1min(A, B_nk, a_scale, b_scale, out=None):
    assert A.dtype == torch.float8_e4m3fn and A.is_cuda and A.is_contiguous()
    assert B_nk.dtype == torch.float8_e4m3fn and B_nk.is_cuda and B_nk.is_contiguous()
    assert a_scale.dtype == torch.float32 and a_scale.is_cuda and a_scale.is_contiguous()
    assert b_scale.dtype == torch.float32 and b_scale.is_cuda and b_scale.is_contiguous()
    M, K = A.shape
    N, K2 = B_nk.shape
    assert K == K2 and K % 128 == 0 and N % 128 == 0 and M % 128 == 0
    assert a_scale.shape == (M // 128, K // 128)
    assert b_scale.shape == (N // 128, K // 128)
    a_scale_dev = a_scale.t().contiguous()
    b_scale_dev = b_scale.t().contiguous()
    if out is None:
        out = torch.empty((M, N), dtype=torch.bfloat16, device=A.device)
    else:
        assert out.shape == (M, N) and out.dtype == torch.bfloat16 and out.is_contiguous()
    stream = torch.cuda.current_stream().cuda_stream
    rc = _load().fp8_hand_r1min_gemm_run(
        M, N, K,
        A.data_ptr(), B_nk.data_ptr(),
        a_scale_dev.data_ptr(), b_scale_dev.data_ptr(),
        out.data_ptr(), stream,
    )
    if rc != 0:
        raise RuntimeError(f"fp8_hand_r1min_gemm_run failed (rc={rc})")
    return out
