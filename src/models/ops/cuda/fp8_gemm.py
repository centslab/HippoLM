"""FP8 E4M3 GEMM via custom CUDA C++ kernel (TMA + warp specialization).

Computes ``Y = (A * a_scale_row) @ (B * b_scale_col).T`` where:
  - A : [M, K] float8_e4m3fn  (per-row scale a_scale [M, 1])
  - B : [N, K] float8_e4m3fn  (per-col scale b_scale [1, N] or scalar)
  - Y : [M, N] bfloat16

The kernel uses TMA loads, warp-specialized producer/consumer
thread-groups, mbarrier-based multi-stage software pipeline, and
``mma.sync.aligned.m16n8k16.row.col.f32.e4m3.e4m3.f32`` for the FP8
tensor core path on sm_120 (Blackwell consumer). Pattern lifted from
the BF16 reference at ``/hy-tmp/sm120_gemm``.

Used by the two-pass W4A8 forward in
:mod:`src.models.ops.nvfp4_linear_w4a8` as an opt-in replacement for
``torch._scaled_mm`` (which routes to cuBLAS). Drop-in API:

    Y = fp8_gemm_scaled(A_fp8, B_fp8, a_scale, b_scale)

Both A and B must be ``torch.float8_e4m3fn`` and contiguous; ``a_scale``
is ``[M, 1]`` FP32, ``b_scale`` is ``[1, N]`` FP32 (or a scalar that
gets broadcast). Returns ``torch.bfloat16`` ``[M, N]`` row-major.

Built per-arch via :file:`scripts/build_fp8_gemm.py`; loaded lazily on
first call (ctypes binding, no JIT). Falls back to
``torch._scaled_mm`` if the .so for the current SM is not built.
"""
from __future__ import annotations

import ctypes
import os
from pathlib import Path
from typing import Optional

import torch


# ---------------------------------------------------------------------------
# Lazy .so loader
# ---------------------------------------------------------------------------
_LIB_DIR = Path(__file__).resolve().parent / "lib"
_loaded_lib: Optional[ctypes.CDLL] = None
_lib_name: Optional[str] = None


def _sm_tag() -> str:
    major, minor = torch.cuda.get_device_capability()
    return f"sm_{major}{minor}"


def _load() -> Optional[ctypes.CDLL]:
    """Bind the FP8 GEMM .so for the current SM arch, or None if missing.

    Naming: ``fp8_gemm_{sm}_{config}.so``. The Python wrapper picks
    one config (currently a single default; real autotune will select
    by shape at first forward).
    """
    global _loaded_lib, _lib_name
    if _loaded_lib is not None:
        return _loaded_lib

    sm = _sm_tag()
    # Config preference: best sweep result at prod FFN shapes first,
    # then the small-M fallback. Empirical ordering (sm_120):
    #   bm128_bn128_bk128_s3_cwg2_wm32_wn64_ds  → 1.04x cuBLAS gate_up
    #   bm64_bn64_bk128_s3_cwg1_wm32_wn32_ds     → small-M fallback
    # (see test/_tmp/probe_fp8_gemm_sweep.py for the sweep; configs
    # with the `_ds` suffix use DIRECT_STORE epilogue — drops the
    # BM*BN*2 Y_out smem so BM=128/BN=128/BK=128 can run NUM_STAGES=3
    # under the 99 KB dynamic-smem cap.)
    preferred = [
        "bm128_bn128_bk128_s3_cwg2_wm32_wn64_ds",
        "bm64_bn64_bk128_s3_cwg1_wm32_wn32_ds",
    ]
    for cfg in preferred:
        so_path = _LIB_DIR / f"fp8_gemm_{sm}_{cfg}.so"
        if so_path.exists():
            _lib_name = so_path.name
            _loaded_lib = ctypes.CDLL(str(so_path), mode=ctypes.RTLD_LOCAL)
            _loaded_lib.gemm_run.restype = None
            _loaded_lib.gemm_run.argtypes = [
                ctypes.c_int, ctypes.c_int, ctypes.c_int,  # M, N, K
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,  # A, B, Y
                ctypes.c_void_p, ctypes.c_void_p,  # a_scale, b_scale
                ctypes.c_void_p,  # workspace (unused)
                ctypes.c_void_p,  # cudaStream_t
            ]
            return _loaded_lib
    return None


def is_available() -> bool:
    """True if a prebuilt .so is loadable for the current SM arch."""
    return _load() is not None


def fp8_gemm_scaled(
    A: torch.Tensor,
    B: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Y = (A * a_scale_row) @ (B * b_scale_col).T → Y [M, N] BF16.

    A : [M, K] float8_e4m3fn  row-major contiguous
    B : [N, K] float8_e4m3fn  row-major contiguous
    a_scale : [M, 1] FP32 contiguous
    b_scale : [1, N] FP32 contiguous (or [N] FP32)
    """
    lib = _load()
    if lib is None:
        raise RuntimeError(
            f"No prebuilt FP8 GEMM .so for {_sm_tag()}. "
            f"Run `python scripts/build_fp8_gemm.py` to build it."
        )

    assert A.dtype == torch.float8_e4m3fn and A.is_cuda
    assert B.dtype == torch.float8_e4m3fn and B.is_cuda
    M, K = A.shape
    N, K2 = B.shape
    assert K == K2, f"K mismatch: A has {K}, B has {K2}"

    if out is None:
        out = torch.empty(M, N, dtype=torch.bfloat16, device=A.device)
    else:
        assert out.shape == (M, N) and out.dtype == torch.bfloat16

    # Flatten scales to match the kernel ABI.
    if a_scale.shape == (M, 1):
        a_scale_flat = a_scale.view(-1).contiguous()
    elif a_scale.shape == (M,):
        a_scale_flat = a_scale.contiguous()
    else:
        raise ValueError(f"a_scale must be [M,1] or [M], got {a_scale.shape}")

    if b_scale.shape == (1, N):
        b_scale_flat = b_scale.view(-1).contiguous()
    elif b_scale.shape == (N,):
        b_scale_flat = b_scale.contiguous()
    else:
        raise ValueError(f"b_scale must be [1,N] or [N], got {b_scale.shape}")

    stream = torch.cuda.current_stream().cuda_stream
    lib.gemm_run(
        M, N, K,
        A.data_ptr(),
        B.data_ptr(),
        out.data_ptr(),
        a_scale_flat.data_ptr(),
        b_scale_flat.data_ptr(),
        0,  # workspace
        stream,
    )
    return out
