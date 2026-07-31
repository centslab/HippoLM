"""FP8 E4M3 GEMM — mainline hybrid: 1×128 BF16 A scale + TensorWise FP8 B.

Confirmed mainline (2026-07-31): per-row 1×128 BF16 activation scale ×
TensorWise scalar B scale. This is the W8A8 GEMM used by the two-pass
W4A8 path (`w4a8_gemm.py`) and the FSFP8-style quantization scheme:

  A : [M, K] float8_e4m3fn, per-row 1×128 BF16 scale
             (a_block_scale [M, K/128] BF16)
  B : [N, K] float8_e4m3fn, TensorWise scale (1 scalar per K-block)
             (b_block_scale [K/128] BF16)
  Y : [M, N] bfloat16

The kernel is the same source file as `fp8_blockwise_gemm.py` — the
hybrid variant instantiates FP8GemmMMA with A_SCALE_1D_BLOCK=true,
BLOCK_SCALE_K=128 (chain matches R10), B_SCALE_TENSORWISE=true, and the
R10 strategy (BLOCK_ACCUM + BLOCK_TILE_SCALE + DIRECT_STORE +
bm128/128/128/s3/cwg2/wm32/wn64).

Two accumulator flavors, separate .so files:
  * hybrid_a1x128_twb       — ACC_RAW_F16 (F16 in-block acc, perf pick)
  * hybrid_a1x128_twb_f32   — F32 acc (overflow-safe W8A8/W4A8 default)

Build:
  python scripts/build_fp8_blockwise_gemm.py --variant hybrid_a1x128_twb
  python scripts/build_fp8_blockwise_gemm.py --variant hybrid_a1x128_twb_f32

See MEMORY.md "W8A8 F16-acc overflow — strategy" (2026-07-31) for why
the F32-acc variant is the production default for W8A8.
"""
from __future__ import annotations

import ctypes
from pathlib import Path
from typing import Literal, Optional

import torch


_LIB_DIR = Path(__file__).resolve().parent / "lib"
_sm_tag_cache: Optional[str] = None


def _sm_tag() -> str:
    global _sm_tag_cache
    if _sm_tag_cache is None:
        major, minor = torch.cuda.get_device_capability()
        _sm_tag_cache = f"sm_{major}{minor}"
    return _sm_tag_cache


def _so_path(acc: Literal["f16", "f32"]) -> Path:
    tag = "_hybrid_a1x128_twb" if acc == "f16" else "_hybrid_a1x128_twb_f32"
    return _LIB_DIR / (
        f"fp8_blockwise_gemm_{_sm_tag()}_"
        f"bm128_bn128_bk128_s3_cwg2_wm32_wn64_ds_blockwise{tag}_bf16.so"
    )


_LIBS: dict[Literal["f16", "f32"], Optional[ctypes.CDLL]] = {}


def _load(acc: Literal["f16", "f32"]) -> ctypes.CDLL:
    if _LIBS.get(acc) is not None:
        return _LIBS[acc]
    so = _so_path(acc)
    if not so.exists():
        variant = "hybrid_a1x128_twb" if acc == "f16" else "hybrid_a1x128_twb_f32"
        raise RuntimeError(
            f"No hybrid_a1x128_twb .so for {_sm_tag()} at {so}. Run "
            f"`python scripts/build_fp8_blockwise_gemm.py --variant {variant}`."
        )
    lib = ctypes.CDLL(str(so), mode=ctypes.RTLD_LOCAL)
    sym = "hybrid_a1x128_twb_bf16_gemm_run" if acc == "f16" else "hybrid_a1x128_twb_f32_bf16_gemm_run"
    fn = getattr(lib, sym)
    fn.restype = None
    fn.argtypes = [
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ]
    _LIBS[acc] = lib
    return lib


def is_available() -> bool:
    if not torch.cuda.is_available():
        return False
    return _so_path("f32").exists()


def fp8_hybrid_gemm(
    A: torch.Tensor,
    B: torch.Tensor,
    a_block_scale: torch.Tensor,
    b_block_scale: torch.Tensor,
    *,
    acc: Literal["f16", "f32"] = "f32",
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Y = (A * a_block_scale_row) @ (B * b_block_scale_tw).T → [M, N] BF16.

    A : [M, K] float8_e4m3fn, row-major contiguous
    B : [N, K] float8_e4m3fn, row-major contiguous
    a_block_scale : per-row 1×128 scale, [M, K/128] BF16 contiguous
    b_block_scale : TensorWise per-K-block scale, [K/128] BF16 contiguous
    acc : "f16" (perf, R10 F16 in-block acc) or "f32" (overflow-safe default)
    """
    assert A.dtype == torch.float8_e4m3fn and A.is_cuda and A.is_contiguous()
    assert B.dtype == torch.float8_e4m3fn and B.is_cuda and B.is_contiguous()
    M, K = A.shape
    N, K2 = B.shape
    assert K == K2, f"K mismatch: A has {K}, B has {K2}"
    assert K % 128 == 0, "K must be multiple of 128 (BLOCK_SCALE_K)"
    assert M % 128 == 0, "M must be multiple of 128 (BM)"
    assert N % 128 == 0, "N must be multiple of 128 (BN)"
    assert a_block_scale.dtype == torch.bfloat16
    assert a_block_scale.shape == (M, K // 128)
    assert a_block_scale.is_cuda and a_block_scale.is_contiguous()
    assert b_block_scale.dtype == torch.bfloat16
    assert b_block_scale.shape == (K // 128,)
    assert b_block_scale.is_cuda and b_block_scale.is_contiguous()

    if out is None:
        out = torch.empty(M, N, dtype=torch.bfloat16, device=A.device)
    else:
        assert out.shape == (M, N) and out.dtype == torch.bfloat16 and out.is_contiguous()

    fn = getattr(_load(acc), (
        "hybrid_a1x128_twb_bf16_gemm_run" if acc == "f16"
        else "hybrid_a1x128_twb_f32_bf16_gemm_run"
    ))
    stream = torch.cuda.current_stream().cuda_stream
    fn(
        M, N, K,
        A.data_ptr(), B.data_ptr(), out.data_ptr(),
        a_block_scale.data_ptr(), b_block_scale.data_ptr(),
        0, stream,
    )
    return out
