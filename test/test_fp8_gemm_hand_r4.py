"""Tests + benchmarks for the hand-written Round 4 FP8 GEMM kernel.

Round 4 = Round 2 + multi-lane TMA:
  - Lane 0 issues TMA_A, lane 1 issues TMA_B in parallel within the
    same producer warp
  - Both lanes share the empty_barrier wait (SIMT lockstep)
  - Same 8 consumer warps (2x4 grid, 64x32 each), NUM_STAGES=2

Verifies:
  1. Correctness vs FP32 per-MmaTile reference
  2. isfinite gate before any timing claim
  3. Throughput vs R2 (single-lane TMA)

Run:
    pytest test/test_fp8_gemm_hand_r4.py -v
"""
from __future__ import annotations

import pytest
import torch

from src.models.ops.cuda.fp8_gemm_hand_r4 import (
    fp8_gemm_hand_r4,
    is_available,
)
from src.models.ops.cuda.fp8_gemm_hand_r2 import fp8_gemm_hand_r2


def _ref_permmatile(A, B_nk, a_scale, b_scale):
    """FP32 reference using per-MmaTile (1x1) scales.

    Scale layout: CUTLASS MN-major K-major in memory; for the per-tile
    ref we broadcast over [m_tile, 128 rows] and [k_tile, 128 cols].
    """
    M, K = A.shape
    N, K2 = B_nk.shape
    a = A.float() * a_scale.float().repeat_interleave(128, dim=0).repeat_interleave(128, dim=1)
    b = B_nk.float() * b_scale.float().repeat_interleave(128, dim=0).repeat_interleave(128, dim=1)
    return (a @ b.t()).to(torch.bfloat16)


@pytest.mark.skipif(not is_available(), reason="hand R4 FP8 GEMM not built")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("M,N,K", [
    (128, 128, 128),         # smallest valid (single MmaTile)
    (1024, 1536, 1536),      # KDA q/k/v/o at prod M
    (4096, 4096, 4096),      # large sweep (32 tiles/dim)
])
def test_r4_correctness(M, N, K):
    torch.manual_seed(0)
    A = torch.randn(M, K, device="cuda", dtype=torch.float32).clamp(-10, 10).to(torch.float8_e4m3fn).contiguous()
    B_nk = torch.randn(N, K, device="cuda", dtype=torch.float32).clamp(-10, 10).to(torch.float8_e4m3fn).contiguous()
    a_scale = (torch.rand(M // 128, K // 128, device="cuda", dtype=torch.float32) + 0.5)
    b_scale = (torch.rand(N // 128, K // 128, device="cuda", dtype=torch.float32) + 0.5)

    out = fp8_gemm_hand_r4(A, B_nk, a_scale, b_scale)
    assert out.shape == (M, N)
    assert out.dtype == torch.bfloat16
    assert torch.isfinite(out).all(), "hand R4: non-finite output"

    ref = _ref_permmatile(A, B_nk, a_scale, b_scale)
    torch.testing.assert_close(out.float(), ref.float(), atol=0.15, rtol=0.10)


def _bench(fn, *, warmup=10, iters=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    times.sort()
    return times[len(times) // 2]


def _tflops(M, N, K, ms):
    return 2.0 * M * N * K / (ms * 1e-3) / 1e12


@pytest.mark.skipif(not is_available(), reason="hand R4 FP8 GEMM not built")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("M,N,K", [
    (1024, 1536, 1536),
    (4096, 1536, 1536),
    (4096, 4096, 4096),
])
def test_benchmark_vs_r2(M, N, K):
    """Compare hand R4 (multi-lane TMA) against R2 (single-lane TMA)."""
    torch.manual_seed(2)
    A = torch.randn(M, K, device="cuda", dtype=torch.float32).clamp(-10, 10).to(torch.float8_e4m3fn).contiguous()
    B_nk = torch.randn(N, K, device="cuda", dtype=torch.float32).clamp(-10, 10).to(torch.float8_e4m3fn).contiguous()
    a_scale = (torch.rand(M // 128, K // 128, device="cuda", dtype=torch.float32) + 0.5)
    b_scale = (torch.rand(N // 128, K // 128, device="cuda", dtype=torch.float32) + 0.5)

    ms_r2 = _bench(lambda: fp8_gemm_hand_r2(A, B_nk, a_scale, b_scale))
    ms_r4 = _bench(lambda: fp8_gemm_hand_r4(A, B_nk, a_scale, b_scale))

    tf_r2 = _tflops(M, N, K, ms_r2)
    tf_r4 = _tflops(M, N, K, ms_r4)
    print(f"\nM={M} N={N} K={K}: "
          f"r2={ms_r2:.3f}ms ({tf_r2:.1f}TF) "
          f"r4={ms_r4:.3f}ms ({tf_r4:.1f}TF) "
          f"r4/r2={tf_r4/tf_r2:.2f}x")