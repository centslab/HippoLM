"""R1-min verification: correctness + throughput vs R0 + CUTLASS 87a.

R1-min = R0 + ONLY NUM_STAGES: 2 -> 3 (no other changes).

Per the kda-correctness-sweep rule, we gate "faster" claims on:
  1. Correctness vs FP32 reference (atol=0.15, rtol=0.10 per MmaTile-scale)
  2. isfinite check
  3. Throughput delta vs the cheaper baseline (R0)
"""
from __future__ import annotations

import pytest
import torch

from src.models.ops.cuda.fp8_gemm_hand_r1_min import (
    fp8_gemm_hand_r1min,
    is_available,
)
from src.models.ops.cuda.fp8_gemm_hand import fp8_gemm_hand_r0


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


@pytest.mark.skipif(not is_available(), reason="hand R1-min FP8 GEMM not built")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("M,N,K", [
    (128, 128, 128),         # smallest valid (single MmaTile)
    (1024, 1536, 1536),      # KDA q/k/v/o at prod M
    (4096, 4096, 4096),      # large sweep (32 tiles/dim)
])
def test_blockwise_correctness(M, N, K):
    torch.manual_seed(0)
    A = torch.randn(M, K, device="cuda", dtype=torch.float32).clamp(-10, 10).to(torch.float8_e4m3fn).contiguous()
    B_nk = torch.randn(N, K, device="cuda", dtype=torch.float32).clamp(-10, 10).to(torch.float8_e4m3fn).contiguous()
    a_scale = (torch.rand(M // 128, K // 128, device="cuda", dtype=torch.float32) + 0.5)
    b_scale = (torch.rand(N // 128, K // 128, device="cuda", dtype=torch.float32) + 0.5)

    out = fp8_gemm_hand_r1min(A, B_nk, a_scale, b_scale)
    assert out.shape == (M, N)
    assert out.dtype == torch.bfloat16
    assert torch.isfinite(out).all(), "hand R1-min: non-finite output"

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


@pytest.mark.skipif(not is_available(), reason="hand R1-min FP8 GEMM not built")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("M,N,K", [
    (1024, 1536, 1536),
    (4096, 4096, 4096),
])
def test_benchmark_vs_r0(M, N, K):
    """Compare hand R1-min (NS=3) against R0 (NS=2) at the same shapes."""
    torch.manual_seed(2)
    A = torch.randn(M, K, device="cuda", dtype=torch.float32).clamp(-10, 10).to(torch.float8_e4m3fn).contiguous()
    B_nk = torch.randn(N, K, device="cuda", dtype=torch.float32).clamp(-10, 10).to(torch.float8_e4m3fn).contiguous()
    a_scale = (torch.rand(M // 128, K // 128, device="cuda", dtype=torch.float32) + 0.5)
    b_scale = (torch.rand(N // 128, K // 128, device="cuda", dtype=torch.float32) + 0.5)

    ms_r0 = _bench(lambda: fp8_gemm_hand_r0(A, B_nk, a_scale, b_scale))
    ms_r1m = _bench(lambda: fp8_gemm_hand_r1min(A, B_nk, a_scale, b_scale))

    tf_r0 = _tflops(M, N, K, ms_r0)
    tf_r1m = _tflops(M, N, K, ms_r1m)
    print(f"\nM={M} N={N} K={K}: "
          f"r0={ms_r0:.3f}ms ({tf_r0:.1f}TF) "
          f"r1min={ms_r1m:.3f}ms ({tf_r1m:.1f}TF) "
          f"r1min/r0={tf_r1m/tf_r0:.2f}x")