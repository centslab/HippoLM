"""Tests + benchmarks for the hand-written Round 0 FP8 GEMM kernel.

Verifies:
  1. Correctness: per-MmaTile path matches a PyTorch reference
  2. isfinite gate before any timing claim
  3. Throughput vs CUTLASS 87a at the same shapes

Run:
    pytest test/test_fp8_gemm_hand.py -v
"""
from __future__ import annotations

import pytest
import torch

from src.models.ops.cuda.fp8_gemm_hand import (
    fp8_gemm_hand_r0,
    is_available,
)


def _ref_permmatile(A, B_nk, a_scale, b_scale):
    """Compute (A * a_scale_per_mma) @ (B_nk.T * b_scale_per_mma) in FP32.

    A is [M, K] FP8, B_nk is [N, K] FP8. a_scale is [M/128, K/128] FP32
    (one scale per MmaTile); b_scale is [N/128, K/128] FP32.
    """
    M, K = A.shape
    N, K2 = B_nk.shape
    a = A.float().reshape(M // 128, 128, K // 128, 128)
    a = a * a_scale.float().reshape(M // 128, 1, K // 128, 1)
    a = a.reshape(M, K)
    b = B_nk.float().reshape(N // 128, 128, K // 128, 128)
    b = b * b_scale.float().reshape(N // 128, 1, K // 128, 1)
    b = b.reshape(N, K)
    return (a @ b.t()).to(torch.bfloat16)


@pytest.mark.skipif(not is_available(), reason="hand R0 FP8 GEMM not built")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("M,N,K", [
    (128, 128, 128),         # smallest valid
    (1024, 1536, 1536),      # KDA q/k/v/o at prod M
    (4096, 4096, 4096),      # large sweep
])
def test_blockwise_correctness(M, N, K):
    torch.manual_seed(0)
    A = torch.randn(M, K, device="cuda", dtype=torch.float32).clamp(-10, 10).to(torch.float8_e4m3fn).contiguous()
    B_nk = torch.randn(N, K, device="cuda", dtype=torch.float32).clamp(-10, 10).to(torch.float8_e4m3fn).contiguous()
    a_scale = (torch.rand(M // 128, K // 128, device="cuda", dtype=torch.float32) + 0.5)
    b_scale = (torch.rand(N // 128, K // 128, device="cuda", dtype=torch.float32) + 0.5)

    out = fp8_gemm_hand_r0(A, B_nk, a_scale, b_scale)
    assert out.shape == (M, N)
    assert out.dtype == torch.bfloat16
    assert torch.isfinite(out).all(), "hand R0: non-finite output"

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


@pytest.mark.skipif(not is_available(), reason="hand R0 FP8 GEMM not built")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("M,N,K", [
    (1024, 1536, 1536),
    (4096, 1536, 1536),
    (4096, 4096, 4096),
])
def test_benchmark_vs_cutlass(M, N, K):
    """Compare hand R0 against CUTLASS 87a at the same shapes."""
    from src.models.ops.cuda.fp8_gemm_cutlass import fp8_blockwise_gemm
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")

    torch.manual_seed(2)
    A = torch.randn(M, K, device="cuda", dtype=torch.float32).clamp(-10, 10).to(torch.float8_e4m3fn).contiguous()
    B_nk = torch.randn(N, K, device="cuda", dtype=torch.float32).clamp(-10, 10).to(torch.float8_e4m3fn).contiguous()
    a_scale = (torch.rand(M // 128, K // 128, device="cuda", dtype=torch.float32) + 0.5)
    b_scale = (torch.rand(N // 128, K // 128, device="cuda", dtype=torch.float32) + 0.5)

    ms_hand = _bench(lambda: fp8_gemm_hand_r0(A, B_nk, a_scale, b_scale))
    ms_cutlass = _bench(lambda: fp8_blockwise_gemm(A, B_nk, a_scale, b_scale))

    tf_h = _tflops(M, N, K, ms_hand)
    tf_c = _tflops(M, N, K, ms_cutlass)
    print(f"\nM={M} N={N} K={K}: "
          f"hand_r0={ms_hand:.3f}ms ({tf_h:.1f}TF) "
          f"cutlass={ms_cutlass:.3f}ms ({tf_c:.1f}TF) "
          f"ratio={tf_h/tf_c:.2f}x")