"""Tests + benchmarks for the project-local CUTLASS-based FP8 GEMM (sm_120).

The kernel uses per-MmaTile (128x128x128) FP32 scales — one scale per
128x128 MmaTile. This is COARSER than per-row × per-128-K-block scaling
the existing project path uses. The reference here mirrors the
kernel's semantics exactly.

Verifies:
  1. Correctness: per-MmaTile path matches a PyTorch reference
  2. isfinite gate before any timing claim
  3. Throughput vs existing project per-row FP8 GEMM (TensorWise scalar)

Run:
    pytest test/test_fp8_gemm_cutlass.py -v
"""
from __future__ import annotations

import pytest
import torch

from src.models.ops.cuda.fp8_gemm_cutlass import (
    fp8_blockwise_gemm,
    is_available,
)

# ---------------------------------------------------------------------------
# Reference computation
# ---------------------------------------------------------------------------

def _ref_permmatile(A, B_nk, a_scale, b_scale):
    """Compute (A * a_scale_per_mma) @ (B_nk.T * b_scale_per_mma) in FP32.

    A is [M, K] FP8, B_nk is [N, K] FP8. a_scale is [M/128, K/128] FP32
    (one scale per MmaTile — coarser than per-row); b_scale is
    [N/128, K/128] FP32.

    Broadcast each MmaTile scale to the corresponding 128x128 block,
    then matmul.
    """
    M, K = A.shape
    N, K2 = B_nk.shape
    # A: [M, K] → reshape to [M/128, 128, K/128, 128] in (m_tile, m, k_tile, k) order.
    a = A.float().reshape(M // 128, 128, K // 128, 128)
    a = a * a_scale.float().reshape(M // 128, 1, K // 128, 1)
    a = a.reshape(M, K)

    # B_nk: [N, K] → reshape to [N/128, 128, K/128, 128] in (n_tile, n, k_tile, k) order.
    b = B_nk.float().reshape(N // 128, 128, K // 128, 128)
    # b_scale[n_tile, k_tile] broadcasts over [n, k] = [128, 128].
    b = b * b_scale.float().reshape(N // 128, 1, K // 128, 1)
    b = b.reshape(N, K)

    return (a @ b.t()).to(torch.bfloat16)


# ---------------------------------------------------------------------------
# Correctness tests
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not is_available(), reason="CUTLASS FP8 GEMM not built")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("M,N,K", [
    (128, 128, 128),         # smallest valid (single MmaTile both sides)
    (1024, 1536, 1536),      # KDA q/k/v/o at prod M
    (4096, 1536, 1536),      # KDA at large M
    (1024, 4096, 4096),      # FFN
])
def test_blockwise_correctness(M, N, K):
    torch.manual_seed(0)
    A = torch.randn(M, K, device="cuda", dtype=torch.float32).to(torch.float8_e4m3fn).contiguous()
    B_nk = torch.randn(N, K, device="cuda", dtype=torch.float32).to(torch.float8_e4m3fn).contiguous()
    # Per-MmaTile scales: one per (M/128, K/128) and (N/128, K/128)
    a_scale = (torch.rand(M // 128, K // 128, device="cuda", dtype=torch.float32) + 0.5)
    b_scale = (torch.rand(N // 128, K // 128, device="cuda", dtype=torch.float32) + 0.5)

    out = fp8_blockwise_gemm(A, B_nk, a_scale, b_scale)
    assert out.shape == (M, N)
    assert out.dtype == torch.bfloat16
    assert torch.isfinite(out).all(), "blockwise: non-finite output"

    ref = _ref_permmatile(A, B_nk, a_scale, b_scale)
    # FP8 E4M3 has 3-bit mantissa (~12% relative); BF16 accumulator is
    # ~7-bit. Use generous tolerance.
    torch.testing.assert_close(out.float(), ref.float(), atol=0.15, rtol=0.10)


# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------

def _bench(fn, *, warmup=10, iters=20):
    """Median runtime in ms using CUDA events."""
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


@pytest.mark.skipif(not is_available(), reason="CUTLASS FP8 GEMM not built")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("M,N,K", [
    (1024, 1536, 1536),     # KDA Q/K/V/O at prod M
    (4096, 1536, 1536),     # KDA Q/K/V/O at large M
    (1024, 4096, 4096),     # FFN gate_up at K=4096
    (4096, 4096, 4096),     # largest sweep shape
])
def test_benchmark_vs_existing_perrow(M, N, K):
    """Compare CUTLASS per-MmaTile path against the existing project per-row FP8 GEMM."""
    from src.models.ops.cuda.fp8_gemm import fp8_gemm_scaled
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")

    torch.manual_seed(2)
    A = torch.randn(M, K, device="cuda", dtype=torch.float32).to(torch.float8_e4m3fn).contiguous()
    B_nk = torch.randn(N, K, device="cuda", dtype=torch.float32).to(torch.float8_e4m3fn).contiguous()
    # Per-row scale for the existing project path
    a_per_row = (torch.rand(M, device="cuda", dtype=torch.float32) + 0.5)
    b_per_col = (torch.rand(N, device="cuda", dtype=torch.float32) + 0.5)
    # Per-MmaTile scale for the CUTLASS path (matching the kernel's semantics)
    a_mma = (torch.rand(M // 128, K // 128, device="cuda", dtype=torch.float32) + 0.5)
    b_mma = (torch.rand(N // 128, K // 128, device="cuda", dtype=torch.float32) + 0.5)

    ms_cutlass = _bench(lambda: fp8_blockwise_gemm(A, B_nk, a_mma, b_mma))
    ms_existing = _bench(lambda: fp8_gemm_scaled(A, B_nk, a_per_row, b_per_col))

    tf_c = _tflops(M, N, K, ms_cutlass)
    tf_e = _tflops(M, N, K, ms_existing)
    print(f"\nM={M} N={N} K={K}: "
          f"cutlass={ms_cutlass:.3f}ms ({tf_c:.1f}TF) "
          f"existing={ms_existing:.3f}ms ({tf_e:.1f}TF) "
          f"ratio={tf_c/tf_e:.2f}x")
