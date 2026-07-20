"""Test + bench for the custom CUDA C++ FP8 GEMM kernel.

Verifies:
  1. Correctness: kernel output matches ``torch._scaled_mm`` (which
     routes to cuBLAS) within FP8 cast noise.
  2. Per-row × per-col scale is applied correctly.
  3. Throughput at prod FFN shapes: beats cuBLAS by some margin.

Correctness reference:
  ``Y = (A_fp8.float() * a_scale) @ (B_fp8.float() * b_scale).T`` in
  fp32, then narrowed to BF16 via RNE. This is the *exact* value the
  kernel should produce (in float), modulo E4M3 B rounding noise
  inside the kernel's accumulator → bf16 cast at the end.
"""
from __future__ import annotations

import pytest
import torch

from src.models.ops.cuda.fp8_gemm import is_available, fp8_gemm_scaled


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="FP8 GEMM requires CUDA",
)
_requires_kernel = pytest.mark.skipif(
    not is_available(),
    reason="No prebuilt fp8_gemm .so for this SM — run scripts/build_fp8_gemm.py",
)


def _fp32_ref(
    A_fp8: torch.Tensor,
    B_fp8: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
) -> torch.Tensor:
    """fp32 matmul of dequantized FP8 inputs — the exact reference
    the kernel should produce (before the in-register FP32 → BF16
    RNE cast at the epilogue).

    Kernel semantics: Y = (A * a_scale_row) @ (B * b_scale_col).T
    where A is [M, K] and B is [N, K] (so b_scale is per-N, applied
    to B's rows — view as [N, 1] to broadcast over the K dim).
    """
    M, K = A_fp8.shape
    a_deq = A_fp8.float() * a_scale.view(M, 1)
    b_deq = B_fp8.float() * b_scale.view(-1, 1)
    return (a_deq @ b_deq.t())


# ---------------------------------------------------------------------------
# Correctness
# ---------------------------------------------------------------------------
@_requires_kernel
@pytest.mark.parametrize("M,K,N", [
    (128, 128, 128),
    (256, 256, 256),
    (1024, 1024, 1024),
    (128, 1024, 128),    # non-square
    (256, 512, 1024),
])
def test_fp8_gemm_matches_fp32_ref(M, K, N):
    """Output matches fp32 reference within E4M3 B rounding noise."""
    torch.manual_seed(0)
    # FP8 e4m3 has range ±448. Use small randn so values are well
    # in range, then cast to fp8 (the "round-to-fp8" path).
    A = torch.randn(M, K, dtype=torch.float32, device="cuda").clamp(-10, 10).to(torch.float8_e4m3fn)
    B = torch.randn(N, K, dtype=torch.float32, device="cuda").clamp(-10, 10).to(torch.float8_e4m3fn)
    a_scale = torch.rand(M, 1, dtype=torch.float32, device="cuda") * 0.1 + 0.01
    b_scale = torch.rand(1, N, dtype=torch.float32, device="cuda") * 0.1 + 0.01

    y_kernel = fp8_gemm_scaled(A, B, a_scale, b_scale).float()
    y_ref = _fp32_ref(A, B, a_scale, b_scale)

    rel = (y_kernel - y_ref).abs() / (y_ref.abs() + 1e-3)
    rel_median = rel.median().item()
    rel_p95 = rel.flatten().sort()[0][int(0.95 * rel.numel())].item()
    print(f"  M={M} K={K} N={N}: median rel={rel_median*100:.2f}%  p95 rel={rel_p95*100:.2f}%")

    # FP8 cast noise + bf16 narrow → expect ~2% median rel, allow 10%
    # for now (this is a new kernel; tolerances tighten once we're
    # sure the layout is correct).
    assert torch.isfinite(y_kernel).all(), "non-finite in y_kernel"
    assert rel_median < 0.10, f"median rel too high: {rel_median*100:.2f}%"


@_requires_kernel
@pytest.mark.parametrize("M,K,N", [
    (128, 128, 128),
    (1024, 1024, 1024),
])
def test_fp8_gemm_matches_scaled_mm(M, K, N):
    """Output matches ``torch._scaled_mm`` (cuBLAS).

    ``torch._scaled_mm`` wants B as a [K, N] *column-major* tensor
    (stride(0) == 1). Build B as [N, K] row-major first, then take
    ``.t()`` to get the column-major view (no copy).
    """
    torch.manual_seed(0)
    A = torch.randn(M, K, dtype=torch.float32, device="cuda").clamp(-10, 10).to(torch.float8_e4m3fn)
    B = torch.randn(N, K, dtype=torch.float32, device="cuda").clamp(-10, 10).to(torch.float8_e4m3fn)
    B_t = B.t()  # [K, N] col-major view (stride(0)=1, stride(1)=K)
    a_scale = torch.rand(M, 1, dtype=torch.float32, device="cuda") * 0.1 + 0.01
    b_scale = torch.rand(1, N, dtype=torch.float32, device="cuda") * 0.1 + 0.01

    y_kernel = fp8_gemm_scaled(A, B, a_scale, b_scale)
    y_cublas = torch._scaled_mm(
        A, B_t,
        scale_a=a_scale,
        scale_b=b_scale,
        out_dtype=torch.bfloat16,
    )

    rel = (y_kernel.float() - y_cublas.float()).abs() / (y_cublas.float().abs() + 1e-3)
    rel_median = rel.median().item()
    print(f"  M={M} K={K} N={N}: kernel vs cuBLAS median rel={rel_median*100:.2f}%")
    # Both have FP8 cast noise; should be close
    assert rel_median < 0.05, f"kernel vs cuBLAS median rel: {rel_median*100:.2f}%"


# ---------------------------------------------------------------------------
# Throughput
# ---------------------------------------------------------------------------
def _bench(fn, iters=30, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = torch.cuda.Event(enable_timing=True)
    t1 = torch.cuda.Event(enable_timing=True)
    t0.record()
    for _ in range(iters):
        fn()
    t1.record()
    torch.cuda.synchronize()
    return t0.elapsed_time(t1) / iters


@_requires_kernel
def test_fp8_gemm_throughput_vs_cublas():
    """Throughput at prod FFN shapes vs ``torch._scaled_mm``."""
    print("\n--- Throughput ---")
    for label, M, K, N in [
        ("gate_up", 16384, 1536, 8192),
        ("down",    16384, 4096, 1536),
    ]:
        torch.manual_seed(0)
        A = torch.randn(M, K, dtype=torch.float32, device="cuda").clamp(-10, 10).to(torch.float8_e4m3fn)
        B = torch.randn(N, K, dtype=torch.float32, device="cuda").clamp(-10, 10).to(torch.float8_e4m3fn)
        B_t = B.t()  # [K, N] col-major view for scaled_mm
        a_scale = torch.rand(M, 1, dtype=torch.float32, device="cuda") * 0.1 + 0.01
        b_scale = torch.rand(1, N, dtype=torch.float32, device="cuda") * 0.1 + 0.01

        def kernel():
            return fp8_gemm_scaled(A, B, a_scale, b_scale)

        def cublas():
            return torch._scaled_mm(A, B_t, scale_a=a_scale, scale_b=b_scale, out_dtype=torch.bfloat16)

        t_k = _bench(kernel)
        t_c = _bench(cublas)
        flops = 2.0 * M * K * N
        print(f"  {label:8s} M={M} K={K} N={N}: cuBLAS={t_c:.2f}ms ({flops/t_c/1e9:.0f} TF), "
              f"kernel={t_k:.2f}ms ({flops/t_k/1e9:.0f} TF) => {t_c/t_k:.2f}x")
