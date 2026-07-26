"""Correctness and benchmark coverage for the FP8 1x32 block-scale GEMM."""
from __future__ import annotations

import pytest
import torch

from src.models.ops.cuda.fp8_blockwise_gemm import fp8_blockwise_gemm, is_available


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not is_available(),
    reason="blockwise FP8 CUDA kernel is unavailable",
)


def _reference(A, B, a_scale, b_scale):
    M, K = A.shape
    N = B.shape[0]
    a = A.float().reshape(M, K // 32, 32) * a_scale.float().unsqueeze(-1)
    b = B.float().reshape(N, K // 32, 32) * b_scale.float().unsqueeze(-1)
    return torch.einsum("mkd,nkd->mn", a, b)


@pytest.mark.parametrize("M,K,N", [(64, 128, 128), (128, 1536, 128), (256, 1536, 256)])
def test_blockwise_gemm_matches_reference(M, K, N):
    torch.manual_seed(0)
    A = torch.randn(M, K, device="cuda").clamp(-10, 10).to(torch.float8_e4m3fn).contiguous()
    B = torch.randn(N, K, device="cuda").clamp(-10, 10).to(torch.float8_e4m3fn).contiguous()
    a_scale = (torch.rand(M, K // 32, device="cuda") * 0.1 + 0.01).bfloat16().contiguous()
    b_scale = (torch.rand(N, K // 32, device="cuda") * 0.1 + 0.01).bfloat16().contiguous()

    out = fp8_blockwise_gemm(A, B, a_scale, b_scale).float()
    ref = _reference(A, B, a_scale, b_scale)
    assert torch.isfinite(out).all()
    assert torch.isfinite(ref).all()
    torch.testing.assert_close(out, ref, atol=0.15, rtol=0.03)


def test_blockwise_gemm_differs_from_rowwise_reference():
    M, K, N = 64, 128, 128
    torch.manual_seed(1)
    A = torch.randn(M, K, device="cuda").to(torch.float8_e4m3fn).contiguous()
    B = torch.randn(N, K, device="cuda").to(torch.float8_e4m3fn).contiguous()
    a_scale = torch.ones(M, K // 32, device="cuda", dtype=torch.bfloat16).contiguous()
    b_scale = torch.ones(N, K // 32, device="cuda", dtype=torch.bfloat16).contiguous()
    out = fp8_blockwise_gemm(A, B, a_scale, b_scale)
    ref = _reference(A, B, a_scale, b_scale).bfloat16()
    assert torch.isfinite(out).all()
    assert (out != 0).any()
    torch.testing.assert_close(out, ref, atol=0.15, rtol=0.03)
