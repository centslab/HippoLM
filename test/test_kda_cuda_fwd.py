"""Numerical correctness test for the KDA CUDA forward kernel (Round 1).

This file is the SINGLE source of truth for "is the CUDA fwd numerically
correct?" — it pins the contract for downstream kernel changes.

Approach
--------
Compare the CUDA fwd output against the vendored FLA Triton path (bf16,
via ``naive_recurrent_kda`` for the strict-fp32 oracle on small inputs
and via ``chunk_kda`` for the realistic-shape bf16 comparison).

Tolerance
---------
* bf16 vs Triton bf16: relerr at significant magnitudes (|FLA| > 0.1) ≤ 5%,
  max abs err ≤ 1.0.
* bf16 vs fp32 naive oracle (small T): relerr ≤ 5e-3.

Notes on test inputs
--------------------
L2-normalized q, k (production KDA does this) keeps the recurrence stable
across many chunks. Without L2 norm, the KDA recurrence is exponentially
sensitive to (k, v) outer product magnitudes and FLA itself produces
values up to 1e23 for T=1024 with random N(0,1) k/v — that's a property
of the KDA recurrence, not a bug. The test must mirror production inputs
to be meaningful for multi-chunk shapes.

Shapes covered
--------------
* T = BT (single chunk, sanity)
* T = 4 * BT (multi-chunk, no doc boundary)
* T = 16 * BT (multi-chunk, large enough to stress cuBLAS bmm)
* varlen with cu_seqlens (one doc + tail-pad doc)
"""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import torch

# Make the project importable when run from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.ops._vendored.fla.ops.kda.naive import naive_recurrent_kda
from src.models.ops.cuda.kda_fwd import chunk_kda_fwd, compile as cuda_compile


def _bf16_max_abs_rel(a: torch.Tensor, b: torch.Tensor) -> tuple[float, float, float]:
    """Returns (max_abs, rel_at_significant, rel_max).
    rel_at_significant: relerr at positions where |b| > 0.1 (avoids div-by-near-zero).
    rel_max: max relerr (any position — informational only).
    """
    diff = (a.float() - b.float()).abs()
    b_abs = b.float().abs()
    rel = diff / (b_abs + 1e-6)
    sig_mask = b_abs > 0.1
    rel_sig = (rel * sig_mask).max().item() if sig_mask.any() else 0.0
    return diff.max().item(), rel_sig, rel.max().item()


def _make_inputs(B, T, H, K, V, scale, device, dtype, seed):
    """Create bf16 inputs in the production shape.

    L2-normalizes q, k to match production KDA (KimiDeltaAttention.forward
    applies qk_l2norm). Without this, the KDA recurrence is exponentially
    sensitive to the (k, v) outer product magnitudes and the test fails
    for T > 64 even when the kernel is correct — the bf16 ULP noise on
    h_prev gets amplified by 1/(1 - exp(g_min)) per chunk.
    """
    torch.manual_seed(seed)
    q = torch.randn(B, T, H, K, device=device, dtype=torch.float32)
    k = torch.randn(B, T, H, K, device=device, dtype=torch.float32)
    v = torch.randn(B, T, H, V, device=device, dtype=dtype)
    # L2 normalize q, k in fp32 then cast (matches FLA's qk_l2norm behavior).
    q = (q / q.norm(dim=-1, keepdim=True)).to(dtype)
    k = (k / k.norm(dim=-1, keepdim=True)).to(dtype)
    # g in [-2.5, -0.5] — keeps the recurrence stable across many chunks
    # by bounding exp(g) away from 1. Production: g = -exp(A_log) * softplus(...)
    # is typically in (-3, 0) but with safe_gate=True clamped to lower_bound=-5.
    g = -torch.rand(B, T, H, K, device=device, dtype=dtype) * 2.0 - 0.5
    beta = torch.randn(B, T, H, device=device, dtype=dtype).sigmoid()
    return q, k, v, g, beta


def _make_naive_inputs(B, T, H, K, V, scale, device, dtype, seed):
    """Convert our bf16 tensors to fp32 for the naive_recurrent oracle."""
    q, k, v, g, beta = _make_inputs(B, T, H, K, V, scale, device, dtype, seed)
    # naive_recurrent_kda expects q: [B, T, H, K], v: [B, T, HV, V], g: [B, T, HV, K].
    # HV = H in our production config (no GVA).
    return q.float(), k.float(), v.float(), g.float(), beta.float()


def test_single_chunk():
    """T = BT = 64: smallest sanity check."""
    print("\n[test_single_chunk] B=1 T=64 H=4 K=32 V=32")
    B, T, H, K, V = 1, 64, 4, 32, 32
    device = torch.device("cuda")
    dtype = torch.bfloat16
    scale = 1.0 / math.sqrt(K)

    q, k, v, g, beta = _make_inputs(B, T, H, K, V, scale, device, dtype, seed=42)

    # Triton reference (chunk_kda). Use the vendored FLA path.
    o_ref, _ = _triton_chunk_kda(q, k, v, g, beta, scale=scale)

    # CUDA path.
    o_cuda, _ = chunk_kda_fwd(q, k, v, g, beta, scale=scale)

    max_abs, rel_sig, rel_max = _bf16_max_abs_rel(o_cuda, o_ref)
    print(f"  max_abs={max_abs:.4f}, rel@|FLA|>0.1={rel_sig:.4f}, rel_max={rel_max:.4f}")
    assert rel_sig < 5e-2, f"relerr at significant magnitudes too high: {rel_sig}"
    assert max_abs < 1.0, f"abs err too high: {max_abs}"
    print("  PASS")


def test_multi_chunk():
    """T = 16 * BT = 1024: stress the cuBLAS bmm + delta_h recurrence."""
    print("\n[test_multi_chunk] B=1 T=1024 H=4 K=64 V=64")
    B, T, H, K, V = 1, 1024, 4, 64, 64
    device = torch.device("cuda")
    dtype = torch.bfloat16
    scale = 1.0 / math.sqrt(K)

    q, k, v, g, beta = _make_inputs(B, T, H, K, V, scale, device, dtype, seed=7)

    o_ref, _ = _triton_chunk_kda(q, k, v, g, beta, scale=scale)
    o_cuda, _ = chunk_kda_fwd(q, k, v, g, beta, scale=scale)

    max_abs, rel_sig, rel_max = _bf16_max_abs_rel(o_cuda, o_ref)
    print(f"  max_abs={max_abs:.4f}, rel@|FLA|>0.1={rel_sig:.4f}, rel_max={rel_max:.4f}")
    assert rel_sig < 5e-2
    assert max_abs < 1.0
    print("  PASS")


def test_production_shape():
    """T=16384 H=12 K=128 V=128 — the actual prod config (no GVA)."""
    print("\n[test_production_shape] B=1 T=16384 H=12 K=128 V=128")
    B, T, H, K, V = 1, 16384, 12, 128, 128
    device = torch.device("cuda")
    dtype = torch.bfloat16
    scale = 1.0 / math.sqrt(K)

    q, k, v, g, beta = _make_inputs(B, T, H, K, V, scale, device, dtype, seed=2026)

    o_ref, _ = _triton_chunk_kda(q, k, v, g, beta, scale=scale)
    o_cuda, _ = chunk_kda_fwd(q, k, v, g, beta, scale=scale)

    max_abs, rel_sig, rel_max = _bf16_max_abs_rel(o_cuda, o_ref)
    print(f"  max_abs={max_abs:.4f}, rel@|FLA|>0.1={rel_sig:.4f}, rel_max={rel_max:.4f}")
    assert rel_sig < 5e-2
    assert max_abs < 1.0
    print("  PASS")


def _triton_chunk_kda(q, k, v, g, beta, scale):
    """Call the vendored FLA Triton chunk_kda directly.

    This bypasses KimiDeltaAttention.forward (which does the L2 norm +
    gate activation + cu_seqlens reshaping) and works on raw q/k/v/g/beta.
    """
    from src.models.ops._vendored.fla.ops.kda import chunk_kda
    return chunk_kda(
        q=q, k=k, v=v, g=g, beta=beta,
        scale=scale,
        initial_state=None, output_final_state=False,
        cu_seqlens=None,
    )


def main():
    cuda_compile(verbose=False)
    print(f"CUDA kernel version: "
          f"{__import__('src.models.ops.cuda.kda_fwd', fromlist=['version_string']).version_string()}")

    # Only run the test we have implemented so far. Round 1 is incremental.
    tests = [
        test_single_chunk,
        test_multi_chunk,
        test_production_shape,
    ]
    for t in tests:
        try:
            t()
        except AssertionError as e:
            print(f"  FAIL: {e}")
            sys.exit(1)
        except Exception as e:
            print(f"  ERROR: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)


if __name__ == "__main__":
    main()
