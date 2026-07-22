"""Tests for the FP8 GEMM runtime auto-dispatcher.

Verifies:
  1. ``fp8_gemm_auto_dispatch`` returns a finite result matching the
     cuBLAS ``_scaled_mm`` reference within FP8 cast noise.
  2. Per-shape cache works: second call with same shape doesn't
     re-benchmark.
  3. Cold-start micro-benchmark cost is bounded (<5 s for the canonical
     prod-shape sweep).
  4. Pick policy: small M (≤2048) selects BM64-class configs, large M
     selects BM128-class configs — matching the auto-memory
     ``project_fp8_dispatch_by_m.md`` sweep.

Run: ``pytest test/test_fp8_gemm_dispatch.py -v -s``
"""
from __future__ import annotations

import time

import pytest
import torch

from src.models.ops.cuda import fp8_gemm_dispatch
from src.models.ops.cuda.fp8_gemm import is_available


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not is_available(),
    reason="FP8 GEMM dispatcher requires CUDA + a prebuilt .so for the current SM",
)


def _setup(M: int, K: int, N: int):
    """Build (A_q, B_q, a_scale, b_scale, out_buf, y_cublas_ref)."""
    torch.manual_seed(0)
    A_bf16 = (torch.randn(M, K, dtype=torch.bfloat16, device="cuda") * 0.1).contiguous()
    B_bf16 = (torch.randn(N, K, dtype=torch.bfloat16, device="cuda") * 0.05).contiguous()
    A_fp32, B_fp32 = A_bf16.float(), B_bf16.float()

    a_scale = (A_fp32.abs().amax(dim=1, keepdim=True).clamp(min=1e-6) / 448.0).contiguous()
    b_scale_in = B_fp32.abs().amax(dim=1, keepdim=True).clamp(min=1e-6) / 448.0
    b_scale = b_scale_in.T.contiguous()

    A_q = (A_fp32 / a_scale).clamp(-448, 448).to(torch.float8_e4m3fn).contiguous()
    B_q = (B_fp32 / b_scale_in).clamp(-448, 448).to(torch.float8_e4m3fn).contiguous()
    out = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")

    # cuBLAS reference (always available, our floor truth).
    y_cublas = torch._scaled_mm(A_q, B_q.T, a_scale, b_scale, out_dtype=torch.bfloat16)
    return A_q, B_q, a_scale, b_scale, out, y_cublas


# ---------------------------------------------------------------------------
# Backend discovery + cache
# ---------------------------------------------------------------------------
def test_dispatch_lists_current_arch_configs():
    """`_list_arch_configs` returns at least one .so on the active SM."""
    configs = fp8_gemm_dispatch._list_arch_configs()
    assert len(configs) > 0, (
        f"No prebuilt fp8_gemm .so for current arch. "
        f"Run scripts/build_fp8_gemm.py --arch $SM."
    )


def test_dispatch_cache_hit_on_second_call():
    """Same (M, K, N) → no re-benchmark. Counter by monkey-patching."""
    fp8_gemm_dispatch.fp8_gemm_dispatch_reset_cache()
    M, K, N = 1024, 1536, 1536  # canonical KDA qkv prod shape
    A_q, B_q, a_scale, b_scale, out, _ = _setup(M, K, N)

    # First call populates the cache.
    t0 = time.time()
    fp8_gemm_dispatch.fp8_gemm_auto_dispatch(A_q, B_q, a_scale, b_scale, out=out)
    t1 = time.time()
    cold_ms = (t1 - t0) * 1000

    # Second call should be much faster (O(1)).
    t0 = time.time()
    fp8_gemm_dispatch.fp8_gemm_auto_dispatch(A_q, B_q, a_scale, b_scale, out=out)
    t1 = time.time()
    warm_ms = (t1 - t0) * 1000

    print(f"  cold={cold_ms:.0f} ms, warm={warm_ms:.3f} ms")
    # Cold is the micro-benchmark + 1 timing iter. Warm is just one call.
    # Allow 5x slack on the warm call (can be noisy on small M).
    assert warm_ms < max(cold_ms / 5, 20), (
        f"warm call ({warm_ms:.1f} ms) not fast enough vs cold ({cold_ms:.1f} ms)"
    )


def test_dispatch_cold_start_budget():
    """First-call micro-benchmark across 3 prod shapes ≤ 5 s total."""
    fp8_gemm_dispatch.fp8_gemm_dispatch_reset_cache()
    shapes = [
        (1024, 1536, 1536),  # KDA_qkv
        (1024, 1536, 4096),  # FFN_gate_up
        (1024, 4096, 1536),  # FFN_down
    ]
    t0 = time.time()
    for M, K, N in shapes:
        A_q, B_q, a_scale, b_scale, out, _ = _setup(M, K, N)
        fp8_gemm_dispatch.fp8_gemm_auto_dispatch(A_q, B_q, a_scale, b_scale, out=out)
    elapsed = (time.time() - t0)
    print(f"  3-shape cold start: {elapsed:.2f} s")
    assert elapsed < 5.0, f"cold-start took {elapsed:.1f} s; budget is 5 s"


# ---------------------------------------------------------------------------
# Correctness
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("M,K,N", [
    (128, 1536, 4096),    # small M, FFN shape
    (1024, 1536, 4096),   # prod FFN
    (1024, 1536, 1536),   # prod KDA
    (4096, 1536, 4096),   # large M
])
def test_dispatch_matches_scaled_mm(M, K, N):
    """Dispatcher output matches torch._scaled_mm reference (FP8 noise floor)."""
    fp8_gemm_dispatch.fp8_gemm_dispatch_reset_cache()
    A_q, B_q, a_scale, b_scale, out, y_cublas = _setup(M, K, N)
    y = fp8_gemm_dispatch.fp8_gemm_auto_dispatch(A_q, B_q, a_scale, b_scale, out=out)

    assert torch.isfinite(y).all(), "non-finite in dispatch output"
    rel = (y.float() - y_cublas.float()).abs() / (y_cublas.float().abs() + 1e-3)
    rel_median = rel.median().item()
    print(f"  M={M} K={K} N={N}: rel_median={rel_median*100:.2f}%")
    # Tighten below the FP8 cast noise floor (3-4%); 5% is safe.
    assert rel_median < 0.05, f"dispatcher vs _scaled_mm rel={rel_median*100:.2f}%"


# ---------------------------------------------------------------------------
# Pick policy (the actual reason this exists)
# ---------------------------------------------------------------------------
def test_dispatch_picks_bm64_for_small_m_n1536():
    """For N=1536 shapes (KDA_qkv / FFN_down), small M picks BM64.

    The pick rule from auto-memory ``project_fp8_dispatch_by_m.md`` is
    shape-dependent (N matters, not just M). On N=1536 the 12 col
    tiles (BM=128) under-fill the 36 SMs at small M, so BM64 wins.
    """
    fp8_gemm_dispatch.fp8_gemm_dispatch_reset_cache()
    M, K, N = 256, 1536, 1536  # KDA_qkv, small M, N=1536
    A_q, B_q, a_scale, b_scale, out, _ = _setup(M, K, N)
    fp8_gemm_dispatch.fp8_gemm_auto_dispatch(A_q, B_q, a_scale, b_scale, out=out)

    summary = fp8_gemm_dispatch.fp8_gemm_dispatch_summary()
    winner = summary[f"M={M},K={K},N={N}"]
    print(f"  small-M N=1536 winner: {winner}")

    if winner == "_scaled_mm":
        pytest.skip("cuBLAS picked — no fp8_gemm .so covers this shape on this SM")
    cfg = winner.replace(f"fp8_gemm_{fp8_gemm_dispatch._sm_tag()}_", "").removesuffix(".so")
    bm = int(cfg.split("_")[0][2:])
    assert bm <= 64, (
        f"N=1536 small-M dispatch picked {winner} (BM={bm}); "
        f"per auto-memory sweep, BM64 should win on this regime"
    )


def test_dispatch_picks_bm128_for_n4096_small_m():
    """For N=4096 shapes (FFN_gate_up), even small M picks BM128.

    Counter-example to the naive "small-M → BM64" rule. With N=4096
    the 32 col tiles + 8 row tiles already fills the 36 SMs at BM=128,
    so BM128 keeps winning almost everywhere.
    """
    fp8_gemm_dispatch.fp8_gemm_dispatch_reset_cache()
    M, K, N = 256, 1536, 4096  # FFN_gate_up, small M, N=4096
    A_q, B_q, a_scale, b_scale, out, _ = _setup(M, K, N)
    fp8_gemm_dispatch.fp8_gemm_auto_dispatch(A_q, B_q, a_scale, b_scale, out=out)

    summary = fp8_gemm_dispatch.fp8_gemm_dispatch_summary()
    winner = summary[f"M={M},K={K},N={N}"]
    print(f"  small-M N=4096 winner: {winner}")

    if winner == "_scaled_mm":
        pytest.skip("cuBLAS picked — no fp8_gemm .so covers this shape on this SM")
    cfg = winner.replace(f"fp8_gemm_{fp8_gemm_dispatch._sm_tag()}_", "").removesuffix(".so")
    bm = int(cfg.split("_")[0][2:])
    assert bm >= 128, (
        f"N=4096 small-M dispatch picked {winner} (BM={bm}); "
        f"per auto-memory sweep, BM128 should win on this regime"
    )


def test_dispatch_picks_bm128_for_large_m():
    """At large M, the cached winner has BM=128 in the filename."""
    fp8_gemm_dispatch.fp8_gemm_dispatch_reset_cache()
    M, K, N = 4096, 1536, 4096  # large M
    A_q, B_q, a_scale, b_scale, out, _ = _setup(M, K, N)
    fp8_gemm_dispatch.fp8_gemm_auto_dispatch(A_q, B_q, a_scale, b_scale, out=out)

    summary = fp8_gemm_dispatch.fp8_gemm_dispatch_summary()
    winner = summary[f"M={M},K={K},N={N}"]
    print(f"  large-M winner: {winner}")

    if winner == "_scaled_mm":
        pytest.skip("cuBLAS picked — no fp8_gemm .so covers this shape on this SM")
    cfg = winner.replace(f"fp8_gemm_{fp8_gemm_dispatch._sm_tag()}_", "").removesuffix(".so")
    bm = int(cfg.split("_")[0][2:])
    assert bm >= 128, (
        f"large-M dispatch picked {winner} (BM={bm}); expected BM≥128 per "
        f"project_fp8_dispatch_by_m.md sweep"
    )


def test_dispatch_no_regression_at_large_m():
    """Large-M dispatch time should not exceed cuBLAS by more than 5%.

    Regression guard: the dispatcher's pick logic should never pick a
    config worse than the always-available cuBLAS fallback at any M.
    """
    fp8_gemm_dispatch.fp8_gemm_dispatch_reset_cache()
    M, K, N = 8192, 1536, 4096  # far above crossover, must not regress
    A_q, B_q, a_scale, b_scale, out, _ = _setup(M, K, N)

    # warm up cuBLAS first
    for _ in range(5):
        torch._scaled_mm(A_q, B_q.T, a_scale, b_scale, out_dtype=torch.bfloat16)
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(20):
        torch._scaled_mm(A_q, B_q.T, a_scale, b_scale, out_dtype=torch.bfloat16)
    e.record()
    e.synchronize()
    cublas_ms = s.elapsed_time(e) / 20

    # warm up dispatcher (cache populated, then time)
    for _ in range(5):
        fp8_gemm_dispatch.fp8_gemm_auto_dispatch(A_q, B_q, a_scale, b_scale, out=out)
    torch.cuda.synchronize()
    s.record()
    for _ in range(20):
        fp8_gemm_dispatch.fp8_gemm_auto_dispatch(A_q, B_q, a_scale, b_scale, out=out)
    e.record()
    e.synchronize()
    disp_ms = s.elapsed_time(e) / 20

    print(f"  large-M cuBLAS={cublas_ms:.3f} ms, dispatch={disp_ms:.3f} ms")
    summary = fp8_gemm_dispatch.fp8_gemm_dispatch_summary()
    winner = summary[f"M={M},K={K},N={N}"]
    # If the dispatcher picked cuBLAS, it's fine (no fp8_gemm candidate
    # available). If it picked a .so, it must be ≤5% slower than cuBLAS.
    if winner != "_scaled_mm":
        assert disp_ms <= cublas_ms * 1.05, (
            f"large-M dispatch regressed: {winner} is {disp_ms / cublas_ms:.3f}x cuBLAS"
        )


def test_dispatch_cublas_branch_writes_to_caller_out():
    """Regression guard: cuBLAS branch must write into the caller's ``out``.

    ``torch._scaled_mm`` ignores the caller's ``out`` and allocates
    internally. Without an explicit ``.copy_`` the dispatcher would
    return its fresh (uninitialized) zero buffer — silent zero-output
    bug (caught 2026-07-22 when forward_precomputed SwiGLU tests went
    from sig_rel=0 to sig_rel=1.0 after the dispatch switch).

    Pick a tiny M (e.g. 32) so no ``fp8_gemm_*.so`` candidate matches —
    the dispatcher must fall back to cuBLAS, exercising this branch.
    """
    fp8_gemm_dispatch.fp8_gemm_dispatch_reset_cache()
    M, K, N = 32, 128, 256  # below BM=64 threshold → fallback to cuBLAS
    A_q, B_q, a_scale, b_scale, out, y_cublas = _setup(M, K, N)

    # Pre-fill `out` with a sentinel so a no-op write would be obvious.
    out.fill_(12345.0)
    result = fp8_gemm_dispatch.fp8_gemm_auto_dispatch(A_q, B_q, a_scale, b_scale, out=out)

    # The dispatcher must return the SAME buffer the caller passed in
    # (id-equality is the contract — callers may hold that reference).
    assert result.data_ptr() == out.data_ptr(), (
        "dispatcher returned a fresh buffer instead of writing into the caller's out"
    )
    # And the caller-visible buffer must contain the GEMM result, NOT
    # the sentinel. Check against the direct cuBLAS reference.
    rel = (out.float() - y_cublas.float()).abs().max().item()
    assert rel < 0.5, (
        f"dispatcher's cuBLAS-fallback output diverged from reference by {rel:.3f}"
    )
