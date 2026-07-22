"""FP8 E4M3 GEMM runtime auto-dispatcher.

Picks the fastest available backend on the current device for each
(M, K, N) shape, with first-call micro-benchmark + per-process cache.
Designed for cross-arch deployment (sm_120 dev box, sm_89 prod, future
sm_100+): per-arch .so discovery in `lib/` falls back to
`torch._scaled_mm` when no prebuilt config matches.

Used as the public entry point when migrating from the legacy
single-config `fp8_gemm_scaled` (which always loads the first entry in
`preferred`).

Architecture (per auto-memory ``project_fp8_dispatch_by_m.md``):
- Per-arch .so discovery: ``fp8_gemm_sm_<arch>_<cfg>.so`` in
  ``src/models/ops/cuda/lib/``. Empty list (e.g. on a new arch before a
  build) → fall through to ``torch._scaled_mm``.
- First-call benchmark: for each ``(M, K, N)`` tuple seen, micro-bench
  every candidate (5 warmup + 10 timed iters per backend, CUDA-event
  median), filter out configs whose tile divisibility doesn't match,
  pick the fastest.
- Cache key: ``(M, K, N)``. Cache value: ``("_scaled_mm", None)`` or
  ``(so_filename, ctypes.CDLL handle)``. Same-shape repeat calls are O(1).
- Cold-start cost: ≤2 s on first unique shape; subsequent calls free.

Per ``feedback_autotune_production.md`` (production kernels are
hardcoded config): the `.so` files ARE the hardcoded config; the
dispatcher just picks between them, no JIT compile. Confirms with
``feedback_test_first.md`` recipe: every new dispatched config is
checked at probe time (the BF16 reference sweep — see
``scripts/probes/probe_fp8_small_m_sweep.py``).

Public API:

    from src.models.ops.cuda.fp8_gemm_dispatch import (
        fp8_gemm_auto_dispatch,         # picks per shape, cached
        fp8_gemm_dispatch_summary,      # show what's cached
        fp8_gemm_dispatch_reset_cache,  # for testing
    )

The legacy ``fp8_gemm_scaled`` in ``fp8_gemm.py`` keeps its
single-config behaviour for backward compatibility with
``use_custom_gemm=True`` in NVFP4 W4A8.
"""
from __future__ import annotations

import ctypes
from pathlib import Path
from typing import Optional

import torch

from src.models.ops.cuda.fp8_gemm import _sm_tag  # re-use existing arch tag


_LIB_DIR = Path(__file__).resolve().parent / "lib"

# ``_CUBLAS`` is the sentinel cache value for "always-available fallback".
_CUBLAS = "_scaled_mm"

# Per (M, K, N) tuple → (kind, lib).
# kind is either _CUBLAS or a `.so` filename string.
_CACHE: dict[tuple[int, int, int], tuple[str, Optional[ctypes.CDLL]]] = {}

# Lazily-loaded ctypes handles by filename.
_LOADED: dict[str, ctypes.CDLL] = {}


def _load_so(so_path: Path) -> ctypes.CDLL:
    if so_path.name not in _LOADED:
        lib = ctypes.CDLL(str(so_path), mode=ctypes.RTLD_LOCAL)
        lib.gemm_run.restype = None
        lib.gemm_run.argtypes = [
            ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p,
        ]
        _LOADED[so_path.name] = lib
    return _LOADED[so_path.name]


def _list_arch_configs() -> list[str]:
    """List prebuilt `.so` filenames for the current SM arch. Sorted."""
    sm = _sm_tag()
    return sorted(p.name for p in _LIB_DIR.glob(f"fp8_gemm_{sm}_*.so"))


def _parse_cfg(so_name: str) -> Optional[tuple[int, int, int]]:
    """Parse (BM, BN, BK) from ``fp8_gemm_sm_<X>_<cfg>.so``.

    Returns ``None`` on parse failure (caller skips the config rather
    than risk dispatching to a config whose tile doesn't match M/K/N).
    """
    cfg = so_name.replace(f"fp8_gemm_{_sm_tag()}_", "").removesuffix(".so")
    try:
        bm = int(cfg.split("_")[0][2:])  # "bm128" → 128
        bn = int(cfg.split("_")[1][2:])  # "bn128" → 128
        bk = int(cfg.split("_")[2][2:])  # "bk128" → 128
    except Exception:
        return None
    return bm, bn, bk


def _time_call(kind: str, lib: Optional[ctypes.CDLL],
               M: int, K: int, N: int,
               A_q: torch.Tensor, B_q: torch.Tensor,
               a_scale: torch.Tensor, b_scale: torch.Tensor,
               out: torch.Tensor) -> float:
    """Median of 10 timed iters (ms) for one backend + shape.

    5 warmup first (CUTLASS heuristic + workspace alloc + JIT for the
    ctypes binding path).
    """
    for _ in range(5):
        _run(kind, lib, M, K, N, A_q, B_q, a_scale, b_scale, out)
    torch.cuda.synchronize()
    times = []
    for _ in range(10):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        _run(kind, lib, M, K, N, A_q, B_q, a_scale, b_scale, out)
        e.record()
        e.synchronize()
        times.append(s.elapsed_time(e))
    times.sort()
    return times[len(times) // 2]


def _run(kind: str, lib: Optional[ctypes.CDLL],
         M: int, K: int, N: int,
         A_q: torch.Tensor, B_q: torch.Tensor,
         a_scale: torch.Tensor, b_scale: torch.Tensor,
         out: torch.Tensor) -> torch.Tensor:
    """Issue the GEMM. Returns the output tensor (may equal ``out``).

    ``torch._scaled_mm`` ignores the caller's ``out`` parameter (it
    allocates internally), so the cuBLAS branch must rebind and copy
    — otherwise the dispatcher would return its uninitialized zero
    buffer. The custom-kernel branch writes into ``out`` in-place.
    """
    if kind == _CUBLAS:
        result = torch._scaled_mm(
            A_q, B_q.T, a_scale, b_scale, out_dtype=torch.bfloat16,
        )
        # In-place copy so the caller's ``out`` reference is honoured
        # (avoids a fresh alloc each call in steady state).
        out.copy_(result)
        return out
    else:
        stream = torch.cuda.current_stream().cuda_stream
        lib.gemm_run(
            M, N, K,
            A_q.data_ptr(), B_q.data_ptr(), out.data_ptr(),
            a_scale.data_ptr(), b_scale.data_ptr(),
            0, stream,
        )
        return out


def _pick_backend(M: int, K: int, N: int,
                  A_q: torch.Tensor, B_q: torch.Tensor,
                  a_scale: torch.Tensor, b_scale: torch.Tensor,
                  out: torch.Tensor) -> tuple[str, Optional[ctypes.CDLL]]:
    """Micro-benchmark every eligible backend for (M, K, N). Returns
    (winner_kind, lib_or_None). When no .so matches, returns the
    ``_scaled_mm`` sentinel."""
    candidates: list[tuple[str, Optional[ctypes.CDLL]]] = [(_CUBLAS, None)]
    for so_name in _list_arch_configs():
        parsed = _parse_cfg(so_name)
        if parsed is None:
            continue
        bm, bn, bk = parsed
        # Kernel hard-asserts: M, N, K must each be divisible by BM/BN/BK.
        # Skip mismatches here so we don't pay a launch that aborts.
        if M % bm or N % bn or K % bk:
            continue
        candidates.append((so_name, _load_so(_LIB_DIR / so_name)))

    if len(candidates) == 1:
        return candidates[0]  # only cuBLAS; nothing to pick

    timings: list[tuple[float, str, Optional[ctypes.CDLL]]] = []
    for kind, lib in candidates:
        try:
            ms = _time_call(kind, lib, M, K, N, A_q, B_q, a_scale, b_scale, out)
            tflops = (2 * M * K * N) / (ms * 1e9)
            timings.append((ms, kind, lib))
        except Exception:
            # A backend misfired (e.g. divisibility check we missed, or
            # the kernel miscompiled for a slightly different AB layout).
            # Skip — the next-best call is unaffected.
            continue

    if not timings:
        return _CUBLAS, None
    timings.sort(key=lambda x: x[0])
    return timings[0][1], timings[0][2]


def fp8_gemm_auto_dispatch(
    A: torch.Tensor,
    B: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """``Y = (A * a_scale_row) @ (B * b_scale_col).T`` → ``Y [M, N] BF16``.

    Auto-dispatches across every prebuilt ``fp8_gemm_sm_<arch>_<cfg>.so``
    on the current device, plus ``torch._scaled_mm``. First call with a
    new ``(M, K, N)`` tuple micro-benches candidates and caches the
    winner (``≤2 s`` cold start, O(1) thereafter). Falls back to
    ``_scaled_mm`` if no `.so` matches the current arch (e.g. on a
    freshly-built env before ``scripts/build_fp8_gemm.py`` runs).

    Args:
        A: ``[M, K]`` ``float8_e4m3fn`` row-major contiguous.
        B: ``[N, K]`` ``float8_e4m3fn`` row-major contiguous.
        a_scale: ``[M, 1]`` or ``[M]`` FP32 contiguous.
        b_scale: ``[1, N]`` or ``[N]`` FP32 contiguous.
        out: optional ``[M, N]`` BF16 output buffer (else freshly empty).

    Returns:
        ``Y [M, N]`` bfloat16.
    """
    assert A.dtype == torch.float8_e4m3fn and A.is_cuda
    assert B.dtype == torch.float8_e4m3fn and B.is_cuda
    M, K = A.shape
    N, K2 = B.shape
    assert K == K2, f"K mismatch: A has {K}, B has {K2}"

    if out is None:
        out = torch.empty(M, N, dtype=torch.bfloat16, device=A.device)
    else:
        assert out.shape == (M, N) and out.dtype == torch.bfloat16

    key = (M, K, N)
    if key not in _CACHE:
        kind, lib = _pick_backend(M, K, N, A, B, a_scale, b_scale, out)
        _CACHE[key] = (kind, lib)

    kind, lib = _CACHE[key]
    out = _run(kind, lib, M, K, N, A, B, a_scale, b_scale, out)
    return out


def fp8_gemm_dispatch_summary() -> dict:
    """Return the current dispatch cache as a {key → winner} dict.

    Useful for logging which backend was picked per shape (e.g. for
    ``print(fp8_gemm_dispatch_summary())`` once per training run).
    """
    return {f"M={k[0]},K={k[1]},N={k[2]}": kind for k, (kind, _) in _CACHE.items()}


def fp8_gemm_dispatch_reset_cache() -> None:
    """Clear the dispatch cache. Test-only — see ``feedback_test_first.md``."""
    _CACHE.clear()
