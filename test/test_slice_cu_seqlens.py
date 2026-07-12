"""Direct unit tests for :func:`_slice_cu_seqlens`.

Why this exists
---------------
The function is exercised indirectly by
``test_chunk_loss_averaging.py`` (the chunk loop) but that test
mixes a real model + CUDA + TP setup. The pure-function
contracts are easier to pin here:

  - Empty chunk (``chunk_start == chunk_end``) returns ``[0]``.
  - A global boundary at ``chunk_start`` becomes local 0
    (the chunk opens a new doc at its left edge).
  - A global boundary at ``chunk_end`` becomes local
    ``chunk_size`` (the chunk closes its last doc at its
    right edge).
  - A global boundary outside ``(chunk_start, chunk_end]`` is
    dropped.
  - The returned tensor starts at 0 and ends at ``chunk_size``
    even when no global boundary lies in range (the function
    synthesizes the bounds defensively).
  - Boundaries are shifted by ``-chunk_start`` (chunk-local
    origin).

These are the contracts the ShortConvolution kernel relies on
to know when to reset the depthwise state. A regression in
``_slice_cu_seqlens`` would silently corrupt the per-doc state
in every chunked training step.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from src.training.loop import _slice_cu_seqlens  # noqa: E402


# --------------------------------------------------------------------------- #
# Empty chunk.                                                                #
# --------------------------------------------------------------------------- #
def test_empty_chunk_returns_single_zero():
    """``chunk_start == chunk_end`` is the degenerate empty chunk;
    the function returns ``[0]`` (a well-defined single-entry
    cu_seqlens) rather than an empty tensor (which would crash
    the ShortConvolution kernel)."""
    cu = torch.tensor([10, 20, 30], dtype=torch.int32)
    out = _slice_cu_seqlens(cu, chunk_start=5, chunk_end=5)
    assert out.tolist() == [0]
    assert out.dtype == torch.int32


# --------------------------------------------------------------------------- #
# Boundary at chunk_start / chunk_end.                                       #
# --------------------------------------------------------------------------- #
def test_boundary_at_chunk_start_becomes_local_zero():
    """A global boundary exactly at ``chunk_start`` becomes the
    chunk's local 0 (the chunk opens a new doc at its left
    edge — same convention as :func:`pack_chunk_aligned`)."""
    cu = torch.tensor([0, 5, 10, 15], dtype=torch.int32)
    out = _slice_cu_seqlens(cu, chunk_start=5, chunk_end=10)
    # Boundary at 5 → local 0; boundary at 10 → local 5
    # (chunk_size).
    assert out.tolist() == [0, 5]


def test_boundary_at_chunk_end_becomes_local_chunk_size():
    """A global boundary at ``chunk_end`` becomes the chunk's
    local ``chunk_size`` (right edge)."""
    cu = torch.tensor([0, 5, 10, 15], dtype=torch.int32)
    out = _slice_cu_seqlens(cu, chunk_start=5, chunk_end=15)
    # 5 → 0, 10 → 5, 15 → 10.
    assert out.tolist() == [0, 5, 10]


def test_boundary_strictly_inside_chunk_is_shifted_by_negative_chunk_start():
    """Boundaries inside ``(chunk_start, chunk_end)`` are
    chunk-local: global value minus ``chunk_start``."""
    cu = torch.tensor([0, 5, 100, 150, 200, 250], dtype=torch.int32)
    out = _slice_cu_seqlens(cu, chunk_start=100, chunk_end=200)
    # In range: 100, 150, 200 (all <= 200). Shifted by -100: 0,
    # 50, 100. The function adds [0] at the start if needed
    # (here 100 -> 0 is already 0, so no extra prepend). Adds
    # [chunk_size=100] at the end if needed (200 -> 100 is
    # already 100, so no extra append).
    assert out.tolist() == [0, 50, 100]


# --------------------------------------------------------------------------- #
# Boundary outside range is dropped.                                         #
# --------------------------------------------------------------------------- #
def test_boundary_outside_range_is_dropped():
    """Boundaries at ``< chunk_start`` or ``> chunk_end`` are
    dropped (they belong to other chunks)."""
    cu = torch.tensor([0, 5, 10, 100, 200, 300], dtype=torch.int32)
    out = _slice_cu_seqlens(cu, chunk_start=100, chunk_end=200)
    # In range: 100, 200. 0, 5, 10 are < chunk_start (dropped).
    # 300 is > chunk_end (dropped).
    # Shifted: 0, 100. The function fills in [0] at start (already
    # 0) and [chunk_size=100] at end (already 100).
    assert out.tolist() == [0, 100]


def test_only_boundaries_outside_range_yields_synthesized_bounds():
    """When no global boundary lies in ``(chunk_start, chunk_end]``,
    the function synthesizes the left + right edges (otherwise
    the ShortConvolution kernel has no markers and would
    treat the chunk as a single doc)."""
    cu = torch.tensor([0, 5, 10, 15], dtype=torch.int32)  # all < 20
    out = _slice_cu_seqlens(cu, chunk_start=20, chunk_end=40)
    # No in-range boundaries. Synthesized: [0, chunk_size=20].
    assert out.tolist() == [0, 20]


# --------------------------------------------------------------------------- #
# First boundary not at 0 → prepend 0.                                       #
# --------------------------------------------------------------------------- #
def test_first_in_range_boundary_not_at_zero_prepends_zero():
    """If the first in-range boundary is not at ``chunk_start``
    (e.g. an internal boundary), the function prepends 0 to mark
    the chunk's local origin as a doc boundary (the chunk
    starts mid-doc and the depthwise state must be reset there
    — same convention as the left-edge case)."""
    cu = torch.tensor([0, 5, 130, 150], dtype=torch.int32)
    out = _slice_cu_seqlens(cu, chunk_start=100, chunk_end=200)
    # In range: 130, 150. Shifted: 30, 50. First is not 0 →
    # prepend 0. Last (50) is not 100 → append 100.
    assert out.tolist() == [0, 30, 50, 100]


# --------------------------------------------------------------------------- #
# dtype preservation.                                                         #
# --------------------------------------------------------------------------- #
def test_dtype_preserved():
    """The output dtype matches the input (int32 / int64 — the
    ShortConvolution kernel reads it as int32 but the helper
    passes through whatever the caller gave)."""
    cu32 = torch.tensor([0, 5, 10, 15], dtype=torch.int32)
    out32 = _slice_cu_seqlens(cu32, chunk_start=5, chunk_end=10)
    assert out32.dtype == torch.int32

    cu64 = torch.tensor([0, 5, 10, 15], dtype=torch.int64)
    out64 = _slice_cu_seqlens(cu64, chunk_start=5, chunk_end=10)
    assert out64.dtype == torch.int64


# --------------------------------------------------------------------------- #
# Multi-boundary dense case (the realistic packed-sequence shape).            #
# --------------------------------------------------------------------------- #
def test_dense_boundaries_chunk_covers_multiple_docs():
    """A realistic packed-sequence chunk: chunk spans
    ``[100, 200)`` of a global sequence with doc boundaries
    every 10 tokens. The chunk-local cu_seqlens must include
    the boundaries at ``100, 110, 120, ..., 200`` (chunk-local
    ``0, 10, 20, ..., 100``)."""
    cu = torch.arange(10, 1010, 10, dtype=torch.int32)  # [10, 20, ..., 1000]
    out = _slice_cu_seqlens(cu, chunk_start=100, chunk_end=200)
    # In-range: 100, 110, 120, ..., 200. Shifted: 0, 10, ..., 100.
    expected = list(range(0, 110, 10))  # 12 entries: 0..100 step 10
    assert out.tolist() == expected


# --------------------------------------------------------------------------- #
# Single-boundary case (the chunk is exactly one doc).                       #
# --------------------------------------------------------------------------- #
def test_chunk_exactly_one_doc():
    """A chunk whose boundaries are exactly ``[chunk_start,
    chunk_end]`` (one full doc) returns ``[0, chunk_size]`` —
    the synthesis branch ensures this even when both
    endpoints are explicit in the global cu_seqlens (no
    prepend/append happens because the values already match)."""
    cu = torch.tensor([0, 100, 200, 300], dtype=torch.int32)
    out = _slice_cu_seqlens(cu, chunk_start=100, chunk_end=200)
    # In range: 100, 200. Shifted: 0, 100. Both ends already
    # match; no prepend / append.
    assert out.tolist() == [0, 100]


# --------------------------------------------------------------------------- #
# Plain-Python runner. Pytest is not a project dependency; collect every      #
# ``def test_*`` and invoke, reporting pass/fail per test.                    #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import inspect
    import traceback

    tests = [
        (name, fn)
        for name, fn in globals().items()
        if name.startswith("test_") and callable(fn)
    ]
    tests.sort(key=lambda kv: inspect.getsourcelines(kv[1])[1])

    passed, failed = 0, 0
    for name, fn in tests:
        try:
            fn()
        except Exception:
            failed += 1
            print(f"  FAIL  {name}")
            traceback.print_exc()
        else:
            passed += 1
            print(f"  PASS  {name}")
    print(f"\n{passed} passed, {failed} failed, {passed + failed} total")
    if failed:
        sys.exit(1)