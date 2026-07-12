"""Tests for the pure-function helpers in :mod:`src.training.loop`.

These helpers are either exported (``:func:`_slice_cu_seqlens```,
``:func:`wsd_lr```) or mirrored here (the chunk-loop math inside
``:func:`_run_training_loop```) so the upcoming training-loop
refactor can proceed against pinned contracts.

What this file pins
-------------------

  * **:func:`_slice_cu_seqlens`** — chunk-local doc-boundary
    projection (corner cases: empty chunk, boundary exactly at
    chunk start / chunk end, all-boundaries-outside, no-inside
    boundary, dtype preservation).
  * **chunk_valid formula** — the per-chunk ``(labels[:, ci*mb+1
    : (ci+1)*mb] != -100).sum()`` count used to weight the
    step-loss (the +1 offset matches FusedLinearCE's label
    shift; dropping it would silently corrupt the loss average).
  * **last_real formula** — the ``max(...)`` over chunks with
    ``nv > 0`` (default ``-1``). Drives the per-chunk loop's
    upper bound and the trailing-empty-chunk skip.
  * **n_chunks divisibility** — ``micro_batch_size`` must divide
    ``seq_len``; default / unset ``micro_batch_size`` falls back
    to one chunk (legacy single-mb behavior).
  * **WSD LR opt.lr contract** — ``peak_lr == 0`` is the
    documented "disable scheduling" opt-out the loop uses for
    smoke runs; ``muon_lr`` and ``learning_rate`` are scaled
    independently by the same schedule.

Each mirror here is a faithful replica of the loop's formula and
MUST be updated in lockstep if the loop formula changes. This is
the same convention used by
:mod:`test.test_chunk_loss_averaging` for the per-chunk step-loss
formula.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from src.training.loop import _slice_cu_seqlens, wsd_lr  # noqa: E402


# =========================================================================== #
# _slice_cu_seqlens — chunk-local doc-boundary projection
# =========================================================================== #
def test_slice_cu_seqlens_includes_boundaries_inside_chunk():
    """Boundaries strictly inside ``(chunk_start, chunk_end]`` become
    local boundaries; outside boundaries are dropped.

    The convention (per support.py docstring): the chunk-local
    tensor starts at 0 (the chunk's own left edge) and ends at
    ``chunk_size`` (the chunk's own right edge); inside
    boundaries are shifted by ``-chunk_start``.
    """
    cu = torch.tensor([0, 10, 25, 35, 50], dtype=torch.int32)
    # Chunk covers [12, 30): boundary at 25 is inside.
    local = _slice_cu_seqlens(cu, chunk_start=12, chunk_end=30)
    assert local.tolist() == [0, 13, 18]


def test_slice_cu_seqlens_excludes_boundaries_outside_chunk():
    """Boundaries <= chunk_start and > chunk_end are dropped.

    Note: the function ALWAYS returns chunk-local coordinates
    (shifted by ``-chunk_start``), not global. Boundary at
    global offset 50 in a [40, 60) chunk becomes local offset 10.
    """
    cu = torch.tensor([0, 10, 25, 35, 50], dtype=torch.int32)
    # Chunk [0, 5): only the 0 boundary is inside (local 0).
    local = _slice_cu_seqlens(cu, chunk_start=0, chunk_end=5)
    assert local.tolist() == [0, 5]
    # Chunk [40, 60): only the 50 boundary is inside (local 10).
    # Prepend 0 (no boundary at local 0), append chunk_size=20.
    local = _slice_cu_seqlens(cu, chunk_start=40, chunk_end=60)
    assert local.tolist() == [0, 10, 20]


def test_slice_cu_seqlens_boundary_at_chunk_start_becomes_local_zero():
    """A global boundary exactly at ``chunk_start`` becomes the
    chunk's local 0 (the chunk opens a new "doc" at its first
    position). The defensive 0-prepend must not double-emit.
    """
    cu = torch.tensor([0, 12, 25], dtype=torch.int32)
    local = _slice_cu_seqlens(cu, chunk_start=12, chunk_end=20)
    assert local[0].item() == 0
    assert local[-1].item() == 20 - 12
    # And the interior [0, 8] has no other boundaries -> just [0, 8].
    assert local.tolist() == [0, 8]


def test_slice_cu_seqlens_boundary_at_chunk_end_becomes_local_chunk_size():
    """A global boundary exactly at ``chunk_end`` becomes the
    chunk's local ``chunk_size``. The defensive end-append must
    not double-emit (the boundary IS the chunk end).
    """
    cu = torch.tensor([0, 5, 20], dtype=torch.int32)
    local = _slice_cu_seqlens(cu, chunk_start=5, chunk_end=20)
    # Boundary at 20 -> local 15; boundary at 5 -> local 0; no other
    # interior boundaries; tail 15 == chunk_size so no double-emit.
    assert local.tolist() == [0, 15]


def test_slice_cu_seqlens_does_not_double_emit_existing_zero():
    """If 0 is already a boundary (chunk_start == 0), the defensive
    0-prepend is a no-op (otherwise we'd see [0, 0, ...])."""
    cu = torch.tensor([0, 12], dtype=torch.int32)
    local = _slice_cu_seqlens(cu, chunk_start=0, chunk_end=8)
    assert local[0].item() == 0
    assert local[-1].item() == 8
    # Exactly 2 entries (no double-prepend).
    assert local.numel() == 2


def test_slice_cu_seqlens_does_not_double_emit_existing_chunk_end():
    """If the chunk end already matches a boundary, the defensive
    end-append is a no-op."""
    cu = torch.tensor([0, 8, 16], dtype=torch.int32)
    local = _slice_cu_seqlens(cu, chunk_start=0, chunk_end=8)
    # Boundaries at 0 and 8; the tail 8 == chunk_size so no double-emit.
    assert local.tolist() == [0, 8]


def test_slice_cu_seqlens_empty_chunk_returns_zero():
    """``chunk_start == chunk_end`` is a degenerate but well-defined
    case: returns a single-entry [0] tensor (the empty-chunk
    sentinel the KDA / ShortConv kernels interpret as "no
    boundaries to reset at").
    """
    cu = torch.tensor([0, 10, 25], dtype=torch.int32)
    local = _slice_cu_seqlens(cu, chunk_start=10, chunk_end=10)
    assert local.tolist() == [0]


def test_slice_cu_seqlens_no_inside_boundary_still_emits_endpoints():
    """When no doc boundary falls inside the chunk range, the
    returned tensor is just ``[0, chunk_size]`` — a single 'doc'
    spanning the whole chunk.
    """
    cu = torch.tensor([0, 50, 100], dtype=torch.int32)
    local = _slice_cu_seqlens(cu, chunk_start=10, chunk_end=20)
    assert local.tolist() == [0, 10]


def test_slice_cu_seqlens_dtype_is_preserved():
    """The returned tensor inherits the input dtype (the kernels
    index into ``cu_seqlens`` so dtype mismatches would be a
    silent corruption).
    """
    for dtype in (torch.int32, torch.int64):
        cu = torch.tensor([0, 10, 20], dtype=dtype)
        local = _slice_cu_seqlens(cu, 5, 15)
        assert local.dtype == dtype


def test_slice_cu_seqlens_many_inside_boundaries_in_order():
    """Multiple inside boundaries are preserved in ascending order
    (the kernel relies on the ascending invariant)."""
    cu = torch.tensor([0, 10, 20, 30, 40, 50], dtype=torch.int32)
    local = _slice_cu_seqlens(cu, chunk_start=10, chunk_end=40)
    # Boundaries at 10 (local 0), 20 (local 10), 30 (local 20), 40 (local 30).
    # The defensive end-append sees local 30 == chunk_size and skips.
    assert local.tolist() == [0, 10, 20, 30]
    assert (local.diff() > 0).all().item(), (
        "slice_cu_seqlens must return ascending boundaries"
    )


# =========================================================================== #
# chunk_valid formula — mirror of run.py:213-220
# =========================================================================== #
def _chunk_valid_mirror(
    labels: torch.Tensor, micro_batch_size: int, n_chunks: int,
) -> list[int]:
    """Faithful replica of :func:`_run_training_loop`'s per-chunk
    valid-token count (lines 213-220).

    Loop formula::

        chunk_valid[ci] = (labels[:, ci*mb+1 : (ci+1)*mb] != -100).sum().item()

    The ``+1`` offset matches the model's label-shift (predict
    ``t+1`` from ``t``), so the count is over the same token
    window FusedLinearCE averages over. If the offset drifts,
    the loss averaging changes magnitude even when the chunk
    contents are identical.
    """
    return [
        int(
            (labels[:, ci * micro_batch_size + 1: (ci + 1) * micro_batch_size] != -100)
            .sum().item()
        )
        for ci in range(n_chunks)
    ]


def test_chunk_valid_uses_plus_one_shift():
    """The +1 offset matches FusedLinearCE's label shift. A naive
    refactor that drops the offset (e.g. ``[ci*mb : (ci+1)*mb]``)
    would over-count by ``labels[:, chunk_start]`` for every
    chunk — silently biasing the loss average toward the chunk
    boundary tokens.
    """
    # 4-token chunk; labels[0]=real, labels[1..3]=-100.
    labels = torch.tensor([[10, -100, -100, -100]])
    # With +1 shift: chunk_valid[0] = (labels[1:4] != -100).sum() = 0
    # Without +1 shift (buggy): would be (labels[0:3] != -100).sum() = 1
    out = _chunk_valid_mirror(labels, micro_batch_size=4, n_chunks=1)
    assert out == [0], (
        f"chunk_valid must use the +1 label-shift offset (matches "
        f"FusedLinearCE); got {out}"
    )


def test_chunk_valid_real_doc_full_chunk():
    """A chunk where every position is a valid target has
    ``chunk_valid == mb - 1`` — the +1 shift means the chunk's
    first token is never counted (it's the model's input, not
    its target). This is consistent with FusedLinearCE's per-
    token CE which also averages over ``mb - 1`` tokens per
    chunk.
    """
    labels = torch.full((1, 8), 42)
    out = _chunk_valid_mirror(labels, micro_batch_size=8, n_chunks=1)
    assert out == [7]


def test_chunk_valid_all_pad_chunk_returns_zero():
    """An all-(-100) chunk (e.g. trailing-pad FFD pack) has
    ``chunk_valid == 0``. The loop uses this for the ``last_real``
    early-stop (see :func:`_last_real_mirror`)."""
    labels = torch.full((1, 8), -100)
    out = _chunk_valid_mirror(labels, micro_batch_size=8, n_chunks=1)
    assert out == [0]


def test_chunk_valid_real_doc_partial_chunk():
    """Mixed valid/-100 within a chunk yields the correct count."""
    labels = torch.tensor([[1, 2, -100, 4, 5, -100, -100, 6]])
    # +1 shift over indices [1..7]: values [2, -100, 4, 5, -100, -100, 6] -> 4 valid.
    out = _chunk_valid_mirror(labels, micro_batch_size=8, n_chunks=1)
    assert out == [4]


def test_chunk_valid_multiple_chunks_independent():
    """Each chunk is computed independently; valid counts do not bleed."""
    labels = torch.tensor([[1, 2, 3, 4, -100, -100, -100, -100]])
    out = _chunk_valid_mirror(labels, micro_batch_size=4, n_chunks=2)
    # Chunk 0: labels[1:4] = [2, 3, 4] -> 3 valid.
    # Chunk 1: labels[5:8] = [-100, -100, -100] -> 0 valid.
    assert out == [3, 0]


def test_chunk_valid_batch_greater_than_one():
    """The formula sums across the batch dim — labels of shape
    ``[B, T]`` count ``B * (per-token valid)``."""
    labels = torch.tensor([
        [1, 2, -100, -100],
        [3, 4, -100, -100],
    ])
    out = _chunk_valid_mirror(labels, micro_batch_size=4, n_chunks=1)
    # +1 shift over indices [1..3] across both batch rows: [2, -100] + [4, -100] = 2 valid.
    assert out == [2]


def test_chunk_valid_zero_chunks_returns_empty():
    """``n_chunks == 0`` produces an empty list (defensive: the
    loop's ``range(0)`` walks no chunks)."""
    labels = torch.full((1, 8), 42)
    out = _chunk_valid_mirror(labels, micro_batch_size=4, n_chunks=0)
    assert out == []


# =========================================================================== #
# last_real formula — mirror of run.py:228-231
# =========================================================================== #
def _last_real_mirror(chunk_valid: list[int]) -> int:
    """Faithful replica of :func:`_run_training_loop`'s
    ``last_real`` detection (lines 228-231).

    Loop formula::

        last_real = max(
            (ci for ci, nv in enumerate(chunk_valid) if nv > 0),
            default=-1,
        )

    Returns ``-1`` when no chunk has any valid token; the loop's
    ``range(last_real + 1)`` then walks ``range(0)`` = nothing,
    which is a well-defined no-op (no chunks, no loss, no backward).
    """
    return max(
        (ci for ci, nv in enumerate(chunk_valid) if nv > 0),
        default=-1,
    )


def test_last_real_returns_last_chunk_with_valid_tokens():
    """The last index with ``nv > 0`` — regardless of how many
    zero chunks follow."""
    chunk_valid = [10, 10, 0, 0, 0]
    assert _last_real_mirror(chunk_valid) == 1


def test_last_real_returns_zero_when_only_first_chunk_has_valid_tokens():
    """When only chunk 0 has valid tokens, ``last_real == 0``
    (the loop still runs chunk 0's forward + backward)."""
    chunk_valid = [10, 0, 0]
    assert _last_real_mirror(chunk_valid) == 0


def test_last_real_returns_neg_one_when_all_chunks_empty():
    """An all-empty chunk_valid returns ``-1`` (the loop's
    ``range(0)`` then walks no chunks — a defined no-op)."""
    chunk_valid = [0, 0, 0, 0]
    assert _last_real_mirror(chunk_valid) == -1


def test_last_real_empty_input_returns_neg_one():
    """Empty input returns ``-1`` (no chunks, no last)."""
    assert _last_real_mirror([]) == -1


def test_last_real_single_full_chunk():
    """Single fully-populated chunk returns ``0``."""
    assert _last_real_mirror([4096]) == 0


def test_last_real_single_empty_chunk():
    """Single empty chunk returns ``-1``."""
    assert _last_real_mirror([0]) == -1


def test_last_real_matches_sparse_prod_repro():
    """The exact prod repro from the chunked_loss_dilution
    historical bug: 5 full chunks + 1 partial + 10 trailing empty
    chunks → ``last_real == 5`` (the loop walks 6 chunks, NOT 16).

    If the loop ran all 16 chunks, the trailing 10 empty chunks
    would have been processed for nothing and the chunk math
    would have over-counted ``n_chunks`` (re-introducing the
    dilution bug the ``last_real`` early-stop was added to fix).
    """
    chunk_valid = [16384] * 5 + [8192] + [0] * 10
    assert _last_real_mirror(chunk_valid) == 5


# =========================================================================== #
# n_chunks divisibility — mirror of run.py:108-121
# =========================================================================== #
def _resolve_n_chunks(seq_len: int, micro_batch_size: int) -> int:
    """Faithful replica of :func:`_run_training_loop`'s ``n_chunks``
    resolution (lines 108-121).

    Rules (in order):

      1. ``micro_batch_size <= 0`` OR ``micro_batch_size > seq_len``
         → fallback to ``micro_batch_size = seq_len`` (single chunk;
         matches legacy single-mb behavior).
      2. ``n_chunks = seq_len // micro_batch_size``.
      3. ``n_chunks * micro_batch_size != seq_len`` →
         ``ValueError`` (defensive: the per-chunk slice would
         silently drop tokens otherwise).
    """
    mbs = micro_batch_size
    if mbs <= 0 or mbs > seq_len:
        mbs = seq_len
    n_chunks = seq_len // mbs
    if n_chunks * mbs != seq_len:
        raise ValueError(
            f"seq_len ({seq_len}) must be a multiple of "
            f"micro_batch_size ({mbs}); got "
            f"seq_len / micro_batch_size = {seq_len / mbs}"
        )
    return n_chunks


@pytest.mark.parametrize(
    "seq_len, micro_batch_size, expected_n_chunks",
    [
        # Production shape: 16 chunks of 16384 = 262144.
        (262144, 16384, 16),
        # Quick smoke: 1 chunk = full seq.
        (16384, 16384, 1),
        # Larger chunk.
        (262144, 32768, 8),
        # Smaller chunk (4× more chunks).
        (262144, 4096, 64),
    ],
)
def test_n_chunks_derivation(seq_len, micro_batch_size, expected_n_chunks):
    """``n_chunks = seq_len / micro_batch_size`` for valid pairs."""
    assert _resolve_n_chunks(seq_len, micro_batch_size) == expected_n_chunks


def test_n_chunks_zero_micro_batch_size_falls_back_to_one():
    """``micro_batch_size == 0`` is treated as 'unspecified' and
    falls back to a single chunk (legacy single-mb behavior)."""
    assert _resolve_n_chunks(16384, 0) == 1


def test_n_chunks_negative_micro_batch_size_falls_back_to_one():
    """``micro_batch_size < 0`` is also 'unspecified'."""
    assert _resolve_n_chunks(16384, -8) == 1


def test_n_chunks_micro_batch_larger_than_seq_len_falls_back():
    """``micro_batch_size > seq_len``: the legacy fallback applies
    (single chunk covering the whole seq)."""
    assert _resolve_n_chunks(4096, 8192) == 1


@pytest.mark.parametrize(
    "seq_len, micro_batch_size",
    [
        (10000, 3000),  # seq_len / mb = 3.33
        (16385, 4),     # +1 off
        (16, 7),        # small case
    ],
)
def test_n_chunks_non_divisible_raises(seq_len, micro_batch_size):
    """``seq_len`` not divisible by ``micro_batch_size`` is a
    :class:`ValueError`."""
    with pytest.raises(ValueError, match="must be a multiple"):
        _resolve_n_chunks(seq_len=seq_len, micro_batch_size=micro_batch_size)


def test_n_chunks_minimum_one_even_for_tiny_seq():
    """``seq_len=1, micro_batch_size=1`` yields 1 chunk (the
    smallest possible valid case)."""
    assert _resolve_n_chunks(1, 1) == 1


def test_n_chunks_default_fallback_is_one_for_short_seq():
    """Quick configs (e.g. ``quick.yml`` with ``seq_len=4096``)
    that don't set ``micro_batch_size`` get 1 chunk via the
    fallback. This is what makes the loop Just Work on small
    test configs."""
    assert _resolve_n_chunks(seq_len=4096, micro_batch_size=0) == 1


# =========================================================================== #
# WSD LR opt.lr contract — pin the LOOP's specific behavior
# =========================================================================== #
# The pure WSD math is comprehensively covered by
# :mod:`test.test_wsd_lr`. The loop-specific contract below is
# what the LOOP does with that math: peak_lr == 0 disables
# scheduling, and both optimizers are updated independently with
# their own peak.

def test_wsd_peak_zero_disables_scheduling():
    """``peak_lr == 0`` is the documented 'disable scheduling'
    opt-out the loop uses for short smoke runs (per the
    ``warmup_steps / decay_steps`` 0/0 defaults in the loop).

    The constant-zero contract: every step returns 0 regardless
    of warmup / decay / step. Optimizers that store ``self.lr``
    would see a constant 0 — they don't update.
    """
    # Various step positions inside and outside warmup/decay.
    for step in [0, 1, 5, 10, 50, 99]:
        lr = wsd_lr(step, peak_lr=0.0, warmup_steps=10, decay_steps=10, max_steps=100)
        assert lr == 0.0, (
            f"peak_lr=0 must always return 0 (disables scheduling); "
            f"step={step} got lr={lr}"
        )


def test_wsd_peak_negative_disables_scheduling():
    """``peak_lr < 0`` is also a 'no-op' (the loop doesn't run
    negative LRs but tests / callers might pass them in)."""
    assert wsd_lr(50, peak_lr=-1.0, warmup_steps=10, decay_steps=10, max_steps=100) == -1.0
    assert wsd_lr(0, peak_lr=-0.001, warmup_steps=0, decay_steps=0, max_steps=10) == -0.001


def test_wsd_muon_and_adamw_updated_independently_with_own_peak():
    """The loop's per-step contract: ``muon_opt.lr`` and
    ``adamw_opt.lr`` are each set from the SAME schedule but with
    DIFFERENT peaks. The peaks (``args.muon_lr`` and
    ``args.learning_rate``) are typically different in production
    (muon_lr ~ 2-4× adamw_lr).

    This test pins: at every step inside the horizon, the ratio
    of the two LRs equals the ratio of their peaks. (If a refactor
    accidentally couples the schedules — e.g. computes
    ``adamw_lr = muon_lr * (adamw_peak / muon_peak)`` — the ratio
    would still hold; this test guards the weaker contract that
    the schedules are independent peaks but applied uniformly.)

    Note: step 0 returns 0 during warmup (peak * 0/warmup), so we
    don't assert ``lr > 0`` at every step — we assert it's within
    [0, peak].
    """
    muon_peak = 0.02
    adamw_peak = 0.005
    warmup, decay, total = 10, 20, 100

    for step in [0, 5, 10, 50, 80, 99]:
        muon_lr = wsd_lr(step, muon_peak, warmup, decay, total)
        adamw_lr = wsd_lr(step, adamw_peak, warmup, decay, total)
        # Both must be in [0, peak] at every step (warmup starts
        # at 0 and ramps up; decay ends at min_lr=0).
        assert 0 <= muon_lr <= muon_peak
        assert 0 <= adamw_lr <= adamw_peak
        # The ratio (muon / adamw) equals the peak ratio at
        # every step (each is scaled by the same schedule
        # factor). Skip the trivial warmup=0 case (0/0 is
        # undefined) and the decay terminator case (both zero).
        if step not in (0,) and not (step >= total - decay
                                    and muon_lr == adamw_lr == 0):
            assert muon_lr / adamw_lr == pytest.approx(
                muon_peak / adamw_peak, rel=1e-12,
            )


def test_wsd_zero_zero_defaults_means_constant_peak():
    """When the loop reads ``warmup_steps=0, decay_steps=0`` (the
    default for smoke runs that don't set them), the schedule
    returns ``peak_lr`` for every step in the horizon. This is
    the constant-LR fallback the loop depends on for legacy
    configs."""
    peak = 0.01
    for step in [0, 1, 50, 99]:
        lr = wsd_lr(step, peak_lr=peak, warmup_steps=0, decay_steps=0, max_steps=100)
        assert lr == peak, (
            f"warmup=0, decay=0 must hold LR at peak; step={step} "
            f"got lr={lr}"
        )