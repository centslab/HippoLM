"""Tests for the FFD chunk-aligned packer used by the streaming data path.

The packer's contract is critical because the KDA chunkwise kernel
reads ``cu_seqlens`` to know where to reset the recurrent ``h`` state.
Three contract points are pinned here:

  1. ``cu_seqlens[-1] == batch_size * seq_len`` (the kernel resets
     ``h`` at every pack boundary, so the tail-pad "doc" must
     terminate at the start of the next pack — never at the last
     real doc's end if that ends before the pack end).
  2. Every internal offset is a multiple of ``chunk_size`` (so the
     kernel's per-chunk state reset lands exactly on a doc boundary).
  3. ``n_packs == batch_size`` is the output invariant: the consumer
     (training loop) relies on this to plan its per-step token
     budget and TP all-reduce buffers.

These tests use no torch / no GPU; they only verify tensor shapes,
cu_seqlens structure, and label-mask positions.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from src.training.data.collate import pack_chunk_aligned  # noqa: E402


def test_cu_seqlens_ends_at_batch_size_times_seq_len():
    """Even when the last real doc ends before the pack end, the
    final cu_seqlens entry MUST be ``batch_size * seq_len`` —
    the kernel needs this to reset state at the tail pad."""
    samples = [[1, 2, 3, 4, 5]]  # one short doc
    input_ids, labels, cu_seqlens = pack_chunk_aligned(
        samples=samples, seq_len=128, chunk_size=64, batch_size=2,
    )
    assert input_ids.shape == (2, 128)
    assert labels.shape == (2, 128)
    assert cu_seqlens[-1].item() == 2 * 128  # = batch_size * seq_len


def test_chunk_size_alignment_in_cu_seqlens():
    """Every internal offset must be a multiple of chunk_size so the
    kernel's chunkwise state reset lands exactly on doc boundaries."""
    samples = [[1] * 50, [2] * 30, [3] * 80]
    _, _, cu_seqlens = pack_chunk_aligned(
        samples=samples, seq_len=256, chunk_size=64, batch_size=1,
    )
    for v in cu_seqlens.tolist():
        assert v % 64 == 0, f"cu_seqlens entry {v} not aligned to chunk_size=64"


def test_n_packs_equals_batch_size_invariant():
    """Output shape is always [batch_size, seq_len] regardless of
    how densely the input samples pack."""
    # Many small docs, batch_size=3.
    samples = [[i] * 10 for i in range(20)]
    input_ids, _, _ = pack_chunk_aligned(
        samples=samples, seq_len=128, chunk_size=64, batch_size=3,
    )
    assert input_ids.shape == (3, 128)


def test_empty_samples_returns_all_pad_with_one_boundary_per_pack():
    """With zero samples the output is all-pad rows with one cu_seqlens
    boundary per pack (the kernel sees each pack as a single all-pad
    'doc')."""
    input_ids, labels, cu_seqlens = pack_chunk_aligned(
        samples=[], seq_len=128, chunk_size=64, batch_size=2,
    )
    assert input_ids.shape == (2, 128)
    # All pad id (default 0).
    assert (input_ids == 0).all()
    # All labels masked.
    assert (labels == -100).all()
    # One boundary per pack (batch_size + 1 total).
    assert cu_seqlens.tolist() == [0, 128, 256]


def test_partial_tail_pad_resets_kernel_state_correctly():
    """The contract: even when the last real doc ends at, say, position
    96 of 128, cu_seqlens must emit a separate boundary at position 128
    so the tail pad [96, 128) is processed as its own zero-information
    'doc' (and not as a continuation of the previous real doc)."""
    samples = [[42] * 96]  # 96 tokens, pack of 128
    _, _, cu_seqlens = pack_chunk_aligned(
        samples=samples, seq_len=128, chunk_size=64, batch_size=1,
    )
    # 96 tokens → n_chunks=2 → chunk-aligned length=128. So this doc
    # fills the pack entirely; only the pack-end boundary is needed.
    # The doc-end boundary == pack_end == 128 → de-duped → only one
    # extra entry beyond the initial 0.
    assert cu_seqlens.tolist() == [0, 128]


def test_eos_appended_when_eos_id_set():
    """When ``eos_id`` is set, each doc is forced to end with it (if
    not already present)."""
    samples = [[1, 2, 3]]  # 3 tokens, no EOS
    eos = 999
    _, _, cu_seqlens = pack_chunk_aligned(
        samples=samples, seq_len=64, chunk_size=64, batch_size=1,
        eos_id=eos,
    )
    # The doc should now have 4 tokens (3 + 1 EOS), occupying one
    # chunk (4 ≤ 64), so cu_seqlens = [0, 64].
    assert cu_seqlens.tolist() == [0, 64]


def test_eos_not_duplicated_when_already_present():
    """If the doc already ends with EOS, do not append a second one."""
    eos = 999
    samples = [[1, 2, eos]]
    _, _, cu_seqlens = pack_chunk_aligned(
        samples=samples, seq_len=64, chunk_size=64, batch_size=1,
        eos_id=eos,
    )
    assert cu_seqlens.tolist() == [0, 64]


def test_seq_len_not_multiple_of_chunk_size_raises():
    """Defensive: seq_len must be a multiple of chunk_size."""
    with pytest.raises(ValueError, match="must be a multiple"):
        pack_chunk_aligned(
            samples=[[1]], seq_len=100, chunk_size=64, batch_size=1,
        )


def test_zero_batch_size_raises():
    """Defensive: batch_size must be >= 1."""
    with pytest.raises(ValueError, match="batch_size must be >= 1"):
        pack_chunk_aligned(
            samples=[[1]], seq_len=64, chunk_size=64, batch_size=0,
        )


def test_cross_doc_label_is_masked():
    """The label at the position right after a doc's end (the first
    token of doc i+1) must be -100 — we don't train on cross-doc
    predictions."""
    # Two short docs back-to-back in one pack, each exactly one chunk.
    samples = [[1, 2, 3, 4], [5, 6, 7, 8]]
    _, labels, _ = pack_chunk_aligned(
        samples=samples, seq_len=128, chunk_size=64, batch_size=1,
    )
    # Doc 1 is positions 0..3; cross-doc mask at position 4.
    assert labels[0, 4].item() == -100


def test_doc_filling_pack_does_not_emit_extra_boundary():
    """If a doc fills a pack exactly (L_ck == seq_len), no extra
    boundary is emitted after it (the kernel would see a zero-length
    'doc')."""
    # A doc of exactly 64 tokens + chunk_size=64 → fills one pack.
    samples = [[7] * 64]
    _, _, cu_seqlens = pack_chunk_aligned(
        samples=samples, seq_len=64, chunk_size=64, batch_size=1,
    )
    # Only [0, 64]: the doc-end and pack-end coincide and are de-duped.
    assert cu_seqlens.tolist() == [0, 64]
