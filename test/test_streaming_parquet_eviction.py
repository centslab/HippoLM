"""Tests for the multi-part parquet iterator's cache-eviction contract.

The user's specific concern: when ``RotatingParquetIterable`` walks
through multiple ``.parquet`` parts in sequence, the just-exhausted
part should be evicted from the local cache so the disk usage stays
bounded by ``max_cache_files``.

These tests inject ``list_parts_fn`` and ``download_fn`` so they
don't touch the network. Each "part" is a tiny valid parquet file
written by pyarrow (4 rows of one int column — enough for the
iterator to think the part is non-empty and trigger pre-download).

Requires pyarrow (already imported by the production code).
"""
from __future__ import annotations

import sys
import threading
from pathlib import Path
from typing import Optional

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from src.training.data.rotating import RotatingParquetIterable  # noqa: E402


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
def _write_tiny_parquet(path: Path, n_rows: int = 4) -> None:
    """Write a tiny valid parquet file with one int column.

    4 rows is enough for the iterator's 50% pre-download trigger to
    fire after 2 samples, and small enough that the test is fast.
    """
    table = pa.table({"idx": list(range(n_rows))})
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, str(path))


def _stub_list_parts(n_parts: int, size_bytes: int = 0):
    """Return a list_parts_fn that yields ``n_parts`` synthetic parts.

    ``size_bytes=0`` makes the iterator's "expected size" check a no-op
    (``if part_size and ...``), so a pre-existing cached file of any
    size is treated as a valid cache hit.
    """

    def _list_parts(ms_name, log):
        # Single subdir; ``(part_index, full_path, size_bytes)`` per part.
        return {
            "data/fake": [
                (i, f"part-{i:05d}.parquet", size_bytes) for i in range(n_parts)
            ]
        }

    return _list_parts


def _stub_download(cache_dir: Path):
    """Return a download_fn that writes a tiny parquet to the cache path."""

    def _download(ms_name, part_path, part_idx, expected_size, cache_path, log):
        _write_tiny_parquet(Path(cache_path))
        return cache_path

    return _download


def _make_iter(
    cache_dir: Path,
    n_parts: int = 3,
    max_cache_files: int = 2,
) -> RotatingParquetIterable:
    """Build an iterator with all the network-touching knobs stubbed."""
    return RotatingParquetIterable(
        ms_name="fake/dataset",
        config_name="fake-config",
        cache_dir=cache_dir,
        max_cache_files=max_cache_files,
        pre_download_pct=0.5,
        list_parts_fn=_stub_list_parts(n_parts),
        download_fn=_stub_download(cache_dir),
    )


# --------------------------------------------------------------------------- #
# Tests                                                                       #
# --------------------------------------------------------------------------- #
def test_evicts_exhausted_part_after_switch(tmp_path: Path):
    """After the iterator crosses from part 0 to part 1, the part-0
    file should be unlink()ed from the cache directory.

    Iterator semantics: each part has 4 rows. The 4 successful
    ``next()`` calls return part-0 samples; the 5th call triggers
    ``StopIteration`` from part 0, runs ``_switch_to_next_part``,
    and returns the first sample from part 1. Only after the 5th
    call has part 0 been evicted.
    """
    it = _make_iter(tmp_path, n_parts=3, max_cache_files=2)
    it.open()
    try:
        # Drain part 0 (4 rows) + trigger switch + yield 1 from part 1.
        for _ in range(5):
            next(it)
        # After exhausting part 0, _switch_to_next_part has been called.
        part0_path = tmp_path / (
            "fake__dataset__fake-config__part00000.snappy.parquet"
        )
        assert part0_path.exists() is False, (
            f"part-0 was not evicted: {part0_path} still on disk"
        )
    finally:
        it.close()


def test_cache_holds_at_most_max_cache_files_after_switch(tmp_path: Path):
    """Across any point of iteration, the cache directory holds at most
    ``max_cache_files`` parquet files (the current part plus one in-flight
    next-part download)."""
    it = _make_iter(tmp_path, n_parts=5, max_cache_files=2)
    it.open()
    try:
        # Walk through enough yields to cross several part boundaries.
        # 5 parts * 4 rows + 5 transitions = 25 yields max.
        for _ in range(25):
            try:
                next(it)
            except StopIteration:
                break
        # Now count files in cache.
        parquet_files = list(tmp_path.glob("*.parquet"))
        assert len(parquet_files) <= 2, (
            f"cache holds {len(parquet_files)} files, expected <= 2: "
            f"{[p.name for p in parquet_files]}"
        )
    finally:
        it.close()


def test_eviction_continues_on_unlink_oserror(tmp_path: Path, monkeypatch):
    """If unlink() raises OSError, the iterator must log and continue
    (no crash). This is the OSError-handling contract in
    ``_switch_to_next_part``."""
    it = _make_iter(tmp_path, n_parts=3, max_cache_files=2)
    it.open()

    real_unlink = Path.unlink

    def flaky_unlink(self, *a, **kw):
        if "part00000" in self.name:
            raise OSError("simulated disk full")
        return real_unlink(self, *a, **kw)

    monkeypatch.setattr(Path, "unlink", flaky_unlink)
    try:
        # Should not raise even though part-0 eviction fails. Drain
        # part 0 (4 yields) + trigger switch (5th call) so eviction is
        # attempted.
        for _ in range(5):
            next(it)
        # Part-1 file should now be the current one.
        assert any(
            "part00001" in p.name for p in tmp_path.iterdir()
        ), "part-1 was never downloaded"
    finally:
        it.close()


def test_reuses_cached_part_on_open(tmp_path: Path):
    """If a part is already on disk and matches the expected size, the
    download_fn is NOT called for that part. This is the restart-from-
    cached-policy contract."""
    # Pre-populate cache with part 0. The stub list_parts returns
    # size_bytes=0, so the iterator's ``part_size`` check is skipped
    # and any existing file is treated as a valid cache hit.
    pre_existing = tmp_path / "fake__dataset__fake-config__part00000.snappy.parquet"
    _write_tiny_parquet(pre_existing)

    download_calls: list[int] = []

    def _spy_download(ms_name, part_path, part_idx, expected_size, cache_path, log):
        download_calls.append(part_idx)
        _write_tiny_parquet(Path(cache_path))
        return cache_path

    it = RotatingParquetIterable(
        ms_name="fake/dataset",
        config_name="fake-config",
        cache_dir=tmp_path,
        max_cache_files=2,
        pre_download_pct=0.5,
        list_parts_fn=_stub_list_parts(3, size_bytes=0),
        download_fn=_spy_download,
    )
    it.open()
    try:
        # Drain part 0 — should NOT call download_fn because it was cached.
        for _ in range(5):  # 4 samples + trigger switch
            next(it)
        assert 0 not in download_calls, (
            f"download_fn was called for part 0 despite cache hit: {download_calls}"
        )
        # Part 1 should have been downloaded (background or sync).
        assert 1 in download_calls, (
            f"download_fn was never called for part 1: {download_calls}"
        )
    finally:
        it.close()


def test_single_part_no_eviction(tmp_path: Path):
    """With only one part, no eviction ever happens — the iterator
    stays on part 0 throughout."""
    it = _make_iter(tmp_path, n_parts=1, max_cache_files=2)
    it.open()
    try:
        for _ in range(4):
            next(it)
        # Still on part 0.
        part0 = tmp_path / (
            "fake__dataset__fake-config__part00000.snappy.parquet"
        )
        assert part0.exists(), "part-0 should not have been evicted"
    finally:
        it.close()


def test_close_joins_in_flight_thread(tmp_path: Path):
    """``close()`` must stop any in-flight background download and join
    the thread within the 2-second timeout. Contract: after ``close()``,
    ``_next_dl_thread`` is either ``None`` or a non-alive thread."""
    # Slow download_fn so the background thread is reliably alive when
    # we call close().
    started = threading.Event()
    proceed = threading.Event()

    def _slow_download(ms_name, part_path, part_idx, expected_size, cache_path, log):
        started.set()
        proceed.wait(timeout=5.0)
        _write_tiny_parquet(Path(cache_path))
        return cache_path

    it = RotatingParquetIterable(
        ms_name="fake/dataset",
        config_name="fake-config",
        cache_dir=tmp_path,
        max_cache_files=2,
        pre_download_pct=0.5,
        list_parts_fn=_stub_list_parts(3, size_bytes=0),
        download_fn=_slow_download,
    )
    it.open()
    try:
        # Trigger the background download by consuming half of part 0.
        for _ in range(2):  # 50% of 4 rows
            next(it)
        # The background thread should now be alive (blocked on `proceed`).
        assert started.wait(timeout=5.0), "background download never started"
        assert it._next_dl_thread is not None and it._next_dl_thread.is_alive(), (
            "background thread should be alive while download is in flight"
        )
    finally:
        # Release the download so the thread can finish.
        proceed.set()
        it.close()
    # After close(), either the thread is None or not alive.
    assert it._next_dl_thread is None or not it._next_dl_thread.is_alive()


def test_open_is_idempotent(tmp_path: Path):
    """Calling open() twice is a no-op (the iterator was previously
    documented to guard against double-init)."""
    it = _make_iter(tmp_path, n_parts=2, max_cache_files=2)
    it.open()
    parts_first = list(it._parts)
    it.open()  # second call
    parts_second = list(it._parts)
    assert parts_first == parts_second
    it.close()
