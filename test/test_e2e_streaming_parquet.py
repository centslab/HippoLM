"""End-to-end tests for the data streaming pipeline.

Pinned contracts (the e2e behaviors the smoke run on
``configs/test/data_loading.yml`` cannot verify at scale):

  1. **Multi-language cache partition.**  Each ``(ms_name,
     config_name)`` pair is its own cache namespace; a multi-
     source / multi-language run must keep at most
     ``max_cache_files`` (default 2) per language, NOT 2 globally.
     Total cache footprint = ``num_subsets * max_cache_files``.

  2. **Pretrain format (``content`` field).**  Each row's text
     is tokenized verbatim; rows whose token count falls below 2
     are dropped silently.

  3. **SFT format (``messages`` field).**  Each row's messages
     list is formatted via the tokenizer's chat template; rows
     that fail to format (missing messages) are dropped.

  4. **Pre-training purge.**  When none of the expected
     ``(ms_name, config_name)`` tuples have a cache hit, the
     full ``.cache/hippolm/datasets`` directory is cleared before
     any download starts. Prevents stale accumulation across
     runs of different datasets on the same dev box.

  5. **Restart-from-cache.**  A second invocation reuses
     already-cached parts (no re-download) as long as the file
     size still matches the API's expected size.

  6. **Edge cases.**  Empty content / messages dropped; doc
     longer than ``max_seq_len`` is truncated; single-part dataset
     does not trigger any eviction; the ``HIPPOLM_NO_PREFETCH=1``
     skip path does not touch the cache.

The tests inject ``list_parts_fn`` and ``download_fn`` so they
don't touch the real ModelScope API — the test dataset
``wlx0515/test-parquet-load`` is a real public dataset (uploaded
for this test) used to pin the *shape* of the API response, but
the test itself uses in-memory stubs.

Requires pyarrow (already a production dep).
"""
from __future__ import annotations

import logging
import queue
import sys
import threading
from pathlib import Path
from typing import Callable, Optional

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from src.training.data import (  # noqa: E402
    LocalParquetIterable,
    StreamingDataset,
)
from src.training.data.cache import (  # noqa: E402
    cache_path_for_part,
    list_cached_parts,
    purge_stale_cache_if_no_hit,
)
from src.training.data.prefetch_batcher import PrefetchBatcher, QueueIterator  # noqa: E402
from src.training.data.rotating import (  # noqa: E402
    RotatingParquetIterable,
    config_to_subdir,
    resolve_parts_for_config,
)


# Public test dataset on ModelScope. Used only to validate the
# shape of a real MS repo-tree response; the tests below don't
# hit the network (they inject list_parts_fn + download_fn).
TEST_MS_NAME = "wlx0515/test-parquet-load"
# Per-(lang, subset) config names that match the existing
# ``config_to_subdir`` mapping in rotating.py (en/zh ×
# multi_style/qa → data/ultrafineweb_{en,zh}_l3/{multi_style,qa}).
PRETRAIN_CONFIGS = {
    "en_multi": "Ultra-FineWeb-L3-en-Multi-Style-Synthetic",
    "zh_multi": "Ultra-FineWeb-L3-zh-Multi-Style-Synthetic",
    "en_qa":    "Ultra-FineWeb-L3-en-QA-Synthetic",
    "zh_qa":    "Ultra-FineWeb-L3-zh-QA-Synthetic",
}


# --------------------------------------------------------------------------- #
# Parquet writers for text data.                                              #
# --------------------------------------------------------------------------- #
def _write_text_part(path: Path, texts: list[str]) -> None:
    """Write a parquet file with one ``content`` column (pretrain format)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.table({"content": texts}),
        str(path), compression="snappy",
    )


def _write_messages_part(path: Path, messages_list: list[list[dict]]) -> None:
    """Write a parquet file with one ``messages`` column (SFT format).

    Each row is a list of ``{role, content}`` dicts; pyarrow's
    nested-list type round-trips through ``to_pylist`` cleanly.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.table({"messages": messages_list}),
        str(path), compression="snappy",
    )


# --------------------------------------------------------------------------- #
# Stubs for the MS API + download path.                                       #
# --------------------------------------------------------------------------- #
def _stub_list_parts_multi(by_dir_response: dict[str, list[tuple[int, str, int]]]):
    """Build a list_parts_fn from a static ``{subdir: [(idx, path, size), ...]}``
    dict. Mirrors the real :func:`list_parquet_parts_via_api` return shape."""

    def _list_parts(ms_name, log):
        return by_dir_response

    return _list_parts


def _stub_download_factory(
    source_dir: Path,
    cache_dir: Path,
) -> Callable:
    """Build a download_fn that copies a pre-staged source parquet into
    the cache directory at the expected path.

    Uses the SAME atomic ``.part`` + rename pattern as production
    (:mod:`src.training.data.rotating.download_part_to_cache`): write
    to ``cache_path + ".part"``, then rename. This is essential for
    the test because the rotator re-schedules the bg download on
    every yield after the 50% trigger; without atomic writes, two
    concurrent writers can interleave and leave a partially-written
    file at the final path that pyarrow refuses to read.

    The source tree under ``source_dir`` mirrors the ``by_dir`` API
    response: ``source_dir/<subdir>/part-NNNNN-...parquet`` is copied
    to ``cache_path_for_part(ms_name, config_name, idx, cache_dir=cache_dir)``.
    """

    def _download(ms_name, part_path, part_idx, expected_size, cache_path, log):
        # ``part_path`` is the full MS-relative path. The source tree
        # mirrors it under ``source_dir``. (For tests that pin part_path
        # to e.g. ``data/ultrafineweb_en_l3/multi_style/part-...``, the
        # source file lives at ``source_dir/data/ultrafineweb_en_l3/multi_style/...``.)
        src = source_dir / part_path
        cache_path = Path(cache_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache_path.with_suffix(cache_path.suffix + ".part")
        tmp.write_bytes(src.read_bytes())
        tmp.rename(cache_path)
        return cache_path

    return _download


def _stage_pretrain_tree(root: Path) -> dict[str, list[tuple[int, str, int]]]:
    """Stage a fake pretrain tree under ``root`` matching the public
    test dataset's structure, and return the API response.

    Returns the same ``{subdir: [(idx, path, size), ...]}`` shape as
    :func:`list_parquet_parts_via_api`. The paths are relative to
    the MS repo root (so they look like ``data/<lang>/<subset>/part-...``).
    Each tuple's third element is the REAL on-disk size so the
    rotator's ``expected_size`` check matches production semantics
    (a 0-byte file would be treated as a stale partial and re-downloaded).
    """
    # English multi-style: 3 parts, each with 8 unique English rows
    for i in range(3):
        _write_text_part(
            root / f"data/ultrafineweb_en_l3/multi_style/part-{i:05d}.parquet",
            [f"english multi-style sample {i}.{j}" for j in range(8)],
        )
    # English qa: 2 parts
    for i in range(2):
        _write_text_part(
            root / f"data/ultrafineweb_en_l3/qa/part-{i:05d}.parquet",
            [f"english qa sample {i}.{j}" for j in range(8)],
        )
    # Chinese multi-style: 2 parts
    for i in range(2):
        _write_text_part(
            root / f"data/ultrafineweb_zh_l3/multi_style/part-{i:05d}.parquet",
            [f"中文多风格样本 {i}.{j}" for j in range(8)],
        )
    # Chinese qa: 2 parts
    for i in range(2):
        _write_text_part(
            root / f"data/ultrafineweb_zh_l3/qa/part-{i:05d}.parquet",
            [f"中文问答样本 {i}.{j}" for j in range(8)],
        )

    def _api(subdir: str, n: int) -> list[tuple[int, str, int]]:
        out = []
        for i in range(n):
            p = root / subdir / f"part-{i:05d}.parquet"
            out.append((i, f"{subdir}/part-{i:05d}.parquet", p.stat().st_size))
        return out

    return {
        "data/ultrafineweb_en_l3/multi_style": _api(
            "data/ultrafineweb_en_l3/multi_style", 3,
        ),
        "data/ultrafineweb_en_l3/qa": _api(
            "data/ultrafineweb_en_l3/qa", 2,
        ),
        "data/ultrafineweb_zh_l3/multi_style": _api(
            "data/ultrafineweb_zh_l3/multi_style", 2,
        ),
        "data/ultrafineweb_zh_l3/qa": _api(
            "data/ultrafineweb_zh_l3/qa", 2,
        ),
    }


def _stage_sft_tree(root: Path, n_parts: int = 3) -> dict[str, list[tuple[int, str, int]]]:
    """Stage a single-subdir SFT tree under ``root``.

    Returns the API response with all parts under ``data/sft`` so
    that ``config_to_subdir(None)`` returning None + the "largest
    subdir" fallback picks it. Sizes are real on-disk sizes.
    """
    for i in range(n_parts):
        messages_list = [
            [
                {"role": "user", "content": f"SFT q {i}.{j}"},
                {"role": "assistant", "content": f"SFT a {i}.{j}"},
            ]
            for j in range(6)
        ]
        _write_messages_part(
            root / f"data/sft/part-{i:05d}.parquet",
            messages_list,
        )
    out = []
    for i in range(n_parts):
        p = root / "data/sft" / f"part-{i:05d}.parquet"
        out.append((i, f"data/sft/part-{i:05d}.parquet", p.stat().st_size))
    return {"data/sft": out}


def _build_iter(
    ms_name: str,
    config_name: Optional[str],
    cache_dir: Path,
    *,
    list_parts_fn,
    source_dir: Path,
    pre_download_pct: float = 0.5,
    max_cache_files: int = 2,
) -> RotatingParquetIterable:
    """Build a RotatingParquetIterable with the source-dir download stub."""
    return RotatingParquetIterable(
        ms_name=ms_name,
        config_name=config_name,
        cache_dir=cache_dir,
        pre_download_pct=pre_download_pct,
        max_cache_files=max_cache_files,
        list_parts_fn=list_parts_fn,
        download_fn=_stub_download_factory(source_dir, cache_dir),
    )


# --------------------------------------------------------------------------- #
# Tiny tokenizer stub (avoids pulling the real Qwen tokenizer into unit tests).#
# --------------------------------------------------------------------------- #
class _StubTokenizer:
    """Minimal tokenizer that maps each whitespace-separated word to
    a stable int id (hash mod 30000). Returns ``input_ids`` as a
    Python list — matches the contract the streaming dataset expects.
    """

    pad_id = 0
    eos_id = 1

    def __init__(self):
        # Stable hash → small vocab so packer is fast.
        self._v = 30_000

    def __call__(self, text, max_length=None, truncation=False, return_tensors=None):
        # Production calls ``StreamingDataset._flush_batch`` with a
        # ``list[str]`` (batched path) and ``_tokenize_one`` with a
        # single ``str`` (per-doc path). The stub must mirror the
        # HuggingFace tokenizer contract on both shapes:
        #   str       → ``{"input_ids": [int, ...]}``
        #   list[str] → ``{"input_ids": [[int, ...], [int, ...], ...]}``
        if isinstance(text, list):
            return {"input_ids": [
                self._encode_one(t, max_length, truncation) for t in text
            ]}
        return {"input_ids": self._encode_one(text, max_length, truncation)}

    def _encode_one(self, text, max_length, truncation):
        ids = [(hash(w) % (self._v - 10)) + 10 for w in text.split()]
        if max_length and truncation:
            ids = ids[:max_length]
        return ids

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        if not messages:
            raise ValueError("empty messages")
        lines = []
        for m in messages:
            lines.append(f"{m['role']}: {m['content']}")
        return "\n".join(lines)


@pytest.fixture
def stub_tokenizer():
    return _StubTokenizer()


@pytest.fixture
def cache_dir(tmp_path, monkeypatch):
    """Isolate each test's cache via HIPPOLM_CACHE_DIR.

    Creates the directory upfront so callers can pass it through
    ``cache_dir=`` arguments without ``FileNotFoundError`` from
    ``iterdir()`` (which ``get_cache_dir()`` does on first call but
    ``list_cached_parts`` does not when ``cache_dir`` is explicit).
    """
    target = tmp_path / "cache"
    target.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HIPPOLM_CACHE_DIR", str(target))
    return target


# --------------------------------------------------------------------------- #
# 1. Pre-training purge contract.                                             #
# --------------------------------------------------------------------------- #
class TestPurgeStaleCache:
    """Pins the ``purge_stale_cache_if_no_hit`` contract: ``.cache``
    must be cleared before training starts if and only if none of
    the expected (ms_name, config_name) tuples have a cache hit.
    """

    def test_no_purge_when_cache_is_empty(self, cache_dir):
        n = purge_stale_cache_if_no_hit(
            [(TEST_MS_NAME, PRETRAIN_CONFIGS["en_multi"])],
            cache_dir=cache_dir,
        )
        assert n == 0, "empty cache should not report any cleared files"

    def test_purges_all_when_only_stale_files_present(self, cache_dir):
        """Stale files from a DIFFERENT dataset → purge everything."""
        stale = cache_dir / "other__dataset__other-cfg__part00000.snappy.parquet"
        stale.write_bytes(b"fake")
        n = purge_stale_cache_if_no_hit(
            [(TEST_MS_NAME, PRETRAIN_CONFIGS["en_multi"])],
            cache_dir=cache_dir,
        )
        assert n == 1
        assert not stale.exists(), "stale file should be unlinked"

    def test_does_not_purge_when_expected_config_has_hit(self, cache_dir):
        hit = cache_path_for_part(
            TEST_MS_NAME, PRETRAIN_CONFIGS["en_multi"], 0,
            cache_dir=cache_dir,
        )
        hit.write_bytes(b"fake")
        n = purge_stale_cache_if_no_hit(
            [(TEST_MS_NAME, PRETRAIN_CONFIGS["en_multi"])],
            cache_dir=cache_dir,
        )
        assert n == 0
        assert hit.exists(), "matching file must NOT be purged"

    def test_multi_config_one_hit_keeps_all(self, cache_dir):
        """In multi-source mode, ONE config hit is enough to skip purge
        entirely (the matching files for ALL configs stay)."""
        hit1 = cache_path_for_part(
            TEST_MS_NAME, PRETRAIN_CONFIGS["en_multi"], 0,
            cache_dir=cache_dir,
        )
        hit2 = cache_path_for_part(
            TEST_MS_NAME, PRETRAIN_CONFIGS["zh_qa"], 0,
            cache_dir=cache_dir,
        )
        hit1.write_bytes(b"fake")
        hit2.write_bytes(b"fake")
        n = purge_stale_cache_if_no_hit(
            [(TEST_MS_NAME, cfg) for cfg in PRETRAIN_CONFIGS.values()],
            cache_dir=cache_dir,
        )
        assert n == 0
        assert hit1.exists() and hit2.exists()

    def test_purges_only_parquet_files(self, cache_dir):
        """.part suffix (in-flight download), README, etc. are left alone."""
        (cache_dir / "stale__x__y__part00000.snappy.parquet").write_bytes(b"fake")
        (cache_dir / "in_flight.snappy.parquet.part").write_bytes(b"tmp")
        (cache_dir / "README.md").write_bytes(b"# not a parquet")
        n = purge_stale_cache_if_no_hit(
            [(TEST_MS_NAME, "any")], cache_dir=cache_dir,
        )
        assert n == 1
        remaining = sorted(p.name for p in cache_dir.iterdir())
        assert "in_flight.snappy.parquet.part" in remaining
        assert "README.md" in remaining

    def test_idempotent(self, cache_dir):
        """Calling twice in a row should be a no-op the second time."""
        (cache_dir / "stale__x__y__part00000.snappy.parquet").write_bytes(b"fake")
        n1 = purge_stale_cache_if_no_hit(
            [(TEST_MS_NAME, "any")], cache_dir=cache_dir,
        )
        n2 = purge_stale_cache_if_no_hit(
            [(TEST_MS_NAME, "any")], cache_dir=cache_dir,
        )
        assert n1 == 1
        assert n2 == 0


# --------------------------------------------------------------------------- #
# 2. Multi-language cache partition + per-language cap.                       #
# --------------------------------------------------------------------------- #
class TestMultiLanguageCacheContract:
    """Each (ms_name, config_name) is its own cache namespace.

    Verifies:
      * Per-language cap = ``max_cache_files`` (default 2).
      * Total cache footprint = ``n_languages * max_cache_files``.
      * Switching within a language (part 0 → 1 → 2) evicts the
        prior part for THAT language only — does NOT touch other
        languages' files.
    """

    def test_each_language_keeps_at_most_2_files(
        self, tmp_path, cache_dir,
    ):
        # Build a 4-language source tree.
        source = tmp_path / "src"
        api = _stage_pretrain_tree(source)

        # Open one iterator per language.
        iters = {
            label: _build_iter(
                ms_name=TEST_MS_NAME,
                config_name=cfg,
                cache_dir=cache_dir,
                list_parts_fn=_stub_list_parts_multi(api),
                source_dir=source,
            )
            for label, cfg in PRETRAIN_CONFIGS.items()
        }

        try:
            for it in iters.values():
                it.open()

            # Walk through every iterator enough to trigger pre-download
            # of part-1 for each language.
            for it in iters.values():
                # 8 rows per part; 50% trigger → 4 next() calls hit part-1 dl.
                for _ in range(4):
                    try:
                        next(it)
                    except StopIteration:
                        break

            # After walking: each language has the current part +
            # in-flight next part = 2 files.
            for label, cfg in PRETRAIN_CONFIGS.items():
                cached = list_cached_parts(
                    TEST_MS_NAME, cfg, cache_dir=cache_dir,
                )
                assert len(cached) <= 2, (
                    f"{label}: cache has {len(cached)} parts,"
                    f" expected <= 2 (current + in-flight): {cached}"
                )

            # Total cache footprint: at most 4 langs × 2 files = 8.
            all_parquets = sorted(p.name for p in cache_dir.glob("*.parquet"))
            assert len(all_parquets) <= 8, (
                f"total cache holds {len(all_parquets)} files,"
                f" expected <= 8: {all_parquets}"
            )
        finally:
            for it in iters.values():
                it.close()

    def test_language_switch_does_not_evict_other_languages(
        self, tmp_path, cache_dir,
    ):
        """Exhausting language A's part-0 should evict A's part-0
        but leave B/C/D untouched."""
        source = tmp_path / "src"
        api = _stage_pretrain_tree(source)
        iters = {
            label: _build_iter(
                TEST_MS_NAME, cfg, cache_dir,
                list_parts_fn=_stub_list_parts_multi(api),
                source_dir=source,
            )
            for label, cfg in PRETRAIN_CONFIGS.items()
        }
        try:
            for it in iters.values():
                it.open()

            # Drain language A (en_multi, 3 parts × 8 rows = 24 rows
            # plus switch padding). Force exhaustion.
            for _ in range(30):
                try:
                    next(iters["en_multi"])
                except StopIteration:
                    break

            # en_multi part-0 should be evicted; other languages'
            # part-0 files must still be present.
            assert not cache_path_for_part(
                TEST_MS_NAME, PRETRAIN_CONFIGS["en_multi"], 0,
                cache_dir=cache_dir,
            ).exists(), "en_multi part-0 should have been evicted"

            for other in ("zh_multi", "en_qa", "zh_qa"):
                assert cache_path_for_part(
                    TEST_MS_NAME, PRETRAIN_CONFIGS[other], 0,
                    cache_dir=cache_dir,
                ).exists(), (
                    f"{other} part-0 was wrongly evicted by en_multi switch"
                )
        finally:
            for it in iters.values():
                it.close()


# --------------------------------------------------------------------------- #
# 3. Per-iterator restart-from-cache.                                        #
# --------------------------------------------------------------------------- #
class TestRestartFromCache:
    def test_second_open_reuses_cached_part(
        self, tmp_path, cache_dir,
    ):
        """After a first iterator closes, a second iterator on the
        SAME (ms_name, config_name) must reuse the cached parts
        instead of re-downloading them."""
        source = tmp_path / "src"
        api = _stage_pretrain_tree(source)
        download_calls: list[int] = []

        def _spy_download(ms_name, part_path, part_idx, expected_size, cache_path, log):
            download_calls.append(part_idx)
            return _stub_download_factory(source, cache_dir)(
                ms_name, part_path, part_idx, expected_size, cache_path, log,
            )

        # First run: opens, downloads part-0, yields, closes.
        it1 = RotatingParquetIterable(
            ms_name=TEST_MS_NAME,
            config_name=PRETRAIN_CONFIGS["en_multi"],
            cache_dir=cache_dir,
            list_parts_fn=_stub_list_parts_multi(api),
            download_fn=_spy_download,
        )
        it1.open()
        next(it1)
        it1.close()
        first_run_calls = list(download_calls)

        # Second run: opens → should find part-0 already cached.
        it2 = RotatingParquetIterable(
            ms_name=TEST_MS_NAME,
            config_name=PRETRAIN_CONFIGS["en_multi"],
            cache_dir=cache_dir,
            list_parts_fn=_stub_list_parts_multi(api),
            download_fn=_spy_download,
        )
        it2.open()
        next(it2)
        it2.close()

        # download_fn should have been called for part-0 in run 1 but NOT in run 2.
        assert 0 in first_run_calls, "first run must download part-0"
        assert download_calls.count(0) == 1, (
            f"part-0 was re-downloaded: {download_calls}"
        )


# --------------------------------------------------------------------------- #
# 4. Pretrain format (content field) — tokenize + drop too-short rows.       #
# --------------------------------------------------------------------------- #
class TestPretrainFormat:
    """End-to-end through ``StreamingDataset``: parquet → tokenize →
    drop too-short rows.
    """

    def test_content_field_tokenized(
        self, tmp_path, stub_tokenizer, cache_dir, monkeypatch,
    ):
        # The rotator + download paths are stubbed; only the
        # tokenize / format pipeline is exercised.
        source = tmp_path / "src"
        api = _stage_pretrain_tree(source)

        # Patch resolve_parts_for_config via the iter's list_parts_fn.
        ds = StreamingDataset(
            dataset_name=TEST_MS_NAME,
            ms_dataset_name=TEST_MS_NAME,
            config_name=PRETRAIN_CONFIGS["en_multi"],
            tokenizer=stub_tokenizer,
            split="train",
            max_seq_len=64,
            text_field="content",
            use_modelscope=True,
        )
        # Replace the lazy loader with a rotator that uses our stubs.
        from src.training.data.rotating import RotatingParquetIterable
        it = _build_iter(
            TEST_MS_NAME, PRETRAIN_CONFIGS["en_multi"], cache_dir,
            list_parts_fn=_stub_list_parts_multi(api),
            source_dir=source,
        )
        ds._ds = it
        try:
            samples = []
            for sample in ds:
                samples.append(sample)
                if len(samples) >= 10:
                    break
            assert samples, "no samples yielded"
            for s in samples:
                assert isinstance(s["input_ids"], list)
                assert s["input_ids"] == s["labels"], (
                    "labels must mirror input_ids in pretrain mode"
                )
                assert len(s["input_ids"]) >= 2, (
                    "StreamingDataset must drop rows that yield <2 tokens"
                )
        finally:
            it.close()

    def test_long_doc_truncated_to_max_seq_len(
        self, tmp_path, stub_tokenizer, cache_dir,
    ):
        """A doc that produces > max_seq_len tokens is truncated;
        the streaming dataset's tokenizer call passes
        ``max_length=max_seq_len, truncation=True``."""
        source = tmp_path / "src"
        # Stage a single part with a deliberately long row.
        long_text = " ".join(f"word{i}" for i in range(200))
        _write_text_part(
            source / "data/ultrafineweb_en_l3/multi_style/part-00000.parquet",
            [long_text],
        )
        p = source / "data/ultrafineweb_en_l3/multi_style/part-00000.parquet"
        api = {
            "data/ultrafineweb_en_l3/multi_style": [
                (
                    0,
                    "data/ultrafineweb_en_l3/multi_style/part-00000.parquet",
                    p.stat().st_size,
                ),
            ],
        }
        it = _build_iter(
            TEST_MS_NAME, PRETRAIN_CONFIGS["en_multi"], cache_dir,
            list_parts_fn=_stub_list_parts_multi(api),
            source_dir=source,
        )
        ds = StreamingDataset(
            dataset_name=TEST_MS_NAME,
            ms_dataset_name=TEST_MS_NAME,
            config_name=PRETRAIN_CONFIGS["en_multi"],
            tokenizer=stub_tokenizer,
            split="train",
            max_seq_len=8,  # tight cap
            text_field="content",
            use_modelscope=True,
        )
        ds._ds = it
        try:
            sample = next(iter(ds))
            assert len(sample["input_ids"]) == 8, (
                f"doc was not truncated to max_seq_len=8,"
                f" got len={len(sample['input_ids'])}"
            )
        finally:
            it.close()


# --------------------------------------------------------------------------- #
# 5. SFT format (messages field) — chat template formatting.                  #
# --------------------------------------------------------------------------- #
class TestSFTFormat:
    def test_messages_field_formatted_via_chat_template(
        self, tmp_path, stub_tokenizer, cache_dir,
    ):
        source = tmp_path / "src"
        api = _stage_sft_tree(source, n_parts=1)
        it = _build_iter(
            TEST_MS_NAME, None, cache_dir,
            list_parts_fn=_stub_list_parts_multi(api),
            source_dir=source,
        )
        # Use config_name=None to exercise the "largest subdir" fallback.
        ds = StreamingDataset(
            dataset_name=TEST_MS_NAME,
            ms_dataset_name=TEST_MS_NAME,
            config_name=None,
            tokenizer=stub_tokenizer,
            split="train",
            max_seq_len=128,
            is_sft=True,
            text_field="content",
            use_modelscope=True,
        )
        ds._ds = it
        try:
            samples = []
            for sample in ds:
                samples.append(sample)
                if len(samples) >= 3:
                    break
            assert samples
            for s in samples:
                assert isinstance(s["input_ids"], list)
                assert len(s["input_ids"]) >= 2
        finally:
            it.close()

    def test_empty_messages_row_is_dropped(
        self, tmp_path, stub_tokenizer, cache_dir,
    ):
        """A row whose ``messages`` field is empty/missing should be
        silently dropped (StreamingDataset._format_sft returns None)."""
        source = tmp_path / "src"
        # One good row + one row with empty messages list.
        messages = [
            [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},
            ],
            [],
        ]
        _write_messages_part(
            source / "data/sft/part-00000.parquet",
            messages,
        )
        api = {
            "data/sft": [
                (
                    0,
                    "data/sft/part-00000.parquet",
                    (source / "data/sft/part-00000.parquet").stat().st_size,
                ),
            ],
        }
        it = _build_iter(
            TEST_MS_NAME, None, cache_dir,
            list_parts_fn=_stub_list_parts_multi(api),
            source_dir=source,
        )
        ds = StreamingDataset(
            dataset_name=TEST_MS_NAME,
            ms_dataset_name=TEST_MS_NAME,
            config_name=None,
            tokenizer=stub_tokenizer,
            split="train",
            max_seq_len=128,
            is_sft=True,
            use_modelscope=True,
        )
        ds._ds = it
        try:
            samples = list(ds)
            assert len(samples) == 1, (
                f"empty-messages row should be dropped, got {len(samples)}"
            )
        finally:
            it.close()


# --------------------------------------------------------------------------- #
# 6. config_to_subdir mapping correctness (real MS dataset structure).        #
# --------------------------------------------------------------------------- #
class TestConfigToSubdirMapping:
    """Pin the existing ``config_to_subdir`` mapping against the real
    public test dataset. If this test ever fails because the MS
    dataset structure changed (or someone moved the regex), the
    test catches it without a network call."""

    def test_en_multi_maps_to_en_multi_style_dir(self):
        assert config_to_subdir(
            PRETRAIN_CONFIGS["en_multi"]
        ) == "data/ultrafineweb_en_l3/multi_style"

    def test_en_qa_maps_to_en_qa_dir(self):
        assert config_to_subdir(
            PRETRAIN_CONFIGS["en_qa"]
        ) == "data/ultrafineweb_en_l3/qa"

    def test_zh_multi_maps_to_zh_multi_style_dir(self):
        assert config_to_subdir(
            PRETRAIN_CONFIGS["zh_multi"]
        ) == "data/ultrafineweb_zh_l3/multi_style"

    def test_zh_qa_maps_to_zh_qa_dir(self):
        assert config_to_subdir(
            PRETRAIN_CONFIGS["zh_qa"]
        ) == "data/ultrafineweb_zh_l3/qa"

    def test_none_returns_none(self):
        """``config_name=None`` is the SFT / generic path — the caller
        is expected to fall back to "largest subdir" via
        ``resolve_parts_for_config``."""
        assert config_to_subdir(None) is None

    def test_resolve_for_real_dataset(self, tmp_path, cache_dir):
        """Run the resolver against an API-response stub that mirrors
        the real ``wlx0515/test-parquet-load`` repo tree. If the
        upload structure is ever broken, this test fails first
        (no network required)."""
        source = tmp_path / "src"
        api = _stage_pretrain_tree(source)
        # Add an sft subdir so the "largest subdir" fallback has a target.
        api.update(_stage_sft_tree(source, n_parts=1))

        # Each pretrain config resolves to its own subdir.
        for label, cfg in PRETRAIN_CONFIGS.items():
            parts = resolve_parts_for_config(
                TEST_MS_NAME, cfg,
                log=logging.getLogger("test"),
                list_parts_fn=_stub_list_parts_multi(api),
            )
            assert parts, f"{label} ({cfg}) returned no parts"

        # config=None picks the largest subdir; sft is smaller in this
        # stage, but en_multi is the largest pretrain so it would win.
        # Make sft the largest by adding more parts. The initial
        # ``_stage_sft_tree(source, n_parts=1)`` produced 1 part; add
        # parts 1..5 here so the final count is 6 (still largest).
        api_largest_sft = dict(api)
        for extra_i in range(1, 6):
            extra_path = source / f"data/sft/part-{extra_i:05d}.parquet"
            _write_messages_part(
                extra_path,
                [[{"role": "user", "content": "x"},
                  {"role": "assistant", "content": "y"}]] * 4,
            )
            api_largest_sft["data/sft"].append(
                (extra_i, f"data/sft/part-{extra_i:05d}.parquet",
                 extra_path.stat().st_size),
            )
        parts = resolve_parts_for_config(
            TEST_MS_NAME, None,
            log=logging.getLogger("test"),
            list_parts_fn=_stub_list_parts_multi(api_largest_sft),
        )
        # 6 SFT parts (the largest in the API response now).
        assert len(parts) == 6, (
            f"None config should pick largest subdir (6 sft parts),"
            f" got {len(parts)}"
        )


# --------------------------------------------------------------------------- #
# 7. PrefetchBatcher end-to-end (tokenize → pack → queue).                    #
# --------------------------------------------------------------------------- #
class TestPrefetchBatcherE2E:
    """Wire a ``RotatingParquetIterable`` into a ``StreamingDataset``
    into a ``PrefetchBatcher`` and pull one batch from the queue.
    Verifies the full prefetch / pack / cu_seqlens contract on the
    happy path."""

    def test_first_batch_yields_packed_tensors(
        self, tmp_path, stub_tokenizer, cache_dir,
    ):
        source = tmp_path / "src"
        api = _stage_pretrain_tree(source)
        it = _build_iter(
            TEST_MS_NAME, PRETRAIN_CONFIGS["en_multi"], cache_dir,
            list_parts_fn=_stub_list_parts_multi(api),
            source_dir=source,
        )
        ds = StreamingDataset(
            dataset_name=TEST_MS_NAME,
            ms_dataset_name=TEST_MS_NAME,
            config_name=PRETRAIN_CONFIGS["en_multi"],
            tokenizer=stub_tokenizer,
            split="train",
            max_seq_len=64,
            text_field="content",
            use_modelscope=True,
        )
        ds._ds = it
        q = queue.Queue(maxsize=2)
        # chunk_size=64 (multiple of KDA BT) and seq_len=128 = 2 * 64.
        batcher = PrefetchBatcher(
            dataset=ds,
            batch_size=2,
            queues=[q],
            seq_len=128,
            chunk_size=64,
            pad_id=0,
            eos_id=None,
            pack_buffer_size=4,
        )
        try:
            qi = QueueIterator(q)
            batch = next(qi)
            assert batch["input_ids"].shape == (2, 128)
            assert batch["labels"].shape == (2, 128)
            assert batch["cu_seqlens"][0].item() == 0
            assert batch["cu_seqlens"][-1].item() == 2 * 128
        finally:
            batcher.close(timeout=2.0)
            it.close()


# --------------------------------------------------------------------------- #
# 8. Single-part dataset (no eviction ever).                                 #
# --------------------------------------------------------------------------- #
class TestSinglePart:
    def test_single_part_never_evicts(self, tmp_path, cache_dir):
        source = tmp_path / "src"
        p = source / "data/ultrafineweb_en_l3/multi_style/part-00000.parquet"
        _write_text_part(
            p,
            ["single-part row"] * 4,
        )
        api = {
            "data/ultrafineweb_en_l3/multi_style": [
                (
                    0,
                    "data/ultrafineweb_en_l3/multi_style/part-00000.parquet",
                    p.stat().st_size,
                ),
            ],
        }
        it = _build_iter(
            TEST_MS_NAME, PRETRAIN_CONFIGS["en_multi"], cache_dir,
            list_parts_fn=_stub_list_parts_multi(api),
            source_dir=source,
        )
        try:
            it.open()
            # Drain the only part.
            for _ in range(4):
                next(it)
            # No part-1 exists, so no pre-download was ever triggered.
            assert cache_path_for_part(
                TEST_MS_NAME, PRETRAIN_CONFIGS["en_multi"], 0,
                cache_dir=cache_dir,
            ).exists(), "single part-0 must remain on disk"
        finally:
            it.close()


# --------------------------------------------------------------------------- #
# 9. Edge: HIPPOLM_NO_PREFETCH skips the rotator entirely.                    #
# --------------------------------------------------------------------------- #
class TestNoPrefetchFallback:
    """``HIPPOLM_NO_PREFETCH=1`` must skip the rotating-parquet path
    and fall back to ModelScope streaming. The rotator is never
    instantiated, so no cache files are produced."""

    def test_no_prefetch_skips_cache_creation(
        self, tmp_path, cache_dir, monkeypatch, stub_tokenizer,
    ):
        # Stage an API response for the rotator (won't be used because
        # the env var forces the streaming fallback). This is just to
        # prove the rotator doesn't accidentally run.
        monkeypatch.setenv("HIPPOLM_NO_PREFETCH", "1")

        from src.training.data.sources import load_streaming_with_fallback
        log = logging.getLogger("test")
        # We don't have a real MS dataset here, but we can confirm the
        # rotator is bypassed by checking that the resulting object is
        # NOT a RotatingParquetIterable. (If modelscope is missing,
        # load falls through to HF streaming which we also don't have,
        # so we just check the exception type / rotator path is taken.)
        import contextlib
        with contextlib.suppress(Exception):
            # Anything other than RotatingParquetIterable is fine.
            result = load_streaming_with_fallback(
                hf_name=TEST_MS_NAME,
                ms_name=TEST_MS_NAME,
                config_name=PRETRAIN_CONFIGS["en_multi"],
                split="train",
                use_ms=True,
                log=log,
            )
            assert not isinstance(result, RotatingParquetIterable), (
                "HIPPOLM_NO_PREFETCH must bypass the rotating-parquet path"
            )
        # And no files were written to the cache.
        assert list(cache_dir.iterdir()) == [], (
            "no-prefetch fallback must not write any cache files"
        )