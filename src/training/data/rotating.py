"""Multi-part parquet iterator with 50% pre-download trigger and
bounded cache.

Replaces the legacy single-part :class:`LocalParquetIterable` for
streaming pretrain: instead of downloading just the first part and
streaming the rest, we discover ALL ``part-NNNNN-*.parquet`` files
in the ModelScope dataset via the repo-tree API, then walk through
them in order. To hide the download latency, we start downloading
the next part as soon as the current part is 50% consumed (in a
background thread). To respect a tight local-disk budget, we evict
the just-exhausted part as soon as the iterator switches to the
next one, so the cache never holds more than ``max_cache_files``
(default 2: the current part and the in-flight next part).

The class is iterable (``__iter__`` / ``__next__``). The first
``open()`` call is a synchronous download of part 0 (we need
something to yield from immediately). All subsequent downloads
are fire-and-forget background threads.
"""
from __future__ import annotations

import logging
import re
import shutil
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import Callable, Iterator, Optional

from .cache import cache_path_for_part, list_cached_parts
from .parquet import LocalParquetIterable


# --------------------------------------------------------------------------- #
# MS API discovery.                                                            #
# --------------------------------------------------------------------------- #
_PART_RE = re.compile(
    r"part-(\d{5})-([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-"
    r"[0-9a-f]{4}-[0-9a-f]{12})-c(\d+)\.snappy\.parquet$"
)


def list_parquet_parts_via_api(
    ms_name: str,
    log: logging.Logger,
    *,
    timeout: float = 30.0,
) -> dict[str, list[tuple[int, str, int]]]:
    """List all ``part-NNNNN-*.snappy.parquet`` files in the MS
    dataset via the repo-tree HTTP API.

    Returns a dict ``{subdir_path: [(part_index, full_path, size_bytes), ...]}``
    with each subdir's parts sorted by ``part_index`` ascending.
    The ``subdir_path`` is the leading directory of the file's
    relative path (e.g. ``data/ultrafineweb_en_l3/qa``).

    Raises :class:`RuntimeError` on HTTP / API error; callers may
    fall back to MS streaming on failure.
    """
    import requests
    r = requests.get(
        f"https://www.modelscope.cn/api/v1/datasets/{ms_name}/repo/tree",
        params={"Revision": "master", "Recursive": "True", "PageSize": 2000},
        timeout=timeout,
    )
    if r.status_code != 200:
        raise RuntimeError(
            f"MS repo-tree API returned HTTP {r.status_code}: {r.text[:200]}"
        )
    data = r.json()
    code = data.get("Code")
    if code not in (None, 200, "200", "Success"):
        raise RuntimeError(
            f"MS repo-tree API code={code} message={data.get('Message')!r}"
        )
    files = data.get("Data", {}).get("Files", [])
    by_dir: dict[str, list[tuple[int, str, int]]] = defaultdict(list)
    for f in files:
        if f.get("Type") != "blob":
            continue
        path = f.get("Path", "")
        m = _PART_RE.search(path)
        if not m:
            continue
        part_idx = int(m.group(1))
        subdir = path.rsplit("/", 1)[0]
        by_dir[subdir].append(
            (part_idx, path, int(f.get("Size", 0) or 0)),
        )
    for d in by_dir:
        by_dir[d].sort(key=lambda x: x[0])
    if not by_dir:
        log.warning(
            f"MS repo-tree for {ms_name} returned no part-*.parquet files"
        )
    return dict(by_dir)


def config_to_subdir(config_name: str | None) -> Optional[str]:
    """Map a yml config_name to the corresponding MS subdir.

    Best-effort: only handles the four known Ultra-FineWeb-L3
    subsets (en/zh × qa/multi_style). Returns ``None`` for anything
    else, signalling the caller to fall back to a generic match
    or to streaming.
    """
    if not config_name:
        return None
    cn = config_name.lower()
    if "en" in cn:
        lang_dir = "ultrafineweb_en_l3"
    elif "zh" in cn:
        lang_dir = "ultrafineweb_zh_l3"
    else:
        return None
    if "multi" in cn:
        subset = "multi_style"
    elif "qa" in cn:
        subset = "qa"
    else:
        return None
    return f"data/{lang_dir}/{subset}"


def resolve_parts_for_config(
    ms_name: str,
    config_name: str | None,
    log: logging.Logger,
    *,
    list_parts_fn: Optional[Callable[[str, logging.Logger], dict]] = None,
) -> list[tuple[int, str, int]]:
    """Return the (part_index, file_path, size) list for the given
    config, or raise on failure. ``list_parts_fn`` is an injection
    point for tests; production uses :func:`list_parquet_parts_via_api`.
    """
    list_parts = list_parts_fn or list_parquet_parts_via_api
    by_dir = list_parts(ms_name, log)
    if not by_dir:
        raise RuntimeError(f"no parquet parts found for {ms_name}")
    subdir = config_to_subdir(config_name)
    if subdir and subdir in by_dir:
        return by_dir[subdir]
    # Fall back: if the user gave a config that we can't map,
    # pick the subdir with the most parquets (the en_qa is the
    # canonical 616-file one and the most likely intended target).
    if subdir is None:
        best = max(by_dir.items(), key=lambda kv: len(kv[1]))
        log.warning(
            f"config_name={config_name!r} has no known subdir mapping;"
            f" using {best[0]!r} (largest: {len(best[1])} parts)"
        )
        return best[1]
    # Subdir is known but absent (e.g. an unrecognised subset).
    raise RuntimeError(
        f"no parquet parts under subdir {subdir!r} for {ms_name}"
    )


# --------------------------------------------------------------------------- #
# Per-part download.                                                           #
# --------------------------------------------------------------------------- #
def download_part_to_cache(
    ms_name: str,
    part_path: str,
    part_index: int,
    expected_size: int,
    cache_path: Path,
    log: logging.Logger,
    *,
    timeout: float = 30.0,
) -> Path:
    """Download a single ``part-NNNNN-*.snappy.parquet`` to the
    local cache. Returns ``cache_path`` on success.

    The download is the existing "HEAD → GET stream" pattern from
    :mod:`src.training.data.prefetch`, lifted out of the
    hard-coded "qa/part-00000" assumption and parameterised on
    ``part_path`` (the relative path inside the MS repo) and
    ``part_index`` (used for the cache filename).

    Idempotent: if ``cache_path`` already exists with the expected
    size, the download is skipped.
    """
    import requests
    from .sources import network_exceptions

    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists():
        if expected_size and cache_path.stat().st_size == expected_size:
            log.info(f"reusing cached part: {cache_path}")
            return cache_path
        # Stale or partial; remove and redownload.
        cache_path.unlink()

    base = f"https://www.modelscope.cn/datasets/{ms_name}/resolve/master"
    auth_url = f"{base}/{part_path}"
    log.info(
        f"downloading part {part_index:05d} -> {cache_path}"
        f" ({expected_size / 1024**2:.1f} MB expected)"
    )
    t0 = time.monotonic()
    got = 0
    last_log = t0
    try:
        with requests.get(auth_url, stream=True, timeout=timeout) as r:
            r.raise_for_status()
            tmp = cache_path.with_suffix(cache_path.suffix + ".part")
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(chunk_size=4 * 1024 * 1024):
                    if not chunk:
                        continue
                    f.write(chunk)
                    got += len(chunk)
                    now = time.monotonic()
                    if now - last_log > 10:
                        rate = got / (now - t0) / 1024
                        pct = (
                            f" ({got / expected_size * 100:.1f}%)"
                            if expected_size else ""
                        )
                        log.info(
                            f"  part {part_index:05d}:"
                            f" {got / 1024**2:.1f} MB in"
                            f" {now - t0:.1f}s, {rate:.0f} KB/s{pct}"
                        )
                        last_log = now
            tmp.rename(cache_path)
    except network_exceptions() as e:
        log.warning(
            f"part {part_index:05d} download failed"
            f" ({type(e).__name__}: {e})"
        )
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass
        raise
    log.info(
        f"  part {part_index:05d}: done in {time.monotonic() - t0:.1f}s"
    )
    return cache_path


# --------------------------------------------------------------------------- #
# The rotating iterator.                                                      #
# --------------------------------------------------------------------------- #
class RotatingParquetIterable:
    """Multi-part parquet iterator with 50% pre-download and
    bounded cache.

    Lifecycle::

        it = RotatingParquetIterable(ms_name, config_name, ...)
        it.open()           # synchronous: download part 0
        for sample in it:   # background download of part N+1
            ...             # starts at 50% of part N
        it.close()          # stop in-flight background download

    Internally the iterator wraps a :class:`LocalParquetIterable`
    for the current part; when that part is exhausted, it switches
    to the next downloaded part and evicts the just-exhausted
    file (so the cache directory holds at most
    ``max_cache_files`` part files at any moment).

    The ``list_parts_fn`` and ``download_fn`` parameters are
    injection points for tests; production callers should leave
    them ``None`` and use the defaults (HTTP API + sync download).
    """

    def __init__(
        self,
        ms_name: str,
        config_name: str | None,
        *,
        cache_dir: Path | None = None,
        pre_download_pct: float = 0.5,
        max_cache_files: int = 2,
        list_parts_fn: Optional[Callable] = None,
        download_fn: Optional[Callable] = None,
        log: Optional[logging.Logger] = None,
    ):
        self._ms_name = ms_name
        self._config_name = config_name
        self._cache_dir = cache_dir
        self._pre_download_pct = float(pre_download_pct)
        self._max_cache_files = int(max_cache_files)
        self._log = log or logging.getLogger(__name__)
        # Part list (filled in by ``open()``).
        self._parts: list[tuple[int, str, int]] = []
        # Index of the part currently being iterated.
        self._part_idx: int = -1
        # Active parquet iterable (a fresh ``LocalParquetIterable``
        # is built for each part to keep the iteration state local).
        self._current_iter: Optional[LocalParquetIterable] = None
        # Total rows in the current part (read from pyarrow).
        self._part_total_rows: int = 0
        # Number of samples already yielded from the current part.
        self._samples_yielded: int = 0
        # Path of the cache file for the current part.
        self._current_path: Optional[Path] = None
        # Path of the cache file already downloaded for the next
        # part (or ``None`` if not yet downloaded).
        self._next_path: Optional[Path] = None
        # Background download thread (if any) and the part index
        # it is downloading.
        self._next_dl_thread: Optional[threading.Thread] = None
        self._next_dl_part_idx: Optional[int] = None
        self._stop = threading.Event()
        self._list_parts_fn = list_parts_fn
        self._download_fn = download_fn or download_part_to_cache
        # Whether ``open()`` has been called.
        self._opened = False

    # ----- public API ----- #

    def open(self) -> None:
        """Synchronously download part 0 (we need it before we can
        yield anything). Raises if no parts are discovered or the
        first download fails.
        """
        if self._opened:
            return
        self._parts = resolve_parts_for_config(
            self._ms_name, self._config_name, self._log,
            list_parts_fn=self._list_parts_fn,
        )
        if not self._parts:
            raise RuntimeError(
                f"no parquet parts found for {self._ms_name}"
                f" (config={self._config_name!r})"
            )
        # Reuse any cached parts that are still on disk and match
        # the expected size (e.g. a previous run that exited after
        # part 0 was downloaded). Re-using cached parts avoids a
        # re-download on a restart.
        cached = set(list_cached_parts(
            self._ms_name, self._config_name, cache_dir=self._cache_dir,
        ))
        # Always start at part 0 — restart-from-part-N would need
        # a saved offset (not implemented).
        self._part_idx = 0
        self._open_current_part()
        self._opened = True
        self._log.info(
            f"RotatingParquetIterable: {len(self._parts)} parts discovered,"
            f" {len(cached)} already on disk,"
            f" pre_download_pct={self._pre_download_pct},"
            f" max_cache={self._max_cache_files}"
        )

    def close(self) -> None:
        """Stop any in-flight background download and wait for it
        to exit (best-effort, 2s timeout). Idempotent.
        """
        self._stop.set()
        t = self._next_dl_thread
        if t is not None and t.is_alive():
            t.join(timeout=2.0)
        self._next_dl_thread = None

    def __iter__(self) -> Iterator[dict]:
        if not self._opened:
            self.open()
        return self

    def __next__(self) -> dict:
        if not self._opened:
            self.open()
        # If we have no current iterator (just opened), build one.
        if self._current_iter is None:
            self._open_current_part()
        try:
            sample = next(self._current_iter)
        except StopIteration:
            # Current part exhausted. Switch to the next part.
            self._switch_to_next_part()
            # ``_switch_to_next_part`` updated ``_current_iter``;
            # pull the first sample from it.
            try:
                sample = next(self._current_iter)
            except StopIteration:
                # ``_parts`` exhausted — terminate.
                raise StopIteration from None
        self._samples_yielded += 1
        # 50% pre-download trigger.
        if (
            self._next_dl_thread is None
            and self._part_idx + 1 < len(self._parts)
            and self._part_total_rows > 0
            and self._samples_yielded
                >= self._part_total_rows * self._pre_download_pct
        ):
            self._schedule_next_download()
        return sample

    # ----- internals ----- #

    def _open_current_part(self) -> None:
        """Ensure the current part is on disk and ``_current_iter`` is set.
        Assumes ``self._part_idx`` is valid.
        """
        part_idx, part_path, part_size = self._parts[self._part_idx]
        cache_path = cache_path_for_part(
            self._ms_name, self._config_name, part_idx,
            cache_dir=self._cache_dir,
        )
        if not cache_path.exists() or (
            part_size and cache_path.stat().st_size != part_size
        ):
            # Synchronous download of part 0 (called from open()) or
            # synchronous fallback when the background download
            # hasn't finished yet (e.g. iter ends before 50%).
            self._download_fn(
                self._ms_name, part_path, part_idx, part_size, cache_path,
                self._log,
            )
        self._current_path = cache_path
        # ``LocalParquetIterable`` is iterable (has ``__iter__``)
        # but not itself an iterator — call ``iter()`` to get one.
        self._current_iter = iter(LocalParquetIterable(cache_path))
        self._samples_yielded = 0
        self._part_total_rows = _count_part_rows(cache_path)

    def _schedule_next_download(self) -> None:
        """Start a background download of the next part. No-op if
        the next part is already in flight or already on disk.
        """
        next_idx = self._part_idx + 1
        if next_idx >= len(self._parts):
            return
        cache_path = cache_path_for_part(
            self._ms_name, self._config_name, next_idx,
            cache_dir=self._cache_dir,
        )
        if cache_path.exists():
            # Already downloaded by an earlier run; nothing to do.
            return
        # No live thread, no live cached file → kick one off.
        self._next_path = cache_path
        self._next_dl_part_idx = next_idx

        def _worker():
            try:
                _, part_path, part_size = self._parts[next_idx]
                self._download_fn(
                    self._ms_name, part_path, next_idx, part_size,
                    cache_path, self._log,
                )
            except Exception as e:
                self._log.warning(
                    f"background download of part {next_idx:05d}"
                    f" failed: {type(e).__name__}: {e}"
                )
            finally:
                self._next_dl_thread = None

        self._next_dl_thread = threading.Thread(
            target=_worker, name=f"prefetch-part-{next_idx:05d}", daemon=True,
        )
        self._next_dl_thread.start()

    def _switch_to_next_part(self) -> None:
        """Move from the just-exhausted part to the next one.
        Evicts the just-exhausted part's file (it is no longer
        needed: the iterator is past it and will not revisit).
        The cache after the switch is the new current part plus
        any in-flight / downloaded next part — by construction
        this is <= ``max_cache_files``.
        """
        if self._part_idx + 1 >= len(self._parts):
            return
        evict_path = self._current_path
        self._part_idx += 1
        # Open the new part. If the background download hasn't
        # finished yet, this will block (synchronous fallback).
        self._open_current_part()
        # The previous "next" is now the current, so drop the
        # ``_next_path`` reference; the new next (if any) will be
        # set by the next 50% trigger.
        self._next_path = None
        self._next_dl_part_idx = None
        # Evict the just-exhausted part. After the switch, the
        # cache holds [new current, optional in-flight next].
        # ``evict_path`` is neither, so it must go to keep the cap.
        if (
            evict_path is not None
            and evict_path != self._current_path
            and evict_path.exists()
        ):
            try:
                evict_path.unlink()
            except OSError as e:
                self._log.warning(
                    f"failed to evict {evict_path}: {e}"
                )


def _count_part_rows(path: Path) -> int:
    """Return the total number of rows in a parquet file (cheap;
    pyarrow reads only the footer)."""
    import pyarrow.parquet as pq
    return pq.ParquetFile(str(path)).metadata.num_rows
