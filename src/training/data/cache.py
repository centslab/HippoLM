"""Local parquet cache directory management.

The cache directory holds the parquet shards that
:class:`src.training.data.rotating.RotatingParquetIterable` has
fetched. The legacy policy (one shard, one config) is preserved
through :func:`cache_path_for`; the new multi-part policy is
exposed via :func:`cache_path_for_part`. The directory is created
lazily on first access; it is NOT created at module import time
(unlike the original train.py behavior).
"""
from __future__ import annotations

import logging
import os
from pathlib import Path


# Project-relative cache root. ``Path(__file__)`` is
# ``<repo>/src/training/data/cache.py``; ``parents[3]`` is the
# repo root. Used as the DEFAULT for :func:`get_cache_dir` so
# that the cache lives next to the code (and travels with the
# checkout) rather than in the user's home directory — this
# matters when the dev box is throwaway (containers, CI runners,
# shared multi-tenant hosts) where ``$HOME`` is not stable
# across runs.
#
# Resolution is intentionally relative to the source file rather
# than the cwd: the cache location is a property of the codebase,
# not of wherever the user happened to invoke ``python`` from.
_PROJECT_CACHE_ROOT = Path(__file__).resolve().parents[3] / ".cache" / "hippolm" / "datasets"


def get_cache_dir() -> Path:
    """Return the local cache directory for downloaded parquet shards.

    Resolution order (first match wins):

      1. ``HIPPOLM_CACHE_DIR`` env var — explicit override for
         callers who need the cache on a different disk (e.g. a
         fast SSD mount) or want to point at a shared location.
      2. ``<repo_root>/.cache/hippolm/datasets`` — project-
         relative default. Travels with the checkout, so two
         worktrees of the same repo on the same machine get
         separate caches (good — part-NN files would otherwise
         collide).
      3. **Removed**: the previous home-relative default
         ``~/.cache/hippolm/datasets`` was the cause of cross-
         environment pollution (CI runner cache from a prior
         task ending up in the developer's home dir, etc.).
         Set ``HIPPOLM_CACHE_DIR=~/.cache/hippolm/datasets`` to
         restore the old behaviour if needed.

    The directory is created on first call (idempotent).
    """
    p = Path(os.environ.get("HIPPOLM_CACHE_DIR") or _PROJECT_CACHE_ROOT)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _filename_for(ms_name: str, config_name: str | None, part_index: int) -> str:
    """Build a cache filename for a specific ``(ms_name, config_name, part_index)``.

    The ``part_index`` is zero-padded to 5 digits so the directory
    sorts by part number under ``ls``. The ``.part`` suffix is used
    during a download; the rename at the end of the download drops
    the suffix.
    """
    return (
        f"{ms_name.replace('/', '__')}__{config_name or 'default'}"
        f"__part{part_index:05d}.snappy.parquet"
    )


def cache_path_for(ms_name: str, config_name: str | None) -> Path:
    """Return the local cache path for the legacy single-shard policy.

    Equivalent to ``cache_path_for_part(ms_name, config_name, 0)``;
    kept as a back-compat re-export for the pre-v0.0.2 callers.
    """
    return cache_path_for_part(ms_name, config_name, part_index=0)


def cache_path_for_part(
    ms_name: str,
    config_name: str | None,
    part_index: int,
    cache_dir: Path | None = None,
) -> Path:
    """Return the local cache path for a specific part of a (ms, config) pair.

    The optional ``cache_dir`` lets tests point at a temp dir; the
    default is :func:`get_cache_dir` (which honors
    ``HIPPOLM_CACHE_DIR``).
    """
    base = cache_dir if cache_dir is not None else get_cache_dir()
    return base / _filename_for(ms_name, config_name, part_index)


def list_cached_parts(
    ms_name: str,
    config_name: str | None,
    cache_dir: Path | None = None,
) -> list[int]:
    """Return the part indices currently on disk for a (ms, config) pair,
    sorted ascending. Used by :class:`RotatingParquetIterable` to
    decide which parts can be reused without a re-download.
    """
    base = cache_dir if cache_dir is not None else get_cache_dir()
    prefix = f"{ms_name.replace('/', '__')}__{config_name or 'default'}__part"
    suffix = ".snappy.parquet"
    out: list[int] = []
    for p in base.iterdir():
        if p.is_file() and p.name.startswith(prefix) and p.name.endswith(suffix):
            mid = p.name[len(prefix):-len(suffix)]
            try:
                out.append(int(mid))
            except ValueError:
                continue
    out.sort()
    return out


def purge_stale_cache_if_no_hit(
    expected: list[tuple[str, str | None]],
    cache_dir: Path | None = None,
    log: logging.Logger | None = None,
) -> int:
    """Pre-training cache integrity check.

    For each ``(ms_name, config_name)`` in ``expected``, check if any
    matching cached parquet file exists in ``cache_dir``. If NONE
    of them have any cache hit (the cache is empty OR holds only
    parquets from a *different* dataset), clear ALL ``*.parquet``
    files from the cache directory to prevent stale accumulation
    across runs of different datasets.

    Returns the number of files cleared. Zero means either:

      - any expected config had a cache hit (caller will reuse), OR
      - the cache was already empty (nothing to clear).

    Idempotent — safe to call multiple times.

    Rationale: the rotator's existing eviction contract keeps at
    most ``max_cache_files`` (default 2) parquets for the *current*
    dataset, but says nothing about parquets left over from a
    *previous* dataset the dev box trained on. Those leftovers
    accumulate disk usage forever. The "no hit for any expected
    config" signal is a strong indicator we're starting a fresh
    training run on a fresh dataset — purge everything to bound
    the cache at zero before re-populating.

    Called from :func:`src.training.loop._setup_worker` once on
    rank 0 before any dataset is instantiated (so the purge happens
    before :class:`RotatingParquetIterable` tries to reuse a
    now-deleted cached part).
    """
    logger = log or logging.getLogger(__name__)
    base = cache_dir if cache_dir is not None else get_cache_dir()

    # 1. Did ANY expected config have a cache hit?
    any_hit = False
    for ms_name, cfg in expected:
        cached = list_cached_parts(ms_name, cfg, cache_dir=base)
        if cached:
            any_hit = True
            logger.info(
                f"cache: hit for {ms_name} (config={cfg!r}):"
                f" {len(cached)} part(s) on disk; skipping purge"
            )
            break

    if any_hit:
        return 0

    # 2. No hit anywhere — clear every parquet in the cache dir.
    cleared = 0
    if not base.exists():
        return 0
    for p in base.iterdir():
        if p.is_file() and p.name.endswith(".snappy.parquet"):
            try:
                p.unlink()
                cleared += 1
            except OSError as e:
                logger.warning(
                    f"cache: failed to clear stale file {p}: {e}"
                )
    logger.info(
        f"cache: no hit for any of {len(expected)} expected config(s);"
        f" cleared {cleared} stale parquet file(s) from {base}"
    )
    return cleared
