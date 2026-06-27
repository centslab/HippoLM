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
