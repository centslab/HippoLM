"""Local parquet cache directory management.

The cache holds at most one ModelScope parquet shard at a time
(see :mod:`src.training.data.prefetch` for the policy rationale).
The directory is created lazily on first access; it is NOT created
at module import time (unlike the original train.py behavior).
"""
from __future__ import annotations

import os
from pathlib import Path


def get_cache_dir() -> Path:
    """Return the local cache directory for downloaded parquet shards.

    Honors the ``HIPPOLM_CACHE_DIR`` env var; defaults to
    ``~/.cache/hippolm/datasets``. Creates the directory on first
    call (idempotent).
    """
    p = Path(
        os.environ.get("HIPPOLM_CACHE_DIR")
        or str(Path.home() / ".cache" / "hippolm" / "datasets")
    )
    p.mkdir(parents=True, exist_ok=True)
    return p


def cache_path_for(ms_name: str, config_name: str | None) -> Path:
    """Return the local cache path for a (ms_name, config_name) shard."""
    return get_cache_dir() / (
        f"{ms_name.replace('/', '__')}__{config_name or 'default'}__part0.snappy.parquet"
    )
