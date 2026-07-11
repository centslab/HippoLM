"""Test :func:`src.training.env.pin_cache_directories` behaviour.

Why: ``modelscope`` writes to ``~/.cache/modelscope`` and
``datasets`` writes to ``~/.cache/huggingface`` by default. Across
cloud providers with varying ``$HOME`` quota and RAID mount naming,
those paths have been observed to silently drift and fill the wrong
disk. :func:`pin_cache_directories` pins both SDK caches to
``<repo>/.cache/<vendor>/`` so the whole cache tree lives under one
root the user controls via the repo checkout location.
"""
from __future__ import annotations

import os

import pytest

from src.training.env import (
    _repo_root,
    configure_runtime_environment,
    pin_cache_directories,
)


_CACHE_ENV_VARS = (
    "HIPPOLM_CACHE_DIR",
    "MODELSCOPE_CACHE",
    "HF_HOME",
    "HF_DATASETS_CACHE",
)


@pytest.fixture
def clean_env(monkeypatch):
    """Strip every cache-related env var around each test so the
    default-resolution branch of :func:`pin_cache_directories`
    runs from a clean slate.
    """
    for k in _CACHE_ENV_VARS:
        monkeypatch.delenv(k, raising=False)
    return monkeypatch


def test_repo_root_contains_src():
    """``_repo_root()`` must point at the directory containing ``src/``
    (parents[2] of ``src/training/env.py``).
    """
    repo = _repo_root()
    assert (repo / "src" / "training" / "env.py").is_file(), (
        f"_repo_root() = {repo}, expected parent of src/training/env.py"
    )


def test_default_layout_under_repo_cache(clean_env):
    """All SDK caches pin to ``<repo>/.cache/<vendor>/`` by default.
    HIPPOLM-side (``HIPPOLM_CACHE_DIR``) is intentionally NOT set by
    this function — it stays under cache.py's own default.
    """
    pin_cache_directories()
    repo_cache = _repo_root() / ".cache"
    assert os.environ["MODELSCOPE_CACHE"] == str(repo_cache / "modelscope")
    assert os.environ["HF_HOME"] == str(repo_cache / "huggingface")
    assert os.environ["HF_DATASETS_CACHE"] == (
        str(repo_cache / "huggingface" / "datasets")
    )
    assert "HIPPOLM_CACHE_DIR" not in os.environ


def test_explicit_exports_survive_pin(clean_env):
    """An explicitly-exported ``MODELSCOPE_CACHE`` / ``HF_HOME`` /
    ``HF_DATASETS_CACHE`` must NOT be overwritten
    (setdefault semantics: explicit beats implicit).
    """
    os.environ["MODELSCOPE_CACHE"] = "/mnt/big/modelscope"
    os.environ["HF_HOME"] = "/mnt/big/huggingface"
    os.environ["HF_DATASETS_CACHE"] = "/mnt/big/huggingface/datasets"
    pin_cache_directories()
    assert os.environ["MODELSCOPE_CACHE"] == "/mnt/big/modelscope"
    assert os.environ["HF_HOME"] == "/mnt/big/huggingface"
    assert os.environ["HF_DATASETS_CACHE"] == "/mnt/big/huggingface/datasets"


def test_configure_runs_pin_before_datasets_import(clean_env, monkeypatch):
    """:func:`configure_runtime_environment` must call
    :func:`pin_cache_directories` BEFORE
    :func:`pin_datasets_retry_config` — the latter imports
    ``datasets``, which snapshots ``HF_DATASETS_CACHE`` into a
    module constant at import time and ignores later env changes.
    """
    from src.training import env as env_mod

    call_order: list[str] = []
    real_pin_cache = env_mod.pin_cache_directories
    real_pin_retry = env_mod.pin_datasets_retry_config

    def wrapped_pin_cache():
        call_order.append("pin_cache_directories")
        real_pin_cache()

    def wrapped_pin_retry():
        call_order.append("pin_datasets_retry_config")
        real_pin_retry()

    monkeypatch.setattr(env_mod, "pin_cache_directories", wrapped_pin_cache)
    monkeypatch.setattr(env_mod, "pin_datasets_retry_config", wrapped_pin_retry)
    configure_runtime_environment()
    assert call_order.index("pin_cache_directories") < call_order.index(
        "pin_datasets_retry_config"
    ), (
        f"pin_cache_directories must run before pin_datasets_retry_config "
        f"(which imports datasets and freezes HF_DATASETS_CACHE). "
        f"Actual call order: {call_order}"
    )


def test_hippolm_cache_dir_not_touched(clean_env):
    """:func:`pin_cache_directories` must not set ``HIPPOLM_CACHE_DIR``
    — that var belongs to :func:`src.training.data.cache.get_cache_dir`
    and is set by the CLI / yml via :mod:`scripts.train`. Keeping the
    two responsibilities split avoids an implicit coupling between
    the SDK-cache pin and the HIPPOLM-side default.
    """
    pin_cache_directories()
    assert "HIPPOLM_CACHE_DIR" not in os.environ