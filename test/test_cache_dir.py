"""Tests for the cache directory resolution in
:mod:`src.training.data.cache`.

The cache location was moved from ``~/.cache/hippolm/datasets``
(home-relative, hardcoded) to ``<repo_root>/.cache/hippolm/datasets``
(project-relative, no hardcoding) so the cache travels with the
checkout. These tests pin the new contract so a future refactor
that accidentally re-introduces a home-relative default gets
caught immediately.

Pinned here:
  * Default cache dir is the project-relative ``.cache/hippolm/datasets``
  * ``HIPPOLM_CACHE_DIR`` env var still overrides (for callers who
    need the cache on a different disk)
  * The cache dir is created on first call (idempotent)
  * The default is NOT the user's home dir (the regression guard
    for the bug that motivated the move)
  * ``cache_path_for_part`` honors a caller-supplied ``cache_dir``
    (the test-injection path)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

# Import order matters: src.training.data.cache uses a module-level
# constant resolved at import time. Importing it after sys.path
# shim means the constant captures THIS repo root, which is what
# the tests want to assert against.
from src.training.data import cache  # noqa: E402
from src.training.data.cache import (  # noqa: E402
    _PROJECT_CACHE_ROOT,
    cache_path_for,
    cache_path_for_part,
    get_cache_dir,
    list_cached_parts,
)


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch, tmp_path):
    """Strip ``HIPPOLM_CACHE_DIR`` from the env around each test so
    the default-path test isn't shadowed by a CI-set env var.
    Tests that want a custom value re-set it via ``monkeypatch.setenv``.
    """
    monkeypatch.delenv("HIPPOLM_CACHE_DIR", raising=False)


# --------------------------------------------------------------------------- #
# Default resolution.                                                          #
# --------------------------------------------------------------------------- #
def test_default_is_project_relative():
    """The default cache dir must be under the project root, not
    the user's home dir. This is the regression guard for the
    bug that motivated moving the cache in the first place."""
    d = get_cache_dir()
    # Project root: 4 levels up from src/training/data/cache.py.
    project_root = Path(cache.__file__).resolve().parents[3]
    assert d == project_root / ".cache" / "hippolm" / "datasets", (
        f"default cache is {d}, expected under {project_root}/.cache"
    )


def test_default_not_in_home():
    """The default cache must not be under ``$HOME``. The previous
    default was ``~/.cache/hippolm/datasets`` which caused cross-
    environment pollution (CI runner cache leaking into a
    developer's home dir, etc.)."""
    d = get_cache_dir()
    home = Path.home()
    assert not str(d).startswith(str(home) + os.sep), (
        f"default cache {d} is under $HOME ({home}); "
        "the project-relative default is the new contract."
    )


def test_default_path_is_absolute():
    """Sanity: the default must be an absolute path (it has to be,
    since ``parents[3]`` is absolute). A relative path would
    silently put the cache wherever the user invoked python from,
    which is a different bug but the same family."""
    d = get_cache_dir()
    assert d.is_absolute()


def test_module_constant_matches_default():
    """The module-level ``_PROJECT_CACHE_ROOT`` must agree with
    what :func:`get_cache_dir` returns when no env var is set.
    If they ever drift, the project-relative default is broken
    in some code path but not others."""
    d = get_cache_dir()
    assert d == _PROJECT_CACHE_ROOT


# --------------------------------------------------------------------------- #
# Env-var override.                                                           #
# --------------------------------------------------------------------------- #
def test_env_var_overrides_default(tmp_path, monkeypatch):
    """``HIPPOLM_CACHE_DIR=/some/path`` must override the
    project-relative default."""
    monkeypatch.setenv("HIPPOLM_CACHE_DIR", str(tmp_path / "custom"))
    d = get_cache_dir()
    assert d == tmp_path / "custom"


def test_env_var_empty_string_falls_back_to_default(monkeypatch):
    """Empty ``HIPPOLM_CACHE_DIR`` is treated as unset (matches the
    behavior of most env-var-as-override patterns; the user can
    always set it to ``.`` if they want cwd-relative)."""
    monkeypatch.setenv("HIPPOLM_CACHE_DIR", "")
    d = get_cache_dir()
    assert d == _PROJECT_CACHE_ROOT


# --------------------------------------------------------------------------- #
# Side effects.                                                                #
# --------------------------------------------------------------------------- #
def test_get_cache_dir_creates_dir(tmp_path, monkeypatch):
    """First call must create the directory. Idempotent on
    subsequent calls (no exception)."""
    target = tmp_path / "new_cache"
    assert not target.exists()
    monkeypatch.setenv("HIPPOLM_CACHE_DIR", str(target))
    d = get_cache_dir()
    assert d.exists()
    assert d.is_dir()
    # Second call must not raise.
    d2 = get_cache_dir()
    assert d2 == d


def test_get_cache_dir_idempotent_under_existing():
    """If the cache dir already exists, :func:`get_cache_dir` must
    not error (mkdir with exist_ok is the standard pattern)."""
    # The default is created on first call, so by the time we
    # get here it exists. A second call must not raise.
    d1 = get_cache_dir()
    d2 = get_cache_dir()
    assert d1 == d2
    assert d1.exists()


# --------------------------------------------------------------------------- #
# cache_path_for_part honors the same resolution.                             #
# --------------------------------------------------------------------------- #
def test_cache_path_for_part_uses_default():
    """When called without a ``cache_dir`` argument,
    :func:`cache_path_for_part` should produce a path under the
    default cache dir (not under ``$HOME``)."""
    p = cache_path_for_part("ms/foo", "cfg", 0)
    assert str(p).startswith(str(_PROJECT_CACHE_ROOT) + os.sep), (
        f"cache_path_for_part produced {p}, expected under "
        f"{_PROJECT_CACHE_ROOT}"
    )


def test_cache_path_for_part_honors_caller_cache_dir(tmp_path):
    """When called WITH a ``cache_dir`` argument (the test-injection
    pattern), the path is under that dir — not the default."""
    test_dir = tmp_path / "test_inject"
    p = cache_path_for_part("ms/foo", "cfg", 7, cache_dir=test_dir)
    assert p.parent == test_dir
    assert p.name.endswith("part00007.snappy.parquet")


def test_cache_path_for_part_filename_format():
    """The filename must include the ms_name, config_name, and
    zero-padded part_index so multiple configs / parts coexist
    in the same cache dir without collision."""
    p = cache_path_for_part("OpenBMB/Ultra-FineWeb-L3", "en_qa", 42)
    assert p.name == (
        "OpenBMB__Ultra-FineWeb-L3__en_qa__part00042.snappy.parquet"
    )


def test_list_cached_parts_finds_existing(tmp_path):
    """Pre-stage a few part files in a temp cache dir; the
    listing function must find all of them in part-index order."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    ms = "ms/foo"
    cfg = "en_qa"
    # Stage parts 2, 5, 1 in a non-sorted order.
    for idx in (2, 5, 1):
        (cache_dir / f"{ms.replace('/', '__')}__{cfg}__part{idx:05d}"
                   ".snappy.parquet").write_bytes(b"")
    out = list_cached_parts(ms, cfg, cache_dir=cache_dir)
    assert out == [1, 2, 5]
