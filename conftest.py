"""Repo-root pytest configuration.

Default ``pytest`` skips ``@pytest.mark.slow`` tests so the full
sweep finishes in seconds, not minutes. The slow tests guard
prod-shape KDA kernel correctness (see
``test/test_kda_chunk_compat.py``); run them on demand with::

    pytest -m slow              # only slow tests
    pytest -m "not slow"        # skip slow (matches the default)
    pytest -m "slow or skip"    # mix slow + any skipif-skipped tests

Why a hook instead of ``addopts = -m 'not slow'``?
``addopts`` would force the filter on every pytest invocation,
which ``-m`` users would have to fight against. The hook instead
adds ``skip`` markers only when the user has *not* passed any
explicit ``-m`` filter — so ``pytest``, ``pytest -m slow``, and
``pytest -m "not slow"`` all do what you'd expect.
"""
from __future__ import annotations

import pytest


def pytest_configure(config):
    """Register the ``slow`` marker so pytest doesn't warn about
    an unknown marker (one warning per parametrized case otherwise)."""
    config.addinivalue_line(
        "markers",
        "slow: prod-shape slow tests; skipped by default. "
        "Run with `pytest -m slow` to include.",
    )


def pytest_collection_modifyitems(config, items):
    """Skip ``@pytest.mark.slow`` tests unless the user explicitly
    selected slow markers via ``-m``."""
    # If the user passed ``-m`` on the CLI (any non-empty value), defer
    # to their filter. ``config.getoption("-m")`` defaults to ``""``,
    # not ``None`` — treat empty as "no filter".
    marker_filter = config.getoption("-m", default="") or ""
    if marker_filter:
        return
    skip_slow = pytest.mark.skip(
        reason="marked @pytest.mark.slow; opt in with `pytest -m slow`"
    )
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)