"""Tests for ``_teardown_worker`` (Phase 3 of the per-rank training worker).

The teardown runs from the ``finally`` clause of
:func:`_run_training_loop` — every exit path (natural end of
training, exception in the loop body, caller bailing out of
:func:`_setup_worker` mid-init) cleans up via this function.
The contract is:

  1. ``prefetcher.close()`` is called if the ctx carries a
     prefetcher (and a close failure is logged, not raised).
  2. ``dist.destroy_process_group()`` is called when the
     process group was initialized (failure logged, not raised).
  3. Safe to call with a partially-populated ctx (e.g. only
     ``prefetcher`` set, only ``args`` set, empty ctx) — no
     AttributeError on missing keys.

These tests pin that contract by mocking ``dist`` and the
prefetcher shutdown helper and observing what gets called in
each branch.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from src.training.loop import _teardown_worker  # noqa: E402


# --------------------------------------------------------------------------- #
# Helpers.                                                                    #
# --------------------------------------------------------------------------- #
def _make_ctx(
    *,
    prefetcher: object | None = None,
    args: object | None = None,
) -> dict:
    """Build a ctx dict with the keys ``_teardown_worker`` reads.

    The function uses ``ctx.get("prefetcher")`` /
    ``ctx.get("args")`` — extra keys are silently ignored, so
    tests can pass a minimal dict and assert on the side
    effects.
    """
    return {"prefetcher": prefetcher, "args": args}


def _patched_dist(
    *,
    available: bool = True,
    initialized: bool = True,
    destroy_raises: bool = False,
):
    """Patch ``torch.distributed`` so the teardown's in-function
    ``import torch.distributed as dist`` picks up the fake.

    Returns a context manager that, on enter, swaps the real
    ``torch.distributed`` with a :class:`MagicMock` whose
    ``is_available`` / ``is_initialized`` /
    ``destroy_process_group`` are pre-configured. On exit, the
    real module is restored.
    """
    fake_dist = MagicMock()
    fake_dist.is_available.return_value = available
    fake_dist.is_initialized.return_value = initialized
    if destroy_raises:
        fake_dist.destroy_process_group.side_effect = RuntimeError("nope")
    return patch("torch.distributed", fake_dist), fake_dist


# --------------------------------------------------------------------------- #
# Empty / minimal ctx — defensive.                                            #
# --------------------------------------------------------------------------- #
def test_empty_ctx_is_safe_no_destroy_no_shutdown():
    """An empty ctx (caller bailed before any setup) must not raise.

    ``prefetcher`` is None → skip close. ``dist`` is not
    initialized (in this test) → skip destroy.
    """
    cm, fake_dist = _patched_dist(initialized=False)
    with cm:
        _teardown_worker(_make_ctx())
    fake_dist.destroy_process_group.assert_not_called()


def test_ctx_with_only_args_skips_noop_cleanup():
    """``args`` set but no prefetcher and dist not initialized →
    teardown is a no-op."""
    args = SimpleNamespace()
    cm, fake_dist = _patched_dist(initialized=False)
    with cm:
        _teardown_worker(_make_ctx(args=args))
    fake_dist.destroy_process_group.assert_not_called()


# --------------------------------------------------------------------------- #
# Prefetcher close branch.                                                    #
# --------------------------------------------------------------------------- #
def test_prefetcher_close_is_called():
    """When ctx carries a prefetcher, ``close()`` is called exactly
    once."""
    prefetcher = MagicMock()
    args = SimpleNamespace()
    cm, _fake_dist = _patched_dist(initialized=False)
    with cm:
        _teardown_worker(_make_ctx(prefetcher=prefetcher, args=args))
    prefetcher.close.assert_called_once()


def test_prefetcher_close_failure_is_logged_not_raised(caplog):
    """A failing prefetcher.close() must NOT propagate — the
    teardown continues to dist.destroy_process_group so the
    process group is still torn down."""
    prefetcher = MagicMock()
    prefetcher.close.side_effect = RuntimeError("prefetcher dead")
    args = SimpleNamespace()
    cm, fake_dist = _patched_dist(initialized=True)
    with caplog.at_level(logging.WARNING, logger="src.training.loop.teardown"):
        with cm:
            _teardown_worker(_make_ctx(prefetcher=prefetcher, args=args))
    fake_dist.destroy_process_group.assert_called_once()
    assert any(
        "prefetcher close failed" in rec.message for rec in caplog.records
    ), f"expected warning about prefetcher close; got: {[r.message for r in caplog.records]}"


# --------------------------------------------------------------------------- #
# dist.destroy_process_group branch.                                          #
# --------------------------------------------------------------------------- #
def test_dist_not_initialized_skips_destroy():
    """If torch.distributed is not initialized, destroy is a no-op
    (avoids raising when called from a single-process / TP-sim
    test)."""
    cm, fake_dist = _patched_dist(initialized=False)
    with cm:
        _teardown_worker(_make_ctx())
    fake_dist.destroy_process_group.assert_not_called()


def test_dist_not_available_skips_destroy():
    """Even if ``is_initialized`` returns True, ``is_available`` False
    skips destroy (defensive — extremely rare in practice)."""
    cm, fake_dist = _patched_dist(available=False, initialized=True)
    with cm:
        _teardown_worker(_make_ctx())
    fake_dist.destroy_process_group.assert_not_called()


def test_dist_destroy_failure_is_logged_not_raised(caplog):
    """A failing ``dist.destroy_process_group`` is logged, not
    raised — otherwise the finally clause would mask the
    original loop-body exception with a teardown exception."""
    prefetcher = MagicMock()
    args = SimpleNamespace()
    cm, _fake_dist = _patched_dist(initialized=True, destroy_raises=True)
    with caplog.at_level(logging.WARNING, logger="src.training.loop.teardown"):
        with cm:
            # Must NOT raise — teardown swallows the destroy error.
            _teardown_worker(_make_ctx(prefetcher=prefetcher, args=args))
    prefetcher.close.assert_called_once()
    assert any(
        "dist.destroy_process_group failed" in rec.message
        for rec in caplog.records
    )


# --------------------------------------------------------------------------- #
# Idempotency.                                                                #
# --------------------------------------------------------------------------- #
def test_teardown_is_idempotent_when_called_twice():
    """The training loop's ``finally`` clause calls teardown once.
    A defensive caller might call it twice (e.g. a wrapper that
    cleans up on a separate exception path). Each branch must be
    safe to re-enter — ``prefetcher.close()`` and
    ``dist.destroy_process_group()`` are both called twice and
    both must not raise."""
    prefetcher = MagicMock()
    args = SimpleNamespace()
    cm, _fake_dist = _patched_dist(initialized=True)
    with cm:
        _teardown_worker(_make_ctx(prefetcher=prefetcher, args=args))
        # Second call: same ctx, must not raise.
        _teardown_worker(_make_ctx(prefetcher=prefetcher, args=args))
    assert prefetcher.close.call_count == 2


def test_teardown_swallows_prefetcher_close_but_still_destroys_dist(caplog):
    """If prefetcher.close() raises but dist.destroy would
    succeed, BOTH warnings are logged and dist.destroy still
    runs (otherwise the process group leaks)."""
    prefetcher = MagicMock()
    prefetcher.close.side_effect = RuntimeError("close kaboom")
    args = SimpleNamespace()
    cm, fake_dist = _patched_dist(initialized=True)
    with caplog.at_level(logging.WARNING, logger="src.training.loop.teardown"):
        with cm:
            _teardown_worker(_make_ctx(prefetcher=prefetcher, args=args))
    fake_dist.destroy_process_group.assert_called_once()
    # Prefetcher warning present, dist warning absent.
    msgs = [rec.message for rec in caplog.records]
    assert any("prefetcher close failed" in m for m in msgs), msgs
    assert not any("dist.destroy_process_group failed" in m for m in msgs), msgs


# --------------------------------------------------------------------------- #
# Plain-Python runner. Pytest is not a project dependency; collect every      #
# ``def test_*`` and invoke, reporting pass/fail per test.                    #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import inspect
    import traceback

    tests = [
        (name, fn)
        for name, fn in globals().items()
        if name.startswith("test_") and callable(fn)
    ]
    tests.sort(key=lambda kv: inspect.getsourcelines(kv[1])[1])

    passed, failed = 0, 0
    for name, fn in tests:
        try:
            fn()
        except Exception:
            failed += 1
            print(f"  FAIL  {name}")
            traceback.print_exc()
        else:
            passed += 1
            print(f"  PASS  {name}")
    print(f"\n{passed} passed, {failed} failed, {passed + failed} total")
    if failed:
        sys.exit(1)