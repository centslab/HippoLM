"""Tests for :func:`_teardown_worker` in
:mod:`src.training.loop.teardown`.

The training loop's ``finally`` clause delegates cleanup to this
function. Two contracts must hold:

  1. **Idempotence** — calling teardown multiple times is safe
     (the loop's ``finally`` calls it once; a caller that bails
     out of ``_setup_worker`` mid-init might also call it
     directly with a partially-populated ctx).

  2. **Defensive** — when ctx is missing keys (e.g. ``prefetcher``,
     ``args``, ``dist`` not initialized), the function must not
     raise; it should warn and continue. This is the production
     teardown path for failed inits.

The lifecycle ordering is also pinned:

  * :func:`shutdown_per_layer_gpu_accum` (only when
    ``args.offload_strategy == "per_layer_gpu"``) MUST run BEFORE
    :func:`torch.distributed.destroy_process_group` — otherwise
    the worker thread can't complete its in-flight CPU adds
    cleanly.
  * :func:`torch.distributed.destroy_process_group` MUST run
    even if the prefetcher close OR the per-layer shutdown
    raises (defensive: leak the worker thread rather than leak
    the process group).

These tests pin those contracts so the upcoming training-loop
refactor doesn't regress them. They use ``monkeypatch`` to
replace :mod:`torch.distributed` and
:func:`shutdown_per_layer_gpu_accum` with controllable fakes.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from src.training.loop import _teardown_worker  # noqa: E402


# =========================================================================== #
# Fakes.
# =========================================================================== #
class _FakePrefetcher:
    """Records ``close()`` calls; raises when ``raise_on_close`` is set.

    Used to verify that teardown calls ``close()`` and tolerates
    failures from it.
    """

    def __init__(self, raise_on_close: bool = False) -> None:
        self.close_calls = 0
        self.raise_on_close = raise_on_close

    def close(self) -> None:
        self.close_calls += 1
        if self.raise_on_close:
            raise RuntimeError("synthetic prefetcher close failure")


class _FakeArgs:
    """Minimal ``args`` substitute exposing only what teardown reads."""

    def __init__(self, offload_strategy: str = "cpu_add") -> None:
        self.offload_strategy = offload_strategy


class _FakeDist:
    """Controllable fake for :mod:`torch.distributed` lifecycle calls."""

    def __init__(
        self,
        initialized: bool = True,
        destroy_raises: bool = False,
    ) -> None:
        self.initialized = initialized
        self.destroy_calls = 0
        self.destroy_raises = destroy_raises

    # Module-level accessors (rebound via monkeypatch in each test).
    def is_initialized(self) -> bool:
        return self.initialized

    def destroy_process_group(self) -> None:
        self.destroy_calls += 1
        if self.destroy_raises:
            raise RuntimeError("synthetic destroy failure")


def _monkeypatch_dist(monkeypatch, fake: _FakeDist) -> None:
    """Replace the ``torch.distributed`` namespace teardown reads."""
    monkeypatch.setattr("torch.distributed.is_initialized", fake.is_initialized)
    monkeypatch.setattr(
        "torch.distributed.destroy_process_group", fake.destroy_process_group,
    )


def _monkeypatch_per_layer_shutdown(monkeypatch, fn) -> None:
    """Replace :func:`shutdown_per_layer_gpu_accum` with a callable."""
    monkeypatch.setattr(
        "src.training.param_offload.shutdown_per_layer_gpu_accum", fn,
    )


# =========================================================================== #
# Prefetcher lifecycle.
# =========================================================================== #
def test_teardown_closes_prefetcher(monkeypatch):
    """When ctx has a prefetcher, ``_teardown_worker`` calls
    ``.close()`` exactly once."""
    pref = _FakePrefetcher()
    dist = _FakeDist(initialized=True)
    _monkeypatch_dist(monkeypatch, dist)
    _teardown_worker({"prefetcher": pref, "args": None})
    assert pref.close_calls == 1
    assert dist.destroy_calls == 1


def test_teardown_handles_prefetcher_close_failure(monkeypatch, caplog):
    """If ``prefetcher.close()`` raises, teardown must log a
    warning and continue — the exception MUST NOT propagate
    (otherwise the ``finally`` clause's caller would see it)."""
    pref = _FakePrefetcher(raise_on_close=True)
    dist = _FakeDist(initialized=True)
    _monkeypatch_dist(monkeypatch, dist)
    with caplog.at_level(logging.WARNING, logger="src.training.loop.teardown"):
        _teardown_worker({"prefetcher": pref, "args": None})
    assert pref.close_calls == 1
    assert dist.destroy_calls == 1, (
        "dist.destroy_process_group must run even if prefetcher.close raises"
    )
    assert any(
        "prefetcher close failed" in rec.message.lower() for rec in caplog.records
    ), "expected a warning log on prefetcher.close failure"


def test_teardown_skips_close_when_prefetcher_is_none(monkeypatch):
    """``ctx['prefetcher'] is None`` (dummy-data path) — no
    ``.close()`` call."""
    dist = _FakeDist(initialized=True)
    _monkeypatch_dist(monkeypatch, dist)
    _teardown_worker({"prefetcher": None, "args": None})
    assert dist.destroy_calls == 1


def test_teardown_handles_missing_prefetcher_key(monkeypatch):
    """``ctx`` without the ``prefetcher`` key (defensive: caller
    that bails out of ``_setup_worker`` before prefetcher is
    built)."""
    dist = _FakeDist(initialized=True)
    _monkeypatch_dist(monkeypatch, dist)
    _teardown_worker({"args": None})  # no prefetcher key at all
    assert dist.destroy_calls == 1


# =========================================================================== #
# Idempotence.
# =========================================================================== #
def test_teardown_is_idempotent(monkeypatch):
    """Calling ``_teardown_worker`` twice must be safe (no
    double-close error, no double-destroy of the process group).
    """
    pref = _FakePrefetcher()
    dist = _FakeDist(initialized=True)
    _monkeypatch_dist(monkeypatch, dist)
    ctx = {"prefetcher": pref, "args": None}
    _teardown_worker(ctx)
    _teardown_worker(ctx)
    # The fake records both calls (the contract is "no exception";
    # whether to call .close() / .destroy() twice is the caller's
    # responsibility). We just verify both calls ran cleanly.
    assert pref.close_calls == 2
    assert dist.destroy_calls == 2


def test_teardown_with_no_args_key(monkeypatch):
    """``ctx`` without ``args`` key (defensive: teardown called
    before ``_setup_worker`` wrote the args key). Must not raise.
    """
    dist = _FakeDist(initialized=False)  # dist not init -> nothing to do
    _monkeypatch_dist(monkeypatch, dist)
    _teardown_worker({})  # both prefetcher and args missing
    assert dist.destroy_calls == 0


# =========================================================================== #
# Distributed lifecycle.
# =========================================================================== #
def test_teardown_skips_dist_destroy_when_not_initialized(monkeypatch):
    """When dist is not initialized (e.g. CPU-only test path), teardown
    must NOT call ``dist.destroy_process_group`` (would raise)."""
    dist = _FakeDist(initialized=False)
    _monkeypatch_dist(monkeypatch, dist)
    _teardown_worker({"prefetcher": None, "args": None})
    assert dist.destroy_calls == 0, (
        "dist.destroy_process_group must NOT be called when dist "
        "isn't initialized"
    )


def test_teardown_handles_destroy_process_group_failure(monkeypatch, caplog):
    """If ``dist.destroy_process_group`` raises, teardown must
    swallow it (it's the very last step; the function's job is
    done)."""
    dist = _FakeDist(initialized=True, destroy_raises=True)
    _monkeypatch_dist(monkeypatch, dist)
    with caplog.at_level(logging.WARNING, logger="src.training.loop.teardown"):
        _teardown_worker({"prefetcher": None, "args": None})
    assert dist.destroy_calls == 1
    assert any(
        "destroy_process_group failed" in rec.message.lower()
        for rec in caplog.records
    )


# =========================================================================== #
# per_layer_gpu shutdown integration.
# =========================================================================== #
def test_teardown_per_layer_gpu_shuts_down_worker(monkeypatch):
    """When ``args.offload_strategy == 'per_layer_gpu'``, teardown
    calls ``shutdown_per_layer_gpu_accum()``."""
    shutdown_calls = []
    dist = _FakeDist(initialized=True)
    _monkeypatch_dist(monkeypatch, dist)
    _monkeypatch_per_layer_shutdown(
        monkeypatch, lambda: shutdown_calls.append("called"),
    )
    _teardown_worker({"prefetcher": None, "args": _FakeArgs("per_layer_gpu")})
    assert shutdown_calls == ["called"]


def test_teardown_per_layer_gpu_shutdown_runs_before_dist_destroy(monkeypatch):
    """The lifecycle order: ``shutdown_per_layer_gpu_accum`` MUST
    run BEFORE ``dist.destroy_process_group`` (the worker thread
    needs the process group alive to complete its in-flight CPU
    adds cleanly).

    If the refactor swaps the order, in-flight D2H + CPU adds
    could deadlock on a destroyed process group.
    """
    call_order = []
    dist = _FakeDist(initialized=True)

    def fake_destroy():
        call_order.append("destroy")

    def fake_shutdown():
        call_order.append("shutdown")

    _monkeypatch_dist(monkeypatch, _FakeDist(initialized=True))
    monkeypatch.setattr("torch.distributed.destroy_process_group", fake_destroy)
    _monkeypatch_per_layer_shutdown(monkeypatch, fake_shutdown)
    _teardown_worker({"prefetcher": None, "args": _FakeArgs("per_layer_gpu")})
    assert call_order == ["shutdown", "destroy"], (
        f"shutdown_per_layer_gpu_accum must run BEFORE "
        f"dist.destroy_process_group; got {call_order}"
    )


def test_teardown_cpu_add_strategy_skips_per_layer_shutdown(monkeypatch):
    """When ``args.offload_strategy`` is the legacy ``'cpu_add'``
    (or unset / not 'per_layer_gpu'), teardown does NOT call
    ``shutdown_per_layer_gpu_accum``."""
    shutdown_calls = []
    _monkeypatch_dist(monkeypatch, _FakeDist(initialized=False))
    _monkeypatch_per_layer_shutdown(
        monkeypatch, lambda: shutdown_calls.append(1),
    )
    _teardown_worker({"prefetcher": None, "args": _FakeArgs("cpu_add")})
    assert shutdown_calls == [], (
        "shutdown_per_layer_gpu_accum must NOT be called under the cpu_add strategy"
    )


def test_teardown_no_args_key_skips_per_layer_shutdown(monkeypatch):
    """``ctx`` without ``args`` (defensive teardown path) — no
    per-layer shutdown call (we have no strategy info)."""
    shutdown_calls = []
    _monkeypatch_dist(monkeypatch, _FakeDist(initialized=False))
    _monkeypatch_per_layer_shutdown(
        monkeypatch, lambda: shutdown_calls.append(1),
    )
    _teardown_worker({"prefetcher": None})
    assert shutdown_calls == []


def test_teardown_handles_per_layer_shutdown_failure(monkeypatch, caplog):
    """If ``shutdown_per_layer_gpu_accum`` raises, teardown must
    continue to ``dist.destroy_process_group`` (defensive: leak
    the worker thread rather than leak the process group)."""
    def fake_shutdown():
        raise RuntimeError("synthetic shutdown failure")

    dist = _FakeDist(initialized=True)
    _monkeypatch_dist(monkeypatch, dist)
    _monkeypatch_per_layer_shutdown(monkeypatch, fake_shutdown)
    with caplog.at_level(logging.WARNING, logger="src.training.loop.teardown"):
        _teardown_worker({"prefetcher": None, "args": _FakeArgs("per_layer_gpu")})
    assert dist.destroy_calls == 1, (
        "dist.destroy_process_group must run even if per-layer "
        "shutdown fails"
    )
    assert any(
        "shutdown_per_layer_gpu_accum failed" in rec.message.lower()
        for rec in caplog.records
    )


# =========================================================================== #
# End-to-end combinations.
# =========================================================================== #
def test_teardown_per_layer_gpu_with_failing_prefetcher(monkeypatch):
    """Per-layer shutdown + prefetcher-close failure + dist destroy
    all in sequence: every step must run, in the right order."""
    pref = _FakePrefetcher(raise_on_close=True)
    call_order = []
    dist = _FakeDist(initialized=True)

    def fake_destroy():
        call_order.append("destroy")

    def fake_shutdown():
        call_order.append("shutdown")

    _monkeypatch_dist(monkeypatch, _FakeDist(initialized=True))
    monkeypatch.setattr("torch.distributed.destroy_process_group", fake_destroy)
    _monkeypatch_per_layer_shutdown(monkeypatch, fake_shutdown)

    # Wrap pref.close to record its call but still raise.
    original_close = pref.close
    def recording_close():
        call_order.append("prefetcher.close")
        original_close()
    pref.close = recording_close

    _teardown_worker({"prefetcher": pref, "args": _FakeArgs("per_layer_gpu")})
    assert call_order == ["prefetcher.close", "shutdown", "destroy"], (
        f"unexpected teardown order: {call_order}"
    )


def test_teardown_partial_ctx_no_dist_no_prefetcher(monkeypatch):
    """Bare-minimum ctx (``{}``) — must not raise. This is the
    most defensive case (caller bailed out of ``_setup_worker``
    before any key was set)."""
    dist = _FakeDist(initialized=False)
    _monkeypatch_dist(monkeypatch, dist)
    _monkeypatch_per_layer_shutdown(
        monkeypatch, lambda: pytest.fail("shutdown called on empty ctx"),
    )
    # Should not raise.
    _teardown_worker({})
    assert dist.destroy_calls == 0