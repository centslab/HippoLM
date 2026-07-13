"""Phase 3 of the training worker: teardown.

:func:`_teardown_worker` is called from the ``finally`` clause
of :func:`_run_training_loop` so every exit path (natural end
of training, exception in the loop body) cleans up. It also
runs a partial ``ctx`` (defensive against a caller bailing out
of :func:`_setup_worker` mid-init) — both the prefetcher
``close()`` and the ``dist.destroy_process_group`` call are
guarded.

``dist`` is imported lazily so the function can be called from
non-distributed unit tests on a partially-populated ctx.
"""
from __future__ import annotations

import logging
from typing import Any, Dict

log = logging.getLogger(__name__)


def _teardown_worker(ctx: Dict[str, Any]) -> None:
    """Close the prefetcher thread and destroy the distributed
    process group. Idempotent: safe to call multiple times (the
    finally clause in :func:`_run_training_loop` calls this, and
    a caller that bails out of setup_worker would call it
    directly with a partially-populated ctx).

    Also stops the async CPU-add worker if one was started in
    :func:`_setup_worker`. The stop function drains any
    in-flight entries first, so the per-param accumulators are
    left consistent on return (callers can inspect them after
    teardown if needed).
    """
    import torch.distributed as dist

    from src.training.param_offload import _stop_cpu_add_worker

    prefetcher = ctx.get("prefetcher")
    if prefetcher is not None:
        try:
            prefetcher.close()
        except Exception as e:
            log.warning("prefetcher close failed: %r", e)
    # Stop the async CPU-add worker first (drains its queue
    # before returning) so the optimizer state is consistent
    # before destroy_process_group runs. Idempotent: a no-op if
    # the worker was never started.
    try:
        _stop_cpu_add_worker()
    except Exception as e:
        log.warning("cpu-add worker stop failed: %r", e)
    if dist.is_available() and dist.is_initialized():
        try:
            dist.destroy_process_group()
        except Exception as e:
            log.warning("dist.destroy_process_group failed: %r", e)