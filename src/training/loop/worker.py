"""Per-rank training worker: thin orchestrator.

:func:`_train_worker` is the public entry point spawned once
per GPU by ``scripts/train.py``. It does almost nothing on its
own — it builds the ``ctx`` dict via :func:`_setup_worker` and
hands it to :func:`_run_training_loop` (which owns the
:func:`_teardown_worker` call via its ``finally`` clause).

The split into the three phases (setup / run / teardown)
keeps each independently testable and keeps the per-step
training loop body readable without the NCCL init / data
plumbing that used to be embedded in the same file.
"""
from __future__ import annotations

import queue
from typing import List, Optional

from .run import _run_training_loop
from .setup import _setup_worker


def _train_worker(
    rank: int,
    args,
    gpus: List[int],
    port: int,
    run_dir_path: str,
    shared_batch_queues: Optional[List[queue.Queue]] = None,
) -> None:
    """Single-worker training entry point (one process per GPU).

    Thin orchestration: setup -> run (which owns the teardown
    via its finally clause).
    """
    ctx = _setup_worker(
        rank, args, gpus, port, run_dir_path, shared_batch_queues,
    )
    _run_training_loop(ctx)