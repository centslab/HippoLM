"""Per-rank training worker: setup, run, teardown.

The training entry point spawns one :func:`_train_worker` per
GPU (one per TP rank). Each worker is split into three phases,
called from a thin orchestration:

  - :func:`_setup_worker`       — env vars, NCCL/gloo init,
    per-worker logging, dataset / dataloader / prefetcher, TP
    model, GradScaler, Muon + AdamW optimizer pair. Returns a
    plain dict that the next two phases consume.
  - :func:`_run_training_loop`  — the per-step forward / backward
    / grad-to-CPU / step / log / checkpoint sequence. This is
    the heart of the worker; everything else is plumbing.
  - :func:`_teardown_worker`    — close the prefetcher thread,
    destroy the distributed process group.

The orchestration :func:`_train_worker` is the public symbol
the entry point imports. The split into three phases keeps the
training loop readable (the per-step logic used to be embedded
in a 400-line monolith with NCCL init / data plumbing mixed in)
and makes each phase independently testable.

Package layout
--------------

The package was split in July 2026 (no behaviour changes —
phases were already separated by comments in the original
``loop.py``):

  * :mod:`.support`  — stateless helpers (:func:`wsd_lr` for the
    WSD LR schedule, :func:`_slice_cu_seqlens` for chunk-local
    doc-boundary slicing).
  * :mod:`.grad_norm`— :func:`_compute_and_clip_grad_norm`
    (all-reduce + clip).
  * :mod:`.setup`    — :func:`_setup_worker` (the per-rank init
    monolith: env vars, NCCL init, model + optimizers).
  * :mod:`.run`      — :func:`_run_training_loop` (per-step
    forward / backward / grad-to-CPU / step / log / checkpoint).
  * :mod:`.teardown` — :func:`_teardown_worker` (prefetcher
    close + process-group destroy).
  * :mod:`.worker`   — :func:`_train_worker` (the thin
    orchestrator).

Public API is preserved by re-exporting every name below; all
callers (``scripts/train.py``, ``src.training.__init__``, the
WSD / slice / grad-norm unit tests) continue to
``import src.training.loop.X`` unchanged.
"""
from __future__ import annotations

from .grad_norm import _compute_and_clip_grad_norm
from .run import _run_training_loop
from .setup import _setup_worker
from .support import _slice_cu_seqlens, wsd_lr
from .teardown import _teardown_worker
from .worker import _train_worker


__all__ = [
    # Public entry point (imported by scripts/train.py and
    # src.training.__init__).
    "_train_worker",
    # Three phases.
    "_setup_worker",
    "_run_training_loop",
    "_teardown_worker",
    # Stateless helpers (re-exported for the unit tests in
    # test/{test_wsd_lr,test_chunk_loss_averaging,
    # test_compute_grad_norm}.py).
    "wsd_lr",
    "_slice_cu_seqlens",
    "_compute_and_clip_grad_norm",
]
