"""Process / launch plumbing for the training entry point.

Public API:
  - :func:`select_tp_gpus`     — auto-detect the largest viable contiguous
    run of 2/4/8 GPUs with enough free VRAM.
  - :func:`expand_for_tp_sim`  — collapse the GPU list when running TP
    in single-GPU simulation mode (gloo backend, N processes pinned
    to the same physical device).

The actual :func:`torch.distributed.init_process_group` call and the
``mp.spawn`` wrapper live in the entry point (PR-6 will factor them
into :mod:`.process_group` and :mod:`.spawn`).
"""
from .gpus import select_tp_gpus
from .tp_sim import expand_for_tp_sim

__all__ = ["select_tp_gpus", "expand_for_tp_sim"]
