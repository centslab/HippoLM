"""Checkpoint save/load for the training loop.

Revives the v0.0.0-era :func:`save_checkpoint` that ships with the
``scripts/train.py`` entry point. The original signature took a
single ``optimizer``; this revision accepts a ``dict[str, Optimizer]``
so the per-device Muon + AdamW pair can both be captured in one
file (the HippoLM optimizer pair is co-equal: skipping one breaks
resume).

The ``scaler`` argument is kept for symmetry with the older code
path, but in practice we run with ``GradScaler(enabled=False)`` —
its ``state_dict`` is a no-op. The argument is preserved so the
entry point's call site is unchanged if/when we later swap in a
real scaler (e.g. for FP16 with explicit loss scaling).

Rank-0-only save is the entry point's responsibility. The
checkpoint file captures rank 0's view of the sharded TP model
and the rank-0 optimizer state; the other ranks' shards are not
persisted here. A full multi-rank save would require
``dist.gather`` / ``all_gather`` of the sharded state dicts, which
is out of scope for this minimal revival.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch

log = logging.getLogger(__name__)


def save_checkpoint(
    model: torch.nn.Module,
    optimizers: Dict[str, torch.optim.Optimizer],
    scaler: Optional[torch.amp.GradScaler],
    step: int,
    loss: float,
    checkpoint_dir: Path,
) -> Path:
    """Persist model + per-optimizer state to ``checkpoint_dir``.

    Returns the path of the written ``.pt`` file. The directory
    is created if missing. The file name embeds the step number
    (``checkpoint_step_{step}.pt``) so multiple snapshots can
    coexist on disk and a stale run can be picked apart by step.

    The ``scaler`` argument is optional but accepted for forward
    compatibility — when AMP is disabled the entry point passes
    a no-op ``GradScaler(enabled=False)`` and we save its (empty)
    state dict.
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = checkpoint_dir / f"checkpoint_step_{step}.pt"
    payload = {
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": {
            name: opt.state_dict() for name, opt in optimizers.items()
        },
        "loss": loss,
    }
    if scaler is not None:
        # ``GradScaler(enabled=False).state_dict()`` is ``{}``;
        # storing it is harmless and makes the file shape uniform.
        payload["scaler_state_dict"] = scaler.state_dict()
    torch.save(payload, path)
    log.info("checkpoint saved: step=%d loss=%.4e path=%s", step, loss, path)
    return path


def load_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizers: Optional[Dict[str, torch.optim.Optimizer]] = None,
    scaler: Optional[torch.amp.GradScaler] = None,
    map_location: str = "cpu",
) -> Tuple[int, float]:
    """Restore model (and optionally optimizers / scaler) from disk.

    Returns ``(step, loss)`` for the caller to resume from. Only
    the optimizer names that exist in both the file and the
    passed-in dict are restored; missing entries (e.g. the file
    was written by an older run with only one optimizer) are
    logged and skipped. Symmetric on the scaler side: missing
    key in the file is silently ignored.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizers is not None and "optimizer_state_dict" in ckpt:
        file_opt = ckpt["optimizer_state_dict"]
        for name, opt in optimizers.items():
            if name in file_opt:
                opt.load_state_dict(file_opt[name])
            else:
                log.warning(
                    "checkpoint %s has no state for optimizer %r;"
                    " leaving that optimizer untouched",
                    path, name,
                )
    if scaler is not None and "scaler_state_dict" in ckpt:
        scaler.load_state_dict(ckpt["scaler_state_dict"])
    return int(ckpt.get("step", 0)), float(ckpt.get("loss", float("nan")))
