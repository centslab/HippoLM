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
    keep_last_n: Optional[int] = None,
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

    ``keep_last_n`` (optional): when set, prune older
    ``checkpoint_step_*.pt`` files in ``checkpoint_dir`` after the
    new save so at most ``keep_last_n`` files remain (newest
    first). ``None`` (default) keeps all files — useful when
    callers want to manage retention themselves (e.g. a separate
    "best" tracker that symlinks to a permanent location).
    ``0`` deletes every existing checkpoint including the one
    just written (use ``1`` to keep only the just-saved file).
    A negative value is a no-op. Pruning happens AFTER
    ``torch.save`` succeeds, so a save failure cannot delete
    existing files.
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
    if keep_last_n is not None:
        prune_old_checkpoints(checkpoint_dir, keep_last_n)
    return path


def prune_old_checkpoints(
    checkpoint_dir: Path,
    keep_last_n: int,
) -> int:
    """Delete ``checkpoint_step_*.pt`` files in ``checkpoint_dir``
    beyond the newest ``keep_last_n``, returning the number of
    files removed.

    Sort key is the step number parsed from the filename
    (``checkpoint_step_{N}.pt``), NOT mtime — wall-clock skew
    between the training process and the filesystem could
    otherwise reorder saves incorrectly (e.g. on a network
    filesystem that lags writes). Files whose names don't match
    the pattern (e.g. ``best.pt``, ``latest.pt`` symlinks, an
    external crash dump) are left untouched.

    Negative ``keep_last_n`` is a no-op. ``0`` deletes every
    existing file (the just-saved one is in the same directory,
    so it IS counted as the newest and preserved; this matches
    the docstring on :func:`save_checkpoint`).
    """
    if keep_last_n < 0:
        return 0
    candidates: list[tuple[int, Path]] = []
    for p in checkpoint_dir.glob("checkpoint_step_*.pt"):
        try:
            # stem is "checkpoint_step_42"; split off the last "_42"
            step_num = int(p.stem.rsplit("_", 1)[-1])
        except (ValueError, IndexError):
            log.debug("ignoring non-conforming checkpoint name: %s", p)
            continue
        candidates.append((step_num, p))
    # Newest first; keep the first keep_last_n, delete the rest.
    candidates.sort(key=lambda x: x[0], reverse=True)
    removed = 0
    for _, p in candidates[keep_last_n:]:
        try:
            p.unlink()
            removed += 1
            log.info("checkpoint pruned (keep_last_n=%d): %s", keep_last_n, p)
        except OSError as e:
            log.warning("failed to prune old checkpoint %s: %r", p, e)
    return removed


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
