"""Stateless helpers for the training loop.

Two functions live here:

  * :func:`wsd_lr` — the Warmup-Stable-Decay LR schedule used
    by both the Muon and AdamW optimizers. Reading the LR off
    the schedule each step (``muon_opt.lr = ...``,
    ``adamw_opt.lr = ...``) and relying on the optimizers to
    read ``self.lr`` at :meth:`step` time is enough — no
    ``torch.optim.lr_scheduler`` wrapper.
  * :func:`_slice_cu_seqlens` — projects a global
    ``cu_seqlens`` tensor (cumulative doc-end offsets across
    the super-long packed sequence) onto a chunk-local
    ``[chunk_start, chunk_end)`` range. Used by the per-chunk
    forward to re-derive the doc-boundary markers the
    ShortConvolution kernel resets at.

The module deliberately has **no internal dependencies** on
the rest of :mod:`src.training.loop` — both helpers can be
imported (and unit-tested) in isolation. The training-loop
phases (:mod:`.setup`, :mod:`.run`, :mod:`.teardown`) import
from here.
"""
from __future__ import annotations

import torch


def wsd_lr(
    step: int,
    peak_lr: float,
    warmup_steps: int,
    decay_steps: int,
    max_steps: int,
    min_lr: float = 0.0,
) -> float:
    """WSD (Warmup-Stable-Decay) learning rate schedule.

    Three phases over the ``max_steps`` training horizon:

      1. **Warmup**  : steps ``[0, warmup_steps)`` —
         linear ramp from ``0`` to ``peak_lr``.
      2. **Stable**  : steps ``[warmup_steps, max_steps - decay_steps)`` —
         constant at ``peak_lr``.
      3. **Decay**   : steps ``[max_steps - decay_steps, max_steps)`` —
         linear ramp from ``peak_lr`` to ``min_lr``.

    Edge cases:

    - ``step < 0`` or ``step >= max_steps``  : ``0`` (clamped; the
      training loop never reaches these but tests / callers may).
    - ``warmup_steps == 0`` : no warmup phase; step 0 is already at
      ``peak_lr``.
    - ``decay_steps == 0``  : no decay phase; LR stays at ``peak_lr``
      to the end (matches the pre-WSD "constant LR" behavior).
    - ``warmup + decay > max_steps`` : stable phase shrinks to zero;
      the warmup phase is still walked first, then the decay phase
      kicks in at ``max_steps - decay_steps`` (overlap region is
      governed by phase priority: warmup > stable > decay).
    - ``peak_lr <= 0`` : returns ``peak_lr`` unchanged (no-op).

    Both optimizers in this codebase (``CPUAdamW`` and ``CPUMuon``)
    read ``self.lr`` at the start of :meth:`step`, so updating
    ``muon_opt.lr`` and ``adamw_opt.lr`` between optimizer
    constructions and each step is enough — no
    ``torch.optim.lr_scheduler`` wrapper needed.
    """
    if peak_lr <= 0.0:
        return peak_lr
    if step < 0 or step >= max_steps:
        return 0.0
    # Phase 1: warmup.
    if warmup_steps > 0 and step < warmup_steps:
        # step 0 -> 0, step warmup_steps-1 -> peak * (warmup-1)/warmup.
        # The very first stable step (== warmup_steps) hits peak.
        return peak_lr * step / warmup_steps
    # Phase 3: decay.
    decay_start = max_steps - decay_steps
    if decay_steps > 0 and step >= decay_start:
        # step decay_start -> peak_lr, step max_steps-1 -> min_lr.
        # Decay spans ``decay_steps`` steps [decay_start, decay_start +
        # decay_steps - 1]; divisor is ``decay_steps - 1`` so the
        # final step lands exactly on min_lr. With decay_steps == 1
        # the single decay step jumps straight to min_lr.
        denom = decay_steps - 1 if decay_steps > 1 else 1
        progress = (step - decay_start) / denom
        return peak_lr + (min_lr - peak_lr) * progress
    # Phase 2: stable.
    return peak_lr


def _slice_cu_seqlens(
    global_cu_seqlens: torch.Tensor,
    chunk_start: int,
    chunk_end: int,
) -> torch.Tensor:
    """Slice a global ``cu_seqlens`` tensor to a chunk-local one.

    The chunked-training invariant: each chunk covers tokens at
    global offsets ``[chunk_start, chunk_end)`` within the
    super-long packed sequence. Doc boundaries (encoded in
    ``global_cu_seqlens`` as cumulative end-offsets) that fall
    inside this range become boundaries in the chunk-local
    ``cu_seqlens``; boundaries outside are dropped.

    The returned tensor:

      - starts at ``0`` (chunk-local origin),
      - lists each global doc-end that lies in
        ``(chunk_start, chunk_end]`` shifted by ``-chunk_start``,
      - ends at ``chunk_end - chunk_start`` (chunk-local right edge).

    Edge cases:

      - A global boundary exactly at ``chunk_start`` becomes the
        chunk's local ``0`` (the chunk opens a new "doc" — same
        convention as :func:`pack_chunk_aligned` where the
        ``pack_end`` entry resets the KDA + ShortConv state).
      - A global boundary exactly at ``chunk_end`` becomes the
        chunk's local ``chunk_size`` (the chunk closes its last
        doc at the chunk's right edge).
      - Empty chunk (``chunk_start == chunk_end``) returns
        ``[0]`` (a degenerate single-entry sequence).

    This is what the model receives as ``cu_seqlens`` for the
    ShortConvolution call (depthwise conv kernel resets at doc
    boundaries). The KDA sub-layer itself does NOT receive
    ``cu_seqlens`` — see ``TPKDA.forward``.
    """
    if chunk_start == chunk_end:
        # Empty chunk: degenerate but well-defined.
        return global_cu_seqlens.new_zeros(1)

    in_range = (global_cu_seqlens >= chunk_start) & (
        global_cu_seqlens <= chunk_end
    )
    local = global_cu_seqlens[in_range] - chunk_start
    # Defensive: ensure 0 at the start and chunk_size at the end.
    chunk_size = chunk_end - chunk_start
    if local.numel() == 0 or local[0].item() != 0:
        local = torch.cat([
            local.new_zeros(1, dtype=local.dtype), local,
        ])
    if local[-1].item() != chunk_size:
        local = torch.cat([
            local, local.new_full((1,), chunk_size, dtype=local.dtype),
        ])
    return local