"""Per-parameter CPU state descriptor + state-only helpers.

This module is the data layer of :mod:`src.training.param_offload`.
It owns:

  * :class:`_ParamState` — the per-parameter CPU-side state
    dataclass used by both optimizers.
  * :class:`OptimizerState` — a :class:`typing.Protocol` declaring
    the minimum shape any per-param state must expose (used by
    :mod:`src.training.diagnostics` to read both optimizers'
    state uniformly without importing the concrete classes).
  * :func:`_accumulator_target` — resolves ``(target_tensor,
    cast_dtype)`` for the offload hook path.
  * :func:`_scale_accum` — in-place scale of a param's accumulator
    by ``coef`` (used by the grad-norm clip path).

The module deliberately avoids importing :class:`CPUAdamW` /
:class:`CPUMuon` (those live in :mod:`.adamw` / :mod:`.muon`) so
the optimizers and the state layer can be reasoned about
independently. The state fields are documented on
:class:`_ParamState` and are dtype-agnostic across AdamW / Muon
storage formats.

History
-------
Quantized muon storage (``int8`` + BF16 per-row scale,
``mxfp8`` + E8M0 per-block scale, and the separate ``accum``
BF16 buffer that was paired with ``int8`` to avoid the
per-mb dequant-add-requant cycle) was removed on 2026-07-12
after long-training runs showed quantization-error
accumulation destabilizing optimization. The state fields
that supported those paths (``mom_scale``, ``mxfp8_block_size``,
``accum``) are gone.

On 2026-07-15 the "merged-accumulator" / mu=1 trick was deleted
in favour of explicit, separately-named buffers:

  * ``s.grad``      — the per-step CPU grad accumulator (BF16,
    one ``.add_()`` per microbatch, zeroed at the end of every
    optimizer step). This is the buffer the post-accumulate-grad
    hook folds each microbatch's grad into. *Why this MUST live
    on CPU*: see the long note in
    :mod:`src.training.loop.grad_norm` — the global L2 grad
    norm is computed across all per-param ``s.grad`` tensors in
    a single fused C++ pass (``fused_l2_norm_sq_bf16``) so that
    grad-clip can be done in one TP all-reduce. Hoisting the
    accumulator back to the GPU just to clip would require an
    extra H2D pass per param + a second reduction; the CPU
    accumulator is the natural single source of truth.

  * ``s.exp_avg``   — AdamW's first moment EMA (β1) / Muon's
    SGD momentum (``β·prev + grad``); preserved across
    optimizer steps. The optimizer step reads ``s.grad`` to
    update ``s.exp_avg``; ``s.grad`` is then zeroed.

  * ``s.exp_avg_sq`` — AdamW's second moment EMA (β2); AdamW
    only. Muon does not use a second moment.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Protocol, Tuple

import torch
import torch.nn as nn


# --------------------------------------------------------------------------- #
# Per-param state descriptor.                                                 #
# --------------------------------------------------------------------------- #
@dataclass
class _ParamState:
    """Per-parameter CPU-side state for either AdamW or Muon.

    Field semantics
    ---------------
    - ``param``: the GPU-side trainable parameter the entry
      belongs to. ``None`` for NVFP4-mode-3 entries (no leaf
      Parameter — the weight lives on the module as FP4 packed
      buffers).
    - ``grad``: per-step CPU grad accumulator. Each microbatch's
      ``.grad`` is ``.add_()``'d into this BF16 pinned buffer
      via the post-accumulate-grad hook (or the manual
      ``accumulate_grads_to_cpu`` path). Zeroed at the end of
      every optimizer step (or by
      :func:`zero_cpu_grad_accum` when ``found_inf`` skips the
      step). The end-of-step grad-norm clip + TP all-reduce
      operate on this tensor — see
      :mod:`src.training.loop.grad_norm`.
    - ``exp_avg``: AdamW's first moment EMA / Muon's SGD
      momentum. Updated in-place by the optimizer step from
      ``grad`` (AdamW: ``β1·prev + (1-β1)·grad``; Muon:
      ``β·prev + grad``). **Preserved across steps** so the EMA
      / momentum smoothing actually has cross-step state to
      smooth. Initial value is zero.
    - ``exp_avg_sq``: AdamW's second moment EMA. Updated in
      place by the AdamW step kernel. ``None`` for Muon entries.
    - ``nvfp4_module`` / ``nvfp4_n`` / ``nvfp4_chunk_ranges``
      / ``nvfp4_grad``: NVFP4-mode-3 ("no BF16 master") plumbing.
      Set when the optimizer's :meth:`register_nvfp4_module`
      binds an :class:`NVFP4*Linear` (``no_bf16_master=True``)
      to this entry. The optimizer step routes the apply
      through ``module.apply_chunk_update``; ``param`` is
      ``None`` on these entries.
    - ``shape``: cached param shape (used by Muon's reshape on
      GPU). For NVFP4 entries this is the module's
      ``(out_features, in_features_per_partition)``.
    - ``step``: per-param step counter (used for bias-correction
      on ``exp_avg_sq``).
    - ``kind``: ``"adamw"`` / ``"muon"`` / ``"adamw_nvfp4"`` /
      ``"muon_nvfp4"``. The optimizer step + the diagnostics
      branch on it.
    - ``factor_chunk``: reusable FP32 pinned scratch for the
      fused AdamW step kernel's per-chunk factor output.
    """

    param: nn.Parameter
    # Per-step CPU grad accumulator (BF16, numel). This is where
    # the post-accumulate-grad hook (or the manual
    # ``accumulate_grads_to_cpu`` path) folds each microbatch's
    # ``.grad`` into. Zeroed at the end of every optimizer step.
    # The end-of-step grad-norm clip + TP all-reduce operate on
    # this tensor — see :mod:`src.training.loop.grad_norm`.
    grad: torch.Tensor | None = None
    # AdamW's first moment EMA / Muon's SGD momentum. Updated
    # in place by the optimizer step from ``grad`` (AdamW:
    # ``β1·prev + (1-β1)·grad``; Muon: ``β·prev + grad``);
    # **preserved across steps** so the EMA / momentum
    # smoothing has cross-step state. Initial value is zero.
    exp_avg: torch.Tensor | None = None
    # AdamW's second moment EMA (β2). ``None`` for Muon entries.
    # Updated in place by the AdamW step kernel. Preserved
    # across steps.
    exp_avg_sq: torch.Tensor | None = None
    # NVFP4-mode-3 ("no BF16 master") support. When this state entry
    # backs an :class:`NVFP4*Linear` (or its TP variants) constructed
    # with ``no_bf16_master=True``, ``param`` is ``None`` and the
    # storage lives on the module as FP4 packed buffers. The optimizer
    # side keeps its CPU pinned grad / exp_avg / exp_avg_sq as BF16
    # (these are optimizer moments, NOT the weight — the user
    # constraint was "no BF16 master weight"; the moments have
    # always been CPU BF16 in this codebase).
    nvfp4_module: object | None = None
    # Per-param shape preserved when ``param is None`` (used by the
    # chunk math in step()).
    nvfp4_n: int = 0
    # Cached ``module.chunk_ranges()`` so we don't re-materialize it
    # every chunk. Invalidated when the module changes its row count
    # (never happens in production; FFN module shapes are static).
    nvfp4_chunk_ranges: list = field(default_factory=list)
    # The side-channel grad the autograd Function stashes on the
    # module between forward/backward and the optimizer step. The
    # post-accumulate-grad hook (:func:`register_grad_offload_hooks`)
    # reads ``module._latest_grad_w`` and folds it into ``s.grad``
    # once per microbatch.
    nvfp4_grad: object | None = None
    # Cached for Muon: original param shape for reshape on GPU.
    shape: tuple = field(default_factory=tuple)
    # Step counter (per param; cheap).
    step: int = 0
    # "adamw" or "muon" or "muon_nvfp4" or "adamw_nvfp4".
    kind: str = "adamw"
    # Reusable FP32 pinned scratch buffer for the fused AdamW
    # step kernel's factor output (one buffer per state,
    # lazily allocated in :meth:`CPUAdamW.step`; size =
    # ``_STREAM_CHUNK_NUMEL`` for param-backed states, sized to
    # the largest NVFP4 chunk otherwise). Keeps the streaming
    # factor + GPU apply loop from re-allocating per chunk.
    factor_chunk: torch.Tensor | None = None
    # --- FP8 2D tight scale EMA storage (Muon-only) ---
    # When ``exp_avg_storage="fp8_2d_tight"`` is set on
    # :class:`CPUMuon`, the SGD momentum EMA ``exp_avg`` is
    # stored as E4M3 + per-(row × col-block) + per-(col × row-block)
    # FP32 scales (the ``lowbit-Muon/quantize_2d`` fine-grained
    # scheme) instead of BF16. The CPUMuon step() dequants the
    # FP8 view to BF16 in-place, runs the standard EMA
    # ``β·prev + grad`` on the BF16 view, then requantizes.
    # ``exp_avg`` stays ``None`` in this mode; the BF16
    # dequant target is ``exp_avg_bf16`` (a CPU scratch
    # buffer that's re-quantized at the end of every step).
    # All four fields are ``None`` for the default BF16 path.
    exp_avg_q: torch.Tensor | None = None
    exp_avg_scale_dim1: torch.Tensor | None = None
    exp_avg_scale_dim2: torch.Tensor | None = None
    exp_avg_bf16: torch.Tensor | None = None
    # Cached param shape (used for 2D reshape on dequant); separate
    # from ``shape`` (which is also used for the GPU reshape) to
    # avoid coupling the two codepaths.
    exp_avg_2d_shape: tuple = field(default_factory=tuple)
    # Cached block size (matches the quantize/dequantize block).
    exp_avg_block: int = 0


# --------------------------------------------------------------------------- #
# Public protocol for per-param state.                                        #
# --------------------------------------------------------------------------- #
class OptimizerState(Protocol):
    """Structural view of a per-param optimizer state entry.

    The training diagnostics (see :mod:`src.training.diagnostics`)
    only need to read a handful of fields off each per-param
    state, but they need to do so uniformly across the two
    optimizers (AdamW and Muon) that the training loop
    instantiates. This :class:`Protocol` declares the minimum
    shape any optimizer state entry must expose to be
    diagnosable.

    Implementations: :class:`_ParamState` (the only concrete
    state in this module, used by both :class:`CPUAdamW` and
    :class:`CPUMuon`). ``CPUAdamW.state`` and ``CPUMuon.state``
    are ``Dict[int, _ParamState]`` and therefore structurally
    satisfy this protocol.

    Field semantics:

    - ``param`` is the GPU-side trainable parameter the entry
      belongs to. Used to read ``.data`` (post-step) for the
      "is the param finite" check.
    - ``kind`` is ``"adamw"`` / ``"muon"`` / ``"muon_nvfp4"`` /
      ``"adamw_nvfp4"``; diagnostics branch on it to pick which
      state tensors to read.
    - ``grad`` is the per-step CPU grad accumulator (BF16
      pinned). The end-of-step grad-norm clip + TP all-reduce
      operate on this tensor. Zeroed by the optimizer step (or
      by :func:`zero_cpu_grad_accum`).
    - ``exp_avg`` is AdamW's first moment EMA / Muon's SGD
      momentum — preserved across steps.
    - ``exp_avg_sq`` is AdamW's second moment EMA (AdamW only;
      ``None`` for Muon).
    - ``shape`` is cached for Muon (the original param shape
      before flatten) so diagnostics can report the failing
      param's full shape on NaN/Inf.
    - ``step`` is the per-param step counter; primarily for
      warmup-aware step reporting.
    """

    param: nn.Parameter
    kind: str
    grad: Optional[torch.Tensor]
    exp_avg: Optional[torch.Tensor]
    exp_avg_sq: Optional[torch.Tensor]
    shape: tuple
    step: int


# --------------------------------------------------------------------------- #
# State-only helpers (no optimizer imports).                                  #
# --------------------------------------------------------------------------- #
def _accumulator_target(s: _ParamState) -> Tuple[torch.Tensor, torch.dtype]:
    """Return ``(target_tensor, cast_dtype)`` for the offload path.

    ``target_tensor`` is the CPU-side tensor that the
    per-microbatch grad is folded into: ``s.grad`` (both AdamW
    and Muon — the post-2026-07-15 layout; the old "merged
    accumulator" design where ``s.m`` / ``s.mom_buf`` doubled
    as the accumulator was removed).

    ``cast_dtype`` is the dtype to cast the GPU grad to before
    the DMA. It matches the accumulator's storage dtype for
    both optimizers (BF16 for AdamW's ``grad`` and for the
    canonical Muon bf16 ``grad``).
    """
    return s.grad, s.grad.dtype


def _scale_accum(s: _ParamState, coef: float) -> None:
    """In-place scale of the per-param grad accumulator by
    ``coef``. Used by
    :func:`src.training.loop._compute_and_clip_grad_norm` to
    clip the global L2 grad norm.

    Resolves the right accumulator for the param:
      - AdamW: ``s.grad`` (BF16, the per-step accumulator).
      - Muon:  ``s.grad`` (BF16, the per-step accumulator;
        was originally ``precision.muon_momentum`` —
        yml-side ``precision:`` block removed 2026-07-23).

    No-op when ``coef == 1.0`` (cheap guard so the grad-norm
    clip stays a no-op for well-behaved steps).
    """
    if coef == 1.0:
        return
    s.grad.mul_(coef)