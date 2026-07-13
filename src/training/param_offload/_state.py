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
``accum``) are gone; only full-precision ``mom_buf`` storage
remains for Muon.
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
    """Per-parameter CPU-side state for either AdamW or Muon."""

    param: nn.Parameter
    # AdamW-specific state (set when ``kind == 'adamw'``).
    # m doubles as the grad accumulator and the optimizer's
    # first moment: each microbatch's ``.grad`` is added to
    # ``m`` in place (mu=1 accumulation — no cross-step β1
    # EMA), and at ``step()`` time ``m`` is consumed as the
    # "gradient" estimate in the AdamW update:
    #     v = β2*v + (1-β2)*m²
    #     update = m / (sqrt(v) + eps)
    # (no β1 division; the bias-correction on m is unnecessary
    # because m is unbiased — it's the raw sum of microbatch
    # grads).
    # ``m`` is BF16 by default: BF16's 8-bit exponent keeps
    # ``v = g²`` from underflowing for typical grad magnitudes.
    m: torch.Tensor | None = None
    exp_avg_sq: torch.Tensor | None = None
    # Muon-specific state (set when ``kind == 'muon'``). The
    # momentum matrix lives in ``mom_buf`` at the configured
    # full-precision storage dtype (bf16 / fp16 / fp32). It
    # doubles as the grad accumulator (mu=1 accumulation).
    # At ``step()`` time ``mom_buf`` is read, orthogonalized
    # via Newton-Schulz, applied to the GPU param, then reset
    # to zero for the next accumulation cycle. Field names
    # are dtype-agnostic so the ``step()`` and ``loop.py``
    # byte-breakdown code works regardless of the configured
    # storage dtype.
    mom_buf: torch.Tensor | None = None
    # NVFP4-mode-3 ("no BF16 master") support. When this state entry
    # backs an :class:`NVFP4*Linear` (or its TP variants) constructed
    # with ``no_bf16_master=True``, ``param`` is ``None`` and the
    # storage lives on the module as FP4 packed buffers. The optimizer
    # side keeps its CPU pinned m / v as BF16 (these are optimizer
    # moments, NOT the weight — the user constraint was "no BF16 master
    # weight"; the moments have always been CPU BF16 in this codebase).
    # The optimizer computes the per-chunk update on the CPU pinned
    # gradient slice, then calls ``module.apply_chunk_update(start,
    # end, grad_chunk, lr)`` which materializes a temp BF16 view,
    # applies the update in place, and re-quantizes to FP4.
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
    # reads ``module._latest_grad_w`` and folds it into ``s.m`` /
    # ``s.mom_buf`` once per microbatch.
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
    - ``m`` / ``exp_avg_sq`` are AdamW's first / second moment
      (both CPU pinned; ``m`` doubles as the grad accumulator
      in the mu=1 accumulation scheme). For Muon both are
      ``None``.
    - ``mom_buf`` is Muon's momentum storage (which doubles as
      the grad accumulator). At the configured full-precision
      dtype (bf16 / fp16 / fp32 — quantized storage removed
      2026-07-12). For AdamW it's ``None``.
    - ``shape`` is cached for Muon (the original param shape
      before flatten) so diagnostics can report the failing
      param's full shape on NaN/Inf.
    - ``step`` is the per-param step counter; primarily for
      warmup-aware step reporting.
    """

    param: nn.Parameter
    kind: str
    m: Optional[torch.Tensor]
    exp_avg_sq: Optional[torch.Tensor]
    mom_buf: Optional[torch.Tensor]
    shape: tuple
    step: int


# --------------------------------------------------------------------------- #
# State-only helpers (no optimizer imports).                                  #
# --------------------------------------------------------------------------- #
def _accumulator_target(s: _ParamState) -> Tuple[torch.Tensor, torch.dtype]:
    """Return ``(target_tensor, cast_dtype)`` for the offload path.

    ``target_tensor`` is the CPU-side tensor that the
    per-microbatch grad is folded into (mu=1 accumulation). For
    AdamW it's ``s.m``; for Muon it's ``s.mom_buf`` directly
    (the merged-accumulator design — quantised storage and the
    separate ``accum`` buffer were removed 2026-07-12).

    ``cast_dtype`` is the dtype to cast the GPU grad to before
    the DMA. It matches the accumulator's storage dtype for
    both optimizers (BF16 for AdamW's ``m`` and for the
    canonical Muon bf16 ``mom_buf``).
    """
    if s.kind == "adamw":
        return s.m, s.m.dtype
    # Muon (and the nvfp4 variants): the merged-accumulator
    # design is in effect — mom_buf is the cycle's accumulator.
    return s.mom_buf, s.mom_buf.dtype


def _scale_accum(s: _ParamState, coef: float) -> None:
    """In-place scale of the per-param accumulator (the cycle's
    grad sum) by ``coef``. Used by
    :func:`src.training.loop._compute_and_clip_grad_norm` to
    clip the global L2 grad norm.

    Resolves the right accumulator for the param:
      - AdamW: ``s.m`` (BF16, the merged accumulator + first
        moment).
      - Muon: ``s.mom_buf`` (BF16 / FP16 / FP32, the merged
        accumulator).

    No-op when ``coef == 1.0`` (cheap guard so the grad-norm
    clip stays a no-op for well-behaved steps).
    """
    if coef == 1.0:
        return
    if s.kind == "adamw":
        s.m.mul_(coef)
    else:
        # Muon (and nvfp4 variants): merged-accumulator design.
        s.mom_buf.mul_(coef)
