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
    cast_dtype, is_mxfp8)`` for the offload hook path.
  * :func:`_quantize_accum_to_mom_buf` — requantizes the int8 /
    mxfp8 accumulator into its storage format (called by
    :meth:`CPUMuon.step` once per cycle).
  * :func:`_scale_accum` — in-place scale of a param's accumulator
    by ``coef`` (used by the grad-norm clip path).

The module deliberately avoids importing :class:`CPUAdamW` /
:class:`CPUMuon` (those live in :mod:`.adamw` / :mod:`.muon`) so
the optimizers and the state layer can be reasoned about
independently. The state fields are documented on
:class:`_ParamState` and are dtype-agnostic across AdamW / Muon
storage formats.
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
    # ``m`` is BF16 by default: the new 5060Ti hardware
    # supports BF16 natively and BF16's 8-bit exponent keeps
    # ``v = g²`` from underflowing for typical grad magnitudes.
    m: torch.Tensor | None = None
    exp_avg_sq: torch.Tensor | None = None
    # Muon-specific state (set when ``kind == 'muon'``). The
    # momentum matrix lives in ``mom_buf`` (any storage dtype:
    # int8 with a BF16 per-row scale, mxfp8 with an E8M0
    # per-block scale, or a raw fp16/bf16/fp32 tensor with no
    # scale). ``mom_buf`` also doubles as the grad accumulator
    # (mu=1 accumulation). At ``step()`` time ``mom_buf`` is
    # read (dequantized for int8/mxfp8), orthogonalized via
    # Newton-Schulz, applied to the GPU param, then reset to
    # zero for the next accumulation cycle. Field names are
    # dtype-agnostic so the ``step()`` and ``loop.py``
    # byte-breakdown code works regardless of the configured
    # storage dtype.
    mom_buf: torch.Tensor | None = None
    mom_scale: torch.Tensor | None = None
    # MXFP8-only: number of elements per E8M0 scale block. ``None``
    # for int8 / fp* storage. The optimizer uses this to reshape
    # ``mom_scale`` from ``[rows, ceil(cols / block_size)]`` to a
    # per-element scale via ``repeat_interleave(block_size, dim=-1)``
    # in the dequant path.
    mxfp8_block_size: int | None = None
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

    # Muon-only: per-microbatch grad accumulator (BF16, pinned).
    # Set ONLY for int8 muon (a separate bf16 ``accum`` lets the
    # per-mb CPU add stay cheap and pushes the int8 requantize to
    # step time). For mxfp8 muon, ``accum`` is ``None`` — the
    # per-mb accumulation flows directly into ``mom_buf`` via the
    # fused C++ kernel (:func:`_mxfp8_apply_grad`), which is the
    # whole point of the mxfp8 design (save the 2 bytes/elt of an
    # extra bf16 accum). For fp* muon, ``mom_buf`` doubles as the
    # accumulator (no separate buffer) — keeps the previously-
    # stable bf16 muon path untouched.
    #
    # Why a separate accumulator exists for quantized muon:
    # when ``mom_buf`` is int8/mxfp8, accumulating a new bf16
    # microbatch grad requires a dequant-add-requant cycle
    # (dequant the existing quantized momentum, add the new
    # bf16 grad, requantize the sum). On CPU this is ~7000 ms
    # per mb at 8-layer smoke scale (and scales linearly to
    # ~32 s/mb at 32 layers) — the dominant cost in
    # ``flush_pending_grads`` and a hard perf cliff for any
    # quantized storage design.
    #
    # A separate bf16 ``accum`` buffer solves this: the per-mb
    # path becomes ``accum.add_(grad_bf16)`` — a single bf16
    # CPU add at ~30 ms for the same size. The expensive
    # dequant-add-requant now happens once per optimizer step
    # (amortized over all microbatches in the cycle) when we
    # requantize ``accum`` back into ``mom_buf`` at the end of
    # :meth:`CPUMuon.step`.
    #
    # Memory cost: doubles the CPU storage for quantized muon
    # (bf16 accum + mxfp8/int8 mom_buf ≈ 3 bytes/elt vs mxfp8-
    # only ≈ 1 byte/elt). Acceptable trade for the per-mb
    # perf win; see ``docs/optimizer_layout.md`` for the
    # full design rationale.
    accum: torch.Tensor | None = None
    # Cached for Muon: original param shape for reshape on GPU.
    shape: tuple = field(default_factory=tuple)
    # Step counter (per param; cheap).
    step: int = 0
    # "adamw" or "muon".
    kind: str = "adamw"


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
    - ``kind`` is ``"adamw"`` or ``"muon"``; diagnostics branch
      on it to pick which state tensors to read.
    - ``m`` / ``exp_avg_sq`` are AdamW's first / second moment
      (both CPU pinned; ``m`` doubles as the grad accumulator
      in the mu=1 accumulation scheme). For Muon both are
      ``None``.
    - ``mom_buf`` / ``mom_scale`` are Muon's momentum storage
      (which doubles as the grad accumulator). ``mom_buf`` is
      at the configured storage dtype (int8 with per-row
      symmetric quantization + BF16 ``mom_scale``; OR mxfp8 with
      per-block E8M0 ``mom_scale``; OR a raw fp16/bf16/fp32
      tensor with ``mom_scale=None``). For AdamW both are
      ``None``.
    - ``mxfp8_block_size`` is the MXFP8-only block size (number
      of elements per E8M0 scale). ``None`` for int8 / fp*
      storage.
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
    mom_scale: Optional[torch.Tensor]
    mxfp8_block_size: Optional[int]
    accum: Optional[torch.Tensor]
    shape: tuple
    step: int


# --------------------------------------------------------------------------- #
# State-only helpers (no optimizer imports).                                  #
# --------------------------------------------------------------------------- #
def _accumulator_target(s: _ParamState) -> Tuple[torch.Tensor, torch.dtype, bool]:
    """Return ``(target_tensor, cast_dtype, is_mxfp8)`` for the
    offload path.

    ``target_tensor`` is the CPU-side tensor that the per-microbatch
    grad is folded into (mu=1 accumulation). For AdamW it's
    ``s.m``; for fp* Muon it's ``s.mom_buf`` directly; for int8
    Muon it's ``s.accum`` (the BF16 mirror that gets requantized
    into ``s.mom_buf`` at step time).

    For mxfp8 Muon, ``s.accum`` is ``None`` and the per-mb
    accumulation flows directly into ``s.mom_buf`` via the
    fused C++ kernel (:func:`CPUMuon._mxfp8_apply_grad`).
    ``target_tensor`` is set to ``s.mom_buf`` (so the hook's
    bookkeeping is uniform), but the flush dispatches based on
    ``is_mxfp8`` and calls the kernel instead of ``.add_()``.

    ``cast_dtype`` is the dtype to cast the GPU grad to before
    the DMA. Always BF16 for the accumulator side — both
    ``s.m`` (AdamW) and ``s.accum`` (int8 muon) are BF16. For
    fp* muon (no separate accum), the cast matches the storage
    dtype of ``s.mom_buf``. For mxfp8, the kernel reads BF16.
    """
    if s.kind == "adamw":
        return s.m, s.m.dtype, False
    # muon
    if s.mxfp8_block_size is not None:
        # mxfp8 muon: the fused kernel does in-place
        # dequant+add+requant. ``mom_buf`` is the cycle-sum.
        # The cast is BF16 (kernel reads bf16 grads); the
        # flush dispatches through ``_mxfp8_apply_grad``
        # instead of ``.add_()``.
        return s.mom_buf, torch.bfloat16, True
    if s.accum is not None:
        # int8 muon: ``accum`` (BF16) is the per-mb
        # accumulator. The cast target is BF16 — a cheap CPU
        # bf16 add, not the old dequant-add-requant cycle.
        return s.accum, torch.bfloat16, False
    # fp* muon: merged-accumulator design, ``mom_buf`` IS the
    # accumulator. Cast matches the storage dtype.
    return s.mom_buf, s.mom_buf.dtype, False


def _quantize_accum_to_mom_buf(s: _ParamState) -> None:
    """Requantize ``s.accum`` (BF16, the per-mb grad sum) into
    ``s.mom_buf`` / ``s.mom_scale`` in the configured storage
    format (int8 with per-row BF16 scale, or mxfp8 with per-block
    E8M0 scale). Runs ONCE per optimizer step (not per
    microbatch), amortized over the accumulation cycle.

    Replaces the old per-mb ``_int8_muon_accumulate`` /
    ``_mxfp8_muon_accumulate`` dequant-add-requant cycle. The
    bf16 ``accum`` accumulator is summed cheaply across mbs
    (just a CPU bf16 add), then we pay one requantize at step
    time to materialize the cycle's gradient into the chosen
    storage format. For 32-layer prod at 16 mbs/step the
    per-mb sync cost drops from ~32 s (old) to ~0.7 s (new);
    the single-step requantize adds ~8 s back, for a net
    3-4x speedup at the optimizer layer.

    Storage format dispatch:
      - ``s.mxfp8_block_size is not None``: mxfp8 (E4M3 + E8M0
        per-block scale, default block_size=32). Pads to
        ``cols_padded`` for the tail block.
      - ``s.mom_scale is not None`` and mxfp8 is not set: int8
        with per-row BF16 scale.
      - fp* storage: not handled here — fp* muon keeps the
        merged-accumulator design (``mom_buf`` IS ``accum``).

    Runs on the CPU. The FP32 work is the only allocation;
    ``mom_buf`` and ``mom_scale`` are updated in place via
    ``copy_``.
    """
    rows, cols = s.shape[0], s.shape[1]
    accum_2d = s.accum.view(rows, cols).float()
    if s.mxfp8_block_size is not None:
        # mxfp8 path: E4M3 + per-block E8M0 scale (OCP MX).
        bs = s.mxfp8_block_size
        cols_p = cols if cols % bs == 0 else cols + (bs - cols % bs)
        n_blocks = cols_p // bs
        # Pad right if cols is not a multiple of bs (the tail
        # block contributes nothing on dequant — it stays zero).
        if cols_p != cols:
            m_padded = torch.nn.functional.pad(accum_2d, (0, cols_p - cols))
        else:
            m_padded = accum_2d
        blocks = m_padded.view(rows, n_blocks, bs)
        # Per-block absmax in FP32; E4M3 max is 448.
        absmax = blocks.abs().amax(dim=-1).float()
        target_scale = (absmax / 448.0).clamp(min=2 ** -127)
        log2 = target_scale.log2()
        e_unclamped = log2.round() + 127.0
        e = e_unclamped.clamp(min=1.0, max=254.0)
        new_scales = e.to(torch.uint8).view(torch.float8_e8m0fnu)
        # ``.float()`` decodes E8M0 bytes to their value
        # (2^(b-127)); the bitcast path through uint8 would give
        # the byte itself.
        new_scale_fp32 = new_scales.float().view(rows, n_blocks, 1)
        scaled = (blocks.float() / new_scale_fp32).clamp(-448.0, 448.0)
        q_e4m3 = scaled.to(torch.float8_e4m3fn)
        # ``copy_`` requires matching shapes (not just element
        # count) — flatten and view as the destination shape to
        # dodge the shape-mismatch error.
        s.mom_buf.copy_(q_e4m3.reshape(-1).view(s.mom_buf.shape))
        s.mom_scale.copy_(new_scales.view(s.mom_scale.shape))
    else:
        # int8 path: per-row symmetric int8 + BF16 per-row scale.
        # Per-row absmax, scale = absmax / 127, clamp to 1e-8 so a
        # dead row doesn't divide by zero.
        row_max = accum_2d.abs().amax(dim=1).clamp(min=1e-8)
        new_scale_fp32 = row_max / 127.0
        new_scale_bf16 = new_scale_fp32.to(torch.bfloat16)
        scale_2d = new_scale_bf16.float().unsqueeze(1)
        q_int8 = (accum_2d / scale_2d).round().clamp(-128, 127).to(torch.int8)
        s.mom_buf.copy_(q_int8.view(-1))
        s.mom_scale.copy_(new_scale_bf16)


def _scale_accum(s: _ParamState, coef: float) -> None:
    """In-place scale of the per-param accumulator (the cycle's
    grad sum) by ``coef``. Used by
    :func:`src.training.loop._compute_and_clip_grad_norm` to
    clip the global L2 grad norm.

    Resolves the right accumulator for the param:
      - AdamW: ``s.m`` (BF16, the merged accumulator + first
        moment).
      - Muon int8 / fp* / AdamW: ``s.accum`` (BF16) when set,
        else ``s.mom_buf`` (merged design).
      - Muon mxfp8: ``s.mom_buf`` (E4M3+E8M0). The clip does a
        dequant-mul-requant via :func:`_requantize_mxfp8` —
        expensive (~one fused-kernel-pass per clipped param),
        but only fires when the norm exceeds the cap (a rare
        event after warm-up). This is the price of the
        no-separate-accum mxfp8 design.

    No-op when ``coef == 1.0`` (cheap guard so the grad-norm
    clip stays a no-op for well-behaved steps).
    """
    if coef == 1.0:
        return
    if s.kind == "adamw":
        s.m.mul_(coef)
    elif s.accum is not None:
        s.accum.mul_(coef)
    elif s.mxfp8_block_size is not None:
        # mxfp8: dequant, scale, requant via the PyTorch
        # helpers. Equivalent numerically to scaling the
        # underlying fp32 accumulator — the dequant/requant
        # carries the bf16 precision loss through. The
        # optimizer-attached path calls ``_dequantize_mxfp8``
        # / ``_requantize_mxfp8`` on the ``CPUMuon`` instance
        # (see :mod:`.muon`); both are duck-typed on the
        # ``opt`` we recover from the state's ``_optimizer``
        # attribute (set by :func:`register_grad_offload_hooks`
        # at hook-install time).
        opt = getattr(s, "_optimizer", None)
        if opt is None:
            # No optimizer attached (test path) — fall back to
            # a plain dequant-mul-requant via PyTorch ops.
            bs = s.mxfp8_block_size
            rows, cols = s.shape[0], s.shape[1]
            cols_p = (cols if cols % bs == 0
                       else cols + (bs - cols % bs))
            n_blocks = cols_p // bs
            q_blocks = s.mom_buf.view(rows, n_blocks, bs).float()
            scale_fp32 = s.mom_scale.float().view(rows, n_blocks, 1)
            m_fp32 = (q_blocks * scale_fp32).view(rows, cols_p)
            m_fp32[:, :cols] *= coef
            E4M3_MAX = 448.0
            absmax = (m_fp32.view(rows, n_blocks, bs)
                      .abs().amax(dim=-1).float())
            target_scale = (absmax / E4M3_MAX) \
                              .clamp(min=2 ** -127)
            log2 = target_scale.log2()
            e_unclamped = log2.round() + 127.0
            e = e_unclamped.clamp(min=1.0, max=254.0)
            new_scales = e.to(torch.uint8).view(torch.float8_e8m0fnu)
            scale_fp32_new = new_scales.float() \
                                .view(rows, n_blocks, 1)
            scaled = (m_fp32.view(rows, n_blocks, bs) / scale_fp32_new) \
                        .clamp(-E4M3_MAX, E4M3_MAX)
            q_e4m3 = scaled.to(torch.float8_e4m3fn)
            s.mom_buf.copy_(q_e4m3.reshape(-1).view(s.mom_buf.shape))
            s.mom_scale.copy_(new_scales.view(s.mom_scale.shape))
        else:
            # Optimizer-attached path: use the existing helpers.
            m_fp32 = opt._dequantize_mxfp8(s)
            m_fp32 *= coef
            opt._requantize_mxfp8(m_fp32, s)
    else:
        # fp* muon (merged design).
        s.mom_buf.mul_(coef)