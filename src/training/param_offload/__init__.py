"""Per-parameter optimizers with CPU offload for HippoLM.

Implements two optimizer variants used together via param groups:

  - :class:`CPUAdamW`     — AdamW with explicit per-step ``grad``
    accumulator (BF16) plus ``exp_avg`` (β1 EMA, BF16) and
    ``exp_avg_sq`` (β2 EMA, BF16) state on CPU pinned memory.
    The forward / backward happen on the GPU in the param's
    training dtype (FP16 in this project). After each
    micro-batch's backward, the per-param post-accumulate-grad
    hook DMA's the GPU ``.grad`` (cast to ``grad.dtype``) and
    folds it into ``grad`` in place on the CPU (BF16, pinned).
    On ``step()``:
      1. ``exp_avg ← β1·exp_avg + (1-β1)·grad``
         (first moment EMA; BF16 in place).
      2. ``exp_avg_sq ← β2·exp_avg_sq + (1-β2)·exp_avg²``
         (second moment EMA; BF16 in place).
      3. Factor ``exp_avg / (sqrt(exp_avg_sq/bc2) + eps)`` is
         computed in FP32 (to recover precision after the BF16
         sqrt), then cast to FP16 and streamed chunk-wise to
         the GPU param and applied in place.
      4. ``grad`` is zeroed for the next accumulation cycle
         (the EMA ``exp_avg`` / ``exp_avg_sq`` are preserved
         across steps so the smoothing has cross-step state).
    Grad dtype is BF16 throughout (matches the new 5060Ti
    hardware; BF16 has FP32-like dynamic range, so no overflow
    on the grad transfer or the EMA update, and v can stay BF16
    because BF16's 8-bit exponent is wide enough to keep
    ``exp_avg_sq = g²`` from underflowing at typical grad
    magnitudes 1e-4 to 1e-3 — 1e-8 is well above BF16's
    smallest normal of ~1.18e-38).

  - :class:`CPUMuon`      — Muon (Newton-Schulz orthogonalization
    of the momentum matrix). The per-step ``grad`` accumulator
    is a dedicated BF16 / FP16 / FP32 pinned buffer; the SGD
    momentum lives in ``exp_avg``. On ``step()``:
      1. ``exp_avg ← β·exp_avg + grad`` (CPU, BF16 in place;
         Keller Jordan's Muon reference formulation, no (1-β)
         factor).
      2. If nesterov: ``g_for_ns = grad + β·exp_avg``;
         else: ``g_for_ns = exp_avg``. (CPU, in storage dtype.)
      3. H2D ``g_for_ns``, cast to FP32, view as 2-D.
      4. Stream NS over rows, apply the update to the GPU param.
      5. ``grad`` is zeroed for the next cycle (the EMA
         ``exp_avg`` is preserved across steps).
    Momentum storage is BF16 throughout (the only supported
    layout since the yml-side ``precision:`` block that let
    operators pick among ``bf16`` / ``fp16`` / ``fp32`` was
    removed 2026-07-23). Both ``grad`` and ``exp_avg`` are
    at BF16; the NS iteration is unaffected (it
    orthogonalizes the raw momentum in FP32 on the GPU
    regardless).
      Quantized storage (``int8`` per-row BF16 scale,
      ``mxfp8`` per-block E8M0 scale) was removed on
      2026-07-12 after long-training runs showed
      quantization-error accumulation destabilizing
      optimization. Pre-removal code is preserved on the
      ``archive/int8-mxfp8-muon`` branch.
      See ``docs/optimizer_layout.md``.

Both optimizers follow the same API surface as ``torch.optim.Optimizer``
minimally — ``step()`` consumes whatever gradients are present on
the parameters and ``zero_grad()`` clears them.

No gradient is stored on the GPU between steps. The training loop
copies each micro-batch's ``.grad`` to CPU and adds it to the
optimizer state's ``grad`` accumulator; only after
``gradient_accumulation_steps`` micro-batches does ``step()``
actually run.

Conventions
-----------
- Param ``.data`` lives on the GPU (FP16).
- ``.grad`` is materialised on the GPU only during ``backward()``;
  immediately after, the training loop calls the per-param
  post-accumulate-grad hook (registered via
  :func:`register_grad_offload_hooks`) which copies ``.grad``
  into the per-param CPU ``grad`` accumulator and frees the
  GPU copy. So between micro-batches the GPU holds zero gradient
  memory.
- Optimizer state on CPU pinned memory:
    AdamW: grad (BF16, numel) + exp_avg (BF16, numel)
           + exp_avg_sq (BF16, numel)
    Muon:  grad (cfg.dtype, numel) + exp_avg (cfg.dtype, numel)
           — both buffers share storage dtype; no scale, no
           separate ``accum``.
- For ~624 M params: AdamW ≈ 6 × 624 M = 3.7 GB CPU RAM
  (grad BF16 + exp_avg BF16 + exp_avg_sq BF16, all 2 bytes/elt).
  Muon: 1.2 GB (bf16/fp16 × 2 buffers) / 2.4 GB (fp32 × 2).
  Quantized storage paths (``int8``, ``mxfp8``) used less RAM
  but were removed on 2026-07-12 (long-training quantization
  error accumulation); pre-removal code lives on the
  ``archive/int8-mxfp8-muon`` branch.

Why this layout is intentional (the "CPU grad accumulator is
unavoidable" note)
-----------------------------------------------------------
The per-step ``grad`` accumulator lives on CPU pinned memory,
not on the GPU. This is NOT a missed optimization — it is a
deliberate design choice driven by the requirement that grad
clip (per the ``max_grad_norm`` setting) must work reliably for
training stability. See :mod:`src.training.loop.grad_norm` for
the full rationale; the short version is: the L2 norm is
computed across all per-param ``grad`` tensors in one fused
C++ pass (``fused_l2_norm_sq_bf16``), TP-all-reduced as a
single FP64 scalar, and clipped in place via another fused
C++ pass (``fused_scale_many_bf16``). Hoisting the accumulator
back to the GPU just to clip would force N kernel launches
instead of one fused pass and add an extra H2D per param +
a second cross-op reduction — a clear loss on every metric
that matters (per-step latency, dispatch overhead, sync
stalls). The CPU accumulator is the minimum-cost substrate
that makes the fused-norm + TP-reduce + in-place scale
pipeline feasible at production scale.

Package layout
--------------
The package was split in July 2026 (no logic changes — class
boundaries are preserved):

  * :mod:`._state`  — :class:`_ParamState` dataclass,
    :class:`OptimizerState` :class:`typing.Protocol`, and the
    state-only helpers (:func:`_accumulator_target`,
    :func:`_scale_accum`).
  * :mod:`.adamw`   — :class:`CPUAdamW`.
  * :mod:`.muon`    — :class:`CPUMuon` and its full-precision
    bf16 / fp16 / fp32 storage-format helpers.
  * :mod:`.offload` — streaming / post-backward plumbing
    (post-accumulate-grad hooks, manual flush, accumulator
    reset, the module-level :data:`_pending_grads` /
    :data:`_manual_flush_param_ids` queues).
  * :mod:`.param_groups` — :func:`build_param_groups`.

Public API is preserved by re-exporting every name below; all
callers (the training loop, every optimizer test, the
diagnostics helpers, the VRAM profiling tools) continue to
``import src.training.param_offload`` unchanged.
"""
from __future__ import annotations

from ._state import (
    OptimizerState,
    _ParamState,
    _accumulator_target,
    _scale_accum,
)
from .adamw import CPUAdamW
from .muon import CPUMuon
from .offload import (
    _cpu_add_async_enabled,
    _cpu_add_worker_thread,
    _drain_cpu_add_queue,
    _enqueue_cpu_add,
    _manual_flush_param_ids,
    _make_offload_hook,
    _pending_grads,
    _start_cpu_add_worker,
    _stop_cpu_add_worker,
    accumulate_grads_to_cpu,
    flush_manual_flush_params,
    flush_pending_grads,
    register_grad_offload_hooks,
    zero_cpu_grad_accum,
)
from .param_groups import build_param_groups


__all__ = [
    # Optimizers
    "CPUAdamW",
    "CPUMuon",
    # State
    "OptimizerState",
    # Streaming / post-backward plumbing
    "accumulate_grads_to_cpu",
    "build_param_groups",
    "flush_manual_flush_params",
    "flush_pending_grads",
    "register_grad_offload_hooks",
    "zero_cpu_grad_accum",
    # Async CPU-add worker (Priority 2 overlap): started by the
    # training loop in _setup_worker, drained at end of step,
    # stopped in _teardown_worker. No-op in sync mode (tests).
    "_cpu_add_async_enabled",
    "_cpu_add_worker_thread",
    "_drain_cpu_add_queue",
    "_enqueue_cpu_add",
    "_start_cpu_add_worker",
    "_stop_cpu_add_worker",
    # Module-level queues (mutated by register_grad_offload_hooks
    # and flush_*_grads; exposed so tests can assert on them).
    "_manual_flush_param_ids",
    "_pending_grads",
    # Private (re-exported for tests + the optional state-only
    # helpers used by the training loop)
    "_ParamState",
    "_accumulator_target",
    "_make_offload_hook",
    "_scale_accum",
]
