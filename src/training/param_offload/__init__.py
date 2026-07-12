"""Per-parameter optimizers with CPU offload for HippoLM.

Implements two optimizer variants used together via param groups:

  - :class:`CPUAdamW`     — AdamW with BF16 m (which doubles as the
    grad accumulator) and BF16 v state on CPU pinned memory. The
    forward / backward happen on the GPU in the param's training
    dtype (FP16 in this project). After each micro-batch's
    backward, the per-param post-accumulate-grad hook DMA's the
    GPU ``.grad`` (cast to ``m.dtype``) and folds it into ``m``
    in place on the CPU (BF16, pinned). On ``step()`` ``m``
    holds the sum of microbatch grads (``mu=1`` accumulation —
    no cross-step β1 EMA), ``v`` is updated in place (BF16, with
    β2 EMA), and the per-element update factor ``m / (sqrt(v) + eps)``
    is computed in FP32 (to recover precision after the BF16 v
    sqrt), then cast to FP16 and streamed chunk-wise to the GPU
    param and applied in place. ``m`` is then reset to zero for
    the next accumulation cycle.
    Grad dtype is BF16 throughout (matches the new 5060Ti
    hardware; BF16 has FP32-like dynamic range, so no overflow
    on the grad transfer or the m update, and v can stay BF16
    because BF16's 8-bit exponent is wide enough to keep
    ``v = g²`` from underflowing at typical grad magnitudes
    1e-4 to 1e-3 — 1e-8 is well above BF16's smallest normal
    of ~1.18e-38).

  - :class:`CPUMuon`      — Muon (Newton-Schulz orthogonalization
    of the momentum matrix). ``mom_buf`` doubles as the grad
    accumulator and the momentum feed for NS — ``mu=1``
    accumulation (just ``mom_buf += g`` per microbatch). On
    ``step()`` ``mom_buf`` is consumed (orthogonalized + applied
    + reset to zero). Momentum storage is configurable via
    ``precision.muon_momentum``:

      * ``bf16`` / ``fp16`` / ``fp32``: full-precision
        momentum, no quantization. ``mom_buf`` is the
        accumulator directly; the add is just a tensor add
        in the storage dtype (no requant). The NS iteration
        is unaffected (it orthogonalizes the raw momentum in
        FP32 on the GPU regardless).

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
optimizer state's accumulator; only after ``gradient_accumulation_steps``
micro-batches does ``step()`` actually run.

Conventions
-----------
- Param ``.data`` lives on the GPU (FP16).
- ``.grad`` is materialised on the GPU only during ``backward()``;
  immediately after, the training loop calls the per-param
  post-accumulate-grad hook (registered via
  :func:`register_grad_offload_hooks`) which copies ``.grad``
  into the per-param CPU accumulator and frees the GPU copy.
  So between micro-batches the GPU holds zero gradient memory.
- Optimizer state on CPU pinned memory:
    AdamW: m (BF16, numel) + v (BF16, numel)
    Muon:  mom_buf (cfg.dtype, numel) — no scale, no
           separate ``accum`` (merged-accumulator design).
- For ~624 M params: AdamW ≈ 4 × 624 M = 2.5 GB CPU RAM
  (m BF16 + v BF16, both 2 bytes/elt).
  Muon: 0.6 GB (fp16/bf16) / 1.2 GB (fp32). The previous
  ``int8``-with-per-row-BF16-scale path used 0.3 GB but
  was removed on 2026-07-12 (long-training quantization
  error accumulation); pre-removal code lives on the
  ``archive/int8-mxfp8-muon`` branch. The merged-accumulator
  design saves 2 bytes/elt on every param vs the legacy
  separate-accumulator design.

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
  * :mod:`.per_layer_gpu_accum` — OPT-IN alternative to the
    streaming path. Per-layer GPU accumulator + worker-thread
    CPU ``add_`` pipeline. Wins 33-44% step time vs the
    default CPU-add path at every MBS measured; see the
    module docstring for the A/B numbers and the v4 design.

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
    _manual_flush_param_ids,
    _make_offload_hook,
    _pending_grads,
    accumulate_grads_to_cpu,
    flush_manual_flush_params,
    flush_pending_grads,
    register_grad_offload_hooks,
    zero_cpu_grad_accum,
)
from .param_groups import build_param_groups
from .per_layer_gpu_accum import (
    flush_per_layer_gpu_accum,
    setup_per_layer_gpu_accum,
    shutdown_per_layer_gpu_accum,
    transfer_per_layer_gpu_accum_join,
)


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
    # Per-layer GPU accumulator (opt-in v4 path; see
    # per_layer_gpu_accum.py for the A/B numbers).
    "setup_per_layer_gpu_accum",
    "flush_per_layer_gpu_accum",
    "transfer_per_layer_gpu_accum_join",
    "shutdown_per_layer_gpu_accum",
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
