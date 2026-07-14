"""Per-step gradient-norm computation + clip.

Single public function :func:`_compute_and_clip_grad_norm`:

  1. Walks every per-param ``_ParamState`` in the optimizer
     pair, accumulating ``||accum||²`` into a local scalar.
  2. All-reduces the scalar across the TP world (NCCL or
     gloo).
  3. When the global ``total_norm`` exceeds ``max_norm``,
     scales each accumulator by ``max_norm / (total_norm + eps)``
     in place.

The accumulator dispatch is:

  - AdamW: ``s.m`` (BF16, merged first moment + grad buffer).
  - Muon (merged-accumulator design): ``s.mom_buf`` (the
    configured full-precision storage dtype: bf16 / fp16 /
    fp32 — quantized storage removed 2026-07-12).

The clip is always a plain ``.mul_(coef)`` because both
accumulators are floating-point (BF16 / FP16 / FP32).

Distributed (``dist``) is imported lazily to keep this module
importable from non-distributed unit tests (the all-reduce
no-ops when the process group is not initialized).
"""
from __future__ import annotations

import torch


def _compute_and_clip_grad_norm(opts, max_norm: float) -> float:
    """Compute the L2 norm across all per-param accumulators in
    the given optimizers, all-reduce across the TP world, and
    clip in place.

    For each param the accumulator is whichever tensor holds
    the sum-of-microbatch-grads for that param at this point:
    ``s.m`` (AdamW) or ``s.mom_buf`` (Muon, merged design).
    Scales the accumulator by ``clip_coef = max_norm /
    (total_norm + 1e-6)`` only when the norm exceeds the cap,
    so well-behaved steps are no-ops.

    The clip is always a plain ``.mul_(coef)`` because both
    accumulators are floating-point (BF16 / FP16 / FP32).

    Implementation: walks all per-param accumulators once via the
    fused C++ kernel :func:`fused_l2_norm_sq_bf16`
    (``src/training/param_offload/cpu_fused.py``). The kernel
    promotes BF16 → FP32 via shift-left-16, squares in FP32,
    and accumulates across all params in FP64 in one OMP
    ``parallel for`` pass. Replaces the legacy Python loop
    ``accum.detach().float().pow(2).sum()`` which paid full
    cross-op dispatch overhead and allocated a fresh FP32 copy
    of each accumulator (1.5 GiB for the tied embed alone) on
    every iteration. ~7.7 s/step saved at base.yml shape.

    Falls back to the per-tensor Python loop if the JIT
    extension failed to compile (no C++ toolchain at runtime);
    correctness is preserved either way, only the speedup is
    lost.
    """
    import torch.distributed as dist

    from src.training.param_offload.cpu_fused import (
        fused_l2_norm_sq_bf16,
    )

    accumulators: list = []
    for opt in opts:
        for s in opt.state.values():
            accum = (
                s.m if s.kind == "adamw"
                else s.mom_buf
            )
            accumulators.append(accum)
    _used_fused, local_sq = fused_l2_norm_sq_bf16(accumulators)
    if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        # Wrap the FP64 scalar in a 1-element tensor so we can use
        # the all-reduce collective (PyTorch doesn't expose a
        # Python-side all-reduce for raw floats).
        local_sq_t = torch.tensor([local_sq], dtype=torch.float64)
        dist.all_reduce(local_sq_t, op=dist.ReduceOp.SUM)
        local_sq = local_sq_t.item()
    total_norm = local_sq ** 0.5
    # Clip only when the norm exceeds the cap.  An earlier revision
    # applied a second unconditional ``s.accum.mul_(max_norm / (total_norm
    # + eps))`` below this guard, which had two bugs:
    #   (a) when total_norm < max_norm it AMPLIFIED small grads
    #       (``max_norm / total_norm > 1``), wrecking training stability;
    #   (b) when total_norm > max_norm the clip was applied twice
    #       (quadratic clip instead of linear).
    if max_norm > 0.0 and total_norm > max_norm:
        clip_coef = max_norm / (total_norm + 1e-6)
        from src.training.param_offload.cpu_fused import (
            fused_scale_many_bf16,
        )
        accumulators = []
        for opt in opts:
            for s in opt.state.values():
                accum = (
                    s.m if s.kind == "adamw"
                    else s.mom_buf
                )
                accumulators.append(accum)
        fused_scale_many_bf16(accumulators, clip_coef)
    return total_norm