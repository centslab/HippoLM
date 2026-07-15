"""Per-step gradient-norm computation + clip.

Single public function :func:`_compute_and_clip_grad_norm`:

  1. Walks every per-param ``_ParamState`` in the optimizer
     pair, accumulating ``||grad||²`` into a local scalar.
  2. All-reduces the scalar across the TP world (NCCL or
     gloo).
  3. When the global ``total_norm`` exceeds ``max_norm`,
     scales each grad accumulator by ``max_norm / (total_norm
     + eps)`` in place.

The accumulator dispatch (post-2026-07-15; was ``s.m`` /
``s.mom_buf`` in the deleted merged-accumulator layout):

  - AdamW: ``s.grad`` (BF16, per-step accumulator — the
    same tensor the per-mb streaming hook folds grads into).
  - Muon:  ``s.grad`` (the configured full-precision storage
    dtype: bf16 / fp16 / fp32).

The clip is always a plain ``.mul_(coef)`` because both
accumulators are floating-point (BF16 / FP16 / FP32).

Distributed (``dist``) is imported lazily to keep this module
importable from non-distributed unit tests (the all-reduce
no-ops when the process group is not initialized).

Why the grad accumulator MUST live on CPU (the "why this is
unavoidable" note)
-------------------------------------------------------
The user's design constraint (recorded here so future maintainers
don't try to "fix" it) is that the per-step grad accumulator is
a CPU pinned buffer — it is NOT a GPU tensor that gets summed +
clipped + summed there. Three reasons this is intentional and
not a missed optimization:

1. **Single fused pass for the L2 norm.** ``s.grad`` is the
   per-param accumulator; the function above walks *all* of them
   in one OMP ``parallel for`` via
   :func:`fused_l2_norm_sq_bf16` and accumulates ``Σ ||g||²``
   into a single FP64 scalar. That scalar is what gets
   all-reduced across the TP world. The fused pass promotes
   BF16 → FP32 via shift-left-16, squares in FP32, and reduces
   in FP64 — all in one kernel. If the accumulator lived on
   the GPU, the fused kernel wouldn't apply (it's a CPU kernel)
   and we'd need a per-param ``.pow(2).sum().item()`` loop that
   ``.item()``-syncs the GPU once per param — at 624 M params
   that would be ~5 s/step of pure sync stalls.

2. **TP all-reduce on a scalar, not per-tensor.** The clip
   threshold ``max_norm`` is a property of the global gradient
   (sum over every param), not per-param. Doing the reduction
   once on a 1-element FP64 tensor is one NCCL/gloo call,
   which is the cheapest collective. Doing it per-param would
   be ``n_params`` collectives and the TP topology doesn't
   benefit from it (the grads are already partitioned by TP
   rank by the optimizer construction).

3. **In-place clip after the reduction.** Once we have the
   global L2 norm, the clip coefficient is
   ``max_norm / (total_norm + eps)`` and we apply it to every
   per-param grad in place via :func:`fused_scale_many_bf16`
   (one OMP pass across all ``s.grad`` tensors). This is the
   same fused-bandwidth trick that made the norm compute
   itself feasible. Doing this on the GPU would require
   ``n_params`` separate kernel launches.

For training stability, gradient clipping is non-negotiable —
it's the single most reliable guard against loss spikes from a
noisy outlier batch, and removing it would require replacing it
with something equally robust. The CPU accumulator is the
minimum-cost substrate that makes all three of the above
feasible at production scale (624 M params, ~2.5 GB of CPU
grad state, 1 TP all-reduce per step).

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
from __future__ import annotations

import torch


def _compute_and_clip_grad_norm(opts, max_norm: float) -> float:
    """Compute the L2 norm across all per-param accumulators in
    the given optimizers, all-reduce across the TP world, and
    clip in place.

    For each param the accumulator is ``s.grad`` (both AdamW
    and Muon — the post-2026-07-15 explicit-accumulator
    layout; the old merged design where ``s.m`` / ``s.mom_buf``
    doubled as the accumulator was deleted). Scales the
    accumulator by ``clip_coef = max_norm / (total_norm + 1e-6)``
    only when the norm exceeds the cap, so well-behaved steps
    are no-ops.

    The clip is always a plain ``.mul_(coef)`` because both
    accumulators are floating-point (BF16 / FP16 / FP32).

    See the module docstring for the rationale on why this
    MUST live on CPU (training-stability requirement: grad
    clip is non-negotiable; the CPU accumulator is the
    minimum-cost substrate for the fused L2-norm + TP
    all-reduce + in-place scale pipeline).
    """
    import torch.distributed as dist

    from src.training.param_offload.cpu_fused import (
        fused_l2_norm_sq_bf16,
    )

    accumulators: list = []
    for opt in opts:
        for s in opt.state.values():
            # Post-2026-07-15: every per-param entry has ``s.grad``
            # (the explicit per-step accumulator). The old dispatch
            # ``s.m if s.kind == "adamw" else s.mom_buf`` was
            # deleted along with the merged-accumulator layout.
            accumulators.append(s.grad)
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
        # Rescan is intentional: the fused_scale_many_bf16 kernel
        # takes a flat list of tensors; rebuilding the list from
        # ``s.grad`` for every optimizer keeps the dispatch
        # symmetric with the norm compute above and avoids
        # carrying the list across the if-branch (which would
        # require an empty-list init + a populate, more code).
        accumulators = []
        for opt in opts:
            for s in opt.state.values():
                accumulators.append(s.grad)
        fused_scale_many_bf16(accumulators, clip_coef)
    return total_norm