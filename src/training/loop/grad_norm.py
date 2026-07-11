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
  - Quantized Muon (int8, mxfp8): ``s.accum`` (BF16, the
    per-mb grad sum; the ``mom_buf`` storage is untouched at
    clip time and requantized from ``accum`` at ``step()``).
  - fp* Muon (merged-accumulator design): ``s.mom_buf``.

The clip is always a plain ``.mul_(coef)`` because all three
accumulators are floating-point (BF16 / FP16 / FP32); the
previous FP8 ``mom_buf`` round-trip via
``_scale_mxfp8_mom_buf`` is gone — the separate-accumulator
redesign moved the FP8 storage away from the clip path.

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
    ``s.m`` (AdamW), ``s.accum`` (quantized muon: int8 /
    mxfp8), or ``s.mom_buf`` (fp* muon, merged design).
    Scales the accumulator by ``clip_coef = max_norm /
    (total_norm + 1e-6)`` only when the norm exceeds the cap,
    so well-behaved steps are no-ops.

    The clip is always a plain ``.mul_(coef)`` because all
    three accumulators are floating-point (BF16 / FP16 / FP32);
    the previous FP8 ``mom_buf`` round-trip via
    ``_scale_mxfp8_mom_buf`` is gone — the separate-accumulator
    redesign moved the FP8 storage away from the clip path.
    """
    import torch.distributed as dist

    local_sq = torch.zeros(1)
    for opt in opts:
        for s in opt.state.values():
            accum = (
                s.m if s.kind == "adamw"
                else (s.accum if s.accum is not None else s.mom_buf)
            )
            local_sq += accum.detach().float().pow(2).sum()
    if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        dist.all_reduce(local_sq, op=dist.ReduceOp.SUM)
    total_norm = local_sq.sqrt().item()
    # Clip only when the norm exceeds the cap.  An earlier revision
    # applied a second unconditional ``s.accum.mul_(max_norm / (total_norm
    # + eps))`` below this guard, which had two bugs:
    #   (a) when total_norm < max_norm it AMPLIFIED small grads
    #       (``max_norm / total_norm > 1``), wrecking training stability;
    #   (b) when total_norm > max_norm the clip was applied twice
    #       (quadratic clip instead of linear).
    if max_norm > 0.0 and total_norm > max_norm:
        clip_coef = max_norm / (total_norm + 1e-6)
        from src.training.param_offload import _scale_accum
        for opt in opts:
            for s in opt.state.values():
                _scale_accum(s, clip_coef)
    return total_norm