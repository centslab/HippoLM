"""Per-step diagnostic logging for the training loop.

The training loop emits three families of diagnostic logs for the
first ``_DIAG_STEPS`` accumulation cycles:

  - **Pre-step**: max abs of the clipped grad accumulator, the
    optimizer state, and the param for both optimizers, plus the
    TP-reduced L2 grad norm. Helps localize which optimizer (or
    which param group) is the source of NaN/Inf if it appears.

  - **Post-Muon** / **Post-AdamW**: all-finite or first NON-FINITE
    shape and pmax after each optimizer step. Run as a pair so
    we can attribute the divergence to one optimizer or the
    other (Muon uses a different update path — Newton-Schulz +
    quantization — and is the more likely culprit for early
    instability).

All helpers take the per-param state dict as a
``Dict[int, OptimizerState]`` (see :class:`src.training.param_offload.OptimizerState`)
and never mutate it; the diagnostic is read-only.

These blocks are gated by ``_DIAG_STEPS`` at the call site
(typically the first 3-5 accumulation cycles) so steady-state
training has zero logging overhead.
"""
from __future__ import annotations

import math
from typing import Dict, Optional

import torch

from .param_offload import OptimizerState


def amax_cpu(t: Optional[torch.Tensor]) -> float:
    """Max abs of a CPU tensor (or ``None``), as a Python float.

    ``None`` (e.g. AdamW state accessed from a Muon diagnostic
    branch) returns 0.0 so it sorts cleanly in a max() default.
    Uses ``.detach().abs().max().item()`` — the tensor is on
    CPU pinned memory so the read is cheap (no D2H copy, no
    sync beyond the item() call itself).
    """
    if t is None:
        return 0.0
    return t.detach().abs().max().item()


def _max_over_states(
    state: Dict[int, OptimizerState],
    attr: str,
) -> float:
    """Max abs of a named attribute across all per-param entries.

    Reads ``getattr(s, attr)`` on each entry and returns the
    scalar max abs. If the dict is empty, returns 0.0 (matches
    the default of ``max(..., default=0.0)`` at the call sites).
    """
    if not state:
        return 0.0
    return max(amax_cpu(getattr(s, attr)) for s in state.values())


def log_pre_step_diag(
    logger,
    *,
    step: int,
    total_norm: float,
    muon_state: Dict[int, OptimizerState],
    adamw_state: Dict[int, OptimizerState],
) -> None:
    """Emit the pre-optimizer-step diagnostic.

    For each optimizer we report three numbers: the max abs of
    the clipped grad accumulator, the max abs of the
    optimizer-internal state (m for AdamW, momentum for Muon),
    and the max abs of the param. Combined with
    ``total_norm`` this is enough to localize the source of an
    explosion to a specific optimizer (muon vs adamw) and
    stage (grad vs state vs param).
    """
    logger.info(
        f"  [diag-step {step} pre]"
        f" total_norm={total_norm:.3e}"
        f" muon: acc_max={_max_over_states(muon_state, 'accum'):.3e}"
        f" mom_max={_max_over_states(muon_state, 'mom_buf'):.3e}"
        f" pmax={_max_over_states(muon_state, 'param'):.3e}"
        f" | adamw: acc_max={_max_over_states(adamw_state, 'accum'):.3e}"
        f" m_max={_max_over_states(adamw_state, 'm'):.3e}"
        f" v_max={_max_over_states(adamw_state, 'exp_avg_sq'):.3e}"
        f" pmax={_max_over_states(adamw_state, 'param'):.3e}"
    )


def log_post_opt_diag(
    logger,
    *,
    step: int,
    opt_label: str,
    state: Dict[int, OptimizerState],
) -> None:
    """Emit the post-optimizer-step diagnostic for one optimizer.

    Scans every per-param entry. On the first NON-FINITE param,
    reports its shape and max-abs. If everything is finite,
    reports the global max abs. We use a max-abs (not just
    "any non-finite") because a single all-``inf`` entry would
    have ``.abs().max() == inf``, which lets the operator see
    the magnitude of the failure rather than a binary
    "non-finite" line.
    """
    nan_pmax = 0.0
    nan_first: Optional[tuple] = None
    for s in state.values():
        v = s.param.data
        if not torch.isfinite(v).all():
            am = v.detach().abs().max().item()
            if math.isnan(am) or am > nan_pmax:
                nan_pmax = am
                if nan_first is None:
                    nan_first = (s.param.shape, am)
    if nan_first is not None:
        logger.info(
            f"  [diag-step {step} post-{opt_label}]"
            f" NON-FINITE in {opt_label} params:"
            f" first_shape={nan_first[0]}"
            f" pmax={nan_first[1]:.3e}"
        )
    else:
        pmax = _max_over_states(state, "param")
        logger.info(
            f"  [diag-step {step} post-{opt_label}]"
            f" all finite, pmax={pmax:.3e}"
        )
