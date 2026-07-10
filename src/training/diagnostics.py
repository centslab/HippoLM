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
    # FP8 dtypes (mxfp8 Muon: E4M3 ``mom_buf`` / E8M0
    # ``mom_scale``) have no CPU reduction kernels in PyTorch
    # 2.9.1 (``NotImplementedError: max_all not implemented for
    # 'Float8_e4m3fn'``). Cast to BF16 first — the round-trip is
    # free on CPU and the abs/max reduction is the same cost as
    # in the original dtype.
    if t.dtype in (torch.float8_e4m3fn, torch.float8_e5m2,
                    torch.float8_e8m0fnu):
        t = t.to(torch.bfloat16)
    return t.detach().abs().max().item()


def _max_over_states(
    state: Dict[int, OptimizerState],
    attr: str,
) -> float:
    """Max abs of a named attribute across all per-param entries.

    Reads ``getattr(s, attr)`` on each entry and returns the
    scalar max abs. If the dict is empty, returns 0.0 (matches
    the default of ``max(..., default=0.0)`` at the call sites).

    NVFP4 mode-3 entries (``s.nvfp4_module is not None``) have
    ``s.param is None`` because the weight lives on the module
    as FP4 packed buffers. Reading ``param`` on those would
    crash; instead we proxy through ``packed_weight`` (the FP4
    byte storage) so the diagnostic still emits a meaningful
    magnitude. The user-visible difference is just the dtype
    of the value (uint8 for FP4 bytes vs whatever the BF16
    weight would be) — the abs-max utility handles both.
    """
    if not state:
        return 0.0
    vals = []
    for s in state.values():
        if s.nvfp4_module is not None:
            vals.append(amax_cpu(s.nvfp4_module.packed_weight))
        else:
            vals.append(amax_cpu(getattr(s, attr)))
    return max(vals)


def log_pre_step_diag(
    logger,
    *,
    step: int,
    total_norm: float,
    muon_state: Dict[int, OptimizerState],
    adamw_state: Dict[int, OptimizerState],
) -> None:
    """Emit the pre-optimizer-step diagnostic.

    For each optimizer we report the max abs of the cycle's
    grad accumulator (``m`` for AdamW; ``accum`` when set for
    quantized muon — int8/mxfp8 — and ``mom_buf`` for fp*
    muon in the merged-accumulator design), the max abs of the
    remaining optimizer-internal state (only ``exp_avg_sq`` for
    AdamW; Muon has no separate internal state at this point
    — ``mom_buf`` / ``mom_scale`` are populated at step end,
    not read here), and the max abs of the param. Combined
    with ``total_norm`` this is enough to localize the source
    of an explosion to a specific optimizer (muon vs adamw)
    and stage (grad vs state vs param).

    For quantized muon the diagnostic reads ``s.accum`` (the
    bf16 cycle sum) rather than ``s.mom_buf`` (which is the
    int8/mxfp8 representation that hasn't been populated yet
    for this cycle — it was reset to zero at the end of the
    previous cycle's step). Reading ``mom_buf`` here would
    report the stale prior cycle's quantized momentum and
    miss any current-cycle divergence.
    """
    # Resolve which tensor to read for each muon param: the
    # separate ``accum`` (quantized) when set, else ``mom_buf``
    # (fp* muon merged design).
    def muon_accum_max(state: Dict[int, OptimizerState]) -> float:
        if not state:
            return 0.0
        vals = []
        for s in state.values():
            t = s.accum if s.accum is not None else s.mom_buf
            vals.append(amax_cpu(t))
        return max(vals)

    logger.info(
        f"  [diag-step {step} pre]"
        f" total_norm={total_norm:.3e}"
        f" muon: mom_max={muon_accum_max(muon_state):.3e}"
        f" pmax={_max_over_states(muon_state, 'param'):.3e}"
        f" | adamw: m_max={_max_over_states(adamw_state, 'm'):.3e}"
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
        # NVFP4 mode-3 entries have no ``s.param``; substitute
        # the FP4 packed buffer so the diagnostic still scans
        # something meaningful (the FP4 byte storage — uint8
        # always finite, so it never triggers the non-finite
        # branch, but the all-finite branch will report its
        # max abs).
        if s.nvfp4_module is not None:
            v = s.nvfp4_module.packed_weight
            if not torch.isfinite(v.float()).all():
                am = v.detach().float().abs().max().item()
                if math.isnan(am) or am > nan_pmax:
                    nan_pmax = am
                    if nan_first is None:
                        nan_first = (s.nvfp4_module.packed_weight.shape, am)
            continue
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
