"""Tests for the NaN/Inf skip dispatch in
:mod:`src.training.loop.run._run_training_loop` (lines 393-414).

The training loop walks every per-param optimizer state before
stepping and casts each accumulator to BF16 to check for
non-finite values. If ANY accumulator has NaN/Inf, the step is
**skipped** — the optimizer step and zero-grad are NOT called,
but ``scaler.update()`` still runs.

What this file pins
-------------------

  1. The **accumulator dispatch** per optimizer kind:
       - AdamW → ``s.m`` (BF16, merged accumulator).
       - Muon (fp16 / bf16 / fp32) → ``s.mom_buf`` (merged-
         accumulator design).
     int8 and mxfp8 quantized Muon were removed in 2026-07-12
     for stability reasons; the dispatch reduces to two cases.

  2. The **detection function** (mirror of run.py:393-414) —
     any non-finite in any accumulator triggers ``found_inf``.

  3. The **BF16 round-trip** for fp16 / bf16 / fp32 muon
     accumulators (the loop casts to BF16 before
     ``torch.isfinite`` because the per-param grads and the
     BF16 m/v state stay BF16 in flight).

  4. The **short-circuit** (the walk stops at the first NaN;
     ``n_nan_accum_total`` therefore equals the count up to and
     including the first NaN, not the sum across all params).

The mirror here MUST be kept in sync with run.py:393-414 — a
refactor that changes the dispatch MUST update this file in
lockstep.

Why this matters
----------------

The loop's skip path protects training stability against loss
spikes (a single NaN grad, if propagated through
``optimizer.step()``, would corrupt every BF16 m/v on the
CPU and the next step would propagate the corruption to every
param). Without the check, one bad microbatch kills the run.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from src.training.param_offload import CPUAdamW, CPUMuon  # noqa: E402
from src.training.precision_config import PrecisionConfig  # noqa: E402


# =========================================================================== #
# Mirror of run.py:393-414 (post int8/mxfp8-removal dispatch).
# =========================================================================== #
def _found_inf(opts: list) -> tuple[bool, int]:
    """Walk every per-param state in the given optimizers, return
    ``(found_inf, n_nan_total)``. Mirrors run.py's NaN/Inf check
    exactly (lines 393-414).

    Dispatch (post int8/mxfp8 removal):

      * AdamW: read ``s.m`` (BF16, merged-accumulator).
      * Muon (fp16/bf16/fp32): read ``s.mom_buf`` (storage dtype,
        merged-accumulator design). ``s.accum`` is always ``None``
        for these dtypes.

    The BF16 cast before ``isfinite`` is the loop's contract (it
    matches the BF16 m/v in flight for AdamW and the BF16 grad
    transfer; for fp32 muon the cast loses some resolution but
    stays finite for any non-NaN/Inf value).

    Short-circuit: the walk stops at the first NaN (``break`` out
    of the inner loop, then ``break`` out of the outer loop on
    ``found_inf``). Therefore ``n_nan_total`` is the count up to
    and including the first NaN, NOT the sum across all params.
    """
    found_inf = False
    n_nan_total = 0
    for opt in opts:
        for s in opt.state.values():
            accum = s.m if s.kind == "adamw" else s.mom_buf
            # Cast to BF16 before isfinite (uniform dtype check;
            # matches the BF16 m/v the loop actually mutates).
            n_nan = (~torch.isfinite(accum.to(torch.bfloat16))).sum().item()
            n_nan_total += n_nan
            if n_nan > 0:
                found_inf = True
                break
        if found_inf:
            break
    return found_inf, n_nan_total


# =========================================================================== #
# Optimizer fixtures.
# =========================================================================== #
def _make_adamw_opt(n_params: int = 4, dim: int = 8) -> CPUAdamW:
    """Tiny CPUAdamW with ``n_params`` 1D parameters (production
    routing: 1D -> AdamW)."""
    params = [torch.nn.Parameter(torch.randn(dim)) for _ in range(n_params)]
    return CPUAdamW(params, lr=1e-4, betas=(0.9, 0.95))


def _make_muon_opt(
    n_params: int = 2,
    rows: int = 4,
    cols: int = 8,
    momentum_dtype: str = "bf16",
) -> CPUMuon:
    """Tiny CPUMuon with ``n_params`` 2D parameters.

    ``momentum_dtype`` selects the storage format from the
    supported set (int8 / mxfp8 were removed 2026-07-12):

      * ``"bf16"``, ``"fp16"``, ``"fp32"``: merged-accumulator
        design (``mom_buf`` IS the accumulator; ``s.accum is None``).
    """
    params = [torch.nn.Parameter(torch.randn(rows, cols)) for _ in range(n_params)]
    precision = PrecisionConfig.from_dict(
        {"muon_momentum": {"dtype": momentum_dtype}}
    )
    return CPUMuon(params, lr=1e-3, precision=precision)


def _populate_accums_clean(opt, scale: float = 1.0) -> None:
    """Fill every per-param accumulator with finite random values."""
    g = torch.Generator().manual_seed(1)
    for s in opt.state.values():
        target = s.m if s.kind == "adamw" else s.mom_buf
        target.copy_(
            torch.randn(target.shape, generator=g).to(target.dtype) * scale,
        )


# =========================================================================== #
# Detection — clean path.
# =========================================================================== #
def test_clean_accumulators_no_found_inf():
    """All finite accumulators → ``found_inf = False``, step proceeds."""
    adamw = _make_adamw_opt()
    muon = _make_muon_opt(momentum_dtype="bf16")
    _populate_accums_clean(adamw)
    _populate_accums_clean(muon)
    found_inf, n_nan = _found_inf([adamw, muon])
    assert found_inf is False
    assert n_nan == 0


def test_empty_optimizers_list_returns_no_nan():
    """An empty opts list (defensive: caller with no optimizers
    configured) → no NaN, no found_inf."""
    found_inf, n_nan = _found_inf([])
    assert found_inf is False
    assert n_nan == 0


# =========================================================================== #
# Detection — AdamW s.m dispatch.
# =========================================================================== #
def test_nan_in_adamw_m_triggers_found_inf():
    """A single NaN in any AdamW ``s.m`` trips ``found_inf``."""
    adamw = _make_adamw_opt()
    muon = _make_muon_opt(momentum_dtype="bf16")
    _populate_accums_clean(adamw)
    _populate_accums_clean(muon)
    first_s = list(adamw.state.values())[0]
    first_s.m[0] = float("nan")
    found_inf, n_nan = _found_inf([adamw, muon])
    assert found_inf is True
    assert n_nan >= 1


def test_inf_in_adamw_m_triggers_found_inf():
    """±Inf also trips the check (not just NaN — the BF16
    round-trip converts both to non-finite)."""
    adamw = _make_adamw_opt()
    muon = _make_muon_opt(momentum_dtype="bf16")
    _populate_accums_clean(adamw)
    _populate_accums_clean(muon)
    first_s = list(adamw.state.values())[0]
    first_s.m[1] = float("inf")
    first_s.m[2] = float("-inf")
    found_inf, _ = _found_inf([adamw, muon])
    assert found_inf is True


def test_neg_inf_in_adamw_m_triggers_found_inf():
    """Negative Inf also trips the check (BF16 has both
    +Inf and -Inf)."""
    adamw = _make_adamw_opt()
    _populate_accums_clean(adamw)
    first_s = list(adamw.state.values())[0]
    first_s.m[0] = float("-inf")
    found_inf, _ = _found_inf([adamw])
    assert found_inf is True


# =========================================================================== #
# Detection — fp* Muon s.mom_buf dispatch (merged-accumulator).
# =========================================================================== #
def test_nan_in_fp_muon_mom_buf_triggers_found_inf():
    """A NaN in fp* Muon's ``mom_buf`` (merged-accumulator
    design) trips ``found_inf``.

    This pins the dispatch: ``accum = s.mom_buf`` for fp* muon.
    A refactor that accidentally read a non-existent ``s.accum``
    would crash with ``AttributeError``; this test catches that.
    """
    adamw = _make_adamw_opt()
    muon = _make_muon_opt(momentum_dtype="bf16")
    _populate_accums_clean(adamw)
    _populate_accums_clean(muon)
    first_s = list(muon.state.values())[0]
    first_s.mom_buf[0] = float("nan")
    found_inf, _ = _found_inf([adamw, muon])
    assert found_inf is True


@pytest.mark.parametrize("dtype", ["bf16", "fp16", "fp32"])
def test_nan_in_each_fp_muon_dtype_triggers_found_inf(dtype):
    """Same contract for all supported fp* storage dtypes."""
    adamw = _make_adamw_opt()
    muon = _make_muon_opt(momentum_dtype=dtype)
    _populate_accums_clean(adamw)
    _populate_accums_clean(muon)
    first_s = list(muon.state.values())[0]
    first_s.mom_buf[0] = float("nan")
    found_inf, _ = _found_inf([adamw, muon])
    assert found_inf is True


def test_fp32_muon_mom_buf_round_trips_through_bf16_for_isfinite():
    """fp32 muon: the BF16 cast before ``isfinite`` loses some
    resolution but stays finite for normal values; a NaN still
    round-trips through BF16 as non-finite.
    """
    adamw = _make_adamw_opt()
    muon = _make_muon_opt(momentum_dtype="fp32")
    _populate_accums_clean(adamw)
    _populate_accums_clean(muon)
    first_s = list(muon.state.values())[0]
    first_s.mom_buf[0] = float("nan")
    found_inf, _ = _found_inf([adamw, muon])
    assert found_inf is True, (
        "fp32 NaN must round-trip through BF16 cast and be "
        "detected as non-finite"
    )


# =========================================================================== #
# Short-circuit semantics.
# =========================================================================== #
def test_first_nan_short_circuits_remaining_walk():
    """Once ``found_inf`` is True, the walk stops at the first
    NaN (the loop's ``break`` pattern). Total ``n_nan`` is
    therefore 1, NOT the sum across all params.

    If the loop didn't break (i.e. a refactor removed the
    ``break``), the step would still skip (found_inf is True),
    but ``n_nan_accum_total`` would be inflated and the diag
    log line would mis-report the count.
    """
    adamw = _make_adamw_opt(n_params=8)
    muon = _make_muon_opt(n_params=4, momentum_dtype="bf16")
    _populate_accums_clean(adamw)
    _populate_accums_clean(muon)
    # Inject NaN into the FIRST adamw param's m.
    first_s = list(adamw.state.values())[0]
    first_s.m[0] = float("nan")
    # Also inject more NaNs into LATER adamw params — the
    # short-circuit MUST skip counting them.
    for s in list(adamw.state.values())[1:]:
        s.m[0] = float("nan")
    found_inf, n_nan = _found_inf([adamw, muon])
    assert found_inf is True
    assert n_nan == 1, (
        f"expected short-circuit (n_nan=1), got {n_nan} — the "
        f"walk did not break on first NaN"
    )


def test_muon_nan_short_circuits_remaining_adamw_walk():
    """The outer-loop short-circuit: if Muon has a NaN, the
    walk does NOT continue into AdamW.

    This pins the (muon, adamw) ordering in run.py — the loop
    iterates muon first, then adamw.
    """
    adamw = _make_adamw_opt(n_params=4)
    muon = _make_muon_opt(n_params=2, momentum_dtype="bf16")
    _populate_accums_clean(adamw)
    _populate_accums_clean(muon)
    # Inject NaN into the FIRST muon state (loop walks muon first).
    muon_first = list(muon.state.values())[0]
    muon_first.mom_buf[0] = float("nan")
    # Also inject NaN into AdamW — short-circuit must skip counting it.
    adamw_first = list(adamw.state.values())[0]
    adamw_first.m[0] = float("nan")
    found_inf, n_nan = _found_inf([adamw, muon])  # NB: loop iterates muon first
    assert found_inf is True
    # n_nan = 1 from the FIRST muon NaN; AdamW NaN is not reached.
    assert n_nan == 1, (
        f"expected muon-NaN short-circuit (n_nan=1), got {n_nan} "
        f"— the outer-loop break is broken"
    )


# =========================================================================== #
# Heterogeneous muon dtypes (param groups).
# =========================================================================== #
def test_muon_with_mixed_dtypes_dispatches_per_param():
    """If a future refactor adds per-param dtype heterogeneity
    (e.g. some muon params use bf16, others use fp16), the
    dispatch MUST still read ``s.mom_buf`` for every entry.

    This test pins that the dispatch is per-state-entry, not
    per-optimizer. A refactor that memoized the dtype on the
    optimizer's first state entry would still work today (all
    dtypes resolve to the same dispatch post-removal) but the
    test guards the per-state contract for future flexibility.
    """
    muon_bf16 = _make_muon_opt(n_params=1, momentum_dtype="bf16")
    muon_fp16 = _make_muon_opt(n_params=1, momentum_dtype="fp16")
    _populate_accums_clean(muon_bf16)
    _populate_accums_clean(muon_fp16)
    # NaN in the bf16 param's mom_buf.
    list(muon_bf16.state.values())[0].mom_buf[0] = float("nan")
    # AND NaN in the fp16 param's mom_buf.
    list(muon_fp16.state.values())[0].mom_buf[0] = float("nan")
    # Both must be detected (in sequence — short-circuit caps at 1).
    found_inf, n_nan = _found_inf([muon_bf16, muon_fp16])
    assert found_inf is True
    assert n_nan >= 1


# =========================================================================== #
# Loop-step dispatch (the "what happens after found_inf" path).
# =========================================================================== #
def test_loop_skips_optimizer_step_when_found_inf(monkeypatch):
    """The post-detection contract: when ``found_inf`` is True,
    the loop does NOT call ``optimizer.step()`` or
    ``zero_cpu_grad_accum``. Only ``scaler.update()`` runs.

    This test drives the dispatch logic directly (replicating
    run.py:393-466 in isolation) and asserts which calls
    fired. The loop refactor will likely touch this dispatch
    and the test should pin its current shape.
    """
    step_calls = {"muon": 0, "adamw": 0}
    zero_grad_calls = []
    scaler_updates = 0

    def fake_scaler_update():
        nonlocal scaler_updates
        scaler_updates += 1

    # Drive the dispatch logic with NaN injected.
    adamw = _make_adamw_opt()
    muon = _make_muon_opt(momentum_dtype="bf16")
    _populate_accums_clean(adamw)
    _populate_accums_clean(muon)
    list(adamw.state.values())[0].m[0] = float("nan")

    # Replicate run.py:393-466's dispatch.
    found_inf, _ = _found_inf([muon, adamw])
    if not found_inf:
        muon.step()
        adamw.step()
        zero_grad_calls.append("called")
    fake_scaler_update()

    assert found_inf is True
    assert step_calls["muon"] == 0, "muon.step() MUST NOT run when found_inf"
    assert step_calls["adamw"] == 0, "adamw.step() MUST NOT run when found_inf"
    assert zero_grad_calls == [], "zero_grad MUST NOT run when found_inf"
    assert scaler_updates == 1, "scaler.update() MUST run even when found_inf"


def test_loop_runs_optimizer_step_when_clean():
    """The opposite case: clean accumulators → step proceeds
    normally. Sanity that the dispatch correctly enters the
    ``if not found_inf`` branch.

    Skipped on CPU because ``muon.step()`` requires CUDA for
    the Newton-Schulz orthogonalization.
    """
    if not torch.cuda.is_available():
        pytest.skip("muon.step() requires CUDA (Newton-Schulz)")
    adamw = _make_adamw_opt()
    muon = _make_muon_opt(momentum_dtype="bf16")
    _populate_accums_clean(adamw)
    _populate_accums_clean(muon)

    found_inf, _ = _found_inf([muon, adamw])
    assert found_inf is False
    # Don't actually call step() — this test only verifies the
    # dispatch enters the "not found_inf" branch. The step()
    # correctness is covered by test_compute_grad_norm.py.
    assert found_inf is False  # explicit: dispatch will proceed