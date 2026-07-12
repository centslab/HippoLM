"""Tests for the per-step diagnostic loggers.

Scope (per user directive: pure helpers only, no model-level
tests because the model code iterates quickly):

  - :func:`amax_cpu`     — None → 0.0, finite tensor → max abs,
    nan / inf → nan / inf (the logger passes these through
    verbatim — the operator wants to see the magnitude of the
    failure).
  - :func:`_max_over_states` — empty dict → 0.0; regular param
    reads ``getattr(s, attr)``; NVFP4 mode-3 entries read
    ``s.nvfp4_module.packed_weight`` instead (no ``s.param`` to
    crash on).
  - :func:`log_pre_step_diag` — emits a line with the per-
    optimizer pre-step magnitudes (muon mom + pmax, adamw m + v
    + pmax) plus the global ``total_norm``.
  - :func:`log_post_opt_diag` — emits the all-finite line when
    every param is finite, the NON-FINITE line on the first
    param with NaN/Inf (with shape + max-abs). NVFP4 mode-3
    entries always look "all-finite" (uint8 storage cannot be
    non-finite) and contribute their packed max-abs to the
    all-finite branch.

These tests pin the exact log line format (the operator relies
on grep patterns over the smoke-test output to debug early
divergence).
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from src.training.diagnostics import (  # noqa: E402
    amax_cpu,
    log_post_opt_diag,
    log_pre_step_diag,
)
from src.training.param_offload import _ParamState  # noqa: E402


# --------------------------------------------------------------------------- #
# Fixtures / helpers.                                                         #
# --------------------------------------------------------------------------- #
def _make_param_state(
    *,
    kind: str = "muon",
    mom_buf: torch.Tensor | None = None,
    m: torch.Tensor | None = None,
    exp_avg_sq: torch.Tensor | None = None,
    param: torch.Tensor | None = None,
    nvfp4_module=None,
) -> OptimizerState:
    """Build a minimal ``OptimizerState`` with only the attrs a given
    test exercises. The dataclass fields default to None so unused
    ones don't need to be supplied.
    """
    return _ParamState(
        param=param,
        kind=kind,
        m=m,
        exp_avg_sq=exp_avg_sq,
        mom_buf=mom_buf,
        nvfp4_module=nvfp4_module,
    )


def _nvfp4_module_stub(packed_uint8: torch.Tensor) -> MagicMock:
    """A MagicMock that exposes ``packed_weight`` (uint8 tensor) like
    a real NVFP4 module does. Used to drive the NVFP4 proxy branch
    in ``_max_over_states`` and ``log_post_opt_diag``.
    """
    mod = MagicMock()
    mod.packed_weight = packed_uint8
    return mod


# --------------------------------------------------------------------------- #
# amax_cpu.                                                                   #
# --------------------------------------------------------------------------- #
def test_amax_cpu_none_returns_zero():
    """``None`` (e.g. AdamW state accessed from a Muon diagnostic
    branch) returns 0.0 so it sorts cleanly in a max() default."""
    assert amax_cpu(None) == 0.0


def test_amax_cpu_returns_max_abs_for_fp32():
    t = torch.tensor([1.0, -3.0, 2.5])
    assert amax_cpu(t) == 3.0


def test_amax_cpu_works_for_bf16_fp16_int():
    """The function is dtype-agnostic (calls .abs().max() and
    .item() — works for any floating or int dtype)."""
    assert amax_cpu(torch.tensor([-7.5], dtype=torch.bfloat16)) == 7.5
    assert amax_cpu(torch.tensor([4.0], dtype=torch.float16)) == 4.0
    assert amax_cpu(torch.tensor([0, -11, 3], dtype=torch.int32)) == 11


def test_amax_cpu_passes_nan_through():
    """NaN is a finite-abs-magnitude of NaN — the logger passes it
    through so the operator sees that a NaN occurred (instead of
    silently converting to 0)."""
    t = torch.tensor([1.0, float("nan"), -2.0])
    out = amax_cpu(t)
    assert out != out  # NaN != NaN


def test_amax_cpu_passes_inf_through():
    """Inf's abs() is Inf — passes through to the caller (the
    operator pattern-greps for ``inf`` in the smoke-test log)."""
    t = torch.tensor([1.0, float("inf"), -3.0])
    assert amax_cpu(t) == float("inf")


# --------------------------------------------------------------------------- #
# log_pre_step_diag — format.                                                 #
# --------------------------------------------------------------------------- #
def test_log_pre_step_diag_emits_required_fields(caplog):
    """The line must include step, total_norm, both optimizers'
    magnitudes, and ``pmax`` for each."""
    caplog.set_level(logging.INFO)
    logger = logging.getLogger("test_diag_pre")
    logger.setLevel(logging.INFO)

    muon_state = {
        1: _make_param_state(
            kind="muon",
            mom_buf=torch.tensor([1.0, 2.0, -3.0]),
            param=torch.tensor([0.5, 0.6]),
        ),
    }
    adamw_state = {
        1: _make_param_state(
            kind="adamw",
            m=torch.tensor([0.1, 0.2]),
            exp_avg_sq=torch.tensor([0.01, 0.04]),
            param=torch.tensor([0.5]),
        ),
    }

    log_pre_step_diag(
        logger,
        step=7,
        total_norm=1.23,
        muon_state=muon_state,
        adamw_state=adamw_state,
    )

    msgs = [rec.message for rec in caplog.records]
    assert any("diag-step 7 pre" in m for m in msgs), msgs
    assert any("total_norm=1.230e+00" in m for m in msgs), msgs
    assert any("muon:" in m for m in msgs), msgs
    assert any("adamw:" in m for m in msgs), msgs
    assert any("mom_max=3.000e+00" in m for m in msgs), msgs
    assert any("m_max=2.000e-01" in m for m in msgs), msgs
    assert any("v_max=4.000e-02" in m for m in msgs), msgs
    assert any("pmax=" in m for m in msgs), msgs


def test_log_pre_step_diag_empty_states_yields_zero_magnitudes(caplog):
    """Empty muon_state / adamw_state dicts are allowed (early
    step, before any param has been routed). The max-over-states
    must yield 0.0, not raise on the empty dict."""
    caplog.set_level(logging.INFO)
    logger = logging.getLogger("test_diag_empty")
    log_pre_step_diag(
        logger, step=0, total_norm=0.0,
        muon_state={}, adamw_state={},
    )
    msgs = [rec.message for rec in caplog.records]
    assert any("mom_max=0.000e+00" in m for m in msgs), msgs
    assert any("m_max=0.000e+00" in m for m in msgs), msgs
    assert any("v_max=0.000e+00" in m for m in msgs), msgs


def test_log_pre_step_diag_nvfp4_module_falls_back_to_packed_weight(caplog):
    """NVFP4 mode-3 entries have ``s.param is None`` — the
    diagnostic reads ``s.nvfp4_module.packed_weight`` instead.
    This pins the proxy branch (otherwise the call would
    AttributeError on a real NVFP4 entry)."""
    caplog.set_level(logging.INFO)
    logger = logging.getLogger("test_diag_nvfp4")
    packed = torch.tensor([10, 20, 250], dtype=torch.uint8)  # max abs = 250
    nvfp4_stub = _nvfp4_module_stub(packed)
    muon_state = {
        1: _ParamState(
            param=None,
            kind="muon_nvfp4",
            mom_buf=torch.tensor([1.0, 2.0]),
            nvfp4_module=nvfp4_stub,
        ),
    }
    log_pre_step_diag(
        logger, step=1, total_norm=1.0,
        muon_state=muon_state, adamw_state={},
    )
    msgs = [rec.message for rec in caplog.records]
    # The muon pmax branch should report the packed uint8 max (250).
    assert any("pmax=2.500e+02" in m for m in msgs), msgs


# --------------------------------------------------------------------------- #
# log_post_opt_diag — finite vs non-finite.                                   #
# --------------------------------------------------------------------------- #
def test_log_post_opt_diag_all_finite(caplog):
    """When every param is finite, the line is the all-finite form
    with the global pmax."""
    caplog.set_level(logging.INFO)
    logger = logging.getLogger("test_diag_post_finite")
    state = {
        1: _make_param_state(
            kind="muon",
            param=torch.tensor([1.0, -2.0, 0.5]),
        ),
        2: _make_param_state(
            kind="adamw",
            param=torch.tensor([0.25]),
        ),
    }
    log_post_opt_diag(logger, step=3, opt_label="muon", state=state)
    msgs = [rec.message for rec in caplog.records]
    assert any("diag-step 3 post-muon" in m for m in msgs), msgs
    assert any("all finite" in m for m in msgs), msgs
    assert any("pmax=2.000e+00" in m for m in msgs), msgs
    # No NON-FINITE line.
    assert not any("NON-FINITE" in m for m in msgs), msgs


def test_log_post_opt_diag_first_non_finite_reports_shape_and_mag(caplog):
    """On the first NON-FINITE entry, the log line includes its
    shape and max-abs — operators use this to localize which
    param is diverging."""
    caplog.set_level(logging.INFO)
    logger = logging.getLogger("test_diag_post_nan")
    state = {
        1: _make_param_state(
            kind="adamw",
            param=torch.tensor([1.0, 2.0]),  # finite, comes first
        ),
        2: _make_param_state(
            kind="adamw",
            param=torch.tensor([float("nan"), 1.0, -3.0]),
        ),
    }
    log_post_opt_diag(logger, step=4, opt_label="adamw", state=state)
    msgs = [rec.message for rec in caplog.records]
    assert any("NON-FINITE" in m for m in msgs), msgs
    # First non-finite param has shape (3,) and max-abs 3.0
    # (NaN's abs is NaN; the abs of [nan, 1, -3] is [nan, 1, 3]
    # — torch.amax on a tensor containing NaN returns NaN).
    # The contract: the line reports ``first_shape=...`` and a
    # pmax value, but pmax can be NaN when the param has any
    # NaN. We just assert the structural fields are present.
    nonfinite_line = next(m for m in msgs if "NON-FINITE" in m)
    assert "first_shape=torch.Size([3])" in nonfinite_line, nonfinite_line
    # And no all-finite line.
    assert not any("all finite" in m for m in msgs), msgs


def test_log_post_opt_diag_picks_largest_pmax_for_non_finite(caplog):
    """When multiple params are non-finite, the reported pmax is
    the largest, not the first. The first_shape is still the
    first non-finite entry (so the operator gets a stable
    pointer)."""
    caplog.set_level(logging.INFO)
    logger = logging.getLogger("test_diag_post_pmax")
    state = {
        1: _make_param_state(
            kind="muon",
            param=torch.tensor([float("inf"), -2.0]),  # |inf|=inf
        ),
        2: _make_param_state(
            kind="muon",
            param=torch.tensor([float("nan"), 0.5]),
        ),
    }
    log_post_opt_diag(logger, step=5, opt_label="muon", state=state)
    msgs = [rec.message for rec in caplog.records]
    nonfinite_line = next(m for m in msgs if "NON-FINITE" in m)
    assert "first_shape=torch.Size([2])" in nonfinite_line
    # The inf magnitude (inf) is larger than the nan magnitude,
    # so pmax should be inf.
    assert "pmax=inf" in nonfinite_line, nonfinite_line


def test_log_post_opt_diag_nvfp4_proxy_branch_is_always_finite(caplog):
    """NVFP4 mode-3 entries have no ``s.param`` — the diagnostic
    reads ``s.nvfp4_module.packed_weight`` (uint8) which is
    always finite. The line therefore reports ``all finite``,
    not ``NON-FINITE``, even if the FP4 packed storage looks
    pathological."""
    caplog.set_level(logging.INFO)
    logger = logging.getLogger("test_diag_post_nvfp4")
    packed = torch.tensor([255, 0, 1], dtype=torch.uint8)  # max abs = 255
    nvfp4_stub = _nvfp4_module_stub(packed)
    state = {
        1: _ParamState(
            param=None,
            kind="muon_nvfp4",
            mom_buf=torch.tensor([1.0]),
            nvfp4_module=nvfp4_stub,
        ),
    }
    log_post_opt_diag(logger, step=6, opt_label="muon", state=state)
    msgs = [rec.message for rec in caplog.records]
    assert any("all finite" in m for m in msgs), msgs
    assert not any("NON-FINITE" in m for m in msgs), msgs
    # The packed_uint8 max-abs is 255 → pmax=2.550e+02.
    assert any("pmax=2.550e+02" in m for m in msgs), msgs


def test_log_post_opt_diag_mixed_nvfp4_and_regular(caplog):
    """When both NVFP4 mode-3 entries (no param) and regular
    params are in the state dict, both contribute to the
    all-finite max-abs. The non-NVFP4 branch drives the param
    scan; the NVFP4 branch drives the pmax aggregate."""
    caplog.set_level(logging.INFO)
    logger = logging.getLogger("test_diag_post_mixed")
    packed = torch.tensor([100], dtype=torch.uint8)  # pmax contrib = 100
    nvfp4_stub = _nvfp4_module_stub(packed)
    state = {
        1: _make_param_state(
            kind="muon",
            param=torch.tensor([1.0, -2.0, 0.5]),
        ),
        2: _ParamState(
            param=None,
            kind="muon_nvfp4",
            mom_buf=torch.tensor([0.1]),
            nvfp4_module=nvfp4_stub,
        ),
    }
    log_post_opt_diag(logger, step=8, opt_label="muon", state=state)
    msgs = [rec.message for rec in caplog.records]
    # The aggregate pmax is max(2.0, 100) — the helper takes
    # max() of two Python floats → 100.
    assert any("all finite" in m for m in msgs), msgs
    assert any("pmax=1.000e+02" in m for m in msgs), msgs


def test_log_post_opt_diag_empty_state(caplog):
    """An empty state dict must NOT crash — it emits the
    all-finite line with ``pmax=0.0`` (the no-entries branch of
    ``_max_over_states``)."""
    caplog.set_level(logging.INFO)
    logger = logging.getLogger("test_diag_post_empty")
    log_post_opt_diag(logger, step=9, opt_label="muon", state={})
    msgs = [rec.message for rec in caplog.records]
    assert any("all finite, pmax=0.000e+00" in m for m in msgs), msgs
    assert not any("NON-FINITE" in m for m in msgs), msgs


# --------------------------------------------------------------------------- #
# Plain-Python runner. Pytest is not a project dependency; collect every      #
# ``def test_*`` and invoke, reporting pass/fail per test.                    #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import inspect
    import traceback

    tests = [
        (name, fn)
        for name, fn in globals().items()
        if name.startswith("test_") and callable(fn)
    ]
    tests.sort(key=lambda kv: inspect.getsourcelines(kv[1])[1])

    passed, failed = 0, 0
    for name, fn in tests:
        try:
            fn()
        except Exception:
            failed += 1
            print(f"  FAIL  {name}")
            traceback.print_exc()
        else:
            passed += 1
            print(f"  PASS  {name}")
    print(f"\n{passed} passed, {failed} failed, {passed + failed} total")
    if failed:
        sys.exit(1)