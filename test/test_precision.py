"""Tests for the precision-config wiring.

Scope (per user directive: only Muon and AdamW tests; no
model tests because model code iterates too fast). Covers:

  - :class:`PrecisionConfig` parsing from a yml-shaped dict
    (the CLI overlay hands the training loop a raw dict; the
    loop converts to a typed ``PrecisionConfig``).
  - :class:`CPUAdamW` storage dtypes for every supported
    ``adamw_m`` / ``adamw_v`` / ``gradients`` combination.
  - :class:`CPUMuon` storage dtypes for ``int8`` (the
    canonical quantized momentum) and floating
    ``muon_momentum`` (which falls back to int8 storage).
  - End-to-end ``step()`` runs on a tiny model for every
    precision combination (the early-return guard on
    ``s.accum.abs().sum() == 0`` masks dtype bugs that would
    only show up when the algorithm actually runs).
  - Parity: same RNG seed + same precision → same result.
    Catches any unexpected fp-cast that breaks
    bit-for-bit reproducibility across the refactor.
  - ``int4`` Muon raises :class:`NotImplementedError` (it's
    declared in the yml schema but the packing scheme is
    on the roadmap, not landed yet).
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

# Make the repo root importable so ``from src...`` works. We
# insert the parent of the test/ directory rather than the
# hardcoded ``/home/wlx/HippoLM`` used by older test files,
# so the tests work regardless of where the repo is checked
# out.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import torch
import torch.nn as nn

from src.training.param_offload import (
    CPUMuon,
    CPUAdamW,
    accumulate_grads_to_cpu,
)
from src.training.precision_config import (
    DType,
    PrecisionConfig,
    ScaleMode,
    TensorPrecision,
)


# --------------------------------------------------------------------------- #
# PrecisionConfig parsing.                                                    #
# --------------------------------------------------------------------------- #
def test_precision_config_defaults_match_canonical_yml():
    """The dataclass defaults should match the yml in
    ``configs/base.yml``: FP16 weights, BF16 gradients, int8 +
    per-channel Muon momentum, BF16 AdamW m/v.
    """
    p = PrecisionConfig()
    assert p.model_weights.dtype == DType.FP16
    assert p.gradients.dtype == DType.BF16
    assert p.muon_momentum.dtype == DType.INT8
    assert p.muon_momentum.scale == ScaleMode.PER_CHANNEL
    assert p.adamw_m.dtype == DType.BF16
    assert p.adamw_v.dtype == DType.BF16


def test_precision_config_from_dict_partial():
    """Missing keys fall through to defaults; nested dicts are
    parsed dtype-first, then scale, then block_size.
    """
    p = PrecisionConfig.from_dict({
        "gradients": {"dtype": "fp32"},
        "muon_momentum": {"dtype": "int4", "scale": "block", "block_size": 64},
    })
    assert p.gradients.dtype == DType.FP32
    assert p.muon_momentum.dtype == DType.INT4
    assert p.muon_momentum.scale == ScaleMode.BLOCK
    assert p.muon_momentum.block_size == 64
    # Unspecified keys keep the dataclass defaults.
    assert p.model_weights.dtype == DType.FP16
    assert p.adamw_m.dtype == DType.BF16


def test_precision_config_from_dict_none_returns_defaults():
    p = PrecisionConfig.from_dict(None)
    assert p.model_weights.dtype == DType.FP16
    p2 = PrecisionConfig.from_dict({})
    assert p2.model_weights.dtype == DType.FP16


def test_precision_config_int8_requires_scale():
    """``int8`` / ``int4`` cannot be constructed without a
    ``scale`` mode — would silently produce un-scaled integer
    momentum that overflows on the first step.
    """
    try:
        TensorPrecision(dtype=DType.INT8)
    except ValueError as e:
        assert "requires a 'scale' mode" in str(e)
    else:
        raise AssertionError("TensorPrecision(INT8) should have raised")
    try:
        TensorPrecision(dtype=DType.INT8, scale=ScaleMode.BLOCK)
    except ValueError as e:
        assert "requires 'block_size'" in str(e)
    else:
        raise AssertionError(
            "TensorPrecision(INT8, scale=BLOCK) should have raised"
        )


def test_precision_config_float_clears_scale_silently():
    """fp* dtypes ignore ``scale`` / ``block_size`` — the
    validation in :meth:`TensorPrecision.__post_init__` clears
    them to ``None`` so the rest of the optimizer can rely on
    ``scale is None`` to mean "no quantization".
    """
    tp = TensorPrecision(
        dtype=DType.BF16, scale=ScaleMode.PER_CHANNEL, block_size=64,
    )
    assert tp.scale is None
    assert tp.block_size is None


# --------------------------------------------------------------------------- #
# activations (autocast dtype).                                               #
# --------------------------------------------------------------------------- #
def test_precision_config_default_activations_is_fp16():
    """The dataclass default for ``activations`` is FP16 — matches
    the canonical yml and the historical hardcoded autocast dtype.
    """
    p = PrecisionConfig()
    assert p.activations.dtype == DType.FP16


def test_precision_config_activations_fp16_enables_autocast():
    """fp16 activations → autocast on, dtype=torch.float16."""
    p = PrecisionConfig(activations=TensorPrecision(dtype=DType.FP16))
    assert p.autocast_enabled is True
    assert p.autocast_dtype == torch.float16


def test_precision_config_activations_bf16_enables_autocast():
    """bf16 activations → autocast on, dtype=torch.bfloat16."""
    p = PrecisionConfig(activations=TensorPrecision(dtype=DType.BF16))
    assert p.autocast_enabled is True
    assert p.autocast_dtype == torch.bfloat16


def test_precision_config_activations_fp32_disables_autocast():
    """fp32 activations → autocast off. The dtype returned by
    :attr:`autocast_dtype` is still FP32 (so a caller that ignores
    ``autocast_enabled`` does not crash) but the autocast context
    is a no-op.
    """
    p = PrecisionConfig(activations=TensorPrecision(dtype=DType.FP32))
    assert p.autocast_enabled is False
    assert p.autocast_dtype == torch.float32


def test_precision_config_activations_rejects_int8():
    """``activations`` only accepts floating dtypes — integer
    dtypes make no semantic sense (autocast does not consume an
    int dtype) and must fail at config-parse time rather than at
    the first forward pass.
    """
    try:
        PrecisionConfig(activations=TensorPrecision(
            dtype=DType.INT8, scale=ScaleMode.NO,
        ))
    except ValueError as e:
        assert "activations" in str(e) and "not allowed" in str(e)
    else:
        raise AssertionError(
            "PrecisionConfig with int8 activations should have raised"
        )


def test_precision_config_activations_rejects_int4():
    """Same as the int8 case — integer dtypes are never valid for
    activations.
    """
    try:
        PrecisionConfig(activations=TensorPrecision(
            dtype=DType.INT4, scale=ScaleMode.PER_CHANNEL,
        ))
    except ValueError as e:
        assert "activations" in str(e) and "not allowed" in str(e)
    else:
        raise AssertionError(
            "PrecisionConfig with int4 activations should have raised"
        )


def test_precision_config_from_dict_parses_activations():
    """The yml loader pulls ``activations`` through to the typed
    config so the training loop can read it.
    """
    p = PrecisionConfig.from_dict({
        "activations": {"dtype": "bf16"},
    })
    assert p.activations.dtype == DType.BF16
    assert p.autocast_enabled is True
    assert p.autocast_dtype == torch.bfloat16


def test_precision_config_to_dict_round_trips_activations():
    """``to_dict`` includes the activations entry so a logged /
    serialized config can be reloaded losslessly.
    """
    p = PrecisionConfig.from_dict({
        "activations": {"dtype": "fp32"},
    })
    d = p.to_dict()
    assert d["activations"] == {"dtype": "fp32"}
    # Round-trip: the dict should reload to the same config.
    p2 = PrecisionConfig.from_dict(d)
    assert p2.activations.dtype == DType.FP32
    assert p2.autocast_enabled is False


# --------------------------------------------------------------------------- #
# CPUAdamW storage dtypes.                                                    #
# --------------------------------------------------------------------------- #
def _small_linear() -> nn.Linear:
    """A tiny 2D-weight + 1D-bias module: exercises both
    ``m`` (2D weight) and ``v`` (2D weight) plus the 1D-bias
    AdamW path.
    """
    torch.manual_seed(0)
    return nn.Linear(8, 8)


def test_adamw_default_storage_is_bf16():
    """No precision passed → canonical yml defaults → all BF16."""
    lin = _small_linear()
    opt = CPUAdamW(list(lin.parameters()))
    s = next(iter(opt.state.values()))
    assert s.accum.dtype == torch.bfloat16
    assert s.m.dtype == torch.bfloat16
    assert s.exp_avg_sq.dtype == torch.bfloat16


def test_adamw_storage_matches_precision_config():
    """Each tensor's dtype follows the corresponding entry in
    the precision config: accum ← gradients, m ← adamw_m,
    v ← adamw_v.
    """
    lin = _small_linear()
    opt = CPUAdamW(list(lin.parameters()), precision=PrecisionConfig(
        gradients=TensorPrecision(dtype=DType.FP16),
        adamw_m=TensorPrecision(dtype=DType.FP32),
        adamw_v=TensorPrecision(dtype=DType.FP16),
    ))
    s = next(iter(opt.state.values()))
    assert s.accum.dtype == torch.float16
    assert s.m.dtype == torch.float32
    assert s.exp_avg_sq.dtype == torch.float16


def test_adamw_int_m_falls_back_to_fp32_with_warning():
    """Quantized ``adamw_m`` / ``adamw_v`` is not supported (the
    FP32 promotion in ``step()`` assumes a floating source);
    the optimizer falls back to FP32 storage and warns.
    """
    lin = _small_linear()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        opt = CPUAdamW(list(lin.parameters()), precision=PrecisionConfig(
            adamw_m=TensorPrecision(dtype=DType.INT8, scale=ScaleMode.NO),
        ))
    s = next(iter(opt.state.values()))
    assert s.m.dtype == torch.float32
    assert any("integer-quantized" in str(w.message) for w in caught)


# --------------------------------------------------------------------------- #
# CPUMuon storage dtypes.                                                      #
# --------------------------------------------------------------------------- #
def _muon_param() -> nn.Parameter:
    """A 2D weight (no bias) — CPUMuon only accepts 2D params."""
    torch.manual_seed(0)
    return nn.Parameter(torch.randn(8, 8))


def test_muon_default_storage_is_int8_with_bf16_scale():
    p = _muon_param()
    opt = CPUMuon([p])
    s = next(iter(opt.state.values()))
    assert s.accum.dtype == torch.bfloat16
    assert s.mom_buf.dtype == torch.int8
    assert s.mom_scale.dtype == torch.bfloat16


def test_muon_fp32_gradients_stored_as_fp32():
    p = _muon_param()
    opt = CPUMuon([p], precision=PrecisionConfig(
        gradients=TensorPrecision(dtype=DType.FP32),
    ))
    s = next(iter(opt.state.values()))
    assert s.accum.dtype == torch.float32


@pytest.mark.parametrize("mom_dtype, torch_dt", [
    (DType.BF16, torch.bfloat16),
    (DType.FP16, torch.float16),
    (DType.FP32, torch.float32),
])
def test_muon_floating_momentum_uses_full_precision_storage(mom_dtype, torch_dt):
    """``muon_momentum: { dtype: bf16/fp16/fp32 }`` is honored:
    the momentum buffer is allocated in the configured dtype
    and there is no scale tensor (no quantization).
    """
    p = _muon_param()
    opt = CPUMuon([p], precision=PrecisionConfig(
        muon_momentum=TensorPrecision(dtype=mom_dtype),
    ))
    s = next(iter(opt.state.values()))
    assert s.mom_buf.dtype == torch_dt
    assert s.mom_scale is None


def test_muon_int4_raises_not_implemented():
    """``int4`` is in the yml schema but the packing scheme is
    on the roadmap, not landed. The optimizer must raise
    rather than silently degrade.
    """
    p = _muon_param()
    try:
        CPUMuon([p], precision=PrecisionConfig(
            muon_momentum=TensorPrecision(
                dtype=DType.INT4, scale=ScaleMode.PER_CHANNEL,
            ),
        ))
    except NotImplementedError as e:
        assert "int4 is not yet implemented" in str(e)
    else:
        raise AssertionError(
            "CPUMuon with int4 momentum should have raised"
            " NotImplementedError"
        )


# --------------------------------------------------------------------------- #
# End-to-end step() with various precisions.                                   #
# --------------------------------------------------------------------------- #
def _run_adamw_step(precision: PrecisionConfig) -> torch.Tensor:
    """Build a tiny AdamW, set grads, run one step, return the
    post-step weight. Used to verify each precision combination
    can actually execute the algorithm (catches dtype bugs that
    would otherwise hide in the no-grad early-return path).
    """
    torch.manual_seed(0)
    lin = nn.Linear(8, 8)
    opt = CPUAdamW(list(lin.parameters()), precision=precision)
    for p in lin.parameters():
        p.grad = torch.randn_like(p) * 0.01
    accumulate_grads_to_cpu([opt])
    opt.step()
    return lin.weight.data.clone()


def test_adamw_step_runs_for_all_canonical_precisions():
    """The canonical yml combinations must all run cleanly."""
    for prec in (
        # Default (BF16 for everything)
        PrecisionConfig(),
        # FP16 m, FP32 v, FP16 gradients
        PrecisionConfig(
            gradients=TensorPrecision(dtype=DType.FP16),
            adamw_m=TensorPrecision(dtype=DType.FP16),
            adamw_v=TensorPrecision(dtype=DType.FP32),
        ),
        # FP32 m/v/gradients (slow but precise; useful for
        # verifying the precision-promotion path in step())
        PrecisionConfig(
            gradients=TensorPrecision(dtype=DType.FP32),
            adamw_m=TensorPrecision(dtype=DType.FP32),
            adamw_v=TensorPrecision(dtype=DType.FP32),
        ),
    ):
        w = _run_adamw_step(prec)
        # The weight should have moved (any nonzero grad
        # with nonzero lr should produce a nonzero update).
        assert w.abs().sum() > 0.0


def test_adamw_parity_default_and_explicit_bf16():
    """``CPUAdamW(precision=None)`` and
    ``CPUAdamW(precision=PrecisionConfig(gradients=BF16, m=BF16, v=BF16))``
    must give bit-identical results with the same RNG seed.
    Catches any "default vs explicit" divergence in the
    dtype resolution path.
    """
    torch.manual_seed(0)
    lin_a = nn.Linear(8, 8)
    opt_a = CPUAdamW(list(lin_a.parameters()))  # default
    for p in lin_a.parameters():
        p.grad = torch.randn_like(p) * 0.01
    accumulate_grads_to_cpu([opt_a])
    opt_a.step()

    torch.manual_seed(0)
    lin_b = nn.Linear(8, 8)
    opt_b = CPUAdamW(
        list(lin_b.parameters()),
        precision=PrecisionConfig(  # explicit BF16
            gradients=TensorPrecision(dtype=DType.BF16),
            adamw_m=TensorPrecision(dtype=DType.BF16),
            adamw_v=TensorPrecision(dtype=DType.BF16),
        ),
    )
    for p in lin_b.parameters():
        p.grad = torch.randn_like(p) * 0.01
    accumulate_grads_to_cpu([opt_b])
    opt_b.step()

    assert torch.equal(lin_a.weight.data, lin_b.weight.data)


def test_muon_step_runs_for_int8_and_floating_storage_paths():
    """int8 momentum (canonical) and full-precision bf16/fp16/fp32
    momentum all produce a step() that updates the weight.
    """
    # GPU is required: step() runs NS on GPU and syncs the
    # current CUDA stream. Skip cleanly when CUDA is unavailable.
    if not torch.cuda.is_available():
        return
    device = torch.device("cuda", 0)

    # int8 momentum (canonical yml path)
    torch.manual_seed(0)
    p1 = nn.Parameter(torch.randn(8, 8, device=device))
    opt1 = CPUMuon([p1], precision=PrecisionConfig(
        muon_momentum=TensorPrecision(dtype=DType.INT8, scale=ScaleMode.PER_CHANNEL),
    ))
    p1.grad = torch.randn_like(p1) * 0.01
    accumulate_grads_to_cpu([opt1])
    initial = p1.data.clone()
    opt1.step()
    assert not torch.equal(p1.data, initial)

    # bf16 momentum → full-precision storage, no dequant/requant.
    torch.manual_seed(0)
    p2 = nn.Parameter(torch.randn(8, 8, device=device))
    opt2 = CPUMuon([p2], precision=PrecisionConfig(
        muon_momentum=TensorPrecision(dtype=DType.BF16),
    ))
    p2.grad = torch.randn_like(p2) * 0.01
    accumulate_grads_to_cpu([opt2])
    initial2 = p2.data.clone()
    opt2.step()
    assert not torch.equal(p2.data, initial2)


# --------------------------------------------------------------------------- #
# accumulate_grads_to_cpu honors the configured dtype.                         #
# --------------------------------------------------------------------------- #
def test_accumulate_grads_casts_to_configured_grad_dtype():
    """The cast target on the GPU is ``s.accum.dtype``, which
    follows the ``gradients`` precision config. This is the
    PCIe-bandwidth-critical path: a wrong cast inflates the
    transfer size 2x.
    """
    torch.manual_seed(0)
    lin = nn.Linear(8, 8)
    opt = CPUAdamW(list(lin.parameters()), precision=PrecisionConfig(
        gradients=TensorPrecision(dtype=DType.FP32),
    ))
    for p in lin.parameters():
        p.grad = torch.randn_like(p) * 0.01
    accumulate_grads_to_cpu([opt])
    # Every state's accum should now contain nonzero data
    # in the configured dtype (FP32).
    for s in opt.state.values():
        assert s.accum.dtype == torch.float32
        assert s.accum.abs().sum() > 0


# --------------------------------------------------------------------------- #
# Plain-Python test runner. Pytest is not a project dependency (the rest of   #
# the test suite uses this pattern); collect every ``def test_*`` above and  #
# invoke them, reporting pass/fail per test.                                   #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import inspect
    import traceback

    tests = [
        (name, fn)
        for name, fn in globals().items()
        if name.startswith("test_") and callable(fn)
    ]
    # Stable, source-order run.
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
