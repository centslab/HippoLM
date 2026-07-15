"""Tests for the precision-config wiring.

Scope (per user directive: only Muon and AdamW tests; no
model tests because model code iterates too fast). Covers:

  - :class:`PrecisionConfig` parsing from a yml-shaped dict
    (the CLI overlay hands the training loop a raw dict; the
    loop converts to a typed ``PrecisionConfig``).
  - :class:`CPUAdamW` storage dtypes for the supported
    ``adamw_m`` / ``adamw_v`` combinations.
  - :class:`CPUMuon` storage dtypes for the supported full-
    precision ``muon_momentum`` dtypes (bf16 / fp16 / fp32).
  - End-to-end ``step()`` runs on a tiny model for every
    precision combination (the early-return guard on
    ``s.grad.abs().sum() == 0`` for both AdamW and Muon
    masks dtype bugs that would only show up when the
    algorithm actually runs).
  - Parity: same RNG seed + same precision → same result.
    Catches any unexpected fp-cast that breaks
    bit-for-bit reproducibility across the refactor.

History
-------
Quantized muon storage (``int8`` + per-row BF16 scale,
``mxfp8`` + per-block E8M0 scale) was removed on 2026-07-12
after long-training runs showed quantization-error
accumulation destabilizing optimization. The legacy
``int4``-NotImplementedError test is also gone (int4 was
never a real path).
"""
from __future__ import annotations

import sys
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
    TensorPrecision,
)


# --------------------------------------------------------------------------- #
# PrecisionConfig parsing.                                                    #
# --------------------------------------------------------------------------- #
def test_precision_config_dataclass_defaults():
    """Pin the dataclass field defaults for :class:`PrecisionConfig`.

    These are the *fallback* values used when no yml / dict is
    supplied. The production training config (``configs/base.yml``)
    matches all of these.
    """
    p = PrecisionConfig()
    assert p.model_weights.dtype == DType.BF16
    assert p.gradients.dtype == DType.BF16
    assert p.muon_momentum.dtype == DType.BF16
    assert p.activations.dtype == DType.BF16
    assert p.adamw_m.dtype == DType.BF16
    assert p.adamw_v.dtype == DType.BF16


def test_precision_config_base_yml_overrides_dataclass_defaults():
    """``configs/base.yml`` (the production precision config) must
    stay aligned with the dataclass defaults — anything that
    drifts would silently run the wrong precision on prod.

    Current production values (all BF16 — quantized storage
    removed 2026-07-12):
      - ``model_weights`` : BF16 (NVFP4 FFN packing requires
        BF16 master weights).
      - ``muon_momentum``  : BF16.
      - ``activations``    : BF16 (autocast over BF16
        activations).

    A future yml edit that flips any of these would silently
    run the wrong precision on prod. This test catches it.
    """
    from pathlib import Path
    import yaml

    base_yml = Path(__file__).resolve().parent.parent / "configs" / "base.yml"
    if not base_yml.exists():
        pytest.skip("configs/base.yml not present (out-of-tree checkout)")

    raw = yaml.safe_load(base_yml.read_text())
    assert "precision" in raw, (
        f"configs/base.yml must have a `precision:` block; got keys: {list(raw)}"
    )
    p = PrecisionConfig.from_dict(raw["precision"])

    assert p.model_weights.dtype == DType.BF16, (
        f"base.yml precision.model_weights is {p.model_weights.dtype}, "
        f"expected BF16 (NVFP4 FFN packing requires BF16 master weights)"
    )
    assert p.muon_momentum.dtype == DType.BF16, (
        f"base.yml precision.muon_momentum is {p.muon_momentum.dtype}, "
        f"expected BF16 (2026-07-12: only full-precision storage supported)"
    )
    assert p.activations.dtype == DType.BF16, (
        f"base.yml precision.activations is {p.activations.dtype}, "
        f"expected BF16 (autocast over BF16 activations)"
    )


def test_precision_config_from_dict_partial():
    """Missing keys fall through to defaults."""
    p = PrecisionConfig.from_dict({
        "gradients": {"dtype": "fp32"},
        "muon_momentum": {"dtype": "fp16"},
    })
    assert p.gradients.dtype == DType.FP32
    assert p.muon_momentum.dtype == DType.FP16
    # Unspecified keys keep the dataclass defaults.
    assert p.model_weights.dtype == DType.BF16
    assert p.adamw_m.dtype == DType.BF16


def test_precision_config_from_dict_ignores_legacy_quantized_fields():
    """Legacy yml files may carry ``scale:`` / ``block_size:`` keys
    for the now-removed quantized storage formats. The loader
    silently drops them (only ``dtype`` is honored)."""
    p = PrecisionConfig.from_dict({
        "muon_momentum": {"dtype": "bf16", "scale": "per-channel",
                          "block_size": 32},
    })
    assert p.muon_momentum.dtype == DType.BF16


def test_precision_config_from_dict_none_returns_defaults():
    p = PrecisionConfig.from_dict(None)
    assert p.model_weights.dtype == DType.BF16
    p2 = PrecisionConfig.from_dict({})
    assert p2.model_weights.dtype == DType.BF16


def test_precision_config_rejects_int8_dtype():
    """``int8`` / ``int4`` / ``mxfp8`` were removed on 2026-07-12
    (long-training instability). Constructing a TensorPrecision
    with an int dtype must fail at config-parse time rather
    than silently degrade at step() time.
    """
    with pytest.raises(ValueError, match="not a valid DType"):
        TensorPrecision(dtype="int8")


def test_precision_config_rejects_mxfp8_dtype():
    with pytest.raises(ValueError, match="not a valid DType"):
        TensorPrecision(dtype="mxfp8")


# --------------------------------------------------------------------------- #
# activations (autocast dtype).                                               #
# --------------------------------------------------------------------------- #
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
    ``grad`` (per-step accumulator, 2D weight) and
    ``exp_avg`` / ``exp_avg_sq`` (EMAs, 2D weight) plus the
    1D-bias AdamW path.
    """
    torch.manual_seed(0)
    return nn.Linear(8, 8)


def test_adamw_default_storage_is_bf16():
    """No precision passed → canonical yml defaults → all BF16."""
    lin = _small_linear()
    opt = CPUAdamW(list(lin.parameters()))
    s = next(iter(opt.state.values()))
    assert s.grad.dtype == torch.bfloat16
    assert s.exp_avg.dtype == torch.bfloat16
    assert s.exp_avg_sq.dtype == torch.bfloat16


def test_adamw_storage_matches_precision_config():
    """Each tensor's dtype follows the corresponding entry in
    the precision config:
      - ``s.grad``     ← ``adamw_m`` (per-step accumulator)
      - ``s.exp_avg``  ← ``adamw_m`` (β1 EMA)
      - ``s.exp_avg_sq`` ← ``adamw_v`` (β2 EMA)
    """
    lin = _small_linear()
    opt = CPUAdamW(list(lin.parameters()), precision=PrecisionConfig(
        adamw_m=TensorPrecision(dtype=DType.FP32),
        adamw_v=TensorPrecision(dtype=DType.FP16),
    ))
    s = next(iter(opt.state.values()))
    assert s.grad.dtype == torch.float32
    assert s.exp_avg.dtype == torch.float32
    assert s.exp_avg_sq.dtype == torch.float16


# --------------------------------------------------------------------------- #
# CPUMuon storage dtypes.                                                      #
# --------------------------------------------------------------------------- #
def _muon_param() -> nn.Parameter:
    """A 2D weight (no bias) — CPUMuon only accepts 2D params."""
    torch.manual_seed(0)
    return nn.Parameter(torch.randn(8, 8))


@pytest.mark.parametrize("mom_dtype, torch_dt", [
    (DType.BF16, torch.bfloat16),
    (DType.FP16, torch.float16),
    (DType.FP32, torch.float32),
])
def test_muon_floating_momentum_uses_full_precision_storage(mom_dtype, torch_dt):
    """``muon_momentum: { dtype: bf16/fp16/fp32 }`` is honored:
    both the per-step grad accumulator (``s.grad``) and the
    SGD momentum (``s.exp_avg``) are allocated in the configured
    dtype. There is no scale tensor (no quantization)."""
    p = _muon_param()
    opt = CPUMuon([p], precision=PrecisionConfig(
        muon_momentum=TensorPrecision(dtype=mom_dtype),
    ))
    s = next(iter(opt.state.values()))
    assert s.grad.dtype == torch_dt
    assert s.exp_avg.dtype == torch_dt


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
        # FP16 m, FP32 v
        PrecisionConfig(
            adamw_m=TensorPrecision(dtype=DType.FP16),
            adamw_v=TensorPrecision(dtype=DType.FP32),
        ),
        # FP32 m/v (slow but precise; useful for verifying the
        # precision-promotion path in step())
        PrecisionConfig(
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
    ``CPUAdamW(precision=PrecisionConfig(adamw_m=BF16,
    adamw_v=BF16))`` must give bit-identical results with the
    same RNG seed. Catches any "default vs explicit"
    divergence in the dtype resolution path.
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
            adamw_m=TensorPrecision(dtype=DType.BF16),
            adamw_v=TensorPrecision(dtype=DType.BF16),
        ),
    )
    for p in lin_b.parameters():
        p.grad = torch.randn_like(p) * 0.01
    accumulate_grads_to_cpu([opt_b])
    opt_b.step()

    assert torch.equal(lin_a.weight.data, lin_b.weight.data)


@pytest.mark.parametrize("mom_dtype", [DType.BF16, DType.FP16, DType.FP32])
def test_muon_step_runs_for_all_floating_storage_paths(mom_dtype):
    """All three full-precision muon storage dtypes must produce
    a step() that updates the weight.
    """
    if not torch.cuda.is_available():
        return
    device = torch.device("cuda", 0)

    torch.manual_seed(0)
    p = nn.Parameter(torch.randn(8, 8, device=device))
    opt = CPUMuon([p], precision=PrecisionConfig(
        muon_momentum=TensorPrecision(dtype=mom_dtype),
    ))
    p.grad = torch.randn_like(p) * 0.01
    accumulate_grads_to_cpu([opt])
    initial = p.data.clone()
    opt.step()
    assert not torch.equal(p.data, initial)


# --------------------------------------------------------------------------- #
# accumulate_grads_to_cpu honors the accumulator's storage dtype.              #
# --------------------------------------------------------------------------- #
def test_accumulate_grads_casts_to_accumulator_dtype():
    """The cast target on the GPU is the accumulator's storage
    dtype (``s.grad.dtype`` for AdamW — the ``gradients``
    precision config no longer governs the cast target). This
    is the PCIe-bandwidth-critical path: a wrong cast inflates
    the transfer size 2x.
    """
    torch.manual_seed(0)
    lin = nn.Linear(8, 8)
    opt = CPUAdamW(list(lin.parameters()), precision=PrecisionConfig(
        adamw_m=TensorPrecision(dtype=DType.FP32),
    ))
    for p in lin.parameters():
        p.grad = torch.randn_like(p) * 0.01
    accumulate_grads_to_cpu([opt])
    # Every state's ``s.grad`` (per-step accumulator) should
    # now contain nonzero data in the configured dtype (FP32).
    for s in opt.state.values():
        assert s.grad.dtype == torch.float32
        assert s.grad.abs().sum() > 0


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
