"""Tests for the optimizer ``state_dict`` / ``load_state_dict`` roundtrip
and the end-to-end ``save_checkpoint`` / ``load_checkpoint`` path.

Background
----------
:class:`CPUAdamW` and :class:`CPUMuon` do not subclass
:class:`torch.optim.Optimizer`, so the stock ``state_dict`` is
unavailable. Until ``state_dict`` / ``load_state_dict`` were
added on both classes, calling :func:`src.training.checkpoint.save_checkpoint`
threw ``AttributeError("'CPUMuon' object has no attribute 'state_dict'")``
every checkpoint_interval steps. These tests pin that fix.

Per-param entries are serialised as a list in insertion order
(matching the model's parameter iteration order from
:func:`build_param_groups`). Tensor data lives on CPU pinned
memory throughout — no CUDA needed for these tests.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from src.training.checkpoint import load_checkpoint, save_checkpoint
from src.training.param_offload import (
    CPUAdamW,
    CPUMuon,
)


# --------------------------------------------------------------------------- #
# Fixtures.                                                                   #
# --------------------------------------------------------------------------- #
@pytest.fixture
def tiny_model():
    """A tiny FP16 model exercising 1D (LayerNorm.weight) and 2D
    (Linear.weight) params so both AdamW and Muon state_dict paths
    get covered. The production param-routing rule sends 1D to
    AdamW and 2D to Muon — matching that routing keeps this fixture
    in sync with how :func:`build_param_groups` constructs the
    real optimizers."""
    torch.manual_seed(0)

    class M(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = nn.Linear(8, 16, bias=False)
            self.norm = nn.LayerNorm(16)

    return M().to(torch.float16)


@pytest.fixture
def adamw_opt(tiny_model):
    # Mirror production routing: 1D params → AdamW.
    return CPUAdamW(
        [p for p in tiny_model.parameters() if p.ndim < 2],
        lr=1e-4,
        betas=(0.9, 0.95),
    )


@pytest.fixture
def muon_opt(tiny_model):
    # Mirror production routing: 2D params → Muon.
    return CPUMuon(
        [p for p in tiny_model.parameters() if p.ndim >= 2],
        lr=1e-3,
    )


def _populate_state(model, adamw, muon, seed: int = 1) -> None:
    """Inject deterministic per-param data so state_dict has
    non-default content to roundtrip."""
    g = torch.Generator().manual_seed(seed)
    for p in model.parameters():
        p.grad = torch.randn(p.shape, generator=g, dtype=p.dtype) * 1e-3
    for s in adamw.state.values():
        # Post-2026-07-15: ``s.grad`` is the per-step
        # accumulator (was ``s.m`` in the merged-accumulator
        # design).
        s.grad.add_(s.param.grad.detach().to("cpu").reshape(-1))
        s.step = 3
    # Muon: post-2026-07-15 explicit-accumulator layout —
    # ``s.grad`` is the per-step accumulator, ``s.exp_avg``
    # is the SGD momentum (zero at this point, preserved
    # across steps). Quantized storage (int8 / mxfp8) was
    # removed 2026-07-12; only full-precision bf16 / fp16 /
    # fp32 storage remains.
    for s in muon.state.values():
        s.grad.add_(
            s.param.grad.detach().to(s.grad.dtype).reshape(-1)
        )
        s.step = 7


# --------------------------------------------------------------------------- #
# CPUAdamW state_dict.                                                        #
# --------------------------------------------------------------------------- #
def test_adamw_state_dict_contains_hyperparams_and_per_param_state(adamw_opt):
    sd = adamw_opt.state_dict()
    assert sd["lr"] == 1e-4
    assert sd["beta1"] == 0.9
    assert sd["beta2"] == 0.95
    # Per-param entries: LayerNorm.weight + LayerNorm.bias (1D → AdamW)
    assert len(sd["state"]) == 2
    for entry in sd["state"]:
        assert entry["kind"] == "adamw"
        assert entry["step"] == 0
        # Post-2026-07-15: state_dict keys are ``grad``,
        # ``exp_avg``, ``exp_avg_sq`` (the merged-accumulator
        # key ``m`` was deleted along with the merged design).
        assert entry["grad"] is not None
        assert entry["exp_avg"] is not None
        assert entry["exp_avg_sq"] is not None


def test_adamw_state_dict_load_state_dict_roundtrip_is_byte_equal(
    tiny_model, adamw_opt
):
    _populate_state(tiny_model, adamw_opt, muon_opt_for(tiny_model, default_muon_lr=1e-3), seed=1)
    sd = adamw_opt.state_dict()

    # Build a fresh optimizer with the same model and restore.
    fresh = CPUAdamW(
        [p for p in tiny_model.parameters() if p.ndim < 2],
        lr=1e-4, betas=(0.9, 0.95),
    )
    fresh.load_state_dict(sd)

    assert fresh.lr == adamw_opt.lr
    assert fresh.beta1 == adamw_opt.beta1
    assert fresh.beta2 == adamw_opt.beta2
    for s_new, s_old in zip(fresh.state.values(), adamw_opt.state.values()):
        # Post-2026-07-15 explicit-accumulator layout: round-trip
        # the per-step grad accumulator + β1 EMA + β2 EMA.
        assert torch.equal(s_new.grad, s_old.grad)
        assert torch.equal(s_new.exp_avg, s_old.exp_avg)
        assert torch.equal(s_new.exp_avg_sq, s_old.exp_avg_sq)
        assert s_new.step == s_old.step


def test_adamw_load_state_dict_rejects_param_count_mismatch(tiny_model):
    opt = CPUAdamW(list(tiny_model.parameters()), lr=1e-4)
    fake = {"lr": 1e-4, "beta1": 0.9, "beta2": 0.95, "eps": 1e-8,
            "weight_decay": 0.0, "state": []}
    with pytest.raises(ValueError, match="param count mismatch"):
        opt.load_state_dict(fake)


def test_adamw_load_state_dict_rejects_shape_mismatch(tiny_model):
    opt = CPUAdamW(list(tiny_model.parameters()), lr=1e-4)
    bad_state = [
        {"shape": [999, 999], "step": 0, "kind": "adamw",
         "grad": torch.zeros(1), "exp_avg": torch.zeros(1),
         "exp_avg_sq": torch.zeros(1)}
    ]
    with pytest.raises(ValueError, match="shape mismatch"):
        opt.load_state_dict(
            {"lr": 1e-4, "beta1": 0.9, "beta2": 0.95, "eps": 1e-8,
             "weight_decay": 0.0, "state": bad_state * len(opt.state)}
        )


# --------------------------------------------------------------------------- #
# CPUMuon state_dict.                                                         #
# --------------------------------------------------------------------------- #
def muon_opt_for(model, default_muon_lr=1e-3):
    return CPUMuon([p for p in model.parameters() if p.ndim >= 2], lr=default_muon_lr)


def adamw_opt_for(model):
    return CPUAdamW([p for p in model.parameters() if p.ndim < 2], lr=1e-4, betas=(0.9, 0.95))


def test_muon_state_dict_contains_hyperparams_and_per_param_state(tiny_model):
    opt = muon_opt_for(tiny_model)
    sd = opt.state_dict()
    assert sd["lr"] == 1e-3
    assert sd["momentum"] == 0.95
    assert sd["nesterov"] is True
    assert sd["ns_steps"] == 5
    # 2D params only: Linear.weight
    assert len(sd["state"]) == 1
    for entry in sd["state"]:
        assert entry["kind"] == "muon"


def test_muon_state_dict_roundtrip_is_byte_equal(tiny_model):
    """Full-precision momentum (the only supported storage since
    2026-07-12 — int8 / mxfp8 quantization removed)."""
    from src.training.param_offload import CPUMuon
    from src.training.precision_config import PrecisionConfig
    precision = PrecisionConfig.from_dict({
        "muon_momentum": {"dtype": "bf16"},
    })
    opt = CPUMuon(
        [p for p in tiny_model.parameters() if p.ndim >= 2],
        lr=1e-3,
        precision=precision,
    )
    for s in opt.state.values():
        # Post-2026-07-15 explicit-accumulator layout: ``s.grad``
        # (per-step accumulator) and ``s.exp_avg`` (SGD
        # momentum) share the configured storage dtype.
        assert s.grad.dtype == torch.bfloat16
        assert s.exp_avg.dtype == torch.bfloat16

    _populate_state(tiny_model, adamw_opt_for(tiny_model), opt, seed=3)
    sd = opt.state_dict()

    fresh = CPUMuon(
        [p for p in tiny_model.parameters() if p.ndim >= 2],
        lr=1e-3,
        precision=precision,
    )
    fresh.load_state_dict(sd)
    for s_new, s_old in zip(fresh.state.values(), opt.state.values()):
        # Round-trip both buffers; both must be byte-equal.
        assert torch.equal(s_new.grad, s_old.grad)
        assert torch.equal(s_new.exp_avg, s_old.exp_avg)
        assert s_new.step == s_old.step


def test_muon_load_state_dict_rejects_param_count_mismatch(tiny_model):
    opt = muon_opt_for(tiny_model)
    fake = {"lr": 1e-3, "momentum": 0.95, "nesterov": True, "ns_steps": 5,
            "weight_decay": 0.0, "state": []}
    with pytest.raises(ValueError, match="param count mismatch"):
        opt.load_state_dict(fake)


# --------------------------------------------------------------------------- #
# End-to-end save_checkpoint / load_checkpoint.                               #
# --------------------------------------------------------------------------- #
def test_save_checkpoint_writes_payload_with_optimizer_state(
    tiny_model, tmp_path
):
    """The original failure mode: save_checkpoint raised AttributeError
    because the CPU-offloaded optimizers had no state_dict()."""
    adamw = adamw_opt_for(tiny_model)
    muon = muon_opt_for(tiny_model)
    _populate_state(tiny_model, adamw, muon, seed=4)

    path = save_checkpoint(
        tiny_model,
        {"muon": muon, "adamw": adamw},
        scaler=None,
        step=42,
        loss=0.123,
        checkpoint_dir=tmp_path,
    )
    assert path.is_file()

    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert payload["step"] == 42
    assert abs(payload["loss"] - 0.123) < 1e-6
    assert "muon" in payload["optimizer_state_dict"]
    assert "adamw" in payload["optimizer_state_dict"]
    # per-param step counters should roundtrip
    for entry in payload["optimizer_state_dict"]["muon"]["state"]:
        assert entry["step"] == 7
    for entry in payload["optimizer_state_dict"]["adamw"]["state"]:
        assert entry["step"] == 3


def test_load_checkpoint_restores_step_loss_and_optimizer_state(
    tiny_model, tmp_path
):
    adamw = adamw_opt_for(tiny_model)
    muon = muon_opt_for(tiny_model)
    _populate_state(tiny_model, adamw, muon, seed=5)

    path = save_checkpoint(
        tiny_model, {"muon": muon, "adamw": adamw}, None,
        step=99, loss=0.5, checkpoint_dir=tmp_path,
    )

    # Fresh model + optimizers with the same architecture
    tiny_model2 = type(tiny_model)().to(torch.float16)
    # Copy weights so the model load doesn't strict-mismatch
    tiny_model2.load_state_dict(tiny_model.state_dict())
    adamw2 = adamw_opt_for(tiny_model2)
    muon2 = muon_opt_for(tiny_model2)

    step, loss = load_checkpoint(
        path, tiny_model2, {"muon": muon2, "adamw": adamw2}, None,
    )
    assert step == 99
    assert loss == 0.5
    for s_new, s_old in zip(adamw2.state.values(), adamw.state.values()):
        # Post-2026-07-15 explicit-accumulator layout: round-trip
        # all three AdamW per-param buffers (grad + exp_avg +
        # exp_avg_sq).
        assert torch.equal(s_new.grad, s_old.grad)
        assert torch.equal(s_new.exp_avg, s_old.exp_avg)
        assert torch.equal(s_new.exp_avg_sq, s_old.exp_avg_sq)
    for s_new, s_old in zip(muon2.state.values(), muon.state.values()):
        # Round-trip the two Muon per-param buffers (grad +
        # exp_avg).
        assert torch.equal(s_new.grad, s_old.grad)
        assert torch.equal(s_new.exp_avg, s_old.exp_avg)