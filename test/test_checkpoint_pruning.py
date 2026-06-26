"""Tests for :func:`prune_old_checkpoints` and the
``keep_last_n`` argument on :func:`save_checkpoint`.

The training loop runs with ``checkpoint_interval: 1`` at the
canonical yml, so a naive "save every step" would leave
``max_steps`` .pt files on disk. The ``keep_last_n`` knob prunes
older files after each save so disk usage stays bounded.

Sort key is the step number parsed from the filename
(``checkpoint_step_{N}.pt``), NOT mtime — wall-clock skew across
machines could otherwise reorder saves incorrectly. Files whose
names don't match the pattern are left untouched.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from src.training.checkpoint import prune_old_checkpoints, save_checkpoint
from src.training.param_offload import (
    CPUAdamW,
    CPUMuon,
    _int8_muon_accumulate,
)


# --------------------------------------------------------------------------- #
# Fixtures.                                                                   #
# --------------------------------------------------------------------------- #
@pytest.fixture
def tiny_model():
    torch.manual_seed(0)
    return nn.Sequential(
        nn.Linear(8, 16), nn.LayerNorm(16), nn.Linear(16, 4),
    ).to(torch.float16)


def _surviving_steps(d: Path) -> list[int]:
    out = []
    for p in d.glob("checkpoint_step_*.pt"):
        try:
            out.append(int(p.stem.rsplit("_", 1)[-1]))
        except ValueError:
            continue
    return sorted(out)


def _populate(model, adamw, muon, seed: int) -> None:
    g = torch.Generator().manual_seed(seed)
    for p in model.parameters():
        p.grad = torch.randn(p.shape, generator=g, dtype=p.dtype) * 1e-3
    for s in adamw.state.values():
        s.m.add_(s.param.grad.detach().to("cpu").reshape(-1))
    for s in muon.state.values():
        if s.mom_scale is not None:
            _int8_muon_accumulate(
                s,
                s.param.grad.detach().to(torch.bfloat16).reshape(-1).to("cpu"),
            )
        else:
            s.mom_buf.add_(s.param.grad.detach().to(s.mom_buf.dtype).reshape(-1))


def _make_saves(model, adamw, muon, d: Path, n: int, keep_last_n):
    """Save checkpoints for steps 1..n with the given retention."""
    for s in range(1, n + 1):
        _populate(model, adamw, muon, seed=s)
        save_checkpoint(
            model, {"muon": muon, "adamw": adamw}, None,
            step=s, loss=1.0 / s, checkpoint_dir=d, keep_last_n=keep_last_n,
        )


# --------------------------------------------------------------------------- #
# Retention behavior.                                                         #
# --------------------------------------------------------------------------- #
def test_keep_last_n_3_over_10_saves_keeps_newest_3(tiny_model, tmp_path):
    adamw = CPUAdamW([p for p in tiny_model.parameters() if p.ndim < 2], lr=1e-4, betas=(0.9, 0.95))
    muon = CPUMuon([p for p in tiny_model.parameters() if p.ndim >= 2], lr=1e-3)
    _make_saves(tiny_model, adamw, muon, tmp_path, n=10, keep_last_n=3)
    assert _surviving_steps(tmp_path) == [8, 9, 10]


def test_keep_last_n_0_deletes_all_including_just_saved(tiny_model, tmp_path):
    adamw = CPUAdamW([p for p in tiny_model.parameters() if p.ndim < 2], lr=1e-4, betas=(0.9, 0.95))
    muon = CPUMuon([p for p in tiny_model.parameters() if p.ndim >= 2], lr=1e-3)
    _make_saves(tiny_model, adamw, muon, tmp_path, n=3, keep_last_n=0)
    assert _surviving_steps(tmp_path) == []


def test_keep_last_n_1_keeps_only_just_saved(tiny_model, tmp_path):
    """The practical "discard everything except the latest" knob."""
    adamw = CPUAdamW([p for p in tiny_model.parameters() if p.ndim < 2], lr=1e-4, betas=(0.9, 0.95))
    muon = CPUMuon([p for p in tiny_model.parameters() if p.ndim >= 2], lr=1e-3)
    _make_saves(tiny_model, adamw, muon, tmp_path, n=5, keep_last_n=1)
    assert _surviving_steps(tmp_path) == [5]


def test_keep_last_n_none_keeps_all_for_backcompat(tiny_model, tmp_path):
    adamw = CPUAdamW([p for p in tiny_model.parameters() if p.ndim < 2], lr=1e-4, betas=(0.9, 0.95))
    muon = CPUMuon([p for p in tiny_model.parameters() if p.ndim >= 2], lr=1e-3)
    _make_saves(tiny_model, adamw, muon, tmp_path, n=3, keep_last_n=None)
    assert _surviving_steps(tmp_path) == [1, 2, 3]


def test_keep_last_n_negative_is_noop(tiny_model, tmp_path):
    """Defensive: a misconfigured negative value should not delete."""
    adamw = CPUAdamW([p for p in tiny_model.parameters() if p.ndim < 2], lr=1e-4, betas=(0.9, 0.95))
    muon = CPUMuon([p for p in tiny_model.parameters() if p.ndim >= 2], lr=1e-3)
    _make_saves(tiny_model, adamw, muon, tmp_path, n=3, keep_last_n=-1)
    assert _surviving_steps(tmp_path) == [1, 2, 3]


# --------------------------------------------------------------------------- #
# Sibling files.                                                              #
# --------------------------------------------------------------------------- #
def test_non_checkpoint_files_are_not_pruned(tiny_model, tmp_path):
    """A user-managed ``best.pt`` in the same dir must survive pruning."""
    adamw = CPUAdamW([p for p in tiny_model.parameters() if p.ndim < 2], lr=1e-4, betas=(0.9, 0.95))
    muon = CPUMuon([p for p in tiny_model.parameters() if p.ndim >= 2], lr=1e-3)
    _make_saves(tiny_model, adamw, muon, tmp_path, n=4, keep_last_n=2)

    sidecar = tmp_path / "best.pt"
    sidecar.write_bytes(b"placeholder")

    _populate(tiny_model, adamw, muon, seed=99)
    save_checkpoint(
        tiny_model, {"muon": muon, "adamw": adamw}, None,
        step=99, loss=0.01, checkpoint_dir=tmp_path, keep_last_n=1,
    )
    assert sidecar.is_file()
    assert _surviving_steps(tmp_path) == [99]


def test_malformed_step_suffix_is_skipped_not_crashed(tiny_model, tmp_path):
    """Files matching the glob but with a non-integer step suffix
    (e.g. ``checkpoint_step_foo.pt``) must be left alone — they're
    not ours to delete."""
    bad = tmp_path / "checkpoint_step_foo.pt"
    bad.write_bytes(b"junk")

    adamw = CPUAdamW([p for p in tiny_model.parameters() if p.ndim < 2], lr=1e-4, betas=(0.9, 0.95))
    muon = CPUMuon([p for p in tiny_model.parameters() if p.ndim >= 2], lr=1e-3)
    _populate(tiny_model, adamw, muon, seed=1)
    save_checkpoint(
        tiny_model, {"muon": muon, "adamw": adamw}, None,
        step=1, loss=1.0, checkpoint_dir=tmp_path, keep_last_n=1,
    )
    assert bad.is_file()
    assert _surviving_steps(tmp_path) == [1]


# --------------------------------------------------------------------------- #
# Direct API.                                                                 #
# --------------------------------------------------------------------------- #
def test_prune_old_checkpoints_returns_deletion_count(tiny_model, tmp_path):
    adamw = CPUAdamW([p for p in tiny_model.parameters() if p.ndim < 2], lr=1e-4, betas=(0.9, 0.95))
    muon = CPUMuon([p for p in tiny_model.parameters() if p.ndim >= 2], lr=1e-3)
    # First: save 5 without pruning (default keep_last_n=None keeps all)
    _make_saves(tiny_model, adamw, muon, tmp_path, n=5, keep_last_n=None)
    assert len(_surviving_steps(tmp_path)) == 5

    removed = prune_old_checkpoints(tmp_path, keep_last_n=2)
    assert removed == 3
    assert _surviving_steps(tmp_path) == [4, 5]


def test_prune_old_checkpoints_returns_zero_when_already_pruned(tiny_model, tmp_path):
    adamw = CPUAdamW([p for p in tiny_model.parameters() if p.ndim < 2], lr=1e-4, betas=(0.9, 0.95))
    muon = CPUMuon([p for p in tiny_model.parameters() if p.ndim >= 2], lr=1e-3)
    _make_saves(tiny_model, adamw, muon, tmp_path, n=2, keep_last_n=None)
    removed = prune_old_checkpoints(tmp_path, keep_last_n=5)
    assert removed == 0
    assert _surviving_steps(tmp_path) == [1, 2]


# --------------------------------------------------------------------------- #
# Surviving checkpoint is loadable.                                           #
# --------------------------------------------------------------------------- #
def test_surviving_checkpoint_is_byte_loadable(tiny_model, tmp_path):
    """After pruning, the kept .pt must still be a complete, loadable
    checkpoint (no half-written state from a race)."""
    adamw = CPUAdamW([p for p in tiny_model.parameters() if p.ndim < 2], lr=1e-4, betas=(0.9, 0.95))
    muon = CPUMuon([p for p in tiny_model.parameters() if p.ndim >= 2], lr=1e-3)
    _make_saves(tiny_model, adamw, muon, tmp_path, n=5, keep_last_n=1)

    survivor = sorted(tmp_path.glob("checkpoint_step_*.pt"))[-1]
    payload = torch.load(survivor, map_location="cpu", weights_only=False)
    assert payload["step"] == 5
    assert abs(payload["loss"] - 0.2) < 1e-6
    assert len(payload["optimizer_state_dict"]["muon"]["state"]) == 2
    assert len(payload["optimizer_state_dict"]["adamw"]["state"]) == 4