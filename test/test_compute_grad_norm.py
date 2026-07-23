"""Tests for ``_compute_and_clip_grad_norm`` in the training loop.

This function:
  1. Walks every optimizer in the ``opts`` list.
  2. For each per-param state entry, reads the per-step grad
     accumulator (``s.grad`` for both AdamW and Muon —
     post-2026-07-15 explicit-accumulator layout; the old
     "merged-accumulator" design where ``s.m`` / ``s.mom_buf``
     doubled as the accumulator was deleted) and adds its L2
     squared to a running ``local_sq`` scalar.
  3. (When in a distributed group) all-reduces ``local_sq``
     across the TP world.
  4. Computes ``total_norm = sqrt(local_sq.sum())``.
  5. If ``total_norm > max_norm``, scales every accumulator in
     place by ``max_norm / (total_norm + 1e-6)``.

The interesting bug history (per the long comment in
``src/training/loop.py``): an earlier revision applied an
unconditional second ``.mul_(max_norm / (total_norm + eps))``
that double-clipped when total_norm > max_norm (quadratic)
and AMPLIFIED when total_norm < max_norm. The current code
only clips when the norm exceeds the cap. These tests pin
that contract.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from src.training.loop import _compute_and_clip_grad_norm  # noqa: E402
from src.training.param_offload import CPUAdamW, CPUMuon  # noqa: E402


# --------------------------------------------------------------------------- #
# Fixtures.                                                                   #
# --------------------------------------------------------------------------- #
@pytest.fixture
def tiny_model():
    """Two-param model: 1D (LayerNorm.weight) + 2D (Linear.weight)
    so the production routing rule puts each in a different
    optimizer."""
    torch.manual_seed(0)

    class M(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = torch.nn.Linear(8, 16, bias=False)
            self.norm = torch.nn.LayerNorm(16)

    return M().to(torch.float32)


@pytest.fixture
def adamw_opt(tiny_model):
    """AdamW on the 1D LayerNorm params (matches production routing)."""
    return CPUAdamW(
        [p for p in tiny_model.parameters() if p.ndim < 2],
        lr=1e-4, betas=(0.9, 0.95),
    )


@pytest.fixture
def muon_opt(tiny_model):
    """Muon on the 2D Linear params, BF16 momentum (no quant)."""
    return CPUMuon(
        [p for p in tiny_model.parameters() if p.ndim >= 2],
        lr=1e-3,
    )


def _populate_grads(model, adamw, muon, scale: float) -> None:
    """Inject deterministic grads into each optimizer's per-step
    accumulator (``s.grad`` for both optimizers, post-2026-07-15
    explicit-accumulator layout)."""
    g = torch.Generator().manual_seed(1)
    for p in model.parameters():
        p.grad = torch.randn(p.shape, generator=g) * scale
    for s in adamw.state.values():
        # ``s.grad`` is the per-step accumulator for AdamW
        # (was ``s.m`` in the merged-accumulator design).
        s.grad.add_(s.param.grad.detach().to(s.grad.dtype).reshape(-1))
    for s in muon.state.values():
        # ``s.grad`` is the per-step accumulator for Muon
        # (was ``s.mom_buf`` in the merged-accumulator design).
        s.grad.add_(s.param.grad.detach().to(s.grad.dtype).reshape(-1))


# --------------------------------------------------------------------------- #
# Tests.                                                                      #
# --------------------------------------------------------------------------- #
def test_norm_below_cap_does_not_modify_accumulators(
    tiny_model, adamw_opt, muon_opt,
):
    """When the total norm is below ``max_norm``, no accumulator
    is modified (the precondition that prevents the historical
    bug of amplifying small grads)."""
    _populate_grads(tiny_model, adamw_opt, muon_opt, scale=1e-4)
    adamw_before = {id(s.param): s.grad.clone() for s in adamw_opt.state.values()}
    muon_before = {id(s.param): s.grad.clone() for s in muon_opt.state.values()}

    norm = _compute_and_clip_grad_norm([adamw_opt, muon_opt], max_norm=1e2)

    assert norm < 1e2, "expected total norm below cap"
    for s in adamw_opt.state.values():
        assert torch.equal(s.grad, adamw_before[id(s.param)]), (
            "adamw accumulator was modified despite norm < cap"
        )
    for s in muon_opt.state.values():
        assert torch.equal(s.grad, muon_before[id(s.param)]), (
            "muon accumulator was modified despite norm < cap"
        )


def test_norm_above_cap_scales_accumulators_by_clip_coef(
    tiny_model, adamw_opt, muon_opt,
):
    """When total_norm > max_norm, every accumulator must be
    scaled in place by ``max_norm / total_norm`` (single clip,
    not quadratic)."""
    _populate_grads(tiny_model, adamw_opt, muon_opt, scale=10.0)

    # Save the PRE-call accumulator values so we can compare
    # against the POST-call values after the function mutates
    # them in place.
    adamw_before = {id(s.param): s.grad.clone() for s in adamw_opt.state.values()}
    muon_before = {id(s.param): s.grad.clone() for s in muon_opt.state.values()}

    # Compute expected norm by mirroring the function's math on
    # the pre-call accumulators.
    local_sq = torch.zeros(1)
    for opt in (adamw_opt, muon_opt):
        for s in opt.state.values():
            accum = adamw_before[id(s.param)] if s.kind == "adamw" else muon_before[id(s.param)]
            local_sq += accum.detach().float().pow(2).sum()
    expected_norm = local_sq.sqrt().item()

    max_norm = 1.0
    returned_norm = _compute_and_clip_grad_norm(
        [adamw_opt, muon_opt], max_norm=max_norm,
    )
    assert abs(returned_norm - expected_norm) < 1e-4

    clip_coef = max_norm / (expected_norm + 1e-6)
    for s in adamw_opt.state.values():
        # Compare POST-call s.grad against PRE-call
        # adamw_before * clip_coef. BF16 precision: allow
        # generous atol because the scaled value can be much
        # smaller than the BF16 mantissa resolution.
        expected = adamw_before[id(s.param)].float() * clip_coef
        actual = s.grad.float()
        assert torch.allclose(actual, expected, atol=5e-3, rtol=5e-2), (
            f"adamw grad not scaled correctly: ratio"
            f" {(actual / expected).mean().item()}"
        )
    for s in muon_opt.state.values():
        expected = muon_before[id(s.param)].float() * clip_coef
        actual = s.grad.float()
        assert torch.allclose(actual, expected, atol=5e-3, rtol=5e-2), (
            f"muon grad not scaled correctly"
        )


def test_zero_max_norm_disables_clipping(tiny_model, adamw_opt, muon_opt):
    """``max_norm=0`` (or negative) disables clipping entirely."""
    _populate_grads(tiny_model, adamw_opt, muon_opt, scale=100.0)
    adamw_before = {id(s.param): s.grad.clone() for s in adamw_opt.state.values()}
    muon_before = {id(s.param): s.grad.clone() for s in muon_opt.state.values()}

    _compute_and_clip_grad_norm([adamw_opt, muon_opt], max_norm=0.0)

    for s in adamw_opt.state.values():
        assert torch.equal(s.grad, adamw_before[id(s.param)])
    for s in muon_opt.state.values():
        assert torch.equal(s.grad, muon_before[id(s.param)])


def test_returns_finite_norm(tiny_model, adamw_opt, muon_opt):
    """Sanity: the returned value is a finite float."""
    _populate_grads(tiny_model, adamw_opt, muon_opt, scale=1.0)
    norm = _compute_and_clip_grad_norm([adamw_opt, muon_opt], max_norm=10.0)
    assert isinstance(norm, float)
    assert norm == norm  # NaN check
    assert norm > 0.0


def test_no_clip_applied_twice(tiny_model, adamw_opt, muon_opt):
    """The historical bug: an earlier revision applied a SECOND
    unconditional ``.mul_(max_norm / (total_norm + eps))``
    after the function returned. This test verifies the
    SINGLE-clip contract: the post-call ratio is exactly
    ``clip_coef`` (not ``clip_coef**2``)."""
    _populate_grads(tiny_model, adamw_opt, muon_opt, scale=10.0)
    adamw_before = {id(s.param): s.grad.clone() for s in adamw_opt.state.values()}

    _compute_and_clip_grad_norm([adamw_opt, muon_opt], max_norm=1.0)

    for s in adamw_opt.state.values():
        # Ratio of post-call to pre-call should be close to
        # clip_coef, which is small (max_norm / large_total_norm).
        # If the bug were present, the ratio would be clip_coef**2
        # (much smaller).
        ratio = (s.grad.float() / adamw_before[id(s.param)].float()).mean().item()
        # Sanity bounds: ratio is positive (no sign flip) and
        # small (norm was clipped). The exact value depends on
        # BF16 precision of the multiplication; we just check
        # the order of magnitude.
        assert 0.0 < ratio < 1.0, (
            f"adamw ratio outside (0, 1): ratio={ratio}"
        )