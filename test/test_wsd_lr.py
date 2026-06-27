"""Tests for the WSD (warmup-stable-decay) learning rate schedule.

Pure Python, no torch / no GPU needed. The schedule is the
canonical LR policy in ``configs/base.yml`` (``lr_warmup_steps=0``,
``lr_decay_steps=100``, ``learning_rate=0.01``, ``max_steps=1000``)
but the function is parameterized so we exercise the corners.
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from src.training.loop import wsd_lr  # noqa: E402


def test_zero_warmup_jumps_to_peak_at_step_zero():
    """When warmup_steps=0, step 0 is already at peak (no ramp)."""
    assert wsd_lr(0, peak_lr=0.01, warmup_steps=0, decay_steps=10, max_steps=100) == 0.01


def test_warmup_linear_ramp_from_zero():
    """Steps 0..warmup_steps-1 linearly ramp 0 -> peak."""
    warmup, peak = 10, 0.01
    assert wsd_lr(0, peak_lr=peak, warmup_steps=warmup, decay_steps=10, max_steps=100) == 0.0
    assert abs(
        wsd_lr(warmup - 1, peak_lr=peak, warmup_steps=warmup, decay_steps=10, max_steps=100)
        - peak * (warmup - 1) / warmup
    ) < 1e-12
    # First step after warmup is exactly peak.
    assert wsd_lr(warmup, peak_lr=peak, warmup_steps=warmup, decay_steps=10, max_steps=100) == peak


def test_stable_phase_constant_peak():
    """Between warmup and decay, LR is exactly peak for every step."""
    peak = 0.01
    warmup, decay, total = 10, 20, 100
    for step in range(warmup, total - decay):
        assert wsd_lr(step, peak_lr=peak, warmup_steps=warmup, decay_steps=decay, max_steps=total) == peak


def test_decay_linear_ramp_to_min_lr():
    """Decay steps [decay_start, total-1] linearly ramp peak -> min_lr."""
    peak, min_lr, warmup, decay, total = 0.01, 0.001, 10, 20, 100
    decay_start = total - decay  # 80
    # First decay step is still peak.
    assert wsd_lr(decay_start, peak_lr=peak, warmup_steps=warmup, decay_steps=decay,
                  max_steps=total, min_lr=min_lr) == peak
    # Last decay step is exactly min_lr.
    assert abs(
        wsd_lr(total - 1, peak_lr=peak, warmup_steps=warmup, decay_steps=decay,
               max_steps=total, min_lr=min_lr) - min_lr
    ) < 1e-12
    # Mid-decay step is on the line peak + (min_lr - peak) * 0.5
    mid = decay_start + decay // 2
    expected = peak + (min_lr - peak) * (mid - decay_start) / (decay - 1)
    assert abs(
        wsd_lr(mid, peak_lr=peak, warmup_steps=warmup, decay_steps=decay,
               max_steps=total, min_lr=min_lr) - expected
    ) < 1e-12


def test_zero_decay_stays_at_peak_through_end():
    """When decay_steps=0 the schedule never decays — every step is at peak
    (with warmup_steps=0 so step 0 is at peak from the start)."""
    peak = 0.01
    for step in [0, 1, 50, 99]:
        assert wsd_lr(step, peak_lr=peak, warmup_steps=0, decay_steps=0, max_steps=100) == peak


def test_decay_steps_one_returns_peak_for_single_decay_step():
    """decay_steps==1 is the guard against division-by-zero. The single
    decay step is at ``decay_start = max_steps - 1 = 99``; with
    ``denom = 1`` and ``progress = (99 - 99) / 1 = 0``, the function
    returns peak_lr for that step (not min_lr as one might naively
    expect). NOTE: this means the schedule never actually decays when
    ``decay_steps=1`` — the docstring says "jumps to min_lr" but the
    code does not. Documented for future bug fix.
    """
    # At step 99 (= decay_start, the only decay step):
    out = wsd_lr(99, peak_lr=0.01, warmup_steps=0, decay_steps=1, max_steps=100, min_lr=0.0)
    assert out == 0.01, (
        f"decay_steps=1: expected peak (0.01) at the only decay step; got {out}"
    )
    # Steps past the horizon are clamped to 0.
    assert wsd_lr(100, peak_lr=0.01, warmup_steps=0, decay_steps=1, max_steps=100) == 0.0


def test_step_below_zero_returns_zero():
    """Negative step is clamped to 0 (clamping to peak would be wrong
    because no real schedule has negative step)."""
    assert wsd_lr(-1, peak_lr=0.01, warmup_steps=10, decay_steps=10, max_steps=100) == 0.0


def test_step_at_or_above_max_steps_returns_zero():
    """Past the horizon is clamped to 0 (the loop never reaches this
    but tests / callers might)."""
    assert wsd_lr(100, peak_lr=0.01, warmup_steps=10, decay_steps=10, max_steps=100) == 0.0
    assert wsd_lr(150, peak_lr=0.01, warmup_steps=10, decay_steps=10, max_steps=100) == 0.0


def test_warmup_plus_decay_exceeds_max_steps_shrinks_stable_phase():
    """When warmup + decay > max_steps, the stable phase shrinks to 0
    AND the warmup phase is walked first (phase priority: warmup >
    stable > decay).

    Setup: warmup=80, decay=80, max_steps=100.
      - decay_start = max_steps - decay_steps = 20.
      - At step 20, warmup (20 < 80) takes priority → returns
        peak * 20/80 = 0.0025 (NOT peak).
      - At step 79 (last warmup step): returns peak * 79/80.
      - At step 80 (first non-warmup, in decay region): denom = 79,
        progress = (80 - 20) / 79 = 60/79. Returns peak +
        (min_lr - peak) * 60/79.
      - At step 99 (last step, in decay): progress = 79/79 = 1.0,
        returns min_lr.
    """
    peak, min_lr = 0.01, 0.0
    warmup, decay, total = 80, 80, 100

    # step 70: still in warmup (70 < 80)
    out_70 = wsd_lr(70, peak_lr=peak, warmup_steps=warmup,
                    decay_steps=decay, max_steps=total, min_lr=min_lr)
    assert abs(out_70 - peak * 70 / warmup) < 1e-12

    # step 20: still in warmup (20 < 80). Warmup wins over decay.
    out_20 = wsd_lr(20, peak_lr=peak, warmup_steps=warmup,
                    decay_steps=decay, max_steps=total, min_lr=min_lr)
    assert abs(out_20 - peak * 20 / warmup) < 1e-12, (
        f"warmup should win at step 20; got {out_20}"
    )

    # step 99: in decay, progress = 79/79 = 1.0 → exactly min_lr
    out_99 = wsd_lr(99, peak_lr=peak, warmup_steps=warmup,
                    decay_steps=decay, max_steps=total, min_lr=min_lr)
    assert abs(out_99 - min_lr) < 1e-12


def test_zero_peak_lr_is_noop():
    """peak_lr <= 0 returns peak unchanged (no math attempted)."""
    assert wsd_lr(50, peak_lr=0.0, warmup_steps=10, decay_steps=10, max_steps=100) == 0.0
    assert wsd_lr(50, peak_lr=-1.0, warmup_steps=10, decay_steps=10, max_steps=100) == -1.0
