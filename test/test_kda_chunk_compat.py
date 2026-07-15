"""KDA chunk_kda cross-shape regression test.

Smoke-tests ``chunk_kda`` (vendored FLA) fwd+bwd across various
parameter combinations. Catches:

- Saved-tensor mismatches (would raise ``not enough values to unpack``).
- NaN propagation (would show non-zero ``nan_count``).
- Missing-flag issues (e.g. ``use_qk_l2norm_in_kernel=True`` is
  required; missing it produces 80% NaN).
- lower_bound range errors (must be in ``[-5, 0)`` per
  ``chunk.py:424``).

Each case is parametrized as its own test, marked
``@pytest.mark.slow``. By default pytest **skips** slow tests
(see ``conftest.py`` at repo root) so the full sweep stays
sub-second on this file; opt back in with::

    pytest -m slow                    # run only slow
    pytest -m "not slow"              # explicit (matches default)
    pytest -m "slow or skip"          # mix slow + cuda-skipped

Why each remaining case is here (the others were dropped on
2026-07-15 to cut redundant prod-shape JIT compiles):

- **prod** (K=128, H=12, T=16384, chunk=64, safe=True, lower=-5):
  baseline shape used in production.
- **k64**: K < BK=128 forces multi-iter intra kernel (different
  Triton codegen than prod).
- **h8**: head-count variation; confirms the kernel handles
  H not equal to 12.
- **chunk32 / chunk128**: chunk_size constexpr variation; guards
  against kernel launch assumptions hardcoded to BT=64.
- **safe_gate=False**: legacy path (no M=16 TC accel); tests
  the kernel branch without safe_gate.
- **lower_bound=-2**: safe_gate=True with a shallower clamp; tests
  the lower-bound guard logic.

Dropped cases (rationale):
- T=4096/8192/32768: no unique coverage vs T=16384 (the
  kernel is block-level; T only affects loop count).
- K=256 (T=4096, H=8): redundant with k64 for "K > BK multi-iter".
- H=16: redundant with h8 (same shape variation).
- lower_bound=-5 (final case): duplicate of prod (same params).
"""
from __future__ import annotations
import sys, math
from pathlib import Path
import pytest
import torch

# Allow running as a script from anywhere
_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from src.models.ops._vendored.fla.ops.kda import chunk_kda

device = torch.device('cuda:0')
dtype = torch.bfloat16


def run_chunk_kda_safe(K, V, H, T, chunk_size, safe_gate, lower_bound, seed=7):
    """Run chunk_kda fwd+bwd; return (nan_count, total) or error string."""
    torch.manual_seed(seed)
    q = torch.randn(1, T, H, K, device=device, dtype=dtype)
    k = torch.randn(1, T, H, K, device=device, dtype=dtype)
    v = torch.randn(1, T, H, V, device=device, dtype=dtype)
    g = -torch.rand(1, T, H, K, device=device, dtype=dtype) * 2.0 - 0.5
    beta = torch.randn(1, T, H, device=device, dtype=dtype).sigmoid()
    do = torch.randn(1, T, H, V, device=device, dtype=dtype)
    scale = 1.0 / math.sqrt(K)
    # TPKDA always provides A_log (shape [HV]) + dt_bias (shape [HV*K])
    # when use_gate_in_kernel=True.
    A_log = torch.zeros(H, device=device, dtype=dtype)
    dt_bias = torch.zeros(H * K, device=device, dtype=dtype)

    common = dict(
        scale=scale,
        chunk_size=chunk_size,
        safe_gate=safe_gate,
        lower_bound=lower_bound,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=safe_gate,
        A_log=A_log,
        dt_bias=dt_bias,
    )

    try:
        o, _ = chunk_kda(q=q, k=k, v=v, g=g, beta=beta, **common)
    except Exception as e:
        return None, f"fwd failed: {type(e).__name__}: {e}"

    # bwd
    q2 = q.detach().requires_grad_(True)
    k2 = k.detach().requires_grad_(True)
    v2 = v.detach().requires_grad_(True)
    g2 = g.detach().requires_grad_(True)
    beta2 = beta.detach().requires_grad_(True)
    try:
        o, _ = chunk_kda(q=q2, k=k2, v=v2, g=g2, beta=beta2, **common)
        o.backward(do)
    except Exception as e:
        return None, f"bwd failed: {type(e).__name__}: {e}"

    nan_count = sum(torch.isnan(t.grad).sum().item() if t.grad is not None else 0
                    for t in [q2, k2, v2, g2, beta2])
    total = sum(t.numel() for t in [q2, k2, v2, g2, beta2])
    return (nan_count, total), None


# Slow cases: each triggers its own Triton kernel compile at prod shape
# (T=16384 + H/K/chunk/safe_gate/lower_bound are constexpr). Parametrize
# keeps each case independent so a single failure pinpoints the param.
_CASES = [
    pytest.param(128, 128, 12, 16384, 64, True,  -5.0,
                 id="prod"),
    pytest.param( 64,  64, 12, 16384, 64, True,  -5.0,
                 id="k64"),
    pytest.param(128, 128,  8, 16384, 64, True,  -5.0,
                 id="h8"),
    pytest.param(128, 128, 12, 16384, 32, True,  -5.0,
                 id="chunk32"),
    pytest.param(128, 128, 12, 16384, 128, True, -5.0,
                 id="chunk128"),
    pytest.param(128, 128, 12, 16384, 64, False, None,
                 id="safe_gate_false"),
    pytest.param(128, 128, 12, 16384, 64, True,  -2.0,
                 id="lower_bound_minus_2"),
]


@pytest.mark.slow
@pytest.mark.parametrize("K, V, H, T, chunk_size, safe_gate, lower_bound",
                         _CASES)
def test_kda_compat(K, V, H, T, chunk_size, safe_gate, lower_bound):
    """One parametrized case: must produce 0 NaN in fwd+bwd grads."""
    if T < chunk_size:
        pytest.skip(f"T={T} < chunk_size={chunk_size}")
    nan_info, err = run_chunk_kda_safe(
        K, V, H, T, chunk_size, safe_gate, lower_bound,
    )
    if err is not None:
        pytest.fail(err[:200])
    nan_count, total = nan_info
    assert nan_count == 0, (
        f"KDA produced {nan_count}/{total} NaN grads "
        f"(K={K} V={V} H={H} T={T} chunk={chunk_size} "
        f"safe_gate={safe_gate} lower_bound={lower_bound})"
    )


if __name__ == "__main__":
    # Manual run mode (also useful when pytest is unavailable).
    # Runs ALL cases regardless of the @slow marker — useful when
    # iterating on the KDA kernel and you don't want to remember
    # ``pytest -m slow``.
    print(f"{'Case':<25s} {'NaN / total':<20s} {'Status'}")
    print("=" * 60)
    all_pass = True
    for case in _CASES:
        # Each entry in _CASES is a pytest.param; extract args + id.
        K, V, H, T, chunk_size, safe_gate, lower_bound = case.values
        desc = case.id
        if T < chunk_size:
            print(f"{desc:<25s} {'SKIP':<20s} (T<chunk_size)")
            continue
        nan_info, err = run_chunk_kda_safe(
            K, V, H, T, chunk_size, safe_gate, lower_bound)
        if err:
            print(f"{desc:<25s} {'FAIL':<20s} {err[:40]}")
            all_pass = False
        else:
            nan_count, total = nan_info
            status = "PASS" if nan_count == 0 else f"NaN {nan_count}/{total}"
            print(f"{desc:<25s} {f'{nan_count}/{total}':<20s} {status}")
            if nan_count > 0:
                all_pass = False
    print("=" * 60)
    print(f"OVERALL: {'ALL PASS' if all_pass else 'FAILURES'}")
    sys.exit(0 if all_pass else 1)