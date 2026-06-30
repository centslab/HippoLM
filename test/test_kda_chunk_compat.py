"""KDA chunk_kda cross-shape regression test.

Smoke-tests `chunk_kda` (vendored FLA) fwd+bwd across various
parameter combinations. Catches:

- Saved-tensor mismatches (would raise `not enough values to unpack`).
- NaN propagation (would show non-zero `nan_count`).
- Missing-flag issues (e.g. `use_qk_l2norm_in_kernel=True` is
  required; missing it produces 80% NaN).
- lower_bound range errors (must be in `[-5, 0)` per
  `chunk.py:424`).

If a case fails, the failure message localizes the issue
(fwd vs bwd, NaN count, exception type, etc.).

This test was promoted from `test/_tmp/test_compat_all.py`
(round-9 KDA optimization, 2026-06-30) — see
`docs/kda_kernel_structure.md` "Round-9 optimization experience"
for the methodology that produced the BK=128 / dhu ns=2 patches
this test guards.
"""
from __future__ import annotations
import sys, math
from pathlib import Path
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


# Test matrix
CASES = [
    # (K, V, H, T, chunk_size, safe_gate, lower_bound, description)
    # Production shape
    (128, 128, 12, 16384, 64, True, -5.0, "PROD K=128 H=12 T=16384 safe_gate=True"),
    # Different K values
    (64, 64, 12, 16384, 64, True, -5.0, "K=64 (BK<128, multi-iter intra)"),
    (256, 256, 8, 4096, 64, True, -5.0, "K=256 (BK=128 still < K)"),
    # Different H
    (128, 128, 8, 16384, 64, True, -5.0, "H=8"),
    (128, 128, 16, 16384, 64, True, -5.0, "H=16"),
    # Different T
    (128, 128, 12, 4096, 64, True, -5.0, "T=4096"),
    (128, 128, 12, 8192, 64, True, -5.0, "T=8192"),
    (128, 128, 12, 32768, 64, True, -5.0, "T=32768"),
    # Different chunk sizes
    (128, 128, 12, 16384, 32, True, -5.0, "chunk_size=32"),
    (128, 128, 12, 16384, 128, True, -5.0, "chunk_size=128"),
    # safe_gate variations
    (128, 128, 12, 16384, 64, False, None, "safe_gate=False (legacy)"),
    (128, 128, 12, 16384, 64, True, -2.0, "lower_bound=-2 (shallower)"),
    (128, 128, 12, 16384, 64, True, -5.0, "lower_bound=-5 (boundary)"),
]


def test_kda_compat():
    """All cases must produce 0 NaN in fwd+bwd grads."""
    failures = []
    for K, V, H, T, chunk_size, safe_gate, lower_bound, desc in CASES:
        if T < chunk_size:
            continue
        nan_info, err = run_chunk_kda_safe(
            K, V, H, T, chunk_size, safe_gate, lower_bound)
        if err:
            failures.append(f"  {desc}: {err[:80]}")
        else:
            nan_count, total = nan_info
            if nan_count > 0:
                failures.append(f"  {desc}: NaN {nan_count}/{total}")
    if failures:
        raise AssertionError(
            "KDA compat failures:\n" + "\n".join(failures)
        )


if __name__ == "__main__":
    # Manual run mode (also useful when pytest is unavailable)
    print(f"{'Case':<60s} {'NaN / total':<20s} {'Status'}")
    print("=" * 95)
    all_pass = True
    for K, V, H, T, chunk_size, safe_gate, lower_bound, desc in CASES:
        if T < chunk_size:
            print(f"{desc:<60s} {'SKIP':<20s} (T<chunk_size)")
            continue
        nan_info, err = run_chunk_kda_safe(
            K, V, H, T, chunk_size, safe_gate, lower_bound)
        if err:
            print(f"{desc:<60s} {'FAIL':<20s} {err[:40]}")
            all_pass = False
        else:
            nan_count, total = nan_info
            status = "PASS" if nan_count == 0 else f"NaN {nan_count}/{total}"
            print(f"{desc:<60s} {f'{nan_count}/{total}':<20s} {status}")
            if nan_count > 0:
                all_pass = False
    print("=" * 95)
    print(f"OVERALL: {'ALL PASS' if all_pass else 'FAILURES'}")
    sys.exit(0 if all_pass else 1)