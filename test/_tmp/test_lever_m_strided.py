"""Lever M + O + P — strided reads + direct u + fused beta/I (June 2026-30).

Lever M (June 2026-30): verifies the strided-read refactor of intra_solve
and wy_fused_transform keeps correctness vs FLA, and removes the 4
.contiguous() calls in __init__.py:310-316 (q_per, k_per, v_per, beta_per).

Lever O (June 2026-30): wy_fused_transform writes u directly to
[T_total, HV, V] strided layout, eliminating the downstream
transpose+contiguous() in __init__.py:405. delta_h reads u via the same
stride pattern, so no consumer change is needed.

Lever P (June 2026-30): intra_solve kernel applies beta row-wise and adds
1.0 on the diagonal — A_kk_fp32 is direct output of `A = I + A_kk * beta`
(lower-tri entries). Saves the wrapper's post-intra `A_kk * beta` (0.40 ms)
and `+ eye` (0.32 ms) at prod.

Setup (mirrors bench/kda_fwd_bench.py inputs):
  * q, k: L2-normalized bf16 (matches production)
  * g: in [-2.5, -0.5] (stable recurrence)
  * beta: sigmoid

Pass criteria (matches docs/verification_thresholds.md):
  * cos ≥ 0.99 at all shapes
  * med_rel < 5% (FLA vs wrapper comparison; wrapper-vs-wrapper is bit-exact)
  * No NaN / Inf
  * Determinism: bit-exact across two runs
  * u comes back in [T, HV, V] shape (Lever O)
  * A_kk_fp32 has correct diagonal = 1.0 (Lever P — implicit via end-to-end
    cos/med_rel vs FLA)

Shapes tested: small, medium, k128, prod.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO))

from src.models.ops.cuda.kda_fwd import chunk_kda_fwd
from src.models.ops._vendored.fla.ops.kda import chunk_kda


SHAPES = {
    "small":  (1,   256,   4,  64,  64),
    "medium": (1,  1024,   4,  64,  64),
    "k128":   (1,  1024,   8, 128, 128),
    "prod":   (1, 16384,  12, 128, 128),
}


def _make_inputs(B, T, H, K, V, device, dtype, seed):
    torch.manual_seed(seed)
    q = torch.randn(B, T, H, K, device=device, dtype=torch.float32)
    k = torch.randn(B, T, H, K, device=device, dtype=torch.float32)
    v = torch.randn(B, T, H, V, device=device, dtype=dtype)
    q = (q / q.norm(dim=-1, keepdim=True)).to(dtype)
    k = (k / k.norm(dim=-1, keepdim=True)).to(dtype)
    g = -torch.rand(B, T, H, K, device=device, dtype=dtype) * 2.0 - 0.5
    beta = torch.randn(B, T, H, device=device, dtype=dtype).sigmoid()
    return q, k, v, g, beta


def test_strided_correctness():
    """All shapes pass vs FLA reference."""
    print(f"\n=== Lever M: strided reads — correctness ===")
    device = "cuda:0"
    dtype = torch.bfloat16
    for name, (B, T, H, K, V) in SHAPES.items():
        scale = 1.0 / math.sqrt(K)
        q, k, v, g, beta = _make_inputs(B, T, H, K, V, device, dtype, seed=7)

        o_wrapper, _ = chunk_kda_fwd(q, k, v, g, beta, scale=scale)
        o_fla, _ = chunk_kda(q=q, k=k, v=v, g=g, beta=beta, scale=scale)
        torch.cuda.synchronize()

        cos = torch.nn.functional.cosine_similarity(
            o_wrapper.flatten(), o_fla.flatten(), dim=0
        ).item()
        diff = (o_wrapper.float() - o_fla.float()).abs()
        max_diff = diff.max().item()
        med_rel = (diff / o_fla.float().abs().clamp(min=1e-6)).median().item()
        finite = torch.isfinite(o_wrapper).all().item()

        status = "PASS" if (cos >= 0.99 and med_rel < 0.05 and finite) else "FAIL"
        print(
            f"  {name:6s}  cos={cos:.6f}  max_diff={max_diff:.2e}  "
            f"med_rel={med_rel:.2e}  finite={finite}  [{status}]"
        )
        assert status == "PASS", f"FAILED at {name}"
        assert cos >= 0.99, f"cos below threshold at {name}"
        assert med_rel < 0.05, f"med_rel above threshold at {name}"
        assert finite, f"non-finite output at {name}"


def test_strided_determinism():
    """Two consecutive runs are bit-exact (Triton deterministic at this shape)."""
    print(f"\n=== Lever M: determinism (prod shape) ===")
    device = "cuda:0"
    dtype = torch.bfloat16
    B, T, H, K, V = SHAPES["prod"]
    scale = 1.0 / math.sqrt(K)
    q, k, v, g, beta = _make_inputs(B, T, H, K, V, device, dtype, seed=7)
    o1, _ = chunk_kda_fwd(q, k, v, g, beta, scale=scale)
    o2, _ = chunk_kda_fwd(q, k, v, g, beta, scale=scale)
    torch.cuda.synchronize()
    diff = (o1.float() - o2.float()).abs().max().item()
    print(f"  max_diff between two runs: {diff:.2e}")
    assert diff == 0.0, "not bit-exact"
    assert torch.isfinite(o1).all().item(), "NaN/Inf in output"


def test_lever_o_direct_u_layout():
    """Lever O: wy_fused_transform writes u directly to [T, HV, V] strided.

    This bypasses the old transpose+contiguous() step in __init__.py:405.
    delta_h reads u via stride HV*V (already handled — see kernel.cu:210).
    """
    print(f"\n=== Lever O: wy_fused_transform direct u write ===")
    import torch
    from src.models.ops.cuda.kda_fwd.triton_wy_transform import wy_fused_transform
    from src.models.ops.cuda.kda_fwd.triton_intra_solve import triton_intra_solve
    from src.models.ops.cuda.kda_fwd.triton_g_cumsum import g_cumsum_fused

    device = "cuda:0"
    dtype = torch.bfloat16
    B, T, H, K, V = SHAPES["prod"]
    BT, BC = 64, 16
    num_chunks = T // BT
    HV = H
    scale = 1.0 / math.sqrt(K)

    torch.manual_seed(7)
    q = torch.randn(B, T, H, K, device=device, dtype=dtype)
    k = torch.randn(B, T, H, K, device=device, dtype=dtype)
    v = torch.randn(B, T, HV, V, device=device, dtype=dtype)
    g = -torch.rand(B, T, HV, K, device=device, dtype=dtype) * 2.0 - 0.5
    beta = torch.randn(B, T, HV, device=device, dtype=dtype).sigmoid()

    q_tok = q.view(T, H, K)
    k_tok = k.view(T, H, K)
    v_tok = v.view(T, HV, V)
    g_tok = g.view(T, HV, K)
    beta_tok = beta.view(T, HV)

    # Run enough of the pipeline to get A_inv.
    _, g_cum_tok = g_cumsum_fused(g_tok, num_chunks, HV, K, BT)
    A_qk, A_kk_fp32 = triton_intra_solve(q_tok, k_tok, g_cum_tok, beta_tok, scale, BT, BC, H=H)
    from src.models.ops.cuda.kda_fwd import _forward_sub
    _forward_sub(A_kk_fp32, BT)

    # Call wy_fused_transform — u must come back in [T, HV, V] shape.
    u, w = wy_fused_transform(A_kk_fp32, v_tok, k_tok, g_cum_tok, beta_tok,
                              T=T, BT=BT, BV=V, BK=K, K=K, V=V, H=H)
    torch.cuda.synchronize()

    assert u.shape == (T, HV, V), f"u shape must be [T, HV, V], got {u.shape}"
    assert u.is_contiguous(), f"u must be contiguous in [T, HV, V] layout"
    assert w.shape == (num_chunks * HV, BT, K), f"w shape must be [N, BT, K], got {w.shape}"
    assert torch.isfinite(u).all().item(), "NaN/Inf in u"
    assert torch.isfinite(w).all().item(), "NaN/Inf in w"
    print(f"  u shape={tuple(u.shape)} contiguous={u.is_contiguous()} finite={torch.isfinite(u).all().item()}")
    print(f"  w shape={tuple(w.shape)}")
    print("  [PASS]")


if __name__ == "__main__":
    test_strided_correctness()
    test_strided_determinism()
    test_lever_o_direct_u_layout()
    print("\nAll Lever M + O + P tests PASSED")