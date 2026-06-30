"""Compare FastKDA fwd (prepare + recurrence) against an FP64 naive reference.

Approach: compute a per-token reference in fp64 (no chunking, no
approximations), then compare FLA chunk_kda and FastKDA fwd against it.
This separates FastKDA's intrinsic error from FLA's intrinsic error
(both are bf16, both use Neumann-series-like approximations of
different kinds).

Important: this test only runs the *forward* path. The backward path
is tested in `test/_tmp/test_fastkda_bwd_grad_match.py` (added later).

Why a tmp test: the FastKDA fwd has a known design tension with the
FLA KDA layer (sigmoid fusion + gate activation scale) that we want to
characterize empirically before committing to a design. Promote to
test/test_fastkda_chunk_compat.py once we know the exact agreement.
"""
from __future__ import annotations
import math
import sys
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO))  # allow both `python test/_tmp/...` and import from src.*

from src.models.ops._triton.fast_kda.prepare import kda_prepare_triton
from src.models.ops._triton.fast_kda.recurrence import kda_recurrence_triton
from src.models.ops._vendored.fla.ops.kda import chunk_kda


def naive_kda_fp64(q, k, v, g, beta, A_log, dt_bias, lower_bound, use_safe_gate):
    """Per-token recurrence in FP64. Reference ground truth.

    Per-token:
        h_t = exp(g_t) * h_{t-1} + k_t outer v_t
        o_t = q_t @ h_t
    where g_t is the *activated* gate per token, in NATURAL log space
    (so exp(g_t) is the per-step decay multiplier).

    Returns (o, final_state) in fp32 (cast down at the end so the
    comparison with bf16 outputs is fair).
    """
    B, T, H, K = q.shape
    _, _, _, V = v.shape
    q = q.double()
    k = k.double()
    v = v.double()
    g = g.double()
    beta = beta.double()
    A_log = A_log.double()
    dt_bias = dt_bias.double()

    if use_safe_gate:
        # FLA convention: g_t = lower_bound * sigmoid(exp(A_log) * (g + dt_bias)) / ln(2)
        #   so exp(g_t) = 2^(lower_bound * sigmoid(...))
        a_exp = torch.exp(A_log).view(1, 1, H, 1)
        g_act = lower_bound * torch.sigmoid(a_exp * (g + dt_bias.view(1, 1, H, K))) / math.log(2)
    else:
        # Standard: g_t = -exp(A_log) * softplus(g + dt_bias)
        a_exp = torch.exp(A_log).view(1, 1, H, 1)
        g_act = -a_exp * torch.nn.functional.softplus(g + dt_bias.view(1, 1, H, K))

    o = torch.zeros(B, T, H, V, dtype=torch.float64, device=q.device)
    h = torch.zeros(B, H, K, V, dtype=torch.float64, device=q.device)
    for t in range(T):
        decay = torch.exp(g_act[:, t])  # [B, H, K]
        # k[:, t] -> [B, H, K] (K-dim), v[:, t] -> [B, H, V]
        kt = k[:, t]  # [B, H, K]
        vt = v[:, t]  # [B, H, V]
        # h[b, h, k_d, v_d] += kt[b, h, k_d] * vt[b, h, v_d]
        h = h * decay.unsqueeze(-1) + torch.einsum('bhk,bhv->bhkv', kt, vt)
        o[:, t] = torch.einsum('bhk,bhkv->bhv', q[:, t], h)
    return o.to(torch.bfloat16), h.to(torch.float32)


def run_one(B, T, H, K, V, use_safe_gate, lower_bound, seed=7):
    """Return (o_ref_bf16, o_fla_bf16, o_fast_bf16, st_ref, st_fla, st_fast) and max diffs."""
    torch.manual_seed(seed)
    device, dtype = 'cuda', torch.bfloat16
    q_raw = torch.randn(B, T, H, K, device=device, dtype=torch.float32)
    k_raw = torch.randn(B, T, H, K, device=device, dtype=torch.float32)
    v = torch.randn(B, T, H, V, device=device, dtype=dtype)
    g_raw = -torch.rand(B, T, H, K, device=device, dtype=torch.float32) * 2.0 - 0.5
    beta_raw = torch.randn(B, T, H, device=device, dtype=torch.float32)
    A_log = torch.rand(H, device=device, dtype=torch.float32) * 0.5 + 0.3
    dt_bias = torch.randn(H, K, device=device, dtype=torch.float32) * 0.5
    scale = 1.0 / math.sqrt(K)

    q = q_raw.to(dtype)
    k = k_raw.to(dtype)
    g = g_raw.to(dtype)
    # beta is pre-sigmoided (FLA KDA layer's contract)
    beta = beta_raw.sigmoid().to(dtype)

    # Reference
    o_ref, st_ref = naive_kda_fp64(
        q_raw, k_raw, v, g_raw, beta_raw, A_log, dt_bias,
        lower_bound=lower_bound if use_safe_gate else None,
        use_safe_gate=use_safe_gate,
    )

    # FLA
    o_fla, st_fla = chunk_kda(
        q=q, k=k, v=v, g=g, beta=beta,
        A_log=A_log, dt_bias=dt_bias,
        use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True,
        safe_gate=use_safe_gate, lower_bound=lower_bound if use_safe_gate else None,
        output_final_state=True, scale=scale,
    )

    # FastKDA. NOTE: prepare kernel applies sigmoid to beta internally —
    # we pass raw (un-sigmoided) beta here. If we pass pre-sigmoided beta,
    # the kernel will sigmoid it again, producing wrong output.
    ws = kda_prepare_triton(q, k, g, beta_raw.to(dtype), A_log, dt_bias,
                            lower_bound if use_safe_gate else 0.0, scale)
    # Round-9: recurrence takes the (i) k_decayed, (ii) K_pre (was
    # k_restored), (iii) Mqk_eff (was Mqk), (iv) raw beta. The recurrence
    # internally applies sigmoid(beta) for v_residual_b.
    o_fast, _, st_fast = kda_recurrence_triton(
        k_decayed=ws['k_decayed'],
        q_decayed=ws['q_decayed'],
        K_pre=ws['K_pre'],
        g_total=ws['g_total'],
        mqk_eff=ws['Mqk_eff'],
        beta=beta_raw.to(dtype),
        v=v,
        initial_state=None,
        output_final_state=True,
    )

    return {
        'o': (o_ref, o_fla, o_fast),
        'st': (st_ref, st_fla, st_fast),
    }


SHAPES = [
    # (B, T, H, K, V, safe_gate, lower_bound, desc)
    (1, 256, 12, 128, 128, True, -5.0, "T=256 safe_gate"),
    (1, 1024, 12, 128, 128, True, -5.0, "T=1024 safe_gate"),
    (1, 4096, 12, 128, 128, True, -5.0, "T=4096 safe_gate"),
    (1, 4096, 12, 128, 128, False, None, "T=4096 NO safe_gate (uses -exp(A)*softplus path)"),
]


def main():
    print(f"{'shape':<50s} {'abs max':>10s} {'cos':>10s} {'mean |r|':>10s} {'med |r|':>10s} {'nan/inf':>10s}")
    print("=" * 110)
    for shape in SHAPES:
        B, T, H, K, V, sg, lb, desc = shape
        try:
            r = run_one(B, T, H, K, V, sg, lb)
            o_ref, o_fla, o_fast = r['o']
            of = o_fast.float()
            oF = o_fla.float()
            diff = of - oF
            abs_max = diff.abs().max().item()
            cos = torch.nn.functional.cosine_similarity(
                of.flatten().unsqueeze(0), oF.flatten().unsqueeze(0)
            ).item()
            # per-element rel diff, masked where ref is small
            mask = oF.abs() > 1e-3
            if mask.any():
                rel = (diff.abs() / oF.abs().clamp_min(1e-6))[mask]
                mean_rel = rel.mean().item()
                med_rel = rel.median().item()
            else:
                mean_rel = med_rel = float('nan')
            nan_inf = (torch.isnan(of).any() | torch.isinf(of).any()).item()
            print(f"{desc:<50s} {abs_max:>10.4e} {cos:>10.6f} {mean_rel:>10.4e} {med_rel:>10.4e} {str(nan_inf):>10s}")
        except Exception as e:
            print(f"{desc:<50s}  ERROR: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
