"""Diagnose the Round-9 FastKDA 17% med_rel mystery.

Background (per user's instruction): cos(FLA, Fast)=0.97 but med_rel=17%,
40x FLA's intrinsic noise floor of 0.4% (bf16 vs fp32). Need to understand
*where* the difference comes from.

Hypothesis: if final_state (h) matches tightly between FLA and Fast but o
doesn't, the divergence is in the o computation (Mqk_eff application). If h
also diverges, the recurrence itself has a problem.

Test plan:
1. Compare final_state (h) directly — should be MUCH tighter than o if o's
   amplification is the culprit (h is [B, H, K, V] fp32; o is [B, T, H, V]
   bf16 with a small-signal projection).
2. Compare o vs FLA's o — same as before.
3. Decompose o = o_state + o_attn. o_state = q_decayed @ h_prev (cross-chunk);
   o_attn = Mqk_eff @ v_residual_b (intra-chunk). If o_state matches FLA but
   o_attn doesn't, the issue is in Mqk_eff or v_residual_b.
4. Multi-shape + multi-seed to see if 17% is consistent (algorithmic) or
   variable (noise).
"""
from __future__ import annotations
import math
import sys
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO))

from src.models.ops._triton.fast_kda.prepare import kda_prepare_triton
from src.models.ops._triton.fast_kda.recurrence import kda_recurrence_triton
from src.models.ops._vendored.fla.ops.kda import chunk_kda


def cos(a, b):
    return torch.nn.functional.cosine_similarity(
        a.float().flatten().unsqueeze(0), b.float().flatten().unsqueeze(0)
    ).item()


def err_stats(a, b, label):
    """Report cos, max_abs, mean_abs, mean_rel, med_rel, sign_agree."""
    af = a.float().flatten()
    bf = b.float().flatten()
    diff = (af - bf).abs()
    max_abs = diff.max().item()
    mean_abs = diff.mean().item()
    denom = bf.abs().clamp_min(1e-3)
    rel = diff / denom
    valid = rel < 5.0
    mean_rel = rel[valid].mean().item() if valid.any() else float('nan')
    med_rel = rel[valid].median().item() if valid.any() else float('nan')
    sign_agree = ((af * bf) > 0).float().mean().item()
    c = cos(af, bf)
    mean_a = af.abs().mean().item()
    mean_b = bf.abs().mean().item()
    print(f"  {label:<18s} cos={c:.4f} |abs|_max={max_abs:.2e} |abs|_mean={mean_abs:.2e} "
          f"mean_rel={mean_rel:.2e} med_rel={med_rel:.2e} sign={sign_agree:.4f} "
          f"mag_a={mean_a:.2e} mag_b={mean_b:.2e}")


def run_shape(B, T, H, K, V, lb, seed):
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

    eps = 1e-6
    q_n = q_raw * torch.rsqrt((q_raw * q_raw).sum(-1, keepdim=True) + eps)
    k_n = k_raw * torch.rsqrt((k_raw * k_raw).sum(-1, keepdim=True) + eps)
    q = q_n.to(dtype); k = k_n.to(dtype); g = g_raw.to(dtype)
    beta_pre = beta_raw.sigmoid().to(dtype)

    # Reference: per-token FP64 with L2 norm (canonical GDR; the one true math)
    o_ref, st_ref = per_token_fp64(q_n.to(torch.float32), k_n.to(torch.float32),
                                    v.float(), g_raw, beta_raw, A_log, dt_bias, lb)

    # FLA
    o_fla, st_fla = chunk_kda(
        q=q, k=k, v=v, g=g, beta=beta_pre,
        A_log=A_log, dt_bias=dt_bias,
        use_qk_l2norm_in_kernel=False,
        use_gate_in_kernel=True,
        safe_gate=True, lower_bound=lb,
        output_final_state=True, scale=scale,
    )

    # FastKDA
    ws = kda_prepare_triton(q, k, g, beta_raw.to(dtype), A_log, dt_bias, lb, scale)
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

    print(f"\n=== B={B} T={T} H={H} K={K} V={V} lb={lb} seed={seed} ===")
    err_stats(o_fla, o_fast, "o FLA vs Fast")
    err_stats(st_fla, st_fast, "h FLA vs Fast")
    err_stats(o_ref, o_fla, "o ref vs FLA")
    err_stats(o_ref, o_fast, "o ref vs Fast")
    err_stats(st_ref, st_fla, "h ref vs FLA")
    err_stats(st_ref, st_fast, "h ref vs Fast")

    print(f"  finite: o_fast={torch.isfinite(o_fast).all().item()} "
          f"h_fast={torch.isfinite(st_fast).all().item()} "
          f"o_fla={torch.isfinite(o_fla).all().item()} "
          f"h_fla={torch.isfinite(st_fla).all().item()}")


def per_token_fp64(q, k, v, g_raw, beta_raw, A_log, dt_bias, lb):
    """Canonical per-token GDR in FP64 (slow; T iterations of fp32 matmul)."""
    q = q.double(); k = k.double(); v = v.double()
    g_raw = g_raw.double(); beta_raw = beta_raw.double()
    A_log = A_log.double(); dt_bias = dt_bias.double()
    B, T, H, K = q.shape
    _, _, _, V = v.shape

    a_exp = torch.exp(A_log).view(1, 1, H, 1)
    g_act = lb * torch.sigmoid(a_exp * (g_raw + dt_bias.view(1, 1, H, K)))

    o = torch.zeros(B, T, H, V, dtype=torch.float64, device=q.device)
    h = torch.zeros(B, H, K, V, dtype=torch.float64, device=q.device)
    for t in range(T):
        h_prev = h
        decay_t = torch.exp(g_act[:, t])
        kt = k[:, t]; vt = v[:, t]
        beta_sig_t = torch.sigmoid(beta_raw[:, t])
        v_residual = vt - torch.einsum('bhk,bhkv->bhv', kt, h_prev)
        v_residual_b = v_residual * beta_sig_t.unsqueeze(-1)
        o[:, t] = torch.einsum('bhk,bhkv->bhv', q[:, t], h_prev)
        h = h_prev * decay_t.unsqueeze(-1) + torch.einsum('bhk,bhv->bhkv', kt, v_residual_b)
    return o.to(torch.bfloat16), h.to(torch.float32)


def main():
    cases = [
        # Multiple shapes + seeds to distinguish algorithmic vs noise
        (1,  256,  8, 128, 128, -5.0,  7, "small"),
        (1, 1024,  8, 128, 128, -5.0,  7, "med-1"),
        (1, 1024,  8, 128, 128, -5.0, 13, "med-2-different-seed"),
        (1, 4096,  8, 128, 128, -5.0,  7, "large"),
        (1, 4096,  8, 128, 128, -5.0, 19, "large-different-seed"),
        (2, 1024,  8, 128, 128, -5.0,  7, "B=2"),
        (1, 4096, 12, 128, 128, -5.0,  7, "H=12-prod-shape"),
    ]
    for B, T, H, K, V, lb, seed, desc in cases:
        try:
            run_shape(B, T, H, K, V, lb, seed)
        except Exception as e:
            print(f"\n=== {desc} ERROR: {type(e).__name__}: {e} ===")


if __name__ == "__main__":
    main()
