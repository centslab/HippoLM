"""Three-way cos similarity: FLA vs naive_ref, FastKDA vs naive_ref, FastKDA vs FLA.

Goal: figure out whether FLA matches naive_ref (the FP64 baseline). If FLA
matches naive but FastKDA doesn't, the bug is FastKDA-side. If FLA doesn't
match naive either, the naive ref is not a ground truth.
"""
from __future__ import annotations
import math
import sys
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from test_fastkda_fwd_match import naive_kda_fp64
from src.models.ops._triton.fast_kda.prepare import kda_prepare_triton
from src.models.ops._triton.fast_kda.recurrence import kda_recurrence_triton
from src.models.ops._vendored.fla.ops.kda import chunk_kda


def _make(B, T, H, K, V, seed=7):
    torch.manual_seed(seed)
    q_raw = torch.randn(B, T, H, K, device='cuda', dtype=torch.float32)
    k_raw = torch.randn(B, T, H, K, device='cuda', dtype=torch.float32)
    v = torch.randn(B, T, H, V, device='cuda', dtype=torch.bfloat16)
    g_raw = -torch.rand(B, T, H, K, device='cuda', dtype=torch.float32) * 2.0 - 0.5
    beta_raw = torch.randn(B, T, H, device='cuda', dtype=torch.float32)
    A_log = torch.rand(H, device='cuda', dtype=torch.float32) * 0.5 + 0.3
    dt_bias = torch.randn(H, K, device='cuda', dtype=torch.float32) * 0.5
    scale = 1.0 / math.sqrt(K)
    q = q_raw.to(torch.bfloat16)
    k = k_raw.to(torch.bfloat16)
    g = g_raw.to(torch.bfloat16)
    beta = beta_raw.sigmoid().to(torch.bfloat16)
    return q_raw, k_raw, v, g_raw, beta_raw, q, k, g, beta, A_log, dt_bias, scale


def stats(name, a, b):
    af = a.float().flatten().unsqueeze(0)
    bf = b.float().flatten().unsqueeze(0)
    cos = torch.nn.functional.cosine_similarity(af, bf).item()
    diff = (a.float() - b.float()).abs()
    rel = (diff / b.float().abs().clamp_min(1e-3)).mean().item()
    print(f'  {name:<30s} cos={cos:.6f} abs_max={diff.max().item():.4e} mean={diff.mean().item():.4e} mean_rel={rel:.4e}')


def main():
    print("=" * 80)
    for B, T, H, K, V in [(1, 256, 12, 128, 128), (1, 1024, 12, 128, 128), (1, 4096, 12, 128, 128)]:
        print(f"\nshape B={B} T={T} H={H} K={K} V={V}")
        q_raw, k_raw, v, g_raw, beta_raw, q, k, g, beta, A_log, dt_bias, scale = _make(B, T, H, K, V)

        o_ref, _ = naive_kda_fp64(
            q_raw, k_raw, v, g_raw, beta_raw, A_log, dt_bias,
            lower_bound=-5.0, use_safe_gate=True,
        )
        o_fla, _ = chunk_kda(
            q=q, k=k, v=v, g=g, beta=beta,
            A_log=A_log, dt_bias=dt_bias,
            use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True,
            safe_gate=True, lower_bound=-5.0,
            output_final_state=True, scale=scale,
        )
        ws = kda_prepare_triton(q, k, g, beta_raw.to(torch.bfloat16), A_log, dt_bias, -5.0, scale)
        o_fast, _, _ = kda_recurrence_triton(
            k_decayed=ws['k_decayed'],
            q_decayed=ws['q_decayed'],
            K_pre=ws['K_pre'],
            g_total=ws['g_total'],
            mqk_eff=ws['Mqk_eff'],
            beta=beta_raw.to(torch.bfloat16),
            v=v,
            initial_state=None,
            output_final_state=True,
        )
        stats('FLA vs ref',       o_fla,  o_ref)
        stats('FastKDA vs ref',   o_fast, o_ref)
        stats('FastKDA vs FLA',   o_fast, o_fla)


if __name__ == "__main__":
    main()
