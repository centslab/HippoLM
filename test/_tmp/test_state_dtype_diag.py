"""Diag: why does fp32-state FlashKDA flow diverge, while bf16-state matches?

Add magnitude tracking and let it run with both dtypes to see what's happening.
"""
from __future__ import annotations
import math
import sys
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO))

from src.models.ops._vendored.fla.ops.kda import chunk_kda


CHUNK = 16


def flashkda_flow_diag(B, T, H, K, V, lb, seed, device, dtype, q_n, k_n, v, g_raw, beta_raw, A_log, dt_bias, scale, state_dtype):
    """Replay FlashKDA's reference, with selectable state dtype."""
    assert state_dtype in (torch.float32, torch.bfloat16), state_dtype
    a_exp = torch.exp(A_log.float()).view(1, 1, H, 1)
    g_act = lb * torch.sigmoid(a_exp * (g_raw + dt_bias.view(1, 1, H, K)))
    NC = (T + CHUNK - 1) // CHUNK

    state = torch.zeros(B, H, K, V, dtype=state_dtype, device=device)

    offs = torch.arange(CHUNK, device=device)
    mag_log = []

    for b in range(B):
        for c in range(NC):
            t_start = c * CHUNK
            t_end = int(min(t_start + CHUNK, T))
            actual_len = t_end - int(t_start)

            for hh in range(H):
                g_chunk = g_act[b, t_start:t_end, hh, :]
                g_cs_local = torch.cumsum(g_chunk, dim=0)
                g_total = g_cs_local[-1]
                if actual_len < CHUNK:
                    g_cs_padded = torch.cat([g_cs_local, torch.zeros(CHUNK - actual_len, K, device=device)], dim=0)
                else:
                    g_cs_padded = g_cs_local
                exp_g_cs = torch.exp(g_cs_padded).to(dtype)
                exp_neg_g_cs = torch.exp(-g_cs_padded).to(dtype)
                exp_g_total = torch.exp(g_total).to(dtype)

                q_chunk = q_n[b, t_start:t_end, hh, :].to(dtype).float()
                k_chunk = k_n[b, t_start:t_end, hh, :].to(dtype).float()
                if actual_len < CHUNK:
                    q_chunk = torch.cat([q_chunk, torch.zeros(CHUNK - actual_len, K, device=device)], dim=0)
                    k_chunk = torch.cat([k_chunk, torch.zeros(CHUNK - actual_len, K, device=device)], dim=0)

                v_chunk = v[b, t_start:t_end, hh, :].float()
                if actual_len < CHUNK:
                    v_chunk = torch.cat([v_chunk, torch.zeros(CHUNK - actual_len, V, device=device)], dim=0)

                beta_chunk = beta_raw[b, t_start:t_end, hh].to(torch.float32)
                if actual_len < CHUNK:
                    beta_chunk = torch.cat([beta_chunk, torch.zeros(CHUNK - actual_len, device=device)], dim=0)
                beta_sig = torch.sigmoid(beta_chunk).to(dtype)
                beta_val_bf16 = beta_sig.unsqueeze(-1)
                beta_val_fp16 = beta_sig.to(torch.float16).unsqueeze(-1)

                q_decayed = (q_chunk * exp_g_cs.float() * scale).to(dtype).float()
                k_decayed = (k_chunk * exp_g_cs.float()).to(dtype).float()
                k_inv = (k_chunk * exp_neg_g_cs.float()).to(dtype).float()
                k_restored = (k_inv * exp_g_total.float()[None, :]).to(dtype).float()

                L = (k_decayed @ k_inv.T).to(torch.float32)
                tril_mask = offs[:, None] > offs[None, :]
                L = torch.where(tril_mask, L * beta_val_fp16.float(),
                                torch.zeros((), device=device))
                L_fp16 = L.to(torch.float16).float()
                Mqk = torch.where(offs[:, None] >= offs[None, :],
                                   (q_decayed @ k_inv.T).to(torch.float32),
                                   torch.zeros((), device=device))

                # CORRECT Neumann via doubling (the kernel's formula is mathematically wrong)
                T = torch.eye(CHUNK, device=device, dtype=torch.float32)
                Lk = L.to(torch.float32)
                for _k in range(int(math.log2(CHUNK))):
                    T = (torch.eye(CHUNK, device=device, dtype=torch.float32) + Lk) @ T
                    Lk = Lk @ Lk
                INV = T.to(dtype).float()

                state_slice = state[b, hh].float()
                v_residual = v_chunk - (k_decayed @ state_slice.T)
                v_residual_b = v_residual * beta_val_bf16
                U = INV @ v_residual_b
                delta_s = (k_restored.T @ U).to(torch.float32)

                g_total_exp = exp_g_total.unsqueeze(-1)
                if state_dtype == torch.float32:
                    h_new = (delta_s + state_slice.T * g_total_exp.float()).T  # [K, V] fp32
                else:
                    h_new = (delta_s.to(dtype) + state_slice.to(dtype) * g_total_exp).T  # bf16

                mag_log.append({
                    'chunk': c, 'head': hh,
                    'state_pre_mean_abs': state_slice.abs().mean().item(),
                    'delta_s_mean_abs': delta_s.abs().mean().item(),
                    'h_new_mean_abs': h_new.float().abs().mean().item(),
                })
                state[b, hh] = h_new.to(state_dtype)

    return state.to(torch.float32), mag_log


def main():
    B, T, H, K, V, lb, seed = 1, 256, 4, 128, 128, -5.0, 7
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

    o_fla, st_fla = chunk_kda(
        q=q_n.to(dtype), k=k_n.to(dtype), v=v,
        g=g_raw.to(dtype), beta=beta_raw.sigmoid().to(dtype),
        A_log=A_log, dt_bias=dt_bias,
        use_qk_l2norm_in_kernel=False,
        use_gate_in_kernel=True,
        safe_gate=True, lower_bound=lb,
        output_final_state=True, scale=scale,
    )

    for state_dtype in [torch.float32, torch.bfloat16]:
        print(f"\n{'='*70}\nstate_dtype={state_dtype}\n{'='*70}")
        st, log = flashkda_flow_diag(B, T, H, K, V, lb, seed, device, dtype, q_n, k_n, v, g_raw, beta_raw, A_log, dt_bias, scale, state_dtype)
        cos = torch.nn.functional.cosine_similarity(st.flatten().unsqueeze(0), st_fla.flatten().unsqueeze(0)).item()
        diff = (st - st_fla).abs()
        print(f"final cos = {cos:.4f}, max_abs = {diff.max().item():.3e}")
        # Print magnitude growth over chunks (head 0)
        print(f"chunk progression (head 0):")
        prev_state = None
        for entry in [e for e in log if e['head']==0]:
            ch = entry['chunk']
            print(f"  c={ch:>3} |state|={entry['state_pre_mean_abs']:.3e}  |delta_s|={entry['delta_s_mean_abs']:.3e}  |h_new|={entry['h_new_mean_abs']:.3e}")
            if ch >= 5:
                break


if __name__ == "__main__":
    main()
