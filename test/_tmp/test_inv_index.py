"""Verify: does FlashKDA's L use row-scaled or column-scaled beta?

From torch_ref.py Line 215-218:
  beta_chunk [CHUNK]
  beta_val_fp16 = beta_activated.to(torch.float16).unsqueeze(-1)  # [CHUNK, 1]
  L = torch.tril(L, diagonal=-1) * beta_val_fp16

beta_val_fp16.shape = [CHUNK, 1]. Broadcasting with L [CHUNK, CHUNK]:
  L[i, j] *= beta[i]   ← ROW scaling

From FastKDA's prepare.py Line 189-191:
  beta_sig = tl.sigmoid(beta).to(tl.float16)     # [BLOCK_M]
  L = tl.where(tril_mask, L * beta_sig[None, :], 0.0).to(tl.bfloat16)
  # beta_sig[None, :] = [1, BLOCK_M], broadcasts as L[i, j] *= beta[j]  ← COLUMN scaling

Test all 4 combinations of (L row/column beta) × (K_pre = INV vs INV^T @ k_restored)
against FLA's reference final_state to find which is correct.
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


def replay(B, T, H, K, V, lb, seed, l_beta_axis, k_pre_axis, device, dtype, q_n, k_n, v, g_raw, beta_raw, A_log, dt_bias, scale):
    """Replay prepare + recurrence in PyTorch.

    l_beta_axis: 0 = row (FlashKDA), 1 = column (FastKDA current)
    k_pre_axis: 0 = K_pre = INV @ k_restored (FastKDA current)
                1 = K_pre = INV^T @ k_restored (suspected fix)
    Returns final_state [B, H, K, V] fp32.
    """
    a_exp = torch.exp(A_log.float()).view(1, 1, H, 1)
    g_act = lb * torch.sigmoid(a_exp * (g_raw + dt_bias.view(1, 1, H, K)))
    NC = (T + CHUNK - 1) // CHUNK
    h = torch.zeros(B, H, K, V, dtype=torch.float32, device=device)

    offs = torch.arange(CHUNK, device=device)
    tril_mask = offs[:, None] > offs[None, :]

    for b in range(B):
        for c in range(NC):
            t_start = c * CHUNK
            t_end = min(t_start + CHUNK, T)
            actual_len = t_end - t_start

            g_chunk = g_act[b, t_start:t_end]
            g_cs = torch.cumsum(g_chunk, dim=0)
            if actual_len < CHUNK:
                pad_g = g_chunk.new_zeros(CHUNK - actual_len, H, K)
                g_total_c = torch.cat([g_cs, torch.zeros(CHUNK - actual_len, H, K, device=device)], dim=0)
                g_total = g_total_c[-1]  # last row
                g_cs = g_total_c
            else:
                g_total = g_cs[-1]
            exp_g_cs = torch.exp(g_cs)
            exp_g_total = torch.exp(g_total)
            exp_neg_g_cs = 1.0 / exp_g_cs

            q_chunk = q_n[b, t_start:t_end].to(torch.float32)
            k_chunk = k_n[b, t_start:t_end].to(torch.float32)
            if actual_len < CHUNK:
                pad_q = q_chunk.new_zeros(CHUNK - actual_len, H, K)
                pad_k = k_chunk.new_zeros(CHUNK - actual_len, H, K)
                q_chunk_f = torch.cat([q_chunk, pad_q], dim=0)
                k_chunk_f = torch.cat([k_chunk, pad_k], dim=0)
            else:
                q_chunk_f, k_chunk_f = q_chunk, k_chunk

            beta_chunk = beta_raw[b, t_start:t_end].to(torch.float32)
            if actual_len < CHUNK:
                pad_b = beta_chunk.new_zeros(CHUNK - actual_len, H)
                beta_chunk_f = torch.cat([beta_chunk, pad_b], dim=0)
            else:
                beta_chunk_f = beta_chunk
            beta_sig = torch.sigmoid(beta_chunk_f)

            q_dec = (q_chunk_f * exp_g_cs * scale).to(dtype).float()
            k_dec = (k_chunk_f * exp_g_cs).to(dtype).float()
            k_inv = (k_chunk_f * exp_neg_g_cs).to(dtype).float()
            k_restored = (k_inv * exp_g_total[None, :, :]).to(dtype).float()

            v_chunk_h = v[b, t_start:t_end].float()
            if actual_len < CHUNK:
                pad_v = v_chunk_h.new_zeros(CHUNK - actual_len, H, V)
                v_chunk_pad = torch.cat([v_chunk_h, pad_v], dim=0)
            else:
                v_chunk_pad = v_chunk_h

            for hh in range(H):
                qd_h = q_dec[:, hh, :]
                kd_h = k_dec[:, hh, :]
                ki_h = k_inv[:, hh, :]
                kr_h = k_restored[:, hh, :]
                bs_h = beta_sig[:, hh]
                exp_gt_h = exp_g_total[hh, :]
                v_chunk_h_v = v_chunk_pad[:, hh, :]

                L_full = (kd_h @ ki_h.T).to(torch.float32)

                # Apply beta with selectable axis
                if l_beta_axis == 0:
                    # ROW scaling: L[i, j] *= beta[i]  (FlashKDA)
                    beta_factor = bs_h[:, None]
                else:
                    # COLUMN scaling: L[i, j] *= beta[j]  (FastKDA current)
                    beta_factor = bs_h[None, :]
                L = torch.where(tril_mask, L_full * beta_factor,
                                torch.zeros((), device=device, dtype=torch.float32))
                L_bf16 = L.to(dtype).float()

                INV_bf16 = (torch.eye(CHUNK, device=device, dtype=dtype) - L_bf16.to(dtype)).float()
                I_plus_L = torch.eye(CHUNK, device=device, dtype=torch.float32) + L_bf16
                for _ in range(4):
                    INV_bf16 = (INV_bf16 @ I_plus_L).to(torch.float32)
                INV_f32 = INV_bf16.to(torch.float32)

                # K_pre with selectable transpose
                if k_pre_axis == 0:
                    # INV @ k_restored (current FastKDA)
                    K_pre = ((INV_f32 @ kr_h).to(dtype)).float()
                else:
                    # INV^T @ k_restored (proposed)
                    K_pre = ((INV_f32.T @ kr_h).to(dtype)).float()

                # v_residual & h update
                v_xc = kd_h @ h[b, hh]  # [CHUNK, V]
                v_res = v_chunk_h_v - v_xc
                v_res_b = v_res * bs_h[:, None]
                h_new = h[b, hh] * exp_gt_h[:, None] + K_pre.T @ v_res_b
                h[b, hh] = h_new
    return h


def err_stats(a, b, label):
    af = a.float().flatten()
    bf = b.float().flatten()
    diff = (af - bf).abs()
    rel = diff / bf.abs().clamp_min(1e-3)
    valid = rel < 5.0
    med_rel = rel[valid].median().item() if valid.any() else float('nan')
    med_abs = diff.median().item()
    mx = diff.max().item()
    print(f"  {label:<35s} med_abs={med_abs:.3e} max_abs={mx:.3e} med_rel={med_rel:.3e}")


def main():
    print("Testing 4 combinations of (L beta row/col) × (K_pre = INV/INV^T @ k)")
    print("All should match FLA within ~2% med_rel if the correct combo is chosen.\n")

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

    print(f"=== shape B={B} T={T} H={H} seed={seed} ===\n")
    for l_axis_name, l_beta_axis in [("row", 0), ("col", 1)]:
        for k_axis_name, k_pre_axis in [("INV@kr", 0), ("INV.T@kr", 1)]:
            label = f"L={l_axis_name}-beta, K_pre={k_axis_name}"
            try:
                st = replay(B, T, H, K, V, lb, seed, l_beta_axis, k_pre_axis,
                            device, dtype, q_n, k_n, v, g_raw, beta_raw, A_log, dt_bias, scale)
                err_stats(st_fla, st, label)
            except Exception as e:
                print(f"  {label:<35s} ERROR: {e}")


if __name__ == "__main__":
    main()
