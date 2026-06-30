"""PyTorch simulation: K_pre = INV @ k_restored vs INV^T @ k_restored.

Hypothesis from the FLA + FlashKDA reference trace:
  delta_s = k_restored^T @ INV @ v_residual_b   (reference)
For h_new += K_pre^T @ v_residual_b to equal this:
  K_pre = (INV^T @ k_restored), NOT (INV @ k_restored).

We replay prepare + recurrence in PyTorch with both formulations, using
EXACT bf16 quantization boundaries matching FastKDA's prepare kernel.
If 'wrong' shows ~6-10% med_rel vs FLA (matching FastKDA's observed
divergence) and 'fix' drops to ~2% (FLA's noise floor), the transpose
bug is confirmed.
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


def replay_per_chunk(B, T, H, K, V, lb, seed, device, dtype, q_n, k_n, v, g_raw, beta_raw, A_log, dt_bias, scale):
    """Replay prepare + recurrence using both K_pre formulations.

    Returns: (st_fla, st_wrong_kpre, st_fixed_kpre)
      Each is fp32 final_state [B, H, K, V].
    """
    a_exp = torch.exp(A_log.float()).view(1, 1, H, 1)
    g_act = lb * torch.sigmoid(a_exp * (g_raw + dt_bias.view(1, 1, H, K)))
    NC = (T + CHUNK - 1) // CHUNK
    h_wrong = torch.zeros(B, H, K, V, dtype=torch.float32, device=device)
    h_fix = torch.zeros(B, H, K, V, dtype=torch.float32, device=device)

    offs = torch.arange(CHUNK, device=device)
    tril_mask = offs[:, None] > offs[None, :]
    diag_mask = offs[:, None] >= offs[None, :]

    for b in range(B):
        for c in range(NC):
            t_start = c * CHUNK
            t_end = min(t_start + CHUNK, T)
            actual_len = t_end - t_start

            # g_cumsum in chunk c
            g_chunk = g_act[b, t_start:t_end]              # [actual_len, H, K]
            g_cs = torch.cumsum(g_chunk, dim=0)
            if actual_len < CHUNK:
                pad_g = g_chunk.new_zeros(CHUNK - actual_len, H, K)
                g_total_c = torch.cat([g_cs, g_cs[-1:] + torch.cumsum(pad_g, dim=0)[1:]], dim=0)[-1]
                exp_g_cs = torch.exp(torch.cat([g_cs, torch.zeros(CHUNK - actual_len, H, K, device=device)], dim=0))
            else:
                g_total_c = g_cs[-1]
                exp_g_cs = torch.exp(g_cs)

            exp_g_total = torch.exp(g_total_c)            # [H, K]
            exp_neg_g_cs = 1.0 / exp_g_cs                  # [CHUNK, H, K]

            q_chunk = q_n[b, t_start:t_end].to(torch.float32)  # [actual_len, H, K]
            k_chunk = k_n[b, t_start:t_end].to(torch.float32)
            if actual_len < CHUNK:
                pad_q = q_chunk.new_zeros(CHUNK - actual_len, H, K)
                pad_k = k_chunk.new_zeros(CHUNK - actual_len, H, K)
                q_chunk_f = torch.cat([q_chunk, pad_q], dim=0)
                k_chunk_f = torch.cat([k_chunk, pad_k], dim=0)
            else:
                q_chunk_f = q_chunk
                k_chunk_f = k_chunk

            beta_chunk = beta_raw[b, t_start:t_end].to(torch.float32)
            if actual_len < CHUNK:
                pad_b = beta_chunk.new_zeros(CHUNK - actual_len, H)
                beta_chunk_f = torch.cat([beta_chunk, pad_b], dim=0)
            else:
                beta_chunk_f = beta_chunk

            # Decays + restoration — bf16 quantized to mirror storage
            q_dec_chunk = (q_chunk_f * exp_g_cs * scale).to(dtype).float()      # [CHUNK, H, K]
            k_dec_chunk = (k_chunk_f * exp_g_cs).to(dtype).float()
            k_inv_chunk = (k_chunk_f * exp_neg_g_cs).to(dtype).float()
            k_restored = k_inv_chunk * exp_g_total[None, :, :]                  # [CHUNK, H, K]
            k_restored = k_restored.to(dtype).float()
            beta_sig = torch.sigmoid(beta_chunk_f)                              # [CHUNK, H]

            v_chunk_h = v[b, t_start:t_end].float()                            # [actual_len, H, V]
            if actual_len < CHUNK:
                pad_v = v_chunk_h.new_zeros(CHUNK - actual_len, H, V)
                v_chunk_pad = torch.cat([v_chunk_h, pad_v], dim=0)
            else:
                v_chunk_pad = v_chunk_h

            for h in range(H):
                qd_h = q_dec_chunk[:, h, :]
                kd_h = k_dec_chunk[:, h, :]
                ki_h = k_inv_chunk[:, h, :]
                kr_h = k_restored[:, h, :]
                bs_h = beta_sig[:, h]
                exp_gt_h = exp_g_total[h, :]
                v_chunk_h_v = v_chunk_pad[:, h, :]

                # L (matches prepare.py: bf16 storage after beta * tril mask)
                L_full = (kd_h @ ki_h.T).to(torch.float32)  # fp32 matmul (matches Triton)
                L = torch.where(tril_mask, L_full * bs_h[:, None], torch.zeros((), device=device, dtype=torch.float32))
                L_bf16 = L.to(dtype).float()

                # INV via Neumann (matches prepare.py)
                INV_bf16 = (torch.eye(CHUNK, device=device, dtype=dtype) - L_bf16.to(dtype)).float()
                I_plus_L = torch.eye(CHUNK, device=device, dtype=torch.float32) + L_bf16
                for _ in range(4):
                    INV_bf16 = (INV_bf16 @ I_plus_L).to(torch.float32)
                INV_f32 = INV_bf16.to(torch.float32)

                # Mqk and Mqk_eff
                Mqk_full = (qd_h @ ki_h.T).to(torch.float32)
                Mqk_bf16 = torch.where(diag_mask, Mqk_full.to(dtype), torch.zeros((), device=device, dtype=dtype)).float()
                # Used by the recurrence only as `Mqk_eff @ v_residual_b` — for
                # this test we only care about h, so we skip Mqk_eff entirely.

                # K_pre two formulations (bf16-quantized as prepare.py does)
                K_pre_wrong = ((INV_f32 @ kr_h).to(dtype)).float()
                K_pre_fix = ((INV_f32.T @ kr_h).to(dtype)).float()

                # v_residual_b = (v - k_dec @ h_prev) * beta_sig  (per V-column independently)
                v_xc = kd_h @ h_wrong[b, h]                  # [CHUNK, V]
                v_res = v_chunk_h_v - v_xc                   # WRONG uses h_wrong
                v_res_b_wrong = v_res * bs_h[:, None]

                v_xc_f = kd_h @ h_fix[b, h]
                v_res_f = v_chunk_h_v - v_xc_f
                v_res_b_fix = v_res_f * bs_h[:, None]

                # h_new = decay * h_prev + K_pre^T @ v_residual_b
                h_wrong_new = h_wrong[b, h] * exp_gt_h[:, None] + K_pre_wrong.T @ v_res_b_wrong
                h_fix_new = h_fix[b, h] * exp_gt_h[:, None] + K_pre_fix.T @ v_res_b_fix

                h_wrong[b, h] = h_wrong_new
                h_fix[b, h] = h_fix_new
    return h_wrong, h_fix


def err_stats(a, b, label):
    af = a.float().flatten()
    bf = b.float().flatten()
    diff = (af - bf).abs()
    mx = diff.max().item()
    med = diff.median().item()
    rel = diff / bf.abs().clamp_min(1e-3)
    valid = rel < 5.0
    med_rel = rel[valid].median().item() if valid.any() else float('nan')
    mean_rel = rel[valid].mean().item() if valid.any() else float('nan')
    print(f"  {label:<25s} med_abs={med:.3e} max_abs={mx:.3e} med_rel={med_rel:.3e} mean_rel={mean_rel:.3e}")


def main():
    print("Bug hypothesis:")
    print("  FastKDA K_pre = INV @ k_restored")
    print("  Correct K_pre = INV^T @ k_restored (matches FLA + FlashKDA reference)")
    print()
    for B, T, H, K, V, seed in [
        (1,  256, 4, 128, 128, 7),
        (1,  256, 4, 128, 128, 13),
        (1, 1024, 4, 128, 128, 7),
    ]:
        print(f"\n=== B={B} T={T} H={H} K={K} V={V} seed={seed} ===")
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

        # FLA reference
        o_fla, st_fla = chunk_kda(
            q=q_n.to(dtype), k=k_n.to(dtype), v=v,
            g=g_raw.to(dtype), beta=beta_raw.sigmoid().to(dtype),
            A_log=A_log, dt_bias=dt_bias,
            use_qk_l2norm_in_kernel=False,
            use_gate_in_kernel=True,
            safe_gate=True, lower_bound=-5.0,
            output_final_state=True, scale=scale,
        )

        # Replay both K_pre formulations
        h_wrong, h_fix = replay_per_chunk(
            B, T, H, K, V, -5.0, seed, device, dtype,
            q_n, k_n, v, g_raw, beta_raw, A_log, dt_bias, scale,
        )

        err_stats(st_fla, h_wrong, "FLA vs WRONG (current)")
        err_stats(st_fla, h_fix, "FLA vs FIX (INV^T)")
        err_stats(h_wrong, h_fix, "WRONG vs FIX")
        err_stats(st_fla, st_fla * 0, "FLA vs ZERO (sanity)")  # should be rel of zero


if __name__ == "__main__":
    main()
