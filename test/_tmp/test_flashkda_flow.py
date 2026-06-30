"""Exact FlashKDA flow vs FLA — verifies the canonical reference.

If the exact FlashKDA flow in PyTorch matches FLA within bf16 noise (~2%
med_rel), then FlashKDA's reference is correct and any deviation by
FastKDA indicates a bug. If even the exact FlashKDA flow differs from
FLA, then FLA and FlashKDA have a substantive difference beyond
storage precision.

FlashKDA flow (per tests/torch_ref.py:217-243):
  1. L = torch.tril(L, diagonal=-1) * beta_val_fp16       # ROW beta
  2. INV = (I - L)^-1 via Neumann (sum I + L + L^2 + ...) in fp16acc
  3. v_residual = v_chunk - (k_decayed @ state.T)
  4. v_residual_b = v_residual * beta_val_bf16
  5. U = INV @ v_residual_b
  6. delta_s = k_restored.T @ U
  7. h_new = state * exp(g_total) + delta_s (with bf16-cast intermediate)
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


def flashkda_flow(B, T, H, K, V, lb, seed, device, dtype, q_n, k_n, v, g_raw, beta_raw, A_log, dt_bias, scale, state_fp32=True):
    """Replay FlashKDA's reference exactly (line-by-line translation)."""
    a_exp = torch.exp(A_log.float()).view(1, 1, H, 1)
    g_act = lb * torch.sigmoid(a_exp * (g_raw + dt_bias.view(1, 1, H, K)))
    NC = (T + CHUNK - 1) // CHUNK

    state = torch.zeros(B, H, K, V, dtype=torch.float32 if state_fp32 else dtype, device=device)

    # FLA's state dtype is bf16 per its convention; try both.
    offs = torch.arange(CHUNK, device=device)

    for b in range(B):
        for c in range(NC):
            t_start = c * CHUNK
            t_end = min(t_start + CHUNK, T)
            actual_len = t_end - t_start

            for hh in range(H):
                # g_cumsum for this head
                g_chunk = g_act[b, t_start:t_end, hh, :]   # [actual_len, K]
                g_cs_local = torch.cumsum(g_chunk, dim=0)
                g_total = g_cs_local[-1] if actual_len == CHUNK else g_cs_local[-1]
                # Pad g to CHUNK
                if actual_len < CHUNK:
                    g_cs_padded = torch.cat([
                        g_cs_local,
                        (g_cs_local[-1:] + torch.cumsum(torch.zeros(CHUNK - actual_len, K, device=device), dim=0))
                    ], dim=0)
                else:
                    g_cs_padded = g_cs_local
                # fp32_ex2_ftz convention used in FlashKDA — just use torch.exp
                exp_g_cs = torch.exp(g_cs_padded).to(dtype)  # [CHUNK, K]
                exp_neg_g_cs = torch.exp(-g_cs_padded).to(dtype)
                exp_g_total = torch.exp(g_total).to(dtype)  # [K]

                q_chunk = q_n[b, t_start:t_end, hh, :].to(dtype).float()  # [actual_len, K]
                k_chunk = k_n[b, t_start:t_end, hh, :].to(dtype).float()
                if actual_len < CHUNK:
                    q_chunk = torch.cat([q_chunk, torch.zeros(CHUNK - actual_len, K, device=device)], dim=0)
                    k_chunk = torch.cat([k_chunk, torch.zeros(CHUNK - actual_len, K, device=device)], dim=0)

                v_chunk = v[b, t_start:t_end, hh, :].float()  # [actual_len, V]
                if actual_len < CHUNK:
                    v_chunk = torch.cat([v_chunk, torch.zeros(CHUNK - actual_len, V, device=device)], dim=0)

                beta_chunk = beta_raw[b, t_start:t_end, hh].to(torch.float32)  # [actual_len]
                if actual_len < CHUNK:
                    beta_chunk = torch.cat([beta_chunk, torch.zeros(CHUNK - actual_len, device=device)], dim=0)
                beta_sig = torch.sigmoid(beta_chunk).to(dtype)
                beta_val_bf16 = beta_sig.unsqueeze(-1)  # [CHUNK, 1]
                beta_val_fp16 = beta_sig.to(torch.float16).unsqueeze(-1)

                # Compute q_decayed, k_decayed, k_inv, k_restored
                q_decayed = (q_chunk * exp_g_cs.float() * scale).to(dtype).float()
                k_decayed = (k_chunk * exp_g_cs.float()).to(dtype).float()
                k_inv = (k_chunk * exp_neg_g_cs.float()).to(dtype).float()
                k_restored = (k_inv * exp_g_total.float()[None, :]).to(dtype).float()

                # L = k_decayed @ k_inv.T (tril + row beta) — Row 211
                L = (k_decayed @ k_inv.T).to(torch.float32)
                # Line 218: L = tril(L, diagonal=-1) * beta_val_fp16
                tril_mask = offs[:, None] > offs[None, :]
                L = torch.where(tril_mask, L * beta_val_fp16.float(), torch.zeros((), device=device))
                L_fp16 = L.to(torch.float16).float()
                Mqk = (q_decayed @ k_inv.T).to(torch.float32)
                Mqk = torch.where(offs[:, None] >= offs[None, :], Mqk, torch.zeros((), device=device))

                # INV via Neumann — Lines 221-229
                INV = (torch.eye(CHUNK, device=device, dtype=torch.float16) - L_fp16.to(torch.float16)).float()
                I = torch.eye(CHUNK, device=device, dtype=torch.float32)
                L_f16 = L_fp16.to(torch.float16).float()
                L2 = (L_f16 @ L_f16).to(torch.float32)
                INV = INV + (INV.to(torch.float16) @ L2.to(torch.float16)).to(torch.float32)
                L4 = (L2 @ L2).to(torch.float32)
                INV = INV + (INV.to(torch.float16) @ L4.to(torch.float16)).to(torch.float32)
                L8 = (L4 @ L4).to(torch.float32)
                INV = INV + (INV.to(torch.float16) @ L8.to(torch.float16)).to(torch.float32)
                INV = INV.to(dtype).float()  # bf16 cast (Line 229)

                # state slice (the h state at chunk start)
                state_slice = state[b, hh].float()  # [K, V]

                # Line 232: v_chunk -= k_decayed @ state.T
                v_residual = v_chunk - (k_decayed @ state_slice.T)  # [CHUNK, V]
                # Line 233: v_chunk *= beta_val_bf16
                v_residual_b = v_residual * beta_val_bf16

                # Line 235: U = INV @ v_chunk
                U = INV @ v_residual_b

                # Line 239: delta_s = k_restored.T @ U
                delta_s = (k_restored.T @ U).to(torch.float32)

                # Line 241-243: h_new = state * exp(g_total) + delta_s
                g_total_exp = exp_g_total.unsqueeze(-1)  # [K, 1]
                if state_fp32:
                    h_new = (delta_s + state_slice.T * g_total_exp.float()).T  # [K, V]
                    state[b, hh] = h_new  # fp32 stored
                else:
                    h_new_bf16 = (delta_s.to(dtype) + state_slice.to(dtype) * g_total_exp)
                    state[b, hh] = h_new_bf16

                # Mask out padding contributions (so they don't propagate)
                if actual_len < CHUNK:
                    # Don't apply the chunk if it's a partial — wait, FLA processes partials too.
                    # Just store as-is.
                    pass

    return state.to(torch.float32)


def err_stats(a, b, label):
    af = a.float().flatten()
    bf = b.float().flatten()
    diff = (af - bf).abs()
    rel = diff / bf.abs().clamp_min(1e-3)
    valid = rel < 5.0
    med_rel = rel[valid].median().item() if valid.any() else float('nan')
    med_abs = diff.median().item()
    mx = diff.max().item()
    cos = torch.nn.functional.cosine_similarity(af.unsqueeze(0), bf.unsqueeze(0)).item()
    print(f"  {label:<35s} cos={cos:.4f} med_abs={med_abs:.3e} max_abs={mx:.3e} med_rel={med_rel:.3e}")


def main():
    print("Test: does FlashKDA's reference flow (line-by-line) match FLA's chunk_kda?")
    print("If yes: FlashKDA = canonical, and FastKDA's deviation is a bug.")
    print("If no: FLA has its own bug/different formulation.\n")

    for B, T, H, K, V, lb, seed in [
        (1,  256, 4, 128, 128, -5.0, 7),
        (1,  256, 4, 128, 128, -5.0, 13),
    ]:
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

        print(f"\n=== B={B} T={T} H={H} seed={seed} ===")
        st_fp32 = flashkda_flow(B, T, H, K, V, lb, seed, device, dtype, q_n, k_n, v, g_raw, beta_raw, A_log, dt_bias, scale, state_fp32=True)
        st_bf16 = flashkda_flow(B, T, H, K, V, lb, seed, device, dtype, q_n, k_n, v, g_raw, beta_raw, A_log, dt_bias, scale, state_fp32=False)
        err_stats(st_fla, st_fp32, "FlashKDA flow (fp32 state)")
        err_stats(st_fla, st_bf16, "FlashKDA flow (bf16 state)")


if __name__ == "__main__":
    main()
