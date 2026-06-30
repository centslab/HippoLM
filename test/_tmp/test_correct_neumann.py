"""Quick test: replay FlashKDA flow with CORRECT Neumann (doubling) formula.
Compare to FLA. Should match within ~2% med_rel (noise floor)."""
import math
import torch
from src.models.ops._vendored.fla.ops.kda import chunk_kda


CHUNK = 16


def flashkda_flow(B, T, H, K, V, lb, seed, q_n, k_n, v, g_raw, beta_raw, A_log, dt_bias, scale, state_dtype, device, dtype):
    """Replay FlashKDA reference with CORRECT doubling formula for (I-L)^-1."""
    a_exp = torch.exp(A_log.float()).view(1, 1, H, 1)
    g_act = lb * torch.sigmoid(a_exp * (g_raw + dt_bias.view(1, 1, H, K)))
    NC = (T + CHUNK - 1) // CHUNK
    state = torch.zeros(B, H, K, V, dtype=state_dtype, device=device)

    offs = torch.arange(CHUNK, device=device)

    for b in range(B):
        for c in range(NC):
            t_start = c * CHUNK
            t_end = int(min(t_start + CHUNK, T))
            actual_len = t_end - t_start

            for hh in range(H):
                g_chunk = g_act[b, t_start:t_end, hh, :]  # [actual_len, K]
                g_cs_local = torch.cumsum(g_chunk, dim=0)
                if actual_len < CHUNK:
                    g_cs_padded = torch.cat([g_cs_local, torch.zeros(CHUNK - actual_len, K, device=device)], dim=0)
                else:
                    g_cs_padded = g_cs_local
                g_total = g_cs_padded[-1]  # always CHUNK×K, so [-1] is well-defined
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

                # CORRECT doubling Neumann for (I - L)^-1 of STRICTLY lower L
                T_neumann = torch.eye(CHUNK, device=device, dtype=torch.float32)
                Lk = L.to(torch.float32)
                for _k in range(int(math.log2(CHUNK))):
                    T_neumann = (torch.eye(CHUNK, device=device, dtype=torch.float32) + Lk) @ T_neumann
                    Lk = Lk @ Lk
                INV = T_neumann.to(dtype).float()

                state_slice = state[b, hh].float()
                v_residual = v_chunk - (k_decayed @ state_slice.T)
                v_residual_b = v_residual * beta_val_bf16
                U = INV @ v_residual_b
                delta_s = (k_restored.T @ U).to(torch.float32)

                g_total_exp = exp_g_total.unsqueeze(-1)
                if state_dtype == torch.float32:
                    h_new = (delta_s + state_slice.T * g_total_exp.float()).T
                else:
                    h_new = (delta_s.to(dtype) + state_slice.to(dtype) * g_total_exp).T
                state[b, hh] = h_new.to(state_dtype)
    return state.to(torch.float32)


def err_stats(a, b, label):
    af = a.float().flatten()
    bf = b.float().flatten()
    diff = (af - bf).abs()
    rel = diff / bf.abs().clamp_min(1e-3)
    valid = rel < 5.0
    med_rel = rel[valid].median().item() if valid.any() else float('nan')
    cos = torch.nn.functional.cosine_similarity(af.unsqueeze(0), bf.unsqueeze(0)).item()
    print(f"  {label:<40s} cos={cos:.4f} max_abs={diff.max().item():.3e} med_rel={med_rel:.3e}")


def main():
    print("Test: does FlashKDA's reference flow (with CORRECT Neumann doubling) match FLA?\n")
    for B, T, H, K, V, lb, seed in [
        (1,  256, 4, 128, 128, -5.0, 7),
        (1, 1024, 4, 128, 128, -5.0, 7),
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

        print(f"=== B={B} T={T} H={H} seed={seed} ===")
        st_fp32 = flashkda_flow(B, T, H, K, V, lb, seed, q_n, k_n, v, g_raw, beta_raw, A_log, dt_bias, scale, torch.float32, device, dtype)
        st_bf16 = flashkda_flow(B, T, H, K, V, lb, seed, q_n, k_n, v, g_raw, beta_raw, A_log, dt_bias, scale, dtype, device, dtype)
        err_stats(st_fla, st_fp32, "FLA vs FlashKDA-corrected (fp32 state)")
        err_stats(st_fla, st_bf16, "FLA vs FlashKDA-corrected (bf16 state)")


if __name__ == "__main__":
    main()
