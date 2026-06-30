"""Per-stage GPU timing for chunk_kda_fwd — DEPRECATED.

DEPRECATED (June 2026-30): this file times the OLD Python bmm loop
implementation, not the current Triton-fused wrappers. Kept for
historical reference only.

Use bench/kda_fwd_stage_breakdown_v2.py for current per-stage timing.

Stages timed (OLD loop, not current):
  1. setup  - reshape, cumsum, exp2 of r_intra/c_intra/r_global/c_global
  2. intra  - the 10-pair bmm loop computing A_qk + A_kk
  3. fsub   - forward_sub kernel (in-place A^-1)
  4. wy     - bmm for u and w
  5. delta  - delta_h kernel
  6. chunk_o- chunk_o kernel

Usage
-----
    python bench/kda_fwd_stage_breakdown.py --shape prod
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from src.models.ops.cuda.kda_fwd import _forward_sub, _delta_h, _chunk_o
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


def _make_event():
    return torch.cuda.Event(enable_timing=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--shape", default="prod", choices=list(SHAPES.keys()))
    p.add_argument("--iters", type=int, default=10)
    args = p.parse_args()

    B, T, H, K, V = SHAPES[args.shape]
    device = torch.device("cuda:0")
    dtype = torch.bfloat16
    scale = 1.0 / math.sqrt(K)
    BT = 64
    BC = 16
    NC = BT // BC
    num_chunks = T // BT
    RCP_LN2 = 1.4426950408889634

    q, k, v, g, beta = _make_inputs(B, T, H, K, V, device, dtype, seed=7)

    # ---- Pre-flatten everything we need before timing ----
    q_tok = q.reshape(T, H, K).contiguous()
    k_tok = k.reshape(T, H, K).contiguous()
    v_tok = v.reshape(T, H, V).contiguous()
    g_tok = g.reshape(T, H, K).contiguous()
    beta_tok = beta.reshape(T, H).contiguous()
    g_cum_fp32 = (g_tok.float() * RCP_LN2).view(num_chunks, BT, H, K).cumsum(dim=1)
    g_cum_tok = g_cum_fp32.to(torch.bfloat16).view(T, H, K).contiguous()
    g_last = g_cum_fp32[:, BT - 1, :, :].contiguous()
    sub_idx = torch.arange(BT, device=device)
    sub_id = sub_idx // BC
    sub_mid = sub_id * BC + (BC // 2)
    g_sub_mid = g_cum_fp32[:, sub_mid, :, :]
    chunk_ids = torch.arange(num_chunks, device=device).repeat_interleave(BT)
    g_cum_tok_fp32 = g_cum_fp32.view(T, H, K).contiguous()
    r_intra = (g_cum_tok_fp32 - g_sub_mid.view(T, H, K)).exp2()
    c_intra = (g_sub_mid.view(T, H, K) - g_cum_tok_fp32).exp2()
    r_global = (g_cum_tok_fp32 - g_last[chunk_ids]).exp2()
    c_global = (g_last[chunk_ids] - g_cum_tok_fp32).exp2()

    q_per = q_tok.view(num_chunks, BT, H, K).transpose(1, 2).contiguous()
    k_per = k_tok.view(num_chunks, BT, H, K).transpose(1, 2).contiguous()
    v_per = v_tok.view(num_chunks, BT, H, V).transpose(1, 2).contiguous()
    g_per = g_cum_fp32.view(num_chunks, BT, H, K).transpose(1, 2).contiguous()
    beta_per = beta_tok.view(num_chunks, BT, H).transpose(1, 2).contiguous()
    q_per_fp32 = q_per.float()
    k_per_fp32 = k_per.float()

    pairs = [(s, s) for s in range(NC)] + [(s_i, s_j) for s_i in range(NC) for s_j in range(s_i)]

    # ---- warmup ----
    for _ in range(5):
        # Recreate A_qk/A_kk and do the work to warm up cuBLAS handle cache
        A_qk = torch.zeros(num_chunks * H, BT, BT, device=device, dtype=torch.float32)
        A_kk = torch.zeros(num_chunks * H, BT, BT, device=device, dtype=torch.float32)
        for s_i, s_j in pairs:
            if s_i == s_j:
                anchor_pos = s_i * BC + (BC // 2)
            else:
                anchor_pos = s_i * BC
            g_anchor = g_per[:, :, anchor_pos, :].contiguous()
            g_row = g_per[:, :, s_i*BC:(s_i+1)*BC, :].contiguous()
            g_col = g_per[:, :, s_j*BC:(s_j+1)*BC, :].contiguous()
            r_block = (g_row - g_anchor.unsqueeze(2)).exp2()
            c_block = (g_anchor.unsqueeze(2) - g_col).exp2()
            q_block = q_per_fp32[:, :, s_i*BC:(s_i+1)*BC, :]
            k_row_block = k_per_fp32[:, :, s_i*BC:(s_i+1)*BC, :]
            k_col_block = k_per_fp32[:, :, s_j*BC:(s_j+1)*BC, :]
            r_b = r_block.reshape(num_chunks * H, BC, K)
            c_b = c_block.reshape(num_chunks * H, BC, K)
            q_b = q_block.reshape(num_chunks * H, BC, K)
            k_row_b = k_row_block.reshape(num_chunks * H, BC, K)
            k_col_b = k_col_block.reshape(num_chunks * H, BC, K)
            A_qk_block = torch.bmm(q_b * r_b, (k_col_b * c_b).transpose(-1, -2)) * scale
            A_kk_block = torch.bmm(k_row_b * r_b, (k_col_b * c_b).transpose(-1, -2))
            A_qk[:, s_i*BC:(s_i+1)*BC, s_j*BC:(s_j+1)*BC] = A_qk_block
            A_kk[:, s_i*BC:(s_i+1)*BC, s_j*BC:(s_j+1)*BC] = A_kk_block
        beta_stacked = beta_per.reshape(num_chunks * H, BT)
        A_kk = A_kk * beta_stacked.unsqueeze(-1)
        mask = torch.tril(torch.ones(BT, BT, device=device, dtype=torch.float32), diagonal=-1)
        A_kk = A_kk * mask.unsqueeze(0)
        eye = torch.eye(BT, device=device, dtype=torch.float32).unsqueeze(0)
        A_kk_fp32 = A_kk + eye
        _forward_sub(A_kk_fp32, BT)
        v_per_stacked = v_per.view(num_chunks * H, BT, V)
        k_per_stacked = k_per.view(num_chunks * H, BT, K)
        beta_stacked = beta_per.reshape(num_chunks * H, BT)
        g_per_stacked = g_per.view(num_chunks * H, BT, K)
        beta_v = (v_per_stacked * beta_stacked.unsqueeze(-1)).contiguous()
        u = torch.bmm(A_kk_fp32, beta_v.to(torch.float32)).to(torch.bfloat16)
        g_cum_exp2 = g_per_stacked.exp2()
        k_with_beta_g = (k_per_stacked * beta_stacked.unsqueeze(-1) * g_cum_exp2.to(torch.bfloat16)).contiguous()
        w = torch.bmm(A_kk_fp32, k_with_beta_g.to(torch.float32)).to(torch.bfloat16)
        chunk_token_base = torch.arange(num_chunks, dtype=torch.int32, device=device) * BT
        doc_chunk_start = torch.tensor([0], dtype=torch.int32, device=device)
        doc_chunk_count = torch.tensor([num_chunks], dtype=torch.int32, device=device)
        v_new_tok = torch.empty(T, H, V, dtype=torch.bfloat16, device=device)
        w_per_hv = w.view(num_chunks, H, BT, K).contiguous()
        u_per_chunk = u.view(num_chunks, H, BT, V)
        u_tok = u_per_chunk.transpose(1, 2).contiguous().view(T, H, V)
        h_per_chunk, h_final = _delta_h(
            k_tok, u_tok, w_per_hv, g_cum_tok,
            chunk_token_base, doc_chunk_start, doc_chunk_count,
            v_new_tok,
            num_chunks, 1, H, H, K, V, 32,
        )
        A_qk_per_hv = A_qk.to(torch.bfloat16).view(num_chunks, H, BT, BT).contiguous()
        o_flat = _chunk_o(
            q_tok, v_new_tok, g_cum_tok, A_qk_per_hv,
            h_per_chunk, chunk_token_base,
            num_chunks, scale, H, H, K, V, 64,
        )
    torch.cuda.synchronize()

    # ---- timed runs ----
    times_setup, times_intra, times_fsub, times_wy, times_delta, times_chunk_o = [], [], [], [], [], []

    for _ in range(args.iters):
        torch.cuda.synchronize()
        t0 = _make_event()
        t1 = _make_event()
        t2 = _make_event()
        t3 = _make_event()
        t4 = _make_event()
        t5 = _make_event()

        # STAGE 1: setup (already done before the loop - measure re-do cost)
        t0.record()
        A_qk = torch.zeros(num_chunks * H, BT, BT, device=device, dtype=torch.float32)
        A_kk = torch.zeros(num_chunks * H, BT, BT, device=device, dtype=torch.float32)

        # STAGE 2: intra (10-pair bmm loop)
        t1.record()
        for s_i, s_j in pairs:
            if s_i == s_j:
                anchor_pos = s_i * BC + (BC // 2)
            else:
                anchor_pos = s_i * BC
            g_anchor = g_per[:, :, anchor_pos, :].contiguous()
            g_row = g_per[:, :, s_i*BC:(s_i+1)*BC, :].contiguous()
            g_col = g_per[:, :, s_j*BC:(s_j+1)*BC, :].contiguous()
            r_block = (g_row - g_anchor.unsqueeze(2)).exp2()
            c_block = (g_anchor.unsqueeze(2) - g_col).exp2()
            q_block = q_per_fp32[:, :, s_i*BC:(s_i+1)*BC, :]
            k_row_block = k_per_fp32[:, :, s_i*BC:(s_i+1)*BC, :]
            k_col_block = k_per_fp32[:, :, s_j*BC:(s_j+1)*BC, :]
            r_b = r_block.reshape(num_chunks * H, BC, K)
            c_b = c_block.reshape(num_chunks * H, BC, K)
            q_b = q_block.reshape(num_chunks * H, BC, K)
            k_row_b = k_row_block.reshape(num_chunks * H, BC, K)
            k_col_b = k_col_block.reshape(num_chunks * H, BC, K)
            A_qk_block = torch.bmm(q_b * r_b, (k_col_b * c_b).transpose(-1, -2)) * scale
            A_kk_block = torch.bmm(k_row_b * r_b, (k_col_b * c_b).transpose(-1, -2))
            A_qk[:, s_i*BC:(s_i+1)*BC, s_j*BC:(s_j+1)*BC] = A_qk_block
            A_kk[:, s_i*BC:(s_i+1)*BC, s_j*BC:(s_j+1)*BC] = A_kk_block
        beta_stacked = beta_per.reshape(num_chunks * H, BT)
        A_kk = A_kk * beta_stacked.unsqueeze(-1)
        mask = torch.tril(torch.ones(BT, BT, device=device, dtype=torch.float32), diagonal=-1)
        A_kk = A_kk * mask.unsqueeze(0)
        eye = torch.eye(BT, device=device, dtype=torch.float32).unsqueeze(0)
        A_kk_fp32 = A_kk + eye

        # STAGE 3: forward_sub
        t2.record()
        _forward_sub(A_kk_fp32, BT)

        # STAGE 4: wy transform
        t3.record()
        v_per_stacked = v_per.view(num_chunks * H, BT, V)
        k_per_stacked = k_per.view(num_chunks * H, BT, K)
        beta_stacked = beta_per.reshape(num_chunks * H, BT)
        g_per_stacked = g_per.view(num_chunks * H, BT, K)
        beta_v = (v_per_stacked * beta_stacked.unsqueeze(-1)).contiguous()
        u = torch.bmm(A_kk_fp32, beta_v.to(torch.float32)).to(torch.bfloat16)
        g_cum_exp2 = g_per_stacked.exp2()
        k_with_beta_g = (k_per_stacked * beta_stacked.unsqueeze(-1) * g_cum_exp2.to(torch.bfloat16)).contiguous()
        w = torch.bmm(A_kk_fp32, k_with_beta_g.to(torch.float32)).to(torch.bfloat16)

        # STAGE 5: delta_h
        t4.record()
        chunk_token_base = torch.arange(num_chunks, dtype=torch.int32, device=device) * BT
        doc_chunk_start = torch.tensor([0], dtype=torch.int32, device=device)
        doc_chunk_count = torch.tensor([num_chunks], dtype=torch.int32, device=device)
        v_new_tok = torch.empty(T, H, V, dtype=torch.bfloat16, device=device)
        w_per_hv = w.view(num_chunks, H, BT, K).contiguous()
        u_per_chunk = u.view(num_chunks, H, BT, V)
        u_tok = u_per_chunk.transpose(1, 2).contiguous().view(T, H, V)
        h_per_chunk, h_final = _delta_h(
            k_tok, u_tok, w_per_hv, g_cum_tok,
            chunk_token_base, doc_chunk_start, doc_chunk_count,
            v_new_tok,
            num_chunks, 1, H, H, K, V, 32,
        )

        # STAGE 6: chunk_o
        t5.record()
        A_qk_per_hv = A_qk.to(torch.bfloat16).view(num_chunks, H, BT, BT).contiguous()
        o_flat = _chunk_o(
            q_tok, v_new_tok, g_cum_tok, A_qk_per_hv,
            h_per_chunk, chunk_token_base,
            num_chunks, scale, H, H, K, V, 64,
        )
        t5.synchronize()

        times_setup.append(t0.elapsed_time(t1))
        times_intra.append(t1.elapsed_time(t2))
        times_fsub.append(t2.elapsed_time(t3))
        times_wy.append(t3.elapsed_time(t4))
        times_delta.append(t4.elapsed_time(t5))
        # chunk_o time = full minus up to t5

    def median(xs):
        xs = sorted(xs)
        return xs[len(xs) // 2]

    total = median(times_setup) + median(times_intra) + median(times_fsub) + median(times_wy) + median(times_delta)
    print(f"\n=== {args.shape}: B={B} T={T} H={H} K={K} V={V} ===")
    print(f"  setup   (init A_qk/A_kk zeros):   {median(times_setup):.3f} ms")
    print(f"  intra   (10-pair bmm loop):       {median(times_intra):.3f} ms")
    print(f"  fsub    (forward_sub kernel):     {median(times_fsub):.3f} ms")
    print(f"  wy      (u, w bmm):               {median(times_wy):.3f} ms")
    print(f"  delta   (delta_h kernel):         {median(times_delta):.3f} ms")
    print(f"  (chunk_o time not measured here — see full bench)")
    print(f"  subtotal: {total:.3f} ms (excludes chunk_o)")
    print(f"\nFull CUDA call: see bench/kda_fwd_bench.py --shape {args.shape}")
    print(f"FLA reference:  see bench/kda_fwd_bench.py --shape {args.shape}")


if __name__ == "__main__":
    main()
