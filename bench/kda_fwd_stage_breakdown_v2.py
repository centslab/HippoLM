"""Per-stage GPU timing for chunk_kda_fwd using the ACTUAL current wrapper.

Times each stage of the fused wrapper (Lever H/M Triton kernels + custom
CUDA delta_h + Triton chunk_o) so we can see where the GPU time goes.

Usage
-----
    python bench/kda_fwd_stage_breakdown_v2.py --shape prod
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from src.models.ops.cuda.kda_fwd import (
    _forward_sub, _delta_h, _chunk_o,
)
from src.models.ops.cuda.kda_fwd.triton_g_cumsum import g_cumsum_fused
from src.models.ops.cuda.kda_fwd.triton_intra_solve import triton_intra_solve
from src.models.ops.cuda.kda_fwd.triton_wy_transform import wy_fused_transform


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


def _event():
    return torch.cuda.Event(enable_timing=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--shape", default="prod", choices=list(SHAPES.keys()))
    p.add_argument("--iters", type=int, default=20)
    args = p.parse_args()

    B, T, H, K, V = SHAPES[args.shape]
    device = torch.device("cuda:0")
    dtype = torch.bfloat16
    scale = 1.0 / math.sqrt(K)
    BT = 64
    BC = 16
    num_chunks = T // BT
    HV = H

    q, k, v, g, beta = _make_inputs(B, T, H, K, V, device, dtype, seed=7)
    q_tok = q.view(T, H, K)
    k_tok = k.view(T, H, K)
    v_tok = v.view(T, HV, V)
    g_tok = g.view(T, HV, K)
    beta_tok = beta.view(T, HV)
    chunk_token_base = torch.arange(num_chunks, dtype=torch.int32, device=device) * BT
    doc_chunk_start = torch.tensor([0], dtype=torch.int32, device=device)
    doc_chunk_count = torch.tensor([num_chunks], dtype=torch.int32, device=device)
    v_new_tok = torch.empty(T, HV, V, dtype=torch.bfloat16, device=device)
    eye = torch.eye(BT, device=device, dtype=torch.float32).unsqueeze(0)

    # ---- warmup ----
    for _ in range(5):
        g_per, g_cum_tok = g_cumsum_fused(g_tok, num_chunks, HV, K, BT)
        A_qk, A_kk_fp32 = triton_intra_solve(q_tok, k_tok, g_cum_tok, beta_tok, scale, BT, BC, H=H)
        _forward_sub(A_kk_fp32, BT)
        u, w = wy_fused_transform(A_kk_fp32, v_tok, k_tok, g_cum_tok, beta_tok,
                                  T=T, BT=BT, BV=V, BK=K, K=K, V=V, H=H)
        w_per_hv = w.view(num_chunks, HV, BT, K).contiguous()
        u_tok = u  # Lever O: u is already [T, HV, V] strided
        h_per_chunk, h_final = _delta_h(
            k_tok, u_tok, w_per_hv, g_cum_tok,
            chunk_token_base, doc_chunk_start, doc_chunk_count,
            v_new_tok,
            num_chunks, 1, H, HV, K, V, 32,
        )
        A_qk_per_hv = A_qk.view(num_chunks, HV, BT, BT).contiguous()
        o_flat = _chunk_o(q_tok, v_new_tok, g_cum_tok, A_qk_per_hv,
                          h_per_chunk, chunk_token_base,
                          num_chunks, scale, H, HV, K, V, 64)
    torch.cuda.synchronize()

    # ---- timed runs ----
    times_gc, times_intra, times_fsub, times_wy, times_reshape, times_delta, times_chunk_o = \
        [], [], [], [], [], [], []

    for _ in range(args.iters):
        torch.cuda.synchronize()
        e0 = _event(); e1 = _event(); e2 = _event(); e3 = _event()
        e4 = _event(); e5 = _event(); e6 = _event()

        e0.record()
        g_per, g_cum_tok = g_cumsum_fused(g_tok, num_chunks, HV, K, BT)

        e1.record()
        A_qk, A_kk_fp32 = triton_intra_solve(q_tok, k_tok, g_cum_tok, beta_tok, scale, BT, BC, H=H)

        e2.record()
        _forward_sub(A_kk_fp32, BT)

        e3.record()
        u, w = wy_fused_transform(A_kk_fp32, v_tok, k_tok, g_cum_tok, beta_tok,
                                  T=T, BT=BT, BV=V, BK=K, K=K, V=V, H=H)

        e4.record()
        w_per_hv = w.view(num_chunks, HV, BT, K).contiguous()
        # Lever O: u is already [T, HV, V] strided — alias is free.
        u_tok = u

        e5.record()
        h_per_chunk, h_final = _delta_h(
            k_tok, u_tok, w_per_hv, g_cum_tok,
            chunk_token_base, doc_chunk_start, doc_chunk_count,
            v_new_tok,
            num_chunks, 1, H, HV, K, V, 32,
        )

        e6.record()
        A_qk_per_hv = A_qk.view(num_chunks, HV, BT, BT).contiguous()
        o_flat = _chunk_o(q_tok, v_new_tok, g_cum_tok, A_qk_per_hv,
                          h_per_chunk, chunk_token_base,
                          num_chunks, scale, H, HV, K, V, 64)
        e6.synchronize()

        times_gc.append(e0.elapsed_time(e1))
        times_intra.append(e1.elapsed_time(e2))
        times_fsub.append(e2.elapsed_time(e3))
        times_wy.append(e3.elapsed_time(e4))
        times_reshape.append(e4.elapsed_time(e5))
        times_delta.append(e5.elapsed_time(e6))
        times_chunk_o.append(e6.elapsed_time(e6))  # placeholder

    def median(xs):
        xs = sorted(xs)
        return xs[len(xs) // 2]

    # Re-time chunk_o cleanly (post-delta_h).
    co_times = []
    for _ in range(args.iters):
        torch.cuda.synchronize()
        e0 = _event(); e1 = _event()
        # Need state from prior run; do another full pipeline but only time last stage
        g_per, g_cum_tok = g_cumsum_fused(g_tok, num_chunks, HV, K, BT)
        A_qk, A_kk_fp32 = triton_intra_solve(q_tok, k_tok, g_cum_tok, beta_tok, scale, BT, BC, H=H)
        _forward_sub(A_kk_fp32, BT)
        u, w = wy_fused_transform(A_kk_fp32, v_tok, k_tok, g_cum_tok, beta_tok,
                                  T=T, BT=BT, BV=V, BK=K, K=K, V=V, H=H)
        w_per_hv = w.view(num_chunks, HV, BT, K).contiguous()
        u_tok = u  # Lever O: u is already [T, HV, V] strided
        h_per_chunk, h_final = _delta_h(
            k_tok, u_tok, w_per_hv, g_cum_tok,
            chunk_token_base, doc_chunk_start, doc_chunk_count,
            v_new_tok,
            num_chunks, 1, H, HV, K, V, 32,
        )
        torch.cuda.synchronize()
        A_qk_per_hv = A_qk.view(num_chunks, HV, BT, BT).contiguous()
        e0.record()
        o_flat = _chunk_o(q_tok, v_new_tok, g_cum_tok, A_qk_per_hv,
                          h_per_chunk, chunk_token_base,
                          num_chunks, scale, H, HV, K, V, 64)
        e1.record(); e1.synchronize()
        co_times.append(e0.elapsed_time(e1))

    total = (median(times_gc) + median(times_intra) + median(times_fsub)
             + median(times_wy) + median(times_reshape) + median(times_delta)
             + median(co_times))
    print(f"\n=== {args.shape}: B={B} T={T} H={H} K={K} V={V} ===")
    print(f"  g_cumsum       (Triton fused):    {median(times_gc):.3f} ms")
    print(f"  intra_solve    (Triton 10-pair):  {median(times_intra):.3f} ms")
    print(f"  fsub+mask+I    (forward_sub):     {median(times_fsub):.3f} ms")
    print(f"  wy_transform   (Triton fused):    {median(times_wy):.3f} ms")
    print(f"  reshape_u      (transpose+contig):{median(times_reshape):.3f} ms")
    print(f"  delta_h        (CUDA WMMA):       {median(times_delta):.3f} ms")
    print(f"  chunk_o        (Triton):          {median(co_times):.3f} ms")
    print(f"  subtotal: {total:.3f} ms")


if __name__ == "__main__":
    main()