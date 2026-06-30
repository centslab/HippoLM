"""Benchmark the fused Triton KDA recurrence kernel.

Times the recurrence step alone and end-to-end (prepare + recurrence)
against FLA's chunk_kda. The recurrence kernel is the second of two
FlashKDA-style kernels; together they replace chunk_kda entirely.

Usage:
    python bench/fast_kda_recurrence_bench.py
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from src.models.ops._triton.fast_kda.prepare import kda_prepare_triton
from src.models.ops._triton.fast_kda.recurrence import kda_recurrence_triton
from src.models.ops._vendored.fla.ops.kda import chunk_kda


CHUNK = 16


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


class CudaTimer:
    def __init__(self):
        self._start = None
        self._end = None

    def __enter__(self):
        torch.cuda.synchronize()
        self._start = torch.cuda.Event(enable_timing=True)
        self._end = torch.cuda.Event(enable_timing=True)
        self._start.record()
        return self

    def __exit__(self, *exc):
        self._end.record()
        torch.cuda.synchronize()
        self.ms = self._start.elapsed_time(self._end)


def _bench(fn, iters):
    """Median ms per call."""
    times = []
    for _ in range(5):
        fn()  # warmup
    torch.cuda.synchronize()
    for _ in range(iters):
        with CudaTimer() as t:
            fn()
        times.append(t.ms)
    times.sort()
    return times[len(times) // 2]


SHAPES = {
    "small":    (1,   256,   4, 128, 128),
    "medium":   (1,  1024,   8, 128, 128),
    "k128":     (1,  2048,   8, 128, 128),
    "prod":     (1, 16384,  12, 128, 128),
    "long":     (1, 32768,  12, 128, 128),
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--shape", default="all", choices=list(SHAPES.keys()) + ["all"])
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args()

    device = torch.device(f"cuda:{args.device}")
    dtype = torch.bfloat16
    torch.cuda.set_device(args.device)

    shapes = SHAPES if args.shape == "all" else {args.shape: SHAPES[args.shape]}

    print(f"\n{'shape':<8} {'T':>6} {'H':>3} {'NC':>5} "
          f"{'prep ms':>10} {'rec ms':>10} {'fwd ms':>10} "
          f"{'FLA ms':>10} {'fwd/FLA':>10} "
          f"{'fwd tok/s':>12}")
    print("-" * 100)

    rows = []
    for name, (B, T, H, K, V) in shapes.items():
        NC = (T + CHUNK - 1) // CHUNK
        scale = 1.0 / math.sqrt(K)
        q, k, v, g, beta = _make_inputs(B, T, H, K, V, device, dtype, args.seed)
        A_log = torch.rand(H, device=device, dtype=torch.float32) * 0.5 + 0.3
        dt_bias = torch.randn(H, K, device=device, dtype=torch.float32) * 0.5

        # 1. Prepare alone
        prep_ms = _bench(
            lambda: kda_prepare_triton(q, k, g, beta, A_log, dt_bias, -5.0, scale),
            args.iters,
        )

        # 2. Recurrence alone (subtract prepare cost)
        def _rec():
            ws = kda_prepare_triton(q, k, g, beta, A_log, dt_bias, -5.0, scale)
            kda_recurrence_triton(
                k_decayed=ws["k_decayed"],
                q_decayed=ws["q_decayed"],
                K_pre=ws["K_pre"],
                g_total=ws["g_total"],
                mqk_eff=ws["Mqk_eff"],
                beta=beta,
                v=v,
            )
        rec_total_ms = _bench(_rec, args.iters)
        rec_only_ms = max(rec_total_ms - prep_ms, 0.01)

        # 3. End-to-end (prep + rec)
        fwd_ms = rec_total_ms

        # 4. FLA reference
        fla_ms = _bench(
            lambda: chunk_kda(q=q, k=k, v=v, g=g, beta=beta, scale=scale),
            args.iters,
        )

        toks = B * T
        ratio = fla_ms / fwd_ms
        fwd_toks = toks / (fwd_ms / 1000)
        print(f"{name:<8} {T:>6} {H:>3} {NC:>5} "
              f"{prep_ms:>10.3f} {rec_only_ms:>10.3f} {fwd_ms:>10.3f} "
              f"{fla_ms:>10.3f} {ratio:>9.2f}x "
              f"{fwd_toks:>12.0f}")
        rows.append((name, T, prep_ms, rec_only_ms, fwd_ms, fla_ms, ratio))

    # Theoretical analysis for prod shape
    print("\n=== Theoretical bandwidth analysis (prod shape: T=16384, H=12) ===")
    B, T, H, K, V = SHAPES["prod"]
    NC = (T + CHUNK - 1) // CHUNK
    dtype_bytes = 2  # bf16
    fp32_bytes = 4

    # =================== Prepare kernel ===================
    print("\n[Prepare] Inputs + workspace I/O:")
    in_bytes_p = (T * H * K) * dtype_bytes * 2  # q, k
    in_bytes_p += T * H * K * dtype_bytes  # g
    in_bytes_p += T * H * dtype_bytes  # beta
    in_bytes_p += H * fp32_bytes + H * K * fp32_bytes  # A_log, dt_bias

    out_bytes_p = (NC * H * CHUNK * K * dtype_bytes * 2  # qd, kr
                   + NC * H * K * fp32_bytes  # gt
                   + NC * H * CHUNK * CHUNK * dtype_bytes)  # Mqk
    total_bytes_p = in_bytes_p + out_bytes_p
    flops_p = NC * H * CHUNK * CHUNK * K * 2 * 2
    print(f"  inputs:   {in_bytes_p/1024/1024:.1f} MB")
    print(f"  outputs:  {out_bytes_p/1024/1024:.1f} MB")
    print(f"  total:    {total_bytes_p/1024/1024:.1f} MB")
    print(f"  bmm FLOPs: {flops_p/1e9:.1f} GFLOPs")
    print(f"  bw limit: {total_bytes_p/448e9*1000:.3f} ms")
    print(f"  tc limit: {flops_p/419e12*1000:.3f} ms (419 TFLOPs bf16)")
    print(f"  → roofline: {max(total_bytes_p/448e9, flops_p/419e12)*1000:.3f} ms")

    # =================== Recurrence kernel ===================
    print("\n[Recurrence] Inputs + state I/O:")
    in_bytes_r = (NC * H * CHUNK * K * dtype_bytes * 2  # qd, kr
                  + NC * H * K * fp32_bytes  # gt
                  + NC * H * CHUNK * CHUNK * dtype_bytes  # Mqk
                  + T * H * V * dtype_bytes  # v
                  + NC * H * K * V * dtype_bytes)  # h_inter (read+written each chunk)
    out_bytes_r = (T * H * V * dtype_bytes  # o
                   + NC * H * K * V * dtype_bytes  # h_inter
                   + H * K * V * fp32_bytes)  # final_state
    total_bytes_r = in_bytes_r + out_bytes_r
    flops_r_per_chunk = (2 * CHUNK * CHUNK * V + 2 * CHUNK * K * V + 2 * K * CHUNK * V)
    flops_r = NC * H * flops_r_per_chunk
    print(f"  reads:    {in_bytes_r/1024/1024:.1f} MB")
    print(f"  writes:   {out_bytes_r/1024/1024:.1f} MB")
    print(f"  total:    {total_bytes_r/1024/1024:.1f} MB")
    print(f"  FLOPs:    {flops_r/1e9:.1f} GFLOPs")
    print(f"  bw limit: {total_bytes_r/448e9*1000:.3f} ms")
    print(f"  tc limit: {flops_r/419e12*1000:.3f} ms (419 TFLOPs bf16)")
    print(f"  AI:       {flops_r/total_bytes_r:.1f} FLOPs/B")
    print(f"  ridge:    419e12/448e9 = {419e12/448e9:.0f} FLOPs/B")
    ridge = 419e12 / 448e9
    if flops_r / total_bytes_r < ridge:
        print(f"  → BANDWIDTH-BOUND (AI {flops_r/total_bytes_r:.0f} << ridge {ridge:.0f})")
    else:
        print(f"  → COMPUTE-BOUND")

    # =================== Combined fwd ===================
    print("\n[Combined fwd] prepare + recurrence:")
    combined_in = (T * H * K * dtype_bytes * 3  # q, k, g
                   + T * H * dtype_bytes  # beta
                   + H * fp32_bytes + H * K * fp32_bytes  # A_log, dt_bias
                   + T * H * V * dtype_bytes)  # v
    combined_out = (T * H * V * dtype_bytes  # o
                    + H * K * V * fp32_bytes)  # final_state
    combined_total = combined_in + combined_out
    combined_flops = flops_p + flops_r
    print(f"  unique I/O: {combined_total/1024/1024:.1f} MB")
    print(f"  total FLOPs: {combined_flops/1e9:.1f} GFLOPs")
    print(f"  bw limit:    {combined_total/448e9*1000:.3f} ms")
    print(f"  tc limit:    {combined_flops/419e12*1000:.3f} ms")
    print(f"  → roofline:  {max(combined_total/448e9, combined_flops/419e12)*1000:.3f} ms")


if __name__ == "__main__":
    main()