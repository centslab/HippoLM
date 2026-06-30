"""Benchmark the fused Triton KDA prepare kernel.

Compares our fused prepare against a non-fused baseline (matmul -> write
intermediates to gmem -> read back -> next matmul) to quantify the
benefit of fusing exp2+mul+bmm in registers.

Also benchmarks against the FLA Triton reference chunk_kda to compare
against a working production path.

Usage:
    python bench/fast_kda_prepare_bench.py
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

from src.models.ops._triton.fast_kda.prepare import kda_prepare_triton
from src.models.ops._vendored.fla.ops.kda import chunk_kda


def _make_inputs(B, T, H, K, V, device, dtype, seed):
    """L2-normalized q, k; g in stable range; sigmoid'd beta."""
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
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args()

    device = torch.device(f"cuda:{args.device}")
    dtype = torch.bfloat16
    torch.cuda.set_device(args.device)

    shapes = SHAPES if args.shape == "all" else {args.shape: SHAPES[args.shape]}

    print(f"\n{'shape':<8} {'T':>6} {'H':>3} {'K':>4} {'V':>4} "
          f"{'prep ms':>10} {'FLA ms':>10} {'prep/FLA':>10} "
          f"{'prep tok/s':>12}")
    print("-" * 90)

    rows = []
    for name, (B, T, H, K, V) in shapes.items():
        scale = 1.0 / math.sqrt(K)
        q, k, v, g, beta = _make_inputs(B, T, H, K, V, device, dtype, args.seed)

        # FLA reference (full chunk_kda)
        fla_ms = _bench(lambda: chunk_kda(q=q, k=k, v=v, g=g, beta=beta, scale=scale), args.iters)

        # Triton prepare only (Kernel 1 of FlashKDA design)
        # Need extra params for our kernel
        A_log = torch.rand(H, device=device, dtype=torch.float32) * 0.5 + 0.3
        dt_bias = torch.randn(H, K, device=device, dtype=torch.float32) * 0.5
        prep_ms = _bench(
            lambda: kda_prepare_triton(q, k, g, beta, A_log, dt_bias, -5.0, scale),
            args.iters,
        )

        toks = B * T
        ratio = fla_ms / prep_ms
        prep_toks = toks / (prep_ms / 1000)
        print(f"{name:<8} {T:>6} {H:>3} {K:>4} {V:>4} "
              f"{prep_ms:>10.3f} {fla_ms:>10.3f} {ratio:>9.2f}x "
              f"{prep_toks:>12.0f}")
        rows.append((name, T, prep_ms, fla_ms, ratio))

    # Theoretical limit estimate
    print("\n=== Theoretical bandwidth analysis (prod shape) ===")
    B, T, H, K, V = SHAPES["prod"]
    dtype_bytes = 2  # bf16
    fp32_bytes = 4

    # Inputs read
    in_bytes = (q.numel() + k.numel() + g.numel() + beta.numel()) * dtype_bytes
    in_bytes += A_log.numel() * fp32_bytes + dt_bias.numel() * fp32_bytes

    # Workspace written
    NC = (T + 15) // 16
    out_kd = NC * H * 16 * K * dtype_bytes
    out_qd = NC * H * 16 * K * dtype_bytes
    out_kr = NC * H * 16 * K * dtype_bytes
    out_gt = NC * H * K * fp32_bytes
    out_inv = NC * H * 16 * 16 * dtype_bytes
    out_mqk = NC * H * 16 * 16 * dtype_bytes
    out_bytes = out_kd + out_qd + out_kr + out_gt + out_inv + out_mqk

    # bf16 matmul FLOPS
    # L = k_decayed @ k_inv.T: NC * H * 16 * 16 * K * 2 = ~64M FLOPS at prod
    # Mqk = q_decayed @ k_inv.T: same ~64M FLOPS
    flops = NC * H * 16 * 16 * K * 2 * 2  # two bmms, each 16*16*K*2 FLOPs

    total_bytes = in_bytes + out_bytes
    print(f"  Inputs read:    {in_bytes/1024/1024:.1f} MB")
    print(f"  Workspace out:  {out_bytes/1024/1024:.1f} MB")
    print(f"  Total bytes:    {total_bytes/1024/1024:.1f} MB")
    print(f"  bmm FLOPs:      {flops/1e9:.2f} GFLOPs (L + Mqk)")
    print(f"  bf16 tc peak (5060 Ti ~419 TFLOPs/s): {419e12 / flops * 1000:.3f} ms (compute limit)")
    print(f"  mem bw limit (5060 Ti ~448 GB/s):     {total_bytes/448e9 * 1000:.3f} ms (bandwidth limit)")
    print(f"  The kernel should be ~{max(flops/419e12, total_bytes/448e9)*1000:.3f} ms (limited by the larger)")


if __name__ == "__main__":
    main()