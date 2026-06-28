"""Per-stage profile: where does CUDA KDA fwd spend its time?

Times each phase of the CUDA pipeline by running the wrapper with
profiler hooks. The phases are:
  1. Python orchestration: cumsum, exp2, view/transpose
  2. cuBLAS bmm: A_qk + A_kk (intra_solve bmm calls)
  3. forward_sub kernel (custom CUDA)
  4. cuBLAS bmm: u + w (wy_transform bmm calls)
  5. delta_h kernel (custom CUDA)
  6. chunk_o kernel (custom CUDA)

This file complements bench/kda_fwd_bench.py (which compares end-to-end
CUDA vs FLA) by breaking down where the gap lives.

Usage
-----
    python bench/kda_fwd_profile.py --shape prod
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

from src.models.ops.cuda.kda_fwd import chunk_kda_fwd
from src.models.ops._vendored.fla.ops.kda import chunk_kda


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


SHAPES = {
    "small":  (1,   256,   4,  64,  64),
    "medium": (1,  1024,   4,  64,  64),
    "k128":   (1,  1024,   8, 128, 128),
    "prod":   (1, 16384,  12, 128, 128),
}


def _time_block(fn, iters: int = 10) -> float:
    """Median ms over `iters` runs (5 warmup)."""
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        e.synchronize()
        times.append(s.elapsed_time(e))
    times.sort()
    return times[len(times) // 2]


def _profile_with_torch_profiler(fn, name: str):
    """Use torch.profiler to get the per-op breakdown."""
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA, torch.profiler.ProfilerActivity.CPU],
        record_shapes=False,
    ) as prof:
        for _ in range(3):
            fn()
        torch.cuda.synchronize()

    print(f"\n=== torch.profiler breakdown for {name} ===")
    print(prof.key_averages().table(
        sort_by="cuda_time_total", row_limit=20,
    ))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--shape", default="prod", choices=list(SHAPES.keys()))
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--torch-profile", action="store_true",
                   help="Also dump torch.profiler table")
    args = p.parse_args()

    B, T, H, K, V = SHAPES[args.shape]
    device = torch.device("cuda:0")
    dtype = torch.bfloat16
    scale = 1.0 / math.sqrt(K)

    q, k, v, g, beta = _make_inputs(B, T, H, K, V, device, dtype, seed=7)

    # End-to-end CUDA vs FLA
    cuda_ms = _time_block(lambda: chunk_kda_fwd(q, k, v, g, beta, scale=scale), args.iters)
    fla_ms = _time_block(lambda: chunk_kda(q=q, k=k, v=v, g=g, beta=beta, scale=scale), args.iters)

    print(f"\nShape {args.shape}: B={B} T={T} H={H} K={K} V={V}")
    print(f"  FLA  total: {fla_ms:.3f} ms")
    print(f"  CUDA total: {cuda_ms:.3f} ms (speedup: {fla_ms/cuda_ms:.2f}x)")
    print(f"  Gap:        {cuda_ms - fla_ms:.3f} ms ({100*(cuda_ms - fla_ms)/fla_ms:.1f}% slower)")

    if args.torch_profile:
        _profile_with_torch_profiler(
            lambda: chunk_kda_fwd(q, k, v, g, beta, scale=scale),
            f"chunk_kda_fwd {args.shape}",
        )
        _profile_with_torch_profiler(
            lambda: chunk_kda(q=q, k=k, v=v, g=g, beta=beta, scale=scale),
            f"chunk_kda (FLA) {args.shape}",
        )


if __name__ == "__main__":
    main()
