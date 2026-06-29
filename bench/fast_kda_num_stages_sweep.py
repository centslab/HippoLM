"""NUM_STAGES sweep on the top configs from the tiling sweep.

C=16 V=16 (current) and C=32 V=32 (best) tested with ns=1,2,3,4 to see
if the adaptive heuristic is missing wins.

Run: python3 bench/fast_kda_num_stages_sweep.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from src.models.ops._triton.fast_kda.recurrence import _kda_recurrence_kernel


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
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        with CudaTimer() as t:
            fn()
        times.append(t.ms)
    times.sort()
    return times[len(times) // 2]


def call_kernel(q_decayed, k_restored, g_total, mqk, v,
                BLOCK_M, BLOCK_V, NUM_STAGES):
    NC, H, _, K = q_decayed.shape
    B, T, Hv, V = v.shape
    assert H == Hv and K == V
    assert V % BLOCK_V == 0
    num_v_splits = V // BLOCK_V

    o = torch.empty(B, T, H, V, dtype=v.dtype, device=v.device)
    h_intermediate = torch.empty(NC, H, K, V, dtype=torch.bfloat16, device=v.device)
    final_state = torch.empty(B, H, K, V, dtype=torch.float32, device=v.device)

    grid = (H, num_v_splits)
    _kda_recurrence_kernel[grid](
        q_decayed, k_restored, g_total, mqk,
        v,
        h_intermediate, None, final_state,
        o,
        B, T, NC, K, V,
        q_decayed.stride(0), q_decayed.stride(1),
        k_restored.stride(0), k_restored.stride(1),
        g_total.stride(0), g_total.stride(1),
        mqk.stride(0), mqk.stride(1),
        v.stride(1), v.stride(2),
        h_intermediate.stride(0), h_intermediate.stride(1),
        0, 0,
        final_state.stride(0), final_state.stride(1),
        o.stride(1), o.stride(2),
        USE_H0=False, STORE_HT=True,
        BLOCK_M=BLOCK_M, BLOCK_K=K, BLOCK_V=BLOCK_V, NUM_STAGES=NUM_STAGES,
    )
    return o, h_intermediate, final_state


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--shape", default="prod")
    p.add_argument("--iters", type=int, default=20)
    args = p.parse_args()

    B, T, H, K, V = 1, 16384, 12, 128, 128
    if args.shape == "small":
        T = 1024
    elif args.shape == "medium":
        T = 4096

    device = torch.device("cuda:0")
    dtype = torch.bfloat16
    g = torch.Generator(device=device).manual_seed(7)

    # Top configs from tiling sweep
    CONFIGS = [
        # (CHUNK, BLOCK_V, label)
        (16, 16, "current"),
        (32, 16, "C=32 V=16"),
        (32, 32, "C=32 V=32 (best)"),
        (16, 32, "C=16 V=32"),
    ]
    NS_CHOICES = [1, 2, 3, 4]

    print(f"\n=== NUM_STAGES sweep (B={B} T={T} H={H} K={V} V={V}) ===\n")
    print(f"{'config':<22} {'NS=1':<10} {'NS=2':<10} {'NS=3':<10} {'NS=4':<10} {'best':<10}")
    print("-" * 75)

    for CHUNK, BLOCK_V, label in CONFIGS:
        NC = (T + CHUNK - 1) // CHUNK
        # Pre-build workspace for this config
        q_decayed = torch.randn(NC, H, CHUNK, K, device=device, dtype=dtype, generator=g)
        k_restored = torch.randn(NC, H, CHUNK, K, device=device, dtype=dtype, generator=g)
        g_total   = torch.randn(NC, H, K,        device=device, dtype=torch.float32, generator=g)
        mqk       = torch.randn(NC, H, CHUNK, CHUNK, device=device, dtype=dtype, generator=g)
        v         = torch.randn(B, T, H, V,      device=device, dtype=dtype, generator=g)

        results = {}
        for ns in NS_CHOICES:
            def _run():
                call_kernel(q_decayed, k_restored, g_total, mqk, v,
                            BLOCK_M=CHUNK, BLOCK_V=BLOCK_V, NUM_STAGES=ns)
            try:
                ms = _bench(_run, args.iters)
                results[ns] = ms
            except Exception as e:
                results[ns] = f"FAIL"
        best_ns = min(results, key=lambda k: results[k] if isinstance(results[k], float) else float('inf'))
        best_ms = results[best_ns] if isinstance(results[best_ns], float) else "FAIL"

        cells = [f"{results[ns]:.3f}" if isinstance(results[ns], float) else "FAIL" for ns in NS_CHOICES]
        print(f"{label:<22} {cells[0]:<10} {cells[1]:<10} {cells[2]:<10} {cells[3]:<10} "
              f"ns={best_ns} {best_ms}")


if __name__ == "__main__":
    main()
