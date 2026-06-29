"""Sweep BLOCK_V and CHUNK for the Triton KDA recurrence kernel.

The fwd analysis showed V-split (96 programs) is at 99% of HBM peak;
V-loop (12 programs) is at 7% (can't saturate 36 SMs). This sweep
explores the design space between these extremes.

Axes:
  BLOCK_V: 4, 8, 16, 32, 64, 128  (V-split granularity)
  CHUNK:   8, 16, 32, 64           (chunk size = M dim of bmm)

For each combination, report:
  - # programs = H * V / BLOCK_V
  - HBM peak %
  - wall time
  - h_prev register size (K * BLOCK_V * 4 bytes fp32)

Run: python bench/fast_kda_tiling_sweep.py
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from src.models.ops._triton.fast_kda.recurrence import _kda_recurrence_kernel


# ===========================================================================
# CUDA timer
# ===========================================================================
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
    times = []
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    for _ in range(iters):
        with CudaTimer() as t:
            fn()
        times.append(t.ms)
    times.sort()
    return times[len(times) // 2]


# ===========================================================================
# Direct kernel call (bypasses wrapper so we can vary CHUNK and BLOCK_V)
# ===========================================================================
def call_kernel(q_decayed, k_restored, g_total, mqk, v,
                BLOCK_M, BLOCK_V, NUM_STAGES, use_h0=False, store_ht=True):
    """Call the recurrence kernel directly with arbitrary tiling."""
    NC, H, _, K = q_decayed.shape
    B, T, Hv, V = v.shape
    assert H == Hv
    assert K == V, f"only K=V=128 supported, got K={K} V={V}"
    assert V % BLOCK_V == 0
    num_v_splits = V // BLOCK_V
    device = v.device
    dtype = v.dtype

    o = torch.empty(B, T, H, V, dtype=dtype, device=device)
    h_intermediate = torch.empty(NC, H, K, V, dtype=torch.bfloat16, device=device)
    if store_ht:
        final_state = torch.empty(B, H, K, V, dtype=torch.float32, device=device)
    else:
        final_state = None
    initial_state = None

    grid = (H, num_v_splits)
    _kda_recurrence_kernel[grid](
        q_decayed, k_restored, g_total, mqk,
        v,
        h_intermediate, initial_state, final_state,
        o,
        B, T, NC, K, V,
        q_decayed.stride(0), q_decayed.stride(1),
        k_restored.stride(0), k_restored.stride(1),
        g_total.stride(0), g_total.stride(1),
        mqk.stride(0), mqk.stride(1),
        v.stride(1), v.stride(2),
        h_intermediate.stride(0), h_intermediate.stride(1),
        0, 0,
        final_state.stride(0) if store_ht else 0,
        final_state.stride(1) if store_ht else 0,
        o.stride(1), o.stride(2),
        USE_H0=False,
        STORE_HT=store_ht,
        BLOCK_M=BLOCK_M,
        BLOCK_K=K,
        BLOCK_V=BLOCK_V,
        NUM_STAGES=NUM_STAGES,
    )
    return o, h_intermediate, final_state


# ===========================================================================
# Sweep
# ===========================================================================
def make_workspace(B, T, H, K, V, CHUNK, device, dtype, seed):
    """Create synthetic workspace tensors (no prepare, just for sweep)."""
    NC = (T + CHUNK - 1) // CHUNK
    g = torch.Generator(device=device).manual_seed(seed)
    q_decayed = torch.randn(NC, H, CHUNK, K, device=device, dtype=dtype, generator=g)
    k_restored = torch.randn(NC, H, CHUNK, K, device=device, dtype=dtype, generator=g)
    g_total   = torch.randn(NC, H, K,        device=device, dtype=torch.float32, generator=g)
    mqk       = torch.randn(NC, H, CHUNK, CHUNK, device=device, dtype=dtype, generator=g)
    v         = torch.randn(B, T, H, V,      device=device, dtype=dtype, generator=g)
    return q_decayed, k_restored, g_total, mqk, v


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--shape", default="prod")
    p.add_argument("--iters", type=int, default=10)
    args = p.parse_args()

    B, T, H, K, V = 1, 16384, 12, 128, 128
    if args.shape == "small":
        T = 1024
    elif args.shape == "medium":
        T = 4096
    elif args.shape == "prod":
        T = 16384

    device = torch.device("cuda:0")
    dtype = torch.bfloat16

    BLOCK_V_CHOICES = [4, 8, 16, 32, 64, 128]
    CHUNK_CHOICES   = [8, 16, 32, 64]

    # HBM peak
    HBM_PEAK_GBPS = 448  # GB/s (GDDR7 28 Gbps, 128-bit)

    print(f"\n=== Tiling sweep (B={B} T={T} H={H} K={V} V={V}) ===\n")
    print(f"{'CHUNK':<6} {'BLOCK_V':<8} {'progs':<7} {'waves':<6} {'h_prev reg':<12} "
          f"{'ms':<8} {'MB':<8} {'% HBM':<7}")
    print("-" * 70)

    NUM_SMS = 36

    for CHUNK in CHUNK_CHOICES:
        NC = (T + CHUNK - 1) // CHUNK
        # Workspace: same total size regardless of CHUNK
        ws_total = NC * H * CHUNK * K * 2 * 3  # qd + kr (bf16)
        ws_total += NC * H * K * 4              # gt (fp32)
        ws_total += NC * H * CHUNK * CHUNK * 2  # Mqk (bf16)
        # V-amp part (per chunk, V-amp is 128/BLOCK_V per program):
        # Each program loads Mqk, qd, kr, gt 128/BLOCK_V times for the V-amp
        # Plus the V and O tensors (NOT V-amped): 2 * T * H * V * 2

        for BLOCK_V in BLOCK_V_CHOICES:
            if V % BLOCK_V != 0:
                continue
            num_v_splits = V // BLOCK_V
            num_progs = H * num_v_splits
            waves = num_progs / NUM_SMS
            h_prev_reg_kb = K * BLOCK_V * 4 / 1024  # fp32, K * BLOCK_V

            # Build workspace for this CHUNK
            q_decayed, k_restored, g_total, mqk, v = make_workspace(
                B, T, H, K, V, CHUNK, device, dtype, seed=7,
            )

            # Adaptive NUM_STAGES: deeper for many chunks
            # Use the wrapper's heuristic
            if NC >= 256:
                ns = 4
            elif NC >= 64:
                ns = 3
            else:
                ns = 2

            def _run():
                call_kernel(q_decayed, k_restored, g_total, mqk, v,
                            BLOCK_M=CHUNK, BLOCK_V=BLOCK_V, NUM_STAGES=ns)
            try:
                ms = _bench(_run, args.iters)
            except Exception as e:
                print(f"{CHUNK:<6} {BLOCK_V:<8} {num_progs:<7} {waves:<6.2f} "
                      f"{h_prev_reg_kb:>5.1f} KB  ns={ns}  {'FAIL':<8} {str(e)[:30]}")
                continue

            # Estimate total gmem traffic (per-program unique I/O)
            # Per chunk per program:
            #   Mqk:    CHUNK*CHUNK*2 bytes (V-amp)
            #   q_dec:  CHUNK*K*2    bytes (V-amp)
            #   k_res:  CHUNK*K*2    bytes (V-amp)
            #   g_total:K*4          bytes (V-amp)
            #   v:      CHUNK*BLOCK_V*2 (no V-amp, only this V-slice)
            #   o:      CHUNK*BLOCK_V*2 (no V-amp, only this V-slice)
            #   h_new:  K*BLOCK_V*2   (write to gmem only at last chunk, ~free)
            per_chunk_v_amp = (CHUNK*CHUNK + CHUNK*K + CHUNK*K) * 2 + K * 4
            per_chunk_v     = CHUNK * BLOCK_V * 2
            per_chunk_total = per_chunk_v_amp + per_chunk_v * 2
            # Per program (NC chunks)
            per_prog_bytes = NC * per_chunk_total
            # Total (all programs)
            total_bytes = num_progs * per_prog_bytes

            pct_hbm = total_bytes / (HBM_PEAK_GBPS * 1e9) / (ms / 1e3) * 100

            print(f"{CHUNK:<6} {BLOCK_V:<8} {num_progs:<7} {waves:<6.2f} "
                  f"{h_prev_reg_kb:>5.1f} KB  ns={ns}  {ms:>6.3f}  {total_bytes/1e6:>6.1f}  {pct_hbm:>6.1f}%")

    print()
    print("Reference: V-split current (CHUNK=16, BLOCK_V=16) → 2.15 ms at 99% of HBM peak")
    print("V-loop tested earlier (CHUNK=16, BLOCK_V=128) → 5.76 ms at 7% of HBM peak")


if __name__ == "__main__":
    main()
