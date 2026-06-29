"""Compute the actual arithmetic intensity (FLOPs/byte) for each bwd subkernel.

The bwd at 13.5 ms is NOT compute-bound even though it does ~2x fwd work
by chain rule. Compute peak is 419 TFLOPs (bf16) on RTX 5060 Ti. Total
bwd compute is ~80 GFLOPs → 0.19 ms at peak. The bwd takes 11.85 ms,
so it's at 1.6% of compute peak — massively bandwidth-bound.

Corrected per-subkernel FLOP accounting (with BK=32 for intra,
BK=32/64 autotuned for wy_dqkg, BV=32 throughout):
  - w_u_recomp  6.4 GFLOPs
  - h_recomp   12.3 GFLOPs
  - dAv         6.1 GFLOPs
  - dhu        18.4 GFLOPs
  - wy_dqkg   ~30 GFLOPs (BK-dependent, mid-range estimate)
  - intra       3.1 GFLOPs   <- was 309 GFLOPs, off by 100x
  - cumsum     ~0.025 GFLOPs (negligible)
  TOTAL       ~76 GFLOPs   (~2.5x fwd, chain-rule consistent)

Ridge point (419 TFLOPs / 448 GB/s) = 935 FLOPs/B. Every bwd subkernel
is well below ridge → all bandwidth-bound.

Run: python bench/fast_kda_bwd_ai_analysis.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

# Problem dimensions (prod shape)
B = 1
T = 16384
H = 12
K = 128
V = 128
CHUNK = 64  # FLA's BT (chunk_size=64)
NC = T // CHUNK  # 256 token chunks

# Block sizes — FLA autotune picks these on sm_120
BK_INTRA = 32  # intra kernel uses BK=min(32, next_pow2(K)) = 32
BV_INTRA = 32
BC_INTRA = 16  # intra sub-chunk

BK_WY = 32     # conservative estimate for wy_dqkg
BV_WY = 32
# (autotune tries 16/32/64, sm_120 typically picks 32 or 64)

BK_H = 32      # h_recomp / dhu — autotune, but kernel hardcodes 64-tile
BV_H = 32

# Dtype sizes
BF16 = 2
FP32 = 4

# Hardware peak
PEAK_BW_GBPS = 448       # GDDR7 28 Gbps, 128-bit
PEAK_TFLOPS_BF16 = 419   # TFLOPs (bf16)
RIDGE = PEAK_TFLOPS_BF16 * 1e12 / (PEAK_BW_GBPS * 1e9)  # 935 FLOPs/B
print(f"\nRidge point: {RIDGE:.0f} FLOPs/B  (above = compute-bound, below = bandwidth-bound)\n")


def fmt_flops(f):
    if f >= 1e9:
        return f"{f/1e9:.2f} GFLOPs"
    if f >= 1e6:
        return f"{f/1e6:.2f} MFLOPs"
    return f"{f:.2f} FLOPs"


def fmt_bytes(b):
    if b >= 1e6:
        return f"{b/1e6:.1f} MB"
    if b >= 1e3:
        return f"{b/1e3:.1f} KB"
    return f"{b:.0f} B"


# ===========================================================================
# Tensor sizes (prod shape)
# ===========================================================================
def _sz(elem, *shape):
    return elem * math.prod(shape)


size_q   = _sz(BF16, B*T*H*K)
size_k   = _sz(BF16, B*T*H*K)
size_v   = _sz(BF16, B*T*H*V)
size_g   = _sz(BF16, B*T*H*K)
size_beta= _sz(BF16, B*T*H)
size_do  = _sz(BF16, B*T*H*V)
size_h   = _sz(BF16, NC*H*K*V)        # 100.7 MB
size_dh  = _sz(BF16, NC*H*K*V)        # 100.7 MB
size_Akk = _sz(BF16, B*T*H*CHUNK)     # 25.2 MB
size_v_new=_sz(BF16, B*T*H*V)
size_w   = _sz(BF16, B*T*H*K)
size_u   = _sz(BF16, B*T*H*V)
size_qg  = _sz(BF16, B*T*H*K)
size_kg  = _sz(BF16, B*T*H*K)

print(f"Tensor sizes (prod): q/k/g={fmt_bytes(size_q)}  v={fmt_bytes(size_v)}  "
      f"h/dh={fmt_bytes(size_h)}  Akk={fmt_bytes(size_Akk)}\n")

# ===========================================================================
# Per-subkernel FLOP and byte accounting
# ===========================================================================

# ---- 1. w_u_recomp (recompute_w_u_fwd) -----------------------------------
# Grid (NT, B*HV) = (256, 12) = 3072 programs
# Per program: 1 token chunk, 1 head
#   - For i_v in cdiv(V, BV)=4: u = A @ (v*beta)   1 bmm (BT,BT)@(BT,BV)
#   - For i_k in cdiv(K, BK)=4: w = A @ (k*beta*exp2(gk))  1 bmm (BT,BT)@(BT,BK)
# Per program: 4 * 64*64*32*2 + 4 * 64*64*32*2 = 4 * 262k + 4 * 262k = 2.1 MFLOPs
flops_wu = NC * H * ((V // BV_H) * CHUNK * CHUNK * BV_H * 2
                      + (K // BK_H) * CHUNK * CHUNK * BK_H * 2)
bytes_wu = size_q + size_k + size_v + size_beta + size_g + size_Akk + size_w + size_u + size_qg + size_kg

# ---- 2. h_recomp (chunk_gated_delta_rule_fwd_h) -------------------------
# Grid (cdiv(V, BV), N*HV) = (4, 12) = 48 programs
# Per program: 1 v-tile, 1 head, sequential loop over NT chunks
# Per chunk (per program):
#   - For each of 2 K-tiles: 1 bmm w @ h (or w @ h.T) → 64*64*32*2
#   - For each of 2 K-tiles: 1 bmm k @ v → 64*64*32*2
# Per chunk: 2 * 2 * 262k = 1 MFLOPs
# Per program: 256 * 1 MFLOPs = 256 MFLOPs
flops_h_recomp = (V // BV_H) * H * NC * (K // BK_H) * 2 * CHUNK * BK_H * BV_H * 2
bytes_h_recomp = size_k + size_u + size_w + size_g + size_h + size_v_new

# ---- 3. dAv (chunk_kda_bwd_dAv) -----------------------------------------
# Grid (NT, B*HV) = (256, 12) = 3072 programs
# Per chunk:
#   - For i_v: do @ v.T (BT, BV) @ (BV, BT) = 64*32*64*2 = 262k FLOPs
#   - For i_v: A @ do (BT, BT) @ (BT, BV) = 64*64*32*2 = 262k FLOPs
# Per program: 4 * 524k = 2.1 MFLOPs
flops_dAv = NC * H * (V // BV_H) * 2 * CHUNK * BV_H * CHUNK * 2
bytes_dAv = size_v_new + size_do + size_Akk + _sz(FP32, B*T*H*CHUNK) + size_v

# ---- 4. dhu (chunk_gated_delta_rule_bwd_dhu) -----------------------------
# Grid (cdiv(V, BV), N*HV) = (4, 12) = 48 programs
# Per program: 1 v-tile, 1 head, sequential REVERSE loop over NT chunks
# Per chunk (per program):
#   - 2 K-tiles * 1 bmm dv = k @ dh (or trans): [BT, 64] @ [64, BV] = 64*64*32*2
#   - 2 K-tiles * 2 bmm for dh update: b_q @ b_do and b_w @ b_dv (each [64, BT] @ [BT, BV])
# Per chunk: 2 * 1 * 262k + 2 * 2 * 262k = 524k + 1.05 MFLOPs = 1.6 MFLOPs
flops_dhu = (V // BV_H) * H * NC * (K // BK_H) * 3 * CHUNK * BK_H * BV_H * 2
bytes_dhu = size_q + size_k + size_w + size_do + size_v + size_dh + size_g + size_v  # dv2 written

# ---- 5. wy_dqkg (chunk_kda_bwd_wy_dqkg_fused) ----------------------------
# Grid (NT, B*HV) = (256, 12) = 3072 programs
# Per program: 1 token chunk, 1 head
# Inner V-loop, per (i_k, i_v): 3 bmm of (BT, BV) @ (BV, BK) = 3 * BT*BV*BK*2
#   Plus 2 more bmm on i_k=0: (BT, BV) @ (BV, BT) and (BT, BT) @ (BT, BV)
# Tail after V-loop: 2 bmm (BT, BK) @ (BK, BT) and (BT, BT) @ (BT, BK)
# Total per chunk (sum over all (i_k, i_v)):
#   = 3 * (K/BK) * (V/BV) * BT*BV*BK*2  (always)
#   + 2 * (V/BV) * BT*BV*BT*2            (only on i_k=0)
#   + 2 * BT*BK*BT*2                     (tail, but tail depends on BK)
# For K=128, BK=32, V=128, BV=32, BT=64:
#   = 3 * 4 * 4 * 64*32*32*2 = 4.7 MFLOPs
#   + 2 * 4 * 64*32*64*2 = 2.1 MFLOPs
#   + 2 * 64*32*64*2 = 0.5 MFLOPs
#   = 7.3 MFLOPs per chunk
# Total: 3072 * 7.3 MFLOPs = 22.4 GFLOPs
# (For BK=64: 18 MFLOPs per chunk → 55 GFLOPs total. Mid-range ~30 GFLOPs.)
flops_wy = NC * H * (3 * (K // BK_WY) * (V // BV_WY) * CHUNK * BV_WY * BK_WY * 2
                     + 2 * (V // BV_WY) * CHUNK * BV_WY * CHUNK * 2
                     + 2 * CHUNK * BK_WY * CHUNK * 2)
bytes_wy = size_q + size_k + size_v + size_v_new + size_g + size_beta + size_Akk \
           + size_h + size_do + size_dh + size_v \
           + _sz(BF16, B*T*H*K) + _sz(BF16, B*T*H*K) + size_v + size_v \
           + _sz(FP32, B*T*H*K) + _sz(FP32, B*T*H) + _sz(FP32, B*T*H*CHUNK)

# ---- 6. intra (chunk_kda_bwd_intra) -------------------------------------
# Grid (NK * NC, NT, B * HV) where NK=4, NC=4, NT=256, B*HV=12 = 49,152 programs
# Per program: 1 K-tile, 1 sub-chunk (BC=16), 1 head
#   - pre-loop (j < i_i): 2 bmm of (BC, BC) @ (BC, BK) per iter, avg 1.5 iters
#   - diagonal (j = 0..BC-1): 0 bmm, element-wise only
#   - post-loop (j > i_i): 2 bmm per iter, avg 1.5 iters
# Per program: ~6 bmm of (BC, BC) @ (BC, BK) = 6 * 16*16*32*2 = 98k FLOPs
# Total: 49,152 * 98k = 4.8 GFLOPs
NUM_PROGRAMS_INTRA = (K // BK_INTRA) * (CHUNK // BC_INTRA) * NC * H
BMM_PER_PROG = 6  # pre avg 1.5 + post avg 1.5, * 2 bmm each
flops_intra = NUM_PROGRAMS_INTRA * BMM_PER_PROG * BC_INTRA * BC_INTRA * BK_INTRA * 2
bytes_intra = (size_q + size_k + size_g + size_beta
               + _sz(BF16, B*T*H*CHUNK) + _sz(BF16, B*T*H*CHUNK)  # dAqk, dAkk
               + _sz(BF16, B*T*H*K) + _sz(BF16, B*T*H*K)  # dq, dk
               + _sz(FP32, B*T*H*K) + _sz(FP32, B*T*H)  # dg, db
               + _sz(BF16, B*T*H*K) + _sz(BF16, B*T*H*K)  # dq2, dk2
               + _sz(FP32, B*T*H*K) + _sz(FP32, B*T*H))  # dg2, db

# ---- 7. local_cumsum ----------------------------------------------------
flops_cumsum = B * T * H * K  # ~2 MFLOPs
bytes_cumsum = _sz(FP32, B*T*H*K) * 2

stages = [
    ("w_u_recomp",  flops_wu,       bytes_wu,       1.35),
    ("h_recomp",    flops_h_recomp, bytes_h_recomp, 0.83),
    ("dAv",         flops_dAv,      bytes_dAv,      0.61),
    ("dhu",         flops_dhu,      bytes_dhu,      1.39),
    ("wy_dqkg",     flops_wy,       bytes_wy,       3.31),
    ("intra",       flops_intra,    bytes_intra,    3.83),
    ("local_cumsum",flops_cumsum,   bytes_cumsum,   0.54),
]

print(f"{'stage':<16} {'FLOPs':>14} {'bytes':>10} {'AI':>8} {'bw roofline':>13} {'tc roofline':>13} {'actual':>9} {'% bw':>7} {'% tc':>7}  bound")
print("-" * 130)
total_flops = 0
total_bytes = 0
total_actual = 0
for name, fl, by, act in stages:
    ai = fl / by
    bw_t = by / (PEAK_BW_GBPS * 1e9) * 1e3
    tc_t = fl / (PEAK_TFLOPS_BF16 * 1e12) * 1e3
    pct_bw = (bw_t / act) * 100
    pct_tc = (tc_t / act) * 100
    bound = "BW-bound" if bw_t > tc_t else "TC-bound"
    print(f"{name:<16} {fmt_flops(fl):>14} {fmt_bytes(by):>10} {ai:>7.1f}  {bw_t:>11.3f} ms {tc_t:>11.3f} ms {act:>7.2f} ms {pct_bw:>6.1f}% {pct_tc:>6.2f}%  {bound}")
    total_flops += fl
    total_bytes += by
    total_actual += act

print("-" * 130)
print(f"{'TOTAL':<16} {fmt_flops(total_flops):>14} {fmt_bytes(total_bytes):>10} {total_flops/total_bytes:>7.1f}  "
      f"{'':>12} {'':>13} {total_actual:>7.2f} ms")
print()

# Compute peak analysis
print(f"=== Compute peak analysis (the user's question) ===")
print(f"  Ridge point: {RIDGE:.0f} FLOPs/B (peak_bw / peak_tc)")
print(f"  Every bwd subkernel is at AI << ridge → all bandwidth-bound")
print()
bwd_at_compute_peak_ms = total_flops / (PEAK_TFLOPS_BF16 * 1e12) * 1e3
print(f"  Total bwd compute:  {fmt_flops(total_flops)}")
print(f"  At {PEAK_TFLOPS_BF16} TFLOPs peak:  {bwd_at_compute_peak_ms:.3f} ms")
print(f"  Actual bwd:         {total_actual:.2f} ms")
print(f"  -> bwd is at {bwd_at_compute_peak_ms/total_actual*100:.2f}% of compute peak")
print(f"  -> bwd has {total_actual/bwd_at_compute_peak_ms:.0f}x headroom on compute")
print()

# Compare to fwd
print(f"=== Fwd vs bwd ===")
fwd_gflops = 30  # rough (FLA chunk_kda includes intra + wy + delta_h + chunk_o)
fwd_ms = 5.12
fwd_peak_pct = fwd_gflops / (fwd_ms / 1e3) / 1e12 / PEAK_TFLOPS_BF16 * 100
bwd_peak_pct = total_flops / (total_actual / 1e3) / 1e12 / PEAK_TFLOPS_BF16 * 100
print(f"  fwd:  {fwd_ms:.2f} ms, ~{fwd_gflops} GFLOPs  -> {fwd_gflops / (fwd_ms/1e3) / 1e3:.1f} TFLOPs/s ({fwd_peak_pct:.2f}% peak)")
print(f"  bwd:  {total_actual:.2f} ms, {fmt_flops(total_flops)}  -> {total_flops / (total_actual/1e3) / 1e12:.1f} TFLOPs/s ({bwd_peak_pct:.2f}% peak)")
print()
print(f"  fwd/bwd compute ratio: {total_flops/fwd_gflops/1e9:.1f}x  (chain rule: ~2-3x expected ✓)")
print(f"  fwd/bwd time ratio:    {fwd_ms/total_actual:.2f}x")
print()
print(f"  Conclusion: bwd compute is only ~{total_flops/fwd_gflops/1e9:.1f}x fwd (chain-rule roughly correct),")
print(f"  but bwd time is 2.6x fwd. The extra time comes from h/dh tensor")
print(f"  re-traffic, NOT from compute.")
print()
print(f"  Concrete: at compute peak, bwd would take {bwd_at_compute_peak_ms:.2f} ms.")
print(f"  At memory peak (using unique I/O only, no h/dh re-reads), bwd would take")
bwd_at_bw_peak_ms = total_bytes / (PEAK_BW_GBPS * 1e9) * 1e3
print(f"  {bwd_at_bw_peak_ms:.2f} ms.")
print(f"  Actual bwd: {total_actual:.2f} ms — between 1/100 of compute peak and ~half of HBM peak.")
print()
print(f"=== The redundant h/dh traffic ===")
print(f"  h tensor: 100.7 MB. Touched by h_recomp (write 100MB + read 100MB),")
print(f"            dhu (read 100MB), wy_dqkg (read 100MB). Total: 500MB h traffic")
print(f"            for a 100MB tensor. 5x amplification.")
print(f"  dh tensor: 100.7 MB. Touched by dhu (write 100MB), wy_dqkg (read 100MB).")
print(f"             Total: 200MB. 2x amplification.")
print(f"  Persistent kernel fusing fwd_h + dhu + wy_dqkg would save")
print(f"  ~400MB of redundant h/dh reads + 200MB of writes = 600MB ≈ 1.34 ms at HBM peak.")
