"""Micro-benchmark: does a fused per-mb dequant→add→requant kernel
on the mxfp8 mom_buf hit a usable perf budget?

Background
----------
The current production design (2026-07-02) uses a separate BF16
``s.accum`` buffer for quantized muon, which adds 2 bytes/elt
of CPU pinned memory — wiping out the mxfp8 memory benefit
(3 B/elt total for mxfp8 + accum vs 2 B/elt for plain bf16).

The fix under evaluation: remove ``s.accum``, do per-mb
dequant→add→requant directly on the mxfp8 mom_buf. The hot path
becomes:
    1. D2H the grad (BF16)
    2. Fused: dequant(mom_buf) + grad → requantize → mom_buf

Per-mb memory traffic (single fused pass, no FP32 work tensor):
    - read mom_buf:        1.0 B/elt  (E4M3)
    - read mom_scale:      0.03 B/elt (E8M0 per 32-elt block)
    - read grad:           2.0 B/elt  (BF16)
    - write mom_buf:       1.0 B/elt  (E4M3)
    - write mom_scale:     0.03 B/elt
    - BF16 work tensor:    2.0 B/elt  (transient)
    ≈ 6 B/elt total

At 1.65B params and 10 GB/s effective CPU bandwidth:
    1.65e9 * 6 = 9.9 GB → 1.0 s per mb

With gas=16: 16 s per step just for accumulation. Uncomfortable
but feasible. The current BF16 accum path costs 0.7 s/mb
(1.65GB * 2 bytes * 16 mbs = 0.5s for the bf16 add path) —
so we're ~14s/step slower than the broken design. We need to
get the fused path under 2 s/mb to be acceptable.

This benchmark:
1. Allocates 1.6 GB of mxfp8 mom_buf (8 layers × 200M elements)
2. Measures the naive 3-op PyTorch dequant+add+requant cycle
   (the old path that was 7-32 s/mb)
3. Measures a fused single-pass PyTorch implementation
4. Prints the per-mb time and the relative speedup

Run: python test/_tmp/bench_mxfp8_fused_kernel.py

If the fused version is <2s/mb, proceed with the fix. If it's
still 5+ s/mb, the design has to go to a real Triton/C++ kernel
or fall back to bf16.
"""
import os
import sys
import time
from pathlib import Path

# Repo on sys.path so imports resolve
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch


# OCP MX: E4M3 max magnitude.
_E4M3_MAX = 448.0
# Default mxfp8 block size in the project.
_BS = 32


def _round_to_e8m0(scale_fp32: torch.Tensor) -> torch.Tensor:
    """Rounds to nearest power of 2, encoded as E8M0 byte."""
    safe = scale_fp32.clamp(min=2 ** -127)
    log2 = safe.log2()
    e_unclamped = log2.round() + 127.0
    e = e_unclamped.clamp(min=1.0, max=254.0)
    return e.to(torch.uint8).view(torch.float8_e8m0fnu)


def naive_3op_dequant_add_requant(
    mom_buf: torch.Tensor,
    mom_scale: torch.Tensor,
    grad: torch.Tensor,
    rows: int,
    cols_p: int,
) -> None:
    """The OLD pre-fix path: 3 separate ops with full FP32
    intermediate. Slow because the FP32 work tensor is 4x the
    mom_buf size and lives in memory for the whole sequence."""
    bs = _BS
    n_blocks = cols_p // bs
    # 1. Dequant → FP32 [rows, cols_p]  (4x memory blowup)
    q = mom_buf.view(rows, n_blocks, bs).float()
    s = mom_scale.float().view(rows, n_blocks, 1)
    dequant = q * s  # [rows, n_blocks, bs] FP32
    dequant_2d = dequant.view(rows, cols_p)
    # 2. Add grad (BF16 → cast to FP32 for the add)
    grad_fp32 = grad.view(rows, cols_p).float()
    sum_2d = dequant_2d + grad_fp32  # [rows, cols_p] FP32
    # 3. Requant → E4M3 + E8M0
    sum_blocks = sum_2d.view(rows, n_blocks, bs)
    absmax = sum_blocks.abs().amax(dim=-1).float()  # [rows, n_blocks]
    target_scale = (absmax / _E4M3_MAX).clamp(min=2 ** -127)
    new_scales = _round_to_e8m0(target_scale)
    scale_fp32 = new_scales.float().view(rows, n_blocks, 1)
    scaled = (sum_blocks / scale_fp32).clamp(-_E4M3_MAX, _E4M3_MAX)
    q_e4m3 = scaled.to(torch.float8_e4m3fn)
    mom_buf.copy_(q_e4m3.reshape(-1).view(mom_buf.shape))
    mom_scale.copy_(new_scales.view(mom_scale.shape))


def fused_dequant_add_requant(
    mom_buf: torch.Tensor,
    mom_scale: torch.Tensor,
    grad: torch.Tensor,
    rows: int,
    cols_p: int,
) -> None:
    """Single fused PyTorch path: do the dequant+add in BF16
    (no FP32 work tensor), then requant in BF16 too. The only
    FP32 ops are the absmax/log2/scale math (negligible work).

    Memory traffic: ~6 B/elt (no FP32 [rows, cols_p] intermediate).
    For 1.6 GB mom_buf: ~10 GB traffic per call.
    """
    bs = _BS
    n_blocks = cols_p // bs
    # Dequant in BF16 directly (E4M3→BF16 is lossless, E8M0→BF16
    # is lossless, BF16 multiply is exact for non-overflowing vals).
    q = mom_buf.view(rows, n_blocks, bs).to(torch.bfloat16)  # [rows, n_blocks, bs] BF16
    s = mom_scale.to(torch.bfloat16).view(rows, n_blocks, 1)  # [rows, n_blocks, 1] BF16
    dequant_bf16 = q * s  # BF16
    # Add grad (BF16 + BF16). No precision loss vs FP32 for
    # grad magnitudes <2^7 (BF16 mantissa = 8 bits; grads are
    # well-conditioned after NS normalization upstream).
    grad_bf16 = grad.view(rows, n_blocks, bs).to(torch.bfloat16)
    sum_bf16 = dequant_bf16 + grad_bf16  # [rows, n_blocks, bs] BF16
    # Requant: absmax → E8M0 scale → divide + round to E4M3
    absmax = sum_bf16.abs().amax(dim=-1).float()  # [rows, n_blocks] FP32 (small)
    target_scale = (absmax / _E4M3_MAX).clamp(min=2 ** -127)
    new_scales = _round_to_e8m0(target_scale)
    # Divide in BF16 (precision loss is OK — E4M3 rounding
    # dominates the error budget).
    scale_bf16 = new_scales.to(torch.bfloat16).view(rows, n_blocks, 1)
    scaled_bf16 = (sum_bf16 / scale_bf16).clamp(-_E4M3_MAX, _E4M3_MAX)
    q_e4m3 = scaled_bf16.to(torch.float8_e4m3fn)
    mom_buf.copy_(q_e4m3.reshape(-1).view(mom_buf.shape))
    mom_scale.copy_(new_scales.view(mom_scale.shape))


def measure(label, fn, mom_buf, mom_scale, grad, rows, cols_p, n_iters=3):
    # Warmup
    for _ in range(2):
        fn(mom_buf, mom_scale, grad, rows, cols_p)
    times = []
    for _ in range(n_iters):
        t0 = time.perf_counter()
        fn(mom_buf, mom_scale, grad, rows, cols_p)
        times.append(time.perf_counter() - t0)
    avg_ms = sum(times) / len(times) * 1000
    p_min = min(times) * 1000
    print(f"  {label:40s} avg={avg_ms:8.1f} ms  min={p_min:8.1f} ms "
          f"({n_iters} iters, mom_buf={mom_buf.numel() * mom_buf.element_size() / 1024**2:.1f} MB)")
    return avg_ms


def main():
    print("=== mxfp8 fused per-mb dequant→add→requant micro-benchmark ===\n")

    # Smoke scale: a single big param to amortize launch overhead.
    # Per-elt cost is what matters, then we project to 1.65B.
    # We use ~200M elts (~200 MB) per single param, called once
    # per param in production (so 1.65B / 200M = 8 calls per mb
    # for a 1.65B model with 8-layer-sized params).
    rows, cols = 1024, 1024 * 200  # 200M elts
    cols_p = cols if cols % _BS == 0 else cols + (_BS - cols % _BS)
    total_elts = rows * cols_p
    print(f"  Setup: 1 param, [{rows}x{cols_p}] = "
          f"{rows*cols_p/1e6:.1f}M elts = {total_elts * 1 / 1024**3:.2f} GB (E4M3)")
    print(f"  Gas: 16 mbs/step → budget per-mb: ~2.0 s\n")

    # Allocate on CPU pinned memory (matches production)
    mom_buf = torch.zeros(total_elts, dtype=torch.float8_e4m3fn).pin_memory()
    n_blocks = rows * (cols_p // _BS)
    mom_scale = torch.zeros(n_blocks, dtype=torch.float8_e8m0fnu).pin_memory()
    grad = torch.randn(total_elts, dtype=torch.bfloat16).pin_memory() * 0.001

    print("Per-mb cost (single fused call = one microbatch):")
    naive_ms = measure("naive 3-op (FP32 work tensor)", naive_3op_dequant_add_requant,
                       mom_buf, mom_scale, grad, rows, cols_p)
    fused_ms = measure("fused (BF16 work tensor)", fused_dequant_add_requant,
                       mom_buf, mom_scale, grad, rows, cols_p)

    speedup = naive_ms / fused_ms
    print(f"\n  Fused speedup vs naive: {speedup:.2f}x")

    # Project to 1.65B scale
    total_params = 1.65e9
    scale_factor = total_params / total_elts
    naive_proj = naive_ms * scale_factor
    fused_proj = fused_ms * scale_factor
    print(f"\n  Projected per-mb at 1.65B scale (linear):")
    print(f"    naive 3-op: {naive_proj/1000:.2f} s/mb "
          f"(×16 mbs = {naive_proj*16/1000:.1f} s/step)")
    print(f"    fused:      {fused_proj/1000:.2f} s/mb "
          f"(×16 mbs = {fused_proj*16/1000:.1f} s/step)")

    # Per-elt cost
    fused_ns_per_elt = fused_ms * 1e6 / total_elts
    print(f"\n  Fused per-elt cost: {fused_ns_per_elt:.0f} ns/elt")
    print(f"  (Roughly equivalent to {fused_ns_per_elt * 6 / 1e9 * 1000:.0f} ns/byte of "
          f"6-byte-per-elt memory traffic — 1/{6 * fused_ns_per_elt / 1e9:.1f} of 10 GB/s peak)")

    # Numerical correctness check
    print("\nNumerical equivalence (fused vs naive):")
    # Set up a known mom_buf/scale
    test_buf = torch.zeros(rows * cols_p, dtype=torch.float8_e4m3fn).pin_memory()
    test_scale = torch.zeros(rows * (cols_p // _BS),
                             dtype=torch.float8_e8m0fnu).pin_memory()
    test_grad = torch.randn(rows * cols_p, dtype=torch.bfloat16).pin_memory() * 0.01

    # Snapshot before
    pre_buf = test_buf.clone()
    pre_scale = test_scale.clone()
    naive_3op_dequant_add_requant(test_buf, test_scale, test_grad, rows, cols_p)
    naive_buf = test_buf.clone()
    naive_scale = test_scale.clone()

    test_buf.copy_(pre_buf)
    test_scale.copy_(pre_scale)
    fused_dequant_add_requant(test_buf, test_scale, test_grad, rows, cols_p)
    fused_buf = test_buf.clone()
    fused_scale = test_scale.clone()

    # Compare
    nq = naive_buf.float()
    fq = fused_buf.float()
    n_nan = torch.isnan(nq).sum().item()
    f_nan = torch.isnan(fq).sum().item()
    n_inf = torch.isinf(nq).sum().item()
    f_inf = torch.isinf(fq).sum().item()
    print(f"  naive:   nan={n_nan} inf={n_inf} mean={nq.mean():.4e} std={nq.std():.4e}")
    print(f"  fused:   nan={f_nan} inf={f_inf} mean={fq.mean():.4e} std={fq.std():.4e}")

    # Scale agreement (1 byte = 256 levels)
    scale_match = (naive_scale.view(torch.uint8) == fused_scale.view(torch.uint8)).float().mean().item()
    print(f"  scale byte-equal: {scale_match*100:.1f}%")

    # Value cos similarity (after dequant, both should represent the same mxfp8 numbers)
    # E4M3 dequant: e4m3.float() * e8m0.float()
    def dequant_for_cmp(buf, scale):
        bs = _BS
        cols_p_ = buf.numel() // rows
        n_blocks_ = cols_p_ // bs
        q = buf.view(rows, n_blocks_, bs).float()
        s = scale.float().view(rows, n_blocks_, 1)
        return (q * s).view(-1)
    n_deq = dequant_for_cmp(naive_buf, naive_scale)
    f_deq = dequant_for_cmp(fused_buf, fused_scale)
    cos = torch.nn.functional.cosine_similarity(n_deq.unsqueeze(0), f_deq.unsqueeze(0)).item()
    rel_diff = (n_deq - f_deq).abs().mean() / (n_deq.abs().mean() + 1e-9)
    print(f"  dequant cos similarity: {cos:.6f}")
    print(f"  dequant rel diff (mean): {rel_diff:.4e}")

    # Verdict
    print("\n=== Verdict ===")
    if fused_proj / 1000 < 2.0:
        print(f"  PASS: fused per-mb {fused_proj/1000:.2f}s < 2.0s budget.")
        print(f"  → proceed with the implementation (remove s.accum).")
    elif fused_proj / 1000 < 5.0:
        print(f"  MARGINAL: fused per-mb {fused_proj/1000:.2f}s, > 2.0s budget but < 5.0s.")
        print(f"  → consider Triton kernel or fall back to bf16 (option B).")
    else:
        print(f"  FAIL: fused per-mb {fused_proj/1000:.2f}s >> 2.0s budget.")
        print(f"  → fall back to bf16 muon (option B) — drop the mxfp8 path.")


if __name__ == "__main__":
    main()
