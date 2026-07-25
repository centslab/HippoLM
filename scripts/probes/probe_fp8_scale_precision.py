"""FP8 scale-format precision study — per-element decomposition.

User question (2026-07-25):
  1. Why does swapping E8M0 for BF16 scale cut average error so much?
  2. Marginal returns from each extra mantissa bit in the scale format
  3. How much does block size change error
  4. Per-element error distribution under each
     (scale_type × granularity × fp8_format) combination

This script measures (1)-(4) directly on synthetic data, then sanity-checks
the headline numbers with a GEMM-level pass.

Three error sources, measured separately so we can attribute:
  A. Scale rounding error  — quantizing the *scale* value to its
     storage format (E8M0/BF16/FP16/FP32), no FP8 in the picture.
     Measures the format's intrinsic precision.
  B. FP8 representation error — perfect scale, just FP8 round-trip.
     Measures the FP8 format's intrinsic precision (E4M3 / E5M2).
  C. Combined error — full pipeline (compute scale → round to format →
     divide → FP8 round → multiply back by rounded scale).

Granularity sweep:
  Per-tensor (1 scale total), per-row (M scales), per-block-K
  (K/BLOCK scales per row) with BLOCK ∈ {1, 16, 32, 64, 128, 256}.

Distributions (per `probe_fp8_scale_mode_distributions_2026_07_24.py`):
  D1 gaussian       — baseline
  D2 heavy_tail     — post-silu_mul-like
  D3 outlier_rows   — 0.5% rows have 10× amax
  D4 post_rmsnorm   — unit-variance
  D5 attn_logit     — wide dynamic range
  D6 block_const    — column-block-constant

Per run: med_rel / p95 / p99 / max_rel / SQNR for each
(scale_type × granularity × fp8_format) tuple.

Note on E8M0: ceil(log2) is the MXFP8 encoder; floor(log2) is also
sweep-able here. ceil(log2) ALWAYS oversizes the scale (over-cover),
floor(log2) ALWAYS undersizes (saturation risk).
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Callable

import torch

_REPO = Path(__file__).resolve().parent.parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


# ===========================================================================
# Constants
# ===========================================================================
E4M3_MAX = 448.0
E5M2_MAX = 57344.0
SCALE_TYPES = ["FP32", "FP16", "BF16", "E8M0"]
GRANULARITIES = ["tensor", "row", "block1", "block16", "block32", "block64", "block128", "block256"]
FP8_FORMATS = ["e4m3", "e5m2"]
DIST_NAMES = ["D1_gaussian", "D2_heavy", "D3_outlier_rows", "D4_post_rmsnorm", "D5_attn_logit", "D6_block_const"]


# ===========================================================================
# Part A — Scale-only precision (no FP8 quantization)
# ===========================================================================
# This measures how much error is introduced *just by rounding the scale
# value to its storage format*. 1000 random scale values drawn from a
# realistic amax distribution (0.001 .. 10.0 covers both KDA and FFN).

def scale_round_to_format(s: torch.Tensor, fmt: str, encoder: str = "ceil_log2") -> torch.Tensor:
    """Round a tensor of scales to a given storage format. encoder ∈
    {ceil_log2, floor_log2, round_log2} for E8M0; ignored otherwise."""
    s = s.float()
    if fmt == "FP32":
        return s
    if fmt == "BF16":
        return s.to(torch.bfloat16).to(torch.float32)
    if fmt == "FP16":
        return s.to(torch.float16).to(torch.float32)
    if fmt == "E8M0":
        # ceil_log2 is the canonical MXFP8 encoder (over-cover).
        # floor_log2 / round_log2 are alternatives.
        log2_s = torch.log2(s.clamp_min(5.877e-39))
        if encoder == "ceil_log2":
            scale_exp = torch.ceil(log2_s).clamp(-127, 127)
        elif encoder == "floor_log2":
            scale_exp = torch.floor(log2_s).clamp(-127, 127)
        elif encoder == "round_log2":
            scale_exp = torch.round(log2_s).clamp(-127, 127)
        else:
            raise ValueError(encoder)
        scale_u8 = (scale_exp + 127).to(torch.uint8)
        scale_f32 = scale_u8.view(torch.float8_e8m0fnu).to(torch.float32)
        return scale_f32
    raise ValueError(fmt)


def part_a_scale_only():
    print("=" * 88)
    print("PART A — Scale-only precision (no FP8 in the picture)")
    print("=" * 88)
    print("Question: 'What is the precision of the scale FORMAT itself?'")
    print("Method:  draw 100k random scales from log-uniform[1e-3, 10],")
    print("         round to each format, measure relative error.")
    print()

    # Generate scales from a log-uniform distribution (matches typical amax distributions).
    log_lo, log_hi = math.log10(1e-3), math.log10(10.0)
    log_uniform = torch.rand(100_000, dtype=torch.float64) * (log_hi - log_lo) + log_lo
    scales = (10.0 ** log_uniform).to(torch.float32).cuda()

    print(f"{'scale_format':14s}  {'encoder':12s}  {'med_rel':>10s}  {'p95_rel':>10s}  "
          f"{'p99_rel':>10s}  {'max_rel':>10s}  {'bits_eq':>10s}")
    print("-" * 88)

    # The "true" scales are already FP32, so FP32 is the reference.
    for fmt in SCALE_TYPES:
        if fmt == "E8M0":
            for enc in ["ceil_log2", "floor_log2", "round_log2"]:
                rounded = scale_round_to_format(scales, fmt, encoder=enc)
                rel = ((rounded - scales).abs() / scales.abs()).float()
                bits = 0  # E8M0 has no mantissa bits
                print(f"{fmt:14s}  {enc:12s}  "
                      f"{rel.median().item()*100:9.3f}%  "
                      f"{torch.quantile(rel, 0.95).item()*100:9.3f}%  "
                      f"{torch.quantile(rel, 0.99).item()*100:9.3f}%  "
                      f"{rel.max().item()*100:9.3f}%  "
                      f"{bits:10d}")
        else:
            rounded = scale_round_to_format(scales, fmt)
            rel = ((rounded - scales).abs() / scales.abs()).float()
            if fmt == "BF16":
                bits = 7  # 7-bit mantissa
            elif fmt == "FP16":
                bits = 10
            else:
                bits = 23
            print(f"{fmt:14s}  {'-':12s}  "
                  f"{rel.median().item()*100:9.3f}%  "
                  f"{torch.quantile(rel, 0.95).item()*100:9.3f}%  "
                  f"{torch.quantile(rel, 0.99).item()*100:9.3f}%  "
                  f"{rel.max().item()*100:9.3f}%  "
                  f"{bits:10d}")

    print()
    print("Interpretation: 'mantissa_bits' is the number of effective mantissa bits.")
    print("E8M0 has 0 mantissa bits → next representable value is 2x away (up to ~100% err).")
    print("BF16 has 7 → roughly 1/128 = 0.78% per-scale error.")
    print("FP16 has 10 → roughly 1/1024 = 0.10% per-scale error.")
    print()


# ===========================================================================
# Part B — Per-element full pipeline (FP8 + scale rounding)
# ===========================================================================
# For each value v in a synthetic tensor:
#   1. Compute ideal scale s_ideal = max_abs(block) / fp8_max
#   2. Compute s_stored = scale_round_to_format(s_ideal, scale_type)
#   3. Compute v_fp8 = round_to_fp8(v / s_stored)
#   4. Compute v_dequant = v_fp8 * s_stored
#   5. rel_err = |v_dequant - v| / |v|

def gen_distribution(name: str, shape: tuple, seed: int = 42, device="cuda"):
    g = torch.Generator(device=device).manual_seed(seed)
    M, K = shape
    if name == "D1_gaussian":
        return torch.randn(M, K, dtype=torch.bfloat16, device=device, generator=g) * 0.1
    if name == "D2_heavy":
        b = torch.randn(M, K, dtype=torch.bfloat16, device=device, generator=g) * 0.05
        tail = (b.abs() > 1.5).to(torch.bfloat16) * b * 3.0
        return b + tail
    if name == "D3_outlier_rows":
        b = torch.randn(M, K, dtype=torch.bfloat16, device=device, generator=g) * 0.05
        row_mask = (torch.rand(M, 1, dtype=torch.bfloat16, device=device, generator=g) < 0.005).to(torch.bfloat16)
        return b + b * row_mask * 10.0
    if name == "D4_post_rmsnorm":
        return torch.randn(M, K, dtype=torch.bfloat16, device=device, generator=g) * 0.5
    if name == "D5_attn_logit":
        small = torch.randn(M, K, dtype=torch.bfloat16, device=device, generator=g)
        big = torch.randn(M, K, dtype=torch.bfloat16, device=device, generator=g) * 5.0
        mask = (torch.rand(M, K, dtype=torch.bfloat16, device=device, generator=g) < 0.10).to(torch.bfloat16)
        return small * (1 - mask) + big * mask
    if name == "D6_block_const":
        b = torch.randn(M, K, dtype=torch.bfloat16, device=device, generator=g) * 0.1
        b = b.reshape(M, K // 32, 32)
        b[:] = b.mean(dim=-1, keepdim=True).expand_as(b)
        return b.reshape(M, K)
    raise ValueError(name)


def quant_blockwise(v: torch.Tensor, block: int, scale_type: str,
                    fp8_max: float, fp8_dtype: torch.dtype) -> tuple:
    """Compute per-block scale (FP32), round to scale_type, then quantize
    each element to FP8. Returns (v_fp8, s_stored) so the caller can
    dequantize and measure error."""
    M, K = v.shape
    assert K % block == 0
    v32 = v.float().reshape(M, K // block, block)
    # Ideal per-block scale (FP32, max-abs)
    s_ideal = v32.abs().amax(dim=-1, keepdim=True).clamp_min(1e-6) / fp8_max  # [M, K/B, 1]
    s_stored = scale_round_to_format(s_ideal, scale_type).reshape(M, K // block, 1)
    # Quantize
    r = (v32 / s_stored).clamp(-fp8_max, fp8_max)
    v_fp8 = r.to(fp8_dtype).reshape(M, K)
    s_full = s_stored.expand(M, K // block, block).reshape(M, K)
    return v_fp8, s_full


def quant_rowwise(v: torch.Tensor, scale_type: str, fp8_max: float, fp8_dtype: torch.dtype):
    M, K = v.shape
    v32 = v.float()
    s_ideal = v32.abs().amax(dim=-1, keepdim=True).clamp_min(1e-6) / fp8_max  # [M, 1]
    s_stored = scale_round_to_format(s_ideal, scale_type)
    r = (v32 / s_stored).clamp(-fp8_max, fp8_max)
    v_fp8 = r.to(fp8_dtype)
    s_full = s_stored.expand(M, K)
    return v_fp8, s_full


def quant_tensorwise(v: torch.Tensor, scale_type: str, fp8_max: float, fp8_dtype: torch.dtype):
    s_ideal = v.float().abs().amax().clamp_min(1e-6) / fp8_max
    s_stored = scale_round_to_format(s_ideal.unsqueeze(0).unsqueeze(0), scale_type).reshape(1, 1)
    r = (v.float() / s_stored).clamp(-fp8_max, fp8_max)
    v_fp8 = r.to(fp8_dtype)
    s_full = s_stored.expand_as(v)
    return v_fp8, s_full


def part_b_per_element():
    print("=" * 88)
    print("PART B — Per-element + per-block SQNR (FP8 + scale rounding)")
    print("=" * 88)
    print("Two metrics:")
    print("  med_rel: per-element |dequant-v|/|v|, median (sensitive to scale rounding)")
    print("  SQNR (dB): 10*log10(signal_power / noise_power) per block averaged")
    print("             Higher = better. SQNR=30dB ≈ 3.2% RMS error.")
    print()

    # Use a fixed shape to make the table readable
    M, K = 1024, 1536  # KDA prod shape

    def sqnr_per_block(v32, deq, block):
        """Compute SQNR per K-block, then average. v32, deq: [M, K] FP32."""
        M, K = v32.shape
        v_b = v32.reshape(M, K // block, block)
        d_b = deq.reshape(M, K // block, block)
        sig = (v_b ** 2).mean(dim=-1).clamp_min(1e-12)
        noise = ((v_b - d_b) ** 2).mean(dim=-1).clamp_min(1e-20)
        sqnr_db = 10.0 * torch.log10(sig / noise)
        return sqnr_db.median().item(), sqnr_db.mean().item()

    for fp8_name in FP8_FORMATS:
        fp8_dtype = torch.float8_e4m3fn if fp8_name == "e4m3" else torch.float8_e5m2
        fp8_max = E4M3_MAX if fp8_name == "e4m3" else E5M2_MAX
        print(f"--- FP8 format: {fp8_name.upper()} (max={fp8_max}) ---")
        print()

        for dist_name in DIST_NAMES:
            v = gen_distribution(dist_name, (M, K))
            v32 = v.float()
            print(f"  Distribution: {dist_name} (v.amax={v32.abs().amax().item():.3f})")

            # SQNR per block (using BF16 as the "good scale" baseline for block size effect)
            print(f"    {'scale':6s}  {'granularity':10s}  {'med_rel':>9s}  {'p99_rel':>9s}  "
                  f"{'sqnr_med':>9s}  {'sqnr_mean':>9s}")
            print(f"    {'-'*6}  {'-'*10}  {'-'*9}  {'-'*9}  {'-'*9}  {'-'*9}")
            for scale_type in SCALE_TYPES:
                for gran_name, gran_fn, block in [
                    ("tensor", quant_tensorwise, None),
                    ("row", quant_rowwise, None),
                    ("block1", quant_blockwise, 1),
                    ("block16", quant_blockwise, 16),
                    ("block32", quant_blockwise, 32),
                    ("block64", quant_blockwise, 64),
                    ("block128", quant_blockwise, 128),
                    ("block256", quant_blockwise, 256),
                ]:
                    if gran_fn == quant_tensorwise:
                        v_fp8, s = gran_fn(v, scale_type, fp8_max, fp8_dtype)
                    elif gran_fn == quant_rowwise:
                        v_fp8, s = gran_fn(v, scale_type, fp8_max, fp8_dtype)
                    else:
                        v_fp8, s = gran_fn(v, block, scale_type, fp8_max, fp8_dtype)
                    deq = v_fp8.float() * s
                    rel = ((deq - v32).abs() / (v32.abs() + 1e-12)).reshape(-1)
                    rel = rel[rel.isfinite()]
                    med_rel = rel.median().item() * 100
                    p99_rel = torch.quantile(rel, 0.99).item() * 100
                    # SQNR per-block (block=32 is reference; just use a fixed block for all)
                    sqnr_m, sqnr_avg = sqnr_per_block(v32, deq, 32)
                    print(f"    {scale_type:6s}  {gran_name:10s}  {med_rel:8.3f}%  {p99_rel:8.3f}%  "
                          f"{sqnr_m:8.2f}dB {sqnr_avg:8.2f}dB")
            print()


# ===========================================================================
# Part C — End-to-end GEMM-level sanity check (E4M3 only, prod shape)
# ===========================================================================

def part_c_gemm():
    print("=" * 88)
    print("PART C — GEMM-level sig_rel (dequant+matmul simulation, prod KDA shape)")
    print("=" * 88)
    print("torch._scaled_mm only accepts FP32 scales for RowWise mode, so we")
    print("simulate the GEMM by dequantizing both operands with the chosen scale")
    print("format and multiplying in BF16. The result is mathematically equivalent")
    print("to a hardware MMA path that consumed the same FP8 + scale.")
    print()

    M, K, N = 1024, 1536, 1536
    g = torch.Generator(device="cuda").manual_seed(42)

    a_bf = torch.randn(M, K, dtype=torch.bfloat16, device="cuda", generator=g) * 0.1
    w_bf = torch.randn(N, K, dtype=torch.bfloat16, device="cuda", generator=g) * 0.05
    ref = torch.nn.functional.linear(a_bf, w_bf)

    fp8_max = E4M3_MAX
    fp8_dtype = torch.float8_e4m3fn

    def gemm_sim(scale_type: str, granularity: str, block: int = 32):
        # Quantize A and B with the chosen scale format and granularity.
        if granularity == "tensor":
            a_q, a_s = quant_tensorwise(a_bf, scale_type, fp8_max, fp8_dtype)
            w_q, w_s = quant_tensorwise(w_bf, scale_type, fp8_max, fp8_dtype)
        elif granularity == "row":
            a_q, a_s = quant_rowwise(a_bf, scale_type, fp8_max, fp8_dtype)
            w_q, w_s = quant_rowwise(w_bf, scale_type, fp8_max, fp8_dtype)
        else:
            a_q, a_s = quant_blockwise(a_bf, block, scale_type, fp8_max, fp8_dtype)
            w_q, w_s = quant_blockwise(w_bf, block, scale_type, fp8_max, fp8_dtype)
        # Dequantize both to FP32 (or BF16 to match hardware accumulator).
        a_deq = (a_q.float() * a_s.float()).to(torch.bfloat16)
        w_deq = (w_q.float() * w_s.float()).to(torch.bfloat16)
        return torch.nn.functional.linear(a_deq, w_deq)

    # For each (scale_type, granularity), run the GEMM and report sig_rel vs BF16 reference
    print(f"  Shape: M={M} K={K} N={N}, Gaussian std=0.1/0.05")
    print()
    print(f"  {'scale':6s}  {'granularity':12s}  {'sig_rel':>9s}  {'cos_sim':>8s}  "
          f"{'med_abs':>9s}  {'max_abs':>9s}")
    print("  " + "-" * 65)
    for scale_type in SCALE_TYPES:
        for gran, blk in [("tensor", 0), ("row", 0), ("block16", 16), ("block32", 32),
                           ("block64", 64), ("block128", 128)]:
            try:
                out = gemm_sim(scale_type, gran, block=blk or 32)
                diff = (out.float() - ref.float()).abs()
                sig_rel = diff.max().item() / (ref.float().abs().max().item() + 1e-9)
                cos = torch.nn.functional.cosine_similarity(
                    out.reshape(-1).float(), ref.reshape(-1).float(), dim=0).item()
                med_abs = diff.median().item()
                max_abs = diff.max().item()
                print(f"  {scale_type:6s}  {gran:12s}  {sig_rel*100:8.3f}%  {cos:8.6f}  "
                      f"{med_abs:8.4f}  {max_abs:8.4f}")
            except Exception as e:
                print(f"  {scale_type:6s}  {gran:12s}  ERR: {e}")
    print()

    # Repeat for the outlier-heavy distribution to show the scale effect more dramatically
    print("  --- Outlier-heavy distribution (D3: 0.5% rows ×10 amax) ---")
    print()
    g2 = torch.Generator(device="cuda").manual_seed(42)
    a_bf = torch.randn(M, K, dtype=torch.bfloat16, device="cuda", generator=g2) * 0.05
    mask = (torch.rand(M, 1, dtype=torch.bfloat16, device="cuda", generator=g2) < 0.005).to(torch.bfloat16)
    a_bf = a_bf + a_bf * mask * 10.0
    w_bf = torch.randn(N, K, dtype=torch.bfloat16, device="cuda", generator=g2) * 0.025
    ref = torch.nn.functional.linear(a_bf, w_bf)
    print(f"  {'scale':6s}  {'granularity':12s}  {'sig_rel':>9s}  {'cos_sim':>8s}  "
          f"{'med_abs':>9s}  {'max_abs':>9s}")
    print("  " + "-" * 65)
    for scale_type in SCALE_TYPES:
        for gran, blk in [("tensor", 0), ("row", 0), ("block16", 16), ("block32", 32),
                           ("block64", 64), ("block128", 128)]:
            try:
                # Inline rebuild for outlier distribution
                if gran == "tensor":
                    a_q, a_s = quant_tensorwise(a_bf, scale_type, fp8_max, fp8_dtype)
                    w_q, w_s = quant_tensorwise(w_bf, scale_type, fp8_max, fp8_dtype)
                elif gran == "row":
                    a_q, a_s = quant_rowwise(a_bf, scale_type, fp8_max, fp8_dtype)
                    w_q, w_s = quant_rowwise(w_bf, scale_type, fp8_max, fp8_dtype)
                else:
                    a_q, a_s = quant_blockwise(a_bf, blk, scale_type, fp8_max, fp8_dtype)
                    w_q, w_s = quant_blockwise(w_bf, blk, scale_type, fp8_max, fp8_dtype)
                a_deq = (a_q.float() * a_s.float()).to(torch.bfloat16)
                w_deq = (w_q.float() * w_s.float()).to(torch.bfloat16)
                out = torch.nn.functional.linear(a_deq, w_deq)
                diff = (out.float() - ref.float()).abs()
                sig_rel = diff.max().item() / (ref.float().abs().max().item() + 1e-9)
                cos = torch.nn.functional.cosine_similarity(
                    out.reshape(-1).float(), ref.reshape(-1).float(), dim=0).item()
                med_abs = diff.median().item()
                max_abs = diff.max().item()
                print(f"  {scale_type:6s}  {gran:12s}  {sig_rel*100:8.3f}%  {cos:8.6f}  "
                      f"{med_abs:8.4f}  {max_abs:8.4f}")
            except Exception as e:
                print(f"  {scale_type:6s}  {gran:12s}  ERR: {e}")
    print()


# ===========================================================================
# Part D — Focused comparison (the user's headline question)
# ===========================================================================

def part_d_headline():
    print("=" * 88)
    print("PART D — Direct E8M0 vs BF16 comparison (user's headline question)")
    print("=" * 88)
    print("Decompose the error difference between E8M0 (MXFP8 default) and BF16")
    print("(blockwise Triton fallback) at 1x32 block size, across distributions.")
    print()

    M, K = 1024, 1536
    for dist_name in DIST_NAMES:
        v = gen_distribution(dist_name, (M, K))
        print(f"  Distribution: {dist_name} (v.amax={v.float().abs().amax().item():.3f})")

        for fp8_name in FP8_FORMATS:
            fp8_dtype = torch.float8_e4m3fn if fp8_name == "e4m3" else torch.float8_e5m2
            fp8_max = E4M3_MAX if fp8_name == "e4m3" else E5M2_MAX

            print(f"    {fp8_name.upper()}  ", end="")
            for scale_type in SCALE_TYPES:
                v_fp8, s = quant_blockwise(v, 32, scale_type, fp8_max, fp8_dtype)
                deq = v_fp8.float() * s
                rel = ((deq - v.float()).abs() / (v.float().abs() + 1e-12)).reshape(-1)
                med = rel.median().item() * 100
                p99 = torch.quantile(rel, 0.99).item() * 100
                print(f" {scale_type}=med{med:5.2f}/p99{p99:5.2f}", end="")
            print()


# ===========================================================================
# Main
# ===========================================================================

def main():
    print(f"GPU: {torch.cuda.get_device_name(0)} (sm_{torch.cuda.get_device_capability(0)[0]}{torch.cuda.get_device_capability(0)[1]})")
    print(f"torch: {torch.__version__}")
    print()

    part_a_scale_only()
    part_b_per_element()
    part_d_headline()
    part_c_gemm()

    print("=" * 88)
    print("DONE")
    print("=" * 88)


if __name__ == "__main__":
    main()