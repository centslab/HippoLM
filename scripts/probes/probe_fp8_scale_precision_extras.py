"""FP8 scale-precision supplementary experiments.

User feedback (2026-07-25) — three additional studies on top of
probe_scale_precision_full_2026_07_25.py:

  PART E — Mantissa-bits sweep (0..23)
    Construct a parametric scale format with N mantissa bits (round-to-nearest).
    For m=0, compare ceil/floor/round_log2 encoders. Sweep m ∈ {0, 1, 2, 3, 4,
    5, 6, 7, 8, 10, 12, 15, 18, 23} at 1x32 block granularity on synthetic +
    real activations. Plot SQNR vs m to get the diminishing-returns curve.

  PART F — 2D blocks + saturation count
    - 2D block sizes (16x16, 32x32, 64x64, 128x128) — these are spatial tiles,
      not just K-axis partitions. Measure SQNR per (block_M, block_K, scale_type).
    - Saturation count vs block size: how many FP8 elements hit ±448 for each
      block size (FP8 main weight side).
    - max_rel and p99_rel vs block size: should grow as block amax gets
      dominated by outliers (smaller relative signal per element).

  PART G — Diagonal cross-matrix
    Full (granularity × scale_format) matrix:
      granularity ∈ {per-tensor, per-row, 1x16, 1x32, 1x64, 1x128, 1xK}
      scale_format ∈ {FP32, BF16, E8M0}
    Show the "trade-off diagonal" the user identified: TS > RS ≈ MX becomes
    obvious once you put the precision × granularity on a 2D grid.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parent.parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


E4M3_MAX = 448.0
E5M2_MAX = 57344.0


# ===========================================================================
# Shared scale-format helpers
# ===========================================================================

def scale_to_mantissa_bits(s: torch.Tensor, m_bits: int, encoder: str = "round_log2") -> torch.Tensor:
    """Round FP32 scale to m_bits mantissa bits (round-to-nearest-even).

    m_bits=0:   power-of-2 only (encoder ∈ ceil/floor/round_log2)
    m_bits≥23: returns FP32 unchanged
    else:       simulated m-bit-mantissa FP format with FP32 exponent range
    """
    s = s.float()
    if m_bits >= 23:
        return s
    if m_bits <= 0:
        log2_s = torch.log2(s.abs().clamp_min(5.877e-39))
        if encoder == "ceil_log2":
            scale_exp = torch.ceil(log2_s).clamp(-127, 127)
        elif encoder == "floor_log2":
            scale_exp = torch.floor(log2_s).clamp(-127, 127)
        else:  # round_log2
            scale_exp = torch.round(log2_s).clamp(-127, 127)
        return (scale_exp + 127).to(torch.uint8).view(torch.float8_e8m0fnu).to(torch.float32) * torch.sign(s).clamp_min(1e-30)
    abs_s = s.abs()
    log2_abs = torch.log2(abs_s.clamp_min(1e-38))
    # ULP for m_bits mantissa = 2^(floor(log2|s|) - m_bits)
    exp_floor = torch.floor(log2_abs)
    ulp = torch.pow(2.0, exp_floor - m_bits)
    rounded_abs = torch.round(abs_s / ulp) * ulp
    return torch.sign(s) * rounded_abs


def scale_round_to_format(s: torch.Tensor, fmt: str) -> torch.Tensor:
    s = s.float()
    if fmt == "FP32":
        return s
    if fmt == "BF16":
        return s.to(torch.bfloat16).to(torch.float32)
    if fmt == "FP16":
        return s.to(torch.float16).to(torch.float32)
    if fmt == "E8M0":
        log2_s = torch.log2(s.abs().clamp_min(5.877e-39))
        scale_exp = torch.ceil(log2_s).clamp(-127, 127)
        out = (scale_exp + 127).to(torch.uint8).view(torch.float8_e8m0fnu).to(torch.float32)
        return out * torch.sign(s).clamp_min(1e-30)
    raise ValueError(fmt)


def quant_blockwise(v: torch.Tensor, block: int, scale_type: str, fp8_max: float, fp8_dtype):
    shape = v.shape
    v_flat = v.float().reshape(-1, shape[-1])
    N, K = v_flat.shape
    assert K % block == 0
    v_b = v_flat.reshape(N, K // block, block)
    s_ideal = v_b.abs().amax(dim=-1, keepdim=True).clamp_min(1e-6) / fp8_max
    s_stored = scale_round_to_format(s_ideal, scale_type).reshape(N, K // block, 1)
    r = (v_b / s_stored).clamp(-fp8_max, fp8_max).to(fp8_dtype)
    v_fp8 = r.reshape(shape).contiguous()
    s_full = s_stored.expand(N, K // block, block).reshape(N, K).reshape(*shape[:-1], shape[-1])
    return v_fp8, s_full


def quant_blockwise_m(s: torch.Tensor, m_bits: int, block: int, fp8_max: float, fp8_dtype, encoder: str = "round_log2"):
    """Like quant_blockwise but uses the parametric mantissa-bits scale format."""
    shape = s.shape
    s_flat = s.float().reshape(-1, shape[-1])
    N, K = s_flat.shape
    assert K % block == 0
    s_b = s_flat.reshape(N, K // block, block)
    s_ideal = s_b.abs().amax(dim=-1, keepdim=True).clamp_min(1e-6) / fp8_max
    s_stored = scale_to_mantissa_bits(s_ideal, m_bits, encoder=encoder).reshape(N, K // block, 1)
    r = (s_b / s_stored).clamp(-fp8_max, fp8_max).to(fp8_dtype)
    v_fp8 = r.reshape(shape).contiguous()
    s_full = s_stored.expand(N, K // block, block).reshape(N, K).reshape(*shape[:-1], shape[-1])
    return v_fp8, s_full


def quant_rowwise(v: torch.Tensor, scale_type: str, fp8_max: float, fp8_dtype):
    shape = v.shape
    v_flat = v.float().reshape(-1, shape[-1])
    N, K = v_flat.shape
    s_ideal = v_flat.abs().amax(dim=-1, keepdim=True).clamp_min(1e-6) / fp8_max
    s_stored = scale_round_to_format(s_ideal, scale_type)
    r = (v_flat / s_stored).clamp(-fp8_max, fp8_max).to(fp8_dtype)
    v_fp8 = r.reshape(shape).contiguous()
    s_full = s_stored.expand(N, K).reshape(*shape[:-1], shape[-1])
    return v_fp8, s_full


def quant_tensorwise(v: torch.Tensor, scale_type: str, fp8_max: float, fp8_dtype):
    s_ideal = v.float().abs().amax().clamp_min(1e-6) / fp8_max
    s_stored = scale_round_to_format(s_ideal.reshape(1, 1), scale_type).reshape(1, 1)
    r = (v.float() / s_stored).clamp(-fp8_max, fp8_max).to(fp8_dtype)
    return r.contiguous(), s_stored.expand_as(v)


def quant_2d_blockwise(v: torch.Tensor, block_m: int, block_k: int, scale_type: str,
                        fp8_max: float, fp8_dtype):
    """Quantize with 2D (M, K) blocks of size (block_m × block_k)."""
    shape = v.shape
    if v.ndim == 3:
        B, M, K = v.shape
        v_flat = v.reshape(B * M, K)
    else:
        M, K = v.shape
        B = None
        v_flat = v
    N, K2 = v_flat.shape
    assert N % block_m == 0, f"N={N} not divisible by block_m={block_m}"
    assert K2 % block_k == 0, f"K={K2} not divisible by block_k={block_k}"
    v_b = v_flat.reshape(N // block_m, block_m, K2 // block_k, block_k)
    s_ideal = v_b.abs().amax(dim=(2, 3), keepdim=True).clamp_min(1e-6) / fp8_max
    s_stored = scale_round_to_format(s_ideal, scale_type)
    r = (v_b / s_stored).clamp(-fp8_max, fp8_max).to(fp8_dtype)
    v_fp8 = r.reshape(N, K2).contiguous()
    s_full = s_stored.expand(N // block_m, block_m, K2 // block_k, block_k).reshape(N, K2)
    if B is not None:
        v_fp8 = v_fp8.reshape(B, M, K)
        s_full = s_full.reshape(B, M, K)
    return v_fp8, s_full


# ===========================================================================
# Distribution generators (same as Part A/B/D)
# ===========================================================================

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


def sqnr_per_row(v_bf: torch.Tensor, deq: torch.Tensor) -> tuple:
    v_bf = v_bf.float().reshape(-1, v_bf.shape[-1])
    deq = deq.float().reshape(-1, deq.shape[-1])
    sig = (v_bf ** 2).mean(dim=-1).clamp_min(1e-12)
    noise = ((v_bf - deq) ** 2).mean(dim=-1).clamp_min(1e-20)
    sqnr_db = 10.0 * torch.log10(sig / noise)
    return sqnr_db.median().item(), sqnr_db.mean().item()


def saturation_count(v_fp8: torch.Tensor, fp8_dtype) -> tuple:
    """Return (count_at_pos_max, count_at_neg_max, total_elements)."""
    pos_max = 448.0 if fp8_dtype == torch.float8_e4m3fn else 57344.0
    neg_max = -448.0 if fp8_dtype == torch.float8_e4m3fn else -57344.0
    v_f = v_fp8.float()
    pos = (v_f == pos_max).sum().item()
    neg = (v_f == neg_max).sum().item()
    return pos, neg, v_f.numel()


def max_rel_p99(v_bf: torch.Tensor, v_fp8: torch.Tensor, s_full: torch.Tensor) -> tuple:
    """Per-element |dequant - v| / max(|v|, rel_floor). rel_floor prevents
    spurious 100% relative errors when v ≈ 0 (where relative error is meaningless).
    rel_floor is set to 1% of the value's FP8 representation quantum so a tiny
    dequant error on a near-zero element isn't reported as 100x error."""
    deq = v_fp8.float() * s_full.float()
    v32 = v_bf.float()
    abs_v = v32.abs()
    # Floor: 1% of the typical magnitude scale (use 1% of amax of the row's block).
    # For simplicity, use 1% of |v| clipped to a global floor.
    floor = torch.clamp(abs_v * 0.01, min=1e-4)
    rel = ((deq - v32).abs() / (abs_v + floor)).reshape(-1)
    rel = rel[rel.isfinite()]
    return rel.max().item() * 100, torch.quantile(rel, 0.99).item() * 100


# ===========================================================================
# PART E — Mantissa-bits sweep
# ===========================================================================

def part_e_mantissa_sweep():
    print("=" * 100)
    print("PART E — Mantissa-bits sweep on the scale format (E[m]M0..E[m]M23 parametric)")
    print("=" * 100)
    print("For each (m_bits ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 18, 23}):")
    print("  Round FP32 scale to m mantissa bits, measure per-element + per-block SQNR")
    print("  At block size 32 (MXFP8 native) on the synthetic Gaussian distribution.")
    print("  m=0 uses ceil_log2 (MXFP8 canonical encoder). Other m use round-to-nearest.")
    print()

    M, K = 1024, 1536
    v = gen_distribution("D1_gaussian", (M, K))
    fp8_max = E4M3_MAX
    fp8_dtype = torch.float8_e4m3fn

    m_bits_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 18, 23]
    print(f"  {'m_bits':>6s}  {'med_rel':>9s}  {'p99_rel':>9s}  {'sqnr_med':>9s}  {'sqnr_mean':>9s}  "
          f"{'format_equivalent':>20s}")
    print("  " + "-" * 80)
    for m in m_bits_list:
        encoder = "ceil_log2" if m == 0 else "round_log2"
        v_fp8, s_full = quant_blockwise_m(v, m, 32, fp8_max, fp8_dtype, encoder=encoder)
        deq = v_fp8.float() * s_full.float()
        rel = ((deq - v.float()).abs() / (v.float().abs() + 1e-12)).reshape(-1)
        rel = rel[rel.isfinite()]
        med = rel.median().item() * 100
        p99 = torch.quantile(rel, 0.99).item() * 100
        sqnr_med, sqnr_avg = sqnr_per_row(v, deq)
        eq_name = ""
        if m == 0:
            eq_name = "E8M0 (UE8M0)"
        elif m == 7:
            eq_name = "BF16-equivalent"
        elif m == 10:
            eq_name = "FP16-equivalent"
        elif m == 23:
            eq_name = "FP32 (control)"
        elif m == 3:
            eq_name = "E5M3-equivalent"
        elif m == 2:
            eq_name = "E5M2-equivalent"
        print(f"  {m:>6d}  {med:8.3f}%  {p99:8.3f}%  {sqnr_med:8.2f}dB {sqnr_avg:8.2f}dB  "
              f"{eq_name:>20s}")
    print()

    # Also on outlier distribution to show the trade-off
    print("  --- Outlier distribution (D3) — same setup ---")
    v = gen_distribution("D3_outlier_rows", (M, K))
    print(f"  {'m_bits':>6s}  {'med_rel':>9s}  {'p99_rel':>9s}  {'sqnr_med':>9s}  {'sqnr_mean':>9s}")
    print("  " + "-" * 60)
    for m in m_bits_list:
        encoder = "ceil_log2" if m == 0 else "round_log2"
        v_fp8, s_full = quant_blockwise_m(v, m, 32, fp8_max, fp8_dtype, encoder=encoder)
        deq = v_fp8.float() * s_full.float()
        rel = ((deq - v.float()).abs() / (v.float().abs() + 1e-12)).reshape(-1)
        rel = rel[rel.isfinite()]
        med = rel.median().item() * 100
        p99 = torch.quantile(rel, 0.99).item() * 100
        sqnr_med, sqnr_avg = sqnr_per_row(v, deq)
        print(f"  {m:>6d}  {med:8.3f}%  {p99:8.3f}%  {sqnr_med:8.2f}dB {sqnr_avg:8.2f}dB")
    print()


# ===========================================================================
# PART F — 2D blocks + saturation count + max_rel/p99 vs block size
# ===========================================================================

def part_f_2d_blocks():
    print("=" * 100)
    print("PART F — 2D blocks, saturation count, max_rel / p99 vs block size")
    print("=" * 100)
    print("Same Gaussian + outlier distributions. For each (block_size, scale_type):")
    print("  - saturations: #elements quantized to ±448 (FP8 max), per the dequantized scale")
    print("  - max_rel, p99: per-element relative error (after FP8 + scale)")
    print("  - sqnr_med: per-row SQNR median")
    print()

    M, K = 1024, 1536
    for dist_name in ["D1_gaussian", "D3_outlier_rows"]:
        v = gen_distribution(dist_name, (M, K))
        v32 = v.float()
        fp8_max = E4M3_MAX
        fp8_dtype = torch.float8_e4m3fn

        print(f"--- Distribution: {dist_name} (amax={v32.abs().amax().item():.3f}) ---")
        print()
        for scale_type in ["BF16", "E8M0"]:
            print(f"  Scale format: {scale_type}")
            print(f"  {'block':>10s}  {'saturations':>12s}  {'max_rel':>9s}  {'p99_rel':>9s}  "
                  f"{'sqnr_med':>9s}")
            print("  " + "-" * 60)
            for block in [16, 32, 64, 128, 256, 512, 1536]:
                v_fp8, s_full = quant_blockwise(v, block, scale_type, fp8_max, fp8_dtype)
                pos, neg, total = saturation_count(v_fp8, fp8_dtype)
                sat_pct = (pos + neg) * 100.0 / total
                mx, p99 = max_rel_p99(v, v_fp8, s_full)
                deq = v_fp8.float() * s_full.float()
                sqnr_med, _ = sqnr_per_row(v, deq)
                print(f"  {block:>10d}  {sat_pct:>10.3f}%  {mx:>8.2f}%  {p99:>8.2f}%  "
                      f"{sqnr_med:>8.2f}dB")
            print()

        # 2D blocks: 16x16, 32x32, 64x64, 128x128 (when both dims divide K and M)
        print(f"  2D block sizes (BF16 scale):")
        print(f"  {'2d_block':>10s}  {'saturations':>12s}  {'max_rel':>9s}  {'p99_rel':>9s}  "
              f"{'sqnr_med':>9s}")
        print("  " + "-" * 60)
        for bm, bk in [(16, 16), (32, 32), (64, 64), (128, 128)]:
            try:
                v_fp8, s_full = quant_2d_blockwise(v, bm, bk, "BF16", fp8_max, fp8_dtype)
                pos, neg, total = saturation_count(v_fp8, fp8_dtype)
                sat_pct = (pos + neg) * 100.0 / total
                mx, p99 = max_rel_p99(v, v_fp8, s_full)
                deq = v_fp8.float() * s_full.float()
                sqnr_med, _ = sqnr_per_row(v, deq)
                print(f"  {bm}x{bk:>5d}  {sat_pct:>10.3f}%  {mx:>8.2f}%  {p99:>8.2f}%  "
                      f"{sqnr_med:>8.2f}dB")
            except AssertionError:
                print(f"  {bm}x{bk:>5d}  (skipped — not divisible)")
        print()


# ===========================================================================
# PART G — Full diagonal cross-matrix (granularity × scale_format)
# ===========================================================================

def part_g_diagonal():
    print("=" * 100)
    print("PART G — Diagonal cross-matrix: (granularity × scale_format) on real activations")
    print("=" * 100)
    print("Original study compared three 'diagonal' points:")
    print("  TS = per-tensor FP32 scale (low granularity, high scale-precision)")
    print("  RS = per-row FP32 scale    (medium granularity, high scale-precision)")
    print("  MX = 1x32 block E8M0 scale (fine granularity, low scale-precision)")
    print()
    print("The conclusion 'TS > RS ≈ MX' from sig_rel is visible in the matrix below.")
    print("This matrix fills in the missing corners:")
    print("  per-tensor × E8M0, per-row × E8M0, 1x32 × BF16, 1x32 × FP32, …")
    print()

    # Use real KDA-like activations: gaussian with realistic amax
    M, K = 1024, 1536
    v = gen_distribution("D1_gaussian", (M, K))
    fp8_max = E4M3_MAX
    fp8_dtype = torch.float8_e4m3fn

    granularities = [
        ("per-tensor", None, None),
        ("per-row", None, None),
        ("1x16", None, 16),
        ("1x32", None, 32),
        ("1x64", None, 64),
        ("1x128", None, 128),
        ("1xK", None, K),
    ]

    scale_formats = ["FP32", "BF16", "E8M0"]

    print(f"  Synthetic Gaussian D1 (M=1024 K=1536), E4M3")
    print()
    print(f"  {'granularity':>14s}  " + "  ".join(f"{s:>16s}" for s in scale_formats))
    print("  " + "-" * 75)
    for gran_name, _, block in granularities:
        row_data = []
        for sf in scale_formats:
            if gran_name == "per-tensor":
                v_fp8, s_full = quant_tensorwise(v, sf, fp8_max, fp8_dtype)
            elif gran_name == "per-row":
                v_fp8, s_full = quant_rowwise(v, sf, fp8_max, fp8_dtype)
            else:
                v_fp8, s_full = quant_blockwise(v, block, sf, fp8_max, fp8_dtype)
            deq = v_fp8.float() * s_full.float()
            sqnr_med, _ = sqnr_per_row(v, deq)
            row_data.append(f"{sqnr_med:>14.2f}dB")
        print(f"  {gran_name:>14s}  " + "  ".join(f"{d:>16s}" for d in row_data))
    print()

    # Also outlier distribution
    print(f"  --- Outlier distribution D3 ---")
    v = gen_distribution("D3_outlier_rows", (M, K))
    print()
    print(f"  {'granularity':>14s}  " + "  ".join(f"{s:>16s}" for s in scale_formats))
    print("  " + "-" * 75)
    for gran_name, _, block in granularities:
        row_data = []
        for sf in scale_formats:
            if gran_name == "per-tensor":
                v_fp8, s_full = quant_tensorwise(v, sf, fp8_max, fp8_dtype)
            elif gran_name == "per-row":
                v_fp8, s_full = quant_rowwise(v, sf, fp8_max, fp8_dtype)
            else:
                v_fp8, s_full = quant_blockwise(v, block, sf, fp8_max, fp8_dtype)
            deq = v_fp8.float() * s_full.float()
            sqnr_med, _ = sqnr_per_row(v, deq)
            row_data.append(f"{sqnr_med:>14.2f}dB")
        print(f"  {gran_name:>14s}  " + "  ".join(f"{d:>16s}" for d in row_data))
    print()


# ===========================================================================
# PART G (real-activation extension) — same matrix on real activations
# ===========================================================================

def part_g_real():
    print("=" * 100)
    print("PART G (real) — Same diagonal cross-matrix on REAL KDA activations")
    print("=" * 100)
    print("Builds a tiny HippoLM model, captures real KDA q_proj activations,")
    print("runs the full (granularity × scale_format) matrix. This is where")
    print("the user's original 'TS > RS ≈ MX' trade-off diagonal becomes visible.")
    print()

    from src.models.config import HippoConfig
    from src.models.model import HippoModel

    cfg = HippoConfig(
        hidden_size=512,
        num_heads=8,
        head_dim=64,
        num_layers=2,
        num_blocks=1,
        intermediate_size=1536,
        vocab_size=1024,
        attention_precision="w16a16",
        ffn_precision="w16a16",
    )
    model = HippoModel(cfg).cuda().to(torch.bfloat16)
    model.eval()

    captured: dict[str, torch.Tensor] = {}

    def make_hook(name):
        def hook(module, inp, out):
            captured[name] = inp[0] if isinstance(inp, tuple) else inp
        return hook

    hooks = []
    for name, module in model.named_modules():
        if type(module).__name__ in {"Linear", "FP8Linear", "NVFP4Linear", "NVFP4LinearW4A8",
                                       "MXFP8Linear", "NVFP4Marlin", "NVFP4LinearTP", "_nvfp4_marlin"}:
            hooks.append(module.register_forward_hook(make_hook(name)))

    B, T = 2, 256
    torch.manual_seed(0)
    input_ids = torch.randint(0, cfg.vocab_size, (B, T), device="cuda")
    with torch.no_grad():
        model(input_ids)
    for h in hooks:
        h.remove()

    # Pick a few representative tensors for the cross-matrix
    target_names = [
        "layers.0.kda.attn.q_proj",
        "layers.0.kda.attn.o_proj",
        "layers.0.ffn.gate_proj",
        "layers.0.ffn.down_proj",
        "lm_head",
    ]

    fp8_max = E4M3_MAX
    fp8_dtype = torch.float8_e4m3fn
    scale_formats = ["FP32", "BF16", "E8M0"]

    for name in target_names:
        if name not in captured:
            continue
        v = captured[name].contiguous()
        if v.numel() < 1024:
            continue
        K = v.shape[-1]
        valid_blocks = [b for b in [16, 32, 64, 128] if K % b == 0]
        if not valid_blocks:
            continue
        granularities = [("per-tensor", None), ("per-row", None)] + [(f"1x{b}", b) for b in valid_blocks]
        if K >= 256:
            granularities.append((f"1x{K}", K))
        print(f"--- {name}  shape={tuple(v.shape)}  amax={v.float().abs().amax().item():.3f} ---")
        print()
        print(f"  {'granularity':>14s}  " + "  ".join(f"{s:>14s}" for s in scale_formats))
        print("  " + "-" * 60)
        for gran_name, block in granularities:
            row_data = []
            for sf in scale_formats:
                if gran_name == "per-tensor":
                    v_fp8, s_full = quant_tensorwise(v, sf, fp8_max, fp8_dtype)
                elif gran_name == "per-row":
                    v_fp8, s_full = quant_rowwise(v, sf, fp8_max, fp8_dtype)
                else:
                    v_fp8, s_full = quant_blockwise(v, block, sf, fp8_max, fp8_dtype)
                deq = v_fp8.float() * s_full.float()
                sqnr_med, _ = sqnr_per_row(v, deq)
                row_data.append(f"{sqnr_med:>12.2f}dB")
            print(f"  {gran_name:>14s}  " + "  ".join(f"{d:>14s}" for d in row_data))
        print()


# ===========================================================================
# PART F (real-activation extension) — max_rel/p99 + 2D blocks on REAL activations
# ===========================================================================

def part_f_real():
    print("=" * 100)
    print("PART F (real) — max_rel / p99 + 2D blocks + weight saturation on REAL activations")
    print("=" * 100)
    print("User request 2026-07-25: extend Part F to real activations + real weights.")
    print("Two sub-parts:")
    print("  F.1 — max_rel / p99 / saturations vs block size on real KDA q_proj (BF16 scale)")
    print("       Test the user's hypothesis 'max_rel / p99 increase with block size'")
    print("  F.2 — 2D blocks (16x16, 32x32, 64x64, 128x128) on real KDA q_proj + FFN down_proj")
    print("  F.3 — Weight saturation count on real model weights (per-row vs 1x32 vs 1x128)")
    print()

    from src.models.config import HippoConfig
    from src.models.model import HippoModel

    cfg = HippoConfig(
        hidden_size=512,
        num_heads=8,
        head_dim=64,
        num_layers=2,
        num_blocks=1,
        intermediate_size=1536,
        vocab_size=1024,
        attention_precision="w16a16",
        ffn_precision="w16a16",
    )
    model = HippoModel(cfg).cuda().to(torch.bfloat16)
    model.eval()

    captured_act: dict[str, torch.Tensor] = {}
    captured_w: dict[str, torch.Tensor] = {}

    def make_hook_act(name):
        def hook(module, inp, out):
            captured_act[name] = inp[0] if isinstance(inp, tuple) else inp
        return hook

    def make_hook_w(name):
        def hook(module, inp, out):
            captured_w[name] = module.weight.detach()
        return hook

    hooks = []
    for name, module in model.named_modules():
        if type(module).__name__ in {"Linear", "FP8Linear", "NVFP4Linear", "NVFP4LinearW4A8",
                                       "MXFP8Linear", "NVFP4Marlin", "NVFP4LinearTP", "_nvfp4_marlin"}:
            hooks.append(module.register_forward_hook(make_hook_act(name)))
            hooks.append(module.register_forward_hook(make_hook_w(name)))

    B, T = 2, 256
    torch.manual_seed(0)
    input_ids = torch.randint(0, cfg.vocab_size, (B, T), device="cuda")
    with torch.no_grad():
        model(input_ids)
    for h in hooks:
        h.remove()

    fp8_max = E4M3_MAX
    fp8_dtype = torch.float8_e4m3fn

    # F.1 — max_rel / p99 / saturations vs block size on real KDA q_proj
    print("--- F.1  KDA q_proj  shape={}  amax={:.3f} ---".format(
        tuple(captured_act["layers.0.kda.attn.q_proj"].shape),
        captured_act["layers.0.kda.attn.q_proj"].float().abs().amax().item()))
    print()
    print("  BF16 scale, varying block size on REAL activations:")
    print(f"  {'block':>10s}  {'saturations':>12s}  {'max_rel':>9s}  {'p99_rel':>9s}  "
          f"{'sqnr_med':>9s}")
    print("  " + "-" * 60)
    v = captured_act["layers.0.kda.attn.q_proj"].contiguous()
    for block in [16, 32, 64, 128, 256, 512]:
        v_fp8, s_full = quant_blockwise(v, block, "BF16", fp8_max, fp8_dtype)
        pos, neg, total = saturation_count(v_fp8, fp8_dtype)
        sat_pct = (pos + neg) * 100.0 / total
        mx, p99 = max_rel_p99(v, v_fp8, s_full)
        deq = v_fp8.float() * s_full.float()
        sqnr_med, _ = sqnr_per_row(v, deq)
        print(f"  {block:>10d}  {sat_pct:>10.3f}%  {mx:>8.2f}%  {p99:>8.2f}%  "
              f"{sqnr_med:>8.2f}dB")
    print()

    # F.2 — 2D blocks on real KDA q_proj + FFN down_proj
    print("--- F.2  2D blocks on real activations ---")
    print()
    for name in ["layers.0.kda.attn.q_proj", "layers.0.ffn.down_proj"]:
        if name not in captured_act:
            continue
        v = captured_act[name].contiguous()
        if v.numel() < 1024:
            continue
        K = v.shape[-1]
        print(f"  --- {name}  shape={tuple(v.shape)}  amax={v.float().abs().amax().item():.3f} ---")
        print()
        # 1D for comparison
        print(f"  1D blocks:")
        print(f"  {'block':>10s}  {'saturations':>12s}  {'max_rel':>9s}  {'p99_rel':>9s}  "
              f"{'sqnr_med':>9s}")
        print("  " + "-" * 60)
        valid_1d = [b for b in [16, 32, 64, 128, 256, 512] if K % b == 0]
        if K == 1536:
            valid_1d.append(1536)
        for block in valid_1d:
            v_fp8, s_full = quant_blockwise(v, block, "BF16", fp8_max, fp8_dtype)
            pos, neg, total = saturation_count(v_fp8, fp8_dtype)
            sat_pct = (pos + neg) * 100.0 / total
            mx, p99 = max_rel_p99(v, v_fp8, s_full)
            deq = v_fp8.float() * s_full.float()
            sqnr_med, _ = sqnr_per_row(v, deq)
            print(f"  {block:>10d}  {sat_pct:>10.3f}%  {mx:>8.2f}%  {p99:>8.2f}%  "
                  f"{sqnr_med:>8.2f}dB")
        print()

        # 2D blocks
        M = v.shape[-2] if v.ndim >= 2 else 1
        valid_2d = [(bm, bk) for bm in [16, 32, 64, 128] for bk in [16, 32, 64, 128]
                     if M % bm == 0 and K % bk == 0]
        print(f"  2D blocks:")
        print(f"  {'2d_block':>10s}  {'saturations':>12s}  {'max_rel':>9s}  {'p99_rel':>9s}  "
              f"{'sqnr_med':>9s}")
        print("  " + "-" * 60)
        # First add per-row for comparison (covers all M x K)
        v_fp8, s_full = quant_rowwise(v, "BF16", fp8_max, fp8_dtype)
        pos, neg, total = saturation_count(v_fp8, fp8_dtype)
        sat_pct = (pos + neg) * 100.0 / total
        mx, p99 = max_rel_p99(v, v_fp8, s_full)
        deq = v_fp8.float() * s_full.float()
        sqnr_med, _ = sqnr_per_row(v, deq)
        print(f"  {'per-row':>10s}  {sat_pct:>10.3f}%  {mx:>8.2f}%  {p99:>8.2f}%  "
              f"{sqnr_med:>8.2f}dB")
        for bm, bk in valid_2d:
            v_fp8, s_full = quant_2d_blockwise(v, bm, bk, "BF16", fp8_max, fp8_dtype)
            pos, neg, total = saturation_count(v_fp8, fp8_dtype)
            sat_pct = (pos + neg) * 100.0 / total
            mx, p99 = max_rel_p99(v, v_fp8, s_full)
            deq = v_fp8.float() * s_full.float()
            sqnr_med, _ = sqnr_per_row(v, deq)
            print(f"  {bm:>3d}x{bk:<6d}  {sat_pct:>10.3f}%  {mx:>8.2f}%  {p99:>8.2f}%  "
                  f"{sqnr_med:>8.2f}dB")
        print()

    # F.3 — Weight saturation count on real model weights
    print("--- F.3  Weight saturation count on real model weights ---")
    print()
    print("  Per Linear layer's weight tensor, count % of FP8 elements at ±448.")
    print("  Different block sizes on the WEIGHT side (K dimension).")
    print()
    weight_names = ["layers.0.kda.attn.q_proj", "layers.0.kda.attn.o_proj",
                     "layers.0.ffn.gate_proj", "layers.0.ffn.down_proj", "lm_head"]
    for name in weight_names:
        if name not in captured_w:
            continue
        w = captured_w[name].contiguous()
        if w.numel() < 1024:
            continue
        K = w.shape[-1]
        print(f"  {name}  shape={tuple(w.shape)}  amax={w.float().abs().amax().item():.3f}")
        print(f"  {'block':>10s}  {'saturations':>12s}  {'sqnr_med':>9s}")
        print("  " + "-" * 40)
        valid_blocks = [b for b in [16, 32, 64, 128, 256, 512] if K % b == 0]
        if K == 1536:
            valid_blocks.append(1536)
        for block in valid_blocks:
            w_fp8, s_full = quant_blockwise(w, block, "BF16", fp8_max, fp8_dtype)
            pos, neg, total = saturation_count(w_fp8, fp8_dtype)
            sat_pct = (pos + neg) * 100.0 / total
            deq = w_fp8.float() * s_full.float()
            sqnr_med, _ = sqnr_per_row(w, deq)
            print(f"  {block:>10d}  {sat_pct:>10.3f}%  {sqnr_med:>8.2f}dB")
        print()


def main():
    print(f"GPU: {torch.cuda.get_device_name(0)} (sm_{torch.cuda.get_device_capability(0)[0]}{torch.cuda.get_device_capability(0)[1]})")
    print(f"torch: {torch.__version__}")
    print()
    part_e_mantissa_sweep()
    part_f_2d_blocks()
    part_g_diagonal()
    part_g_real()
    part_f_real()
    print("=" * 100)
    print("DONE")
    print("=" * 100)


if __name__ == "__main__":
    main()