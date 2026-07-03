"""Benchmark the C++ fused mxfp8 kernel vs the PyTorch path.

Verifies:
1. Numerical correctness vs the PyTorch reference (dequant+add+requant
   as separate ops).
2. Per-mb wall-clock at the 1.65B-equivalent scale.
3. The savings from not needing the BF16 accum buffer.

If the C++ kernel hits <2s/mb at 1.65B, the design (option A) is
viable. If not, fall back to option B (bf16 muon).
"""
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from src.training.ops.mxfp8_accum import (
    fused_mxfp8_dequant_add_requant,
    requantize_mxfp8,
)

# OCP MX: E4M3 max magnitude.
_E4M3_MAX = 448.0
_BS = 32


def _round_to_e8m0(scale_fp32: torch.Tensor) -> torch.Tensor:
    safe = scale_fp32.clamp(min=2 ** -127)
    log2 = safe.log2()
    e_unclamped = log2.round() + 127.0
    e = e_unclamped.clamp(min=1.0, max=254.0)
    return e.to(torch.uint8).view(torch.float8_e8m0fnu)


def pytorch_reference(
    mom_buf: torch.Tensor,
    mom_scale: torch.Tensor,
    grad: torch.Tensor,
    rows: int,
    cols_p: int,
) -> None:
    """The PyTorch reference: separate dequant + add + requant ops."""
    bs = _BS
    n_blocks = cols_p // bs
    q = mom_buf.view(rows, n_blocks, bs).to(torch.bfloat16)
    # mom_scale is [rows, n_blocks] (matches the project's
    # scale_shape_fn = lambda shape: (shape[0], (shape[1] + bs - 1) // bs))
    s = mom_scale.view(rows, n_blocks).to(torch.bfloat16).unsqueeze(-1)
    dequant_bf16 = q * s
    grad_bf16 = grad.view(rows, n_blocks, bs).to(torch.bfloat16)
    sum_bf16 = dequant_bf16 + grad_bf16
    absmax = sum_bf16.abs().amax(dim=-1).float()
    target_scale = (absmax / _E4M3_MAX).clamp(min=2 ** -127)
    new_scales = _round_to_e8m0(target_scale)
    scale_bf16 = new_scales.to(torch.bfloat16).view(rows, n_blocks, 1)
    scaled_bf16 = (sum_bf16 / scale_bf16).clamp(-_E4M3_MAX, _E4M3_MAX)
    q_e4m3 = scaled_bf16.to(torch.float8_e4m3fn)
    mom_buf.copy_(q_e4m3.reshape(-1).view(mom_buf.shape))
    mom_scale.copy_(new_scales.view(mom_scale.shape))


def main():
    print("=== mxfp8 C++ fused kernel benchmark ===\n")

    # Warmup the kernel (JIT compile is done, just first call)
    rows, cols = 1024, 1024
    cols_p = cols
    n = rows * cols_p
    print("Warmup...")
    for _ in range(3):
        buf = torch.zeros(n, dtype=torch.float8_e4m3fn).pin_memory()
        # Use raw byte construction (e8m0 0x7F = 1.0 = 2^0)
        scale_uint8 = torch.full((rows, cols_p // _BS), 127, dtype=torch.uint8)
        scale = scale_uint8.view(torch.float8_e8m0fnu).pin_memory()
        g = torch.randn(n, dtype=torch.bfloat16).pin_memory() * 0.001
        fused_mxfp8_dequant_add_requant(buf, scale, g, rows, cols)
    print("  done\n")

    # Test correctness
    print("Correctness vs PyTorch reference:")
    rows, cols = 256, 256
    cols_p = cols
    n = rows * cols_p
    n_blocks_per_row = cols_p // _BS
    torch.manual_seed(0)
    # Per-block scales: each block gets its own absmax → scale.
    init_vals_2d = torch.randn(rows, cols_p, dtype=torch.bfloat16) * 0.01
    init_blocks = init_vals_2d.view(rows, n_blocks_per_row, _BS)
    init_absmax = init_blocks.abs().amax(dim=-1).float()  # [rows, n_blocks_per_row]
    target_scale = (init_absmax / _E4M3_MAX).clamp(min=2**-127)
    init_scale = _round_to_e8m0(target_scale)  # [rows, n_blocks_per_row] e8m0
    # IMPORTANT: write e8m0 BYTES via uint8 view, not via the
    # float constructor (which would quantize 112 → 134, etc).
    init_scale = init_scale.view(torch.uint8).view(torch.float8_e8m0fnu)
    init_buf = (init_blocks / init_scale.float().unsqueeze(-1)) \
                .clamp(-_E4M3_MAX, _E4M3_MAX).to(torch.float8_e4m3fn) \
                .reshape(-1)

    grad_vals = torch.randn(n, dtype=torch.bfloat16) * 0.005

    # PyTorch reference
    buf_p = init_buf.clone().pin_memory()
    scale_p = init_scale.clone().pin_memory()
    g_p = grad_vals.clone().pin_memory()
    pytorch_reference(buf_p, scale_p, g_p, rows, cols_p)

    # C++ kernel
    buf_c = init_buf.clone().pin_memory()
    scale_c = init_scale.clone().pin_memory()
    g_c = grad_vals.clone().pin_memory()
    fused_mxfp8_dequant_add_requant(buf_c, scale_c, g_c, rows, cols)

    # Compare
    def dequant(buf, scale, rows, cols_p):
        bs = _BS
        nb = cols_p // bs
        q = buf.view(rows, nb, bs).to(torch.bfloat16).float()
        s = scale.view(rows, nb, 1).float()
        return (q * s).reshape(-1)
    nq = dequant(buf_p, scale_p, rows, cols_p)
    cq = dequant(buf_c, scale_c, rows, cols_p)
    n_nan = torch.isnan(nq).sum().item()
    c_nan = torch.isnan(cq).sum().item()
    n_inf = torch.isinf(nq).sum().item()
    c_inf = torch.isinf(cq).sum().item()
    cos = torch.nn.functional.cosine_similarity(nq.unsqueeze(0), cq.unsqueeze(0)).item()
    rel_diff = (nq - cq).abs().mean() / (nq.abs().mean() + 1e-9)
    scale_match = (scale_p.view(torch.uint8) == scale_c.view(torch.uint8)).float().mean().item()
    print(f"  PyTorch:  nan={n_nan} inf={n_inf}")
    print(f"  C++:       nan={c_nan} inf={c_inf}")
    print(f"  cos sim:   {cos:.6f}")
    print(f"  rel diff:  {rel_diff:.4e}")
    print(f"  scale match: {scale_match*100:.1f}%")
    assert n_nan == c_nan and n_inf == c_inf, "NaN/Inf mismatch"
    assert rel_diff < 0.1, f"rel diff too large: {rel_diff}"
    assert scale_match > 0.9, f"scale agreement too low: {scale_match}"
    print(f"  PASS\n")

    # Per-mb wall-clock at 1.65B-equivalent scale
    print("Per-mb wall-clock (single big param, projected to 1.65B):")
    rows, cols = 1024, 1024 * 200  # 200M elts per single param
    cols_p = cols
    n = rows * cols_p
    buf = torch.zeros(n, dtype=torch.float8_e4m3fn).pin_memory()
    # Use raw byte construction (e8m0 0x7F = 1.0)
    scale_uint8 = torch.full((rows, cols_p // _BS), 127, dtype=torch.uint8)
    scale = scale_uint8.view(torch.float8_e8m0fnu).pin_memory()
    g = torch.randn(n, dtype=torch.bfloat16).pin_memory() * 0.001

    # C++ fused
    for _ in range(2):
        fused_mxfp8_dequant_add_requant(buf, scale, g, rows, cols)
    times = []
    for _ in range(3):
        t0 = time.perf_counter()
        fused_mxfp8_dequant_add_requant(buf, scale, g, rows, cols)
        times.append(time.perf_counter() - t0)
    cpp_ms = min(times) * 1000

    # PyTorch reference
    for _ in range(2):
        pytorch_reference(buf, scale, g, rows, cols_p)
    times = []
    for _ in range(3):
        t0 = time.perf_counter()
        pytorch_reference(buf, scale, g, rows, cols_p)
        times.append(time.perf_counter() - t0)
    pt_ms = min(times) * 1000

    print(f"  C++ fused:   min={cpp_ms:8.1f} ms (per call)")
    print(f"  PyTorch:     min={pt_ms:8.1f} ms (per call)")
    print(f"  C++ speedup: {pt_ms/cpp_ms:.2f}x")

    # Project to 1.65B (8 calls per mb at 200M elts each)
    n_per_param = 200e6
    n_total = 1.65e9
    calls_per_mb = n_total / n_per_param
    cpp_proj = cpp_ms * calls_per_mb / 1000
    pt_proj = pt_ms * calls_per_mb / 1000
    print(f"\n  Projected per-mb at 1.65B scale ({calls_per_mb:.1f} params × 200M elts):")
    print(f"    C++ fused:  {cpp_proj:.2f} s/mb  (×16 mbs = {cpp_proj*16:.1f} s/step)")
    print(f"    PyTorch:    {pt_proj:.2f} s/mb  (×16 mbs = {pt_proj*16:.1f} s/step)")

    # Per-elt cost
    per_elt_ns = cpp_ms * 1e6 / n
    print(f"\n  C++ per-elt cost: {per_elt_ns:.0f} ns/elt")

    # Verdict
    print("\n=== Verdict ===")
    if cpp_proj < 2.0:
        print(f"  PASS: C++ per-mb {cpp_proj:.2f}s < 2.0s budget.")
        print(f"  → mxfp8 design is viable WITHOUT a separate accum buffer.")
        print(f"  → memory saving: 2 bytes/elt (no accum) = {n_total * 2 / 1024**3:.1f} GB saved at 1.65B.")
    elif cpp_proj < 5.0:
        print(f"  MARGINAL: C++ per-mb {cpp_proj:.2f}s.")
        print(f"  → consider SIMD optimization (AVX-512) or fall back to bf16.")
    else:
        print(f"  FAIL: C++ per-mb {cpp_proj:.2f}s >> 2.0s.")
        print(f"  → fall back to bf16 muon (option B).")


if __name__ == "__main__":
    main()
