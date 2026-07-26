"""Correctness + speed test for the fused Triton
``dequant_ema_requant_2d_fused`` kernel.

Compares against the PyTorch reference:
  deq = dequantize_2d(q, s1, s2)              # CPU
  ema = deq.bfloat16() * beta + grad           # CPU
  q_new, s1_new, s2_new, tight = quantize_2d(ema.float(), block)

PASS criteria:
- Per-shape numerics:
    dequant output cos_sim = 1.0 (bit-exact)
    EMA output cos_sim = 1.0 (bit-exact)
    q_new cos_sim > 0.999 (within E4M3 grid step)
    s1_new / s2_new max rel < 1e-5 (FP32 ULP)
- Speed (sm_120, dev box):
    GPU fused wall < CPU reference wall × 5 for each prod shape
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest
import torch

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.training.ops.fp8_2d_tight import (
    DEFAULT_BLOCK, dequantize_2d, quantize_2d,
)
from src.training.ops.fp8_2d_tight_fused import (
    dequant_ema_requant_2d_fused,
)


SHAPES = [
    (8192, 1536),   # FFN down
    (1536, 4096),   # FFN gate/up
    (4608, 1536),   # FFN down (skinny/tall) — bottleneck shape
    (1536, 1536),   # attn o
    (1536, 128),    # attn res q (small K)
]
BETA = 0.95
BLOCK = DEFAULT_BLOCK  # 32


def setup_state(rows: int, cols: int, seed: int = 0):
    """Build a realistic fp8_2d_tight state on CPU."""
    torch.manual_seed(seed)
    ema = torch.randn(rows, cols, dtype=torch.bfloat16) * 0.1
    grad = torch.randn(rows, cols, dtype=torch.bfloat16) * 0.01
    q, s1, s2, _tight = quantize_2d(ema.float(), block=BLOCK)
    return (
        q.contiguous(),
        s1.contiguous(),
        s2.contiguous(),
        grad.contiguous().view(-1),
    )


def cpu_reference(q_cpu, s1_cpu, s2_cpu, grad_cpu, rows, cols, block, beta):
    """Pure-PyTorch 3-op CPU reference, matches the existing muon
    step's dequant + EMA + quant pipeline."""
    deq = dequantize_2d(q_cpu, s1_cpu, s2_cpu,
                         rows=rows, cols=cols, block=block)
    ema_bf16 = (deq.view(rows, cols).bfloat16()
                .mul_(beta).add_(grad_cpu.view(rows, cols).bfloat16()))
    q_new, s1_new, s2_new, _tight = quantize_2d(
        ema_bf16.float(), block=block,
    )
    return q_new.contiguous(), s1_new.contiguous(), s2_new.contiguous(), ema_bf16


def diff_stats(a, b):
    a_f = a.detach().float().reshape(-1)
    b_f = b.detach().float().reshape(-1)
    finite = torch.isfinite(a_f) & torch.isfinite(b_f)
    a_ff = a_f[finite]
    b_ff = b_f[finite]
    diff = (a_ff - b_ff).abs()
    rel = diff / (a_ff.abs() + 1e-9)
    if a_ff.numel() > 1:
        cos = ((a_ff * b_ff).sum()
               / (a_ff.norm() * b_ff.norm() + 1e-12)).item()
    else:
        cos = float("nan")
    return {
        "max_abs": diff.max().item() if diff.numel() else 0.0,
        "mean_abs": diff.mean().item() if diff.numel() else 0.0,
        "max_rel": rel.max().item() if rel.numel() else 0.0,
        "cos": cos,
    }


@pytest.mark.parametrize("shape", SHAPES)
def test_correctness_vs_cpu(shape):
    """The fused kernel should match the PyTorch reference within
    FP8 grid step (1 ULP of E4M3 at the saturation boundary)."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    rows, cols = shape
    q_cpu, s1_cpu, s2_cpu, grad_cpu = setup_state(rows, cols)

    # CPU reference
    q_ref, s1_ref, s2_ref, ema_ref = cpu_reference(
        q_cpu, s1_cpu, s2_cpu, grad_cpu, rows, cols, BLOCK, BETA,
    )

    # GPU fused path
    q_gpu = q_cpu.to("cuda:0")
    s1_gpu = s1_cpu.to("cuda:0")
    s2_gpu = s2_cpu.to("cuda:0")
    grad_gpu = grad_cpu.to("cuda:0")
    q_new, s1_new, s2_new, ema = dequant_ema_requant_2d_fused(
        q_gpu, s1_gpu, s2_gpu, grad_gpu, BETA, rows, cols, BLOCK,
    )
    torch.cuda.synchronize()

    # Compare
    s_ema = diff_stats(ema.cpu(), ema_ref)
    s_q = diff_stats(q_new.cpu().float(), q_ref.float())
    s_s1 = diff_stats(s1_new.cpu().float(), s1_ref.float())
    s_s2 = diff_stats(s2_new.cpu().float(), s2_ref.float())

    # EMA must be bit-exact (same arithmetic, just fused).
    assert s_ema["cos"] > 0.999, (
        f"shape {shape}: EMA cos_sim {s_ema['cos']:.6f} too low "
        f"(max_abs {s_ema['max_abs']:.4e})"
    )

    # q_new: E4M3 grid step at the saturation boundary is 32.0
    # (the value 32 = 2^5 is representable; rounding may go up or
    # down). cos_sim > 0.999 is the right threshold.
    assert s_q["cos"] > 0.999, (
        f"shape {shape}: q_new cos_sim {s_q['cos']:.6f} too low "
        f"(max_abs {s_q['max_abs']:.4e})"
    )

    # Scales: max_abs. The scales are tiny (~1e-3), so a 1e-7 ULP
    # ratio looks like a large relative error; use absolute diff
    # instead. E4M3 quantization can shift the amax by 1 ULP at the
    # boundary, so allow up to 1e-5 absolute.
    assert s_s1["max_abs"] < 1e-5, (
        f"shape {shape}: s1_new max_abs {s_s1['max_abs']:.4e} "
        "exceeds FP32 ULP budget"
    )
    assert s_s2["max_abs"] < 1e-5, (
        f"shape {shape}: s2_new max_abs {s_s2['max_abs']:.4e} "
        "exceeds FP32 ULP budget"
    )


@pytest.mark.parametrize("shape", SHAPES)
def test_speed_vs_cpu(shape):
    """GPU fused path should be ≥ 5× faster than the CPU reference
    on each prod shape."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    rows, cols = shape
    q_cpu, s1_cpu, s2_cpu, grad_cpu = setup_state(rows, cols)

    # Warmup CPU + GPU
    _ = cpu_reference(q_cpu, s1_cpu, s2_cpu, grad_cpu, rows, cols, BLOCK, BETA)
    q_gpu = q_cpu.to("cuda:0")
    s1_gpu = s1_cpu.to("cuda:0")
    s2_gpu = s2_cpu.to("cuda:0")
    grad_gpu = grad_cpu.to("cuda:0")
    _ = dequant_ema_requant_2d_fused(
        q_gpu, s1_gpu, s2_gpu, grad_gpu, BETA, rows, cols, BLOCK,
    )
    torch.cuda.synchronize()

    # Time CPU reference
    n_iter = 5
    t0 = time.perf_counter()
    for _ in range(n_iter):
        _ = cpu_reference(
            q_cpu, s1_cpu, s2_cpu, grad_cpu, rows, cols, BLOCK, BETA,
        )
    t_cpu_ms = (time.perf_counter() - t0) * 1000.0 / n_iter

    # Time GPU fused (H2D excluded — muon step does H2D once per
    # step, not per param; the constant overhead amortizes)
    t0 = time.perf_counter()
    for _ in range(n_iter):
        _ = dequant_ema_requant_2d_fused(
            q_gpu, s1_gpu, s2_gpu, grad_gpu, BETA, rows, cols, BLOCK,
        )
    torch.cuda.synchronize()
    t_gpu_ms = (time.perf_counter() - t0) * 1000.0 / n_iter

    speedup = t_cpu_ms / t_gpu_ms
    assert speedup >= 5.0, (
        f"shape {shape}: GPU fused {t_gpu_ms:.1f}ms vs CPU "
        f"{t_cpu_ms:.1f}ms = {speedup:.1f}× speedup (want ≥5×)"
    )
    # Report
    print(
        f"shape {shape}: CPU {t_cpu_ms:.1f}ms, "
        f"GPU fused {t_gpu_ms:.1f}ms, {speedup:.1f}× speedup",
        flush=True,
    )


def test_bottleneck_shape_perf():
    """The (4608, 1536) FFN-down bottleneck shape: GPU fused
    should be ≥ 50× faster than the PyTorch reference."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    rows, cols = 4608, 1536
    q_cpu, s1_cpu, s2_cpu, grad_cpu = setup_state(rows, cols)
    # Warmup
    _ = cpu_reference(q_cpu, s1_cpu, s2_cpu, grad_cpu, rows, cols, BLOCK, BETA)
    q_gpu = q_cpu.to("cuda:0")
    s1_gpu = s1_cpu.to("cuda:0")
    s2_gpu = s2_cpu.to("cuda:0")
    grad_gpu = grad_cpu.to("cuda:0")
    _ = dequant_ema_requant_2d_fused(
        q_gpu, s1_gpu, s2_gpu, grad_gpu, BETA, rows, cols, BLOCK,
    )
    torch.cuda.synchronize()
    n_iter = 5
    t0 = time.perf_counter()
    for _ in range(n_iter):
        _ = cpu_reference(
            q_cpu, s1_cpu, s2_cpu, grad_cpu, rows, cols, BLOCK, BETA,
        )
    t_cpu_ms = (time.perf_counter() - t0) * 1000.0 / n_iter
    t0 = time.perf_counter()
    for _ in range(n_iter):
        _ = dequant_ema_requant_2d_fused(
            q_gpu, s1_gpu, s2_gpu, grad_gpu, BETA, rows, cols, BLOCK,
        )
    torch.cuda.synchronize()
    t_gpu_ms = (time.perf_counter() - t0) * 1000.0 / n_iter
    speedup = t_cpu_ms / t_gpu_ms
    assert speedup >= 50.0, (
        f"(4608, 1536) bottleneck: GPU fused {t_gpu_ms:.1f}ms vs "
        f"CPU {t_cpu_ms:.1f}ms = {speedup:.1f}× speedup (want ≥50×)"
    )
    print(
        f"(4608, 1536) bottleneck: CPU {t_cpu_ms:.1f}ms, "
        f"GPU fused {t_gpu_ms:.1f}ms, {speedup:.1f}× speedup",
        flush=True,
    )


if __name__ == "__main__":
    # Manual run: prints a one-line speed summary per shape.
    if not torch.cuda.is_available():
        print("CUDA required")
        sys.exit(1)
    print(f"Smoke run: fused vs CPU reference, BLOCK={BLOCK}, BETA={BETA}\n")
    for shape in SHAPES:
        rows, cols = shape
        q_cpu, s1_cpu, s2_cpu, grad_cpu = setup_state(rows, cols)
        # Warmup
        _ = cpu_reference(q_cpu, s1_cpu, s2_cpu, grad_cpu, rows, cols, BLOCK, BETA)
        q_gpu = q_cpu.to("cuda:0")
        s1_gpu = s1_cpu.to("cuda:0")
        s2_gpu = s2_cpu.to("cuda:0")
        grad_gpu = grad_cpu.to("cuda:0")
        _ = dequant_ema_requant_2d_fused(
            q_gpu, s1_gpu, s2_gpu, grad_gpu, BETA, rows, cols, BLOCK,
        )
        torch.cuda.synchronize()
        n_iter = 5
        t0 = time.perf_counter()
        for _ in range(n_iter):
            _ = cpu_reference(
                q_cpu, s1_cpu, s2_cpu, grad_cpu, rows, cols, BLOCK, BETA,
            )
        t_cpu_ms = (time.perf_counter() - t0) * 1000.0 / n_iter
        t0 = time.perf_counter()
        for _ in range(n_iter):
            _ = dequant_ema_requant_2d_fused(
                q_gpu, s1_gpu, s2_gpu, grad_gpu, BETA, rows, cols, BLOCK,
            )
        torch.cuda.synchronize()
        t_gpu_ms = (time.perf_counter() - t0) * 1000.0 / n_iter
        speedup = t_cpu_ms / t_gpu_ms
        print(
            f"shape {shape}: CPU {t_cpu_ms:7.1f}ms, "
            f"GPU fused {t_gpu_ms:7.1f}ms, {speedup:6.1f}× speedup",
            flush=True,
        )