"""Regression test for the per-mb CPU sync bottleneck.

Bug: with ``muon_momentum: mxfp8`` (or int8), the merged-accumulator
design forces a full dequant-add-requant on the CPU pinned momentum
buffer for every microbatch. At prod scale (32 layers, ~232M muon
elements) this costs ~32s per mb — the dominant cost in
``flush_pending_grads``.

Fix: a separate bf16 ``accum`` buffer holds the per-mb grad sum
across microbatches; ``mom_buf`` is only updated at ``step()``
time. The per-mb path becomes ``accum.add_(grad_bf16)`` (cheap
CPU bf16 add), eliminating the dequant/requant cycle.

This test asserts the per-mb CPU work for mxfp8 muon is within a
budget that lets ``flush_pending_grads`` complete in <2s for the
8-layer smoke config (was ~8s on the bug). It also asserts the
final mom_buf and param match a reference (bf16 momentum) to
within quant-noise, so the redesign doesn't change correctness.

NOT a full training-loop test — just the per-mb + per-step
work in CPUMuon.
"""
import os
import sys
import time
from pathlib import Path

# Repo on sys.path so imports resolve
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from src.training.param_offload import (
    CPUMuon,
    _accumulator_target,
    accumulate_grads_to_cpu,
)
from src.training.precision_config import DType, PrecisionConfig, ScaleMode, TensorPrecision


def _make_precision(mom_dtype: DType) -> PrecisionConfig:
    """Build a PrecisionConfig with the given muon_momentum dtype.
    Goes through TensorPrecision's post_init so block_size auto-
    defaults to 32 for mxfp8 and scale mode defaults to per-
    channel for int8 (matches the canonical training config)."""
    if mom_dtype == DType.MXFP8:
        mom_tp = TensorPrecision(dtype=mom_dtype)  # post_init sets block_size=32, scale='block'
    elif mom_dtype.is_integer:
        # int8 needs an explicit scale mode. Use per-channel
        # (per-row for 2D weights) — the canonical muon int8
        # path; matches the int8 default in test_precision.py
        # and the original CPUMuon __init__.
        mom_tp = TensorPrecision(dtype=mom_dtype, scale=ScaleMode.PER_CHANNEL)
    else:
        mom_tp = TensorPrecision(dtype=mom_dtype)
    return PrecisionConfig(
        model_weights=TensorPrecision(dtype=DType.BF16),
        gradients=TensorPrecision(dtype=DType.BF16),
        activations=TensorPrecision(dtype=DType.BF16),
        muon_momentum=mom_tp,
        adamw_m=TensorPrecision(dtype=DType.BF16),
        adamw_v=TensorPrecision(dtype=DType.BF16),
    )


def _build_param(rows: int, cols: int, dtype: torch.dtype = torch.bfloat16):
    p = torch.nn.Parameter(torch.randn(rows, cols, dtype=dtype) * 0.02)
    p._mock_id = id(p)  # marker
    return p


def test_mxfp8_per_mb_flush_is_fast():
    """At 8-layer prod-ish scale, the per-mb CPU work for mxfp8
    muon must be <2s — close to bf16 baseline. Pre-fix it was ~8s.
    """
    if not torch.cuda.is_available():
        import pytest
        pytest.skip("CUDA not available")
    device = "cuda"

    # Smoke scale: 8 layers × ~26M muon elements per layer ≈ 26M
    # Use a single big param to keep the test simple and measure
    # the per-mb cost clearly.
    n_layers = 8
    rows, cols = 1536, 4096  # typical gate/up_proj shape
    n = rows * cols

    params = [_build_param(rows, cols).to(device) for _ in range(n_layers)]

    prec = _make_precision(DType.MXFP8)
    opt = CPUMuon(params, lr=0.01, weight_decay=0.0, precision=prec)

    # Simulate gradient_accumulation_steps=16 microbatches per step.
    n_mbs = 16
    grads = [torch.randn_like(p) * 0.001 for p in params]

    # Per-mb: simulate the offload hook (D2H + accumulate). We
    # bypass the hook machinery and call accumulate_grads_to_cpu
    # directly (which also calls flush_pending_grads internally
    # when sync_device is set). This is exactly what the
    # per-mb hot path does in production.
    per_mb_times = []
    for mb_i in range(n_mbs):
        # Fake a backward: set .grad on each param
        for p, g in zip(params, grads):
            p.grad = g.clone()
        t0 = time.perf_counter()
        accumulate_grads_to_cpu([opt], sync_device=0)
        per_mb_times.append(time.perf_counter() - t0)

    avg_ms = sum(per_mb_times) / len(per_mb_times) * 1000
    p95_ms = sorted(per_mb_times)[int(0.95 * len(per_mb_times))] * 1000

    # Pre-fix: avg ~7000ms, p95 ~7500ms (CPU dequant-add-requant).
    # Post-fix: avg ~50ms, p95 ~80ms (just bf16 .add_()).
    # Threshold set generously above bf16 baseline (~30ms avg) to
    # catch a regression without flaking on noisy CI.
    assert avg_ms < 2000, (
        f"per-mb flush avg={avg_ms:.1f}ms exceeds 2000ms budget — "
        f"the mxfp8 dequant-add-requant per-mb CPU bottleneck is "
        f"back. Pre-fix this was ~7000ms."
    )
    print(f"  [PASS] per-mb flush avg={avg_ms:.1f}ms p95={p95_ms:.1f}ms")


def test_mxfp8_final_param_close_to_bf16():
    """After a step, the post-step param for mxfp8 muon should
    match the bf16-muon post-step param to within quantization
    error. The redesign should NOT change the algorithmic
    output (only the per-mb path). Compare against a bf16-
    momentum CPUMuon reference.

    NOTE: we compare ``p.data`` post-step rather than the
    momentum buffer, because the bf16 path's ``mom_buf`` is
    reset to zero at step end (it's consumed by the
    orthogonalize and discarded). Only the mxfp8 path keeps
    ``mom_buf`` populated for state_dict observability — and
    even there, it's the quantized representation of the
    cycle's grad sum, not the update that was applied. The
    param itself is the same object both paths updated in
    place, so it's the right thing to compare.
    """
    if not torch.cuda.is_available():
        import pytest
        pytest.skip("CUDA not available")
    device = "cuda"

    torch.manual_seed(0)
    rows, cols = 256, 256
    n = rows * cols
    n_mbs = 4

    def run_with(dtype: DType) -> torch.Tensor:
        # Fresh param per run for isolation. Same init seed
        # → same starting params, same grads.
        torch.manual_seed(42)
        p = torch.nn.Parameter(torch.randn(rows, cols, dtype=torch.bfloat16, device=device) * 0.02)
        prec = _make_precision(dtype)
        opt = CPUMuon([p], lr=0.01, weight_decay=0.0, precision=prec)

        for _ in range(n_mbs):
            p.grad = torch.randn_like(p) * 0.01
            accumulate_grads_to_cpu([opt], sync_device=0)
        opt.step()
        return p.data.detach().float().cpu()

    ref = run_with(DType.BF16)
    mxfp8 = run_with(DType.MXFP8)

    # bf16 is the "exact" reference. mxfp8 has quantization
    # error in the grad-sum storage that feeds NS — E4M3 has a
    # 3-bit mantissa plus per-block E8M0 scale rounding.
    # The NS update itself happens in FP16, identical for both
    # paths. Relative diff should be well under 5% (the mxfp8
    # error is bounded by ~3% from quantization).
    diff = (mxfp8 - ref).abs().max()
    rel = diff / ref.abs().max()
    assert rel < 0.05, (
        f"mxfp8 final param diverges from bf16 by rel={rel:.4f} (>5%). "
        f"Redesign changed the algorithm — should be numerics-only."
    )
    print(f"  [PASS] mxfp8 vs bf16 post-step param rel_diff={rel:.4f}")


def test_int8_per_mb_flush_is_fast():
    """Same as test_mxfp8_per_mb_flush_is_fast but for int8 storage.
    Same fix applies — separate accum for any quantized storage.
    """
    if not torch.cuda.is_available():
        import pytest
        pytest.skip("CUDA not available")
    device = "cuda"

    n_layers = 8
    rows, cols = 1536, 4096
    params = [_build_param(rows, cols).to(device) for _ in range(n_layers)]

    prec = _make_precision(DType.INT8)
    opt = CPUMuon(params, lr=0.01, weight_decay=0.0, precision=prec)

    n_mbs = 16
    grads = [torch.randn_like(p) * 0.001 for p in params]

    per_mb_times = []
    for mb_i in range(n_mbs):
        for p, g in zip(params, grads):
            p.grad = g.clone()
        t0 = time.perf_counter()
        accumulate_grads_to_cpu([opt], sync_device=0)
        per_mb_times.append(time.perf_counter() - t0)

    avg_ms = sum(per_mb_times) / len(per_mb_times) * 1000
    assert avg_ms < 2000, (
        f"int8 per-mb flush avg={avg_ms:.1f}ms exceeds budget. "
        f"Same bottleneck as mxfp8."
    )
    print(f"  [PASS] int8 per-mb flush avg={avg_ms:.1f}ms")


def test_bf16_momentum_unchanged():
    """The redesign should NOT change the bf16 momentum path —
    that's the previously-stable behavior. With bf16 momentum,
    there's no separate accum; mom_buf still doubles as the
    accumulator (no per-mb dequant needed).
    """
    if not torch.cuda.is_available():
        import pytest
        pytest.skip("CUDA not available")
    device = "cuda"

    torch.manual_seed(0)
    rows, cols = 256, 256
    p = torch.nn.Parameter(torch.randn(rows, cols, dtype=torch.bfloat16, device=device) * 0.02)
    prec = _make_precision(DType.BF16)
    opt = CPUMuon([p], lr=0.01, weight_decay=0.0, precision=prec)
    s = next(iter(opt.state.values()))

    # Per-mb: target = s.mom_buf (no separate accum), bf16 cast
    target, dtype = _accumulator_target(s)
    assert target is s.mom_buf
    assert dtype == torch.bfloat16
    print(f"  [PASS] bf16 momentum uses merged accum (target=mom_buf)")


if __name__ == "__main__":
    print("=== mxfp8 muon per-mb flush perf ===")
    test_mxfp8_per_mb_flush_is_fast()
    print("\n=== int8 muon per-mb flush perf ===")
    test_int8_per_mb_flush_is_fast()
    print("\n=== bf16 momentum unchanged ===")
    test_bf16_momentum_unchanged()
    print("\n=== mxfp8 final param matches bf16 reference ===")
    test_mxfp8_final_param_close_to_bf16()
    print("\nALL PASSED")
