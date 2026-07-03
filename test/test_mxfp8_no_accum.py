"""Regression test: mxfp8 muon has NO bf16 ``s.accum`` buffer.

Historical context (2026-07-02 → 2026-07-03):

  The mxfp8 muon design originally allocated a separate BF16
  ``s.accum`` buffer alongside ``s.mom_buf``, mirroring the
  int8 design's fast-per-mb trick. That made per-mb CPU sync
  cheap (~50ms) but defeated the raison d'être of mxfp8 —
  saving the 2 bytes/elt that a separate accumulator would
  have cost.

  Fix: remove ``s.accum`` for mxfp8. Per-mb accumulation flows
  through a fused C++ kernel (:func:`src.training.ops.mxfp8_accum.
  fused_mxfp8_dequant_add_requant`) that does dequant + add +
  requant in one pass directly into ``s.mom_buf``.

This test guards the design intent. It does NOT assert tight
numerical correctness — the per-mb dequant+add+requant
naturally introduces ~12% E4M3 quant error per block, and
over many gradient-accumulation steps the error compounds
until training diverges. This is exactly why mxfp8 is **not**
the production default (see ``configs/base.yml`` — muon_momentum
ships as ``bf16``). The test config ``configs/test/muon_mxfp8.yml``
still exists for memory-constrained experiments and CI smoke
coverage of the code path.

What this test asserts
----------------------
1. ``CPUMuon`` allocates ``s.accum is None`` for mxfp8 muon
   params. (The design intent of the fix.)
2. ``_accumulator_target`` returns the right ``is_mxfp8`` flag
   so the flush path dispatches through the fused C++ kernel
   instead of a plain ``target.add_(src)``.
3. ``_accumulator_target`` for int8 muon keeps ``s.accum``
   as the target and ``is_mxfp8 = False`` (int8 path unchanged).
4. ``_accumulator_target`` for bf16 muon returns ``s.mom_buf``
   as the merged-accumulator target (unchanged from prior).
5. The fused C++ kernel exists and runs without producing
   NaN/Inf on the E4M3/E8M0 path (a sanity smoke — the
   detailed kernel correctness test lives in
   ``test/_tmp/bench_mxfp8_cpp_kernel.py``).
"""
from __future__ import annotations

import sys
from pathlib import Path

# Repo on sys.path so imports resolve when run via pytest.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from src.training.param_offload import (
    CPUMuon,
    _accumulator_target,
    accumulate_grads_to_cpu,
)
from src.training.precision_config import DType, PrecisionConfig, ScaleMode, TensorPrecision


def _make_precision(mom_dtype: DType) -> PrecisionConfig:
    """Build a PrecisionConfig for the given muon_momentum dtype.

    Goes through TensorPrecision's __post_init__ so mxfp8 gets
    block_size=32 + scale='block' automatically, and int8 gets
    scale='per_channel' (the canonical muon int8 path)."""
    if mom_dtype == DType.MXFP8:
        mom_tp = TensorPrecision(dtype=mom_dtype)
    elif mom_dtype.is_integer:
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


def test_mxfp8_muon_has_no_separate_accum():
    """mxfp8 muon must NOT allocate a bf16 ``s.accum`` buffer.

    The whole point of mxfp8 is to save the 2 bytes/elt that
    ``s.accum`` would cost. Asserting ``s.accum is None`` is the
    direct guard against re-introducing the bug.
    """
    if not torch.cuda.is_available():
        import pytest
        pytest.skip("CUDA not available")

    device = "cuda"
    rows, cols = 256, 256
    p = torch.nn.Parameter(torch.randn(rows, cols, dtype=torch.bfloat16, device=device) * 0.02)
    prec = _make_precision(DType.MXFP8)
    opt = CPUMuon([p], lr=0.01, weight_decay=0.0, precision=prec)
    s = next(iter(opt.state.values()))

    # Direct check: mxfp8 must NOT have a bf16 accum.
    assert s.accum is None, (
        "mxfp8 muon re-allocated a bf16 s.accum — this is the "
        "2026-07-02 regression (wastes 2 bytes/elt of CPU pinned "
        "memory that mxfp8 was supposed to save)."
    )

    # The accumulator target for the per-mb flush must be s.mom_buf
    # (not a separate accum), with is_mxfp8=True so the flush
    # dispatches through the fused C++ kernel.
    target, cast_dtype, is_mxfp8 = _accumulator_target(s)
    assert target is s.mom_buf
    assert cast_dtype == torch.bfloat16
    assert is_mxfp8 is True, (
        "_accumulator_target must flag mxfp8 so the flush dispatches "
        "through CPUMuon._mxfp8_apply_grad (fused C++ kernel) instead "
        "of a plain target.add_(src) — that path would NaN on the "
        "dtype mismatch."
    )
    print(f"  [PASS] mxfp8 muon: s.accum=None, target=s.mom_buf, is_mxfp8=True")


def test_int8_muon_still_has_accum():
    """int8 muon still uses the bf16 ``s.accum`` path — the fix
    was scoped to mxfp8 ONLY (the user request: 'mxfp8设计的目的
    就是省内存'). int8 per-mb dequant+add+requant is the expensive
    bottleneck we keep avoiding with the bf16 accum trick.
    """
    if not torch.cuda.is_available():
        import pytest
        pytest.skip("CUDA not available")

    device = "cuda"
    rows, cols = 256, 256
    p = torch.nn.Parameter(torch.randn(rows, cols, dtype=torch.bfloat16, device=device) * 0.02)
    prec = _make_precision(DType.INT8)
    opt = CPUMuon([p], lr=0.01, weight_decay=0.0, precision=prec)
    s = next(iter(opt.state.values()))

    assert s.accum is not None, (
        "int8 muon must keep its bf16 s.accum — the fix was scoped "
        "to mxfp8 only."
    )
    target, cast_dtype, is_mxfp8 = _accumulator_target(s)
    assert target is s.accum
    assert is_mxfp8 is False
    print(f"  [PASS] int8 muon: s.accum is bf16, fast per-mb path unchanged")


def test_bf16_momentum_uses_merged_accum():
    """bf16 momentum uses the merged-accumulator design: ``s.mom_buf``
    doubles as the per-mb accumulator. No s.accum, no fused kernel —
    just a plain bf16 .add_() per mb.

    This was the previously-stable behavior and must NOT change
    with the mxfp8 redesign.
    """
    if not torch.cuda.is_available():
        import pytest
        pytest.skip("CUDA not available")

    device = "cuda"
    rows, cols = 256, 256
    p = torch.nn.Parameter(torch.randn(rows, cols, dtype=torch.bfloat16, device=device) * 0.02)
    prec = _make_precision(DType.BF16)
    opt = CPUMuon([p], lr=0.01, weight_decay=0.0, precision=prec)
    s = next(iter(opt.state.values()))

    assert s.accum is None
    target, cast_dtype, is_mxfp8 = _accumulator_target(s)
    assert target is s.mom_buf
    assert cast_dtype == torch.bfloat16
    assert is_mxfp8 is False
    print(f"  [PASS] bf16 momentum: s.accum=None, target=s.mom_buf, merged design")


def test_mxfp8_fused_kernel_runs_and_is_finite():
    """Smoke: the fused C++ kernel populates s.mom_buf with E4M3
    values and s.mom_scale with E8M0 scales, and the dequantized
    result is finite (no NaN from a missed E8M0=0 div-by-zero or
    from a wrong sign bit in the kernel's bitcast).

    Detailed kernel correctness (rel-diff vs PyTorch reference,
    sign-bit regression, etc.) lives in
    ``test/_tmp/bench_mxfp8_cpp_kernel.py``. This test only
    guards the dispatch + finiteness of the production path.
    """
    if not torch.cuda.is_available():
        import pytest
        pytest.skip("CUDA not available")

    device = "cuda"
    torch.manual_seed(0)
    rows, cols = 256, 256

    p = torch.nn.Parameter(torch.randn(rows, cols, dtype=torch.bfloat16, device=device) * 0.02)
    prec = _make_precision(DType.MXFP8)
    opt = CPUMuon([p], lr=0.01, weight_decay=0.0, precision=prec)
    s = next(iter(opt.state.values()))

    p.grad = torch.randn(rows, cols, dtype=torch.bfloat16, device=device) * 0.01
    accumulate_grads_to_cpu([opt], sync_device=0)

    # The mom_buf dtype must remain E4M3 (the kernel didn't write
    # back as bf16 — that would mean the dispatch is wrong).
    assert s.mom_buf.dtype == torch.float8_e4m3fn
    assert s.mom_scale.dtype == torch.float8_e8m0fnu

    # Finiteness: a NaN here means the E8M0=0 init didn't take
    # (or a sign-bit regression slipped in). NOT a tight rel-diff
    # check — that lives in the kernel bench.
    dequant = opt._dequantize_mxfp8(s)
    assert torch.isfinite(dequant).all(), (
        "fused mxfp8 kernel produced NaN/Inf — likely E8M0=0 "
        "on an unfilled scale entry or a sign-bit regression."
    )
    # And the dequantized scale is non-zero (E8M0 byte 1 = 2^-126
    # is the safe floor — never zero).
    assert dequant.abs().max() > 0, "dequant is all zero — kernel wrote nothing"
    print(f"  [PASS] fused kernel: finite dequant, max-abs={dequant.abs().max().item():.3e}")


if __name__ == "__main__":
    print("=== mxfp8 no-accum design regression ===\n")
    test_mxfp8_muon_has_no_separate_accum()
    test_int8_muon_still_has_accum()
    test_bf16_momentum_uses_merged_accum()
    test_mxfp8_fused_kernel_runs_and_is_finite()
    print("\nALL PASSED")