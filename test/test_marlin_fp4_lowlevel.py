"""Low-level correctness test for the Marlin FP4 ctypes loader.

This test exists to catch regressions in the *ctypes binding layer* of
:mod:`src.models.ops.nvfp4_marlin` — distinct from the high-level
``NVFP4Linear`` integration test at :file:`test/test_ffn_nvfp4_marlin.py`.

Why a separate low-level test
-----------------------------
The high-level test only exercises the path through
``NVFP4Linear.forward → marlin_nvfp4_matmul → _MarlinNvFp4Matmul.apply``.
If a regression silently breaks ``_MarlinNvFp4Matmul.forward`` itself
(ctypes argtypes misaligned, mangled symbol renamed by torch / CUDA,
``global_scale`` exponent off-by-one, ``eff_scale`` zeroed out, …),
the integration test will still show whatever its broader assertions
happen to cover.

We pin the ctypes-layer contract directly with a tiny forward call
using the exact same arg layout as the loader expects:

  - shape:        x[M,K] BF16 -> out[M,N] BF16
  - finiteness:   no NaN/Inf
  - non-zero:     catches "eff_scale wrong → zero kernel output"
                  (one of the two latent bugs shipped in the initial
                  Marlin integration; see project memory
                  ``project_marlin_production.md``)
  - cache hit:    a second forward against the same ``packed`` buffer
                  must reach the cache and return identical output
  - cache miss:   replacing the ``packed`` buffer (simulating an
                  optimizer step / repack) must invalidate the cache
                  and the new output must match a fresh first-call

Backward contract: ``_MarlinNvFp4Matmul`` returns BF16 grads on ``x``
and ``w_master`` (STE). We assert finiteness + non-zero, which catches
the "w_master was detached" or "grad flow broken at ctypes boundary"
class of regressions.

Run:

    python -m pytest test/test_marlin_fp4_lowlevel.py -v
"""
from __future__ import annotations

import ctypes
from pathlib import Path
import sys

import pytest
import torch

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from src.models.ops import nvfp4_marlin
from src.models.ops.nvfp4_marlin import (
    _MarlinNvFp4Matmul,
    _build_marlin_scales_caches,
    _resolve_so,
    quantize_nvfp4_with_global_scale,
)


# ---------------------------------------------------------------------------
# Skips — gate the test on the hardware the kernel actually supports.
# ---------------------------------------------------------------------------
def _cuda_sm80_or_newer() -> bool:
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability()
    return major >= 8


# Skip the whole module on boxes where the kernel cannot run. Done at
# collection time (no expensive setup fires in the body of the tests).
pytestmark = pytest.mark.skipif(
    not _cuda_sm80_or_newer(),
    reason="Marlin FP4 needs CUDA sm_80+; not available on this box",
)


# ---------------------------------------------------------------------------
# Fixtures — small, deterministic, ctypes-friendly shapes.
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def tiny_quantized_weight() -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, int, int,
]:
    """Build a (weight_master, packed, scales, global_scale, K) fixture.

    ``M=K=128, N=64`` (8 K-groups of 16). Small enough that a failed
    ctypes call surfaces as a segfault or zero output immediately, not
    as a slow timeout. ``weight_master`` is BF16; the packed pair is
    produced by the real ``quantize_nvfp4_with_global_scale`` (the
    same codepath the production ``repack_weights`` uses), so the
    fixture exercises the full quantization contract.
    """
    torch.manual_seed(0)
    M = 8
    N = 64
    K = 128
    block_size = 16

    w_master = torch.randn(N, K, dtype=torch.bfloat16, device="cuda")
    packed, scales, global_scale = quantize_nvfp4_with_global_scale(
        w_master, block_size=block_size,
    )
    return w_master, packed, scales, global_scale, K, block_size, M


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_resolve_so_finds_prebuilt_for_current_arch():
    """``_resolve_so`` must locate the per-arch .so (or the legacy
    fixed-name fallback) for whatever SM is reporting on this device.
    Catches: a build script rename, an .so deleted post-build, or a
    device capability misreport (sm tag set to e.g. '121' when the
    shipped .so is '120').
    """
    kind = "kernel_only"
    path = _resolve_so(kind)
    assert Path(path).is_file(), f"_resolve_so({kind!r}) returned non-existent path: {path}"
    # Must be one of the two naming conventions.
    assert path.endswith(".so"), f"unexpected non-.so path: {path}"


def test_ensure_libs_loaded_binds_marlin_mm_symbol():
    """After ``_ensure_libs_loaded`` runs, ``_marlin_mm_fn`` and
    ``_repack_fn`` must be non-None callable ctypes objects. Catches
    a mangled-name drift (torch 2.12 / CUDA 13 may rename
    ``_ZN6marlin9marlin_mm...``) before we even try a forward call.
    """
    nvfp4_marlin._libs_loaded = False
    nvfp4_marlin._marlin_mm_fn = None
    nvfp4_marlin._repack_fn = None
    nvfp4_marlin._ensure_libs_loaded()
    assert nvfp4_marlin._marlin_mm_fn is not None, "_marlin_mm_fn is None after loader bind"
    assert nvfp4_marlin._repack_fn is not None, "_repack_fn is None after loader bind"
    # ctypes function objects expose .argtypes / .restype — the assertion
    # confirms the bind didn't fall through to a generic void* signature.
    assert nvfp4_marlin._marlin_mm_fn.argtypes is not None
    assert len(nvfp4_marlin._marlin_mm_fn.argtypes) == 35, (
        f"marlin_mm_fn should have 35 args, got {len(nvfp4_marlin._marlin_mm_fn.argtypes)}"
    )


def _make_caches_holder(
    scales: torch.Tensor,
    global_scale: torch.Tensor,
    size_k: int,
    size_n: int,
    block_size: int = 16,
) -> torch.nn.Module:
    """Build a tiny ``nn.Module`` whose ``_scales_for_kernel`` /
    ``_global_scale_adj`` attributes are populated by
    ``_build_marlin_scales_caches``. Mirrors the production
    ``NVFP4*Linear.repack_weights`` setup, but for tests that exercise
    ``_MarlinNvFp4Matmul.apply`` directly without going through the
    module wrapper.
    """
    holder = torch.nn.Module()
    _build_marlin_scales_caches(
        holder, scales, global_scale,
        size_k=size_k, size_n=size_n, block_size=block_size,
    )
    return holder


def test_marlin_fp4_forward_shape_dtype_finite_nonzero(tiny_quantized_weight):
    """Single-tensor forward invariants: shape, dtype, finiteness, non-zero.

    Catches four distinct regressions in one test:

      1. Shape contract drift (kernel writes wrong output dims).
      2. Dtype contract drift (kernel writes FP32 instead of BF16).
      3. ``eff_scale`` zeroed out (the loader bind forgets to pass
         ``global_scale_adj`` → kernel returns 0; see project memory
         ``project_marlin_production.md``).
      4. argtypes misalignment (kernel reads garbage args, returns
         NaN/Inf or segfaults).
    """
    w_master, packed, scales, global_scale, K, block_size, M = tiny_quantized_weight
    N = w_master.shape[0]
    holder = _make_caches_holder(scales, global_scale, K, N, block_size)

    x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    out = _MarlinNvFp4Matmul.apply(
        x, w_master, packed,
        holder._scales_for_kernel, holder._global_scale_adj,
        None, K, block_size,
    )

    assert out.shape == (M, w_master.shape[0]), (
        f"out.shape={tuple(out.shape)} != ({M}, {w_master.shape[0]})"
    )
    assert out.dtype == torch.bfloat16, f"out.dtype={out.dtype} != bfloat16"
    assert torch.isfinite(out).all().item(), (
        f"Marlin forward produced NaN/Inf: out[0,:4]={out[0,:4].tolist()}"
    )
    # Non-zero guard: the 'eff_scale wrong' bug shipped with the kernel
    # returned all-zero output. abs().sum() > 0 is the cheapest check.
    assert out.abs().sum().item() > 0.0, (
        "Marlin forward produced all-zero output "
        "(eff_scale / global_scale binding likely broken — see "
        "project memory project_marlin_production.md)"
    )


def test_marlin_fp4_recompute_path_is_deterministic(tiny_quantized_weight):
    """Two forward calls with the same inputs must return bit-identical
    output.

    Previously this test relied on the module-level ``_cached_repack``
    dict to skip repacking on the second call. That cache is now gone —
    the large ``repacked`` buffer is recomputed each forward (the
    ``_repack_for_marlin`` ctypes call is ~110 µs). The repack output
    must still be deterministic on the same input.
    """
    w_master, packed, scales, global_scale, K, block_size, M = tiny_quantized_weight
    N = w_master.shape[0]
    holder = _make_caches_holder(scales, global_scale, K, N, block_size)

    x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    out1 = _MarlinNvFp4Matmul.apply(
        x, w_master, packed,
        holder._scales_for_kernel, holder._global_scale_adj,
        None, K, block_size,
    )
    out2 = _MarlinNvFp4Matmul.apply(
        x, w_master, packed,
        holder._scales_for_kernel, holder._global_scale_adj,
        None, K, block_size,
    )
    assert torch.equal(out1, out2), (
        "Recompute-path Marlin forward returned different output on "
        "second call (repack + scale-processing is not deterministic)"
    )


def test_marlin_fp4_repack_rebuilds_caches(tiny_quantized_weight):
    """After re-quantizing with a new BF16 master (simulating an
    optimizer step + ``repack_weights``), the small Marlin-side
    caches (``_scales_for_kernel``, ``_global_scale_adj``) must be
    rebuilt against the new weights. Otherwise the kernel uses stale
    scales against fresh data and produces wrong output.

    This pins the contract that ``NVFP4*Linear.repack_weights``
    satisfies by calling ``_build_marlin_scales_caches`` after the
    fresh quantize.
    """
    w_master, packed, scales, global_scale, K, block_size, M = tiny_quantized_weight
    N = w_master.shape[0]
    holder = _make_caches_holder(scales, global_scale, K, N, block_size)

    x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    out_before = _MarlinNvFp4Matmul.apply(
        x, w_master, packed,
        holder._scales_for_kernel, holder._global_scale_adj,
        None, K, block_size,
    )

    # Simulate an optimizer step: change the BF16 master, re-quantize,
    # copy the new packed/scales/global_scale in place (same data_ptr
    # as before), and rebuild the small caches. This is exactly what
    # production's ``NVFP4Linear.repack_weights`` does.
    new_w = w_master + 0.1 * torch.randn_like(w_master)
    new_packed, new_scales, new_global_scale = quantize_nvfp4_with_global_scale(
        new_w, block_size=block_size,
    )
    packed.copy_(new_packed)
    scales.copy_(new_scales)
    global_scale.copy_(new_global_scale)
    _build_marlin_scales_caches(
        holder, new_scales, new_global_scale,
        size_k=K, size_n=N, block_size=block_size,
    )

    out_after = _MarlinNvFp4Matmul.apply(
        x, new_w, packed,
        holder._scales_for_kernel, holder._global_scale_adj,
        None, K, block_size,
    )
    # Output must have changed (we changed the BF16 master).
    assert not torch.equal(out_before, out_after), (
        "Forward output did not change after repack — caches were not "
        "rebuilt against the new weights, or repack_weights's "
        "_build_marlin_scales_caches hook is broken."
    )

    # Re-running with the same (now-new) state must be stable.
    out_after_repeat = _MarlinNvFp4Matmul.apply(
        x, new_w, packed,
        holder._scales_for_kernel, holder._global_scale_adj,
        None, K, block_size,
    )
    assert torch.equal(out_after, out_after_repeat), (
        "Marlin forward returned different output on repeat call after "
        "repack"
    )


def test_marlin_fp4_backward_grads_finite_nonzero(tiny_quantized_weight):
    """STE bwd: ``grad_x`` and ``grad_w`` (on ``w_master``) must be
    finite and non-zero.

    Catches:

      - autograd Function detached ``w_master`` (grad_w -> None)
      - ctypes bind changed ctx.save_for_backward layout (autograd
        bwd raises a shape error)
      - bwd math regressed to NaN (e.g. zero-row grad_out)
    """
    w_master, packed, scales, global_scale, K, block_size, M = tiny_quantized_weight
    N = w_master.shape[0]
    holder = _make_caches_holder(scales, global_scale, K, N, block_size)

    x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    w_master = w_master.detach().clone().requires_grad_(True)

    y = _MarlinNvFp4Matmul.apply(
        x, w_master, packed,
        holder._scales_for_kernel, holder._global_scale_adj,
        None, K, block_size,
    )
    y.sum().backward()

    assert x.grad is not None, "x.grad is None (autograd did not flow through x)"
    assert w_master.grad is not None, (
        "w_master.grad is None — w_master was likely detached "
        "from the autograd graph in forward()"
    )
    assert torch.isfinite(x.grad).all().item(), "x.grad has NaN/Inf"
    assert torch.isfinite(w_master.grad).all().item(), "w_master.grad has NaN/Inf"
    assert x.grad.abs().sum().item() > 0.0, "x.grad is all-zero"
    assert w_master.grad.abs().sum().item() > 0.0, "w_master.grad is all-zero"


def test_marlin_fp4_with_bias_adds_bias(tiny_quantized_weight):
    """The ctypes layer passes an empty b_bias void* slot; the
    bias add happens Python-side (``out + bias``). This test pins
    that contract: ``out[..., n]`` shifts by ``bias[n]`` exactly.
    """
    w_master, packed, scales, global_scale, K, block_size, M = tiny_quantized_weight
    N = w_master.shape[0]
    holder = _make_caches_holder(scales, global_scale, K, N, block_size)

    bias = torch.randn(w_master.shape[0], dtype=torch.bfloat16, device="cuda")
    x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")

    out_no_bias = _MarlinNvFp4Matmul.apply(
        x, w_master, packed,
        holder._scales_for_kernel, holder._global_scale_adj,
        None, K, block_size,
    )
    out_with_bias = _MarlinNvFp4Matmul.apply(
        x, w_master, packed,
        holder._scales_for_kernel, holder._global_scale_adj,
        bias, K, block_size,
    )

    # bias adds to every row of the M-dim output.
    expected = out_no_bias + bias
    assert torch.allclose(out_with_bias, expected, atol=1e-3, rtol=1e-3), (
        f"Bias-add path diverged from out+bias broadcast: "
        f"max_abs={(out_with_bias - expected).abs().max().item():.4f}"
    )
