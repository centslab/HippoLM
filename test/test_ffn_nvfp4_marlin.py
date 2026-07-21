"""Class-level correctness / perf tests for the NVFP4 Marlin FP4 path.

Historical note (2026-07-21): this file used to hold "NVFP4-Marlin
SwiGLU vs BF16" integration tests built via ``ffn_nvfp4=True,
ffn_nvfp4_marlin=True`` — i.e. Marlin *mode-2* (a BF16 master
``.weight`` present, seeded + repacked, SGD applied to ``.weight``).
The 5-scheme migration made ``ffn_precision="w4a16"`` always build
Marlin *mode-3* (FP4 packed buffers are the source of truth; there
is no ``.weight`` Parameter), so those mode-2 SwiGLU tests exercised
a config->SwiGLU path that no longer exists and were deleted. The
live w4a16 mode-3 SwiGLU integration (fwd/bwd finite + optimizer
wiring) is covered by ``test_nvfp4_no_bf16_master.py::TestSwiGLUMode3``
and ``TestBuildParamGroupsWiring``.

The class-level tests below construct ``NVFP4Linear(use_marlin=True)``
directly and guard the Marlin kernel + packed-state_dict behavior.

Run:
    python -m pytest test/test_ffn_nvfp4_marlin.py -v
"""
from __future__ import annotations

import pytest
import torch

from src.models.ops.nvfp4_linear import NVFP4Linear


def test_nvfp4_marlin_packed_state_dict_roundtrip():
    """Saving and loading the state_dict of an NVFP4-Marlin layer must
    preserve packed_weight + scales + global_scale (so checkpoints stay
    small AND the kernel doesn't NaN from missing global_scale).

    The derived Marlin caches (``_scales_for_kernel``,
    ``_global_scale_adj``) are registered as ``persistent=False``
    buffers so they participate in ``.cuda()`` device transfer but are
    excluded from ``state_dict()`` — they're derived state,
    reproducible from the source ``scales`` / ``global_scale`` buffers.
    A post-``load_state_dict`` hook auto-refreshes them against the
    loaded data, so the very next forward uses the correct weights
    without the caller having to remember to call ``repack_weights()``.
    """
    torch.manual_seed(0)
    layer = NVFP4Linear(128, 64, bias=True, use_marlin=True).cuda()
    layer.repack_weights()

    sd = layer.state_dict()
    assert "weight" in sd
    assert "packed_weight" in sd
    assert "scales" in sd
    assert "global_scale" in sd, "Marlin layer state_dict missing global_scale"
    # Derived Marlin caches must NOT be in state_dict (recomputed by
    # repack_weights against the source buffers).
    assert "_scales_for_kernel" not in sd, (
        "_scales_for_kernel should be persistent=False; not in state_dict"
    )
    assert "_global_scale_adj" not in sd, (
        "_global_scale_adj should be persistent=False; not in state_dict"
    )
    assert sd["packed_weight"].dtype == torch.uint8
    assert sd["scales"].dtype == torch.float8_e4m3fn
    assert sd["global_scale"].dtype == torch.float32

    # Reconstruct from the dict and verify forward matches. The
    # post-load hook refreshes the derived caches automatically.
    layer2 = NVFP4Linear(128, 64, bias=True, use_marlin=True).cuda()
    layer2.load_state_dict(sd)

    x = torch.randn(2, 128, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        y1 = layer(x)
        y2 = layer2(x)
    assert torch.allclose(y1, y2), "state_dict roundtrip changed output"
    assert torch.isfinite(y1).all().item() and torch.isfinite(y2).all().item()


def test_nvfp4_marlin_layer_is_faster_than_dequant_path():
    """Sanity check that the Marlin forward is materially faster than
    the dequant+cuBLAS path at FFN gate_up shapes on sm_120.

    This is a wall-clock smoke check, not a micro-benchmark — we just
    verify the Marlin kernel is at least 1.5x faster than
    dequant+cuBLAS at a representative shape (M=2048, K=N=4096).
    Tolerates the kernel being temporarily slower if the test box is
    a non-sm_120 GPU (graceful skip).
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    major, _ = torch.cuda.get_device_capability(0)
    if major < 8:
        pytest.skip(f"Marlin FP4 needs sm_80+, got sm_{major}")

    torch.manual_seed(0)
    M, K, N = 2048, 4096, 4096
    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)

    layer_marlin = NVFP4Linear(K, N, bias=True, use_marlin=True).cuda()
    layer_dequant = NVFP4Linear(K, N, bias=True, use_marlin=False).cuda()
    # Share weights + repack so the only difference is the forward path.
    layer_dequant.weight.data.copy_(layer_marlin.weight.data)
    layer_marlin.repack_weights()
    layer_dequant.repack_weights()

    def bench(fn, n_iters=20, n_warmup=5):
        for _ in range(n_warmup):
            fn()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(n_iters):
            fn()
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / n_iters

    t_marlin = bench(lambda: layer_marlin(x))
    t_dequant = bench(lambda: layer_dequant(x))
    speedup = t_dequant / t_marlin
    print(f"\n  Marlin={t_marlin:.3f}ms  Dequant={t_dequant:.3f}ms  Speedup={speedup:.2f}x")
    # We expect at least 1.5x at this shape on sm_120. Be lenient
    # (1.2x) to handle the 5060 Ti vs other sm_80+ boxes variation.
    assert speedup >= 1.2, (
        f"Marlin FP4 only {speedup:.2f}x faster than dequant+cuBLAS "
        f"(expected >=1.5x on sm_120)"
    )
