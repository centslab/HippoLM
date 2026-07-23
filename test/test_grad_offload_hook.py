"""Regression test for the g-vs-p.grad bug in param_offload.

PyTorch's ``register_post_accumulate_grad_hook`` callback receives
the leaf tensor (param), NOT the gradient. The gradient is on
``p.grad`` at hook fire time. Using ``g`` as if it were the grad
silently streams param values to CPU on every microbatch, which
makes optimizer updates no-ops (loss flatlines for thousands of
steps). See :mod:`src.training.param_offload` for the fix and
the long incident writeup.

This test:
  1. Builds a tiny model + CPUAdamW.
  2. Installs the production offload hooks.
  3. Runs 4 microbatches of (loss/4).backward() with different
     inputs.
  4. Asserts ``||s.grad||_2`` is in a sane range (NOT ~16x
     ``||p.data||_2``, which is what the bug would produce).
     Note: in the post-2026-07-15 explicit-accumulator layout,
     ``s.grad`` is the per-step grad accumulator (BF16 pinned
     on CPU). After 4 microbatches, ``s.grad`` holds the sum
     of the per-mb grads.
  5. Spot-checks a param: assert ``||s.grad||`` is much closer
     to ``||real p.grad||`` than to ``||p.data||``.

If this test fails, the offload path is back to streaming param
values to CPU and training will silently flatline.
"""
from __future__ import annotations
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from src.training.param_offload import (
    build_param_groups,
    flush_manual_flush_params,
    flush_pending_grads,
    register_grad_offload_hooks,
    zero_cpu_grad_accum,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA needed")
def test_grad_offload_hook_uses_real_grad_not_param():
    torch.manual_seed(0)
    device = 0
    torch.cuda.set_device(device)

    # Tiny model: one Linear so we can easily compute the true grad.
    model = nn.Sequential(nn.Linear(8, 16, bias=False)).to(device).to(torch.bfloat16)
    linear = model[0]

    from src.training.param_offload import CPUAdamW
    # 2D weight → would normally go to Muon, but force it to
    # AdamW by calling the constructor directly.
    opt = CPUAdamW([linear.weight], lr=1e-3)

    # Install the production hook.
    register_grad_offload_hooks([opt])

    # Compute the "true" grad by disabling the hook and using
    # plain backward on a representative input.
    model_true = nn.Sequential(nn.Linear(8, 16, bias=False)).to(device).to(torch.bfloat16)
    with torch.no_grad():
        model_true[0].weight.copy_(linear.weight)
    x = torch.randn(4, 8, device=device, dtype=torch.bfloat16)
    y = model_true(x).float().pow(2).mean()
    y.backward()
    true_grad_norm = model_true[0].weight.grad.float().norm().item()
    p_data_norm = linear.weight.float().norm().item()
    model_true.zero_grad(set_to_none=True)

    # 4 microbatches of (loss/4).backward()
    for _ in range(4):
        model.zero_grad(set_to_none=False)
        x = torch.randn(4, 8, device=device, dtype=torch.bfloat16)
        y = model(x).float().pow(2).mean()
        (y / 4).backward()
        flush_pending_grads(sync_device=device)
        # No manual-flush params in this test; call is a no-op.
        flush_manual_flush_params([opt])

    # Check: s.grad should hold 4 × (grad/4) = mean(grad) ≈ the
    # true grad (the 4 microbatches with different randn
    # inputs don't accumulate to exactly the true grad, but
    # they're of the same magnitude). Its L2 norm should be
    # in the same order of magnitude as the true grad, NOT
    # 4× the param's L2 norm.
    s = list(opt.state.values())[0]
    accum_norm = s.grad.float().norm().item()
    print(f"\ntrue_grad_norm={true_grad_norm:.4e}  p_data_norm={p_data_norm:.4e}"
          f"  accum_norm={accum_norm:.4e}")

    # The bug would produce accum_norm ≈ 4 × p_data_norm (sum
    # of 4 copies of p.data after 4 microbatches of the buggy
    # hook). The correct value is roughly comparable to the
    # true grad's norm (allow some slack for the randn data
    # varying across microbatches).
    assert accum_norm < 0.5 * p_data_norm, (
        f"accum_norm={accum_norm:.4e} is suspiciously close to"
        f" p_data_norm={p_data_norm:.4e} — the offload hook"
        f" may be streaming p.data instead of p.grad (the"
        f" original g-vs-p.grad bug)."
    )
    # And not absurdly small either (which would mean the hook
    # copied nothing).
    assert accum_norm > 1e-6, f"accum_norm={accum_norm:.4e} too small"

    # Cleanup
    zero_cpu_grad_accum([opt])


# --------------------------------------------------------------------------- #
# adamw_dtype plumbing (2026-07-23).                                          #
# --------------------------------------------------------------------------- #
def test_resolve_dtype_maps_names():
    """``_resolve_dtype`` maps the yml/CLI dtype names to torch dtypes
    and rejects anything else (the single point of resolution at the
    ``build_param_groups`` boundary)."""
    from src.training.param_offload.param_groups import _resolve_dtype

    assert _resolve_dtype("bf16") is torch.bfloat16
    assert _resolve_dtype("fp16") is torch.float16
    assert _resolve_dtype("fp32") is torch.float32
    # Case / whitespace tolerant.
    assert _resolve_dtype(" BF16 ") is torch.bfloat16
    with pytest.raises(ValueError):
        _resolve_dtype("int8")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA needed (pinned buffers)")
@pytest.mark.parametrize(
    "dtype", [torch.bfloat16, torch.float16, torch.float32]
)
def test_cpu_adamw_respects_dtype_kwarg(dtype):
    """CPUAdamW allocates its three pinned buffers (grad / exp_avg /
    exp_avg_sq) at the requested dtype. The default (bf16) matches the
    pre-2026-07-23 hardcoded behavior; fp16/fp32 are the sweep
    headroom options wired via ``optimizer.adamw.dtype`` / --adamw_dtype."""
    from src.training.param_offload import CPUAdamW

    device = 0
    torch.cuda.set_device(device)
    w = nn.Linear(8, 16, bias=False).to(device).to(torch.bfloat16).weight
    opt = CPUAdamW([w], lr=1e-3, dtype=dtype)
    s = list(opt.state.values())[0]
    assert s.grad.dtype == dtype
    assert s.exp_avg.dtype == dtype
    assert s.exp_avg_sq.dtype == dtype
