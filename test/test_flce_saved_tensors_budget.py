"""Regression test for Opt-3: FLCE dw chunking VRAM contract.

Historical context
------------------
Before commit ``feat(flce): accumulate dw into embed_weight.grad
during forward (Opt-3)`` (2026-07-04), :class:`_TiedFusedLCEFunction`
saved ``(dx, dw, embed_weight)`` in ``ctx.save_for_backward``:

    dx:           [N, H] BF16 =  16384 × 1536 × 2 =   48 MiB
    dw:           [V, H] BF16 = 248320 × 1536 × 2 =  727 MiB
    embed_weight: [V, H] BF16 = same               =  727 MiB
    TOTAL:                                           1502 MiB

The ``dw`` and ``embed_weight`` were both full ``[V, H]`` BF16
copies of data that the embed parameter already owned. The
``embed_weight`` save was needed only for ``embed_weight.grad is
None`` detection in backward (could have been elided with a
``hasattr`` / sentinel flag); the ``dw`` save was needed so that
the kernel's do-scaling could happen in backward before the
``add_`` into ``embed_weight.grad``.

Opt-3 collapsed both saves by moving the ``add_`` into forward:
the kernel returns ``dw`` as before, but instead of stashing it
in ``ctx`` we immediately scatter it into ``embed_weight.grad``.
Backward only needs ``dx`` (for the gradient of the hidden
state), so ``ctx.save_for_backward(dx)`` — total saved_tensors
budget drops to ~48 MiB.

What this test pins
-------------------
The exact ``len(saved_tensors)`` and the shape of the saved
tensor. A future refactor that re-adds ``dw`` or ``embed_weight``
to ``save_for_backward`` (e.g. to do-scaling the accumulated
grad at backward time) would silently regress ~1454 MiB of
saved_tensors per chunked microbatch — easily enough to push
the per-step driver peak from 13916 MiB back above the 16311
MiB 5060 Ti 16G limit. This test would catch that.

The ``do == 1.0`` contract
--------------------------
Opt-3 assumes ``do == 1.0`` for the FLCE call (the standard
PyTorch seed gradient for a scalar loss output). The fla
kernel confirms this with its own skip-mul:
``if torch.ne(do, torch.tensor(1.0, ...))``. If any future
caller seeds ``do != 1.0`` here, the grad will be off by that
factor. We do NOT assert ``do == 1.0`` in the kernel because
fla already gates the skip on it; this test only pins the
saved_tensors budget.
"""
from __future__ import annotations

import gc
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from src.models.tp_layers import init_tp


def _build_minimal_setup(
    v_size: int = 1024,
    h_size: int = 256,
    n_tokens: int = 32,
    seed: int = 0,
):
    init_tp(world_size=1, devices=[0], backend="gloo")
    torch.manual_seed(seed)
    device = 0
    torch.cuda.set_device(device)
    embed = nn.Embedding(v_size, h_size).to(device=device, dtype=torch.bfloat16)
    nn.init.normal_(embed.weight, std=0.05)
    hidden = torch.randn(n_tokens, h_size, device=device, dtype=torch.bfloat16)
    target = torch.randint(0, v_size, (n_tokens,), device=device)
    return embed, hidden, target


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA requires GPU")
def test_tied_flce_saved_tensors_only_dx():
    """Opt-3 VRAM contract: _TiedFusedLCEFunction saved_tensors must
    contain ONLY dx — no dw, no embed_weight. At prod config this
    drops the saved_tensors budget from 1502 MiB to 48 MiB.

    Failure mode this catches:
      - Someone re-adds dw to save_for_backward to "do scaling
        in backward". This silently regresses ~727 MiB/chunk.
      - Someone re-adds embed_weight to save_for_backward for
        grad-allocation detection. This silently regresses
        another ~727 MiB/chunk.
    """
    from src.models.tp_model.lm_head import _TiedFusedLCEFunction

    embed, hidden, target = _build_minimal_setup(v_size=2048, h_size=512, n_tokens=128, seed=1)
    embed.zero_grad()

    # Forward only — don't call backward so saved_tensors stay alive
    # on the autograd Function's ctx for the gc walk below. We MUST
    # hold a reference to ``loss`` so the autograd graph isn't
    # garbage-collected (the loss tensor is the only Python reference
    # to the graph in this test).
    loss = _TiedFusedLCEFunction.apply(
        hidden, target, embed.weight, embed.weight,
        0, 2048, -100, 8,
    )
    assert loss.grad_fn is not None, "loss.grad_fn is None — autograd graph not built"

    # Find the autograd Function node. The backward class is
    # named ``_TiedFusedLCEFunctionBackward`` (PyTorch's
    # convention: forward class + 'Backward' suffix).
    target_name = "_TiedFusedLCEFunctionBackward"
    found_node = None
    for obj in gc.get_objects():
        if type(obj).__name__ != target_name:
            continue
        found_node = obj
        break
    assert found_node is not None, (
        f"no {target_name} node in gc after forward — backward graph not built"
    )

    saved = found_node.saved_tensors
    assert isinstance(saved, tuple), f"saved_tensors is {type(saved)}, not tuple"

    # Opt-3: only dx is saved. Old code would have 3 (dx, dw, embed_weight).
    assert len(saved) == 1, (
        f"_TiedFusedLCEFunctionBackward.saved_tensors has {len(saved)} tensors, "
        f"expected 1 (only dx). Opt-3 contract violated — dw or embed_weight "
        f"was re-added to save_for_backward, regressing ~1454 MiB at prod."
    )

    only_tensor = saved[0]
    assert isinstance(only_tensor, torch.Tensor)
    # The single saved tensor must be dx, shape [N, H].
    expected_shape = (hidden.shape[0], hidden.shape[1])
    assert tuple(only_tensor.shape) == expected_shape, (
        f"saved tensor shape {tuple(only_tensor.shape)} != expected dx "
        f"shape {expected_shape}"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA requires GPU")
def test_tied_flce_grad_scatter_into_embed_weight():
    """Opt-3 correctness: dw must be added to embed_weight.grad
    DURING forward (not backward), so that backward doesn't need
    to touch .grad for the FLCE contribution.

    Verifies:
      1. embed_weight.grad is allocated by forward (first-call case).
      2. embed_weight.grad[start:vp] matches the fla kernel's
         reference dw (within BF16 noise).
    """
    from src.models.tp_model.lm_head import _TiedFusedLCEFunction
    from src.models.ops._vendored.fla.modules.fused_linear_cross_entropy import (
        fused_linear_cross_entropy_forward,
    )

    embed, hidden, target = _build_minimal_setup(v_size=512, h_size=256, n_tokens=64, seed=42)
    embed.zero_grad()
    assert embed.weight.grad is None, "precondition: grad should be None"

    # Run the new autograd Function (forward only — no backward).
    loss_new = _TiedFusedLCEFunction.apply(
        hidden, target, embed.weight, embed.weight,
        0, 512, -100, 8,
    )

    # (1) First-call allocation: embed_weight.grad must exist after forward.
    assert embed.weight.grad is not None, (
        "embed_weight.grad was not allocated during forward — Opt-3 first-call "
        "alloc path is broken"
    )
    assert embed.weight.grad.shape == embed.weight.shape

    # (2) The slice [start:vp] must match the reference dw from fla.
    _, _, dw_ref, _ = fused_linear_cross_entropy_forward(
        hidden, target, embed.weight, None,
        ignore_index=-100, num_chunks=8, reduction="mean",
    )
    grad_slice = embed.weight.grad[:dw_ref.shape[0]]
    max_abs_diff = (grad_slice.float() - dw_ref.float()).abs().max().item()
    assert max_abs_diff < 1e-3, (
        f"embed_weight.grad[start:vp] mismatch: max abs diff {max_abs_diff:.4e} "
        f"vs fla kernel reference"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA requires GPU")
def test_tied_flce_grad_accumulates_across_calls():
    """Opt-3 correctness: multiple forward calls (simulating
    multiple microbatches in a chunked-training step) must
    accumulate into embed_weight.grad via add_, not overwrite.

    This is critical for the chunked-training path (b173879):
    each chunk calls _TiedFusedLCEFunction once, and the embed
    gradient must be the SUM of all chunks' contributions.
    """
    from src.models.tp_model.lm_head import _TiedFusedLCEFunction
    from src.models.ops._vendored.fla.modules.fused_linear_cross_entropy import (
        fused_linear_cross_entropy_forward,
    )

    embed, hidden, target = _build_minimal_setup(v_size=512, h_size=256, n_tokens=32, seed=99)
    embed.zero_grad()

    # First forward
    _TiedFusedLCEFunction.apply(
        hidden, target, embed.weight, embed.weight,
        0, 512, -100, 8,
    )
    grad_after_first = embed.weight.grad.clone()

    # Second forward with different inputs
    hidden2 = torch.randn(32, 256, device=0, dtype=torch.bfloat16)
    target2 = torch.randint(0, 512, (32,), device=0)
    _TiedFusedLCEFunction.apply(
        hidden2, target2, embed.weight, embed.weight,
        0, 512, -100, 8,
    )

    # The second call's dw should be ADDED to the first.
    _, _, dw_second, _ = fused_linear_cross_entropy_forward(
        hidden2, target2, embed.weight, None,
        ignore_index=-100, num_chunks=8, reduction="mean",
    )
    expected = grad_after_first + dw_second
    max_abs_diff = (embed.weight.grad.float() - expected.float()).abs().max().item()
    assert max_abs_diff < 1e-3, (
        f"multi-call accumulation wrong: max abs diff {max_abs_diff:.4e} "
        f"vs expected grad_after_first + dw_second"
    )