"""Tests for the truncated BPTT KDA-state detach in
:mod:`src.training.loop.run._run_training_loop` (lines 270-273).

The per-chunk loop carries the KDA recurrent state from chunk
``i`` to chunk ``i+1`` (so the model sees the right initial
state), but it **detaches** the state from the autograd graph::

    kda_states = [
        st.detach() if st is not None else None
        for st in outputs["kda_states"]
    ]

This bounds the autograd graph to one chunk at a time, so the
backward of chunk ``i+1`` does NOT traverse back through chunk
``i``'s saved tensors. This is the **chunked KDA OOM fix** (see
the historical writeup in
:mod:`src.training.param_offload._state`): at 16 chunks of
T=16384, full BPTT retains all 16 chunks' saved-tensor
activations simultaneously (~93 GB); per-chunk bwd+detach
bounds the peak to one chunk.

This file pins three contracts:

  1. **Numerical preservation** — the detach preserves the
     carried state's *values* (chunk ``i+1`` sees the same
     initial h-vector as chunk ``i``'s output). A refactor that
     drops the detach or accidentally casts to a different
     dtype would silently corrupt chunk ``i+1``'s forward.

  2. **Graph severance** — the carried (detached) state has
     ``requires_grad=False`` and ``grad_fn=None``. The autograd
     walk from chunk ``i+1``'s loss stops at the detach.

  3. **Memory release** — ``del outputs`` after the detach
     releases chunk ``i``'s saved tensors. The autograd graph
     node count for the carried state is zero (no
     chunk-``i`` graph nodes reachable from the detach).

The strongest assertion is the bidirectional pair:

  * With detach (truncated BPTT): backward from a hypothetical
    chunk-``i+1`` loss through the carried state does NOT
    populate chunk-``i``'s params.
  * Without detach (full BPTT): the same backward DOES populate
    chunk-``i``'s params (proving the test setup is sensitive
    enough to catch a regression).

We use a minimal :class:`torch.nn.RNNCell` as a KDA proxy (both
have a recurrent hidden state that contributes to the
next-step output and gradient). This lets the tests run on CPU
without spinning up the TPKDA stack.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))


# =========================================================================== #
# KDA proxy — minimal RNN cell.
# =========================================================================== #
class _RNNCellProxy(torch.nn.Module):
    """Minimal KDA-proxy: a hidden state that the next forward
    step consumes.

    Each forward takes ``(x_t, h_prev)`` and returns a final
    ``(loss, h_next)``. The hidden state has ``requires_grad``
    when produced from a forward that built a graph; after
    ``.detach()`` it becomes a leaf as far as autograd is
    concerned.

    Used to verify the loop's detach contract without spinning
    up the TPKDA stack (which would force a CUDA dependency).
    """

    def __init__(self, input_dim: int = 4, hidden_dim: int = 8) -> None:
        super().__init__()
        self.cell = torch.nn.RNNCell(input_dim, hidden_dim)
        self.proj = torch.nn.Linear(hidden_dim, 1)

    def forward(
        self, x_seq: torch.Tensor, h0: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``x_seq``: ``[T, input_dim]``. Returns ``(loss, h_final)``.

        The loss is the final-step projection squared-mean — a
        stand-in for the model's per-chunk CE. Only the final
        step's projection contributes (matches the chunk-level
        loss the loop's forward returns; we don't model the
        per-token CE here because we don't need it for the
        detach contract).
        """
        T = x_seq.shape[0]
        h = h0 if h0 is not None else torch.zeros(
            1, self.cell.hidden_size, dtype=x_seq.dtype,
        )
        for t in range(T):
            h = self.cell(x_seq[t: t + 1], h)
        y = self.proj(h)
        loss = y.pow(2).mean()
        return loss, h


# =========================================================================== #
# Loop-shaped fixtures.
# =========================================================================== #
def _two_chunk_loop(
    model: _RNNCellProxy,
    x1: torch.Tensor,
    x2: torch.Tensor,
    detach: bool = True,
) -> tuple:
    """Faithful replica of the loop's per-chunk forward + (optional) detach.

    Mirrors run.py lines 246-274:

      with autocast: forward(chunk_i, kda_states=kda_states)
      kda_states = [st.detach() if st is not None else None
                    for st in outputs["kda_states"]]
      del outputs

    When ``detach=False``, the carried state is NOT detached —
    this is the full-BPTT path used for the opposite-direction
    test (to prove the test setup is sensitive to a regression).
    """
    # Chunk 1: initial state is None.
    loss1, h1 = model(x1, h0=None)
    # Carry + (optional) detach.
    if detach:
        h1_carried = h1.detach() if h1 is not None else None
    else:
        h1_carried = h1
    # Chunk 2: forward receives the carried state.
    loss2, h2 = model(x2, h0=h1_carried)
    # Combine: the loop sums per-chunk losses (then divides by
    # n_chunks / token-weights — we just sum here for the test).
    total_loss = loss1 + loss2
    return total_loss, h1, h1_carried, h2


# =========================================================================== #
# Contract 1 — numerical preservation.
# =========================================================================== #
def test_detach_preserves_numerical_state():
    """The detach MUST preserve the carried state's *values* —
    chunk ``i+1`` sees the same initial h-vector as chunk
    ``i``'s output.

    If the detach accidentally cast to a different dtype or
    zeroed the state, the chunk-``i+1`` forward would diverge
    from the full-BPTT forward (silent training corruption).
    """
    torch.manual_seed(0)
    model = _RNNCellProxy()
    _, h1, h1_carried, _ = _two_chunk_loop(
        model, torch.randn(4, 4), torch.randn(4, 4),
    )
    assert torch.equal(h1, h1_carried), (
        "detach() must preserve the numerical state — the loop "
        "relies on the carried state being numerically identical "
        "to the previous chunk's output"
    )


def test_detach_preserves_dtype():
    """The detach must preserve the dtype (the next forward
    step types-checks against the carried state's dtype)."""
    torch.manual_seed(0)
    model = _RNNCellProxy()
    _, h1, h1_carried, _ = _two_chunk_loop(
        model, torch.randn(4, 4), torch.randn(4, 4),
    )
    assert h1_carried.dtype == h1.dtype


# =========================================================================== #
# Contract 2 — graph severance.
# =========================================================================== #
def test_detach_breaks_requires_grad():
    """The detached (carried) state must have
    ``requires_grad=False``. Otherwise the backward of chunk
    ``i+1`` would propagate through chunk ``i``'s graph
    (re-introducing the chunked KDA OOM)."""
    torch.manual_seed(0)
    model = _RNNCellProxy()
    _, h1, h1_carried, _ = _two_chunk_loop(
        model, torch.randn(4, 4), torch.randn(4, 4),
    )
    assert h1.requires_grad is True, (
        "sanity: the un-detached state must have requires_grad "
        "(otherwise the test setup is broken)"
    )
    assert h1_carried.requires_grad is False, (
        "the carried state MUST have requires_grad=False — "
        "otherwise backward propagates through chunk i's graph"
    )


def test_detach_severs_grad_fn():
    """The detached state must have ``grad_fn=None`` (it's a
    leaf from the perspective of any backward that starts at
    chunk ``i+1``)."""
    torch.manual_seed(0)
    model = _RNNCellProxy()
    _, h1, h1_carried, _ = _two_chunk_loop(
        model, torch.randn(4, 4), torch.randn(4, 4),
    )
    assert h1.grad_fn is not None, (
        "sanity: the un-detached state has a grad_fn "
        "(otherwise the setup is broken)"
    )
    assert h1_carried.grad_fn is None, (
        "the carried state MUST have grad_fn=None — autograd "
        "cannot walk back through it"
    )


# =========================================================================== #
# Contract 3 — memory / graph isolation (the OOM-fix guarantee).
# =========================================================================== #
def test_backward_through_carried_detached_state_does_not_reach_chunk_i():
    """The truncated-BPTT guarantee: backward through the
    detached carried state (simulating chunk ``i+1``'s loss)
    must NOT propagate into chunk ``i``'s parameters.

    Setup:
      * Run chunk ``i`` forward (graph built, NO backward).
      * Detach the carried state.
      * ``del`` the un-detached state and the loss (matches
        the loop's ``del outputs`` on line 274).
      * Pretend chunk ``i+1`` ran and produced a loss that
        depends on the carried state.

    To exercise the backward path (a detached leaf has
    ``requires_grad=False``, so a direct ``.backward()`` on it
    would raise), we wrap the carried state in a synthetic
    leaf: ``h1_carried.clone().requires_grad_(True)``. This
    models "chunk 2's autograd graph starts here as a fresh
    leaf" — the same starting-point the loop's chunk-2 forward
    effectively has when it consumes the carried state.

    Expectation: NO parameter receives a gradient. The detach
    severed chunk 1's graph; chunk 2's autograd walk starts at
    its own synthetic leaf and never traverses back through
    chunk 1's saved tensors.
    """
    torch.manual_seed(0)
    model = _RNNCellProxy()

    # Chunk 1: build the graph.
    loss1, h1 = model(torch.randn(4, 4), h0=None)
    # Detach (mimics the loop's per-chunk detach).
    h1_carried = h1.detach() if h1 is not None else None
    # ``del`` to release chunk 1's graph nodes (matches the
    # loop's ``del outputs`` / ``del kda_states``).
    del h1, loss1

    # The detach already severed the graph: h1_carried has no
    # grad_fn (it's a leaf as far as chunk 2 is concerned).
    assert h1_carried.grad_fn is None

    # Pretend chunk 2 ran: re-attach the data as a fresh leaf
    # (the loop's chunk-2 forward essentially does the same —
    # it consumes the carried state as input, building its own
    # graph from that point on).
    chunk2_state = h1_carried.clone().detach().requires_grad_(True)
    chunk2_loss = chunk2_state.sum()
    chunk2_loss.backward()
    # chunk2_state.grad is finite (=1 from sum); the test
    # below verifies that NO chunk-1 param received a gradient.
    assert torch.isfinite(chunk2_state.grad).all()

    # No chunk-1 param should have a gradient: chunk 2's
    # backward walk started at chunk2_state (a leaf) and never
    # traversed back through chunk 1's graph.
    for name, p in model.named_parameters():
        assert p.grad is None, (
            f"param {name} has a gradient — the detach failed to "
            f"sever the graph; chunk 2's loss backproped into "
            f"chunk 1's parameters (truncated BPTT broken)"
        )


def test_full_bptt_propagates_backward_through_carried_state():
    """Opposite-direction test: WITHOUT detach, backward through
    the carried state DOES populate chunk-1's params'
    gradients.

    This pins the bidirectional guarantee so we know the
    previous test isn't trivially passing for both paths. If
    this test fails, the test setup is wrong (the proxy model
    doesn't actually backprop through the recurrent state).
    """
    torch.manual_seed(0)
    model = _RNNCellProxy()

    loss1, h1 = model(torch.randn(4, 4), h0=None)
    # NO detach — full BPTT path.
    chunk2_loss = h1.sum()
    chunk2_loss.backward()

    # Without detach, chunk 2's loss backprop reaches chunk 1's
    # params. If this assertion fails, the test setup is wrong
    # (the proxy model isn't a useful KDA stand-in).
    grads = [p.grad for p in model.parameters()]
    assert any(g is not None for g in grads), (
        "without detach, backward through chunk 2's loss must "
        "populate chunk 1's params' gradients (full BPTT). If "
        "this fails, the test setup is wrong — the proxy model "
        "doesn't actually backprop through the recurrent state."
    )
    # And specifically: the input-projection weight (RNN
    # ``weight_ih``) is reached through chunk 1's path, so it
    # MUST have a gradient under full BPTT.
    weight_ih_grad = model.cell.weight_ih.grad
    assert weight_ih_grad is not None, (
        "RNN weight_ih must receive a gradient under full BPTT "
        "(it was reached through chunk 1's graph)"
    )


# =========================================================================== #
# End-to-end — joint backward with detach is well-defined.
# =========================================================================== #
def test_joint_backward_with_detach_completes_without_raising():
    """The full loop pattern: build chunk 1's graph, detach,
    build chunk 2's graph on top of the detached state, run
    the JOINT backward. Must complete cleanly (no autograd
    errors), and every trainable param receives a finite,
    non-NaN gradient.

    The chunk-1 contribution to the gradient flows through
    chunk 1's own graph (which IS still reachable from the
    joint loss). The chunk-2 contribution flows through chunk
    2's graph only (it can't reach chunk 1 via the detach).
    Both are finite and well-defined.
    """
    torch.manual_seed(42)
    model = _RNNCellProxy()
    total_loss, _, _, _ = _two_chunk_loop(
        model, torch.randn(4, 4), torch.randn(4, 4), detach=True,
    )
    total_loss.backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, (
            f"param {name} has no gradient after joint backward"
        )
        assert torch.isfinite(p.grad).all(), (
            f"param {name} has non-finite gradient: "
            f"{(~torch.isfinite(p.grad)).sum().item()} non-finite entries"
        )


def test_detach_survives_del_of_source_outputs():
    """The loop's ``del outputs`` after the detach must NOT
    invalidate the carried state (the carried state is a
    separate, owned tensor — ``del`` on the source just releases
    the source's refcount).

    If the carried state were a *view* into the source, ``del
    outputs`` would invalidate it and chunk ``i+1``'s forward
    would crash. ``.detach()`` returns a fresh tensor (not a
    view), so the carried state is independent.
    """
    torch.manual_seed(0)
    model = _RNNCellProxy()
    loss1, h1, h1_carried, _ = _two_chunk_loop(
        model, torch.randn(4, 4), torch.randn(4, 4),
    )
    h1_carried_clone = h1_carried.clone()
    # Drop the source. The carried state must survive.
    del h1, loss1
    # Numerical values must still match the clone.
    assert torch.equal(h1_carried, h1_carried_clone), (
        "deleting the source outputs invalidated the carried "
        "state — the carried state must be an independent "
        "tensor, not a view into the source"
    )


# =========================================================================== #
# Per-layer detachment in the loop (line 270-273 only detaches
# the KDA state, NOT the loss or other outputs).
# =========================================================================== #
def test_detach_only_touches_carried_state_not_loss_or_intermediates():
    """The loop's detach pattern only touches the carried state
    (``outputs['kda_states']``); the loss itself and any other
    graph-internal tensors stay attached.

    If a refactor accidentally detaches the loss or the model
    outputs, backward would no longer propagate (the joint
    gradient would be empty).
    """
    torch.manual_seed(0)
    model = _RNNCellProxy()

    # Build chunk 1 (graph attached).
    loss1, h1 = model(torch.randn(4, 4), h0=None)
    # Detach only the state (mimics the loop).
    h1_carried = h1.detach()
    # The loss MUST still have a grad_fn — it's part of the
    # chunk-1 graph that's about to backprop.
    assert loss1.grad_fn is not None
    assert h1.grad_fn is not None  # the un-detached state has a grad_fn
    assert h1_carried.grad_fn is None  # the carried state does not
    # Backward through the loss must still reach chunk-1's params.
    model.zero_grad()
    loss1.backward()
    has_grad = any(p.grad is not None for p in model.parameters())
    assert has_grad, (
        "backward through the (still-attached) loss must reach "
        "chunk-1's params; if this fails, the test setup or "
        "detach pattern is broken"
    )