"""Block Attention Residuals (Block AttnRes).

From "Attention Residuals" (arXiv:2603.15031) by Kimi Team.
Replaces fixed residual accumulation with learned softmax attention
over block-level representations.

This module moved from ``src/models/block_attn_res.py`` to
``src/models/ops/attn_res.py`` as part of the v0.0.1 ops refactor.

Multi-head, head-dim-TP-sharded
--------------------------------
The pseudo-query is now a matrix of shape ``[num_heads, head_dim]``
(8 heads × 128 dim for the v0.0.0 1024-hidden model) rather than a
single ``[hidden_size]`` vector, so each head has its own learned
query. Attention is computed per head and the per-head outputs are
concatenated back to ``[B, T, hidden_size]``.

Under TP, each rank holds ``num_heads // world`` query heads and
operates on the corresponding slice of the residual stream's
hidden dim (i.e. on the rows ``[rank * hpp, (rank + 1) * hpp)`` of
the ``[H, D_h]`` head axis). The per-rank output is
``[B, T, hpp * D_h]``; an :func:`all_gather` of the head dim
restores the full ``[B, T, hidden_size]`` residual before it's
consumed by the next layer.

VRAM notes
----------
The :func:`torch.stack` of block tensors (``[N+1, B, T, D]``) is
the largest transient in this module. Multi-head only changes
the *einsum* shapes; the V/K storage layout is identical to
single-head, so the stack cost is unchanged.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.norms import RMSNorm
from src.models.tp_layers import _TP_GROUP, get_tp_rank, get_tp_world_size


class _AllGatherAlongHeadDim(torch.autograd.Function):
    """Autograd-aware all-gather along the head dim for BlockAttnRes.

    Forward: gather each rank's ``[B, T, hpp * D_h]`` slice along
    the last dim and concatenate to ``[B, T, H * D_h]`` (full
    hidden).

    Backward: take the upstream ``[B, T, H * D_h]`` gradient and
    return this rank's head slice ``[B, T, hpp * D_h]`` — the
    standard "all-gather of an output" backward.

    Why a custom Function? In PyTorch 2.4+ the autograd
    registration for ``_allgather_base_`` (the underlying NCCL
    op behind ``torch.distributed.all_gather_into_tensor``)
    was removed. Calling the bare NCCL allgather on a tensor
    that flows into ``loss.backward()`` triggers the
    ``c10d::_allgather_base_: an autograd kernel was not
    registered`` warning and silently corrupts the gradient:
    the per-rank head slice receives no usable gradient, so
    ``BlockAttnRes.query`` (the pseudo-query) cannot learn.
    The wrapper makes the gradient path explicit.

    Falls back to identity when ``world == 1`` (no comm, no
    autograd concern). For ``world > 1`` the forward issues
    the allgather inside this Function's ``forward`` so the
    autograd graph sees a single node with a registered
    backward, and the gradient flowing back to ``query`` is
    exactly the local head slice of the upstream gradient.

    Implementation note: PyTorch's autograd context
    (``ctx``) in modern versions does not preserve arbitrary
    Python attributes set in ``forward`` to ``backward``
    (the ``backward`` ctx is a different object: an
    ``_AllGatherAlongHeadDimBackward`` whose ``__getattr__``
    only delegates to ``saved_tensors``). We therefore route
    the per-call metadata (``world``, ``rank``, ``hpp``,
    ``head_dim``) through ``ctx.save_for_backward`` as a
    small int32 tensor, and unpack it in ``backward``. This
    is the same pattern used elsewhere in this codebase
    (see ``_AllReduceMax`` in ``src.models.tp_layers``).
    """

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,  # [B, T, hpp * D_h] on this rank's device
        world: int,
        rank: int,
        hpp: int,
        head_dim: int,
    ) -> torch.Tensor:  # [B, T, H * D_h]
        if world == 1:
            # No-op: local slice is already the full hidden.
            # Save the metadata so backward is well-defined and
            # matches the wrapper's contract (returns grad_out).
            ctx.save_for_backward(torch.tensor(
                [world, rank, hpp, head_dim], dtype=torch.int32,
                device=x.device,
            ))
            return x
        # Each rank's slice is contiguous. All-gather a flat
        # ``[world, B*T*hpp*D_h]`` buffer and reshape+permute to
        # ``[B, T, H*D_h]``.
        bt = x.shape[:-1]
        flat = x.reshape(-1)  # [B*T*hpp*D_h]
        buf = torch.empty(
            world * flat.numel(), dtype=x.dtype, device=x.device,
        )
        handle = torch.distributed.all_gather_into_tensor(
            buf, flat, group=_TP_GROUP, async_op=True,
        )
        handle.wait()
        # buf is [world, B*T*hpp*D_h]; reshape each row to
        # [B, T, hpp*D_h] and concat along the last dim to get
        # [B, T, H*D_h].
        per_rank = buf.view(world, *bt, x.shape[-1])
        out = per_rank.movedim(0, -2).reshape(*bt, world * x.shape[-1])
        ctx.save_for_backward(torch.tensor(
            [world, rank, hpp, head_dim], dtype=torch.int32,
            device=x.device,
        ))
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):  # [B, T, H * D_h]
        (meta,) = ctx.saved_tensors
        world, rank, hpp, head_dim = (int(v) for v in meta.tolist())
        if world == 1:
            # Identity: the gradient is the local slice, which
            # is the full hidden when world=1.
            return grad_out, None, None, None, None
        # The backward of an all-gather of an output is a slice:
        # rank r's contribution to the forward was the slice
        # ``grad_out[..., r*hpp*D_h : (r+1)*hpp*D_h]`` along
        # the last dim. Return only that slice as the gradient
        # w.r.t. the local input.
        start = rank * hpp * head_dim
        end = start + hpp * head_dim
        return grad_out[..., start:end], None, None, None, None


def _all_gather_along_head_dim(
    x: torch.Tensor,
    world: int,
    rank: int,
    hpp: int,
    head_dim: int,
) -> torch.Tensor:
    """All-gather a ``[B, T, hpp * D_h]`` tensor along the last
    dim and concatenate to ``[B, T, H * D_h]`` (= full hidden).

    The all-gather is wrapped in a custom
    :class:`torch.autograd.Function` (see
    :class:`_AllGatherAlongHeadDim`) so the backward of the
    boundary is well-defined: the gradient flowing back to
    ``BlockAttnRes.query`` is exactly the local head slice of
    the upstream gradient.

    Falls back to identity when ``world == 1`` (no comm, no
    autograd concern). For ``world > 1`` the wrapper is
    essential — the bare NCCL allgather has no autograd kernel
    in PyTorch 2.4+, which would silently corrupt the gradient
    for the pseudo-query.
    """
    return _AllGatherAlongHeadDim.apply(x, world, rank, hpp, head_dim)


class BlockAttnRes(nn.Module):
    """Block-boundary attention residual, multi-head and TP-shardable.

    In the official Kimi design, AttnRes is invoked only at block
    boundaries (every ``block_size`` layers) to compute the input
    to the next block. Within a block the connection is the
    standard residual ``x = x + Sublayer(x)``.

    For the boundary between block ``n-1`` and block ``n``, the
    input to the first sub-layer of block ``n`` is::

        h = sum_i softmax( w^T * RMSNorm(b_i) ) * b_i
        where b_i in {b_0, b_1, ..., b_{n-1}}

    The pseudo-query ``w`` is per-head (``[num_heads, head_dim]``)
    and shared across all boundaries; initialised to 0 so the
    start-of-training attention is uniform (paper requirement).

    Under TP, the query is sharded along ``num_heads`` and each
    rank holds ``num_heads // world`` heads. The output is
    ``[B, T, hidden_size]`` after an all-gather of the head dim
    across the TP group.

    Note: the all-gather path needs an autograd-aware wrapper
    (see ``_AllGatherAlongHeadDim``); the bare NCCL allgather
    loses gradient through the boundary, which would silently
    break the pseudo-query's learning. The bug is fixed by
    :class:`_AllGatherAlongHeadDim` in :mod:`src.models.tp_layers`
    — this module calls that wrapper, not NCCL directly.
    """

    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        assert self.hidden_size == self.num_heads * self.head_dim, (
            f"hidden_size={self.hidden_size} must equal "
            f"num_heads*head_dim={self.num_heads * self.head_dim}"
        )

        # Per-head learnable pseudo-query. Initialized to 0 so all
        # attention weights are uniform at the start of training
        # (paper requirement).
        # Under TP, the query is sharded along ``num_heads`` and
        # each rank holds ``num_heads // world`` heads. The init
        # value is still 0 so the start-of-training uniform-attention
        # contract holds.
        self.world = get_tp_world_size()
        assert self.num_heads % self.world == 0, (
            f"num_heads={self.num_heads} not divisible by tp_world={self.world}"
        )
        self.hpp = self.num_heads // self.world
        self.rank = get_tp_rank()
        # Shape on each rank: [hpp, head_dim].
        self.query = nn.Parameter(torch.zeros(self.hpp, self.head_dim))

        # RMSNorm on keys. We normalise the *full* hidden, then
        # slice on the head dim — a head-local RMSNorm would mix
        # the per-head axes with the global residual statistics,
        # which deviates from the paper. Single global RMSNorm is
        # cheap (one row-wise normalisation over a few MB).
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, blocks: list[torch.Tensor]) -> torch.Tensor:
        """Compute the input to the next block via AttnRes.

        Args:
            blocks: list of completed block representations,
                each ``[B, T, hidden_size]``. Contains ``b_0``
                (the embedding) and ``b_1, ..., b_{n-1}``.

        Returns:
            Aggregated hidden state ``[B, T, hidden_size]``. When
            ``world > 1`` the head dim is all-gathered across the
            TP group.
        """
        V = torch.stack(blocks, dim=0)  # [N, B, T, D]
        N, B, T, D = V.shape

        # Normalise keys (full hidden, then slice).
        K = self.norm(V)  # [N, B, T, D]
        # Reshape to per-head: [N, B, T, H, D_h].
        K = K.view(N, B, T, self.num_heads, self.head_dim)
        V_h = V.view(N, B, T, self.num_heads, self.head_dim)

        # TP slice: this rank's heads.
        start = self.rank * self.hpp
        K_local = K[..., start:start + self.hpp, :].contiguous()  # [N, B, T, hpp, D_h]
        V_local = V_h[..., start:start + self.hpp, :].contiguous()  # [N, B, T, hpp, D_h]

        # Attention logits per head: w_l[h, :] @ K[..., h, :]
        # query: [hpp, D_h], K_local: [N, B, T, hpp, D_h]
        # -> logits: [N, B, T, hpp]
        logits = torch.einsum("hd,nbthd->nbth", self.query, K_local)
        weights = F.softmax(logits, dim=0)  # softmax over N (block depth)

        # Per-head weighted sum: out[b, t, h, :] = sum_n weights[n, b, t, h] * V_local[n, b, t, h, :]
        # weights: [N, B, T, hpp], V_local: [N, B, T, hpp, D_h]
        # -> out: [B, T, hpp, D_h]
        out = torch.einsum("nbth,nbthd->bthd", weights, V_local)
        # Flatten heads: [B, T, hpp * D_h] (a slice of full hidden).
        out = out.reshape(B, T, self.hpp * self.head_dim)

        # All-gather across the head dim to restore the full
        # residual. The wrapper is autograd-aware: backward
        # slices the upstream gradient to the local head slice,
        # which is what the pseudo-query's gradient needs.
        out = _all_gather_along_head_dim(
            out, self.world, self.rank, self.hpp, self.head_dim,
        )
        return out
