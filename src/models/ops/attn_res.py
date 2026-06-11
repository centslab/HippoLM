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


def _all_gather_along_head_dim(x: torch.Tensor) -> torch.Tensor:
    """All-gather a ``[B, T, hpp * D_h]`` tensor along the last
    dim and concatenate to ``[B, T, H * D_h]`` (= full hidden).

    Falls back to identity when world==1.
    """
    world = get_tp_world_size()
    if world == 1:
        return x
    # Each rank's slice is contiguous. We all-gather a flat
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
    return out


class BlockAttnRes(nn.Module):
    """Block-level attention residual computation, multi-head and
    TP-shardable.

    For each sub-layer, computes input as softmax attention over:
    - b_0: token embedding (always included)
    - b_1, ..., b_{n-1}: completed block representations
    - partial_block: intra-block partial sum (if provided)

    Attention weights are computed from a learned per-head
    pseudo-query ``w_l`` of shape ``[num_heads, head_dim]`` and
    RMSNorm-normalized keys. The output is ``[B, T, hidden_size]``
    (after an all-gather of the head dim across the TP group).
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

    def forward(
        self,
        blocks: list[torch.Tensor],
        partial_block: torch.Tensor | None,
    ) -> torch.Tensor:
        """Compute attention residual.

        Args:
            blocks: List of [batch_size, seq_len, hidden_size] tensors.
                Contains b_0 (embedding) and completed block representations.
            partial_block: [batch_size, seq_len, hidden_size] or None.
                Intra-block partial sum for the current block.

        Returns:
            Aggregated hidden state [batch_size, seq_len, hidden_size]
            on this rank's device. When world>1, this is the
            all-gathered result across the head dim.
        """
        if partial_block is None:
            V = torch.stack(blocks, dim=0)  # [N, B, T, D]
        else:
            V = torch.stack(blocks + [partial_block], dim=0)  # [N+1, B, T, D]
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
        # residual. When world==1 this is a no-op.
        out = _all_gather_along_head_dim(out)
        return out
