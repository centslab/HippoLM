"""Vocab-parallel embedding for the TP path.

This is the v0.1.0 production shape: each rank holds a
``[vocab // world, hidden]`` shard of the logical embed. The
forward does a masked lookup + all-reduce to produce the full
``[B, T, H]`` hidden on every rank; the backward scatters
``dhidden`` into the local shard via ``index_add_`` (no
inter-rank comm).

Replaces the v0.0.0 replicated embed path (which duplicated the
embed's weights and AdamW state across every rank — 7x waste at
TP=8). See :class:`TPShardedEmbed`'s docstring for the memory
analysis.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from src.models.tp_layers import get_tp_rank, get_tp_world_size, tp_all_reduce_sum


class _ShardedEmbedLookup(torch.autograd.Function):
    """Custom autograd for vocab-sharded embedding lookup.

    Forward (per rank):
      - weight:           [vp, H]  (the local vocab shard)
      - input_ids:        [B, T]   (replicated, full vocab indices)
      - Returns hidden:   [B, T, H] (replicated via all-reduce)

    The forward does a masked lookup ``weight[clamp(input_ids -
    start, 0, vp - 1)]`` and multiplies by a ``[B, T]`` mask that is
    True only for tokens belonging to this rank's vocab shard, then
    all-reduces the result across the TP group. The all-reduce
    sums the per-rank partial embeddings into the full ``[B, T, H]``
    hidden on every rank.

    Backward:
      - dhidden:  [B, T, H] (replicated, flowing in from the layers)
      - Returns dw: [vp, H] (gradient for the local weight shard)

    The backward does NOT need any inter-rank comm. Each rank
    receives the full dhidden; it scatters dhidden into the local
    weight shard using ``index_add_`` (the per-row mask is already
    zero for off-rank rows, so the scatter is a no-op for them).
    """

    @staticmethod
    def forward(ctx, weight, input_ids, start, vp):
        # ``input_ids`` lives on this rank's device; ``weight`` is
        # the local [vp, H] shard. Both are on the same device (we
        # constructed the embed with .to(device=d)).
        local_ids = input_ids - start                       # [B, T]
        mask = (local_ids >= 0) & (local_ids < vp)          # [B, T] bool
        local_ids_safe = local_ids.clamp(0, vp - 1)
        local_emb = weight[local_ids_safe]                  # [B, T, H]
        # Zero out off-rank rows so the all-reduce sums to the
        # correct value (only the owning rank contributes
        # non-zero to each row of the sum).
        local_emb = local_emb * mask.unsqueeze(-1).to(local_emb.dtype)
        hidden = tp_all_reduce_sum(local_emb)               # [B, T, H]
        ctx.save_for_backward(input_ids, local_ids_safe, mask)
        ctx.start = start
        ctx.vp = vp
        return hidden

    @staticmethod
    def backward(ctx, dhidden):
        input_ids, local_ids_safe, mask = ctx.saved_tensors
        # dhidden: [B, T, H] on every rank (it flows backward from
        # the layers, which see the replicated full hidden).
        B, T, H = dhidden.shape
        flat_dh = dhidden.reshape(B * T, H)                 # [B*T, H]
        flat_mask = mask.reshape(B * T)                     # [B*T]
        flat_ids = local_ids_safe.reshape(B * T)            # [B*T]
        # Zero out dhidden for off-rank rows. The index_add_ would
        # otherwise write those rows to the clamped index (0 or
        # vp-1), corrupting the local weight gradient.
        masked = flat_dh * flat_mask.unsqueeze(-1).to(flat_dh.dtype)
        dw = masked.new_zeros(ctx.vp, H)                    # [vp, H]
        # index_add_ accumulates: dw[flat_ids[i], :] += masked[i, :]
        # for each i. Multiple (b, t) pairs with the same token
        # naturally sum (correct gradient for a token that appears
        # multiple times in the sequence).
        dw.index_add_(0, flat_ids, masked)
        return dw, None, None, None


class TPShardedEmbed(nn.Embedding):
    """``nn.Embedding`` with the vocab dim sharded across the TP group.

    Memory characteristics (vs. the replicated v0.0.0 embed)
    --------------------------------------------------------

    Replicated (V=248320, H=1024, FP16):
      * embed param:  V * H * 2 B = 508 MB per device, duplicated
        across world devices (waste = (world - 1) * 508 MB).
      * AdamW state:  V * H * 2 * 2 B = 2032 MB per device
        (BF16 m + BF16 v on the embed's optimizer group).

    Sharded (world=8, vp = V / 8 = 31040):
      * embed param:  vp * H * 2 B = 64 MB per device (no
        duplication).
      * AdamW state:  vp * H * 2 * 2 B = 256 MB per device.
      * Per-forward comm: one all-reduce of [B, T, H] = 8 MB
        (the masked partial embeddings are summed to the full
        hidden).

    Net saving at TP=8: 7 * 508 + 7 * 256 = 5352 MB across the
    node, or 669 MB per device.

    Forward (per rank)
    ------------------
    1. Masked lookup: ``weight[clamp(input_ids - start, 0, vp-1)]``
       multiplied by a [B, T] bool mask (True only for tokens in
       this rank's vocab shard).
    2. ``tp_all_reduce_sum`` of the masked [B, T, H] partial
       embeddings. The result is the full [B, T, H] hidden, on
       every rank.

    Backward
    --------
    No inter-rank comm. Each rank scatters dhidden into the local
    weight shard via ``index_add_``; off-rank rows have mask=False
    so they contribute zero.

    Interaction with tied lm_head
    -----------------------------
    The matmul ``hidden @ weight.T`` is naturally a column-parallel
    matmul on the local [vp, H] shard: the result is [N, vp]
    sharded logits, which is exactly what :class:`TPFusedLceLoss`
    expects. The FusedLinearCE's ``dw`` is a [vp, H] gradient that
    goes into ``weight.grad`` (the local shard's grad) via the
    custom autograd in :class:`_TiedFusedLCEFunction`. PyTorch's
    autograd sums the FusedLinearCE dw and the lookup backward dw
    (both are gradients for the same parameter from different
    paths) into ``weight.grad`` at the end of backward.
    """

    def __init__(self, vocab_size: int, hidden_size: int, *args, **kwargs) -> None:
        # Save the LOGICAL vocab size for ``forward`` to compute
        # the per-token rank routing. The parent's
        # ``num_embeddings`` is the local shard size.
        self.logical_vocab_size = vocab_size
        self.world = get_tp_world_size()
        assert vocab_size % self.world == 0, (
            f"vocab_size={vocab_size} not divisible by tp_world={self.world}"
        )
        self.vocab_per_partition = vocab_size // self.world
        self.rank = get_tp_rank()
        self._start = self.rank * self.vocab_per_partition
        # Initialize the parent ``nn.Embedding`` with the LOCAL
        # shard size. The weight has shape [vp, H] and lives on
        # this rank's device.
        super().__init__(self.vocab_per_partition, hidden_size, *args, **kwargs)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return _ShardedEmbedLookup.apply(
            self.weight, input_ids, self._start, self.vocab_per_partition
        )

    def extra_repr(self) -> str:
        return (
            f"logical_vocab_size={self.logical_vocab_size}, "
            f"vocab_per_partition={self.vocab_per_partition}, "
            f"hidden_size={self.embedding_dim}, "
            f"tp_world={self.world}, tp_rank={self.rank}, "
            f"shard_start={self._start}"
        )
