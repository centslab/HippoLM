"""Block Attention Residuals (Block AttnRes).

From "Attention Residuals" (arXiv:2603.15031) by Kimi Team.
Replaces fixed residual accumulation with learned softmax attention
over block-level representations.

This module moved from ``src/models/block_attn_res.py`` to
``src/models/ops/attn_res.py`` as part of the v0.0.1 ops refactor.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.norms import RMSNorm


class BlockAttnRes(nn.Module):
    """Block-level attention residual computation.

    For each sub-layer, computes input as softmax attention over:
    - b_0: token embedding (always included)
    - b_1, ..., b_{n-1}: completed block representations
    - partial_block: intra-block partial sum (if provided)

    Attention weights are computed from a learned pseudo-query w_l
    and RMSNorm-normalized keys.
    """

    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size

        # Learnable pseudo-query. Initialized to 0 so all attention weights
        # are uniform at the start of training (paper requirement).
        self.query = nn.Parameter(torch.zeros(config.hidden_size))

        # RMSNorm on keys
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
        """
        if partial_block is None:
            # Only attend over completed blocks
            V = torch.stack(blocks, dim=0)  # [N, B, T, D]
        else:
            # Attend over completed blocks + current partial sum
            V = torch.stack(blocks + [partial_block], dim=0)  # [N+1, B, T, D]

        # Normalize keys
        K = self.norm(V)  # Same shape as V

        # Compute attention logits: w_l^T * RMSNorm(v_i)
        # query: [D], K: [N, B, T, D] -> logits: [N, B, T]
        logits = torch.einsum("d,nbtd->nbt", self.query, K)
        weights = F.softmax(logits, dim=0)  # Normalize over depth dimension

        # Weighted aggregation: sum_i weight_i * v_i
        h = torch.einsum("nbt,nbtd->btd", weights, V)  # [B, T, D]
        return h
