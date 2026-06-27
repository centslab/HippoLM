"""Column-row parallel SwiGLU for the TP path.

Fused gate+up projection (one ``ColumnParallelLinear`` with
output = ``2 * intermediate_size``) + row-parallel down projection
with all-reduce. Saves 1 kernel launch per SwiGLU vs the unfused
3-projection baseline.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.tp_layers import ColumnParallelLinear, RowParallelLinear


class TPSwiGLU(nn.Module):
    """Column-row parallel SwiGLU.

    gate_proj:  hidden  -> intermediate (column parallel, sharded)
    up_proj:    hidden  -> intermediate (column parallel, sharded)
    down_proj:  intermediate (sharded) -> hidden (row parallel, all-reduce)
    """

    def __init__(self, config, device=None, dtype=None) -> None:
        super().__init__()
        # Fused gate+up projection: one ColumnParallelLinear with
        # output = 2 * intermediate_size. The first ``intermediate``
        # output channels are the gate, the next ``intermediate``
        # are the up. This halves the matmul kernel-launch count
        # vs. two separate ColumnParallelLinear's (3 -> 2 per
        # SwiGLU). Bias: when ``use_bias`` is True the fused bias
        # is one tensor of size ``2 * intermediate_per_partition``
        # on this rank (gate half then up half); we expose it as
        # ``self.gate_up_bias`` so the forward can split it.
        self.gate_up_proj = ColumnParallelLinear(
            config.hidden_size, 2 * config.intermediate_size,
            bias=config.use_bias, device=device, dtype=dtype,
        )
        self.down_proj = RowParallelLinear(
            config.intermediate_size, config.hidden_size,
            bias=config.use_bias, device=device, dtype=dtype,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is replicated (full hidden). gate_up_proj produces
        # sharded intermediate slices on this rank: the first
        # ``inter_per_partition`` channels are the gate, the next
        # ``inter_per_partition`` are the up. down is row-parallel
        # and all-reduces back to full hidden.
        gu = self.gate_up_proj(x)
        inter_per_partition = self.gate_up_proj.out_features_per_partition // 2
        gate, up = gu.split(inter_per_partition, dim=-1)
        return self.down_proj(F.silu(gate) * up)
