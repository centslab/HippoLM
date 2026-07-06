"""Column-row parallel SwiGLU for the TP path.

Fused gate+up projection (one ``ColumnParallelLinear`` with
output = ``2 * intermediate_size``) + row-parallel down projection
with all-reduce. Saves 1 kernel launch per SwiGLU vs the unfused
3-projection baseline.

When ``config.ffn_nvfp4`` is True, the projections use
:class:`NVFP4ColumnParallelLinear` / :class:`NVFP4RowParallelLinear`
instead of the BF16 versions. Weight storage is NVFP4 packed; the
matmul still runs in BF16 (dequant-on-fwd). The optimizer updates
the BF16 master weight; :func:`repack_nvfp4_weights` re-quantizes
it after each step.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.tp_layers import ColumnParallelLinear, RowParallelLinear
from src.models.ops.nvfp4_tp import NVFP4ColumnParallelLinear, NVFP4RowParallelLinear


class TPSwiGLU(nn.Module):
    """Column-row parallel SwiGLU.

    gate_proj:  hidden  -> intermediate (column parallel, sharded)
    up_proj:    hidden  -> intermediate (column parallel, sharded)
    down_proj:  intermediate (sharded) -> hidden (row parallel, all-reduce)
    """

    def __init__(self, config, device=None, dtype=None) -> None:
        super().__init__()
        # ``ffn_nvfp4_bf16_only`` toggles the W4A16 FFN linears to a
        # plain-BF16 mode: the BF16 master is the only GPU storage, no
        # ``packed_weight`` / ``scales`` buffers are allocated. Saves
        # the per-layer packed/scales VRAM at the cost of losing the
        # FP4 round-trip in the forward (the matmul is plain BF16 GEMM).
        # Useful on small-VRAM boxes where the packed buffer footprint
        # outweighs the dequant-on-fwd benefit.
        if (
            getattr(config, "ffn_nvfp4", False)
            and not getattr(config, "ffn_nvfp4_bf16_only", False)
        ):
            ColCls, RowCls = NVFP4ColumnParallelLinear, NVFP4RowParallelLinear
        else:
            ColCls, RowCls = ColumnParallelLinear, RowParallelLinear

        # Fused gate+up projection: one ColumnParallelLinear with
        # output = 2 * intermediate_size. The first ``intermediate``
        # output channels are the gate, the next ``intermediate``
        # are the up. This halves the matmul kernel-launch count
        # vs. two separate ColumnParallelLinear's (3 -> 2 per
        # SwiGLU). Bias: when ``use_bias`` is True the fused bias
        # is one tensor of size ``2 * intermediate_per_partition``
        # on this rank (gate half then up half); we expose it as
        # ``self.gate_up_bias`` so the forward can split it.
        self.gate_up_proj = ColCls(
            config.hidden_size, 2 * config.intermediate_size,
            bias=config.use_bias, device=device, dtype=dtype,
        )
        self.down_proj = RowCls(
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