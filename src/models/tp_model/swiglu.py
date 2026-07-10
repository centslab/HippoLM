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

When ``config.ffn_nvfp4_no_bf16_master`` is True (and ``ffn_nvfp4``
AND ``ffn_nvfp4_marlin`` are both True), the BF16 master weight is
not allocated on any rank — the FP4 packed buffers are the only
persistent state, the optimizer streams BF16 views through the
module's ``material/commit/apply_chunk_update`` API during the step.
This is the storage mode that scales to MoE (each expert would
otherwise need its own BF16 master = linear scaling in N_experts).
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
        #
        # ``ffn_nvfp4_marlin`` flips the forward matmul to vLLM's
        # Marlin FP4 kernel (~3.5x speedup at FFN shapes on sm_120).
        # Backward stays as BF16 matmul (STE for quantize noise).
        if (
            getattr(config, "ffn_nvfp4", False)
            and not getattr(config, "ffn_nvfp4_bf16_only", False)
        ):
            ColCls, RowCls = NVFP4ColumnParallelLinear, NVFP4RowParallelLinear
            use_marlin = getattr(config, "ffn_nvfp4_marlin", False)
            # Mode (3) requires Marlin — see NVFP4ColumnParallelLinear.
            no_bf16_master = (
                use_marlin
                and getattr(config, "ffn_nvfp4_no_bf16_master", False)
            )
        else:
            ColCls, RowCls = ColumnParallelLinear, RowParallelLinear
            use_marlin = False
            no_bf16_master = False

        # Fused gate+up projection: one ColumnParallelLinear with
        # output = 2 * intermediate_size. The first ``intermediate``
        # output channels are the gate, the next ``intermediate``
        # are the up. This halves the matmul kernel-launch count
        # vs. two separate ColumnParallelLinear's (3 -> 2 per
        # SwiGLU). Bias: when ``use_bias`` is True the fused bias
        # is one tensor of size ``2 * intermediate_per_partition``
        # on this rank (gate half then up half); we expose it as
        # ``self.gate_up_bias`` so the forward can split it.
        # ``use_marlin`` propagates the ffn_nvfp4_marlin flag into
        # the NVFP4 layer constructors; the plain BF16 TP classes
        # don't accept it (no NVFP4 buffers to allocate).
        col_kwargs = {"bias": config.use_bias, "device": device, "dtype": dtype}
        row_kwargs = {"bias": config.use_bias, "device": device, "dtype": dtype}
        if use_marlin:
            col_kwargs["use_marlin"] = True
            row_kwargs["use_marlin"] = True
            if no_bf16_master:
                col_kwargs["no_bf16_master"] = True
                row_kwargs["no_bf16_master"] = True
        self.gate_up_proj = ColCls(
            config.hidden_size, 2 * config.intermediate_size, **col_kwargs,
        )
        self.down_proj = RowCls(
            config.intermediate_size, config.hidden_size, **row_kwargs,
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