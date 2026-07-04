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

Pre-RMSNorm fusion (Opt-1)
--------------------------
The module absorbs the previous-layer ``mlp_norm`` into the
``gate_up_proj`` forward (set ``prenorm=True`` on the
``gate_up_proj``). The fused path uses FLA's ``rms_norm_linear``
(BF16) or :class:`_RmsNormNvfp4Matmul` (NVFP4) — both fuse
RMSNorm + the linear into a single autograd Function, dropping one
full ``[T, hidden]`` save vs the unfused ``rms_norm`` + ``linear``
chain (~48 MiB per layer at prod shape).

Activation memory savings stack across all 32 layers (~1500 MiB at
prod shape) without changing the optimizer state. See ``vram_debug``
benchmark and ``docs/gradient_checkpointing.md``.
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

    When ``config.ffn_prenorm_fusion`` is True, ``gate_up_proj`` is
    constructed with ``prenorm=True`` and absorbs the pre-RMSNorm
    that used to live in :class:`TPHippoLayer.mlp_norm`. The forward
    therefore takes the **pre-norm** residual stream (not the
    normalized tensor). The caller must NOT apply an external
    RMSNorm before calling this module in fusion mode.
    """

    def __init__(self, config, device=None, dtype=None) -> None:
        super().__init__()
        if getattr(config, "ffn_nvfp4", False):
            ColCls, RowCls = NVFP4ColumnParallelLinear, NVFP4RowParallelLinear
        else:
            ColCls, RowCls = ColumnParallelLinear, RowParallelLinear

        prenorm = getattr(config, "ffn_prenorm_fusion", False)
        norm_eps = getattr(config, "rms_norm_eps", 1e-6)

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
            prenorm=prenorm, norm_eps=norm_eps,
        )
        self.down_proj = RowCls(
            config.intermediate_size, config.hidden_size,
            bias=config.use_bias, device=device, dtype=dtype,
        )
        self._prenorm = prenorm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is replicated (full hidden). When ``_prenorm`` is True,
        # x is the **pre-norm** residual stream and ``gate_up_proj``
        # does its own RMSNorm internally via the fused Function.
        # When False, x is the **post-norm** tensor (caller applied
        # mlp_norm separately) and ``gate_up_proj`` does matmul only.
        gu = self.gate_up_proj(x)
        inter_per_partition = self.gate_up_proj.out_features_per_partition // 2
        gate, up = gu.split(inter_per_partition, dim=-1)
        return self.down_proj(F.silu(gate) * up)