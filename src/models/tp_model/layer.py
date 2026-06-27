"""Single TP layer: attn_norm -> TPKDA -> mlp_norm -> TPSwiGLU.

The block-boundary AttnRes is a single per-device replicated
module that lives on :class:`TPHippoModel`; it is invoked once
per non-first block to compute the next block's input. Inside
a block the residual is standard ``x = x + SubLayer(x)``.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from src.models.norms import RMSNorm

from .kda import TPKDA
from .swiglu import TPSwiGLU


class TPHippoLayer(nn.Module):
    """TP version of HippoLayer. KDA is sharded along the head dim
    via :class:`TPKDA`; FFN is sharded via :class:`TPSwiGLU`.

    AttnRes no longer lives inside each layer. The block-boundary
    AttnRes is a single per-device replicated module
    (``TPHippoModel.replicated_per_device[d]["attn_res"]``);
    it is invoked once per non-first block to compute the next
    block's input. Inside a block the residual is standard
    ``x = x + SubLayer(x)``.
    """

    def __init__(self, layer_idx: int, config, device=None, dtype=None) -> None:
        super().__init__()
        self.layer_idx = layer_idx

        # NB: ``device`` may be 0 (an int) which is falsy in Python
        # — guard with ``is not None`` to avoid the conditional
        # silently falling through to the no-op branch.
        def _to(m: nn.Module) -> nn.Module:
            if device is None:
                return m
            if dtype is not None:
                return m.to(device=device, dtype=dtype)
            return m.to(device=device)

        self.attn_norm = _to(RMSNorm(config.hidden_size, eps=config.rms_norm_eps))
        self.mlp_norm = _to(RMSNorm(config.hidden_size, eps=config.rms_norm_eps))
        self.kda = _to(TPKDA(config, layer_idx=layer_idx))
        self.ffn = TPSwiGLU(config, device=device, dtype=dtype)

    def forward(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Standard residual transformer layer on the local device.

        Args:
            x: ``[B, T, hidden_size]`` replicated hidden state.
            cu_seqlens: optional ``[total_docs + 1]`` long tensor
                with global offsets across the flattened
                ``batch_size * seq_len`` sequence. Forwarded
                to the KDA sub-layer only; FFN / RMSNorm do not
                depend on the doc layout.

        Returns:
            ``[B, T, hidden_size]`` after KDA and FFN with
            standard residual connections.
        """
        x = x + self.kda(self.attn_norm(x), cu_seqlens=cu_seqlens)
        x = x + self.ffn(self.mlp_norm(x))
        return x
