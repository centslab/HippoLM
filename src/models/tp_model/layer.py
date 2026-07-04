"""Single TP layer: attn_norm -> TPKDA -> mlp_norm -> TPSwiGLU.

The block-boundary AttnRes is a single per-device replicated
module that lives on :class:`TPHippoModel`; it is invoked once
per non-first block to compute the next block's input. Inside
a block the residual is standard ``x = x + SubLayer(x)``.

Pre-RMSNorm fusion (Opt-1)
--------------------------
When ``config.ffn_prenorm_fusion`` is True, the FFN path absorbs
``mlp_norm`` into ``TPSwiGLU.gate_up_proj`` (a single fused
autograd Function that does ``rms_norm(x) @ W.T`` in one call).
This saves one full ``[T, hidden]`` save per layer (~48 MiB at
prod shape; ~1500 MiB stacked across 32 layers).

In fusion mode:
  * ``self.mlp_norm`` is None (no separate RMSNorm module).
  * ``self.ffn(x)`` expects ``x`` to be the pre-norm residual
    stream (not the post-norm tensor).

When ``config.ffn_prenorm_fusion`` is False (default for backward
compat), the original ``rms_norm`` + ``gate_up_proj`` chain is
preserved — ``mlp_norm`` is applied externally.
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

    When ``config.ffn_prenorm_fusion`` is True, ``self.mlp_norm``
    is omitted and the FFN absorbs the RMSNorm into its first
    matmul.
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
        self._ffn_prenorm_fusion = getattr(config, "ffn_prenorm_fusion", False)
        if self._ffn_prenorm_fusion:
            # mlp_norm is absorbed into gate_up_proj (see TPSwiGLU doc).
            self.mlp_norm = None
        else:
            self.mlp_norm = _to(RMSNorm(config.hidden_size, eps=config.rms_norm_eps))
        self.kda = _to(TPKDA(config, layer_idx=layer_idx))
        self.ffn = TPSwiGLU(config, device=device, dtype=dtype)

    def forward(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor | None = None,
        initial_state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Standard residual transformer layer on the local device.

        Args:
            x: ``[B, T, hidden_size]`` replicated hidden state.
            cu_seqlens: optional ``[chunk_local_total_docs + 1]``
                long tensor with offsets across the chunk's
                flattened sequence. Forwarded to the KDA
                sub-layer's ShortConvolution only (so the
                depthwise conv resets at doc boundaries).
            initial_state: optional ``[1, hpp, head_k_dim,
                head_v_dim]`` float32 — the KDA recurrent state
                at the start of this chunk. ``None`` (or a list
                entry of ``None`` at the model level) means
                "start from zeros".

        Returns:
            ``(output, final_state)``: ``output`` is
            ``[B, T, hidden_size]`` after KDA and FFN with
            standard residual connections; ``final_state`` is
            the KDA state at the last token, to be passed as
            ``initial_state`` to the next chunk's layer call.
        """
        kda_out, final_state = self.kda(
            self.attn_norm(x), cu_seqlens=cu_seqlens,
            initial_state=initial_state,
        )
        x = x + kda_out
        if self._ffn_prenorm_fusion:
            # FFN absorbs its own RMSNorm via the fused Function in
            # gate_up_proj. Pass the pre-norm residual stream.
            x = x + self.ffn(x)
        else:
            x = x + self.ffn(self.mlp_norm(x))
        return x, final_state
