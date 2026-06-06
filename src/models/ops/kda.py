"""Kimi Delta Attention (KDA) module.

A fast delta-rule linear attention from the vendored
``flash-linear-attention`` library (``src.models.ops._vendored.fla``).
Supports chunked parallel and fused recurrent modes.

The KDA module (``KimiDeltaAttention``) is a verbatim copy from the
fla library, with its ``from fla.X import Y`` lines rewritten to
import from the local vendored namespace. This keeps the
implementation tuneable in-tree without forking fla.

Reference: Kimi Linear (https://arxiv.org/abs/2510.26692)
"""
import torch
import torch.nn as nn

from src.models.ops._vendored.fla.layers.kda import KimiDeltaAttention


class KDA(nn.Module):
    """Kimi Delta Attention wrapper.

    Simple wrapper around KimiDeltaAttention for global (non-causal) attention.
    No position encoding by design (NoPE).
    """

    def __init__(self, config, layer_idx: int = 0):
        super().__init__()
        self.attn = KimiDeltaAttention(
            hidden_size=config.hidden_size,
            head_dim=config.head_dim,
            num_heads=config.num_heads,
            num_v_heads=config.num_heads,
            expand_v=config.expand_v,
            mode=config.kda_mode,
            use_short_conv=config.use_short_conv,
            allow_neg_eigval=config.allow_neg_eigval,
            safe_gate=config.safe_gate,
            lower_bound=config.lower_bound,
            conv_size=config.conv_size,
            conv_bias=config.conv_bias,
            layer_idx=layer_idx,
            norm_eps=config.rms_norm_eps,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor [batch_size, seq_len, hidden_size]

        Returns:
            Output tensor [batch_size, seq_len, hidden_size]
        """
        output, _, _ = self.attn(
            hidden_states=x,
            attention_mask=None,
            past_key_values=None,
            use_cache=False,
            output_attentions=False,
        )
        return output
