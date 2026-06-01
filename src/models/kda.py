"""Kimi Delta Attention (KDA) module.

A fast delta-rule linear attention from flash-linear-attention.
Supports chunked parallel and fused recurrent modes.
"""
import torch
import torch.nn as nn

from fla.layers.kda import KimiDeltaAttention


class KDA(nn.Module):
    """Kimi Delta Attention wrapper.

    Simple wrapper around KimiDeltaAttention for global (non-causal) attention.
    No position encoding by design (NoPE).
    """

    def __init__(self, config):
        super().__init__()
        self.attn = KimiDeltaAttention(
            hidden_size=config.hidden_size,
            head_dim=config.head_dim,
            num_heads=config.num_heads,
            num_v_heads=config.num_kv,  # Grouped value attention
            mode="chunk",
            use_short_conv=False,  # No local convolution for global attention
            allow_neg_eigval=False,
            safe_gate=False,
            layer_idx=0,
            norm_eps=config.rms_norm_eps,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor [batch_size, seq_len, hidden_size]

        Returns:
            Output tensor [batch_size, seq_len, hidden_size]
        """
        # KDA expects [B, T, D] and returns [B, T, D]
        output, _, _ = self.attn(
            hidden_states=x,
            attention_mask=None,
            past_key_values=None,
            use_cache=False,
            output_attentions=False,
        )
        return output
