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

    def forward(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor ``[batch_size, seq_len, hidden_size]``.
            cu_seqlens: optional ``[total_docs + 1]`` long tensor
                with global offsets across the flattened
                ``[batch_size * seq_len]`` sequence. When set, the
                KDA layer is called with ``batch_size=1`` (the
                batch dim is folded into the sequence dim) and the
                kernel resets the recurrent state at each
                ``cu_seqlens`` boundary. Without ``cu_seqlens`` the
                layer treats every row of the input as an
                independent sequence (the legacy single-doc-per-row
                contract).

        Returns:
            Output tensor ``[batch_size, seq_len, hidden_size]``.
            When ``cu_seqlens`` is set the output is reshaped back
            from the ``[1, batch_size * seq_len, hidden_size]``
            layout the kernel produced.
        """
        if cu_seqlens is None:
            # Legacy path: one doc per row, no varlen state reset.
            output, _, _ = self.attn(
                hidden_states=x,
                attention_mask=None,
                past_key_values=None,
                use_cache=False,
                output_attentions=False,
            )
            return output

        # Packed path: fold the batch dim into the sequence dim so
        # the kernel sees ``batch_size=1`` (its hard requirement
        # when ``cu_seqlens`` is supplied; see
        # :func:`chunk_kda`). The vendored KDA layer picks
        # ``cu_seqlens`` up from ``**kwargs`` and threads it
        # through to the conv1d (when ``use_short_conv=True``) and
        # to the chunkwise recurrence.
        B, T, H = x.shape
        x_flat = x.reshape(1, B * T, H)
        output_flat, _, _ = self.attn(
            hidden_states=x_flat,
            attention_mask=None,
            past_key_values=None,
            use_cache=False,
            output_attentions=False,
            cu_seqlens=cu_seqlens,
        )
        return output_flat.reshape(B, T, H)
