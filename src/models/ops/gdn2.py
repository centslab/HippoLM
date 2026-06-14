"""Gated DeltaNet 2 (GDN2) module.

A successor to KDA that splits KDA's scalar beta gate into two channel-wise
gates: an erase gate ``b in R^K`` on the key axis and a write gate ``w in R^V``
on the value axis. The matrix-state recurrence is

    S_t = (I - k_t (b_t * k_t)^T) Diag(exp(g_t)) S_{t-1} + k_t (w_t * v_t)^T

so KDA is the special case ``b = w = beta`` (scalar broadcast). The vendored
kernels and the layer module live under :mod:`src.models.ops._vendored.fla`;
this file is the thin wrapper used by :mod:`src.models.model`.

The wrapper mirrors the layout used by :mod:`src.models.ops.kda`: the only job
is to translate :class:`src.models.config.HippoConfig` fields into the
``GatedDeltaNet2`` constructor arguments and expose a plain
``(B, T, hidden)`` forward signature. Cache / attention-mask plumbing is
deliberately not surfaced — HippoLM's training path is global, packed-batch,
NoPE; the inference backend has its own entry point.

Reference: ``Gated DeltaNet-2: Decoupling Erase and Write in Linear Attention``
(NVIDIA, 2025). Adapted from
https://github.com/fla-org/flash-linear-attention/blob/main/fla/layers/gdn2.py.
"""
import torch
import torch.nn as nn

from src.models.ops._vendored.fla.layers.gdn2 import GatedDeltaNet2


class GDN2(nn.Module):
    """Gated DeltaNet 2 wrapper.

    Thin wrapper around :class:`GatedDeltaNet2` for global (NoPE, non-causal)
    attention. The KDA-side optimization flags carried by the wrapper
    (``allow_neg_eigval``, ``use_short_conv``, ``conv_size``, ``conv_bias``) are
    accepted unchanged to keep the :class:`HippoConfig` surface stable across
    the KDA → GDN2 swap.

    GDN2 has no ``safe_gate`` / ``lower_bound`` knobs — its gate path is the
    standard softplus-on-pre-activation, scaled by ``-exp(A_log)``. The
    corresponding fields on :class:`HippoConfig` are silently ignored.
    """

    def __init__(self, config, layer_idx: int = 0):
        super().__init__()
        self.attn = GatedDeltaNet2(
            hidden_size=config.hidden_size,
            head_dim=config.head_dim,
            num_heads=config.num_heads,
            num_v_heads=config.num_heads,
            expand_v=config.expand_v,
            mode=config.gdn2_mode,
            use_short_conv=config.use_short_conv,
            allow_neg_eigval=config.allow_neg_eigval,
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
