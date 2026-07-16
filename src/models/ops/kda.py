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


# Modules inside KimiDeltaAttention that should run their GEMM in FP8
# E4M3 when ``config.kda_fp8`` is True. ``o_proj`` is included for
# output projection; ``b_proj`` is the small (1536 -> 12) Linear that
# produces the beta scalar (post-sigmoid; the test sweep showed this
# path has 0.98% sig_rel — essentially noise-free).
_DIRECT_FP8_LINEARS = ("q_proj", "k_proj", "v_proj", "o_proj", "b_proj")
# Sequential pairs (f_proj is the gate log-bottleneck, g_proj is the
# value-side gate). Each pair has two Linears; both get FP8.
_SEQUENTIAL_FP8_LINEARS = ("f_proj", "g_proj")


def _replace_linear_with_fp8(old: nn.Linear) -> nn.Linear:
    """Swap an ``nn.Linear`` for ``FP8Linear`` with copied weights.

    Used by :meth:`KDA.__init__` to convert KimiDeltaAttention's
    projection layers in place. State-dict keys are preserved
    (``weight`` / ``bias`` have the same names + shapes), so BF16
    checkpoints load without conversion.

    The replacement keeps the *BF16 leaf* weight (FP8Linear stores
    ``weight`` as ``nn.Parameter`` of dtype BF16). The optimizer
    updates that leaf; the FP8 quantize happens on the fly in the
    forward pass.
    """
    from src.models.ops.fp8_linear import FP8Linear

    has_bias = old.bias is not None
    new = FP8Linear(
        old.in_features, old.out_features,
        bias=has_bias,
        device=old.weight.device,
        dtype=old.weight.dtype,
    )
    new.weight.data.copy_(old.weight.data)
    if has_bias:
        new.bias.data.copy_(old.bias.data)
    return new


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

        # FP8 W8A8 path for the projection layers (q/k/v/o/b_proj +
        # f_proj[0..1] + g_proj[0..1]). The kernel itself, the
        # gating path (A_log / dt_bias / FusedRMSNormGated), and the
        # rest of the model stay in BF16.
        # See ``src.models.ops.fp8_linear`` for the per-row-act +
        # per-channel-weight E4M3 GEMM via ``torch._scaled_mm``.
        if getattr(config, "kda_fp8", False):
            for name in _DIRECT_FP8_LINEARS:
                old = getattr(self.attn, name)
                setattr(self.attn, name, _replace_linear_with_fp8(old))
            for seq_name in _SEQUENTIAL_FP8_LINEARS:
                seq = getattr(self.attn, seq_name)
                for i in range(len(seq)):
                    seq[i] = _replace_linear_with_fp8(seq[i])

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
