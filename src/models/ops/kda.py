"""Kimi Delta Attention (KDA) module.

A fast delta-rule linear attention from the vendored
``flash-linear-attention`` library (``src.models.ops._vendored.fla``).
Supports chunked parallel and fused recurrent modes.

The KDA module (``KimiDeltaAttention``) is a verbatim copy from the
fla library, with its ``from fla.X import Y`` lines rewritten to
import from the local vendored namespace. This keeps the
implementation tuneable in-tree without forking fla.

Reference: Kimi Linear (https://arxiv.org/abs/2510.26692)

Precision is scheme-driven via ``config.attention_precision``
(``HippoConfig``, see the 2026-07-21 5-scheme migration):

  * ``"w16a16"`` (default) — no FP8 swap; plain ``nn.Linear``.
  * ``"w8a8"``   — ``FP8Linear(fp8_bwd=True)`` (per-row act +
                   per-channel weight E4M3; FP8 forward AND
                   backward via ``_scaled_mm``). Same scheme as
                   the FFN W4A8 autograd contract.
  * ``"w8a16"``  — no BF16-GEMM W8A16 KDA kernel on sm_120 today
                   (FP8 storage + BF16 GEMM requires a dequant-to-
                   BF16 path). Falls back to ``w16a16`` (plain
                   ``nn.Linear``) with a logged warning.
  * ``"w4a8"``   — no NVFP4 KDA kernel on sm_120 today; falls
                   back to ``w8a8`` (``FP8Linear(fp8_bwd=True)``)
                   with a logged warning. NVFP4 → FP8 GEMM is the
                   closest available match for the spec's
                   "NVFP4 storage + FP8 GEMM".
  * ``"w4a16"``  — no NVFP4 KDA kernel on sm_120 today; falls
                   back to ``w16a16`` (plain ``nn.Linear``) with
                   a logged warning. NVFP4 → BF16 GEMM is the
                   closest available match for the spec's
                   "NVFP4 storage + BF16 GEMM".

Precision-sensitive projections stay as ``nn.Linear`` (BF16 GEMM)
in all cases (see the projection-name lists below for the
rationale).

The legacy boolean flags ``kda_fp8`` / ``kda_mxfp8`` were removed
on 2026-07-21; the scheme is the single source of truth. MXFP8
is no longer supported (the b12x dense GEMM was not maintained
and has been removed).
"""
from __future__ import annotations

import warnings

import torch
import torch.nn as nn

from src.models.ops._vendored.fla.layers.kda import KimiDeltaAttention


# Modules inside KimiDeltaAttention that should run their GEMM in FP8
# E4M3 when ``config.attention_precision`` is ``"w8a8"`` (or any
# non-w16a16 scheme that resolves to FP8 — currently that's just
# w8a8 and the w4a8 fallback).
#
# Precision-sensitive projections stay as ``nn.Linear`` (BF16 GEMM):
#
#   * ``f_proj`` (sequential 2-Linear gate pre-exp): per-module
#     noise probe showed 5.11% sig_rel at its output (the highest
#     of any KDA projection), driven by compounded FP8 noise across
#     two GEMMs in series. The output feeds ``exp(-A_log.exp())``
#     in chunk_kda, so any quant noise gets amplified by the
#     exponential before driving the per-token decay rate. Too
#     sensitive to quantize. Reverted to BF16 on 2026-07-21.
#
#   * ``b_proj``: beta gate; small-output Linear whose FP8 path is
#     already BF16 via the div-by-16 fallback at prod H=12. The
#     remaining (q, k, v, o) projections are kept in FP8 because:
#       - ``l2norm`` inside chunk_kda absorbs magnitude noise for
#         q, k (only direction matters post-l2norm);
#       - ``o_proj``'s 3.66% intrinsic noise is the same as q/k/v
#         — not more precision-sensitive (probe 2026-07-21). The
#         high whole-KDA sig_rel at the residual stream is mostly
#         upstream chunk_kda accumulation; adding 1pp absolute
#         from o_proj's FP8 noise is acceptable for the ~45ms/step
#         BF16 GEMM saving.
_DIRECT_FP8_LINEARS = ("q_proj", "k_proj", "v_proj", "o_proj")
# Sequential pairs kept in FP8. ``g_proj`` survives because its
# output feeds ``o_norm`` (FusedRMSNormGated) which divides by
# ``||x||`` — magnitude quant noise is damped; per-module noise
# probe showed only 1.60% sig_rel at its output (lowest of any
# KDA projection).
_SEQUENTIAL_FP8_LINEARS = ("g_proj",)


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
        fp8_bwd=True,
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

        # Resolve the attention_precision scheme to a concrete
        # replacement function (or None for the no-swap w16a16
        # case). See the module docstring for the full spec; the
        # table is:
        #
        #   w16a16 → no swap (plain nn.Linear throughout)
        #   w8a8   → _replace_linear_with_fp8 (FP8Linear fp8_bwd=True)
        #   w8a16  → no kernel → w16a16 fallback (nn.Linear) + warn
        #   w4a8   → no kernel → w8a8 fallback + warn
        #   w4a16  → no kernel → w16a16 fallback + warn
        scheme = getattr(config, "attention_precision", "w16a16")
        replace_fn = self._resolve_replace_fn(scheme)

        if replace_fn is not None:
            for name in _DIRECT_FP8_LINEARS:
                old = getattr(self.attn, name)
                setattr(self.attn, name, replace_fn(old))
            for seq_name in _SEQUENTIAL_FP8_LINEARS:
                seq = getattr(self.attn, seq_name)
                for i in range(len(seq)):
                    seq[i] = replace_fn(seq[i])

    @staticmethod
    def _resolve_replace_fn(scheme: str):
        """Map the attention_precision scheme to a Linear replacement.

        Returns ``None`` for ``w16a16`` (no swap). For schemes
        without a KDA kernel on sm_120, falls back to the closest
        available scheme and emits a RuntimeWarning.
        """
        if scheme == "w16a16":
            return None
        if scheme == "w8a8":
            return _replace_linear_with_fp8
        if scheme == "w8a16":
            # No BF16-GEMM W8A16 KDA kernel on sm_120 today.
            # Fall back to plain nn.Linear (BF16 GEMM, BF16 grads)
            # with a warning — preserves the BF16 GEMM contract.
            warnings.warn(
                "attention_precision='w8a16' is not supported on sm_120 "
                "today (no BF16-GEMM W8A16 KDA kernel); falling back to "
                "'w16a16' (plain nn.Linear).",
                RuntimeWarning, stacklevel=2,
            )
            return None
        if scheme == "w4a8":
            # No NVFP4 KDA kernel on sm_120 today. The closest
            # available match for the spec's "NVFP4 storage + FP8
            # GEMM" is FP8Linear (FP8 GEMM via _scaled_mm).
            warnings.warn(
                "attention_precision='w4a8' is not supported on sm_120 "
                "today (no NVFP4 KDA kernel); falling back to 'w8a8' "
                "(FP8Linear, FP8 GEMM + FP8 bwd).",
                RuntimeWarning, stacklevel=2,
            )
            return _replace_linear_with_fp8
        if scheme == "w4a16":
            # No NVFP4 KDA kernel on sm_120 today. The closest
            # available match for the spec's "NVFP4 storage + BF16
            # GEMM" is plain nn.Linear (BF16 GEMM, BF16 grads).
            warnings.warn(
                "attention_precision='w4a16' is not supported on sm_120 "
                "today (no NVFP4 KDA kernel); falling back to 'w16a16' "
                "(plain nn.Linear, BF16 GEMM + BF16 grads).",
                RuntimeWarning, stacklevel=2,
            )
            return None
        raise AssertionError(
            f"attention_precision must be one of 'w16a16', 'w8a16', "
            f"'w8a8', 'w4a16', 'w4a8'; got {scheme!r}"
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