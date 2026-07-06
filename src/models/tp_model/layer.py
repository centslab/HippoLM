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
        initial_state: torch.Tensor | None = None,
        save_residual: bool = True,
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

        ``save_residual`` is accepted for API stability (some
        callers — e.g. the block-level ``_block_forward`` — pass
        ``False`` historically) but is currently unused: the
        standard ``x = x + sub`` always allocates a fresh sum
        tensor. See "opt-5/1 RMSNorm/residual no-op" in the
        project memory for the empirical / theoretical analysis
        showing this is a true zero-savings no-op (the saved
        ``x`` in the rmsnorm Function is a reference already
        held by the residual chain's autograd history; removing
        the save frees no storage in this project's setup).

        Returns:
            ``(output, final_state)``: ``output`` is
            ``[B, T, hidden_size]`` after KDA and FFN with
            standard residual connections; ``final_state`` is
            the KDA state at the last token, to be passed as
            ``initial_state`` to the next chunk's layer call.
        """
        del save_residual  # empirically a zero-savings no-op — see memory note
        kda_out, final_state = self.kda(
            self.attn_norm(x), cu_seqlens=cu_seqlens,
            initial_state=initial_state,
        )
        x = x + kda_out
        x = x + self.ffn(self.mlp_norm(x))
        return x, final_state
