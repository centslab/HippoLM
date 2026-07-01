"""SwiGLU FFN layer.

Standard SwiGLU ``(silu(x @ W_gate) * x @ W_up) @ W_down``.

When ``config.ffn_nvfp4`` is True, the three linear projections
store their weights in NVFP4 packed format (E2M1 + FP8-e4m3fn 1x16
microblock scales) instead of BF16. The matmul still runs in BF16
(dequant-on-fwd) since PyTorch 2.9.1's ``torch._scaled_mm`` NVFP4
path requires BOTH A and B to be FP4-packed. The optimizer updates
the BF16 master weight; :func:`repack_nvfp4_weights` is called
once per training step to keep the FP4 buffers in sync.
"""
import torch.nn as nn
import torch.nn.functional as F

from src.models.ops.nvfp4_linear import NVFP4Linear


class SwiGLU(nn.Module):
    """SwiGLU feed-forward network.

    gate = Swish(x @ W_gate)
    up   = x @ W_up
    out  = (gate * up) @ W_down
    """

    def __init__(self, config):
        super().__init__()
        Cls = NVFP4Linear if getattr(config, "ffn_nvfp4", False) else nn.Linear
        self.gate_proj = Cls(
            config.hidden_size, config.intermediate_size, bias=config.use_bias,
        )
        self.up_proj = Cls(
            config.hidden_size, config.intermediate_size, bias=config.use_bias,
        )
        self.down_proj = Cls(
            config.intermediate_size, config.hidden_size, bias=config.use_bias,
        )

    def forward(self, x):
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)