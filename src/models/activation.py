"""SwiGLU FFN layer.

Standard SwiGLU ``(silu(x @ W_gate) * x @ W_up) @ W_down``.

When ``config.ffn_nvfp4`` is True, the three linear projections
store their weights in NVFP4 packed format (E2M1 + FP8-e4m3fn 1x16
microblock scales) instead of BF16. The matmul still runs in BF16
(dequant-on-fwd) since PyTorch 2.9.1's ``torch._scaled_mm`` NVFP4
path requires BOTH A and B to be FP4-packed. The optimizer updates
the BF16 master weight; :func:`repack_nvfp4_weights` is called
once per training step to keep the FP4 buffers in sync.

When ``config.ffn_nvfp4_marlin`` is True (and ``ffn_nvfp4`` is
also True), the forward matmul uses vLLM's Marlin FP4 kernel
(BF16 MMA + register dequant + cp.async double-buffered prefetch)
instead of the dequant+cuBLAS path. ~3.5x speedup at FFN shapes
on sm_120 (47-49 TFLOPS).
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
        use_nvfp4 = getattr(config, "ffn_nvfp4", False)
        # ``use_marlin`` only takes effect when ``ffn_nvfp4`` is also
        # True (the NVFP4 modules own the use_marlin flag — nn.Linear
        # ignores it).
        use_marlin = use_nvfp4 and getattr(config, "ffn_nvfp4_marlin", False)
        Cls = NVFP4Linear if use_nvfp4 else nn.Linear
        kwargs = {"bias": config.use_bias}
        if use_nvfp4:
            kwargs["use_marlin"] = use_marlin
        self.gate_proj = Cls(config.hidden_size, config.intermediate_size, **kwargs)
        self.up_proj = Cls(config.hidden_size, config.intermediate_size, **kwargs)
        self.down_proj = Cls(config.intermediate_size, config.hidden_size, **kwargs)

    def forward(self, x):
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)