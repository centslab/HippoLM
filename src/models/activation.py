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

When ``config.ffn_nvfp4_no_bf16_master`` is True (and ``ffn_nvfp4``
AND ``ffn_nvfp4_marlin`` are both True), the BF16 master weight is
not allocated at all on either CPU or GPU — only the FP4 packed
buffers exist. The optimizer streams BF16 views through the module's
``material/commit/apply_chunk_update`` API during the step (see
:class:`NVFP4Linear` and :mod:`src.training.param_offload`). This is
the storage mode that scales to MoE: a per-expert BF16 master at
8+ experts and 32 layers would dominate VRAM.
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
        # Mode (3) requires Marlin — the dequant+cuBLAS no-leaf Function
        # exists in nvfp4_linear but isn't wired into the SwiGLU path
        # yet (would need a per-call back-reference; see the comment at
        # NVFP4Linear.__init__).
        no_bf16_master = (
            use_marlin and getattr(config, "ffn_nvfp4_no_bf16_master", False)
        )
        Cls = NVFP4Linear if use_nvfp4 else nn.Linear
        kwargs = {"bias": config.use_bias}
        if use_nvfp4:
            kwargs["use_marlin"] = use_marlin
            kwargs["no_bf16_master"] = no_bf16_master
        self.gate_proj = Cls(config.hidden_size, config.intermediate_size, **kwargs)
        self.up_proj = Cls(config.hidden_size, config.intermediate_size, **kwargs)
        self.down_proj = Cls(config.intermediate_size, config.hidden_size, **kwargs)

    def forward(self, x):
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)