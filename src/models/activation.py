"""SwiGLU FFN layer.

Standard SwiGLU ``(silu(x @ W_gate) * x @ W_up) @ W_down``.

Precision is scheme-driven via ``config.ffn_precision``
(``HippoConfig``, see the 2026-07-21 5-scheme migration):

  * ``"w16a16"`` (default) — plain ``nn.Linear`` (BF16 GEMM).
  * ``"w8a16"``  — no W8A16 FFN kernel on sm_120 today; falls
                   back to plain ``nn.Linear`` (BF16) with a
                   logged warning. Listed in the enum for
                   completeness; not used by the canonical
                   ``base.yml`` config.
  * ``"w8a8"``   — ``FP8Linear(fp8_bwd=True)`` (per-row act +
                   per-channel weight FP8 E4M3; FP8 forward AND
                   backward via ``_scaled_mm``).
  * ``"w4a8"``   — NVFP4 mode-3 two-pass:
                   ``NVFP4LinearW4A8`` (NVFP4 packed weights +
                   FP8 E4M3 activations via Triton dequant +
                   ``_scaled_mm``).
  * ``"w4a16"``  — NVFP4 mode-3 Marlin: ``NVFP4Linear(
                   use_marlin=True, no_bf16_master=True)``
                   (NVFP4 packed weights + BF16 MMA via Marlin;
                   FP4 packed buffers are the source of truth).

The kernel choice (two-pass vs Marlin mode-3, FP8 vs BF16 GEMM,
FP8 vs BF16 grads) is hardcoded per scheme — there is no per-
config knob. MXFP4 is not supported for any scheme.

The legacy boolean flags ``ffn_nvfp4`` / ``ffn_nvfp4_marlin`` /
``ffn_nvfp4_no_bf16_master`` were removed on 2026-07-21; the
scheme is the single source of truth.
"""
import warnings

import torch.nn as nn

from src.models.ops.nvfp4_linear import NVFP4Linear
from src.models.ops.nvfp4_linear_w4a8 import NVFP4LinearW4A8
from src.models.ops.fp8_linear import FP8Linear


class SwiGLU(nn.Module):
    """SwiGLU feed-forward network.

    gate = Swish(x @ W_gate)
    up   = x @ W_up
    out  = (gate * up) @ W_down
    """

    def __init__(self, config):
        super().__init__()
        scheme = getattr(config, "ffn_precision", "w16a16")
        Cls, kwargs = self._resolve_class(scheme, config)
        self.gate_proj = Cls(config.hidden_size, config.intermediate_size, **kwargs)
        self.up_proj = Cls(config.hidden_size, config.intermediate_size, **kwargs)
        self.down_proj = Cls(config.intermediate_size, config.hidden_size, **kwargs)

    @staticmethod
    def _resolve_class(scheme: str, config) -> tuple[type, dict]:
        """Map the FFN precision scheme to a Linear class + kwargs.

        Each scheme's kernel is hardcoded — no per-config knobs.
        MXFP4 is not supported (the w4* schemes use NVFP4 only).
        """
        bias = config.use_bias
        if scheme == "w16a16":
            return nn.Linear, {"bias": bias}
        if scheme == "w8a16":
            # No W8A16 FFN kernel on sm_120 today (FP8 storage +
            # BF16 GEMM requires a dequant-to-BF16 path). Fall
            # back to plain BF16 with a warning. Listed for
            # completeness; the canonical config doesn't use it.
            warnings.warn(
                "ffn_precision='w8a16' is not supported on sm_120 today "
                "(no W8A16 FFN kernel — FP8 storage + BF16 GEMM); "
                "falling back to 'w16a16' (BF16).",
                RuntimeWarning,
                stacklevel=2,
            )
            return nn.Linear, {"bias": bias}
        if scheme == "w8a8":
            # FP8 weight storage + FP8 GEMM + FP8 grads. The
            # FP8Linear default ``fp8_bwd=True`` matches the
            # w8a8 contract (FP8 E4M3 backward via two
            # ``_scaled_mm`` calls).
            return FP8Linear, {"bias": bias, "fp8_bwd": True}
        if scheme == "w4a8":
            # NVFP4 storage + FP8 GEMM (two-pass Triton dequant +
            # ``_scaled_mm``). BF16 master weight is the leaf;
            # backward is BF16 STE (spec says FP8 bwd + FP8 grads;
            # current implementation is BF16 — tracked as TODO).
            return NVFP4LinearW4A8, {"bias": bias}
        if scheme == "w4a16":
            # NVFP4 storage + BF16 GEMM (Marlin mode-3). Marlin
            # kernel + no-BF16-master are hardcoded — the kernel
            # choice IS the scheme.
            return NVFP4Linear, {
                "bias": bias,
                "use_marlin": True,
                "no_bf16_master": True,
            }
        raise AssertionError(
            f"ffn_precision must be one of 'w16a16', 'w8a16', 'w8a8', "
            f"'w4a16', 'w4a8'; got {scheme!r}"
        )

    def forward(self, x):
        gate = nn.functional.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)