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
                   ``NVFP4W4A8SwiGLU`` (NVFP4 packed weights +
                   FP8 E4M3 activations via Triton dequant +
                   ``_scaled_mm``). The wrapper shares a single
                   ``act_quant`` across gate+up and exposes
                   ``forward_precomputed(x, a_fp8, a_s)`` so
                   an upstream :func:`rmsnorm_fp8_with_passthrough`
                   can skip the redundant per-FFN quant.
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

Producer-side FP8 quant fusions (fp8_rmsnorm /
fp8_residual / fp8_silu_mul) were removed on 2026-07-23 — see
:class:`HippoConfig` for the rationale. The fused kernels in
``src/models/ops/{rmsnorm_fp8,fp8_residual,silu_mul_fp8}.py``
are kept for unit-test coverage but no longer wired into the
SwiGLU forward path. ``forward_precomputed`` remains available
for callers that want to route pre-quantized activations into
the W4A8 wrapper directly.
"""
import warnings

import torch
import torch.nn as nn

from src.models.ops.nvfp4_linear import NVFP4Linear
from src.models.ops.nvfp4_linear_w4a8 import NVFP4LinearW4A8, NVFP4W4A8SwiGLU
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
        bias = bool(config.use_bias)
        # Special case: w4a8 uses the NVFP4W4A8SwiGLU wrapper which
        # shares a single act_quant across gate+up and exposes a
        # forward_precomputed hook for the upstream RmsNormFp8STE
        # passthrough. All other schemes build the three projections
        # directly (no FFN-level fusion available).
        if scheme == "w4a8":
            self.ffn_inner: nn.Module = NVFP4W4A8SwiGLU(
                config.hidden_size, config.intermediate_size, bias=bias,
            )
            # Expose the inner's three projections as direct attributes
            # so external access (state_dict, tests, debug) keeps the
            # ``ffn.gate_proj.weight`` path working.
            self.gate_proj = self.ffn_inner.gate_proj
            self.up_proj = self.ffn_inner.up_proj
            self.down_proj = self.ffn_inner.down_proj
            self._uses_inner = True
        else:
            Cls, kwargs = self._resolve_class(scheme, config)
            self.gate_proj = Cls(config.hidden_size, config.intermediate_size, **kwargs)
            self.up_proj = Cls(config.hidden_size, config.intermediate_size, **kwargs)
            self.down_proj = Cls(config.intermediate_size, config.hidden_size, **kwargs)
            self.ffn_inner = None
            self._uses_inner = False

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
        if self._uses_inner:
            return self.ffn_inner(x)
        gate = nn.functional.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)

    def forward_precomputed(
        self,
        x: torch.Tensor,
        a_fp8: torch.Tensor,
        a_s: torch.Tensor,
    ) -> torch.Tensor:
        """Forward with caller-supplied FP8+scale for the shared act_quant.

        Only available when ``ffn_precision="w4a8"`` (the
        :class:`NVFP4W4A8SwiGLU` wrapper). ``x`` is the BF16 input
        (kept for the silu_quant_fused backward's saved tensor);
        ``a_fp8`` / ``a_s`` are the FP8 representation of ``x`` to
        route into gate+up. Saves the ~105 us / FFN redundant
        ``quantize_act_fp8_fused`` call that :meth:`forward` would
        otherwise issue internally.
        """
        if not self._uses_inner:
            raise RuntimeError(
                "SwiGLU.forward_precomputed is only supported when "
                "ffn_precision='w4a8' (uses NVFP4W4A8SwiGLU inner). "
                f"Got scheme with ffn_inner={self.ffn_inner!r}."
            )
        return self.ffn_inner.forward_precomputed(x, a_fp8, a_s)