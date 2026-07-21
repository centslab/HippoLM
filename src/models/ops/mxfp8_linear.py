"""MXFP8 Linear — b12x CuTe dense GEMM forward + BF16 STE backward.

Drop-in ``nn.Linear`` replacement that runs its forward GEMM through
the vendored ``b12x`` MXFP8 dense kernel (per-32-element-block E4M3
values + UE8M0 pow2 scales on *both* activation and weight). At the
KDA prod shape (M=16384, K=N=1536) on sm_120 the b12x MXFP8 GEMM
hits ~103 TFLOPS (~106% of the 97 TFLOPS FP8 dense spec) — measurably
faster than ``torch._scaled_mm`` W8A8 (~95 TFLOPS) at large M.

Why MXFP8 vs the W8A8 ``FP8Linear`` path
----------------------------------------
Both quantize to E4M3, so the *representation* noise floor is the
same. The difference is scale granularity:

  * W8A8 (``FP8Linear``): per-row activation scale (FP32) +
    per-output-channel weight scale (FP32). One scale per row / col.
  * MXFP8 (this module): per-32-element-block UE8M0 scale on both
    operands. 48 scales per 1536-wide row instead of 1.

The finer block granularity is the bigger lever on noise (per the
W8A8-vs-W4A8 probe, granularity beat representation) — but on the
Gaussian synthetic sweep MXFP8's UE8M0 (pow2-only) scale roughly
cancels the block-granularity win, landing at ~3.75% sig_rel,
about the same as W8A8's 3.7%. On real (non-Gaussian, outlier-heavy)
activations the per-block scale should pull ahead.

Backward (STE — same contract as ``FP8Linear`` default path)
------------------------------------------------------------
The forward's MXFP8 quantization is discarded in backward. The
backward re-runs BF16 matmul against the BF16 master weight:

    grad_x = grad_out @ w_bf16      (BF16 matmul)
    grad_w = grad_out.T @ x_bf16    (BF16 matmul)
    grad_b = grad_out.sum(dim=0)    (if bias present)

There is no MXFP8 backward kernel in b12x (the dense GEMM is a
forward-only serving primitive), so a matching-precision FP8 bwd
is not available. The BF16 STE bwd is bit-equivalent to
``nn.Linear`` at the BF16 rounding-noise floor, and is the same
scheme ``FP8Linear`` and NVFP4 mode-2 use.

Weight pack cache (Strategy 2 — skip-N MXFP8 quant)
---------------------------------------------------
``quantize_mxfp8_rows_torch`` + ``pack_mxfp8_linear_weight`` are a
pure function of the BF16 master weight, so the packed weight is
cached on the module and invalidated only when the BF16 leaf's
``_version`` bumps (the optimizer step). In a bench loop the cache
hits every forward after the first; in training the hit rate is
(N_microbatches-1)/N_microbatches.

State dict: BF16 master ``weight`` (+ optional ``bias``) only —
identical to ``nn.Linear``, so a BF16 checkpoint loads without
conversion.

Shape constraint
----------------
MXFP8 requires ``in_features % 32 == 0`` (the block size). The b12x
packer additionally pads K to a multiple of 128 internally. If
``in_features`` is not a multiple of 32 the constructor silently
flips to ``bf16_only=True`` (bit-exact with ``nn.Linear``).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


_MXFP8_BLOCK = 32  # MXFP8_SCALE_VEC_SIZE — UE8M0 scale block along K
# The b12x dense MXFP8 GEMM consumes a 128-wide K tile, and the weight
# helper ``quantize_mxfp8_rows_torch`` requires K % 128 == 0. Projections
# whose K is not a multiple of 128 fall back to BF16 (e.g. the small
# inner Linear of a sequential gate).
_MXFP8_K_MULTIPLE = 128


# ---------------------------------------------------------------------------
# Lazy b12x import (the CuTe kernel compiles on first use; keep it out of
# module-import time so environments without b12x still import this file).
# ---------------------------------------------------------------------------
def _b12x_api():
    """Import and return the b12x MXFP8 API, or ``None`` if unavailable."""
    try:
        from b12x.gemm.mxfp8_linear import (  # type: ignore
            is_mxfp8_linear_supported,
            mxfp8_linear,
            pack_mxfp8_linear_weight,
        )
        from b12x.gemm.wo_projection import quantize_mxfp8_rows_torch  # type: ignore
    except Exception:  # pragma: no cover - b12x not installed
        return None
    return (
        is_mxfp8_linear_supported,
        mxfp8_linear,
        pack_mxfp8_linear_weight,
        quantize_mxfp8_rows_torch,
    )


def is_mxfp8_available() -> tuple[bool, str | None]:
    """Whether b12x is importable AND the GPU supports MXFP8 MMA."""
    api = _b12x_api()
    if api is None:
        return False, "b12x not importable"
    is_supported = api[0]
    return is_supported()


# ---------------------------------------------------------------------------
# Autograd Function: b12x MXFP8 forward + BF16 STE backward
# ---------------------------------------------------------------------------
class _MXFP8Matmul(torch.autograd.Function):
    """MXFP8 forward (b12x dense GEMM) with STE BF16 backward."""

    @staticmethod
    def forward(ctx, x, w_bf16, bias, module):  # type: ignore[override]
        N = w_bf16.shape[0]
        packed = module._get_or_refresh_pack(w_bf16)
        mxfp8_linear = module._mxfp8_linear_fn
        # b12x does the activation MXFP8 quant internally and returns
        # BF16. ``expected_m`` defaults to the token count (autotune
        # hint); bias is added inside mxfp8_linear if passed, but we
        # add it in the STE-friendly BF16 path here for clarity.
        out = mxfp8_linear(x, packed, bias=None)
        if bias is not None:
            out = out + bias

        ctx.save_for_backward(x, w_bf16)
        ctx.has_bias = bias is not None
        return out

    @staticmethod
    def backward(ctx, grad_out):  # type: ignore[override]
        x, w = ctx.saved_tensors
        N = w.shape[0]
        K = w.shape[1]

        grad_out_2d = grad_out.reshape(-1, N)
        x_2d = x.reshape(-1, K)

        # STE: re-run in BF16 against the BF16 leaf weight.
        grad_x_2d = grad_out_2d @ w                 # [M, K]
        grad_w = grad_out_2d.t() @ x_2d             # [N, K]
        grad_bias = grad_out_2d.sum(dim=0) if ctx.has_bias else None

        grad_x = grad_x_2d.reshape(x.shape)
        # 4 forward inputs: x, w_bf16, bias, module.
        return grad_x.to(x.dtype), grad_w.to(w.dtype), grad_bias, None


# ---------------------------------------------------------------------------
# Module
# ---------------------------------------------------------------------------
class MXFP8Linear(nn.Module):
    """``nn.Linear`` drop-in whose forward GEMM runs in MXFP8 (b12x).

    Storage: BF16 master ``weight`` (the optimizer leaf) + optional
    ``bias``. The MXFP8-packed weight is a derived buffer recomputed
    (and cached) from the master each time the master changes.

    Args:
        in_features  : K (must be a multiple of 32 for the MXFP8 path)
        out_features : N
        bias         : include a bias parameter
        device       : parameter device
        dtype        : parameter dtype (BF16 by default)
        bf16_only    : escape hatch — skip MXFP8, use plain ``F.linear``
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        device=None,
        dtype=None,
        bf16_only: bool = False,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        api = _b12x_api()
        if api is None:
            bf16_only = True
            self._mxfp8_linear_fn = None
            self._pack_fn = None
            self._quantize_fn = None
        else:
            (_, mxfp8_linear, pack_fn, quantize_fn) = api
            self._mxfp8_linear_fn = mxfp8_linear
            self._pack_fn = pack_fn
            self._quantize_fn = quantize_fn

        # MXFP8 block size is 32 along K, but the b12x dense kernel +
        # weight quantizer require K % 128 == 0. K that isn't a
        # multiple of 128 falls back to BF16 (e.g. a small inner
        # sequential-gate Linear with K=32).
        if not bf16_only and in_features % _MXFP8_K_MULTIPLE != 0:
            bf16_only = True
        self.bf16_only = bf16_only

        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, device=device, dtype=dtype or torch.bfloat16),
        )
        if bias:
            self.bias = nn.Parameter(
                torch.empty(out_features, device=device, dtype=dtype or torch.bfloat16),
            )
        else:
            self.register_parameter("bias", None)

        # Pack cache (Strategy 2). Holds the MXFP8LinearWeight keyed by
        # the BF16 master's ``_version``. Recomputed only when the
        # optimizer / loader bumps the version.
        self._pack_cache = None
        self._pack_w_ver: int = -1

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            bound = 1.0 / math.sqrt(self.in_features) if self.in_features > 0 else 0.0
            nn.init.uniform_(self.bias, -bound, bound)

    def _get_or_refresh_pack(self, w_bf16: torch.Tensor):
        """Quantize + pack the BF16 master to MXFP8, cached by version."""
        w_ver = w_bf16._version
        if self._pack_cache is None or self._pack_w_ver != w_ver:
            # quantize_mxfp8_rows_torch returns FP8 E4M3 values [N, K]
            # + UE8M0 scale_rows (1, N, K/32). pack expects [N, K/32].
            rows = self._quantize_fn(w_bf16)
            scale_rows_2d = rows.scale_rows.squeeze(0)  # [N, K/32]
            self._pack_cache = self._pack_fn(rows.values, scale_rows_2d)
            self._pack_w_ver = w_ver
        return self._pack_cache

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.bf16_only:
            return F.linear(x, self.weight, self.bias)
        return _MXFP8Matmul.apply(x, self.weight, self.bias, self)

    def extra_repr(self) -> str:
        if self.bf16_only:
            storage = "BF16(bf16_only)"
        else:
            storage = "MXFP8-E4M3 (per-32-block UE8M0, b12x fwd, BF16 STE bwd)"
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, weight_storage={storage}"
        )
