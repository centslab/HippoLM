"""Normalization layers for HippoLM.

The :class:`RMSNorm` here is a thin wrapper over fla-org's fused
``rms_norm`` Triton kernel (see
``src/models/ops/_vendored/fla/modules/layernorm.py``). Two reasons
to use the fused kernel rather than a naive Python ``mean + rsqrt +
mul`` composition:

1. **Dtype discipline under autocast.** The naive version calls
   ``x.pow(2).mean(-1, keepdim=True)``; ``mean`` is in PyTorch's
   autocast "promote" list and silently upcasts the result to FP32,
   propagating FP32 through the rest of the chain. Under a BF16
   autocast (the canonical :class:`PrecisionConfig` setting) this
   means the residual stream becomes FP32 for the rest of the
   forward pass, doubling activation memory. The fla kernel keeps
   the normalized output in the input dtype and only allocates
   FP32 for the per-row ``rstd`` (shape ``[N]``, a few KB) and the
   weight parameter — the rest of the saved-for-backward state is
   in the input dtype.

2. **Speed.** The fused Triton kernel does the squared-mean /
   rsqrt / weight-multiply in one launch, no per-op round-trips.

The wrapper preserves the original class name and call site so the
:class:`HippoLayer` and :class:`BlockAttnRes` modules don't need to
change.

Import note: the fla ``layernorm`` module is imported inside
:meth:`forward` rather than at module top-level. The fla
``modules/__init__.py`` triggers a load of
``src.models.ops.__init__.py`` which (via ``attn_res``) re-enters
this module — a circular import that breaks eager top-level import.
The first forward call pays the deferred import cost; subsequent
calls hit Python's import cache.
"""
import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (fla fused).

    Drop-in replacement for the naive ``x * rsqrt(mean(x^2) + eps) * w``
    composition. The ``weight`` parameter is the affine scale; bias
    is not supported (matches the original naive implementation).
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape ``[..., dim]``.

        Returns:
            Normalized tensor of the same shape and dtype as ``x``.
            The internal ``rstd`` (FP32) is computed by the fused
            kernel and not surfaced; the saved-for-backward state
            is the per-row ``rstd`` plus the weight — both small.
        """
        from src.models.ops._vendored.fla.modules.layernorm import rms_norm
        return rms_norm(x, self.weight, None, eps=self.eps)

    def extra_repr(self) -> str:
        return f"eps={self.eps}"
