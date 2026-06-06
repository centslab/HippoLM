"""Backwards-compat shim: BlockAttnRes moved to :mod:`src.models.ops.attn_res`.

The Block Attention Residuals implementation has moved into
:mod:`src.models.ops` as part of the v0.0.1 ops refactor. This shim
re-exports :class:`BlockAttnRes` from the new location.
"""
from src.models.ops.attn_res import BlockAttnRes

__all__ = ["BlockAttnRes"]
