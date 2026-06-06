"""HippoLM model operations.

This package contains:
- The attention / MLP operations used by HippoModel:
  * :class:`BlockAttnRes` -- Block Attention Residuals
  * :class:`KDA`          -- Kimi Delta Attention wrapper
- ``_vendored.fla``        -- vendored copy of the fla library
                              (``flash-linear-attention``) for in-tree
                              tuning of the KDA chunked kernel
- ``cuda``                 -- custom CUDA kernels (planned)
"""
from src.models.ops.attn_res import BlockAttnRes
from src.models.ops.kda import KDA

__all__ = ["BlockAttnRes", "KDA"]
