"""Tensor-parallel HippoModel package.

Layout
------

The 1398-line ``tp_model.py`` was split in June 2026 along
class boundaries (no logic changes). Each module owns one
self-contained component:

  * :mod:`.embed`  — :class:`TPShardedEmbed` (vocab-parallel
    embedding) and the :class:`_ShardedEmbedLookup` custom
    autograd that powers it.
  * :mod:`.swiglu` — :class:`TPSwiGLU` (column-row parallel
    SwiGLU FFN with fused gate+up projection).
  * :mod:`.lm_head` — :class:`TPFusedLceLoss` (training path:
    fused matmul + cross-entropy with online chunked softmax)
    and :class:`TPLmHead` (inference shim that returns the
    full sharded logits).
  * :mod:`.kda`    — :class:`TPKDA` (column-parallel Q/K/V +
    row-parallel O projection; the KDA chunkwise op itself
    is rank-local, no inter-rank comm).
  * :mod:`.layer`  — :class:`TPHippoLayer` (one
    attn_norm → TPKDA → mlp_norm → SwiGLU residual block).
  * :mod:`.model`  — :class:`TPHippoModel` (the full model:
    embed, per-block AttnRes at block boundaries, all layers,
    lm_head + fused CE).

Public API re-exports
---------------------

The package re-exports every public class so that
``from src.models.tp_model import TPHippoModel`` keeps
working for every existing caller (training loop, eval
server, all tests).
"""
from __future__ import annotations

from .embed import TPShardedEmbed, _ShardedEmbedLookup
from .kda import TPKDA
from .layer import TPHippoLayer
from .lm_head import TPFusedLceLoss, TPLmHead, _TiedFusedLCEFunction
from .model import TPHippoModel
from .swiglu import TPSwiGLU

__all__ = [
    "TPHippoModel",
    "TPHippoLayer",
    "TPKDA",
    "TPSwiGLU",
    "TPShardedEmbed",
    "TPFusedLceLoss",
    "TPLmHead",
    # Private but imported by tests
    "_ShardedEmbedLookup",
    "_TiedFusedLCEFunction",
]
