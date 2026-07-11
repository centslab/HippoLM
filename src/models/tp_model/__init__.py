"""Tensor-parallel HippoModel package.

Layout
------

The 1398-line ``tp_model.py`` was split in June 2026 along
class boundaries (no logic changes). Each module owns one
self-contained component:

  * :mod:`._primitives` — TP building blocks (:class:`ColumnParallelLinear`,
    :class:`RowParallelLinear`, TP process-group lifecycle, all-reduce
    autograd functions). Moved here from the old top-level
    :mod:`src.models.tp_layers` module during the 2026-07 refactor.
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
server, all tests). ``TPHippoModel`` is loaded lazily via the
module-level :pep:`562` ``__getattr__`` below — eager-loading it
here would create a circular import because ``tp_model.model``
imports :class:`BlockAttnRes` from :mod:`src.models.ops.attn_res`,
which itself imports :mod:`._primitives` from this package (a
chicken-and-egg loop triggered any time someone imports
:mod:`src.models.ops.attn_res` before :mod:`src.models.tp_model`
has finished initialising).
"""
from __future__ import annotations

from .embed import TPShardedEmbed, _ShardedEmbedLookup
from .kda import TPKDA
from .layer import TPHippoLayer
from .lm_head import TPFusedLceLoss, TPLmHead, _TiedFusedLCEFunction
from .swiglu import TPSwiGLU


def __getattr__(name: str):
    """PEP 562 lazy attribute access.

    Defers the ``TPHippoModel`` import to attribute-access time
    so that :mod:`src.models.ops.attn_res` can import
    :mod:`._primitives` from this package during its own
    initialisation without triggering a circular import through
    :mod:`.model` (which depends on :class:`BlockAttnRes`).
    """
    if name == "TPHippoModel":
        from .model import TPHippoModel as _TPHippoModel
        # Cache on the module so subsequent attribute lookups
        # skip the import dance entirely (Python looks up
        # module attributes via __getattr__ only when normal
        # attribute lookup fails).
        globals()["TPHippoModel"] = _TPHippoModel
        return _TPHippoModel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


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
