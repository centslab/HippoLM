"""Backwards-compat shim: KDA moved to :mod:`src.models.ops.kda`.

The KDA implementation has moved into :mod:`src.models.ops` as part of
the v0.0.1 ops refactor so that the chunked-parallel kernel and its
Triton sources are tunable in-tree. This shim re-exports
:class:`KDA` from the new location.
"""
from src.models.ops.kda import KDA

__all__ = ["KDA"]
