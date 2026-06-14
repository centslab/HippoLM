"""Backwards-compat shim: GDN2 lives in :mod:`src.models.ops.gdn2`.

The GDN2 layer is the successor to KDA. The vendored kernel and layer are
kept under :mod:`src.models.ops._vendored.fla` so they remain tunable in-tree;
this module re-exports the :class:`GDN2` wrapper from
:mod:`src.models.ops.gdn2` for code that imports the model layers by name.
"""
from src.models.ops.gdn2 import GDN2

__all__ = ["GDN2"]
