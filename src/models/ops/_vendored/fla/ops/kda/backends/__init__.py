# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""KDA backends."""

from src.models.ops._vendored.fla.ops.backends import BackendRegistry, dispatch
from src.models.ops._vendored.fla.ops.kda.backends.fastkda import FastKDABackend
from src.models.ops._vendored.fla.ops.kda.backends.flashkda import FlashKDABackend
from src.models.ops._vendored.fla.ops.kda.backends.tilelang import KDATileLangBackend

kda_registry = BackendRegistry("kda")
# FastKDA (in-tree Triton, priority 2) is registered first; it
# handles inference and rejects training (fwd-only) so the FLA
# default chunk path is used for backward. FlashKDA (priority 3) is
# the external-package fallback; TileLang is registered last.
kda_registry.register(FastKDABackend())
kda_registry.register(FlashKDABackend())
kda_registry.register(KDATileLangBackend())


__all__ = ['dispatch', 'kda_registry']
