# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from src.models.ops._vendored.fla.modules.convolution import ImplicitLongConvolution, LongConvolution, ShortConvolution
from src.models.ops._vendored.fla.modules.fused_bitlinear import BitLinear, FusedBitLinear
from src.models.ops._vendored.fla.modules.fused_cross_entropy import FusedCrossEntropyLoss
from src.models.ops._vendored.fla.modules.fused_kl_div import FusedKLDivLoss
from src.models.ops._vendored.fla.modules.fused_linear_cross_entropy import FusedLinearCrossEntropyLoss
from src.models.ops._vendored.fla.modules.fused_norm_gate import (
    FusedLayerNormGated,
    FusedLayerNormSwishGate,
    FusedLayerNormSwishGateLinear,
    FusedRMSNormGated,
    FusedRMSNormSwishGate,
    FusedRMSNormSwishGateLinear,
)
from src.models.ops._vendored.fla.modules.l2norm import L2Norm
from src.models.ops._vendored.fla.modules.layernorm import GroupNorm, GroupNormLinear, LayerNorm, LayerNormLinear, RMSNorm, RMSNormLinear
from src.models.ops._vendored.fla.modules.mlp import GatedMLP
from src.models.ops._vendored.fla.modules.rotary import RotaryEmbedding
from src.models.ops._vendored.fla.modules.token_shift import TokenShift

__all__ = [
    'BitLinear',
    'FusedBitLinear',
    'FusedCrossEntropyLoss',
    'FusedKLDivLoss',
    'FusedLayerNormGated',
    'FusedLayerNormSwishGate',
    'FusedLayerNormSwishGateLinear',
    'FusedLinearCrossEntropyLoss',
    'FusedRMSNormGated',
    'FusedRMSNormSwishGate',
    'FusedRMSNormSwishGateLinear',
    'GatedMLP',
    'GroupNorm',
    'GroupNormLinear',
    'ImplicitLongConvolution',
    'L2Norm',
    'LayerNorm',
    'LayerNormLinear',
    'LongConvolution',
    'RMSNorm',
    'RMSNormLinear',
    'RotaryEmbedding',
    'ShortConvolution',
    'TokenShift',
]
