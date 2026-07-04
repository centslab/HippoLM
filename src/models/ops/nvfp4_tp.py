"""NVFP4 Column/Row-Parallel linear layers for the TP path.

These mirror :class:`ColumnParallelLinear` and :class:`RowParallelLinear`
in :mod:`src.models.tp_layers`, but store their weights in NVFP4 packed
format. The sharding math (out/world for column, in/world for row) is
identical to the BF16 versions; the only thing that changes is the
storage dtype and the matmul implementation.

Construction
------------
Both modules accept the **logical** ``in_features`` / ``out_features``
(what the rest of the model sees) and compute the per-partition shape
from the TP world size. The BF16 master weight is the per-partition
shape; the packed buffers are derived from it via ``repack_weights``.
This matches the existing BF16 TP layers' state-dict shape contract.

Forward
-------
Column-parallel: dequantize FP4 -> BF16 (per partition), ``F.linear``.
Row-parallel:    dequantize FP4 -> BF16 (per partition), ``F.linear``,
                  then all-reduce the output (reuses the standard
                  :func:`tp_all_reduce`).

Backward
--------
Standard BF16 matmul backward (STE for the quantize noise), same as the
non-parallel :class:`NVFP4Linear`. The all-reduce on the row-parallel
backward is handled by the existing :class:`tp_all_reduce` machinery —
its backward is identity (sum's gradient is 1), so grad_out flows
through unchanged into the local matmul backward.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.ops.nvfp4 import dequantize_nvfp4
from src.models.tp_layers import _TP_WORLD_SIZE, _TP_RANK, tp_all_reduce


class _NVFP4RowMatmul(torch.autograd.Function):
    """Row-parallel NVFP4 matmul: dequant -> BF16 matmul -> all-reduce.

    The column-parallel case is byte-identical to the non-parallel
    :class:`_NVFP4Matmul` (no all-reduce, bias added before matmul),
    so it reuses that class directly. Only the row-parallel path
    needs its own Function because the bias is added AFTER the
    all-reduce (matching :class:`RowParallelLinear`).

    Backward: the all-reduce's backward is identity (sum's gradient
    is 1 on each rank), so grad_out flows through unchanged into the
    standard BF16 matmul backward below.
    """

    @staticmethod
    def forward(  # type: ignore[override]
        ctx, x, w_master, packed_w, scales_w, bias, K_orig, block_size,
    ):
        w_bf16 = dequantize_nvfp4(
            packed_w, scales_w, K_orig,
            block_size=block_size, out_dtype=x.dtype,
        )
        ctx.save_for_backward(x, w_bf16)
        ctx.has_bias = bias is not None
        out = F.linear(x, w_bf16, None)  # bias added after all-reduce
        # Match the RowParallelLinear contract: only rank 0 adds the
        # bias so the all-reduce sums to one effective bias.
        if bias is not None and _TP_RANK == 0:
            out = out + bias
        return tp_all_reduce(out)

    @staticmethod
    def backward(ctx, grad_out):  # type: ignore[override]
        x, w_bf16 = ctx.saved_tensors
        K = w_bf16.shape[1]
        N = w_bf16.shape[0]
        grad_out_2d = grad_out.reshape(-1, N)
        x_2d = x.reshape(-1, K)
        grad_x = grad_out @ w_bf16
        grad_w = (grad_out_2d.t().to(x_2d.dtype)) @ x_2d
        grad_bias = grad_out_2d.sum(dim=0) if ctx.has_bias else None
        return grad_x, grad_w, None, None, grad_bias, None, None


# ---------------------------------------------------------------------------
# nn.Module wrappers
# ---------------------------------------------------------------------------
class NVFP4ColumnParallelLinear(nn.Module):
    """Column-parallel NVFP4 linear.

    Weight shape: ``[out_features_per_partition, in_features]`` (BF16 master).

    Optional ``prenorm`` mode absorbs a pre-RMSNorm into the forward:
    the module holds its own ``norm_weight`` parameter and the forward
    applies ``rms_norm_linear`` (RMSNorm + the NVFP4 BF16-activation
    matmul fused in one autograd Function). The fused path drops one
    full ``x`` save vs the unfused ``rms_norm`` → ``_NVFP4Matmul``
    chain (~48 MiB per layer at prod shape).
    """

    def __init__(
        self, in_features: int, out_features: int, bias: bool = False,
        device=None, dtype=None, block_size: int = 16,
        prenorm: bool = False, norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        assert block_size == 16, f"NVFP4 block_size must be 16, got {block_size}"
        world = _TP_WORLD_SIZE
        assert out_features % world == 0, (
            f"out_features={out_features} not divisible by tp_world={world}"
        )
        self.in_features = in_features
        self.out_features = out_features
        self.out_features_per_partition = out_features // world
        self.block_size = block_size
        self.prenorm = prenorm
        self.norm_eps = norm_eps

        self.weight = nn.Parameter(
            torch.empty(
                self.out_features_per_partition, in_features,
                device=device, dtype=dtype or torch.bfloat16,
            )
        )
        K_padded = in_features + (-in_features % block_size)
        self.register_buffer(
            "packed_weight",
            torch.zeros(
                self.out_features_per_partition, K_padded // 2,
                device=device, dtype=torch.uint8,
            ),
        )
        self.register_buffer(
            "scales",
            torch.zeros(
                self.out_features_per_partition, K_padded // block_size,
                device=device, dtype=torch.float8_e4m3fn,
            ),
        )
        if bias:
            self.bias = nn.Parameter(
                torch.empty(
                    self.out_features_per_partition,
                    device=device, dtype=dtype or torch.bfloat16,
                )
            )
        else:
            self.register_parameter("bias", None)
        if prenorm:
            # Replicated RMSNorm weight — not TP-sharded (the residual
            # stream is replicated, so the norm is too). Same dtype as
            # the activation so fla rms_norm_linear can run without
            # dtype juggling.
            self.norm_weight = nn.Parameter(
                torch.ones(in_features, device=device, dtype=dtype or torch.bfloat16)
            )
        else:
            self.register_parameter("norm_weight", None)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
        self.repack_weights()

    @torch.no_grad()
    def repack_weights(self) -> None:
        from src.models.ops.nvfp4 import quantize_nvfp4
        packed, scales = quantize_nvfp4(self.weight.data, block_size=self.block_size)
        self.packed_weight.copy_(packed)
        self.scales.copy_(scales)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.prenorm:
            from src.models.ops.nvfp4_linear import _RmsNormNvfp4Matmul
            return _RmsNormNvfp4Matmul.apply(
                x, self.norm_weight, self.weight,
                self.packed_weight, self.scales, self.bias,
                self.in_features, self.block_size, self.norm_eps,
            )
        # The column-parallel case is byte-identical to the base
        # _NVFP4Matmul (no all-reduce, bias in the F.linear call),
        # so we reuse that Function rather than re-implementing.
        from src.models.ops.nvfp4_linear import _NVFP4Matmul
        return _NVFP4Matmul.apply(
            x, self.weight, self.packed_weight, self.scales, self.bias,
            self.in_features, self.block_size,
        )


class NVFP4RowParallelLinear(nn.Module):
    """Row-parallel NVFP4 linear with all-reduce on output.

    Weight shape: ``[out_features, in_features_per_partition]`` (BF16 master).
    """

    def __init__(
        self, in_features: int, out_features: int, bias: bool = False,
        device=None, dtype=None, block_size: int = 16,
    ) -> None:
        super().__init__()
        assert block_size == 16, f"NVFP4 block_size must be 16, got {block_size}"
        world = _TP_WORLD_SIZE
        assert in_features % world == 0, (
            f"in_features={in_features} not divisible by tp_world={world}"
        )
        self.in_features = in_features
        self.out_features = out_features
        self.in_features_per_partition = in_features // world
        self.block_size = block_size

        self.weight = nn.Parameter(
            torch.empty(
                out_features, self.in_features_per_partition,
                device=device, dtype=dtype or torch.bfloat16,
            )
        )
        K_padded = self.in_features_per_partition + (-self.in_features_per_partition % block_size)
        self.register_buffer(
            "packed_weight",
            torch.zeros(
                out_features, K_padded // 2,
                device=device, dtype=torch.uint8,
            ),
        )
        self.register_buffer(
            "scales",
            torch.zeros(
                out_features, K_padded // block_size,
                device=device, dtype=torch.float8_e4m3fn,
            ),
        )
        if bias:
            self.bias = nn.Parameter(
                torch.empty(out_features, device=device, dtype=dtype or torch.bfloat16)
            )
        else:
            self.register_parameter("bias", None)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
        self.repack_weights()

    @torch.no_grad()
    def repack_weights(self) -> None:
        from src.models.ops.nvfp4 import quantize_nvfp4
        packed, scales = quantize_nvfp4(self.weight.data, block_size=self.block_size)
        self.packed_weight.copy_(packed)
        self.scales.copy_(scales)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _NVFP4RowMatmul.apply(
            x, self.weight, self.packed_weight, self.scales, self.bias,
            self.in_features_per_partition, self.block_size,
        )