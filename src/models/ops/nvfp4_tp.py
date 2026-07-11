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

Mode (3) — no BF16 master
-------------------------
Passing ``no_bf16_master=True`` (Marlin-only) drops the BF16 master
weight and lets the optimizer stream-dequant the FP4 buffers at chunk
granularity (see :class:`NVFP4Linear` in :mod:`nvfp4_linear`). For
column-parallel this just changes the autograd Function signature; for
row-parallel we add a dedicated ``_MarlinNvFp4RowMatmulNoLeafImpl`` (the
all-reduce contract differs from the base class).

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
through unchanged into the local matmul backward. In mode (3) ``grad_w``
is stashed on the owning module via a side-channel; the custom
optimizer step in :mod:`src.training.param_offload` consumes it.
"""
from __future__ import annotations

import ctypes
import weakref
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.ops.nvfp4 import dequantize_nvfp4
from src.models.ops.nvfp4_linear import (
    _NVFP4NoLeafMatmul,
    _nvfp4_post_load_repack_hook,
)
from src.models.tp_model._primitives import _TP_WORLD_SIZE, _TP_RANK, tp_all_reduce


class _NVFP4RowMatmul(torch.autograd.Function):
    """Row-parallel NVFP4 matmul: dequant -> BF16 matmul -> all-reduce.

    **Mode (2) only** (legacy BF16 master present). The column-parallel
    case is byte-identical to the non-parallel
    :class:`nvfp4_linear._NVFP4Matmul` (no all-reduce, bias added
    before matmul), so it reuses that class directly. Only the
    row-parallel path needs its own Function because the bias is added
    AFTER the all-reduce (matching :class:`RowParallelLinear`).

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


class _NVFP4RowMatmulNoLeaf(torch.autograd.Function):
    """Row-parallel NVFP4 matmul — no BF16 master, side-channel grad_w.

    Forward: dequant FP4 -> BF16, F.linear, bias-on-rank-0-after, all-
    reduce. Identical TP contract to :class:`_NVFP4RowMatmul`.

    Backward: ``grad_x = grad_out @ W^T`` (the all-reduce's backward is
    identity so grad_out flows through unchanged). ``grad_w`` is stashed
    on the owning module via a weakref captured at forward time.
    """

    @staticmethod
    def forward(  # type: ignore[override]
        ctx, x, packed_w, scales_w, bias, K_orig, block_size, module_ref,
    ):
        w_bf16 = dequantize_nvfp4(
            packed_w, scales_w, K_orig,
            block_size=block_size, out_dtype=x.dtype,
        )
        ctx.save_for_backward(x, w_bf16)
        ctx.has_bias = bias is not None
        ctx.module_ref = module_ref
        ctx.K_orig = K_orig
        ctx.input_shape = x.shape
        out = F.linear(x, w_bf16, None)  # bias added after all-reduce
        if bias is not None and _TP_RANK == 0:
            out = out + bias
        return tp_all_reduce(out)

    @staticmethod
    def backward(ctx, grad_out):  # type: ignore[override]
        x, w_bf16 = ctx.saved_tensors
        K = ctx.K_orig
        N = w_bf16.shape[0]
        grad_out_2d = grad_out.reshape(-1, N)
        x_2d = x.reshape(-1, K)
        grad_x = grad_out @ w_bf16
        grad_w = (grad_out_2d.t().to(x_2d.dtype)) @ x_2d
        grad_bias = grad_out_2d.sum(dim=0) if ctx.has_bias else None
        module = ctx.module_ref()
        if module is not None:
            module._stash_grad_w(grad_w)
        return grad_x, None, None, grad_bias, None, None, None


class _MarlinNvFp4RowMatmul(torch.autograd.Function):
    """Row-parallel NVFP4 matmul via Marlin fused kernel + all-reduce.

    Same TP contract as :class:`_NVFP4RowMatmul`: bias added by rank 0
    after the all-reduce. The forward uses vLLM's Marlin FP4 kernel
    (BF16 MMA + register dequant + cp.async double-buffered prefetch)
    for ~3.5x speedup over the dequant+cuBLAS path at FFN shapes.

    Backward: standard BF16 matmul backward on the BF16 master weight
    (STE for the quantize noise). The all-reduce's backward is identity.

    The ``scales_for_kernel`` / ``global_scale_adj`` buffers are precomputed
    by ``_build_marlin_scales_caches`` and stashed on the owning module
    (see ``NVFP4RowParallelLinear.repack_weights``). The large
    ``repacked`` buffer is recomputed each forward inside
    ``_MarlinNvFp4Matmul.forward`` — see ``_build_marlin_scales_caches``
    for the design note.

    **Mode (2) only** (BF16 master present). Mode (3) variant is
    :class:`_MarlinNvFp4RowMatmulNoLeaf` below.
    """

    @staticmethod
    def forward(  # type: ignore[override]
        ctx, x, w_master, packed_w, scales_for_kernel, global_scale_adj, bias, K_orig, block_size,
        scales_e4m3=None, global_scale=None,
    ):
        from src.models.ops.nvfp4_marlin import marlin_nvfp4_matmul
        out = marlin_nvfp4_matmul(
            x, w_master, packed_w, scales_for_kernel, global_scale_adj, None,
            scales_e4m3, global_scale,
        )
        ctx.save_for_backward(x, w_master)
        ctx.has_bias = bias is not None
        ctx.bwd_packed_w = packed_w
        ctx.bwd_scales_e4m3 = scales_e4m3
        ctx.bwd_global_scale = global_scale
        ctx.bwd_block_size = block_size
        ctx.bwd_N = w_master.shape[0]
        ctx.bwd_K = K_orig
        # Stash the original (unflattened) input shape so backward can
        # reshape the 2-D kernel output back to whatever the upstream
        # caller passed in (e.g. [B, T, K] for a 3-D activation).
        ctx.bwd_input_shape = x.shape
        if bias is not None and _TP_RANK == 0:
            out = out + bias
        return tp_all_reduce(out)

    @staticmethod
    def backward(ctx, grad_out):  # type: ignore[override]
        from src.models.ops.nvfp4_marlin import _marlin_bwd_grad_x
        x, w_bf16 = ctx.saved_tensors
        K = ctx.bwd_K
        N = ctx.bwd_N
        grad_out_2d = grad_out.reshape(-1, N)
        x_2d = x.reshape(-1, K)
        if (ctx.bwd_packed_w is not None
                and ctx.bwd_scales_e4m3 is not None
                and ctx.bwd_global_scale is not None):
            grad_x_2d = _marlin_bwd_grad_x(
                grad_out_2d, ctx.bwd_packed_w, ctx.bwd_scales_e4m3,
                ctx.bwd_global_scale, N, K, ctx.bwd_block_size,
            )
            # Reshape back to the original (unflattened) input shape.
            grad_x = grad_x_2d.reshape(*ctx.bwd_input_shape[:-1], K)
        else:
            grad_x = grad_out @ w_bf16
        grad_w = (grad_out_2d.t().to(x_2d.dtype)) @ x_2d
        grad_bias = grad_out_2d.sum(dim=0) if ctx.has_bias else None
        return grad_x, grad_w, None, None, None, grad_bias, None, None, None, None


class _MarlinNvFp4RowMatmulNoLeafImpl(torch.autograd.Function):
    """Row-parallel Marlin FP4 matmul — mode (3), no BF16 master.

    Forward: Marlin FP4 kernel call, bias-on-rank-0, all-reduce.
    Backward: Marlin bwd grad_x + dequantized W for the BF16 fallback;
    ``grad_w`` stashed on the module via weakref.
    """

    @staticmethod
    def forward(  # type: ignore[override]
        ctx, x, packed_w, scales_for_kernel, global_scale_adj, bias, K_orig, block_size,
        scales_e4m3, global_scale, module_ref,
    ):
        from src.models.ops.nvfp4_marlin import (
            _ensure_libs_loaded, _repack_for_marlin, _kbfloat16, _kfe2m1f,
            _kfe4m3fn, _marlin_mm_fn,
            dequantize_marlin_nvfp4 as dequantize_nvfp4,
            _marlin_bwd_grad_x,
        )
        _ensure_libs_loaded()
        assert x.dtype == torch.bfloat16
        orig_shape = x.shape
        K = K_orig
        x_2d = x.reshape(-1, K)
        M = x_2d.shape[0]
        # packed_w is 2-D [N, K//2]; the leading dim IS N.
        N = packed_w.shape[0]

        repacked = _repack_for_marlin(packed_w, size_k=K, size_n=N)
        out = torch.empty(M, N, dtype=torch.bfloat16, device=x.device)
        stream = torch.cuda.current_stream().cuda_stream

        workspace = torch.zeros(132 * 128, dtype=torch.int32, device=x.device)
        empty_f32 = torch.empty(0, dtype=torch.float32, device=x.device)
        empty_bf16 = torch.empty(0, dtype=torch.bfloat16, device=x.device)
        empty_i32 = torch.empty(0, dtype=torch.int32, device=x.device)

        a_type = _kbfloat16()
        b_type = _kfe2m1f()
        c_type = _kbfloat16()
        s_type = _kfe4m3fn()

        num_groups = K // block_size
        sms = torch.cuda.get_device_properties(0).multi_processor_count

        _marlin_mm_fn(  # type: ignore[name-defined]
            ctypes.c_void_p(x_2d.data_ptr()),
            ctypes.c_void_p(repacked.data_ptr()),
            ctypes.c_void_p(out.data_ptr()),
            ctypes.c_void_p(empty_f32.data_ptr()),
            ctypes.c_void_p(empty_bf16.data_ptr()),
            ctypes.c_void_p(empty_f32.data_ptr()),
            ctypes.c_void_p(scales_for_kernel.data_ptr()),
            ctypes.c_void_p(global_scale_adj.data_ptr()),
            ctypes.c_void_p(empty_i32.data_ptr()),
            ctypes.c_void_p(empty_i32.data_ptr()),
            ctypes.c_void_p(empty_i32.data_ptr()),
            ctypes.c_void_p(empty_bf16.data_ptr()),
            ctypes.c_int(M), ctypes.c_int(N), ctypes.c_int(K),
            ctypes.c_int(x_2d.stride(0)),
            ctypes.c_void_p(workspace.data_ptr()),
            ctypes.c_void_p(ctypes.addressof(a_type)),
            ctypes.c_void_p(ctypes.addressof(b_type)),
            ctypes.c_void_p(ctypes.addressof(c_type)),
            ctypes.c_void_p(ctypes.addressof(s_type)),
            ctypes.c_bool(False), ctypes.c_bool(False), ctypes.c_bool(True), ctypes.c_bool(False),
            ctypes.c_int(num_groups), ctypes.c_int(block_size), ctypes.c_int(0),
            ctypes.c_void_p(stream),
            ctypes.c_int(-1), ctypes.c_int(-1), ctypes.c_int(sms),
            ctypes.c_bool(False), ctypes.c_bool(False), ctypes.c_bool(False),
        )

        ctx.save_for_backward(x)
        ctx.has_bias = bias is not None
        ctx.module_ref = module_ref
        ctx.bwd_packed_w = packed_w
        ctx.bwd_scales_e4m3 = scales_e4m3
        ctx.bwd_global_scale = global_scale
        ctx.bwd_block_size = block_size
        ctx.bwd_N = N
        ctx.bwd_K = K
        ctx.bwd_input_shape = x.shape
        out = out.reshape(*orig_shape[:-1], N)
        if bias is not None and _TP_RANK == 0:
            out = out + bias
        return tp_all_reduce(out)

    @staticmethod
    def backward(ctx, grad_out):  # type: ignore[override]
        (x,) = ctx.saved_tensors
        # Dequant FP4 → BF16 (the no-leaf Function saves only x).
        # Use the Marlin-aware dequant that includes global_scale
        # (the legacy ``dequantize_nvfp4`` from nvfp4.py doesn't).
        from src.models.ops.nvfp4_marlin import dequantize_marlin_nvfp4
        w_bf16 = dequantize_marlin_nvfp4(
            ctx.bwd_packed_w, ctx.bwd_scales_e4m3,
            ctx.bwd_global_scale, ctx.bwd_K,
            block_size=ctx.bwd_block_size, out_dtype=x.dtype,
        )
        K = w_bf16.shape[1]
        N = w_bf16.shape[0]
        grad_out_2d = grad_out.reshape(-1, N)
        x_2d = x.reshape(-1, K)
        # BF16 cuBLAS bwd (Marlin FP4 bwd kernel produces NaN/Inf
        # at FFN scales per project memory — stay with the BF16
        # path the legacy tests rely on).
        grad_x = grad_out @ w_bf16
        grad_w = (grad_out_2d.t().to(x_2d.dtype)) @ x_2d
        grad_bias = grad_out_2d.sum(dim=0) if ctx.has_bias else None
        module = ctx.module_ref()
        if module is not None:
            module._stash_grad_w(grad_w)
        return grad_x, None, None, None, None, grad_bias, None, None, None, None, None


# ---------------------------------------------------------------------------
# nn.Module wrappers
# ---------------------------------------------------------------------------
class NVFP4ColumnParallelLinear(nn.Module):
    """Column-parallel NVFP4 linear.

    Weight shape: ``[out_features_per_partition, in_features]`` (BF16 master
    in mode 2; FP4 only in mode 3).
    """

    def __init__(
        self, in_features: int, out_features: int, bias: bool = False,
        device=None, dtype=None, block_size: int = 16,
        bf16_only: bool = False,
        use_marlin: bool = False,
        no_bf16_master: bool = False,
    ) -> None:
        super().__init__()
        assert block_size == 16, f"NVFP4 block_size must be 16, got {block_size}"
        world = _TP_WORLD_SIZE
        assert out_features % world == 0, (
            f"out_features={out_features} not divisible by tp_world={world}"
        )
        if no_bf16_master and not use_marlin:
            raise ValueError(
                "no_bf16_master=True currently requires use_marlin=True"
            )
        self.in_features = in_features
        self.out_features = out_features
        self.out_features_per_partition = out_features // world
        self.block_size = block_size
        self.bf16_only = bf16_only
        self.use_marlin = use_marlin
        self.no_bf16_master = no_bf16_master

        # Mode (2) only — see NVFP4Linear for the rationale.
        if not no_bf16_master and not bf16_only:
            self.weight = nn.Parameter(
                torch.empty(
                    self.out_features_per_partition, in_features,
                    device=device, dtype=dtype or torch.bfloat16,
                )
            )

        K_padded = in_features + (-in_features % block_size)
        if not bf16_only:
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
            if use_marlin:
                self.register_buffer(
                    "global_scale",
                    torch.ones(1, dtype=torch.float32, device=device),
                )
                self.register_buffer(
                    "_scales_for_kernel",
                    torch.empty(0, dtype=torch.float8_e4m3fn, device=device),
                    persistent=False,
                )
                self.register_buffer(
                    "_global_scale_adj",
                    torch.ones(1, dtype=torch.float32, device=device),
                    persistent=False,
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
        # Side-channel state (mode 3).
        self._latest_grad_w: Optional[torch.Tensor] = None
        self.register_load_state_dict_post_hook(_nvfp4_post_load_repack_hook)
        self._init_weights()

    # ------------------------------------------------------------------
    # Mode (3) chunk API — delegates to the same logic as NVFP4Linear
    # (we keep the storage contract identical so the optimizer can
    # treat base and TP variants uniformly).
    # ------------------------------------------------------------------
    def material_chunk(self, start: int, end: int) -> torch.Tensor:
        from src.models.ops.nvfp4_marlin import dequantize_marlin_nvfp4
        packed_chunk = self.packed_weight[start:end].contiguous()
        scales_chunk = self.scales[start:end].contiguous()
        return dequantize_marlin_nvfp4(
            packed_chunk, scales_chunk, self.global_scale,
            self.in_features, self.block_size,
        )

    def commit_chunk(self, start: int, end: int, bf16_chunk: torch.Tensor) -> None:
        from src.models.ops.nvfp4_marlin import quantize_nvfp4_with_global_scale
        packed, scales, _ = quantize_nvfp4_with_global_scale(
            bf16_chunk, block_size=self.block_size,
        )
        self.packed_weight[start:end].copy_(packed)
        self.scales[start:end].copy_(scales)

    def material_bf16_view(self) -> torch.Tensor:
        from src.models.ops.nvfp4_marlin import dequantize_marlin_nvfp4
        return dequantize_marlin_nvfp4(
            self.packed_weight, self.scales, self.global_scale,
            self.in_features, self.block_size,
        )

    def commit_bf16_view(self, bf16: torch.Tensor) -> None:
        from src.models.ops.nvfp4_marlin import (
            quantize_nvfp4_with_global_scale, _build_marlin_scales_caches,
        )
        packed, scales, gs = quantize_nvfp4_with_global_scale(
            bf16, block_size=self.block_size,
        )
        self.packed_weight.copy_(packed)
        self.scales.copy_(scales)
        self.global_scale.copy_(gs.view(1))
        _build_marlin_scales_caches(
            self, self.scales, self.global_scale,
            size_k=self.in_features, size_n=self.out_features_per_partition,
            block_size=self.block_size,
        )

    def apply_chunk_update(
        self, start: int, end: int, grad_chunk: torch.Tensor, factor: float,
    ) -> None:
        bf16 = self.material_chunk(start, end)
        bf16.sub_(grad_chunk.to(bf16.dtype), alpha=factor)
        self.commit_chunk(start, end, bf16)
        from src.models.ops.nvfp4_marlin import _build_marlin_scales_caches
        _build_marlin_scales_caches(
            self, self.scales, self.global_scale,
            size_k=self.in_features, size_n=self.out_features_per_partition,
            block_size=self.block_size,
        )

    def decay_and_apply_chunk(
        self,
        start: int, end: int,
        update_chunk: torch.Tensor, lr: float,
        wd_factor: float = 1.0,
    ) -> None:
        """Muon-style apply. See ``NVFP4Linear.decay_and_apply_chunk``
        for the math. ``start`` / ``end`` are local-partition rows
        (``out_features_per_partition``).
        """
        bf16 = self.material_chunk(start, end)
        if wd_factor != 1.0:
            bf16.mul_(wd_factor)
        bf16.sub_(update_chunk.to(bf16.dtype), alpha=lr)
        self.commit_chunk(start, end, bf16)
        from src.models.ops.nvfp4_marlin import _build_marlin_scales_caches
        _build_marlin_scales_caches(
            self, self.scales, self.global_scale,
            size_k=self.in_features, size_n=self.out_features_per_partition,
            block_size=self.block_size,
        )

    def chunk_ranges(self) -> list[tuple[int, int]]:
        rows_per_chunk = max(1, (4 * 1024 * 1024) // max(1, self.in_features))
        ranges = []
        start = 0
        while start < self.out_features_per_partition:
            end = min(start + rows_per_chunk, self.out_features_per_partition)
            ranges.append((start, end))
            start = end
        return ranges

    def _stash_grad_w(self, grad_w: torch.Tensor) -> None:
        self._latest_grad_w = grad_w.detach()

    def _consume_grad_w(self) -> Optional[torch.Tensor]:
        g = self._latest_grad_w
        self._latest_grad_w = None
        return g

    def _init_weights(self) -> None:
        if self.bf16_only:
            if self.bias is not None:
                nn.init.zeros_(self.bias)
            return
        if not self.no_bf16_master:
            nn.init.xavier_uniform_(self.weight)
            if self.bias is not None:
                nn.init.zeros_(self.bias)
            self.repack_weights()
            return
        # Mode (3) — init FP4 buffers from a small Xavier seed.
        if self.bias is not None:
            nn.init.zeros_(self.bias)
        from src.models.ops.nvfp4_marlin import quantize_nvfp4_with_global_scale
        N, K = self.out_features_per_partition, self.in_features
        for start in range(0, N, 64):
            end = min(start + 64, N)
            chunk = torch.empty(
                end - start, K, device=self.packed_weight.device,
                dtype=torch.bfloat16,
            )
            nn.init.xavier_uniform_(chunk)
            packed, scales, _ = quantize_nvfp4_with_global_scale(
                chunk, block_size=self.block_size,
            )
            self.packed_weight[start:end].copy_(packed)
            self.scales[start:end].copy_(scales)
        if self.use_marlin:
            # Populate the Marlin-side caches. The legacy constructor
            # calls repack_weights() which does this; mode-3 skips
            # the re-quantization (no BF16 master to quantize from)
            # but still needs the cache buffers populated, otherwise
            # the kernel reads from an empty buffer and crashes.
            from src.models.ops.nvfp4_marlin import _build_marlin_scales_caches
            _build_marlin_scales_caches(
                self, self.scales, self.global_scale,
                size_k=self.in_features,
                size_n=self.out_features_per_partition,
                block_size=self.block_size,
            )

    @torch.no_grad()
    def repack_weights(self) -> None:
        if self.bf16_only:
            return
        if self.use_marlin:
            from src.models.ops.nvfp4_marlin import (
                quantize_nvfp4_with_global_scale,
                _build_marlin_scales_caches,
            )
            if not self.no_bf16_master:
                packed, scales, global_scale = quantize_nvfp4_with_global_scale(
                    self.weight.data, block_size=self.block_size,
                )
                self.packed_weight.copy_(packed)
                self.scales.copy_(scales)
                self.global_scale.copy_(global_scale.view(1))
            _build_marlin_scales_caches(
                self, self.scales, self.global_scale,
                size_k=self.in_features,
                size_n=self.out_features_per_partition,
                block_size=self.block_size,
            )
            return
        if not self.no_bf16_master:
            from src.models.ops.nvfp4 import quantize_nvfp4
            packed, scales = quantize_nvfp4(
                self.weight.data, block_size=self.block_size,
            )
            self.packed_weight.copy_(packed)
            self.scales.copy_(scales)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.bf16_only:
            return F.linear(x, self.weight, self.bias)
        if self.use_marlin:
            if self.no_bf16_master:
                from src.models.ops.nvfp4_marlin import (
                    _NVFP4MarlinNoLeafMatmul,
                )
                return _NVFP4MarlinNoLeafMatmul.apply(
                    x, self.packed_weight,
                    self._scales_for_kernel, self._global_scale_adj,
                    self.bias,
                    self.in_features, self.block_size,
                    self.scales, self.global_scale,
                    weakref.ref(self),
                )
            from src.models.ops.nvfp4_marlin import marlin_nvfp4_matmul
            return marlin_nvfp4_matmul(
                x, self.weight, self.packed_weight,
                self._scales_for_kernel, self._global_scale_adj,
                self.bias,
                self.scales, self.global_scale,
            )
        # Non-Marlin mode (2) — column-parallel maps to base
        # _NVFP4Matmul (no all-reduce, bias in F.linear).
        from src.models.ops.nvfp4_linear import _NVFP4Matmul
        return _NVFP4Matmul.apply(
            x, self.weight, self.packed_weight, self.scales, self.bias,
            self.in_features, self.block_size,
        )


class NVFP4RowParallelLinear(nn.Module):
    """Row-parallel NVFP4 linear with all-reduce on output.

    Weight shape: ``[out_features, in_features_per_partition]`` (BF16 master
    in mode 2; FP4 only in mode 3).
    """

    def __init__(
        self, in_features: int, out_features: int, bias: bool = False,
        device=None, dtype=None, block_size: int = 16,
        bf16_only: bool = False,
        use_marlin: bool = False,
        no_bf16_master: bool = False,
    ) -> None:
        super().__init__()
        assert block_size == 16, f"NVFP4 block_size must be 16, got {block_size}"
        world = _TP_WORLD_SIZE
        assert in_features % world == 0, (
            f"in_features={in_features} not divisible by tp_world={world}"
        )
        if no_bf16_master and not use_marlin:
            raise ValueError(
                "no_bf16_master=True currently requires use_marlin=True"
            )
        self.in_features = in_features
        self.out_features = out_features
        self.in_features_per_partition = in_features // world
        self.block_size = block_size
        self.bf16_only = bf16_only
        self.use_marlin = use_marlin
        self.no_bf16_master = no_bf16_master

        if not no_bf16_master and not bf16_only:
            self.weight = nn.Parameter(
                torch.empty(
                    out_features, self.in_features_per_partition,
                    device=device, dtype=dtype or torch.bfloat16,
                )
            )

        K_padded = self.in_features_per_partition + (-self.in_features_per_partition % block_size)
        if not bf16_only:
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
            if use_marlin:
                self.register_buffer(
                    "global_scale",
                    torch.ones(1, dtype=torch.float32, device=device),
                )
                self.register_buffer(
                    "_scales_for_kernel",
                    torch.empty(0, dtype=torch.float8_e4m3fn, device=device),
                    persistent=False,
                )
                self.register_buffer(
                    "_global_scale_adj",
                    torch.ones(1, dtype=torch.float32, device=device),
                    persistent=False,
                )
        if bias:
            self.bias = nn.Parameter(
                torch.empty(out_features, device=device, dtype=dtype or torch.bfloat16)
            )
        else:
            self.register_parameter("bias", None)
        self._latest_grad_w: Optional[torch.Tensor] = None
        self.register_load_state_dict_post_hook(_nvfp4_post_load_repack_hook)
        self._init_weights()

    # Mode (3) chunk API — same surface as column-parallel.
    def material_chunk(self, start: int, end: int) -> torch.Tensor:
        from src.models.ops.nvfp4_marlin import dequantize_marlin_nvfp4
        packed_chunk = self.packed_weight[start:end].contiguous()
        scales_chunk = self.scales[start:end].contiguous()
        return dequantize_marlin_nvfp4(
            packed_chunk, scales_chunk, self.global_scale,
            self.in_features_per_partition, self.block_size,
        )

    def commit_chunk(self, start: int, end: int, bf16_chunk: torch.Tensor) -> None:
        from src.models.ops.nvfp4_marlin import quantize_nvfp4_with_global_scale
        packed, scales, _ = quantize_nvfp4_with_global_scale(
            bf16_chunk, block_size=self.block_size,
        )
        self.packed_weight[start:end].copy_(packed)
        self.scales[start:end].copy_(scales)

    def material_bf16_view(self) -> torch.Tensor:
        from src.models.ops.nvfp4_marlin import dequantize_marlin_nvfp4
        return dequantize_marlin_nvfp4(
            self.packed_weight, self.scales, self.global_scale,
            self.in_features_per_partition, self.block_size,
        )

    def commit_bf16_view(self, bf16: torch.Tensor) -> None:
        from src.models.ops.nvfp4_marlin import (
            quantize_nvfp4_with_global_scale, _build_marlin_scales_caches,
        )
        packed, scales, gs = quantize_nvfp4_with_global_scale(
            bf16, block_size=self.block_size,
        )
        self.packed_weight.copy_(packed)
        self.scales.copy_(scales)
        self.global_scale.copy_(gs.view(1))
        _build_marlin_scales_caches(
            self, self.scales, self.global_scale,
            size_k=self.in_features_per_partition, size_n=self.out_features,
            block_size=self.block_size,
        )

    def apply_chunk_update(
        self, start: int, end: int, grad_chunk: torch.Tensor, factor: float,
    ) -> None:
        bf16 = self.material_chunk(start, end)
        bf16.sub_(grad_chunk.to(bf16.dtype), alpha=factor)
        self.commit_chunk(start, end, bf16)
        from src.models.ops.nvfp4_marlin import _build_marlin_scales_caches
        _build_marlin_scales_caches(
            self, self.scales, self.global_scale,
            size_k=self.in_features_per_partition, size_n=self.out_features,
            block_size=self.block_size,
        )

    def decay_and_apply_chunk(
        self,
        start: int, end: int,
        update_chunk: torch.Tensor, lr: float,
        wd_factor: float = 1.0,
    ) -> None:
        """Muon-style apply. See ``NVFP4Linear.decay_and_apply_chunk``
        for the math. ``start`` / ``end`` are rows in the global
        ``out_features`` dim (since the row-parallel partition is
        sharded along the **input** axis, not the output).
        """
        bf16 = self.material_chunk(start, end)
        if wd_factor != 1.0:
            bf16.mul_(wd_factor)
        bf16.sub_(update_chunk.to(bf16.dtype), alpha=lr)
        self.commit_chunk(start, end, bf16)
        from src.models.ops.nvfp4_marlin import _build_marlin_scales_caches
        _build_marlin_scales_caches(
            self, self.scales, self.global_scale,
            size_k=self.in_features_per_partition, size_n=self.out_features,
            block_size=self.block_size,
        )

    def chunk_ranges(self) -> list[tuple[int, int]]:
        rows_per_chunk = max(1, (4 * 1024 * 1024) // max(1, self.in_features_per_partition))
        ranges = []
        start = 0
        while start < self.out_features:
            end = min(start + rows_per_chunk, self.out_features)
            ranges.append((start, end))
            start = end
        return ranges

    def _stash_grad_w(self, grad_w: torch.Tensor) -> None:
        self._latest_grad_w = grad_w.detach()

    def _consume_grad_w(self) -> Optional[torch.Tensor]:
        g = self._latest_grad_w
        self._latest_grad_w = None
        return g

    def _init_weights(self) -> None:
        if self.bf16_only:
            if self.bias is not None:
                nn.init.zeros_(self.bias)
            return
        if not self.no_bf16_master:
            nn.init.xavier_uniform_(self.weight)
            if self.bias is not None:
                nn.init.zeros_(self.bias)
            self.repack_weights()
            return
        if self.bias is not None:
            nn.init.zeros_(self.bias)
        from src.models.ops.nvfp4_marlin import quantize_nvfp4_with_global_scale
        N, K = self.out_features, self.in_features_per_partition
        for start in range(0, N, 64):
            end = min(start + 64, N)
            chunk = torch.empty(
                end - start, K, device=self.packed_weight.device,
                dtype=torch.bfloat16,
            )
            nn.init.xavier_uniform_(chunk)
            packed, scales, _ = quantize_nvfp4_with_global_scale(
                chunk, block_size=self.block_size,
            )
            self.packed_weight[start:end].copy_(packed)
            self.scales[start:end].copy_(scales)
        if self.use_marlin:
            # See NVFP4ColumnParallelLinear._init_weights — same
            # cache-population requirement.
            from src.models.ops.nvfp4_marlin import _build_marlin_scales_caches
            _build_marlin_scales_caches(
                self, self.scales, self.global_scale,
                size_k=self.in_features_per_partition,
                size_n=self.out_features,
                block_size=self.block_size,
            )

    @torch.no_grad()
    def repack_weights(self) -> None:
        if self.bf16_only:
            return
        if self.use_marlin:
            from src.models.ops.nvfp4_marlin import (
                quantize_nvfp4_with_global_scale,
                _build_marlin_scales_caches,
            )
            if not self.no_bf16_master:
                packed, scales, global_scale = quantize_nvfp4_with_global_scale(
                    self.weight.data, block_size=self.block_size,
                )
                self.packed_weight.copy_(packed)
                self.scales.copy_(scales)
                self.global_scale.copy_(global_scale.view(1))
            _build_marlin_scales_caches(
                self, self.scales, self.global_scale,
                size_k=self.in_features_per_partition,
                size_n=self.out_features,
                block_size=self.block_size,
            )
            return
        if not self.no_bf16_master:
            from src.models.ops.nvfp4 import quantize_nvfp4
            packed, scales = quantize_nvfp4(
                self.weight.data, block_size=self.block_size,
            )
            self.packed_weight.copy_(packed)
            self.scales.copy_(scales)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.bf16_only:
            out = F.linear(x, self.weight, None)
            if self.bias is not None and _TP_RANK == 0:
                out = out + self.bias
            return tp_all_reduce(out)
        if self.use_marlin:
            if self.no_bf16_master:
                return _MarlinNvFp4RowMatmulNoLeafImpl.apply(
                    x, self.packed_weight,
                    self._scales_for_kernel, self._global_scale_adj,
                    self.bias,
                    self.in_features_per_partition, self.block_size,
                    self.scales, self.global_scale,
                    weakref.ref(self),
                )
            return _MarlinNvFp4RowMatmul.apply(
                x, self.weight, self.packed_weight,
                self._scales_for_kernel, self._global_scale_adj,
                self.bias,
                self.in_features_per_partition, self.block_size,
                self.scales, self.global_scale,
            )
        if self.no_bf16_master:
            return _NVFP4RowMatmulNoLeaf.apply(
                x, self.packed_weight, self.scales, self.bias,
                self.in_features_per_partition, self.block_size,
                weakref.ref(self),
            )
        return _NVFP4RowMatmul.apply(
            x, self.weight, self.packed_weight, self.scales, self.bias,
            self.in_features_per_partition, self.block_size,
        )
