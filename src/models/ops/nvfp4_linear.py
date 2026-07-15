"""W4A16 NVFP4 Linear (FFN weight stored in NVFP4 packed format, BF16 activation).

This module exposes a drop-in replacement for ``F.linear`` whose weight
**storage** is NVFP4 packed (uint8 + FP8-e4m3fn microblock scales). The
matmul itself runs in BF16 (dequantize-on-fwd); see the FP4-matmul
note below for why.

Storage modes
-------------
There are three storage modes, selected at construction time:

  1. ``bf16_only=True`` — plain BF16 GEMM, identical numerics to a
     plain ``nn.Linear``. The packed buffers are not allocated. Used by
     :class:`src.models.opt5_bf16_only_ffn` to opt out of NVFP4.

  2. ``bf16_only=False, no_bf16_master=False`` (**legacy**):
     ``self.weight`` is an ``nn.Parameter`` (BF16, the optimizer's view);
     ``packed_weight`` / ``scales`` are derived buffers repacked after
     each optimizer step via :meth:`repack_weights`. The bwd path does
     not actually read ``self.weight.data`` (it only uses it as a graph
     anchor — STE for the quantize noise), but it still exists. This
     is the historical default.

  3. ``bf16_only=False, no_bf16_master=True`` (**new, MoE-friendly**):
     there is no BF16 master anywhere. ``packed_weight``,
     ``scales``, and ``global_scale`` are the only persistent state
     (FP4 packed on GPU; the optimizer streams BF16 views through
     material/commit during the step). The autograd Function returns
     ``grad_x`` only — ``grad_w`` is stashed on the module via a
     side-channel so the custom optimizer step can consume it.

Mode (3) exists because the BF16 master scales linearly with the
number of experts in a future MoE FFN. Eliminating it saves
~10-12 MiB / FFN module / TP-rank; at (num_experts=8, num_layers=32,
TP=1) on the 5060 Ti dev box that is ~2.5 GiB headroom before any
FFN shape changes — i.e. the storage cliff is moved from "switch on
NVFP4" to "add the first expert".

FP4-matmul note
---------------
PyTorch's ``torch._scaled_mm`` NVFP4 path (``scale_block_size=16``)
requires BOTH A and B to be FP4-packed (W4A4 only). A pure BF16 x FP4
mixed-precision GEMM is not exposed in 2.9.1. We therefore dequantize
to BF16 before the matmul, keeping the autograd contract identical
to a plain Linear. Swapping in a real FP4 GEMM is a follow-up that
touches only ``_NVFP4Matmul.forward``.

Autograd design (STE for the quantize round-trip)
-------------------------------------------------
The forward dequantizes ``packed_weight`` / ``scales`` to BF16 and
calls ``F.linear``. The dequantized BF16 is conceptually a quantized
view of the underlying weight; the gradient should therefore flow
into the FP4 buffers. In mode (2) we use ``self.weight`` as a graph
anchor (forward ignores it, backward returns its grad — STE pattern).
In mode (3) there is no leaf for autograd to attach a gradient to,
so the new :class:`_NVFP4NoLeafMatmul` instead stashes ``grad_w`` on
a back-reference to the owning module via a side-channel. The custom
optimizer step in :mod:`src.training.param_offload` consumes the
stashed grad and applies it at the FP4 packed level (streaming
material/commit at chunk granularity).
"""
from __future__ import annotations

import weakref
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.ops.nvfp4 import dequantize_nvfp4


# Chunk granularity for the optimizer's stream-dequant path. Each
# forward pass materializes a single BF16 [N, K] view (large), but the
# optimizer step iterates in chunks so the peak BF16 surface stays flat
# (~10 MiB regardless of module size). 4M elements = 8 MiB at BF16,
# matching the AdamW chunk budget in :mod:`src.training.param_offload`.
_CHUNK_NUMEL = 4 * 1024 * 1024  # 4M BF16 elements per chunk ≈ 8 MiB


def _nvfp4_post_load_repack_hook(
    module: torch.nn.Module, incompatible_keys: object,
) -> None:
    """Post-``load_state_dict`` hook: refresh derived Marlin caches.

    Installed on every ``NVFP4*Linear`` instance via
    :func:`torch.nn.Module.register_load_state_dict_post_hook`. After a
    checkpoint load, the source ``packed_weight`` / ``scales`` /
    ``global_scale`` buffers reflect the loaded data, but the
    per-module ``_scales_for_kernel`` and ``_global_scale_adj`` are
    stale (they're not in the state_dict — they're derived state). The
    hook invokes ``repack_weights()`` to recompute them against the
    freshly-loaded source buffers, so the very next forward uses the
    correct weights.

    The hook is a no-op for ``bf16_only=True`` modules
    (``repack_weights`` short-circuits) and for modules that don't have
    the derived caches (defensive guard in case a future module type
    inherits this hook).

    For mode (3) (``no_bf16_master=True``) this is still needed — the
    hook simply rebuilds the Marlin-side caches against the freshly
    loaded FP4 buffers.
    """
    repack = getattr(module, "repack_weights", None)
    if repack is None or not hasattr(module, "_scales_for_kernel"):
        return
    # In mode (3) ``repack_weights`` no longer touches a BF16 master; it
    # only rebuilds the small Marlin caches. We don't need to re-quantize
    # the source FP4 buffers because they ARE the source of truth in
    # mode (3).
    repack()


class _NVFP4Matmul(torch.autograd.Function):
    """BF16 act @ NVFP4-packed weight (dequantized to BF16 in forward).

    **Mode (2) only** (legacy BF16 master present). Inputs (only ``x``
    and ``w_master`` need gradients):

        x            : [..., K] in BF16 (activation, requires_grad)
        w_master     : [N, K] in BF16 (the BF16 master weight, requires_grad).
                       Used as a graph anchor; the forward does NOT consume it.
                       The backward returns the standard matmul grad on it so
                       the optimizer can update it. STE for the quantize noise.
        packed_w     : [N, K_padded // 2] uint8 (FP4-packed, no grad)
        scales_w     : [N, K_padded // 16] fp8_e4m3fn (no grad)
        bias         : [N] or None
        K_orig       : int (the un-padded K)
        block_size   : int (NVFP4 microblock size, fixed at 16)
    """

    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        x: torch.Tensor,
        w_master: torch.Tensor,
        packed_w: torch.Tensor,
        scales_w: torch.Tensor,
        bias: torch.Tensor | None,
        K_orig: int,
        block_size: int,
    ) -> torch.Tensor:
        # Dequantize FP4 -> BF16 for the matmul (the actual work).
        w_bf16 = dequantize_nvfp4(
            packed_w, scales_w, K_orig,
            block_size=block_size,
            out_dtype=x.dtype,
        )
        # Save (x, w_bf16) for backward. w_bf16 is the dequantized view;
        # its grad in backward is the conceptual grad on w_master (STE).
        ctx.save_for_backward(x, w_bf16)
        ctx.has_bias = bias is not None
        return F.linear(x, w_bf16, bias)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):  # type: ignore[override]
        x, w_bf16 = ctx.saved_tensors
        # Standard BF16 matmul backward (we can't use F.linear's backward
        # because w_bf16 isn't a leaf that the autograd graph knows about
        # as a parameter — we dequantized it from packed/scales).
        K = w_bf16.shape[1]
        N = w_bf16.shape[0]
        grad_out_2d = grad_out.reshape(-1, N)
        x_2d = x.reshape(-1, K)
        # grad_x = grad_out @ W  (broadcasts over leading dims).
        grad_x = grad_out @ w_bf16
        # grad_w = grad_out.T @ x  -> shape [N, K]. This is what we want
        # to flow into ``self.weight`` (the BF16 master).
        grad_w = (grad_out_2d.t().to(x_2d.dtype)) @ x_2d
        grad_bias = grad_out_2d.sum(dim=0) if ctx.has_bias else None
        return grad_x, grad_w, None, None, grad_bias, None, None


# ---------------------------------------------------------------------------
# Mode (3) — no BF16 master. Side-channel autograd for grad_w delivery.
# Defined here (not in nvfp4_marlin) so the dequant+cuBLAS path is local
# to nvfp4_linear; the Marlin variant lives in nvfp4_marlin alongside the
# kernel call it wraps.
# ---------------------------------------------------------------------------
class _NVFP4NoLeafMatmul(torch.autograd.Function):
    """BF16 act @ NVFP4-packed weight — no BF16 master in the autograd graph.

    Forward: dequantize FP4 -> BF16, ``F.linear``. Bias added inside
    ``F.linear`` (column-parallel / no-allreduce case).

    Backward: ``grad_x = grad_out @ W_dequant`` (standard matmul
    backward). ``grad_w`` is computed but NOT returned — instead it is
    stashed on the owning module's ``_latest_grad_w`` attribute via a
    weakref captured at forward time. The custom optimizer step in
    :mod:`src.training.param_offload` consumes that stashed tensor
    once per step (clearing it after use).

    The weakref is required because the Function is held by the
    autograd graph (which can outlive the module if the module is
    freed without the graph being collected first); the weakref
    silently no-ops if the module has been GC'd.
    """

    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        x: torch.Tensor,
        packed_w: torch.Tensor,
        scales_w: torch.Tensor,
        bias: torch.Tensor | None,
        K_orig: int,
        block_size: int,
        module_ref,  # weakref to NVFP4Linear (mode 3)
    ) -> torch.Tensor:
        # Dequantize FP4 -> BF16 for the matmul.
        w_bf16 = dequantize_nvfp4(
            packed_w, scales_w, K_orig,
            block_size=block_size, out_dtype=x.dtype,
        )
        ctx.save_for_backward(x, w_bf16)
        ctx.has_bias = bias is not None
        ctx.module_ref = module_ref
        ctx.K_orig = K_orig
        # Stash shape so backward can reshape 2-D output back.
        ctx.input_shape = x.shape
        return F.linear(x, w_bf16, bias)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):  # type: ignore[override]
        x, w_bf16 = ctx.saved_tensors
        K = ctx.K_orig
        N = w_bf16.shape[0]
        grad_out_2d = grad_out.reshape(-1, N)
        x_2d = x.reshape(-1, K)
        # grad_x = grad_out @ W  -> shape matches x up to the last dim.
        grad_x = grad_out @ w_bf16
        # grad_w = grad_out.T @ x  -> shape [N, K]. Stash to module.
        grad_w = (grad_out_2d.t().to(x_2d.dtype)) @ x_2d
        module = ctx.module_ref()
        if module is not None:
            # Replace any prior grad_w (only one backward runs per
            # forward; the optimizer is single-consumer).
            module._stash_grad_w(grad_w)
        grad_bias = grad_out_2d.sum(dim=0) if ctx.has_bias else None
        return grad_x, None, None, grad_bias, None, None, None


class NVFP4Linear(nn.Module):
    """``nn.Linear``-equivalent module that stores its weight in NVFP4.

    See module docstring for the autograd / state design.

    State dict keys (mode 2 — BF16 master present):
        weight                 : the BF16 master weight (shape [N, K])
        packed_weight          : uint8 buffer [N, K // 2]   (FP4)
        scales                 : fp8_e4m3fn buffer [N, K // 16]
        bias                   : present iff ``bias=True``

    State dict keys (mode 3 — no BF16 master):
        packed_weight          : uint8 buffer [N, K // 2]   (FP4)
        scales                 : fp8_e4m3fn buffer [N, K // 16]
        global_scale           : float32 [1]   (Marlin scheme only)
        bias                   : present iff ``bias=True``
        (no ``weight`` key)

    Mode 2 is the legacy default for back-compat with existing
    checkpoints. Mode 3 is the new storage model — pass
    ``no_bf16_master=True`` at construction.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        device=None,
        dtype=None,
        block_size: int = 16,
        bf16_only: bool = False,
        use_marlin: bool = False,
        no_bf16_master: bool = False,
    ) -> None:
        super().__init__()
        assert block_size == 16, (
            f"NVFP4 spec requires block_size=16, got {block_size}"
        )
        # Mode (3) requires Marlin: only the Marlin path defines the
        # no-leaf autograd Function today. The dequant+cuBLAS variant
        # lives below for completeness, but at the call site we gate
        # it on use_marlin.
        if no_bf16_master and not use_marlin:
            raise ValueError(
                "no_bf16_master=True currently requires use_marlin=True "
                "(the dequant+cuBLAS path is mode-2-only). The mode-3 "
                "side-channel autograd Function is implemented in "
                "_NVFP4MarlinNoLeafMatmul; a dequant+cuBLAS no-leaf "
                "Function exists (_NVFP4NoLeafMatmul) and is wired "
                "below, but verify the SwiGLU pipeline before mixing."
            )
        self.in_features = in_features
        self.out_features = out_features
        self.block_size = block_size
        self.bf16_only = bf16_only
        self.use_marlin = use_marlin
        self.no_bf16_master = no_bf16_master

        if not bf16_only:
            K_padded = in_features + (-in_features % block_size)
            self.register_buffer(
                "packed_weight",
                torch.zeros(out_features, K_padded // 2, device=device, dtype=torch.uint8),
            )
            self.register_buffer(
                "scales",
                torch.zeros(out_features, K_padded // block_size, device=device, dtype=torch.float8_e4m3fn),
            )
            if use_marlin:
                self.register_buffer(
                    "global_scale",
                    torch.ones(1, dtype=torch.float32, device=device),
                )
                # Marlin-side caches (rebuilt by _build_marlin_scales_caches
                # in repack_weights).
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
        # Mode (2) registers the BF16 master Parameter (legacy default).
        # Mode (3) does not — the optimizer updates the FP4 buffers
        # directly via the chunked material/commit/apply API.
        if not no_bf16_master and not bf16_only:
            self.weight = nn.Parameter(
                torch.empty(out_features, in_features, device=device, dtype=dtype or torch.bfloat16)
            )
        elif no_bf16_master and not bf16_only:
            # The module does not own a leaf BF16 Parameter. We register
            # an attribute so optimizer / autograd introspection sees a
            # consistent surface (named_parameters returns just bias).
            pass

        if bias:
            self.bias = nn.Parameter(
                torch.empty(out_features, device=device, dtype=dtype or torch.bfloat16)
            )
        else:
            self.register_parameter("bias", None)

        # Side-channel state for mode (3): the autograd Function stashes
        # grad_w on this attribute after each backward; the custom
        # optimizer step clears it on consumption.
        self._latest_grad_w: Optional[torch.Tensor] = None
        # Optional callback installed by the optimizer's
        # :meth:`register_nvfp4_module`. When set, ``_stash_grad_w``
        # routes through the callback (which does D2H + worker
        # enqueue) instead of stashing on ``_latest_grad_w``. See
        # the docstring on :meth:`_stash_grad_w` for the why.
        self._nvfp4_offload_cb: Optional[Callable[[torch.Tensor], None]] = None

        # Production-safe state-dict hook.
        self.register_load_state_dict_post_hook(
            _nvfp4_post_load_repack_hook,
        )

        self._init_weights()

    def _init_weights(self) -> None:
        if self.bf16_only:
            if self.bias is not None:
                nn.init.zeros_(self.bias)
            return
        # Mode (2): initialize the BF16 master then repack.
        if not self.no_bf16_master:
            nn.init.xavier_uniform_(self.weight)
            if self.bias is not None:
                nn.init.zeros_(self.bias)
            self.repack_weights()
            return
        # Mode (3): no BF16 master to initialize. Initialize the FP4
        # buffers to a deterministic small-valued range so the first
        # forward is well-defined. We materialize a temporary small BF16
        # tensor, quantize it, and copy in.
        # Use Xavier-uniform scaled to a small FP4-friendly range.
        if self.bias is not None:
            nn.init.zeros_(self.bias)
        self._init_fp4_from_seed()

    @torch.no_grad()
    def _init_fp4_from_seed(self) -> None:
        """Initialize FP4 buffers from a deterministic BF16 seed.

        Avoids allocating a full [N, K] BF16 master just for init; uses
        a chunked Xavier draw over the row axis and quantizes chunk-by-
        chunk into the packed buffers. The seed comes from the module's
        own RNG (inherited from ``nn.Module._init_weights``'s caller).
        """
        from src.models.ops.nvfp4_marlin import (
            quantize_nvfp4_with_global_scale,
        )
        N, K = self.out_features, self.in_features
        # We use a single-pass material + commit so the on-device peak
        # stays bounded — but init is rare, so the simpler in-place
        # path is fine. Allocate one row at a time to keep peak low.
        for start in range(0, N, 64):
            end = min(start + 64, N)
            chunk = torch.empty(
                end - start, K, device=self.packed_weight.device,
                dtype=torch.bfloat16,
            )
            nn.init.xavier_uniform_(chunk)
            packed, scales, gs = quantize_nvfp4_with_global_scale(
                chunk, block_size=self.block_size,
            )
            self.packed_weight[start:end].copy_(packed)
            self.scales[start:end].copy_(scales)
        if self.use_marlin:
            # Compute global_scale as the 99th-percentile of all per-block
            # block-scales (the natural scale of the column-wise max);
            # quantize_nvfp4_with_global_scale already produces this but
            # it's a per-chunk value. For init we approximate with 1.0.
            self.global_scale.copy_(torch.ones_like(self.global_scale))
            # Refresh the Marlin-side caches. The legacy constructor
            # calls repack_weights() which does this; mode-3 skips the
            # re-quantization (no BF16 master to quantize from) but
            # still needs the cache buffers populated, otherwise the
            # kernel reads from an empty buffer and crashes with OOB.
            from src.models.ops.nvfp4_marlin import _build_marlin_scales_caches
            _build_marlin_scales_caches(
                self, self.scales, self.global_scale,
                size_k=self.in_features, size_n=self.out_features,
                block_size=self.block_size,
            )

    # ------------------------------------------------------------------
    # Mode (3) chunk API — used by the custom optimizer in param_offload.
    # ------------------------------------------------------------------
    def material_chunk(
        self, start: int, end: int,
    ) -> torch.Tensor:
        """Materialize a BF16 [end-start, K] view of the FP4 buffers.

        The returned tensor is a fresh BF16 allocation — the caller
        owns it and must either discard it (e.g. apply grad inplace
        then call :meth:`commit_chunk`) or treat it as a read-only
        probe. The peak VRAM impact is the chunk size only, not the
        full module.

        No-grad. Run inside ``torch.no_grad()`` at the call site.
        """
        from src.models.ops.nvfp4_marlin import dequantize_marlin_nvfp4
        # The simpler dequantize_nvfp4 doesn't multiply by global_scale,
        # but the repacked FP4 buffers DO have an implicit global_scale
        # baked in (Marlin scheme). Use the marlin dequant for accuracy.
        packed_chunk = self.packed_weight[start:end].contiguous()
        scales_chunk = self.scales[start:end].contiguous()
        return dequantize_marlin_nvfp4(
            packed_chunk, scales_chunk, self.global_scale,
            self.in_features, self.block_size,
        )

    def commit_chunk(
        self, start: int, end: int, bf16_chunk: torch.Tensor,
    ) -> None:
        """Quantize ``bf16_chunk`` back into the FP4 buffers (rows
        ``[start, end)``). Pairs with :meth:`material_chunk`.

        No-grad. Run inside ``torch.no_grad()`` at the call site.
        """
        from src.models.ops.nvfp4_marlin import (
            quantize_nvfp4_with_global_scale,
        )
        packed, scales, gs = quantize_nvfp4_with_global_scale(
            bf16_chunk, block_size=self.block_size,
        )
        self.packed_weight[start:end].copy_(packed)
        self.scales[start:end].copy_(scales)
        # Global scale is recomputed from the entire matrix, not per
        # chunk; the optimizer pipeline is responsible for refreshing
        # it once per step (see param_offload's _finalize_fp4_step).
        # As a local approximation we copy the chunk-local global scale
        # in and let the post-step pass overwrite if needed.
        # NB: the current param_offload finalize path recomputes from
        # a fresh quantize over the full weight; this line is a
        # placeholder for in-progress updates and is overwritten by
        # finalize.

    def material_bf16_view(self) -> torch.Tensor:
        """Materialize a FULL BF16 [N, K] view of the FP4 buffers.

        Convenience for callers that genuinely need the full weight
        (checkpoint save, debugging, tests). Peak VRAM cost: the full
        module's BF16 footprint. Prefer :meth:`material_chunk` in
        the optimizer hot path.

        No-grad.
        """
        from src.models.ops.nvfp4_marlin import dequantize_marlin_nvfp4
        return dequantize_marlin_nvfp4(
            self.packed_weight, self.scales, self.global_scale,
            self.in_features, self.block_size,
        )

    def commit_bf16_view(self, bf16: torch.Tensor) -> None:
        """Re-quantize a FULL BF16 [N, K] tensor into the FP4 buffers.

        Companion to :meth:`material_bf16_view`. Recomputes the
        global_scale from the full tensor (the per-chunk
        :meth:`commit_chunk` cannot — it sees only a slice).

        No-grad.
        """
        from src.models.ops.nvfp4_marlin import (
            quantize_nvfp4_with_global_scale,
        )
        packed, scales, gs = quantize_nvfp4_with_global_scale(
            bf16, block_size=self.block_size,
        )
        self.packed_weight.copy_(packed)
        self.scales.copy_(scales)
        self.global_scale.copy_(gs.view(1))
        # Refresh the Marlin-side caches against the new global_scale.
        from src.models.ops.nvfp4_marlin import _build_marlin_scales_caches
        _build_marlin_scales_caches(
            self, self.scales, self.global_scale,
            size_k=self.in_features, size_n=self.out_features,
            block_size=self.block_size,
        )

    def apply_chunk_update(
        self, start: int, end: int, grad_chunk: torch.Tensor, factor: float,
        rebuild_scales_cache: bool = True,
    ) -> None:
        """Apply ``W[start:end] -= factor * grad_chunk`` at the FP4 level.

        No full BF16 master is materialized. The flow:

          1. material the chunk BF16 view
          2. subtract ``factor * grad_chunk`` in place
          3. commit the updated chunk back to FP4
          4. (optionally) refresh Marlin scales cache

        ``rebuild_scales_cache`` (default True) controls the final
        scales-cache rebuild. The optimizer's per-chunk loop typically
        passes ``False`` for all but the last chunk, so the cache is
        rebuilt exactly once per module per step (vs once per chunk
        per step). The cached ``scales_for_kernel`` is consumed by
        the *next* step's fwd; the per-step
        ``repack_nvfp4_weights`` call at the end of the training
        loop also rebuilds it, so the chunk-internal rebuilds are
        pure dead compute that the deferred-change removes.

        Caller is the optimizer, which holds ``grad_chunk`` (a slice
        of the stashed ``_latest_grad_w``). No-grad.
        """
        bf16 = self.material_chunk(start, end)
        bf16.sub_(grad_chunk.to(bf16.dtype), alpha=factor)
        self.commit_chunk(start, end, bf16)
        if rebuild_scales_cache:
            from src.models.ops.nvfp4_marlin import _build_marlin_scales_caches
            _build_marlin_scales_caches(
                self, self.scales, self.global_scale,
                size_k=self.in_features, size_n=self.out_features,
                block_size=self.block_size,
            )

    def decay_and_apply_chunk(
        self,
        start: int, end: int,
        update_chunk: torch.Tensor, lr: float,
        wd_factor: float = 1.0,
        rebuild_scales_cache: bool = True,
    ) -> None:
        """Apply Muon-style ``W = W * wd_factor - lr * update`` at the
        FP4 level (no full BF16 master materialize).

        Flow:
          1. materialize the BF16 view of rows ``[start, end)``
          2. ``bf16 *= wd_factor``        (decoupled WD)
          3. ``bf16 -= lr * update_chunk`` (the NS output)
          4. commit back to FP4
          5. (optionally) refresh Marlin scales cache

        See :meth:`apply_chunk_update` for the
        ``rebuild_scales_cache`` contract.

        ``update_chunk`` is the per-row Newton-Schulz output for the
        chunk. We cast to BF16 here (matches ``material_chunk``'s
        output dtype); the NS run itself stays in FP16 on the GPU.

        Companion of :meth:`apply_chunk_update` (AdamW-style). Both
        keep the peak GPU surface bounded to one chunk (~8 MiB BF16
        for the canonical FFN module size).
        """
        bf16 = self.material_chunk(start, end)
        if wd_factor != 1.0:
            bf16.mul_(wd_factor)
        bf16.sub_(update_chunk.to(bf16.dtype), alpha=lr)
        self.commit_chunk(start, end, bf16)
        if rebuild_scales_cache:
            from src.models.ops.nvfp4_marlin import _build_marlin_scales_caches
            _build_marlin_scales_caches(
                self, self.scales, self.global_scale,
                size_k=self.in_features, size_n=self.out_features,
                block_size=self.block_size,
            )

    def chunk_ranges(self) -> list[tuple[int, int]]:
        """Yield ``(start, end)`` row ranges covering the full weight.

        Chunk size is the same AdamW budget as the rest of the
        training loop (:data:`_CHUNK_NUMEL`, 4M BF16 elements / 8 MiB).
        Returns a list (not a generator) so callers can len() it for
        progress reporting.
        """
        rows_per_chunk = max(1, _CHUNK_NUMEL // max(1, self.in_features))
        ranges = []
        start = 0
        while start < self.out_features:
            end = min(start + rows_per_chunk, self.out_features)
            ranges.append((start, end))
            start = end
        return ranges

    # ------------------------------------------------------------------
    # Side-channel grad_w delivery (mode 3 only).
    # ------------------------------------------------------------------
    def _stash_grad_w(self, grad_w: torch.Tensor) -> None:
        # Production path (CPUAdamW / CPUMuon with mode-3 modules
        # registered via :meth:`register_nvfp4_module`): the
        # optimizer installs ``_nvfp4_offload_cb`` so D2H + worker
        # enqueue happens HERE, inside the autograd backward call.
        # This lets the per-NVFP4 D2H overlap with subsequent
        # layers' bwd kernels instead of waiting for the post-
        # backward :func:`accumulate_grads_to_cpu` sweep
        # (which costs ~22ms/chunk of pure-Python per-param
        # work on the dev shape, ~hundreds of ms at prod shape).
        #
        # Legacy path (tests, unit-test paths without optimizer
        # registration): fall back to stashing on
        # ``self._latest_grad_w`` and let accumulate_grads_to_cpu
        # pick it up post-backward.
        cb = getattr(self, "_nvfp4_offload_cb", None)
        if cb is not None:
            cb(grad_w)
            return
        self._latest_grad_w = grad_w.detach()

    def _consume_grad_w(self) -> Optional[torch.Tensor]:
        g = self._latest_grad_w
        self._latest_grad_w = None
        return g

    # ------------------------------------------------------------------
    # Legacy mode (2) — re-quantize BF16 master -> FP4 buffers.
    # Mode (3) — rebuild Marlin caches only.
    # ------------------------------------------------------------------
    @torch.no_grad()
    def repack_weights(self) -> None:
        """Re-quantize the BF16 master weight into the NVFP4 buffer pair.

        **Mode (2)**: call AFTER each optimizer step (NOT after each
        forward). Keeps the packed buffers in sync with the (now-
        updated) BF16 master, so the next forward sees the latest
        quantized weights.

        **Mode (3)**: no-op unless Marlin caches need refreshing (e.g.
        after a checkpoint load). The FP4 buffers ARE the source of
        truth, no re-quantization needed.
        """
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
            # Recompute the small Marlin-side buffers. See
            # ``_build_marlin_scales_caches`` for the design note.
            N = self.out_features
            scales = self.scales
            global_scale = self.global_scale
            _build_marlin_scales_caches(
                self, scales, global_scale,
                size_k=self.in_features, size_n=N,
                block_size=self.block_size,
            )
            return
        # Non-Marlin mode (2) — simple quantize path.
        from src.models.ops.nvfp4 import quantize_nvfp4
        packed, scales = quantize_nvfp4(self.weight.data, block_size=self.block_size)
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
                # The Function captures a weakref to this module for
                # the grad_w side-channel.
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
        # Non-Marlin mode (2) — dequant+cuBLAS path.
        from src.models.ops.nvfp4_linear import _NVFP4Matmul as _leaf  # local
        return _leaf.apply(
            x, self.weight,
            self.packed_weight, self.scales,
            self.bias,
            self.in_features, self.block_size,
        )

    def extra_repr(self) -> str:
        if self.bf16_only:
            storage = "BF16(bf16_only)"
        elif self.use_marlin:
            leaf = "no-bf16-master" if self.no_bf16_master else "bf16-master"
            storage = f"NVFP4(E2M1+FP8-e4m3fn 1x16) [Marlin {leaf}]"
        else:
            storage = "NVFP4(E2M1+FP8-e4m3fn 1x16)"
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, block_size={self.block_size}, "
            f"weight_storage={storage}"
        )



def repack_nvfp4_weights(model: nn.Module) -> int:
    """Walk ``model`` and call ``repack_weights()`` on every NVFP4 module.

    Looks for the NVFP4 modules shipped with this project:

      - :class:`NVFP4Linear`          (single-GPU SwiGLU path)
      - :class:`NVFP4ColumnParallelLinear`  (TP column-parallel SwiGLU)
      - :class:`NVFP4RowParallelLinear`     (TP row-parallel SwiGLU)

    All three expose a ``repack_weights()`` method. For mode (2) it
    re-quantizes the BF16 master into the FP4 buffer pair; for mode (3)
    it only refreshes the Marlin caches against the (already-current)
    FP4 buffers.

    Returns the number of NVFP4 modules repacked. ``0`` is a valid
    return value when NVFP4 is not enabled (the function is a no-op).
    """
    count = 0
    # Local imports to avoid a circular dependency at module-load time
    # (this module is imported by both activation.py and tp_model.swiglu,
    # which themselves are imported here at runtime, not at top level).
    from src.models.ops.nvfp4_tp import (
        NVFP4ColumnParallelLinear,
        NVFP4RowParallelLinear,
    )
    target_types = (NVFP4Linear, NVFP4ColumnParallelLinear, NVFP4RowParallelLinear)
    for module in model.modules():
        if isinstance(module, target_types):
            module.repack_weights()
            count += 1
    return count
