"""Muon with CPU-offloaded momentum.

:class:`CPUMuon` is the other half of
:mod:`src.training.param_offload`'s optimizer pair. It owns the
Newton-Schulz orthogonalization step and the per-storage-dtype
dispatch (full-precision bf16 / fp16 / fp32; quantized int8 /
mxfp8 storage was removed on 2026-07-12 after long-training
instability — pre-removal code preserved on the
``archive/int8-mxfp8-muon`` branch). The split file layout keeps
the optimizer algorithm in one place without the offload
plumbing interleaving it.

Implementation notes are on the class docstring; see
:mod:`src.training.param_offload` for the package-level design
rationale.
"""
from __future__ import annotations

from typing import List

import torch
import torch.nn as nn

from ._state import _ParamState, _accumulator_target
from .cpu_fused import fused_zero_many
from .offload import _make_nvfp4_offload_cb


class CPUMuon:
    """Muon with CPU-side ``grad`` (per-step accumulator) and
    ``exp_avg`` (SGD momentum — preserved across steps).
    Params live on the GPU; grads are streamed to CPU by the
    post-accumulate-grad hook (see :func:`register_grad_offload_hooks`)
    or by the manual :func:`accumulate_grads_to_cpu` path.

    Memory layout per trainable param:
        CPU pinned: grad (cfg dtype, numel),
                    exp_avg (cfg dtype, numel)
        GPU:         param (FP16), .grad (transient)

    The two CPU buffers have distinct, explicitly-named roles
    (post-2026-07-15; the previous "merged-accumulator" design
    where ``mom_buf`` doubled as the accumulator and the
    momentum — with β effectively equal to 1, because the
    accumulator was reset at every step — was deleted):

    * ``s.grad``    — per-step grad accumulator. Each
      microbatch's ``.grad`` is added to ``s.grad`` in place
      (one ``.add_()`` per microbatch). Zeroed at the end of
      every :meth:`step` (or by :func:`zero_cpu_grad_accum`
      when ``found_inf`` skips the step). The end-of-step
      grad-norm clip + TP all-reduce operate on this tensor —
      see :mod:`src.training.loop.grad_norm` for the "why this
      must live on CPU" rationale.
    * ``s.exp_avg`` — SGD momentum: ``β·prev + grad`` (Keller
      Jordan's Muon reference formulation). Preserved across
      steps so the momentum smoothing has cross-step state.
      If ``nesterov=True``, the input to the NS iteration is
      ``grad + β·exp_avg``; otherwise it's just ``exp_avg``.
      Initial value is zero.

    Storage dtype is BF16 (full-precision; was originally
    driven by ``precision.muon_momentum.dtype`` — the yml-side
    ``precision:`` block that let the yml switch among
    ``bf16`` / ``fp16`` / ``fp32`` was removed 2026-07-23):

    * The Newton-Schulz output precision is unaffected by the
      storage dtype (NS orthogonalizes the dequantized / raw
      FP32 momentum on the GPU regardless); the storage
      precision only affects how much the EMA buffer is
      rounded between steps.

    * Quantized storage (``int8`` per-row BF16 scale,
      ``mxfp8`` per-block E8M0 scale) was removed on
      2026-07-12. The bf16 canonical storage is fine for
      training at our scale and avoids the per-step
      requantize cost. See ``docs/optimizer_layout.md`` for
      the post-removal layout.

    The training loop's :func:`accumulate_grads_to_cpu` adds
    the GPU ``.grad`` to ``s.grad`` (CPU, at the storage
    dtype). On :meth:`step` we compute the SGD momentum
    ``exp_avg ← β·exp_avg + grad`` on CPU, compute the
    Nesterov-corrected input on CPU, H2D + cast to FP32, NS,
    apply the update to the GPU param, then zero ``s.grad``
    for the next cycle (preserving ``exp_avg`` across steps).

    Then stream each row-chunk to the GPU, run 5 NS iterations
    in FP16 (tensor cores), and apply the update to the GPU
    param. The NS path is independent of the storage dtype.
    """

    _NS_COEFFS = (3.4445, -4.7750, 2.0315)

    def __init__(
        self,
        params: List[nn.Parameter],
        lr: float = 1e-3,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        weight_decay: float = 0.0,
        exp_avg_storage: str = "bf16",
        block_size: int = 32,
    ) -> None:
        """CPU-offloaded Muon with BF16 state throughout.

        Storage (BF16 for both buffers; the only supported
        layout since the legacy ``precision:`` yml block was
        removed 2026-07-23):

        - ``s.grad``    (per-step accumulator)
        - ``s.exp_avg`` (SGD momentum EMA)

        The ``momentum`` and ``nesterov`` hyperparameters are
        consumed by :meth:`step` (since the 2026-07-15 refactor
        that deleted the mu=1 merged-accumulator trick): the
        step computes ``exp_avg = β·exp_avg + grad`` and (when
        ``nesterov=True``) the Nesterov correction
        ``g_for_ns = grad + β·exp_avg`` before Newton-Schulz.

        The ``exp_avg_storage`` / ``block_size`` parameters
        control the optional FP8 2D-tight scale EMA layout
        (see ``project_2d_tight_scale_shipped.md``).
        """
        self.lr = lr
        self.momentum = momentum
        self.nesterov = nesterov
        self.ns_steps = ns_steps
        self.weight_decay = weight_decay

        # Storage mode for the SGD momentum EMA ``exp_avg``.
        # - ``"bf16"`` (default): full-precision BF16 buffer.
        # - ``"fp8_2d_tight"``: E4M3 + per-(row × col-block) +
        #   per-(col × row-block) FP32 scales (the
        #   ``lowbit-Muon/quantize_2d`` fine-grained scheme);
        #   dequant/re-quant per step. Brings NS output drift
        #   from 14-21% down to ~1.45% on B/D regimes at the
        #   prod shape (see
        #   ``docs/auto-memory/project_2d_tight_scale.md``).
        if exp_avg_storage not in ("bf16", "fp8_2d_tight"):
            raise ValueError(
                f"exp_avg_storage must be 'bf16' or 'fp8_2d_tight', "
                f"got {exp_avg_storage!r}"
            )
        self.exp_avg_storage = exp_avg_storage
        self.block_size = block_size
        if exp_avg_storage == "fp8_2d_tight":
            # Lazy import: this is the only site that needs the
            # quant primitives, and keeping the import at use
            # time avoids a hard import dependency from the
            # default BF16 path.
            from ..ops.fp8_2d_tight import quantize_2d, dequantize_2d
            self._quantize_2d = quantize_2d
            self._dequantize_2d = dequantize_2d

        # Storage: BF16 for both ``grad`` and ``exp_avg``. The
        # legacy ``precision.muon_momentum`` knob (which could
        # route to fp16 / fp32 / fp8) was removed on 2026-07-23
        # with the yml-side ``precision:`` block.
        storage_dtype = torch.bfloat16

        self.state: dict[int, _ParamState] = {}
        seen: set[int] = set()
        for p in params:
            if not p.requires_grad:
                continue
            if id(p) in seen:
                continue
            seen.add(id(p))
            if p.ndim < 2:
                raise ValueError(
                    f"CPUMuon got {p.ndim}D param of shape {tuple(p.shape)}; "
                    f"use CPUAdamW for 1D params."
                )
            n = p.numel()
            shape = tuple(p.shape)
            st = _ParamState(
                param=p,
                grad=torch.zeros(
                    n, dtype=storage_dtype, device="cpu",
                ).pin_memory(),
                kind="muon",
                shape=shape,
            )
            if exp_avg_storage == "bf16":
                # Legacy BF16 storage; unchanged.
                st.exp_avg = torch.zeros(
                    n, dtype=storage_dtype, device="cpu",
                ).pin_memory()
            else:
                # FP8 2D tight scale storage. See
                # :mod:`src.training.ops.fp8_2d_tight` for the
                # contract. All four buffers are pinned CPU so
                # the per-step dequant+EMA+requant stays on the
                # CPU side; the resulting H2D into NS is the
                # same one copy as the BF16 path.
                rows, cols = shape[0], shape[1]
                # Fall back to the BF16 buffer for params whose
                # shape isn't a multiple of ``block_size`` on
                # both axes (matches the
                # ``lowbit-Muon/quantize_2d`` reshape contract).
                # In production the only such params are small:
                # ``layers.X.kda.attn.b_proj.weight`` (12, 1536)
                # — the KDA b-projection has 12 rows = num_heads
                # — and ``attn_res.query`` (12, 128) — the
                # AttnRes query has 12 rows = num_heads and 128
                # cols = head_dim. Total BF16 fallback size:
                # ~40 KB across the whole model. FP8 savings on
                # these would be ~73 KB; adding padding support
                # for that is not worth the code complexity. The
                # step() path branches on ``s.exp_avg is not None``
                # so the BF16 fallback runs without further changes.
                if rows % block_size == 0 and cols % block_size == 0:
                    n_row_blocks = rows // block_size
                    n_col_blocks = cols // block_size
                    # scale_dim1: per-(row × col-block) amax / E4M3_MAX,
                    # shape (rows, cols // block_size).
                    # scale_dim2: per-(col × row-block) amax / E4M3_MAX,
                    # shape (cols, rows // block_size).
                    # This is the fine-grained layout from
                    # lowbit-Muon/src/quant.py::quantize_2d; the coarse
                    # per-(row-block × all-cols) layout doesn't isolate
                    # outliers to their col-block and gives ~10% NS
                    # drift (vs 1.45% here).
                    st.exp_avg_q = torch.zeros(
                        n, dtype=torch.float8_e4m3fn, device="cpu",
                    ).pin_memory()
                    st.exp_avg_scale_dim1 = torch.zeros(
                        rows, n_col_blocks, dtype=torch.float32, device="cpu",
                    ).pin_memory()
                    st.exp_avg_scale_dim2 = torch.zeros(
                        cols, n_row_blocks, dtype=torch.float32, device="cpu",
                    ).pin_memory()
                    st.exp_avg_bf16 = torch.zeros(
                        n, dtype=storage_dtype, device="cpu",
                    ).pin_memory()
                    st.exp_avg_2d_shape = shape
                    st.exp_avg_block = block_size
                    # exp_avg stays None in this mode; the BF16
                    # scratch buffer is exp_avg_bf16.
                    st.exp_avg = None
                else:
                    # Non-multiple-of-block param: keep the legacy
                    # BF16 buffer. The FP8 storage savings on
                    # these tiny params are negligible vs the
                    # code complexity of padding support.
                    st.exp_avg = torch.zeros(
                        n, dtype=storage_dtype, device="cpu",
                    ).pin_memory()
            self.state[id(p)] = st

    def register_nvfp4_module(self, module) -> None:
        """Register an :class:`NVFP4*Linear` with
        ``no_bf16_master=True`` so its FP4 packed buffers get Muon
        updates.

        Mirrors the per-param entry created in :meth:`__init__` —
        BF16 storage for both ``grad`` and ``exp_avg`` (the only
        supported layout since the legacy ``precision:`` yml block
        was removed 2026-07-23).

        In all cases the apply at step time routes through
        :meth:`NVFP4Linear.apply_chunk_update` (the module-side
        material / commit / apply API). No full BF16 master is
        ever materialized.
        """
        assert getattr(module, "no_bf16_master", False), (
            f"register_nvfp4_module requires no_bf16_master=True; "
            f"got {type(module).__name__}"
        )
        cols = getattr(
            module, "in_features_per_partition", module.in_features,
        )
        rows = module.out_features
        n = rows * cols
        shape = (rows, cols)
        key = id(module)
        if key in self.state:
            return

        # Replicate the storage decision from __init__ but keyed
        # on the module's shape (not a Parameter's). Same BF16
        # storage so all state is consistent across the model.
        storage_dtype = torch.bfloat16

        st = _ParamState(
            param=None,
            grad=torch.zeros(
                n, dtype=storage_dtype, device="cpu",
            ).pin_memory(),
            exp_avg=torch.zeros(
                n, dtype=storage_dtype, device="cpu",
            ).pin_memory(),
            kind="muon_nvfp4",
            shape=shape,
            nvfp4_module=module,
            nvfp4_n=n,
            nvfp4_chunk_ranges=module.chunk_ranges(),
        )
        self.state[key] = st
        # Install the in-backward D2H callback on the module.
        # Routes BF16 grad_w → cast → D2H → event-record →
        # worker-enqueue through the custom autograd Function's
        # ``backward()`` (via :meth:`NVFP4Linear._stash_grad_w`)
        # so the per-NVFP4 D2H overlaps with subsequent layers'
        # bwd kernels instead of waiting for
        # :func:`accumulate_grads_to_cpu` post-backward.
        #
        # Resolved at registration time so the closure holds
        # only the target tensor + cast dtype (no per-call
        # state-dict lookup on the hot path).
        target, cast_dtype = _accumulator_target(st)
        module._nvfp4_offload_cb = _make_nvfp4_offload_cb(
            module, target, cast_dtype,
        )

    def zero_grad(self, set_to_none: bool = True) -> None:
        for s in self.state.values():
            # NVFP4 mode-3: no Param.grad to free (the autograd
            # Function stashes grad_w on the module between
            # forward/backward and the optimizer step; that's
            # consumed by accumulate_grads_to_cpu).
            if s.nvfp4_module is not None:
                continue
            s.param.grad = None
            # NOTE: do NOT zero ``grad`` here — that would discard
            # the user's gradient accumulation across micro-batches.
            # The training loop is responsible for resetting
            # ``s.grad`` at the end of each accumulation cycle (via
            # :meth:`step` or :func:`zero_cpu_grad_accum`).

    def _newton_schulz(self, x: torch.Tensor) -> torch.Tensor:
        a, b, c = self._NS_COEFFS
        # Cast to FP16 for tensor-core matmul speed. The
        # orthogonalization is robust to FP16 noise for typical
        # gradient magnitudes.
        x = x.to(torch.float16)
        # Normalize by Frobenius norm before NS. The polynomial
        # coefficients (a, b, c) = (3.4445, -4.7750, 2.0315) have
        # fixed points at singular values σ ≈ 0.868 and σ ≈ 1.265;
        # the iteration only converges for σ in that band. Real
        # gradient matrices can have singular values well outside
        # the band (after accumulation, condition numbers of 100+
        # are common); without this normalization NS diverges and
        # every param becomes NaN on the second step.
        # The standard Muon reference (Keller Jordan's muon.py)
        # normalizes by ``X.norm() + eps`` for the same reason.
        eps = 1e-7
        x = x / (x.norm() + eps)
        if x.size(0) > x.size(1):
            g = x
            for _ in range(self.ns_steps):
                gt = g.t()
                xtx = gt @ g                                  # [in, in]
                inner = (
                    a * torch.eye(xtx.size(0), device=g.device, dtype=g.dtype)
                    + b * xtx
                    + c * xtx @ xtx
                )
                g = g @ inner
            return g
        else:
            g = x
            for _ in range(self.ns_steps):
                ggt = g @ g.t()                               # [out, out]
                inner = (
                    a * torch.eye(ggt.size(0), device=g.device, dtype=g.dtype)
                    + b * ggt
                    + c * ggt @ ggt
                )
                g = inner @ g
            return g

    # Per-chunk row count for the streaming NS path. Each chunk
    # is a 2-D ``[CHUNK_ROWS, cols]`` matrix; for embed
    # (cols=1024) the chunk is ``[4096, 1024]`` = 8 MB FP16.
    _STREAM_CHUNK_ROWS = 4096

    def step(self) -> None:
        """One Muon step across every trainable param in this
        optimizer's state.

        For each param with a non-zero ``s.grad`` (the per-cycle
        grad accumulator):

        1. Compute SGD momentum on CPU:
           ``exp_avg ← β·exp_avg + grad`` (BF16 in place).
        2. Compute the Nesterov-corrected input on CPU:
           ``g_for_ns = grad + β·exp_avg`` (nesterov=True) or
           ``g_for_ns = exp_avg`` (nesterov=False).
        3. H2D ``g_for_ns``, cast to FP32, view as 2-D.
        4. Stream NS over rows and apply the update to the GPU
           param: ``param -= lr * NS(g_for_ns)``.
        5. Stream sync + ``s.grad.zero_()`` for the next cycle
           (``s.exp_avg`` is preserved across steps).

        State stays on CPU pinned memory; the per-mb VRAM peak
        is one param's worth of buffers (~21 MB for a
        1024×3072 down_proj) regardless of storage dtype.
        """
        lr = self.lr
        beta = self.momentum
        nesterov = self.nesterov
        wd = self.weight_decay
        CHUNK_ROWS = self._STREAM_CHUNK_ROWS
        # Collect grad tensors for the deferred fused zero at
        # the end of step(). Only entries that actually did work
        # (had non-zero accumulated grad) are included — the
        # early-exit skip below means inactive params' grads are
        # already zero, so re-zeroing them would be wasted
        # bandwidth.
        grads_to_zero: list = []
        # Per-param early-exit + sync are intentional (see notes
        # below). An experiment that hoisted the early-exit and
        # per-param stream sync out of the loop into a single
        # final sync showed the all-async path runs ~2.5× slower
        # on a 5060 Ti (~700 ms median for legacy per-param pattern
        # vs ~1700 ms for the refactor) at base.yml shape (~224
        # Muon entries). The bottleneck is GPU-side — without
        # per-param syncs the long H2D / matmul / D2H queue
        # causes PyTorch's CUDA stream to stall (probably
        # pinned-memory DMA back-pressure or matmul-launch
        # oversubscription). The legacy per-param
        # ``.abs().sum().item()`` adds ~5 µs of host bookkeeping
        # per call but is essentially free; the per-param stream
        # ``synchronize()`` returns ~immediately when only one
        # matmul is in flight.
        # -----
        # Lesson: do **not** "optimize" this loop by removing the
        # per-param sync or merging it into one final sync — the
        # empirical floor is set by GPU stream behavior, not by
        # Python-side overhead. If a future cuBLAS / CUDA runtime
        # change makes the all-async path faster, this comment
        # should be re-validated against a fresh microbench.
        for s in self.state.values():
            # Per-step grad accumulator (post-2026-07-15; was
            # ``s.mom_buf`` in the merged-accumulator design).
            g_buf = s.grad
            # Early-exit on no accumulated grad. The storage
            # dtype is always a floating dtype (bf16 / fp16 /
            # fp32 — quantized storage removed 2026-07-12).
            # ``g_buf`` is in pinned CPU memory; ``.item()`` is
            # a host-side scalar read — ~5 µs, negligible. The
            # host-vs-GPU interleaving it introduces is actually
            # what keeps the per-param stream from oversubscribing
            # (see the loop preamble).
            if g_buf.abs().sum().item() == 0:
                continue
            shape = s.shape
            rows, cols = shape[0], shape[1]
            module = s.nvfp4_module  # None for legacy params
            # For mode-3 we don't have ``s.param.device`` (no leaf
            # Parameter). The packed buffers live on whatever device
            # the module was moved to via ``.cuda()`` /
            # ``register_load_state_dict_post_hook`` —
            # ``module.packed_weight.device`` is the source of truth.
            device = (
                module.packed_weight.device if module is not None
                else s.param.device
            )
            # ---- SGD momentum update on CPU (post-2026-07-15;
            # replaces the merged-accumulator trick where the
            # accumulator WAS the momentum, with β=1 effectively).
            # ``exp_avg ← β·exp_avg + grad`` in BF16 in place
            # (same storage dtype as ``g_buf`` so the EMA
            # doesn't accumulate dtype-cast rounding). For
            # ``β = 0.95`` and an EMA of per-cycle grads, the
            # magnitudes stay well inside BF16's 7-bit mantissa
            # range — same precision as the legacy FP32 path in
            # practice.
            #
            # This is the only place the EMA smoothing happens:
            # before this update, ``s.exp_avg`` holds the prior
            # step's momentum (or zero on step 0); after the
            # update, it holds the current step's momentum and
            # is preserved across steps for the next ``step()``.
            #
            # In FP8 2D tight scale mode, ``s.exp_avg`` is
            # ``None`` and the BF16 working buffer is
            # ``s.exp_avg_bf16``. We dequant the FP8 storage
            # into that buffer at the start, run the same
            # in-place ``β·prev + grad`` math on the BF16
            # buffer, then requantize the result back into the
            # FP8 + scales at the end. The actual storage on
            # disk / CPU is FP8 + 2 × FP32 scale tensors; the
            # BF16 buffer is per-step scratch that is
            # overwritten each step.
            if s.exp_avg is not None:
                exp_avg = s.exp_avg
            else:
                # FP8 2D tight scale path.
                rows, cols = s.exp_avg_2d_shape[0], s.exp_avg_2d_shape[1]
                deq = self._dequantize_2d(
                    s.exp_avg_q, s.exp_avg_scale_dim1, s.exp_avg_scale_dim2,
                    rows=rows, cols=cols, block=s.exp_avg_block,
                )
                s.exp_avg_bf16.copy_(deq.reshape(-1))
                exp_avg = s.exp_avg_bf16
            exp_avg.mul_(beta).add_(g_buf)
            if s.exp_avg is None:
                # FP8 2D tight scale: requantize BF16 back to FP8.
                rows, cols = s.exp_avg_2d_shape[0], s.exp_avg_2d_shape[1]
                q, s1, s2, _tight = self._quantize_2d(
                    exp_avg.view(rows, cols), block=s.exp_avg_block,
                )
                s.exp_avg_q.copy_(q)
                s.exp_avg_scale_dim1.copy_(s1)
                s.exp_avg_scale_dim2.copy_(s2)
            # ---- Nesterov correction on CPU. The classical Muon
            # formulation (Keller Jordan): if nesterov, the
            # input to NS is ``g + β·buf`` (the "look-ahead"),
            # otherwise it's just ``buf``. Computing on CPU
            # avoids an extra GPU buffer + an extra H2D for the
            # look-ahead term; the cost is one CPU-side
            # elementwise add per step (negligible vs the H2D).
            if nesterov:
                g_for_ns = g_buf.add(exp_avg, alpha=beta)
            else:
                g_for_ns = exp_avg
            # ---- H2D + cast to FP32 (NS input is always FP32).
            orth_input = g_for_ns.to(device, non_blocking=True) \
                                    .float().view(rows, cols)

            # ---- Decoupled weight decay: applied once to the
            # full GPU param (in place), before the streaming
            # NS apply loop. For mode-3 NVFP4 the full-param
            # materialize we'd need here defeats the whole point
            # of the chunked apply — instead we fold the WD
            # scalar into each chunk's apply as
            # ``bf16 *= (1 - lr * wd)``. Scalar multiplication
            # distributes over the chunk partition, so the
            # per-chunk math is mathematically identical to the
            # legacy "once-on-full-param" version. ----
            if wd != 0.0 and module is None:
                s.param.data.mul_(1.0 - lr * wd)
            wd_factor = 1.0 - lr * wd if wd != 0.0 else 1.0

            # ---- Stream NS over rows; orth_input is already on
            # GPU so no per-chunk H2D is needed. For mode-3
            # NVFP4: the apply destination is the module's
            # ``decay_and_apply_chunk`` (material BF16 view,
            # fold the WD scalar in, subtract lr * update_chunk,
            # commit back to FP4 — no full BF16 master
            # materialize). The NS run itself is identical to
            # legacy; only the apply destination differs. ----
            m_fp16 = orth_input.to(torch.float16)
            if module is None:
                target_dtype = s.param.dtype
            else:
                # Mode-3: material_chunk returns BF16; the
                # commit back to FP4 quantizes from BF16.
                # Casting the update to BF16 here keeps the
                # math the same precision as legacy.
                target_dtype = torch.bfloat16
            # Defer the per-chunk Marlin scales-cache rebuild
            # to the last chunk only: the cache is consumed by
            # the *next* step's fwd, and the training loop's
            # end-of-step ``repack_nvfp4_weights`` already
            # rebuilds it once per module per step. Per-chunk
            # rebuilds (the pre-change default) are dead
            # compute — (N_chunks-1) wasted rebuilds per
            # module per step. Each rebuild is ~800 µs at
            # base.yml FFN shapes (microbench
            # ``bench_nvfp4_apply_chunk.py``).
            n_chunks = (rows + CHUNK_ROWS - 1) // CHUNK_ROWS
            for chunk_i in range(n_chunks):
                r_start = chunk_i * CHUNK_ROWS
                r_end = min(r_start + CHUNK_ROWS, rows)
                m_chunk = m_fp16[r_start:r_end]
                update = self._newton_schulz(m_chunk)
                update = update.to(target_dtype)
                if module is None:
                    s.param.data[r_start:r_end].add_(update, alpha=-lr)
                else:
                    # Module rows align 1:1 with the optimizer's
                    # ``m_fp16[r_start:r_end]`` view (both are
                    # the local-partitioned FP4 weight, in our
                    # row-major convention). For TP row-parallel
                    # the partition is along the input axis, not
                    # output — but ``module.apply_chunk_update``
                    # and ``module.chunk_ranges`` already use the
                    # local row indexing the module exposes, so
                    # passing the same ``(r_start, r_end)``
                    # works without translation.
                    module.decay_and_apply_chunk(
                        r_start, r_end, update, lr, wd_factor=wd_factor,
                        rebuild_scales_cache=(chunk_i == n_chunks - 1),
                    )

            # ---- Sync the device stream before next param's H2D.
            # See the loop preamble for why this stays per-param
            # despite the obvious "1 sync at end is cheaper"
            # intuition — empirically the per-param version is
            # ~2.5× faster than the all-async version on 5060 Ti.
            # ----
            torch.cuda.current_stream(device).synchronize()

            # ---- Cycle-end housekeeping (deferred): queue
            # ``s.grad`` for the fused zero at the bottom of
            # step(). Replacing the per-param ``s.grad.zero_()``
            # with one C++ ``memset``-style call across all
            # entries eliminates the per-param Python+dispatch
            # overhead (~50 us each) and unlocks cross-op CPU
            # bandwidth via OpenMP. See
            # :mod:`src.training.param_offload.cpu_fused`. ----
            grads_to_zero.append(s.grad)

        # ---- End-of-step fused zero (all grads in one
        # C++ call). Falls back to per-tensor ``.zero_()`` if the
        # JIT extension failed to compile (no C++ toolchain at
        # runtime). ``s.exp_avg`` is preserved across steps
        # (NOT zeroed here) — only the per-step accumulator
        # ``s.grad`` is reset. ----
        fused_zero_many(grads_to_zero)

    # ------------------------------------------------------------------ #
    # Checkpoint save/load (resume).                                     #
    # ------------------------------------------------------------------ #
    # See the equivalent block on :class:`CPUAdamW` for the design.
    # Full-precision BF16 storage is the only layout
    # supported post-2026-07-12; quantized storage was
    # removed along with int8 / mxfp8 storage. The saved entry
    # contains ``grad`` (per-step accumulator — zero at save time
    # in steady state) and ``exp_avg`` (SGD momentum — preserved
    # across steps). ``load_state_dict`` restores both in place.
    #
    # State_dict keys changed on 2026-07-15 (was ``"mom_buf"`` for
    # the merged accumulator); pre-change checkpoints cannot be
    # loaded by the new code.
    def state_dict(self) -> dict:
        sd_entries = []
        for s in self.state.values():
            entry = {
                "shape": list(s.shape),
                "step": s.step,
                "kind": s.kind,
                "grad": s.grad,
            }
            if s.exp_avg is not None:
                entry["exp_avg"] = s.exp_avg
            else:
                # FP8 2D tight scale storage. All five fields are
                # required for an exact load; shape + block_size
                # allow the loader to sanity-check the layout
                # matches.
                entry["exp_avg_q"] = s.exp_avg_q
                entry["exp_avg_scale_dim1"] = s.exp_avg_scale_dim1
                entry["exp_avg_scale_dim2"] = s.exp_avg_scale_dim2
                entry["exp_avg_2d_shape"] = list(s.exp_avg_2d_shape)
                entry["exp_avg_block"] = s.exp_avg_block
            sd_entries.append(entry)
        return {
            "lr": self.lr,
            "momentum": self.momentum,
            "nesterov": self.nesterov,
            "ns_steps": self.ns_steps,
            "weight_decay": self.weight_decay,
            "state": sd_entries,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        self.lr = float(state_dict["lr"])
        self.momentum = float(state_dict["momentum"])
        self.nesterov = bool(state_dict["nesterov"])
        self.ns_steps = int(state_dict["ns_steps"])
        self.weight_decay = float(state_dict["weight_decay"])
        saved = state_dict.get("state", [])
        current = list(self.state.values())
        if len(saved) != len(current):
            raise ValueError(
                f"CPUMuon.load_state_dict: param count mismatch "
                f"(file has {len(saved)} entries, current optimizer "
                f"has {len(current)}). The model architecture likely "
                f"changed since this checkpoint was written."
            )
        for cur, sav in zip(current, saved):
            cur.step = int(sav.get("step", 0))
            if tuple(sav.get("shape", ())) != tuple(cur.shape):
                idx = current.index(cur)
                raise ValueError(
                    f"CPUMuon.load_state_dict: shape mismatch at "
                    f"param index {idx}: file={tuple(sav.get('shape', ()))} "
                    f"current={tuple(cur.shape)}."
                )
            if sav.get("grad") is not None:
                cur.grad.copy_(sav["grad"])
            if cur.exp_avg is not None:
                if sav.get("exp_avg") is not None:
                    cur.exp_avg.copy_(sav["exp_avg"])
            else:
                # FP8 2D tight scale path — restore all five
                # fields. A saved entry without them is a hard
                # error (no silent fallback to BF16; that would
                # mask a storage-mode mismatch and the next
                # step() would re-dequant a zero FP8 buffer).
                for k in ("exp_avg_q", "exp_avg_scale_dim1",
                         "exp_avg_scale_dim2", "exp_avg_2d_shape",
                         "exp_avg_block"):
                    if k not in sav:
                        raise ValueError(
                            f"CPUMuon.load_state_dict: FP8 2D tight "
                            f"scale storage on the live optimizer "
                            f"but saved entry missing key {k!r}"
                        )
                if tuple(sav["exp_avg_2d_shape"]) != tuple(cur.exp_avg_2d_shape):
                    sav_shape = tuple(sav["exp_avg_2d_shape"])
                    cur_shape = tuple(cur.exp_avg_2d_shape)
                    raise ValueError(
                        f"CPUMuon.load_state_dict: exp_avg_2d_shape "
                        f"mismatch (file={sav_shape} current={cur_shape})"
                    )
                if int(sav["exp_avg_block"]) != int(cur.exp_avg_block):
                    sav_block = int(sav["exp_avg_block"])
                    cur_block = int(cur.exp_avg_block)
                    raise ValueError(
                        f"CPUMuon.load_state_dict: exp_avg_block "
                        f"mismatch (file={sav_block} current={cur_block})"
                    )
                cur.exp_avg_q.copy_(sav["exp_avg_q"])
                cur.exp_avg_scale_dim1.copy_(sav["exp_avg_scale_dim1"])
                cur.exp_avg_scale_dim2.copy_(sav["exp_avg_scale_dim2"])