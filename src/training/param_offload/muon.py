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

from typing import List, Optional

import torch
import torch.nn as nn

from ..precision_config import PrecisionConfig
from ._state import _ParamState
from .cpu_fused import fused_zero_many


class CPUMuon:
    """Muon with configurable momentum storage on CPU pinned memory.

    State per param:
        CPU pinned: mom_buf (cfg dtype, numel)
        GPU:         param (FP16), .grad (transient)

    Merged-accumulator design: ``mom_buf`` doubles as the
    grad accumulator (mu=1 accumulation — each microbatch's grad
    is added to ``mom_buf`` in place). At ``step()`` time
    ``mom_buf`` is read (cast to FP32), orthogonalized via
    Newton-Schulz, applied to the GPU param, then reset to
    zero for the next accumulation cycle. The cross-step μ
    momentum smoothing has nothing to smooth across cycles
    (the buffer resets at every step).

    Storage dtype is ``precision.muon_momentum.dtype``:

    * ``bf16`` / ``fp16`` / ``fp32``: full-precision storage.
      The Newton-Schulz output precision is unaffected by the
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
    the GPU ``.grad`` to ``s.mom_buf`` (CPU, at the storage
    dtype). On :meth:`step` we H2D ``mom_buf``, cast to FP32,
    NS, apply the update to the GPU param, then recast back
    to the storage dtype and D2H.

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
        precision: Optional[PrecisionConfig] = None,
    ) -> None:
        """CPU-offloaded Muon with configurable precision.

        ``precision`` controls:

        - ``s.mom_buf`` (momentum + grad accumulator, merged)
          ← ``precision.muon_momentum`` (bf16 / fp16 / fp32).

        The ``gradients`` precision config is no longer used
        (there is no separate accumulator buffer).

        Supported momentum dtypes: full-precision ``bf16`` /
        ``fp16`` / ``fp32``. (Quantized ``int8`` / ``mxfp8``
        was removed on 2026-07-12.)
        """
        self.lr = lr
        self.momentum = momentum
        self.nesterov = nesterov
        self.ns_steps = ns_steps
        self.weight_decay = weight_decay
        precision = precision or PrecisionConfig()

        # fp* muon: full-precision storage at the configured
        # dtype; no scale, no dequant/requant in step().
        mom_storage_dtype = precision.muon_momentum.dtype.to_torch()

        # Keep the precision config so register_nvfp4_module can
        # mirror the storage decisions for module-based (no-BF16-
        # master) entries.
        self._precision = precision

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
                mom_buf=torch.zeros(
                    n, dtype=mom_storage_dtype, device="cpu",
                ).pin_memory(),
                kind="muon",
                shape=shape,
            )
            self.state[id(p)] = st

    def register_nvfp4_module(self, module) -> None:
        """Register an :class:`NVFP4*Linear` with
        ``no_bf16_master=True`` so its FP4 packed buffers get Muon
        updates.

        Mirrors the per-param entry created in :meth:`__init__` —
        the dtype config (``precision.muon_momentum``) controls
        the storage format (``mom_buf`` dtype only; no scale, no
        separate ``accum`` in the post-2026-07-12 design).

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

        # Replicate the storage decision from __init__ but keyed on
        # the module's shape (not a Parameter's). Same precision
        # config so all storage is consistent across the model.
        precision = self._precision
        mom_storage_dtype = precision.muon_momentum.dtype.to_torch()

        st = _ParamState(
            param=None,
            mom_buf=torch.zeros(
                n, dtype=mom_storage_dtype, device="cpu",
            ).pin_memory(),
            kind="muon_nvfp4",
            shape=shape,
            nvfp4_module=module,
            nvfp4_n=n,
            nvfp4_chunk_ranges=module.chunk_ranges(),
        )
        self.state[key] = st

    def zero_grad(self, set_to_none: bool = True) -> None:
        for s in self.state.values():
            # NVFP4 mode-3: no Param.grad to free (the autograd
            # Function stashes grad_w on the module between
            # forward/backward and the optimizer step; that's
            # consumed by accumulate_grads_to_cpu).
            if s.nvfp4_module is not None:
                continue
            s.param.grad = None

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

        Merged-accumulator design (post-2026-07-12): ``mom_buf``
        doubles as the accumulator for every storage dtype.
        Per-mb grad accumulation happens via
        :func:`accumulate_grads_to_cpu` (a cheap CPU add at the
        storage dtype — bf16 by default). ``mom_buf`` is
        consumed at step() time: H2D, cast to FP32, NS, apply
        update, recast back to storage dtype, D2H, zero for the
        next cycle.

        No cross-step μ momentum smoothing is applied — the
        gradient accumulator resets to zero at the end of
        every step, so the smoothing has nothing to smooth
        across cycles. The momentum-smoothing benefit was the
        only reason for the μ coefficient in the original
        SGD-momentum design; with mu=1 accumulation the same
        effect is achieved trivially by the per-cycle sum.

        State stays on CPU pinned memory; the per-mb VRAM peak
        is one param's worth of buffers (~21 MB for a
        1024×3072 down_proj) regardless of storage dtype.
        """
        lr = self.lr
        wd = self.weight_decay
        CHUNK_ROWS = self._STREAM_CHUNK_ROWS
        # Collect mom_buf tensors for the deferred fused zero at
        # the end of step(). Only entries that actually did work
        # (had non-zero accumulated grad) are included — the
        # early-exit skip below means inactive params' mom_bufs
        # are already zero, so re-zeroing them would be wasted
        # bandwidth.
        mom_bufs_to_zero: list = []
        for s in self.state.values():
            # Merged-accumulator design (all storage dtypes):
            # ``mom_buf`` is the cycle's accumulated grad.
            g_buf = s.mom_buf
            # Early-exit on no accumulated grad. The storage
            # dtype is always a floating dtype (bf16 / fp16 /
            # fp32 — quantized storage removed 2026-07-12).
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
            # fp* muon: merged-accumulator design. H2D mom_buf,
            # cast to FP32 (NS input is always FP32).
            mom_buf_gpu = g_buf.to(device, non_blocking=True)
            m_fp32_gpu = mom_buf_gpu.float().view(rows, cols)

            # ---- No SGD update here: the accumulator IS the
            # accumulated grad (mu=1). We orthogonalize it as-is. ----

            # ---- Recast FP32 back to the configured storage
            # dtype + D2H (the only precision loss the
            # full-precision path incurs). ----
            new_mom = m_fp32_gpu.to(s.mom_buf.dtype).view(-1)
            s.mom_buf.copy_(new_mom, non_blocking=True)
            orth_input = m_fp32_gpu

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
            # Ensures the D2H above completed; otherwise the next
            # param could overwrite the pinned host buffer before
            # this DMA landed. ----
            torch.cuda.current_stream(device).synchronize()

            # ---- Cycle-end housekeeping (deferred): queue
            # ``mom_buf`` for the fused zero at the bottom of
            # step(). Replacing the per-param ``s.mom_buf.zero_()``
            # with one C++ ``memset``-style call across all
            # entries eliminates the per-param Python+dispatch
            # overhead (~50 us each) and unlocks cross-op CPU
            # bandwidth via OpenMP. See
            # :mod:`src.training.param_offload.cpu_fused`. ----
            mom_bufs_to_zero.append(s.mom_buf)

        # ---- End-of-step fused zero (all mom_bufs in one
        # C++ call). Falls back to per-tensor ``.zero_()`` if the
        # JIT extension failed to compile (no C++ toolchain at
        # runtime). ----
        fused_zero_many(mom_bufs_to_zero)

    # ------------------------------------------------------------------ #
    # Checkpoint save/load (resume).                                     #
    # ------------------------------------------------------------------ #
    # See the equivalent block on :class:`CPUAdamW` for the design.
    # Full-precision storage (``mom_scale is None``) is the only
    # path supported post-2026-07-12; quantized storage
    # (``mom_scale is not None``) was removed along with int8 /
    # mxfp8 storage. The saved entry contains ``mom_buf`` only;
    # ``load_state_dict`` restores it in place.
    def state_dict(self) -> dict:
        return {
            "lr": self.lr,
            "momentum": self.momentum,
            "nesterov": self.nesterov,
            "ns_steps": self.ns_steps,
            "weight_decay": self.weight_decay,
            "state": [
                {
                    "shape": list(s.shape),
                    "step": s.step,
                    "kind": s.kind,
                    "mom_buf": s.mom_buf,
                }
                for s in self.state.values()
            ],
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
                raise ValueError(
                    f"CPUMuon.load_state_dict: shape mismatch at "
                    f"param index {current.index(cur)}: file="
                    f"{tuple(sav.get('shape', ()))} current="
                    f"{tuple(cur.shape)}."
                )
            if sav.get("mom_buf") is not None:
                cur.mom_buf.copy_(sav["mom_buf"])
