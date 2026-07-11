"""Muon with CPU-offloaded momentum.

:class:`CPUMuon` is the other half of
:mod:`src.training.param_offload`'s optimizer pair. It owns the
Newton-Schulz orthogonalization step and the per-storage-dtype
dispatch (int8 + per-row BF16 scale, mxfp8 + per-block E8M0, or
full-precision fp16/bf16/fp32). The split file layout keeps the
optimizer algorithm in one place without the offload plumbing
interleaving it.

Implementation notes are on the class docstring; see
:mod:`src.training.param_offload` for the package-level design
rationale.
"""
from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn

from ..ops.mxfp8_accum import fused_mxfp8_dequant_add_requant
from ..precision_config import DType, PrecisionConfig
from ._state import _ParamState, _quantize_accum_to_mom_buf


class CPUMuon:
    """Muon with configurable momentum storage on CPU pinned memory.

    State per param:
        CPU pinned: mom_buf (cfg dtype, numel),
                    mom_scale (BF16, rows or None)
        GPU:         param (FP16), .grad (transient)

    No separate accumulator buffer: ``mom_buf`` doubles as the
    grad accumulator (mu=1 accumulation — each microbatch's grad
    is added to ``mom_buf`` in place). At ``step()`` time
    ``mom_buf`` is read (dequantized for int8), orthogonalized
    via Newton-Schulz, applied to the GPU param, then reset to
    zero for the next accumulation cycle. The cross-step μ
    momentum smoothing has nothing to smooth across cycles
    (the buffer resets at every step).

    Storage dtype is ``precision.muon_momentum.dtype``:

    * ``int8`` + per-channel scale (canonical default): 1 byte/elt
      momentum + a tiny ``rows``-sized BF16 scale tensor. The
      2-D momentum matrix ``m`` is stored as a flat int8 buffer
      reshaped to ``[rows, cols]`` plus a per-row BF16 scale
      factor (symmetric quantization::

          scale[i] = max(|m[i, :]|) / 127
          q[i, j] = round(m[i, j] / scale[i]).clip(-128, 127)

      ). The BF16 scale (vs FP16 in the v0.0.2 design) is the
      key numerical choice for the new 5060Ti hardware: BF16's
      8-bit exponent matches FP32, so the per-row scale never
      overflows even for very small per-row maxes. FP16's 5-bit
      exponent could underflow ``scale`` to 0 for very quiet
      rows, which would then divide-by-zero on dequant. BF16
      has the same 2-byte footprint as FP16.
    * ``mxfp8`` + per-block E8M0 scale: 1 byte/elt E4M3 momentum
      + 1 byte per ``block_size``-element block of E8M0 scale
      (OCP MX spec; default ``block_size=32``). The 2-D momentum
      matrix ``m`` is stored as a flat ``float8_e4m3fn`` buffer
      reshaped to ``[rows, cols_padded]`` (``cols_padded`` rounds
      up to a multiple of ``block_size`` — the tail block is all
      zeros so it contributes nothing on dequant) plus a per-block
      ``float8_e8m0fnu`` scale tensor of shape
      ``[rows, cols_padded // block_size]``. The quantize recipe::

          for each block of `block_size` contiguous elements along cols:
              absmax = max(|m|) within the block
              target_scale = absmax / E4M3_MAX
              scale = round_to_nearest_power_of_2(target_scale)
              q[i] = round_to_e4m3(m[i] / scale)

      Same per-mb dequant → add → requant overhead as int8 (the
      scale tensor tracks each block's growing max-abs across
      accumulation); the storage savings vs int8 are negligible
      (E8M0 is 1 byte vs BF16 2 bytes, but mxfp8 has ~cols/block_size
      scale entries per row). The numerical advantage is the
      per-block scale granularity — within-row scale variation
      is captured without sacrificing global dynamic range.
    * ``bf16`` / ``fp16`` / ``fp32``: full-precision momentum, no
      quantization. ``mom_scale`` is ``None``. The Newton-Schulz
      output precision is unaffected by the storage dtype (NS
      orthogonalizes the dequantized / raw FP32 momentum on the
      GPU regardless); the storage precision only affects how
      much the EMA buffer is rounded between steps.

    The training loop's :func:`accumulate_grads_to_cpu` adds the
    GPU ``.grad`` to ``accum`` (CPU, BF16). On :meth:`step` we:

    * ``int8``: dequant to FP32 on the GPU, SGD in FP32,
      requant to int8 + BF16 scale, D2H the new storage.
    * ``mxfp8``: dequant to FP32 on the GPU, SGD in FP32,
      requant to E4M3 + E8M0 scale, D2H the new storage.
    * ``fp16``/``bf16``/``fp32``: H2D the raw momentum, SGD in
      FP32, cast back to the storage dtype, D2H. No dequant /
      requant cycle.

    Then stream each row-chunk to the GPU, run 5 NS iterations
    in FP16 (tensor cores), and apply the update to the GPU
    param. The NS path is identical for both storage variants.
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

        - ``s.mom_buf``   (momentum + grad accumulator, merged)
          ← ``precision.muon_momentum`` (int8 with per-row
          quantization, mxfp8 with per-block E8M0 scale, or raw
          fp16/bf16/fp32). ``mom_buf`` doubles as the accumulator
          (mu=1 accumulation).
        - ``s.mom_scale`` (scale tensor) ← set for int8 (BF16
          per-row) or mxfp8 (E8M0 per-block). ``None`` for fp*
          storage.
        - ``s.mxfp8_block_size`` (block size for E8M0 scale) ←
          set for mxfp8 (default 32), ``None`` for int8 / fp*.

        The ``gradients`` precision config is no longer used
        (there is no separate accumulator buffer).

        Supported momentum dtypes: ``int8`` (default, per-row
        BF16 scale), ``mxfp8`` (per-block E8M0 scale, default
        block_size=32), and full-precision ``bf16`` / ``fp16`` /
        ``fp32``. ``int4`` raises :class:`NotImplementedError`
        (the packing scheme is on the roadmap but not landed).
        """
        self.lr = lr
        self.momentum = momentum
        self.nesterov = nesterov
        self.ns_steps = ns_steps
        self.weight_decay = weight_decay
        precision = precision or PrecisionConfig()

        # Resolve momentum storage. int8 / mxfp8 are the quantized
        # storage paths (with their respective scale tensors);
        # fp* is full-precision. int4 is on the roadmap but not yet
        # implemented (the mom_buf would need to pack 2 elements
        # per byte and the dequant path would need to unpack).
        # The ``gradients`` precision config is no longer used
        # (no separate accumulator buffer — mom_buf doubles as
        # the accumulator).
        mom_dtype_cfg = precision.muon_momentum.dtype
        if mom_dtype_cfg == DType.INT4:
            raise NotImplementedError(
                "CPUMuon: muon_momentum dtype=int4 is not yet implemented."
                " Use int8 (the current default), mxfp8, or a floating"
                " dtype (bf16 / fp16 / fp32)."
            )
        # int8 → 1 byte/elt int8 + a BF16 per-row scale.
        # mxfp8 → 1 byte/elt E4M3 + 1 byte per `block_size`-elt
        #          E8M0 scale (default block_size=32; set in
        #          TensorPrecision.__post_init__ when dtype=mxfp8).
        # fp*  → full-precision storage at the configured dtype;
        #         no scale, no dequant/requant in step().
        if mom_dtype_cfg.is_integer:
            mom_storage_dtype = mom_dtype_cfg.to_torch()  # int8
            scale_dtype = torch.bfloat16
            scale_shape_fn = lambda shape: (shape[0],)
            block_size = None
            quantized = True
        elif mom_dtype_cfg.is_mxfp:
            mom_storage_dtype = mom_dtype_cfg.to_torch()  # float8_e4m3fn
            scale_dtype = torch.float8_e8m0fnu
            block_size = precision.muon_momentum.block_size
            # Padded cols is rounded up to a multiple of block_size
            # so the partial tail block is full of zeros (which
            # quantize to zero with a zero scale — contributes
            # nothing on dequant).
            scale_shape_fn = lambda shape, bs=block_size: (
                shape[0], (shape[1] + bs - 1) // bs,
            )
            quantized = True
        else:
            mom_storage_dtype = mom_dtype_cfg.to_torch()  # bf16/fp16/fp32
            scale_dtype = None
            scale_shape_fn = None
            block_size = None
            quantized = False

        # Keep the precision config so register_nvfp4_module can
        # mirror the storage decisions for module-based (no-BF16-
        # master) entries — without this the fp4 module path
        # wouldn't know whether to allocate int8+mxfp8+BF16 scale
        # or just an FP32 mom_buf.
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
            if scale_shape_fn is None:
                scale_shape: tuple[int, ...] = ()
            else:
                scale_shape = scale_shape_fn(shape)
            mom_scale = (
                torch.zeros(scale_shape, dtype=scale_dtype, device="cpu").pin_memory()
                if scale_dtype is not None else None
            )
            # E8M0 byte 0 decodes to 0.0 → divide-by-zero on dequant.
            # Initialize all mxfp8 scale bytes to 1 (=2^-126, the safe
            # floor) so a freshly-loaded param doesn't NaN out on the
            # first dequant (the kernel overwrites these every cycle
            # anyway, so this is purely a pre-first-step floor).
            if scale_dtype is torch.float8_e8m0fnu:
                mom_scale = torch.full(
                    scale_shape, 1, dtype=torch.uint8,
                    device="cpu",
                ).view(torch.float8_e8m0fnu).pin_memory()
            # For MXFP8 storage, the per-block scale layout
            # requires ``mom_buf`` to be sized ``rows * cols_p``
            # (cols rounded up to a multiple of block_size), not
            # the original ``rows * cols``. The trailing padded
            # block is filled with zeros (no per-mb contribution)
            # so it contributes nothing on dequant. Without this
            # padding, the ``view(rows, n_blocks, bs)`` reshape
            # in the dequant/requant helpers would raise on any
            # param whose K dim is not a multiple of ``block_size``
            # (e.g. KDA f_proj1/g_proj1 in tiny models).
            storage_n = n
            if block_size is not None and p.ndim == 2:
                cols_dim = shape[1]
                cols_p = cols_dim if cols_dim % block_size == 0 \
                    else cols_dim + (block_size - cols_dim % block_size)
                storage_n = shape[0] * cols_p
            # Separate bf16 accumulator for int8 muon only. For
            # mxfp8 we now do per-mb in-place accumulation on the
            # mxfp8 ``mom_buf`` itself via the fused C++ kernel
            # (:func:`_mxfp8_apply_grad`), which is what the
            # mxfp8 design was for: save the 2 bytes/elt of an
            # extra bf16 accum. For fp* muon the merged-accumulator
            # design still applies (``mom_buf`` doubles as the
            # accumulator), so ``accum`` stays ``None``.
            is_mxfp8_param = (block_size is not None and p.ndim == 2)
            accum = (
                torch.zeros(n, dtype=torch.bfloat16, device="cpu").pin_memory()
                if quantized and not is_mxfp8_param else None
            )
            st = _ParamState(
                param=p,
                mom_buf=torch.zeros(storage_n, dtype=mom_storage_dtype, device="cpu").pin_memory(),
                mom_scale=mom_scale,
                mxfp8_block_size=block_size,
                accum=accum,
                kind="muon",
                shape=shape,
            )
            self.state[id(p)] = st

    def register_nvfp4_module(self, module) -> None:
        """Register an :class:`NVFP4*Linear` with
        ``no_bf16_master=True`` so its FP4 packed buffers get Muon
        updates.

        Mirrors the per-param entry created in :meth:`__init__` —
        the dtype config (``precision.muon_momentum``) controls the
        storage format (``mom_buf`` dtype + optional scale + optional
        separate ``accum``). For the canonical config (int8 + BF16
        per-row scale) the per-mb accumulator is ``s.accum`` and the
        step requires a dequant-add-requant at step end. For
        ``bf16`` Muon (full precision) we use the merged-accumulator
        design (``mom_buf`` IS the accumulator; no separate
        ``accum``).

        In all cases the apply at step time routes through
        :meth:`NVFP4Linear.apply_chunk_update` (the module-side
        material / commit / apply API). No full BF16 master is ever
        materialized.
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

        # Replicate the storage decisions from __init__ but keyed on
        # the module's shape (not a Parameter's). Same precision
        # config so int8 vs mxfp8 vs bf16 stays consistent across
        # the model — the only thing that varies is the absence of
        # an ``nn.Parameter`` to point at.
        precision = self._precision  # populated below if missing
        mom_dtype_cfg = precision.muon_momentum.dtype
        if mom_dtype_cfg == DType.INT4:
            raise NotImplementedError(
                "CPUMuon: muon_momentum dtype=int4 is not yet implemented."
            )
        if mom_dtype_cfg.is_integer:
            mom_storage_dtype = mom_dtype_cfg.to_torch()
            scale_dtype = torch.bfloat16
            scale_shape: tuple[int, ...] = (rows,)
            block_size = None
            quantized = True
        elif mom_dtype_cfg.is_mxfp:
            mom_storage_dtype = mom_dtype_cfg.to_torch()
            scale_dtype = torch.float8_e8m0fnu
            block_size = precision.muon_momentum.block_size
            cols_p = cols if cols % block_size == 0 \
                else cols + (block_size - cols % block_size)
            scale_shape = (rows, cols_p // block_size)
            quantized = True
        else:
            mom_storage_dtype = mom_dtype_cfg.to_torch()
            scale_dtype = None
            scale_shape = ()
            block_size = None
            quantized = False

        storage_n = n
        if block_size is not None:
            cols_p = cols if cols % block_size == 0 \
                else cols + (block_size - cols % block_size)
            storage_n = rows * cols_p

        is_mxfp8_param = block_size is not None
        accum = (
            torch.zeros(n, dtype=torch.bfloat16, device="cpu").pin_memory()
            if quantized and not is_mxfp8_param else None
        )
        mom_scale = (
            torch.zeros(scale_shape, dtype=scale_dtype, device="cpu").pin_memory()
            if scale_dtype is not None else None
        )
        # E8M0 floor: initialize all scale bytes to 1 (= 2^-126).
        if scale_dtype is torch.float8_e8m0fnu:
            mom_scale = torch.full(
                scale_shape, 1, dtype=torch.uint8, device="cpu",
            ).view(torch.float8_e8m0fnu).pin_memory()

        st = _ParamState(
            param=None,
            mom_buf=torch.zeros(storage_n, dtype=mom_storage_dtype, device="cpu").pin_memory(),
            mom_scale=mom_scale,
            mxfp8_block_size=block_size,
            accum=accum,
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

    def _dequantize(self, s: _ParamState) -> torch.Tensor:
        """Return dequantized FP32 momentum, shape ``s.shape``.

        Runs on the CPU. Reads ``s.mom_buf`` (int8) and
        ``s.mom_scale`` (BF16); returns ``[rows, cols]`` in FP32.
        The FP32 cast of the BF16 scale is free (BF16→FP32 is
        lossless), and the per-row scale is then broadcast across
        the cols dimension.

        Only valid when ``s.mom_buf.dtype == torch.int8``. For
        floating-dtype momentum, read ``s.mom_buf.float()``
        directly (no scale to multiply).
        """
        rows, cols = s.shape[0], s.shape[1]
        q_2d = s.mom_buf.view(rows, cols).float()
        scale_2d = s.mom_scale.float().unsqueeze(1)  # [rows, 1], FP32
        return q_2d * scale_2d  # FP32, [rows, cols]

    def _requantize(self, m_fp32: torch.Tensor, s: _ParamState) -> None:
        """Per-row symmetric int8 quantize ``m_fp32`` into
        ``s.mom_buf`` (int8) / ``s.mom_scale`` (BF16).

        ``m_fp32`` is the SGD-updated momentum in FP32, shape
        ``s.shape``. The scale is BF16 in storage but computed
        in FP32 here for the divide (so the scale's dynamic
        range is FP32-quality, not BF16-quality, and any tiny
        mantissa loss is acceptable on the way to a 2-byte
        scale). Only valid when ``s.mom_buf.dtype == torch.int8``.
        """
        rows, cols = s.shape[0], s.shape[1]
        m_2d = m_fp32.view(rows, cols)
        # Per-row max abs. Clamp to 1e-8 so the scale never
        # collapses to 0 (a row of exact zeros would otherwise
        # divide-by-zero in the quantize step).
        row_max = m_2d.abs().amax(dim=1).clamp(min=1e-8)
        new_scale_fp32 = (row_max / 127.0)
        # Cast to BF16 for storage; the per-row divide in the
        # dequant path promotes BF16→FP32 losslessly.
        new_scale_bf16 = new_scale_fp32.to(torch.bfloat16)
        # Quantize in FP32 for the divide (lossless); then
        # cast the int8 to int8 storage.
        scale_2d = new_scale_bf16.float().unsqueeze(1)  # [rows, 1], FP32
        q_int8 = (m_2d / scale_2d).round().clamp(-128, 127).to(torch.int8)
        s.mom_buf.copy_(q_int8.view(-1))
        s.mom_scale.copy_(new_scale_bf16)

    # ------------------------------------------------------------------ #
    # MXFP8 (E4M3 elements + per-block E8M0 scale) helpers.             #
    # ------------------------------------------------------------------ #
    # These mirror the int8 helpers above but use block-scaled
    # E8M0 + E4M3 quantization (OCP MX spec). Same per-mb
    # dequant-add-requant overhead as int8; finer within-row scale
    # granularity.
    #
    # Storage layout (per-param ``s``):
    #   mom_buf          : float8_e4m3fn,    shape [rows, cols_padded]
    #                      (cols_padded rounds up to a multiple of
    #                      ``block_size``; the tail block is all zeros
    #                      so it contributes nothing on dequant)
    #   mom_scale        : float8_e8m0fnu,   shape [rows, cols_padded // block_size]
    #   mxfp8_block_size : int (default 32)
    #
    # The ``view`` to [rows, cols_padded] is a free reshape because
    # mom_buf is contiguous 1-D.

    _E4M3_MAX = 448.0

    @staticmethod
    def _round_to_e8m0(scale_fp32: torch.Tensor) -> torch.Tensor:
        """Round FP32 scale values to the nearest E8M0-representable
        power of 2 (banker's rounding on the exponent).

        E8M0 has 8 exponent bits, no mantissa, no sign — values are
        pure powers of 2 in [2^-126, 2^127]. We use byte 1 (smallest
        representable, 2^-126) as the underflow floor so a
        block of all-zero values doesn't produce a 0-byte scale
        (E8M0 0-byte decodes to 0.0 → divide-by-zero on dequant;
        for an all-zero block the quantized values are also zero,
        so the scale value is moot, but we need a finite non-NaN
        scale for the divide).

        We bitcast via uint8 + ``.view(float8_e8m0fnu)`` (not
        ``.to(float8_e8m0fnu)``, which does value conversion).
        """
        safe = scale_fp32.float().clamp(min=2 ** -127)
        log2 = safe.log2()
        e_unclamped = log2.round() + 127.0
        e = e_unclamped.clamp(min=1.0, max=254.0)
        return e.to(torch.uint8).view(torch.float8_e8m0fnu)

    def _mxfp8_padded_cols(self, s: _ParamState) -> int:
        """Number of cols after MXFP8 right-padding to a multiple
        of ``s.mxfp8_block_size``. Used by both the dequant and
        requant paths so they agree on the layout."""
        bs = s.mxfp8_block_size
        cols = s.shape[1]
        return cols if cols % bs == 0 else cols + (bs - cols % bs)

    def _mxfp8_apply_grad(self, s: _ParamState, src_bf16: torch.Tensor) -> None:
        """Per-mb in-place accumulate ``src_bf16`` (BF16, numel)
        into ``s.mom_buf`` via the fused mxfp8 dequant+add+requant
        kernel.

        This is the per-mb hot path that replaces the prior
        "separate bf16 accum + step-end requantize" design. The
        mxfp8 storage format exists precisely to skip the
        per-mb dequant+add+requant cost on a separate accum
        buffer; we now pay the dequant+add+requant per mb
        (in the kernel) directly on the mxfp8 storage and
        save the 2 bytes/elt of CPU pinned memory for ``accum``.

        For dims where ``cols % block_size == 0`` (the production
        case for hidden_dim 1024+, intermediate 3072), the
        ``src_bf16`` length equals ``mom_buf.numel()`` exactly
        and the kernel is called directly. For dims where
        ``cols % block_size != 0``, ``mom_buf`` is sized to
        ``rows * cols_padded`` (one padded tail block) and we
        zero-pad the trailing ``rows * (cols_padded - cols)``
        elements of ``src_bf16`` before the kernel call.
        """
        rows, cols = s.shape[0], s.shape[1]
        bs = s.mxfp8_block_size
        cols_p = self._mxfp8_padded_cols(s)
        if cols_p != cols:
            # Pad the trailing block with zeros. The kernel sees
            # the same len = rows * cols_p as ``mom_buf``.
            tail = rows * (cols_p - cols)
            pad = torch.zeros(tail, dtype=torch.bfloat16,
                              device="cpu").pin_memory()
            src_padded = torch.cat(
                [src_bf16.view(rows, cols)[:, :cols].reshape(-1), pad],
            ).pin_memory()
            fused_mxfp8_dequant_add_requant(
                s.mom_buf, s.mom_scale, src_padded, rows, cols,
                block_size=bs,
            )
        else:
            fused_mxfp8_dequant_add_requant(
                s.mom_buf, s.mom_scale, src_bf16, rows, cols,
                block_size=bs,
            )

    def _dequantize_mxfp8(self, s: _ParamState) -> torch.Tensor:
        """Dequantize MXFP8 ``s.mom_buf`` (E4M3) + ``s.mom_scale``
        (E8M0) into a FP32 ``[rows, cols]`` tensor (the padded tail
        block is cropped out — it's all zeros anyway).

        Runs on the CPU. The per-block scale broadcast across the
        block's elements via ``repeat_interleave(block_size, dim=-1)``
        then we crop to the original cols.

        Only valid when ``s.mxfp8_block_size is not None``.
        """
        rows, cols = s.shape[0], s.shape[1]
        bs = s.mxfp8_block_size
        cols_p = self._mxfp8_padded_cols(s)
        n_blocks = cols_p // bs
        # E4M3 → FP32 (lossless); reshape to [rows, n_blocks, bs].
        q_blocks = s.mom_buf.view(rows, n_blocks, bs).float()
        # E8M0 → FP32 via ``.float()`` (value conversion: byte b
        # decodes to 2^(b-127)). NOT via ``.view(torch.uint8).float()``
        # — that bitcasts to uint8 first and would give the byte value
        # itself (e.g. byte 113 → 113.0), not the power-of-2 it encodes
        # (2^-14 ≈ 6.1e-5). Easy bug to make and easy to miss because
        # the values are still in a "plausible" range.
        scale_fp32 = s.mom_scale.float().view(rows, n_blocks, 1)
        # Per-element scale (broadcast across each block's bs elts).
        out = q_blocks * scale_fp32
        # Flatten and crop the padded tail (which is zero anyway).
        return out.view(rows, cols_p)[:, :cols].reshape(rows, cols)

    def _requantize_mxfp8(self, m_fp32: torch.Tensor, s: _ParamState) -> None:
        """Per-block MXFP8 quantize ``m_fp32`` (FP32 ``[rows, cols]``)
        into ``s.mom_buf`` (E4M3 ``[rows, cols_padded]``) +
        ``s.mom_scale`` (E8M0 ``[rows, cols_padded // block_size]``).

        The padded tail block is filled with zeros so it contributes
        nothing on dequant.

        Only valid when ``s.mxfp8_block_size is not None``.
        """
        rows, cols = s.shape[0], s.shape[1]
        bs = s.mxfp8_block_size
        cols_p = self._mxfp8_padded_cols(s)
        n_blocks = cols_p // bs
        # Reshape to blocks; pad cols to cols_p if needed.
        if cols_p != cols:
            m_padded = torch.nn.functional.pad(
                m_fp32.view(rows, cols), (0, cols_p - cols),
            )
        else:
            m_padded = m_fp32.view(rows, cols)
        blocks = m_padded.view(rows, n_blocks, bs)
        # Per-block absmax in FP32. E4M3 max is 448, so the per-block
        # scale is absmax / 448, then round to nearest power of 2
        # for E8M0.
        absmax = blocks.abs().amax(dim=-1).float()           # [rows, n_blocks]
        target_scale = (absmax / self._E4M3_MAX).clamp(min=2 ** -127)
        scales = self._round_to_e8m0(target_scale)            # E8M0
        # Per-element scale broadcast across each block's bs elts.
        # ``scales.float()`` decodes the E8M0 byte to its value
        # (2^(b-127)); the bitcast path would give the byte itself.
        scale_fp32 = scales.float() \
                        .view(rows, n_blocks, 1)
        scaled = (blocks.float() / scale_fp32).clamp(-self._E4M3_MAX, self._E4M3_MAX)
        # Cast FP32 → E4M3 (PyTorch does round-to-nearest-even).
        q_e4m3 = scaled.to(torch.float8_e4m3fn)
        # ``copy_`` requires shapes to match — flatten the src to
        # the destination's 1-D shape (not just same numel).
        s.mom_buf.copy_(q_e4m3.reshape(-1).view(s.mom_buf.shape))
        s.mom_scale.copy_(scales.view(s.mom_scale.shape))

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

        Per-mb grad accumulation now happens into a separate
        bf16 ``accum`` buffer (set only for quantized muon —
        int8 / mxfp8). The per-mb hot path is a cheap CPU bf16
        ``.add_()`` (~30 ms / mb at 8-layer smoke) instead of
        the old dequant-add-requant cycle (~7 s / mb). The
        expensive requantize into ``mom_buf`` / ``mom_scale``
        runs ONCE per param per step here, amortized over all
        microbatches in the cycle.

        Storage-dispatch summary (the path for each param
        depends on its configured ``muon_momentum`` dtype):

        * **int8 + BF16 per-row scale** (canonical quantized
          default): read ``s.accum`` (bf16) → H2D as FP32
          on the GPU → NS → apply update → requant ``s.accum``
          → ``s.mom_buf`` / ``s.mom_scale`` on the CPU at
          step end → reset ``s.accum``.
        * **mxfp8 + E8M0 per-block scale** (experimental): same
          as int8 but the step-end requant uses per-block E4M3
          + E8M0 (OCP MX). Pads to ``cols_padded`` for the
          tail block.
        * **Full-precision** (``fp16``/``bf16``/``fp32``, i.e.
          ``s.mom_scale is None``): merged-accumulator design
          (``mom_buf`` IS the accumulator). H2D ``mom_buf``,
          cast to FP32, NS, apply update, recast back to
          storage dtype, D2H, reset ``mom_buf``. No separate
          ``accum``.

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

        See :file:`test/_tmp/test_muon_gpu_step.py` for the
        timing comparison (76% muon step reduction at production
        scale, ~903 ms saved per step on a 1024-hidden,
        3072-intermediate model). See :file:`test/_tmp/
        test_muon_separate_accum.py` for the per-mb CPU sync
        budget regression test (would catch the pre-fix design
        where int8/mxfp8 paid a full dequant-add-requant per mb).
        """
        lr = self.lr
        wd = self.weight_decay
        CHUNK_ROWS = self._STREAM_CHUNK_ROWS
        for s in self.state.values():
            # For quantized muon: the cycle's accumulated grad
            # lives in ``s.accum`` (bf16, pinned). ``s.mom_buf``
            # is zero at the start of the cycle and gets
            # populated at step() end via
            # :func:`_quantize_accum_to_mom_buf`.
            # For fp* muon: ``mom_buf`` doubles as the
            # accumulator (no separate ``accum``).
            use_separate_accum = s.accum is not None
            g_buf = s.accum if use_separate_accum else s.mom_buf
            # Early-exit on no accumulated grad. ``s.accum`` is
            # always bf16 (no FP8 CPU reduction concerns); the
            # FP8 branch only applies to fp* muon where
            # ``s.mom_buf`` is in a quantized storage format and
            # has no CPU sum reduction kernels.
            if use_separate_accum:
                if g_buf.abs().sum().item() == 0:
                    continue
            else:
                if g_buf.dtype in (torch.float8_e4m3fn, torch.float8_e5m2,
                                    torch.float8_e8m0fnu):
                    if g_buf.to(torch.bfloat16).abs().sum().item() == 0:
                        continue
                elif g_buf.abs().sum().item() == 0:
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
            is_mxfp8 = s.mxfp8_block_size is not None
            quantized = s.mom_scale is not None  # int8 OR mxfp8
            # mxfp8 muon no longer has a separate ``accum`` —
            # the per-mb fused kernel accumulates directly into
            # ``mom_buf``. So this branch behaves like the old
            # fp* muon merged-accumulator design (read ``mom_buf``
            # directly, no requantize back at step end).
            if is_mxfp8:
                use_separate_accum = False
            # For fp* muon this is ``s.mom_buf`` (the merged
            # accumulator). ----
            if use_separate_accum:
                m_fp32_gpu = g_buf.to(device, non_blocking=True).float() \
                                            .view(rows, cols)
            else:
                # Same as the prior merged-accumulator fp* /
                # int8 / mxfp8 dequant code (untouched).
                mom_buf_gpu = g_buf.to(device, non_blocking=True)
                if is_mxfp8:
                    bs = s.mxfp8_block_size
                    cols_p = self._mxfp8_padded_cols(s)
                    scale_gpu = s.mom_scale.to(device, non_blocking=True)
                    n_blocks = cols_p // bs
                    q_blocks = mom_buf_gpu.float().view(rows, n_blocks, bs)
                    scale_fp32 = scale_gpu.float() \
                                    .view(rows, n_blocks, 1)
                    m_fp32_blocks = q_blocks * scale_fp32
                    m_fp32_gpu = m_fp32_blocks.view(rows, cols_p)[:, :cols] \
                                                        .reshape(rows, cols)
                elif quantized:
                    scale_gpu = s.mom_scale.to(device, non_blocking=True)
                    q_2d = mom_buf_gpu.float().view(rows, cols)
                    scale_2d = scale_gpu.float().unsqueeze(1)
                    m_fp32_gpu = q_2d * scale_2d
                else:
                    m_fp32_gpu = mom_buf_gpu.float().view(rows, cols)

            # ---- No SGD update here: the accumulator IS the
            # accumulated grad (mu=1). We orthogonalize it as-is. ----

            # ---- Requantize (quantized only) + D2H. For the
            # separate-accumulator path the requantize happens
            # on the CPU at step end (see _quantize_accum_to_mom_buf).
            # For mxfp8 muon (no separate accum) the cycle-sum IS
            # ``mom_buf`` already — there is no step-end requantize
            # back to ``mom_buf`` because we just consumed it for
            # NS. The cycle reset below zeros ``mom_buf`` so the
            # next cycle's per-mb fused accumulate starts fresh.
            # For fp* muon the recast happens on the GPU here
            # (same as before). ----
            if use_separate_accum:
                # Step-end requantize is done on CPU below (after
                # NS, just before the cycle reset). For now, the
                # FP32 buffer on GPU IS the NS input.
                orth_input = m_fp32_gpu
            elif is_mxfp8:
                # mxfp8 muon (no ``accum``): just take the
                # dequantized fp32 from H2D as the NS input. The
                # post-NS update is applied directly to the GPU
                # param; ``mom_buf`` is reset to zero below.
                orth_input = m_fp32_gpu
            elif quantized:
                # int8 per-row requant.
                row_max = m_fp32_gpu.abs().amax(dim=1).clamp(min=1e-8)
                new_scale_fp32 = row_max / 127.0
                new_scale_bf16 = new_scale_fp32.to(torch.bfloat16)
                new_scale_2d = new_scale_bf16.float().unsqueeze(1)
                q_int8_gpu = (m_fp32_gpu / new_scale_2d).round() \
                                .clamp(-128, 127).to(torch.int8)
                s.mom_buf.copy_(q_int8_gpu.view(-1), non_blocking=True)
                s.mom_scale.copy_(new_scale_bf16, non_blocking=True)
                orth_input = m_fp32_gpu
            else:
                # Cast FP32 back to the configured storage dtype
                # for fp* muon (the round-trip through the narrower
                # dtype is the only precision loss the full-
                # precision path incurs).
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
            # GPU so no per-chunk H2D is needed (vs the prior
            # CPU-path design which H2D'd each chunk's FP32
            # slice). For mode-3 NVFP4: the apply destination
            # is the module's ``decay_and_apply_chunk`` (material
            # BF16 view, fold the WD scalar in, subtract lr *
            # update_chunk, commit back to FP4 — no full BF16
            # master materialize). The NS run itself is identical
            # to legacy; only the apply destination differs. ----
            m_fp16 = orth_input.to(torch.float16)
            if module is None:
                target_dtype = s.param.dtype
            else:
                # Mode-3: material_chunk returns BF16; the
                # commit back to FP4 quantizes from BF16. Casting
                # the update to BF16 here keeps the math the
                # same precision as legacy (NS already runs in
                # FP16, the cast is the same either way).
                target_dtype = torch.bfloat16
            for r_start in range(0, rows, CHUNK_ROWS):
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
                    )

            # ---- Sync the device stream before next param's H2D.
            # Ensures the D2H above completed; otherwise the next
            # param could overwrite the pinned host buffer before
            # this DMA landed. ----
            torch.cuda.current_stream(device).synchronize()

            # ---- Cycle-end housekeeping ----
            if use_separate_accum:
                # Quantized muon: requantize accum (bf16) into
                # mom_buf (the configured storage format) for
                # state_dict observability + next-cycle
                # continuity. ONE requantize per param per step,
                # not one per mb.
                _quantize_accum_to_mom_buf(s)
                # Reset accum to zero for the next cycle.
                s.accum.zero_()
            else:
                # fp* muon: merged-accumulator design. ``mom_buf``
                # is the accumulator and resets to zero here.
                s.mom_buf.zero_()

    # ------------------------------------------------------------------ #
    # Checkpoint save/load (resume).                                     #
    # ------------------------------------------------------------------ #
    # See the equivalent block on :class:`CPUAdamW` for the design.
    # Both quantised (``mom_scale is not None``) and full-precision
    # storage are handled uniformly: the saved entry contains
    # whichever of ``mom_buf`` / ``mom_scale`` is non-None, and
    # ``load_state_dict`` restores them in place into the current
    # optimizer's pre-allocated buffers.
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
                    "mom_scale": s.mom_scale,
                    "mxfp8_block_size": s.mxfp8_block_size,
                    # Quantized muon only: bf16 per-cycle grad
                    # accumulator. ``None`` for fp* muon (no
                    # separate buffer). Save/restore to support
                    # mid-cycle resume (the cycle's accumulated
                    # grad persists across save/load).
                    "accum": s.accum,
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
            if sav.get("mom_scale") is not None:
                cur.mom_scale.copy_(sav["mom_scale"])
            # Restore the per-cycle bf16 accumulator (quantized
            # muon only). For fp* muon ``s.accum`` stays ``None``,
            # so the .copy_ is skipped.
            if sav.get("accum") is not None and cur.accum is not None:
                cur.accum.copy_(sav["accum"])
            # mxfp8_block_size is metadata-only (no tensor); keep
            # the current optimizer's value if not in the file
            # (e.g. an older int8 checkpoint loaded into a freshly-
            # constructed mxfp8 optimizer would mismatch — caught
            # by the explicit check below).
            file_bs = sav.get("mxfp8_block_size")
            if file_bs is not None and cur.mxfp8_block_size is not None \
                    and file_bs != cur.mxfp8_block_size:
                raise ValueError(
                    f"CPUMuon.load_state_dict: mxfp8 block_size"
                    f" mismatch at param index {current.index(cur)}:"
                    f" file={file_bs} current={cur.mxfp8_block_size}."
                )