"""Per-parameter optimizers with CPU offload for HippoLM.

Implements two optimizer variants used together via param groups:

  - :class:`CPUAdamW`     — AdamW with BF16 m and BF16 v state on
    CPU pinned memory. The forward / backward happen on the GPU
    in the param's training dtype (FP16 in this project). After
    each micro-batch's backward, :func:`accumulate_grads_to_cpu`
    copies the GPU ``.grad`` to a per-param CPU accumulator
    (BF16, pinned) and frees the GPU copy. On ``step()`` the
    accumulator is consumed: m, v are updated in place on the
    CPU (BF16); the per-element update factor is computed in
    FP32 (to recover precision after the BF16 v sqrt), then
    cast to FP16 and streamed chunk-wise to the GPU param and
    applied in place. This is the DeepSpeed ZeRO-Offload
    pattern with BF16 state.
    Grad dtype is BF16 throughout (matches the new 5060Ti
    hardware; BF16 has FP32-like dynamic range, so no overflow
    on the grad transfer or the m update, and v can stay BF16
    because BF16's 8-bit exponent is wide enough to keep
    ``v = g²`` from underflowing at typical grad magnitudes
    1e-4 to 1e-3 — 1e-8 is well above BF16's smallest normal
    of ~1.18e-38).

  - :class:`CPUMuon`      — Muon (Newton-Schulz orthogonalization
    of the momentum matrix). Momentum storage is configurable
    via ``precision.muon_momentum``:

      * ``int8`` (default) + **BF16 per-row scale**: 1 byte/elt
        momentum + a tiny ``rows``-sized BF16 scale tensor.
        The dequantize + SGD update + requantize cycle runs on
        the CPU (one full pass per param per step; the NS
        iteration is the only GPU step and is streamed over
        rows in 4096-row chunks).
      * ``bf16`` / ``fp16`` / ``fp32``: full-precision
        momentum, no quantization. Doubles / quadruples the
        Muon CPU RAM vs int8 (no scale tensor needed).
        The NS iteration is unaffected (it orthogonalizes the
        dequantized / raw momentum in FP32 on the GPU
        regardless of storage precision).

Both optimizers follow the same API surface as ``torch.optim.Optimizer``
minimally — ``step()`` consumes whatever gradients are present on
the parameters and ``zero_grad()`` clears them.

No gradient is stored on the GPU between steps. The training loop
copies each micro-batch's ``.grad`` to CPU and adds it to the
optimizer state's accumulator; only after ``gradient_accumulation_steps``
micro-batches does ``step()`` actually run.

Conventions
-----------
- Param ``.data`` lives on the GPU (FP16).
- ``.grad`` is materialised on the GPU only during ``backward()``;
  immediately after, the training loop calls
  :func:`accumulate_grads_to_cpu` which copies ``.grad`` to the
  per-param CPU accumulator and frees the GPU copy. So between
  micro-batches the GPU holds zero gradient memory.
- Optimizer state on CPU pinned memory:
    AdamW: m (BF16, numel) + v (BF16, numel) + accum (BF16, numel)
    Muon:  mom_buf (cfg.dtype, numel)
           + mom_scale (BF16, rows)  [int8 only; None for fp*]
           + accum (BF16, numel)
- For ~624 M params: AdamW ≈ 6 × 624 M = 3.7 GB CPU RAM
  (m BF16 + v BF16 + accum BF16, all 2 bytes/elt).
  Muon: 0.6 GB (int8), 1.2 GB (fp16/bf16), 2.5 GB (fp32).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Protocol, Tuple

import torch
import torch.nn as nn

from .precision_config import DType, PrecisionConfig, TensorPrecision


# --------------------------------------------------------------------------- #
# Per-param state descriptor.                                                 #
# --------------------------------------------------------------------------- #
@dataclass
class _ParamState:
    """Per-parameter CPU-side state for either AdamW or Muon."""

    param: nn.Parameter
    # The CPU accumulator (BF16). For both AdamW and Muon this is
    # the destination of :func:`accumulate_grads_to_cpu`: each
    # micro-batch's ``.grad`` (cast to BF16 before the DMA) is
    # added in here on the CPU, and the training loop's
    # :func:`zero_cpu_grad_accum` zeros it out at the end of the
    # accumulation cycle.
    accum: torch.Tensor
    # AdamW-specific state (set when ``kind == 'adamw'``).
    # m and v are both BF16: the new 5060Ti hardware supports
    # BF16 natively and BF16's 8-bit exponent keeps ``v = g²``
    # from underflowing for typical grad magnitudes. (The
    # previous FP32 v was a holdover from the FP16 era, where
    # FP16's 5-bit exponent would underflow ``g²`` to 0 for
    # grad magnitudes ~1e-4 to 1e-3 — that was the original
    # justification. BF16 doesn't have this problem.)
    m: torch.Tensor | None = None
    exp_avg_sq: torch.Tensor | None = None
    # Muon-specific state (set when ``kind == 'muon'``). The
    # momentum matrix ``m`` lives in ``mom_buf`` (any storage
    # dtype: int8 with a BF16 per-row scale, or a raw fp16/bf16/
    # fp32 tensor with no scale). Field names are dtype-agnostic
    # so the ``step()`` and ``loop.py`` byte-breakdown code
    # works regardless of the configured storage dtype.
    mom_buf: torch.Tensor | None = None
    mom_scale: torch.Tensor | None = None
    # Cached for Muon: original param shape for reshape on GPU.
    shape: tuple = field(default_factory=tuple)
    # Step counter (per param; cheap).
    step: int = 0
    # "adamw" or "muon".
    kind: str = "adamw"


# --------------------------------------------------------------------------- #
# Public protocol for per-param state.                                        #
# --------------------------------------------------------------------------- #
class OptimizerState(Protocol):
    """Structural view of a per-param optimizer state entry.

    The training diagnostics (see :mod:`src.training.diagnostics`)
    only need to read a handful of fields off each per-param
    state, but they need to do so uniformly across the two
    optimizers (AdamW and Muon) that the training loop
    instantiates. This :class:`Protocol` declares the minimum
    shape any optimizer state entry must expose to be
    diagnosable.

    Implementations: :class:`_ParamState` (the only concrete
    state in this module, used by both :class:`CPUAdamW` and
    :class:`CPUMuon`). ``CPUAdamW.state`` and ``CPUMuon.state``
    are ``Dict[int, _ParamState]`` and therefore structurally
    satisfy this protocol.

    Field semantics:

    - ``param`` is the GPU-side trainable parameter the entry
      belongs to. Used to read ``.data`` (post-step) for the
      "is the param finite" check.
    - ``accum`` is the CPU pinned BF16 grad accumulator
      (destination of :func:`accumulate_grads_to_cpu`).
    - ``kind`` is ``"adamw"`` or ``"muon"``; diagnostics branch
      on it to pick which state tensors to read.
    - ``m`` / ``exp_avg_sq`` are AdamW's first / second moment
      (both BF16 CPU pinned). For Muon both are ``None``.
    - ``mom_buf`` / ``mom_scale`` are Muon's momentum storage.
      ``mom_buf`` is the EMA buffer at the configured storage
      dtype (int8 with per-row symmetric quantization + BF16
      ``mom_scale``, OR a raw fp16/bf16/fp32 tensor with
      ``mom_scale=None``). For AdamW both are ``None``.
    - ``shape`` is cached for Muon (the original param shape
      before flatten) so diagnostics can report the failing
      param's full shape on NaN/Inf.
    - ``step`` is the per-param step counter; primarily for
      warmup-aware step reporting.
    """

    param: nn.Parameter
    accum: torch.Tensor
    kind: str
    m: Optional[torch.Tensor]
    exp_avg_sq: Optional[torch.Tensor]
    mom_buf: Optional[torch.Tensor]
    mom_scale: Optional[torch.Tensor]
    shape: tuple
    step: int


# --------------------------------------------------------------------------- #
# CPU AdamW                                                                   #
# --------------------------------------------------------------------------- #
class CPUAdamW:
    """AdamW with CPU-side m (BF16) and v (BF16) state, GPU-side
    params.

    Memory layout per trainable param:
        CPU pinned: m (BF16, numel), v (BF16, numel), accum (BF16, numel)
        GPU:         param (training dtype, e.g. FP16), .grad (transient)

    On :meth:`step`:
        1. Read the latest accumulated grad (CPU pinned, BF16).
        2. Update m, v in place (BF16 / BF16).
        3. Compute the update factor in FP32 (to recover the
           mantissa precision lost to the BF16 v sqrt), then
           cast to FP16 (chunked, streamed to the GPU).
        4. Apply ``p -= lr * factor`` in place on the GPU.

    The training loop is expected to call
    :func:`accumulate_grads_to_cpu` after each micro-batch's
    backward() to copy ``.grad`` into ``accum`` and free the GPU
    copy. :meth:`step` then sees the accumulated grad on CPU.
    """

    def __init__(
        self,
        params: List[nn.Parameter],
        lr: float = 1e-4,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        precision: Optional[PrecisionConfig] = None,
    ) -> None:
        """CPU-offloaded AdamW with configurable per-tensor precision.

        ``precision`` controls the storage dtypes of:

        - ``s.accum``   (the grad accumulator)  ← ``precision.gradients``
        - ``s.m``       (1st moment)            ← ``precision.adamw_m``
        - ``s.exp_avg_sq`` (2nd moment)         ← ``precision.adamw_v``

        Defaults (when ``precision`` is ``None``) match the canonical
        yml: BF16 for all three. The :meth:`step` algorithm is
        dtype-agnostic — it always promotes to FP32 for the
        divisor / factor math — so any combination of these
        dtypes produces the same numerical answer up to casting
        error.
        """
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        # If no precision passed, use the dataclass defaults (which
        # already match the canonical yml).
        precision = precision or PrecisionConfig()

        # Resolve dtypes once (avoids per-param .to_torch() calls).
        accum_dtype = precision.gradients.dtype.to_torch()
        m_dtype = precision.adamw_m.dtype.to_torch()
        v_dtype = precision.adamw_v.dtype.to_torch()
        # Quantized m / v are not supported (AdamW's m and v are
        # magnitudes / squared magnitudes; quantizing them loses
        # the precision that the FP32 promotion in step() is
        # supposed to recover). Fall back to FP32 storage and
        # warn — explicit user override of an unsupported
        # combination should not silently change the math.
        if precision.adamw_m.dtype.is_integer:
            m_dtype = torch.float32
            import warnings
            warnings.warn(
                f"CPUAdamW: adamw_m dtype={precision.adamw_m.dtype.value}"
                f" is integer-quantized; falling back to FP32 storage."
                f" Quantizing m/v is not supported (the FP32"
                f" promotion in step() assumes a floating source).",
                stacklevel=2,
            )
        if precision.adamw_v.dtype.is_integer:
            v_dtype = torch.float32
            import warnings
            warnings.warn(
                f"CPUAdamW: adamw_v dtype={precision.adamw_v.dtype.value}"
                f" is integer-quantized; falling back to FP32 storage.",
                stacklevel=2,
            )

        self.state: dict[int, _ParamState] = {}
        seen: set[int] = set()
        for p in params:
            if not p.requires_grad:
                continue
            if id(p) in seen:
                continue
            seen.add(id(p))
            n = p.numel()
            self.state[id(p)] = _ParamState(
                param=p,
                accum=torch.zeros(n, dtype=accum_dtype, device="cpu").pin_memory(),
                exp_avg_sq=torch.zeros(n, dtype=v_dtype, device="cpu").pin_memory(),
                m=torch.zeros(n, dtype=m_dtype, device="cpu").pin_memory(),
                kind="adamw",
                shape=p.shape,
            )

    def zero_grad(self, set_to_none: bool = True) -> None:
        for s in self.state.values():
            s.param.grad = None
            # NOTE: do NOT zero accum here — that would discard
            # the user's gradient accumulation across micro-batches.
            # The training loop is responsible for resetting accum
            # at the start of each accumulation cycle (via
            # :func:`zero_cpu_grad_accum`).

    # Per-chunk size for the streaming factor / update path.
    # 4 M elements is the sweet spot on V100 / 5060Ti PCIe: 8 MB
    # FP16 per chunk keeps the GPU transient well under the 16 GB
    # ceiling and overlaps the H2D copy with the next chunk's
    # compute.
    _STREAM_CHUNK_NUMEL = 4 * 1024 * 1024

    def step(self) -> None:
        beta1, beta2 = self.beta1, self.beta2
        lr = self.lr
        eps = self.eps
        wd = self.weight_decay
        CHUNK = self._STREAM_CHUNK_NUMEL
        for s in self.state.values():
            if s.accum.abs().sum().item() == 0:
                # No accumulated grad (shouldn't happen if the
                # training loop is correct, but guard against
                # unnecessary work).
                continue
            s.step += 1
            step = s.step
            bc1 = 1.0 - beta1 ** step
            bc2 = 1.0 - beta2 ** step
            g = s.accum
            m = s.m
            v = s.exp_avg_sq
            # Update m (BF16, in place) and v (BF16, in place).
            # The dedicated ``m`` buffer keeps ``m`` and ``g``
            # (which is ``s.accum``) aliased-free: the chained
            # ``m.mul_(beta1).add_(g, ...)`` would otherwise
            # mutate ``g`` mid-chain and produce the wrong
            # update on every step.
            #
            # ``addcmul_`` on BF16: BF16 has FP32 exponent range
            # so ``g*g`` (for g ~ 1e-3) is ~1e-6, well above the
            # BF16 smallest normal. The mantissa is 7 bits which
            # is the precision bottleneck — but we recover
            # precision in the per-element factor by promoting
            # to FP32 below before the sqrt.
            m.mul_(beta1).add_(g, alpha=1.0 - beta1)
            v.mul_(beta2).addcmul_(g, g, value=1.0 - beta2)
            # Decoupled weight decay: applied in place on the GPU
            # once, before the streaming factor / apply loop.
            # Folding it into the per-element factor would also
            # work but would require an extra H2D per chunk.
            if wd != 0.0:
                s.param.data.mul_(1.0 - lr * wd)
            # Streaming update: chunk the param so the peak GPU
            # memory for the FP32 divisor / FP16 factor transfer
            # is bounded to ``CHUNK`` elements regardless of
            # ``numel``. The chunked path is bit-identical to
            # the un-chunked path because the operations are
            # element-wise.
            #
            # Numerics: do the divisor math in FP32 even though
            # ``g``, ``m``, and ``v`` are all BF16. Promoting
            # v_chunk to FP32 before the sqrt recovers the
            # 7-bit BF16 mantissa precision for the divisor,
            # which is what ``1/sqrt(v/bc2)`` is most sensitive
            # to (a small relative error in v becomes a large
            # relative error in 1/sqrt(v)). The factor is then
            # cast back to FP16 for the GPU apply — that
            # final cast is the precision bottleneck, not the
            # v promotion.
            p_flat = s.param.data.view(-1)
            n = p_flat.numel()
            for start in range(0, n, CHUNK):
                end = min(start + CHUNK, n)
                v_chunk_fp32 = v[start:end].float()
                denom = (v_chunk_fp32 / bc2).sqrt_().add_(eps)  # FP32, chunk
                factor = (m[start:end].float() / bc1) / denom   # FP32, chunk
                # Stream the FP32 factor to the GPU as the param
                # dtype (FP16) and apply in place.
                p_flat[start:end].add_(
                    factor.to(
                        device=s.param.device, dtype=s.param.dtype,
                        non_blocking=True,
                    ),
                    alpha=-lr,
                )
            # Reset accum to zero for the next accumulation cycle.
            s.accum.zero_()


# --------------------------------------------------------------------------- #
# CPU Muon                                                                    #
# --------------------------------------------------------------------------- #
class CPUMuon:
    """Muon with configurable momentum storage on CPU pinned memory.

    State per param:
        CPU pinned: mom_buf (cfg dtype, numel),
                    mom_scale (BF16, rows or None),
                    accum (BF16, numel)
        GPU:         param (FP16), .grad (transient)

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

        - ``s.accum``     (grad accumulator)     ← ``precision.gradients``
        - ``s.mom_buf``   (momentum storage)     ← ``precision.muon_momentum``
          (int8 with per-row quantization, or raw fp16/bf16/fp32)
        - ``s.mom_scale`` (per-row scale)        ← only set for int8
          storage (BF16 — the scale needs FP32-like dynamic range
          to avoid underflow on quiet rows; BF16 is the cheapest
          dtype that guarantees this). ``None`` for fp* storage.

        Supported momentum dtypes: ``int8`` (default),
        ``bf16`` / ``fp16`` / ``fp32`` (full-precision storage,
        no quantization). ``int4`` raises :class:`NotImplementedError`
        (the packing scheme is on the roadmap but not landed).
        """
        self.lr = lr
        self.momentum = momentum
        self.nesterov = nesterov
        self.ns_steps = ns_steps
        self.weight_decay = weight_decay
        precision = precision or PrecisionConfig()

        accum_dtype = precision.gradients.dtype.to_torch()
        # Resolve momentum storage. int8 is the only fully-supported
        # quantized storage; int4 is on the roadmap but not yet
        # implemented (the mom_buf would need to pack 2 elements
        # per byte and the dequant path would need to unpack).
        mom_dtype_cfg = precision.muon_momentum.dtype
        if mom_dtype_cfg == DType.INT4:
            raise NotImplementedError(
                "CPUMuon: muon_momentum dtype=int4 is not yet implemented."
                " Use int8 (the current default) or a floating dtype"
                " (bf16 / fp16 / fp32)."
            )
        # int8 → 1 byte/elt momentum + a BF16 per-row scale.
        # fp*  → full-precision storage at the configured dtype;
        #         no scale, no dequant/requant in step().
        if mom_dtype_cfg.is_integer:
            mom_storage_dtype = mom_dtype_cfg.to_torch()  # int8
            scale_dtype = torch.bfloat16
        else:
            mom_storage_dtype = mom_dtype_cfg.to_torch()  # bf16/fp16/fp32
            scale_dtype = None  # no scale for full-precision storage

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
            mom_scale = (
                torch.zeros(shape[0], dtype=scale_dtype, device="cpu").pin_memory()
                if scale_dtype is not None else None
            )
            st = _ParamState(
                param=p,
                accum=torch.zeros(n, dtype=accum_dtype, device="cpu").pin_memory(),
                mom_buf=torch.zeros(n, dtype=mom_storage_dtype, device="cpu").pin_memory(),
                mom_scale=mom_scale,
                kind="muon",
                shape=shape,
            )
            self.state[id(p)] = st

    def zero_grad(self, set_to_none: bool = True) -> None:
        for s in self.state.values():
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

    def _newton_schulz(self, x: torch.Tensor) -> torch.Tensor:
        a, b, c = self._NS_COEFFS
        # Cast to FP16 for tensor-core matmul speed. The
        # orthogonalization is robust to FP16 noise for typical
        # gradient magnitudes.
        x = x.to(torch.float16)
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

        Two storage paths share the same scaffolding (H2D state,
        SGD in FP32, D2H new state, then stream NS over rows):

        * **Quantized** (``s.mom_scale is not None``, i.e.
          ``int8`` + BF16 scale): dequant to FP32 on the GPU,
          SGD in FP32, requant to int8 + BF16 scale, D2H the
          new storage.
        * **Full-precision** (``s.mom_scale is None``, i.e.
          ``fp16``/``bf16``/``fp32``): H2D the raw momentum,
          SGD in FP32, cast back to the storage dtype, D2H.
          No dequant / requant.

        Both paths run the EMA + NS work on the GPU HBM
        (bandwidth-bound, ~50x faster than CPU DRAM). State
        stays on CPU pinned memory; the per-mb VRAM peak is one
        param's worth of buffers (~21 MB for a 1024×3072
        down_proj) regardless of storage dtype.

        See :file:`test/_tmp/test_muon_gpu_step.py` for the
        timing comparison (76% muon step reduction at production
        scale, ~903 ms saved per step on a 1024-hidden,
        3072-intermediate model).
        """
        mom = self.momentum
        lr = self.lr
        wd = self.weight_decay
        nesterov = self.nesterov
        CHUNK_ROWS = self._STREAM_CHUNK_ROWS
        for s in self.state.values():
            if s.accum.abs().sum().item() == 0:
                continue
            shape = s.shape
            rows, cols = shape[0], shape[1]
            device = s.param.device
            quantized = s.mom_scale is not None

            # ---- H2D state to GPU (async; pinned host memory →
            # real async DMA with non_blocking=True) ----
            mom_buf_gpu = s.mom_buf.to(device, non_blocking=True)
            accum_gpu = s.accum.to(device, non_blocking=True)

            # ---- Get the SGD input as FP32 [rows, cols] on GPU.
            # Quantized path: dequant int8 with the BF16 per-row
            # scale. Full-precision path: just cast the storage
            # dtype to FP32 (lossless for fp32; bf16/fp16 widen
            # to FP32 in one instruction). ----
            if quantized:
                scale_gpu = s.mom_scale.to(device, non_blocking=True)
                q_2d = mom_buf_gpu.float().view(rows, cols)
                scale_2d = scale_gpu.float().unsqueeze(1)     # [rows, 1]
                m_fp32_gpu = q_2d * scale_2d                 # FP32 [rows, cols]
            else:
                m_fp32_gpu = mom_buf_gpu.float().view(rows, cols)

            # ---- GPU SGD: m = beta*m + (1-beta)*grad ----
            # accum is BF16; cast to FP32 for the add (matches
            # the CPU path's behavior; BF16 add would lose
            # precision on the EMA).
            g_2d = accum_gpu.float().view(rows, cols)
            m_fp32_gpu.mul_(mom).add_(g_2d, alpha=1.0 - mom)

            # ---- Requant (quantized) or recast (full-precision)
            # + D2H the new momentum storage. ----
            if quantized:
                row_max = m_fp32_gpu.abs().amax(dim=1).clamp(min=1e-8)
                new_scale_fp32 = row_max / 127.0
                new_scale_bf16 = new_scale_fp32.to(torch.bfloat16)
                new_scale_2d = new_scale_bf16.float().unsqueeze(1)
                q_int8_gpu = (m_fp32_gpu / new_scale_2d).round() \
                                .clamp(-128, 127).to(torch.int8)
                # q_int8_gpu is 2D; flatten back to 1D to match
                # s.mom_buf's storage.
                s.mom_buf.copy_(q_int8_gpu.view(-1), non_blocking=True)
                s.mom_scale.copy_(new_scale_bf16, non_blocking=True)
            else:
                # Cast FP32 back to the configured storage dtype
                # (the SGD is in FP32; the round-trip through the
                # narrower dtype is the only precision loss the
                # full-precision path incurs, vs the quantized
                # path's per-row int8 quantization).
                new_mom = m_fp32_gpu.to(s.mom_buf.dtype).view(-1)
                s.mom_buf.copy_(new_mom, non_blocking=True)

            # ---- Decoupled weight decay: applied once to the
            # full GPU param (in place), before the streaming
            # NS apply loop. ----
            if wd != 0.0:
                s.param.data.mul_(1.0 - lr * wd)

            # ---- Stream NS over rows; m_fp32_gpu is already on
            # GPU so no per-chunk H2D is needed (vs the prior
            # CPU-path design which H2D'd each chunk's FP32
            # slice). ----
            m_fp16 = m_fp32_gpu.to(torch.float16)
            for r_start in range(0, rows, CHUNK_ROWS):
                r_end = min(r_start + CHUNK_ROWS, rows)
                m_chunk = m_fp16[r_start:r_end]
                update = self._newton_schulz(m_chunk)
                update = update.to(s.param.dtype)
                s.param.data[r_start:r_end].add_(update, alpha=-lr)

            # ---- Sync the device stream before next param's H2D.
            # Ensures the D2H above completed; otherwise the next
            # param could overwrite the pinned host buffer before
            # this DMA landed. ----
            torch.cuda.current_stream(device).synchronize()

            # Reset grad accumulator for next accumulation cycle.
            s.accum.zero_()


# --------------------------------------------------------------------------- #
# Public API: stream grads to CPU and reset                                   #
# --------------------------------------------------------------------------- #
def accumulate_grads_to_cpu(
    optimizers: List,
    sync_device: int | None = None,
) -> None:
    """Manual-path copy of each trainable param's ``.grad`` to its
    optimizer state's CPU accumulator and free the GPU copy.

    This is the post-backward batched path. It is kept for the
    manual code paths (unit tests that set ``.grad`` directly,
    or any caller that has not installed the per-param streaming
    hooks via :func:`register_grad_offload_hooks`). The training
    loop uses the streaming path.

    The async ``.to("cpu")`` is issued first (with a flattened
    view of the grad, so the resulting ``src_cpu`` matches the
    1-D ``accum`` tensor), then we sync the current CUDA device
    (or ``sync_device`` if given) before performing the in-place
    CPU add. This avoids the race where ``s.accum.add_(src)`` runs
    on the CPU before the DMA copy populates ``src``.

    The cast target is read from ``s.accum.dtype`` (the
    ``gradients`` precision in :class:`PrecisionConfig`). BF16 is
    the default; FP16 / FP32 also work, and integer dtypes
    (int8 / int4) cast the rounded-integer grad into the
    accumulator's storage — but in practice the training loop
    uses a floating dtype for ``gradients`` because the
    grad-to-accumulator add is an FP add, not an int add.

    The cast happens BEFORE the DMA so the transfer itself
    matches the storage dtype. We do the cast on the GPU first,
    then the DMA, so the PCIe transfer is already in the
    target dtype.
    """
    pending: list[tuple[torch.Tensor, torch.Tensor]] = []
    for opt in optimizers:
        for s in opt.state.values():
            g = s.param.grad
            if g is None:
                continue
            # Cast on the GPU first (matches s.accum's dtype;
            # no FP16 overflow on the DMA if accum is BF16), then
            # flatten, then DMA.
            src_cpu = (
                g.detach()
                .to(s.accum.dtype)
                .reshape(-1)
                .to("cpu", non_blocking=True)
            )
            pending.append((s.accum, src_cpu))
            # Free GPU grad immediately so the GPU memory is
            # available for the next micro-batch's forward.
            s.param.grad = None

    if sync_device is not None:
        torch.cuda.synchronize(sync_device)
    else:
        torch.cuda.synchronize()

    for target, src in pending:
        target.add_(src)


# --------------------------------------------------------------------------- #
# Streaming per-param D2H via post-accumulate-grad hooks.                      #
# --------------------------------------------------------------------------- #
# Module-level queue for in-flight D2H transfers issued by the
# hooks. Each entry is ``(accum, src_cpu)`` where ``src_cpu`` is
# a CPU tensor whose data is being DMA'd in asynchronously.
# :func:`flush_pending_grads` syncs CUDA and applies the adds.
_pending_grads: list[tuple[torch.Tensor, torch.Tensor]] = []

# Module-level set of param ``id(p)`` for params that need a
# *post-backward* manual flush instead of (or in addition to) the
# per-param streaming hook. The only param in this category for
# HippoLM is the tied ``embed_tokens.weight``: the FusedLinearCE
# grad arrives via a custom autograd Function that fires AFTER
# the natural-path ``accumulate_grad_`` (which is what triggers
# the per-param hook). If we cleared ``p.grad`` in the hook, the
# manual ``add_`` from the custom autograd would lazy-create a
# separate grad tensor, splitting the embed's grad across GPU +
# CPU. Skipping the hook for these params and reading
# ``p.grad`` once at ``flush_pending_grads`` time sees the
# final (post-add_) grad and copies the whole thing.
_manual_flush_param_ids: set[int] = set()


def register_grad_offload_hooks(
    optimizers: List,
    manual_flush_params: Optional[List[nn.Parameter]] = None,
) -> None:
    """Install a per-param post-accumulate-grad hook that streams
    each param's grad to the CPU as it is computed.

    Why hooks (and not a post-backward loop):

      The post-backward path (``accumulate_grads_to_cpu``)
      iterates over the optimizer state after ``loss.backward()``
      has returned. By that point autograd has already populated
      ``.grad`` for every trainable param on the GPU. The peak
      GPU memory is therefore ``model + activations + sum(grads)``
      at the instant backward finishes, before the function has
      a chance to issue the first DMA. For a 624 M-param model
      in FP16, ``sum(grads) ≈ 1.2 GB`` — the user's report.

      A post-accumulate-grad hook fires as autograd propagates
      the grad to a specific param, BEFORE the next param's
      backward runs. The hook can issue the DMA and clear the
      GPU grad immediately, so the peak GPU grad memory is
      bounded to "at most one param's grad" (the one autograd is
      currently propagating to) plus the cast-buffer transient.

    The hook closes over the per-param ``_ParamState`` so it
    can find the right CPU accumulator. Each hook returns
    ``None`` (no replacement grad) and also explicitly sets
    ``p.grad = None`` as a belt-and-suspenders — the engine's
    "return None leaves .grad as-is" semantics mean our explicit
    clear is the source of truth.

    The cast target is ``s.accum.dtype`` (the
    ``gradients`` precision in :class:`PrecisionConfig`). BF16 is
    the default and matches the PCIe bandwidth: BF16 and FP16
    are both 2 bytes/elt so the DMA size is the same, but the
    BF16 accumulator matches the BF16 m/v storage in the
    AdamW path so no further cast is needed at step() time.

    ``manual_flush_params`` (optional): list of params whose
    grad arrives via a custom autograd Function that fires
    AFTER the natural-path ``accumulate_grad_`` (e.g. tied
    embeddings whose LCE grad is added by a custom Function in
    the model's backward). For these params the per-param hook
    is **skipped** (would clear the in-progress grad), and the
    manual ``accumulate_grads_to_cpu`` path runs once per
    microbatch in :func:`flush_pending_grads`. See the comment
    on ``_manual_flush_param_ids`` for the gory details.

    Must be called AFTER ``build_param_groups`` (so the per-param
    state exists) and BEFORE the first ``backward()``. Typically
    called once per worker in :func:`_setup_worker`.
    """
    _manual_flush_param_ids.clear()
    if manual_flush_params:
        _manual_flush_param_ids.update(id(p) for p in manual_flush_params)
    seen: set[int] = set()
    for opt in optimizers:
        for s in opt.state.values():
            p = s.param
            if id(p) in seen:
                continue
            seen.add(id(p))
            if id(p) in _manual_flush_param_ids:
                # Skip the per-param hook; flush_pending_grads
                # handles this param via the manual path.
                continue
            p.register_post_accumulate_grad_hook(_make_offload_hook(s))
    # Defensive: clear any stale entries (e.g. from a previous
    # run sharing the module-level queue).
    _pending_grads.clear()


def _make_offload_hook(s: _ParamState):
    """Build the post-accumulate-grad hook for one param. Closes
    over ``s`` so the hook can find the CPU accumulator and the
    target dtype without a per-call dict lookup.

    PyTorch contract reminder: per the
    :meth:`torch.Tensor.register_post_accumulate_grad_hook`
    docstring, the argument passed to the hook is the **leaf
    tensor** (``param``), NOT the gradient. The gradient lives
    on ``param.grad`` at the moment the hook fires. Using ``g``
    as if it were the grad (the previous behaviour in this
    file) silently streamed the param values to CPU every
    microbatch — the root cause of a multi-day "loss not
    decreasing" incident. Always read ``p.grad`` here.
    """
    p = s.param
    accum = s.accum
    target_dtype = accum.dtype

    def hook(g: torch.Tensor | None) -> None:
        # ``g`` is the leaf param per the PyTorch API contract.
        # We need the grad, which is on ``p.grad`` at this
        # point. (See the docstring above for the why.)
        grad = p.grad
        if grad is None:
            # The param had no grad this backward (e.g. it was
            # in a no-grad branch). Nothing to offload.
            return None
        # Cast on the GPU (so the DMA carries the accum dtype,
        # not the param's training dtype), flatten to match the
        # 1-D ``accum`` shape, then issue the async D2H. The
        # .detach() prevents the resulting CPU tensor from
        # carrying autograd metadata (we never want to backprop
        # through a grad-DMA).
        src = (
            grad.detach()
            .to(target_dtype)
            .reshape(-1)
            .to("cpu", non_blocking=True)
        )
        _pending_grads.append((accum, src))
        # Free the GPU grad right now so the memory is available
        # for the next param's backward. The hook return value
        # replaces .grad per PyTorch's contract; returning None
        # means "don't replace", so our explicit clear is what
        # actually empties .grad.
        p.grad = None
        return None

    return hook


def flush_manual_flush_params(optimizers: List) -> None:
    """Copy the GPU grad of every ``manual_flush`` param to its
    CPU accumulator and free the GPU copy.

    Intended to be called by the training loop AFTER the full
    microbatch ``backward()`` has returned, alongside
    :func:`flush_pending_grads`. By the time this runs, any
    custom-autograd manual ``add_`` (e.g. the FusedLinearCE dw
    scatter in :class:`_TiedFusedLCEFunction`) has already
    landed in ``p.grad``, so the value we read here is the
    *complete* microbatch grad.

    This is the per-microbatch ``accumulate_grads_to_cpu`` path
    scoped to just the params in
    :data:`_manual_flush_param_ids` — all the other params go
    through the streaming post-accumulate-grad hook instead.

    Idempotent within a single microbatch: calling it twice
    before the next backward would no-op the second time
    (the GPU grad is cleared after the first call).
    """
    if not _manual_flush_param_ids:
        return
    pending: list[tuple[torch.Tensor, torch.Tensor]] = []
    seen: set[int] = set()
    for opt in optimizers:
        for s in opt.state.values():
            p = s.param
            if id(p) in seen:
                continue
            seen.add(id(p))
            if id(p) not in _manual_flush_param_ids:
                continue
            g = p.grad
            if g is None:
                continue
            src_cpu = (
                g.detach()
                .to(s.accum.dtype)
                .reshape(-1)
                .to("cpu", non_blocking=True)
            )
            pending.append((s.accum, src_cpu))
            p.grad = None
    if not pending:
        return
    torch.cuda.synchronize()
    for target, src in pending:
        target.add_(src)


def flush_pending_grads(sync_device: int | None = None) -> None:
    """Sync CUDA and apply the CPU-side adds for any D2H transfers
    issued by the post-accumulate-grad hooks since the last flush.

    The training loop calls this after each microbatch's
    ``backward()`` and before the next forward pass. The
    accumulate-on-CPU side is commutative and idempotent within
    a single microbatch, so the order of pending entries does
    not matter.

    Idempotent and safe to call when no DMAs are pending (no
    synchronize, no work). Safe to call multiple times per
    microbatch, though the training loop calls it once.

    NOTE: this function only flushes the **per-param-hook**
    queue. Params whose per-param hook was deliberately skipped
    (see :data:`_manual_flush_param_ids` — typically the tied
    embed whose grad arrives via a custom autograd Function
    that fires after the natural-path accumulate_grad_) need
    :func:`flush_manual_flush_params` called separately, after
    this function.

    For the manual-only path (no hooks installed at all — e.g.
    unit tests that set ``.grad`` directly), use
    :func:`accumulate_grads_to_cpu` instead.
    """
    if not _pending_grads:
        return
    if sync_device is not None:
        torch.cuda.synchronize(sync_device)
    else:
        torch.cuda.synchronize()
    # apply_ in place; order doesn't matter for the add.
    for target, src in _pending_grads:
        target.add_(src)
    _pending_grads.clear()


def zero_cpu_grad_accum(optimizers: List) -> None:
    """Reset CPU accumulators (call this AFTER step() to start the
    next accumulation cycle).
    """
    for opt in optimizers:
        for s in opt.state.values():
            s.accum.zero_()


def build_param_groups(
    model: nn.Module,
    device: int,
    lr_muon: float = 1e-3,
    lr_adamw: float = 1e-4,
    weight_decay: float = 0.01,
    muon_momentum: float = 0.95,
    precision: Optional[PrecisionConfig] = None,
) -> Tuple[List, List]:
    """Build a (muon_optimizer, adamw_optimizer) pair for a single
    rank's model fragment.

    Standard Muon routing:
        - 1D params (RMSNorm.weight, BlockAttnRes.query,
          KDA A_log/dt_bias with ``_no_weight_decay``): AdamW
        - 2D Linear weights: Muon
        - Embedding + lm_head: AdamW (per user spec)

    ``precision`` (optional :class:`PrecisionConfig`) is forwarded
    to both optimizers; the model weights themselves are
    constructed at the dtype from ``precision.model_weights`` by
    the caller (``scripts.train`` / ``src.training.loop``).
    Defaults to the canonical yml precision when ``None``.
    """
    muon_params: list[nn.Parameter] = []
    adamw_params: list[nn.Parameter] = []
    seen: set[int] = set()
    for name, p in model.named_parameters_per_device(device):
        if not p.requires_grad or id(p) in seen:
            continue
        seen.add(id(p))
        # Per user spec: lm_head and embed_tokens -> AdamW.
        # The AttnRes pseudo-query lives under
        # ``replicated.{device}.attn_res.query`` so it is also
        # caught by the ``replicated.`` prefix here.
        if name.startswith("lm_head.") or name.startswith("replicated."):
            adamw_params.append(p)
        # KDA short-conv weights are 3D (nn.Conv1d: [D, 1, W]).
        # CPUMuon._newton_schulz does ``g.t()`` which only works on
        # 2-D matrices, so we must route them to AdamW. These params
        # are tiny (393K total) so the precision/regularization
        # difference vs Muon is negligible. Match by name suffix
        # (``...q_conv1d.weight``) for explicitness; the 3D guard
        # is a backstop in case fla adds more depthwise convs.
        elif (p.ndim == 3 or any(
                name.endswith(suffix) for suffix in
                (".q_conv1d.weight", ".k_conv1d.weight", ".v_conv1d.weight"))):
            adamw_params.append(p)
        elif p.ndim < 2:
            # 1D: norms, A_log, dt_bias, o_norm.weight, lm_head.bias.
            adamw_params.append(p)
        else:
            muon_params.append(p)

    muon_opt = CPUMuon(
        muon_params,
        lr=lr_muon,
        momentum=muon_momentum,
        nesterov=True,
        ns_steps=5,
        weight_decay=0.0,   # decoupled wd applied elsewhere if needed
        precision=precision,
    )
    adamw_opt = CPUAdamW(
        adamw_params,
        lr=lr_adamw,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=weight_decay,
        precision=precision,
    )
    return muon_opt, adamw_opt
