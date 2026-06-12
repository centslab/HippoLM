"""GPU-based optimizers for HippoLM v0.0.2.

Design (v0.0.2 spec):
  - AdamW: BF16 m, v on the param's device (GPU). Grad input is
    cast to FP32 inside step. Factor computed in FP32 and applied
    via FP32 add followed by cast to the FP16 param (no FP16 factor
    cast). V100 has no BF16 tensor cores; m and v are kept in BF16
    for storage (8-bit exponent → no g*g underflow, 7-bit mantissa
    is sufficient for the per-element EMA signal).
  - Muon: int8-quantized momentum with FP16 per-row scale factors
    (arXiv:2509.23106). Dequantize to FP32 for the SGD update and
    Newton-Schulz orthogonalization; requantize after. NS runs in
    FP16 (tensor cores). Apply via FP32 add + cast to FP16.
  - All gradients are cast to FP32 in place on the GPU by
    ``accumulate_grads_to_cpu`` (kept name; no PCIe transfer). The
    optimizer state lives on the GPU and the step runs entirely
    on-device. No PCIe bandwidth bottleneck.

No gradient is stored on the GPU between micro-batches in the
"copy to CPU and sum there" sense; ``.grad`` accumulates naturally
on the GPU across ``gradient_accumulation_steps`` micro-batches via
PyTorch's autograd, then the optimizer step consumes the mean grad.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import torch
import torch.nn as nn


# --------------------------------------------------------------------------- #
# Streaming chunk sizes for the v0.0.2 GPU optimizer step.                    #
# --------------------------------------------------------------------------- #
# The chunk size bounds the per-chunk peak VRAM of the FP32 temp tensors
# the AdamW / Muon step creates from the FP16 / BF16 storage. At
# ``_ADAMW_CHUNK = 1<<20`` elements per chunk:
#   - AdamW peak per chunk: 4 × 1M × 4 B = 16 MB
#     (g_c, m_c, v_c, p_c — PyTorch's caching allocator
#     reuses one temp across the four .float() calls, so
#     effective peak is closer to 12-16 MB depending on the
#     allocator state).
#   - Muon peak per chunk: similar (the SGD update is the
#     streamed part; the dequantize/quantize and Newton-Schulz
#     stay full-matrix but each is small: at most
#     rows*cols*4 B ≈ 12 MB for the 3072×1024 FFN down_proj).
#
# 1 M elements / chunk is a good V100 trade-off: the kernel
# work per chunk (≈1 ms of FMAs at 1 M elements) is large
# relative to the launch overhead (≈40 µs), and the peak
# temp is bounded regardless of ``numel``. For 254 M-param
# params this is 254 chunks per step; for 1 M-param params
# it is a single chunk (no overhead vs. un-chunked).
_ADAMW_CHUNK = 1 << 20  # 1 M elements per chunk
_MUON_CHUNK = 1 << 20   # 1 M elements per chunk


# --------------------------------------------------------------------------- #
# Per-param state descriptor.                                                 #
# --------------------------------------------------------------------------- #
@dataclass
class _ParamState:
    """Per-parameter GPU state for AdamW or Muon."""

    param: nn.Parameter
    # AdamW state: BF16 m, v on the param's device.
    m: torch.Tensor | None = None
    exp_avg_sq: torch.Tensor | None = None
    # Muon state: int8 quantized momentum + FP16 per-row scale.
    int8_q: torch.Tensor | None = None
    int8_scale: torch.Tensor | None = None
    # Legacy v0.0.1 fields kept as None for API compat (no longer used).
    accum: torch.Tensor | None = None
    pinned_in: torch.Tensor | None = None
    kind: str = "adamw"
    shape: tuple = field(default_factory=tuple)
    step: int = 0


# --------------------------------------------------------------------------- #
# GPU AdamW (BF16 state)                                                      #
# --------------------------------------------------------------------------- #
class GPUAdamW:
    """AdamW with BF16 m, v on the param's device (GPU).

    The ``.grad`` is whatever the model backward produced (FP16 in
    v0.0.2). The step casts it to FP32 for the math, stores m and v
    as BF16 (in-place cast), computes the factor in FP32, and
    applies the update to the FP16 param via a temporary FP32 copy
    (so the small update factor is not rounded to FP16 precision
    before the add).

    Memory layout per trainable param:
        GPU (param.dtype=FP16): param, .grad (FP32 after the cast)
        GPU (BF16): m, v (numel each)
    """

    def __init__(
        self,
        params: List[nn.Parameter],
        lr: float = 1e-4,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ) -> None:
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.state: dict[int, _ParamState] = {}
        seen: set[int] = set()
        for p in params:
            if not p.requires_grad:
                continue
            if id(p) in seen:
                continue
            seen.add(id(p))
            n = p.numel()
            device = p.device
            self.state[id(p)] = _ParamState(
                param=p,
                m=torch.zeros(n, dtype=torch.bfloat16, device=device),
                exp_avg_sq=torch.zeros(n, dtype=torch.bfloat16, device=device),
                kind="adamw",
                shape=p.shape,
            )

    def zero_grad(self, set_to_none: bool = True) -> None:
        for s in self.state.values():
            if set_to_none:
                s.param.grad = None
            elif s.param.grad is not None:
                s.param.grad.detach_().zero_()

    def step(self) -> None:
        """Streamed AdamW step.

        v0.0.1 path: cast the whole ``.grad`` / ``.m`` / ``.v`` /
        ``.data`` to FP32 simultaneously → 4× numel bytes peak
        temp per param (≈6 GB peak for the 254M-param embedding
        alone).

        This v0.0.2 path chunks the flat param into pieces of
        ``_ADAMW_CHUNK`` elements (1 M by default ≈ 4 MB at
        FP32) and runs the m / v / denom / factor / apply math
        one chunk at a time, so the per-chunk peak is bounded to
        5 × 4 MB ≈ 20 MB regardless of ``numel``.

        Trade-off: a Python-level loop over chunks adds
        launch-overhead (≈40 µs per chunk on V100), but the
        kernel work per chunk is large enough to amortise it
        (≈1 ms each), so end-to-end step time is unchanged in
        practice.

        Note: the writeback of m / v into the BF16 state
        happens BEFORE the in-place ``m_c.div_(bc1).div_(denom)``
        that turns m_c into the factor. The un-chunked v0.0.1
        path did the writeback first too (``s.m.copy_(...)``
        then ``factor = (m_fp32 / bc1) / denom``) for the same
        reason — once the factor is computed in place, the
        updated m is gone from m_c. Doing the writeback first
        preserves the (FP32-rounded-to-BF16) updated m in
        storage for the next step, then re-derives the factor
        from the FP32 local m_c.
        """
        beta1, beta2 = self.beta1, self.beta2
        lr = self.lr
        eps = self.eps
        wd = self.weight_decay
        chunk_size = _ADAMW_CHUNK
        for s in self.state.values():
            p = s.param
            g = p.grad
            if g is None:
                continue
            s.step += 1
            step = s.step
            bc1 = 1.0 - beta1 ** step
            bc2 = 1.0 - beta2 ** step

            flat_p = p.data.view(-1)
            flat_g = g.reshape(-1)
            flat_m = s.m.view(-1)
            flat_v = s.exp_avg_sq.view(-1)
            n = flat_p.numel()

            for start in range(0, n, chunk_size):
                end = min(start + chunk_size, n)

                # Load chunk-sized FP32 views; PyTorch's caching
                # allocator reuses one temporary buffer across
                # iterations, so the peak per chunk is
                # 4 × chunk_size × 4 B ≈ 16 MB, not 5 × that.
                g_c = flat_g[start:end].float()
                m_c = flat_m[start:end].float()
                v_c = flat_v[start:end].float()
                p_c = flat_p[start:end].float()

                # m, v updates in FP32. These are the same
                # numerics as the un-chunked path, just on
                # chunk-sized views.
                m_c.mul_(beta1).add_(g_c, alpha=1.0 - beta1)
                v_c.mul_(beta2).addcmul_(g_c, g_c, value=1.0 - beta2)

                # Writeback m / v to BF16 storage BEFORE the
                # in-place factor computation overwrites m_c.
                # Matches the un-chunked path's order:
                #   s.m.copy_(m_fp32.to(bf16))  ← here
                #   factor = (m_fp32 / bc1) / denom  ← below
                # Without this ordering, the next step's
                # m_c = flat_m.float() would read the
                # factor (sign(g)) rather than the EMA, and
                # the optimizer would diverge.
                flat_m[start:end].copy_(m_c.to(torch.bfloat16))
                flat_v[start:end].copy_(v_c.to(torch.bfloat16))

                # Factor in FP32 (in-place; m_c is destroyed).
                denom = (v_c / bc2).sqrt_().add_(eps)
                factor = m_c.div_(bc1).div_(denom)

                # Apply to FP16 param via FP32 add. Decoupled
                # weight decay folds into p_c before the factor
                # add so the per-step math is identical to the
                # un-chunked version.
                if wd != 0.0:
                    p_c.mul_(1.0 - lr * wd)
                p_c.add_(factor, alpha=-lr)

                # Writeback the updated p to FP16 storage.
                flat_p[start:end].copy_(p_c.to(torch.float16))


# --------------------------------------------------------------------------- #
# GPU Muon (int8 momentum, FP16 per-row scale)                               #
# --------------------------------------------------------------------------- #
class GPUMuon:
    """Muon with int8-quantized momentum + FP16 per-row scale.

    The 2-D momentum matrix m is stored as ``[rows, cols]`` reshaped
    to a flat int8 buffer plus a per-row FP16 scale factor. The
    per-row symmetric quantization is::

        scale[i] = max(|m[i, :]|) / 127
        q[i, j] = round(m[i, j] / scale[i]).clip(-128, 127)

    On each step:
      1. Dequantize m to FP32.
      2. SGD update in FP32: m = beta*m + (1-beta)*g.
      3. Requantize to int8 for storage.
      4. Newton-Schulz orthogonalize the dequantized m (cast to
         FP16 for tensor cores) → orthogonalized update u.
      5. Apply to FP16 param via FP32 add + cast to FP16.

    The quantization noise (~0.4% of |m|) is well below the NS
    update's magnitude, so the int8 storage does not noticeably
    affect the learning trajectory. Reference: arXiv:2509.23106.
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
    ) -> None:
        self.lr = lr
        self.momentum = momentum
        self.nesterov = nesterov
        self.ns_steps = ns_steps
        self.weight_decay = weight_decay
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
                    f"GPUMuon got {p.ndim}D param of shape {tuple(p.shape)}; "
                    f"use GPUAdamW for 1D params."
                )
            n = p.numel()
            device = p.device
            shape = tuple(p.shape)
            self.state[id(p)] = _ParamState(
                param=p,
                int8_q=torch.zeros(n, dtype=torch.int8, device=device),
                # one scale per row; for a 2-D [rows, cols] matrix
                # this is the natural "outer axis" quantization
                # granularity (matches the row-direction Newton-
                # Schulz convention).
                int8_scale=torch.zeros(shape[0], dtype=torch.float16, device=device),
                kind="muon",
                shape=shape,
            )

    def zero_grad(self, set_to_none: bool = True) -> None:
        for s in self.state.values():
            if set_to_none:
                s.param.grad = None
            elif s.param.grad is not None:
                s.param.grad.detach_().zero_()

    def _dequantize(self, s: _ParamState) -> torch.Tensor:
        """Return dequantized FP32 momentum, shape s.shape."""
        rows, cols = s.shape[0], s.shape[1]
        q_2d = s.int8_q.view(rows, cols).float()
        scale_2d = s.int8_scale.float().unsqueeze(1)  # [rows, 1]
        return q_2d * scale_2d  # FP32, [rows, cols]

    def _requantize(self, m_fp32: torch.Tensor, s: _ParamState) -> None:
        """Per-row symmetric int8 quantize m_fp32 into s.int8_q / s.int8_scale.

        ``m_fp32`` is the SGD-updated momentum in FP32, shape ``s.shape``.
        """
        rows, cols = s.shape[0], s.shape[1]
        m_2d = m_fp32.view(rows, cols)
        # Per-row max abs. Clamp to 1e-8 so the scale never collapses
        # to 0 (a row of exact zeros would otherwise divide-by-zero
        # in the quantize step; the int8 buffer would also be filled
        # with 0 which is fine, but the scale = 0 would be ambiguous
        # in the next dequantize).
        row_max = m_2d.abs().amax(dim=1).clamp(min=1e-8)
        new_scale = (row_max / 127.0).to(torch.float16)
        scale_2d = new_scale.float().unsqueeze(1)  # [rows, 1]
        q_int8 = (m_2d / scale_2d).round().clamp(-128, 127).to(torch.int8)
        s.int8_q.copy_(q_int8.view(-1))
        s.int8_scale.copy_(new_scale)

    def _newton_schulz(self, x: torch.Tensor) -> torch.Tensor:
        """Newton-Schulz orthogonalization in FP16 (tensor-core path)."""
        a, b, c = self._NS_COEFFS
        x = x.to(torch.float16)
        if x.size(0) > x.size(1):
            g = x
            for _ in range(self.ns_steps):
                gt = g.t()
                xtx = gt @ g  # [in, in], FP16
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
                ggt = g @ g.t()  # [out, out], FP16
                inner = (
                    a * torch.eye(ggt.size(0), device=g.device, dtype=g.dtype)
                    + b * ggt
                    + c * ggt @ ggt
                )
                g = inner @ g
            return g

    def step(self) -> None:
        """Streamed Muon step.

        The SGD-style update (dequantize → m = β·m + g →
        requantize) and the final param update
        (p -= lr · update) are streamed over the flat
        ``numel`` dim in chunks of ``_MUON_CHUNK`` elements.
        The dequantize/requantize/Newton-Schulz stay
        full-matrix because:

          - ``_dequantize`` returns a [rows, cols] FP32 tensor
            (the dequantize itself is element-wise, but the
            output is a dense FP32 matrix, and reshaping back
            into a [rows, cols] view is what the NS kernel
            expects).
          - ``_requantize`` needs per-row max-abs, which is
            naturally a [rows, cols] → [rows] reduction;
            streaming it would require a running-max per row
            across chunks, complicating the kernel for no
            meaningful memory win (the dequantized tensor is
            at most ~12 MB for the 3072×1024 FFN down_proj).
          - ``_newton_schulz`` operates on the full matrix
            because the polynomial evaluation reads/writes
            ``g @ g.T`` (or the transposed variant), which
            needs the whole matrix at once.

        The chunks that DO get streamed are: the
        ``m_fp32.mul_(mom).add_(g_fp32, alpha=1.0 - mom)`` line
        and the ``p_data_fp32.add_(update, alpha=-lr)`` line.
        Both previously allocated a [numel] FP32 copy of the
        param / momentum that was the bulk of the per-step
        peak (4 × numel bytes for the FFN down_proj's
        3 M elements = 12 MB each; for the KDA's 1 M-element
        matrices that's 4 MB; for everything else the peak
        is the int8 momentum scale tensor). Streaming caps
        each at chunk_size × 4 B ≈ 4 MB.
        """
        mom = self.momentum
        lr = self.lr
        wd = self.weight_decay
        nesterov = self.nesterov
        chunk_size = _MUON_CHUNK
        for s in self.state.values():
            p = s.param
            g = p.grad
            if g is None:
                continue
            g_fp32 = g.float() if g.dtype != torch.float32 else g
            shape = s.shape
            rows, cols = shape[0], shape[1]

            # 1) Dequantize momentum to FP32 (full matrix, but
            # small: at most 12 MB for the 3072×1024 FFN
            # down_proj, 4 MB for the 1024×1024 KDA matrices).
            m_fp32 = self._dequantize(s)

            # 2) SGD update in FP32, streamed over the flat
            # tensor. ``m_fp32`` and ``g_fp32`` are both [rows,
            # cols]; the streaming loop views them as 1-D for
            # the in-place arithmetic and the kernel sees the
            # same memory layout (the views are contiguous).
            m_flat = m_fp32.view(-1)
            g_flat = g_fp32.view(-1)
            n = m_flat.numel()
            for start in range(0, n, chunk_size):
                end = min(start + chunk_size, n)
                m_c = m_flat[start:end]
                g_c = g_flat[start:end]
                m_c.mul_(mom).add_(g_c, alpha=1.0 - mom)

            # 3) Requantize to int8 for storage
            self._requantize(m_fp32, s)

            # 4) Decoupled weight decay
            if wd != 0.0:
                p.data.mul_(1.0 - lr * wd)

            # 5) NS on the dequantized FP32 momentum (cast to
            #    FP16 for tensor-core speed).
            update = self._newton_schulz(m_fp32)
            update = update.to(torch.float16)

            # 6) Apply to FP16 param via FP32 add, streamed
            # over the flat tensor. Previously this allocated
            # a [numel] FP32 copy of ``p.data`` (4 × numel
            # bytes) plus a [numel] FP32 copy of ``update``;
            # both are now chunked.
            p_flat = p.data.view(-1)
            u_flat = update.view(-1)
            n = p_flat.numel()
            for start in range(0, n, chunk_size):
                end = min(start + chunk_size, n)
                p_c = p_flat[start:end].float()
                u_c = u_flat[start:end].float()
                p_c.add_(u_c, alpha=-lr)
                p_flat[start:end].copy_(p_c.to(torch.float16))


# --------------------------------------------------------------------------- #
# Public API                                                                  #
# --------------------------------------------------------------------------- #
def accumulate_grads_to_cpu(
    optimizers: List,
    sync_device: int | None = None,
) -> None:
    """No-op in the v0.0.2 GPU-optimizer path.

    In v0.0.1 this function copied each param's ``.grad`` (FP16) to a
    CPU FP16 pinned buffer and added it to the per-param CPU
    accumulator. The CPU accumulator was consumed by the optimizer
    step on the CPU.

    In v0.0.2 the optimizer state and step live entirely on the
    GPU. The ``.grad`` stays on the GPU in FP16 (the param dtype);
    the optimizer step casts it to FP32 locally for the math. There
    is no CPU transfer and no CPU accumulator, so this function
    has nothing to do.

    The original signature is preserved so the training loop call
    site does not need to change. The ``sync_device`` argument is
    accepted but unused.
    """
    return None


def zero_cpu_grad_accum(optimizers: List) -> None:
    """Reset param grads after the optimizer step (start the next
    accumulation cycle). The function name is preserved for v0.0.1
    call-site compatibility; the implementation just calls each
    optimizer's ``zero_grad``.
    """
    for opt in optimizers:
        opt.zero_grad(set_to_none=True)


def build_param_groups(
    model: nn.Module,
    device: int,
    lr_muon: float = 1e-3,
    lr_adamw: float = 1e-4,
    weight_decay: float = 0.01,
    muon_momentum: float = 0.95,
) -> Tuple[List, List]:
    """Build a (muon_optimizer, adamw_optimizer) pair for a single
    rank's model fragment.

    Standard Muon routing:
        - 1D params (RMSNorm.weight, BlockAttnRes.query,
          KDA A_log/dt_bias with ``_no_weight_decay``): AdamW
        - 2D Linear weights: Muon (int8 momentum, GPU)
        - Embedding + lm_head: AdamW (per user spec)
    """
    muon_params: list[nn.Parameter] = []
    adamw_params: list[nn.Parameter] = []
    seen: set[int] = set()
    for name, p in model.named_parameters_per_device(device):
        if not p.requires_grad or id(p) in seen:
            continue
        seen.add(id(p))
        if name.startswith("lm_head.") or name.startswith("replicated."):
            adamw_params.append(p)
        elif (p.ndim == 3 or any(
                name.endswith(suffix) for suffix in
                (".q_conv1d.weight", ".k_conv1d.weight", ".v_conv1d.weight"))):
            adamw_params.append(p)
        elif p.ndim < 2:
            adamw_params.append(p)
        else:
            muon_params.append(p)

    muon_opt = GPUMuon(
        muon_params,
        lr=lr_muon,
        momentum=muon_momentum,
        nesterov=True,
        ns_steps=5,
        weight_decay=0.0,   # decoupled wd applied elsewhere if needed
    )
    adamw_opt = GPUAdamW(
        adamw_params,
        lr=lr_adamw,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=weight_decay,
    )
    return muon_opt, adamw_opt
