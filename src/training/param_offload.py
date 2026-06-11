"""Per-parameter optimizers with CPU offload for HippoLM v0.0.1.

Implements two optimizer variants used together via param groups:

  - :class:`CPUAdamW`     — AdamW with FP16 m, v state on CPU pinned
    memory. Forward / backward happen on GPU in FP16 (with tensor
    cores); after gradient accumulation the GPU-side ``.grad`` is
    streamed to CPU where the optimizer step is computed. The
    per-element update factor is computed in FP32 (to avoid FP16
    underflow on the divisor), then cast back to FP16 and applied
    in place on the GPU. This is the DeepSpeed ZeRO-Offload
    pattern with FP16 state.

  - :class:`CPUMuon`     — Muon (Newton-Schulz orthogonalization of
    the momentum matrix) with FP16 momentum on CPU pinned memory.
    The NS iteration is a sequence of matrix multiplies, which is
    *much* faster on GPU than on CPU, so each step streams the
    gradient (or accumulated momentum) to the GPU, runs 5 NS
    iterations in FP16 (tensor cores), and applies the resulting
    update to the GPU parameter directly. The CPU-side momentum
    state is only updated by accumulating the grad, which is cheap.

Both optimizers follow the same API surface as ``torch.optim.Optimizer``
minimally — ``step()`` consumes whatever gradients are present on
the parameters and ``zero_grad()`` clears them.

No gradient is stored on the GPU between steps. The training loop
copies each micro-batch's ``.grad`` to CPU and adds it to the
optimizer state's accumulator; only after ``gradient_accumulation_steps``
micro-batches does ``step()`` actually run.

Conventions
-----------
- Param ``.data`` lives on the GPU.
- ``.grad`` is materialized on the GPU only during ``backward()``;
  immediately after, the training loop calls
  :func:`stream_grads_to_cpu` which copies ``.grad`` to the per-param
  CPU accumulator and frees the GPU copy. So between micro-batches
  the GPU holds zero gradient memory.
- Optimizer state (m, v for AdamW; momentum for Muon) is FP16 on
  CPU pinned memory. For 624M params, this is 2 * 2 * 624M bytes
  = 2.5 GB CPU RAM (AdamW) or 1 * 2 * 624M = 1.25 GB (Muon). The
  halving vs FP32 is a deliberate v0.0.1 trade for CPU RAM and
  DMA bandwidth; FP32-divisor math in AdamW preserves precision
  where it matters.
"""
from __future__ import annotations

import math
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Iterable, List, Tuple

import torch
import torch.nn as nn


# --------------------------------------------------------------------------- #
# Per-param state descriptor.                                                 #
# --------------------------------------------------------------------------- #
@dataclass
class _ParamState:
    """Per-parameter CPU-side state for either AdamW or Muon."""

    param: nn.Parameter
    # The CPU accumulator. For AdamW, the optim step is on CPU, so
    # we accumulate raw grad into m/v directly. For Muon, we
    # accumulate grad into momentum (so the user's intent of
    # "stream grad to CPU and accumulate" maps to the standard
    # SGD-with-momentum update; NS is applied at step time on the
    # GPU).
    accum: torch.Tensor            # FP16, CPU pinned, shape = param.numel()
    # Pre-allocated buffer for the GPU->CPU copy target. Avoids
    # allocating a new pinned tensor on every micro-batch.
    pinned_in: torch.Tensor | None = None
    # Whether this state is for AdamW or Muon. Set by the factory.
    kind: str = "adamw"             # "adamw" or "muon"
    # AdamW-specific (also reused as Muon's momentum buffer; same
    # field is overloaded to keep the dataclass slim).
    exp_avg_sq: torch.Tensor | None = None
    # AdamW-specific: persistent first-moment buffer (m). Lives in
    # its own allocation so that ``m = beta1*m + (1-beta1)*g`` is
    # not aliased with the incoming grad ``g = s.accum``. An
    # earlier version of this code re-purposed ``s.accum`` as m
    # (zeroing it after the step) but that meant ``g`` and ``m``
    # were the same tensor during the in-place update, so the
    # chained ``m.mul_(beta1).add_(g, alpha=1-beta1)`` mutated
    # ``g`` mid-chain and produced ``m = 0.99*X`` instead of the
    # intended ``0.1*X`` (9.9x error on every step). Keeping
    # ``m`` separate is the standard AdamW layout and makes the
    # aliasing bug impossible by construction. Set to None for Muon.
    m: torch.Tensor | None = None
    # Cached for Muon: original param shape for reshape on GPU.
    shape: tuple = field(default_factory=tuple)
    # Step counter (per param; cheap).
    step: int = 0


# --------------------------------------------------------------------------- #
# CPU AdamW                                                                   #
# --------------------------------------------------------------------------- #
class CPUAdamW:
    """AdamW with CPU-side m, v state, GPU-side params.

    Memory layout per trainable param:
        CPU pinned: m (FP16, numel), v (FP16, numel), accum (FP16, numel)
        GPU:         param (training dtype, e.g. FP16), .grad (transient, FP16)

    The state buffers are FP16 (per v0.0.1 design — we trade a small
    amount of AdamW precision for halved CPU RAM and a 2x faster
    CPU-side m/v update). The divisor math in :meth:`step` is
    up-cast to FP32 to avoid FP16 underflow when ``v`` is small.

    On ``step()``:
        1. Read the latest accumulated grad (CPU pinned, FP16).
        2. Update m, v in place (FP16).
        3. Compute the update factor in FP32 then cast back to FP16
           (cheap numel cast, well under a millisecond for 254M elts).
        4. Stream the FP16 factor to the GPU param and apply
           ``p -= lr * factor`` (tensor-core matmul territory).

    The training loop is expected to call
    :func:`accumulate_grads_to_cpu` after each micro-batch's
    backward() to copy ``.grad`` into ``accum`` and free the GPU
    copy. ``step()`` then sees the accumulated grad on CPU.
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
        # Filter requires_grad, dedup by id() (param aliasing can
        # happen with tied weights).
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
                accum=torch.zeros(n, dtype=torch.float16, device="cpu").pin_memory(),
                # v MUST be FP32: an FP16 v can underflow to 0 for
                # entries where the grad magnitude is small
                # (1e-4 to 1e-3), and that makes the AdamW
                # denominator fall back to ``eps`` (1e-8),
                # blowing the per-element factor up to ~1e9 * m
                # which overflows the FP16 cast and corrupts
                # the param with inf. FP32 v never underflows
                # (FP32 has ~28 bits of dynamic range, way more
                # than the ~12 bits of FP16). m stays FP16
                # because the m update is bounded in magnitude
                # by the grad scale and FP16 precision is
                # sufficient for the per-element update.
                exp_avg_sq=torch.zeros(n, dtype=torch.float32, device="cpu").pin_memory(),
                # Persistent first-moment buffer. Allocated
                # separately from ``accum`` so that the in-place
                # m update cannot alias the incoming grad.
                m=torch.zeros(n, dtype=torch.float16, device="cpu").pin_memory(),
                kind="adamw",
                shape=p.shape,
            )

    def zero_grad(self, set_to_none: bool = True) -> None:
        for s in self.state.values():
            s.param.grad = None
            # NOTE: do NOT zero accum here — that would discard
            # the user's gradient accumulation across micro-batches.
            # The training loop is responsible for resetting accum
            # at the start of each accumulation cycle.

    # Per-chunk size for the streaming factor / update path.
    # 4 M elements is the sweet spot on V100 PCIe: 8 MB FP16 per
    # chunk keeps the GPU transient well under the 16 GB ceiling
    # and overlaps the H2D copy with the next chunk's compute.
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
            # m lives in its own buffer (``s.m``), initialised
            # to 0 in :meth:`__init__``. We update it in place;
            # on step 1, m = beta1*0 + (1-beta1)*g = 0.1*g.
            # The previous v0.0.1 layout over-loaded ``s.accum``
            # as both the grad accumulator AND m, but that made
            # ``g`` and ``m`` the same tensor for the in-place
            # update chain (``m.mul_(beta1).add_(g, 1-beta1)``),
            # which mutates ``g`` mid-chain to ``0.9*g_orig``
            # and produces ``m = 0.99*X`` instead of ``0.1*X``
            # (9.9x error on every step). The dedicated ``s.m``
            # buffer makes the aliasing bug impossible.
            m = s.m
            v = s.exp_avg_sq
            # Update m (FP16, in place) and v (FP32, in place).
            m.mul_(beta1).add_(g, alpha=1.0 - beta1)
            v.mul_(beta2).addcmul_(g, g, value=1.0 - beta2)
            # Compute update: -lr * (m/bc1) / (sqrt(v/bc2) + eps) - lr*wd*p
            # We need a CPU copy of the param to update; instead
            # we apply the update in place on the GPU by streaming
            # the per-element update factor.
            #
            # update_factor = (m/bc1) / (sqrt(v/bc2) + eps) + wd*p
            # but p is on GPU and we want to avoid streaming it to
            # CPU. Trick: stream a *small* CPU scalar (the update
            # factor) and apply on GPU. But the factor is
            # per-element, not a scalar. So we either:
            #   (a) stream the whole param to CPU, update on CPU, stream back, OR
            #   (b) stream the per-element factor to GPU and apply there.
            #
            # (b) is cheaper memory-wise (factor is the same size
            # as the param but we transfer it once per step). (a)
            # requires transferring the param in both directions
            # (1x extra) plus keeping a CPU copy.
            #
            # We use (b) for memory efficiency.
            #
            # FP16 SAFETY: do the divisor math in FP32. The m
            # update is in FP16 (in place); the v update is in
            # FP32 (so v never underflows). Pure FP16
            # ``1/sqrt(v)`` would underflow when v ≈ 0 (v_min in
            # FP16 is ~6e-8, but the reciprocal of a small v
            # rounds to 0). The up-cast is a ``numel`` op — cheap
            # relative to the GPU transfer.
            #
            # STREAMING: for large params (e.g. embed_tokens at
            # ``[vocab, hidden] = [248320, 1024]`` = 254 M elts,
            # 508 MB FP16 for the factor) we chunk the
            # factor-compute / apply loop so the peak GPU memory
            # for ``factor`` is bounded to ``CHUNK`` elements.
            # The chunks are independent (each is a slice along
            # the flat param view), so the result is bit-identical
            # to the non-streaming path.
            p_flat = s.param.data.view(-1)
            n = p_flat.numel()
            if n <= CHUNK:
                denom = (v / bc2).sqrt_().add_(eps)            # FP32, numel
                factor = (m.float() / bc1) / denom             # FP32, numel
                if wd != 0.0:
                    s.param.data.mul_(1.0 - lr * wd)
                factor_gpu = factor.to(
                    device=s.param.device, dtype=s.param.dtype,
                    non_blocking=True,
                ).view_as(s.param.data)
                # DIAG: per-stage telemetry for the first few steps
                # so we can localize where inf/nan is introduced on
                # the embed (the post-step pmax went to inf on step 0).
                import os as _os_diag
                import logging as _logging_diag
                if _os_diag.environ.get("HIPPOLM_ADAMW_DIAG") == "1" and step <= int(
                    _os_diag.environ.get("HIPPOLM_DIAG_STEPS", "3")
                ):
                    _log_diag = _logging_diag.getLogger(__name__)
                    _has = lambda t, dt: t.detach().to(dt, copy=True)
                    _factor_fp16 = factor.to(torch.float16).to(torch.float32)
                    _factor_gpu_fp32 = _has(factor_gpu, torch.float32)
                    def _stats(name, t):
                        f = t.detach().float()
                        nan_n = torch.isnan(f).sum().item()
                        inf_n = torch.isinf(f).sum().item()
                        return (
                            f"{name}: max_abs={f.abs().max().item():.3e}"
                            f" min_abs={f.abs().min().item():.3e}"
                            f" nan={int(nan_n)} inf={int(inf_n)}"
                            f" numel={f.numel()}"
                        )
                    _log_diag.info(
                        f"[adamw-diag step={step} shape={tuple(s.shape)}]"
                        f" {_stats('m(after update, fp16)', m)}"
                        f" | {_stats('v(after update, fp16)', v)}"
                        f" | {_stats('denom(fp32)', denom)}"
                        f" | {_stats('factor(fp32)', factor)}"
                        f" | {_stats('factor(fp16->fp32)', _factor_fp16)}"
                        f" | {_stats('factor_gpu(fp32)', _factor_gpu_fp32)}"
                    )
                s.param.data.add_(factor_gpu, alpha=-lr)
            else:
                # Streaming path: chunk-process the param.
                if wd != 0.0:
                    s.param.data.mul_(1.0 - lr * wd)
                for start in range(0, n, CHUNK):
                    end = min(start + CHUNK, n)
                    v_chunk = v[start:end]
                    denom = (v_chunk / bc2).sqrt_().add_(eps)   # FP32, chunk
                    factor = (m[start:end].float() / bc1) / denom
                    # Stream the FP32 factor to the GPU as the
                    # param dtype (FP16) and apply in place.
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
    """Muon (refactored) with CPU momentum, GPU Newton-Schulz.

    State per param:
        CPU pinned: accum (FP16, numel), momentum (FP16, numel)
        GPU:         param (FP16), .grad (transient, FP16)

    The state buffers are FP16 (per v0.0.1 design). Momentum updates
    are SGD-style ``m = beta*m + (1-beta)*g``; running them in FP16
    is safe because (a) the magnitude stays bounded by the gradient
    scale, and (b) the NS iteration in :meth:`_newton_schulz` casts
    to FP16 anyway, so the precision boundary is consistent.

    The training loop's :func:`accumulate_grads_to_cpu` adds the
    GPU ``.grad`` to ``accum`` (CPU, FP16). On ``step()`` we read
    ``accum``, update momentum, then run NS on the GPU.
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
                    f"CPUMuonV2 got {p.ndim}D param of shape {tuple(p.shape)}; "
                    f"use CPUAdamW for 1D params."
                )
            n = p.numel()
            st = _ParamState(
                param=p,
                accum=torch.zeros(n, dtype=torch.float16, device="cpu").pin_memory(),
                kind="muon",
                shape=p.shape,
            )
            # Separate momentum buffer.
            st.exp_avg_sq = torch.zeros(n, dtype=torch.float16, device="cpu").pin_memory()
            self.state[id(p)] = st

    def zero_grad(self, set_to_none: bool = True) -> None:
        for s in self.state.values():
            s.param.grad = None

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
                # Build the inner polynomial in FP16.
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
    # is a 2-D ``[CHUNK, cols]`` matrix; for embed (cols=1024)
    # the chunk is ``[CHUNK, 1024]`` = 8 MB FP16 at CHUNK=4096.
    _STREAM_CHUNK_ROWS = 4096

    def step(self) -> None:
        mom = self.momentum
        lr = self.lr
        wd = self.weight_decay
        nesterov = self.nesterov
        CHUNK_ROWS = self._STREAM_CHUNK_ROWS
        for s in self.state.values():
            if s.accum.abs().sum().item() == 0:
                continue
            # Update momentum: m = beta*m + grad
            m = s.exp_avg_sq
            m.mul_(mom).add_(s.accum, alpha=1.0 - mom)
            shape = s.shape
            rows, cols = shape[0], shape[1]
            # Decoupled weight decay (applied once to the full param
            # rather than per-chunk to keep semantics identical to
            # the non-streaming path).
            if wd != 0.0:
                s.param.data.mul_(1.0 - lr * wd)
            if rows * cols <= CHUNK_ROWS * cols:
                # Small param: single chunk.
                g = m.view(shape).to(s.param.device, non_blocking=True)
                update = self._newton_schulz(g)
                update = update.to(s.param.dtype)
                s.param.data.add_(update, alpha=-lr)
            else:
                # Streaming path: chunk along the row dim so each
                # NS iteration is on a 2-D slice well under the
                # V100's 16 GB ceiling. The flat momentum
                # ``m`` (1-D, ``numel``) is sliced in strides of
                # ``CHUNK_ROWS * cols`` and reshaped per chunk.
                for r_start in range(0, rows, CHUNK_ROWS):
                    r_end = min(r_start + CHUNK_ROWS, rows)
                    flat_start = r_start * cols
                    flat_end = r_end * cols
                    m_chunk_flat = m[flat_start:flat_end]
                    g = m_chunk_flat.view(r_end - r_start, cols).to(
                        s.param.device, non_blocking=True,
                    )
                    update = self._newton_schulz(g)
                    update = update.to(s.param.dtype)
                    s.param.data[r_start:r_end].add_(update, alpha=-lr)
            # Reset grad accumulator for next accumulation cycle.
            s.accum.zero_()


# --------------------------------------------------------------------------- #
# Public API: stream grads to CPU and reset                                   #
# --------------------------------------------------------------------------- #
def accumulate_grads_to_cpu(
    optimizers: List,
    sync_device: int | None = None,
) -> None:
    """Copy each trainable param's ``.grad`` to its optimizer state's
    CPU accumulator and free the GPU copy.

    Safe to call on either AdamW or Muon states: the state object
    has the same ``accum`` slot in both.

    The async ``.to("cpu")`` is issued first (with a flattened
    view of the grad, so the resulting ``src_cpu`` matches the
    1-D ``accum`` tensor), then we sync the current CUDA device
    (or ``sync_device`` if given) before performing the in-place
    CPU add. This avoids the race where ``s.accum.add_(src)`` runs
    on the CPU before the DMA copy populates ``src``.
    """
    pending: list[tuple[torch.Tensor, torch.Tensor]] = []
    for opt in optimizers:
        for s in opt.state.values():
            g = s.param.grad
            if g is None:
                continue
            # Issue the async copy; flatten so the destination
            # ``accum`` (1-D, ``numel``, FP16) and the source have
            # the same shape. Cast to FP16 BEFORE the DMA so the
            # transfer itself is FP16 → halves the PCIe bandwidth
            # vs FP32 and matches the destination buffer dtype.
            src_cpu = (
                g.detach()
                .reshape(-1)
                .to("cpu", non_blocking=True)
                .to(torch.float16)
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


def zero_cpu_grad_accum(optimizers: List) -> None:
    """Reset CPU accumulators (call this AFTER step() to start the
    next accumulation cycle)."""
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
) -> Tuple[List, List]:
    """Build a (muon_optimizer, adamw_optimizer) pair for a single
    rank's model fragment.

    Standard Muon routing:
        - 1D params (RMSNorm.weight, BlockAttnRes.query,
          KDA A_log/dt_bias with ``_no_weight_decay``): AdamW
        - 2D Linear weights: Muon
        - Embedding + lm_head: AdamW (per user spec)
    """
    muon_params: list[nn.Parameter] = []
    adamw_params: list[nn.Parameter] = []
    seen: set[int] = set()
    for name, p in model.named_parameters_per_device(device):
        if not p.requires_grad or id(p) in seen:
            continue
        seen.add(id(p))
        # Per user spec: lm_head and embed_tokens -> AdamW.
        if name.startswith("lm_head.") or name.startswith("replicated."):
            adamw_params.append(p)
        # BlockAttnRes pseudo-query is now 2-D
        # ``[num_heads // world, head_dim]`` rather than a 1-D
        # vector. It's a learned attention parameter, not a
        # Linear weight, so it should go to AdamW regardless
        # of its ndim. Match by exact suffix.
        elif name.endswith(".attn_res.query") or name.endswith(".mlp_res.query"):
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
    )
    adamw_opt = CPUAdamW(
        adamw_params,
        lr=lr_adamw,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=weight_decay,
    )
    return muon_opt, adamw_opt
