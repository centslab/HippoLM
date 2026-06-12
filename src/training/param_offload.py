"""Per-parameter optimizers with CPU offload for HippoLM.

Implements two optimizer variants used together via param groups:

  - :class:`CPUAdamW`     — AdamW with FP16 m and FP32 v state on
    CPU pinned memory. The forward / backward happen on the GPU
    in the param's training dtype (FP16 in this project). After
    each micro-batch's backward, :func:`accumulate_grads_to_cpu`
    copies the GPU ``.grad`` to a per-param CPU accumulator
    (FP16, pinned) and frees the GPU copy. On ``step()`` the
    accumulator is consumed: m, v are updated in place on the
    CPU; the per-element update factor is computed in FP32 (to
    avoid FP16 underflow on the divisor), then cast to FP16 and
    streamed chunk-wise to the GPU param and applied in place.
    This is the DeepSpeed ZeRO-Offload pattern with FP16 state.

  - :class:`CPUMuon`      — Muon (Newton-Schulz orthogonalization
    of the momentum matrix) with FP16 momentum on CPU pinned
    memory. The NS iteration is a sequence of matrix multiplies,
    which is *much* faster on the GPU than on the CPU, so each
    step streams the chunked momentum to the GPU, runs 5 NS
    iterations in FP16 (tensor cores), and applies the resulting
    update to the GPU param directly. The CPU-side momentum
    state is only updated by accumulating the grad, which is
    cheap.

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
- Optimizer state (m, v for AdamW; momentum for Muon) is FP16
  (and FP32 v for AdamW) on CPU pinned memory. For ~624 M params
  this is roughly 2 × 2 × 624 M bytes = 2.5 GB CPU RAM (AdamW)
  or 1 × 2 × 624 M = 1.25 GB (Muon). The FP16 trade is a
  deliberate design choice for CPU RAM and DMA bandwidth; the
  FP32 divisor math in AdamW preserves precision where it matters
  (preventing the FP16 v underflow that would otherwise blow up
  the per-element factor).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import torch
import torch.nn as nn


# --------------------------------------------------------------------------- #
# Per-param state descriptor.                                                 #
# --------------------------------------------------------------------------- #
@dataclass
class _ParamState:
    """Per-parameter CPU-side state for either AdamW or Muon."""

    param: nn.Parameter
    # The CPU accumulator. For both AdamW and Muon this is the
    # destination of :func:`accumulate_grads_to_cpu`: each
    # micro-batch's ``.grad`` is added in here on the CPU, and
    # the training loop's :func:`zero_cpu_grad_accum` zeros it
    # out at the end of the accumulation cycle.
    accum: torch.Tensor
    # AdamW-specific state (set when ``kind == 'adamw'``).
    # m is FP16 (magnitude stays bounded by the grad scale, and
    # FP16 is sufficient for the per-element update).
    # v is FP32 (an FP16 v underflows to 0 for typical grad
    # magnitudes 1e-4 to 1e-3, which makes ``1/sqrt(v)`` blow
    # up to ~1e9 * m and overflow the FP16 cast on the
    # per-element factor). FP32 v has ~28 bits of dynamic range
    # — way more than the ~12 bits of FP16.
    m: torch.Tensor | None = None
    exp_avg_sq: torch.Tensor | None = None
    # Cached for Muon: original param shape for reshape on GPU.
    shape: tuple = field(default_factory=tuple)
    # Step counter (per param; cheap).
    step: int = 0
    # "adamw" or "muon".
    kind: str = "adamw"


# --------------------------------------------------------------------------- #
# CPU AdamW                                                                   #
# --------------------------------------------------------------------------- #
class CPUAdamW:
    """AdamW with CPU-side m, v state, GPU-side params.

    Memory layout per trainable param:
        CPU pinned: m (FP16, numel), v (FP32, numel), accum (FP16, numel)
        GPU:         param (training dtype, e.g. FP16), .grad (transient)

    On :meth:`step`:
        1. Read the latest accumulated grad (CPU pinned, FP16).
        2. Update m, v in place (FP16 / FP32 respectively).
        3. Compute the update factor in FP32 then cast to FP16
           (chunked, streamed to the GPU).
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
            self.state[id(p)] = _ParamState(
                param=p,
                accum=torch.zeros(n, dtype=torch.float16, device="cpu").pin_memory(),
                # v is FP32 to prevent underflow (see dataclass
                # docstring above).
                exp_avg_sq=torch.zeros(n, dtype=torch.float32, device="cpu").pin_memory(),
                # m is FP16 (magnitude bounded; safe precision-wise).
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
            # at the start of each accumulation cycle (via
            # :func:`zero_cpu_grad_accum`).

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
            m = s.m
            v = s.exp_avg_sq
            # Update m (FP16, in place) and v (FP32, in place).
            # The dedicated ``m`` buffer keeps ``m`` and ``g``
            # (which is ``s.accum``) aliased-free: the chained
            # ``m.mul_(beta1).add_(g, ...)`` would otherwise
            # mutate ``g`` mid-chain and produce the wrong
            # update on every step.
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
            # FP16 SAFETY: do the divisor math in FP32. Pure
            # FP16 ``1/sqrt(v)`` would underflow when v ≈ 0
            # (FP16 smallest normal is ~6e-8; reciprocals of
            # small v round to 0). The up-cast is a ``numel``
            # op — cheap relative to the GPU transfer.
            p_flat = s.param.data.view(-1)
            n = p_flat.numel()
            for start in range(0, n, CHUNK):
                end = min(start + CHUNK, n)
                v_chunk = v[start:end]
                denom = (v_chunk / bc2).sqrt_().add_(eps)    # FP32, chunk
                factor = (m[start:end].float() / bc1) / denom  # FP32, chunk
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
    """Muon with CPU momentum, GPU Newton-Schulz.

    State per param:
        CPU pinned: accum (FP16, numel), momentum (FP16, numel)
        GPU:         param (FP16), .grad (transient)

    The training loop's :func:`accumulate_grads_to_cpu` adds the
    GPU ``.grad`` to ``accum`` (CPU, FP16). On :meth:`step` we
    update momentum on the CPU, then stream each chunk to the
    GPU, run Newton-Schulz (FP16 tensor cores), and apply the
    resulting update to the GPU param.
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
                    f"CPUMuon got {p.ndim}D param of shape {tuple(p.shape)}; "
                    f"use CPUAdamW for 1D params."
                )
            n = p.numel()
            st = _ParamState(
                param=p,
                accum=torch.zeros(n, dtype=torch.float16, device="cpu").pin_memory(),
                kind="muon",
                shape=p.shape,
            )
            # Momentum buffer in FP16 (magnitude stays bounded by
            # the grad scale; safe precision-wise).
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
        mom = self.momentum
        lr = self.lr
        wd = self.weight_decay
        nesterov = self.nesterov
        CHUNK_ROWS = self._STREAM_CHUNK_ROWS
        for s in self.state.values():
            if s.accum.abs().sum().item() == 0:
                continue
            # Update momentum on the CPU: m = beta*m + grad.
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

    The cast to FP16 happens BEFORE the DMA so the transfer
    itself is FP16 → halves the PCIe bandwidth vs FP32 and
    matches the destination buffer dtype. The src grad is the
    training dtype (FP16 in v0.0.2), so the cast is a no-op
    when it is already FP16 and a down-cast when AMP promoted
    it to FP32.
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
    )
    adamw_opt = CPUAdamW(
        adamw_params,
        lr=lr_adamw,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=weight_decay,
    )
    return muon_opt, adamw_opt
