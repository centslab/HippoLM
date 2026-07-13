"""AdamW with CPU-offloaded state.

:class:`CPUAdamW` is one half of :mod:`src.training.param_offload`'s
optimizer pair. It mirrors the per-param design of the rest of
the package (state lives on CPU pinned memory; per-mb grads are
streamed to CPU via the post-accumulate-grad hook in
:mod:`.offload`). The split file layout keeps the optimizer
itself here so the algorithmic code is in one place without the
``Muon`` / offload plumbing interleaving it.

Implementation notes are kept on the class docstring; see
:mod:`src.training.param_offload` for the package-level design
rationale (storage layouts, the merged-accumulator trick, etc.).
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from ..precision_config import PrecisionConfig
from ._state import _ParamState


class CPUAdamW:
    """AdamW with CPU-side m (BF16) and v (BF16) state, GPU-side
    params. ``m`` doubles as the grad accumulator (mu=1
    accumulation — each microbatch's grad is added to ``m``
    in place, no cross-step β1 EMA).

    Memory layout per trainable param:
        CPU pinned: m (BF16, numel), v (BF16, numel)
        GPU:         param (training dtype, e.g. FP16), .grad (transient)

    No separate accumulator buffer: ``m`` IS the accumulator.
    On :meth:`step`:
        1. ``m`` holds the sum of microbatch grads (raw, no β1
           EMA across steps).
        2. Update ``v`` in place (BF16): ``v = β2*v + (1-β2)*m²``.
        3. Compute the update factor ``m / (sqrt(v) + eps)`` in
           FP32 (to recover the mantissa precision lost to the
           BF16 v sqrt), then cast to FP16 (chunked, streamed
           to the GPU).
        4. Apply ``p -= lr * factor`` in place on the GPU.
        5. Reset ``m`` to zero for the next accumulation cycle.

    The training loop is expected to fold each micro-batch's
    ``.grad`` into ``m`` via the post-accumulate-grad hook
    (see :func:`register_grad_offload_hooks`) or the manual
    :func:`accumulate_grads_to_cpu`. :meth:`step` then sees
    the accumulated grad already on CPU in ``m``.
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

        - ``s.m``       (1st moment + grad accumulator) ← ``precision.adamw_m``
        - ``s.exp_avg_sq`` (2nd moment)                 ← ``precision.adamw_v``

        The grad accumulator dtype is implicit in ``adamw_m``
        (m doubles as the accumulator; we don't have a separate
        ``gradients`` accumulator anymore — saving 2 bytes/elt
        per param). The cast target in the offload hook is
        ``s.m.dtype`` (typically BF16).

        Defaults (when ``precision`` is ``None``) match the canonical
        yml: BF16 for both. The :meth:`step` algorithm is
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
        m_dtype = precision.adamw_m.dtype.to_torch()
        v_dtype = precision.adamw_v.dtype.to_torch()

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
                exp_avg_sq=torch.zeros(n, dtype=v_dtype, device="cpu").pin_memory(),
                m=torch.zeros(n, dtype=m_dtype, device="cpu").pin_memory(),
                kind="adamw",
                shape=p.shape,
            )

    def register_nvfp4_module(self, module) -> None:
        """Register an ``NVFP4*Linear`` constructed with
        ``no_bf16_master=True`` so its FP4 packed buffers get AdamW
        updates.

        The module replaces the role of ``self.weight``:
            - persistent state lives in ``module.packed_weight`` /
              ``module.scales`` / ``module.global_scale``
            - autograd's bwd stashes ``grad_w`` on
              ``module._latest_grad_w`` (consumed per-step)
            - the optimizer computes its update using the same CPU
              pinned m / v tensors it would for a Parameter-based
              param, then routes the apply through
              ``module.apply_chunk_update(start, end, grad_chunk, lr)``

        CPU-pinned m / v are stored as BF16 (matches the legacy
        CPUAdamW default precision). The user's prior sweep on
        AdamW's v showed BF16 is fine — v stores squared grads
        (1e-3 to 1e3 typical magnitude, well inside BF16's 8-bit
        exponent range) and the FP32 promotion at sqrt time in
        step() recovers all mantissa precision we need. Pinned
        BF16 means half the CPU bandwidth on these D2H paths vs
        FP32 (the default FP32 I had set initially was overly
        conservative and unnecessarily expensive).
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
        key = id(module)
        if key in self.state:
            return
        # Match the precision the canonical yml uses for CPUAdamW
        # (BF16 for both m and v) — do NOT pick up a per-config
        # quantization here; the user's note was specifically that
        # "AdamW 的 v 只是吃动态范围，精度不是那么敏感" so we just
        # use the safe BF16 default that legacy AdamW uses.
        self.state[key] = _ParamState(
            param=None,
            exp_avg_sq=torch.zeros(n, dtype=torch.bfloat16, device="cpu").pin_memory(),
            m=torch.zeros(n, dtype=torch.bfloat16, device="cpu").pin_memory(),
            kind="adamw_nvfp4",
            shape=(rows, cols),
            nvfp4_module=module,
            nvfp4_n=n,
            nvfp4_chunk_ranges=module.chunk_ranges(),
        )

    def zero_grad(self, set_to_none: bool = True) -> None:
        for s in self.state.values():
            # NVFP4 mode-3: no Param.grad to free. The stashed
            # grad_w on the module is consumed by
            # accumulate_grads_to_cpu instead of via a hook.
            if s.nvfp4_module is not None:
                continue
            s.param.grad = None
            # NOTE: do NOT zero ``m`` here — that would discard
            # the user's gradient accumulation across micro-batches
            # (``m`` doubles as the accumulator). The training
            # loop is responsible for resetting ``m`` at the
            # start of each accumulation cycle (via
            # :func:`zero_cpu_grad_accum`).

    # Per-chunk size for the streaming factor / update path.
    # 4 M elements is the sweet spot on V100 / 5060Ti PCIe: 8 MB
    # FP16 per chunk keeps the GPU transient well under the 16 GB
    # ceiling and overlaps the H2D copy with the next chunk's
    # compute.
    _STREAM_CHUNK_NUMEL = 4 * 1024 * 1024

    def step(self) -> None:
        beta2 = self.beta2
        lr = self.lr
        eps = self.eps
        wd = self.weight_decay
        CHUNK = self._STREAM_CHUNK_NUMEL
        for s in self.state.values():
            g = s.m
            if g.abs().sum().item() == 0:
                # No accumulated grad (shouldn't happen if the
                # training loop is correct, but guard against
                # unnecessary work).
                continue
            s.step += 1
            step = s.step
            bc2 = 1.0 - beta2 ** step
            v = s.exp_avg_sq
            # ``g`` (= s.m) is the sum of microbatch grads.
            # No β1 EMA: in the merged-accumulator design, m IS
            # the accumulator, so its value across steps is the
            # raw sum of one accumulation cycle's grads. β1 was
            # only ever smoothing the cross-step accumulator-to-
            # accumulator transition; in this design the
            # accumulator resets to zero at the end of every
            # step, so β1 smoothing has nothing to smooth.
            #
            # ``addcmul_`` on BF16: BF16 has FP32 exponent range
            # so ``g*g`` (for g ~ 1e-3) is ~1e-6, well above the
            # BF16 smallest normal. The mantissa is 7 bits which
            # is the precision bottleneck — but we recover
            # precision in the per-element factor by promoting
            # to FP32 below before the sqrt.
            v.mul_(beta2).addcmul_(g, g, value=1.0 - beta2)
            # Decoupled weight decay: applied in place on the GPU
            # once, before the streaming factor / apply loop.
            # Folding it into the per-element factor would also
            # work but would require an extra H2D per chunk.
            # For mode-3 (NVFP4) entries, ``s.param is None`` —
            # weight decay is folded into the per-chunk
            # ``module.decay_and_apply_chunk`` call below.
            if wd != 0.0 and s.param is not None:
                s.param.data.mul_(1.0 - lr * wd)
            # Streaming update: chunk the param so the peak GPU
            # memory for the FP32 divisor / FP16 factor transfer
            # is bounded to ``CHUNK`` elements regardless of
            # ``numel``. The chunked path is bit-identical to
            # the un-chunked path because the operations are
            # element-wise.
            #
            # Numerics: do the divisor math in FP32 even though
            # ``g`` and ``v`` are both BF16. Promoting
            # v_chunk to FP32 before the sqrt recovers the
            # 7-bit BF16 mantissa precision for the divisor,
            # which is what ``1/sqrt(v/bc2)`` is most sensitive
            # to (a small relative error in v becomes a large
            # relative error in 1/sqrt(v)). The factor is then
            # cast back to FP16 for the GPU apply — that
            # final cast is the precision bottleneck, not the
            # v promotion.
            #
            # No β1 division on the numerator (g is unbiased —
            # it's the raw sum, not an EMA), but v still has
            # its bias-correction bc2.
            # ---- Mode (3) NVFP4: chunk along the row axis of
            # the module (same granularity budget as the FFN
            # module list — ~8 MiB BF16 peak). The apply routes
            # through ``module.apply_chunk_update`` which
            # materializes a temp BF16 view, applies the update,
            # and commits back to FP4. The optimizer-side math
            # (v update + factor) is identical to the param
            # case — only the apply destination differs.
            # apply_chunk_update(start, end, grad_chunk, lr)
            # does ``bf16 -= lr * grad_chunk`` where grad_chunk
            # here is the AdamW update factor (m / sqrt(v/bc2) +
            # eps), matching the legacy param.add_(factor, alpha=-lr)
            # which does ``param -= lr * factor``.
            if s.nvfp4_module is not None:
                module = s.nvfp4_module
                cols = (
                    module.in_features_per_partition
                    if hasattr(module, "in_features_per_partition")
                    else module.in_features
                )
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
                n_chunks = len(s.nvfp4_chunk_ranges)
                for ci, (r_start, r_end) in enumerate(s.nvfp4_chunk_ranges):
                    flat_start = r_start * cols
                    flat_end = r_end * cols
                    v_chunk_fp32 = v[flat_start:flat_end].float()
                    denom = (v_chunk_fp32 / bc2).sqrt_().add_(eps)
                    factor = g[flat_start:flat_end].float() / denom
                    # ``factor`` is 1-D (flat slice of pinned
                    # gradient accumulator on CPU); reshape to 2-D
                    # and move to the module's device so it matches
                    # ``material_chunk``'s (chunk_rows, K) output
                    # (apply_chunk_update does
                    # ``bf16 -= lr * factor`` element-wise).
                    factor = factor.view(r_end - r_start, cols)
                    factor = factor.to(module.packed_weight.device, non_blocking=True)
                    module.apply_chunk_update(
                        r_start, r_end, factor, lr,
                        rebuild_scales_cache=(ci == n_chunks - 1),
                    )
                s.m.zero_()
                continue
            p_flat = s.param.data.view(-1)
            n = p_flat.numel()
            for start in range(0, n, CHUNK):
                end = min(start + CHUNK, n)
                v_chunk_fp32 = v[start:end].float()
                denom = (v_chunk_fp32 / bc2).sqrt_().add_(eps)  # FP32, chunk
                factor = g[start:end].float() / denom           # FP32, chunk
                # Stream the FP32 factor to the GPU as the param
                # dtype (FP16) and apply in place.
                p_flat[start:end].add_(
                    factor.to(
                        device=s.param.device, dtype=s.param.dtype,
                        non_blocking=True,
                    ),
                    alpha=-lr,
                )
            # Reset m to zero for the next accumulation cycle.
            # ``m`` doubles as the accumulator; this is the only
            # point in the cycle where the accumulation result
            # is consumed.
            s.m.zero_()

    # ------------------------------------------------------------------ #
    # Checkpoint save/load (resume).                                     #
    # ------------------------------------------------------------------ #
    # The training loop periodically calls ``save_checkpoint`` (see
    # :mod:`src.training.checkpoint`) which in turn calls
    # ``opt.state_dict()`` on each optimizer. The CPU-offloaded
    # design does not subclass :class:`torch.optim.Optimizer`, so the
    # stock ``state_dict`` is unavailable — we serialize the per-param
    # state (CPU pinned tensors + scalar metadata) ourselves.
    #
    # Per-param entries are emitted as a list in ``self.state``'s
    # insertion order, which is the order :func:`build_param_groups`
    # added params in (i.e. the model's parameter iteration order).
    # That order is deterministic across save/load, so
    # ``load_state_dict`` can match the i-th saved entry to the
    # i-th current entry without a stable id key (Python ``id(p)``
    # does not survive a process restart). The matching tensors are
    # copied in place — the per-param buffers already exist on the
    # freshly-constructed optimizer, only their contents are
    # restored.
    def state_dict(self) -> dict:
        return {
            "lr": self.lr,
            "beta1": self.beta1,
            "beta2": self.beta2,
            "eps": self.eps,
            "weight_decay": self.weight_decay,
            "state": [
                {
                    "shape": list(s.shape),
                    "step": s.step,
                    "kind": s.kind,
                    "m": s.m,
                    "exp_avg_sq": s.exp_avg_sq,
                }
                for s in self.state.values()
            ],
        }

    def load_state_dict(self, state_dict: dict) -> None:
        self.lr = float(state_dict["lr"])
        self.beta1 = float(state_dict["beta1"])
        self.beta2 = float(state_dict["beta2"])
        self.eps = float(state_dict["eps"])
        self.weight_decay = float(state_dict["weight_decay"])
        saved = state_dict.get("state", [])
        current = list(self.state.values())
        if len(saved) != len(current):
            raise ValueError(
                f"CPUAdamW.load_state_dict: param count mismatch "
                f"(file has {len(saved)} entries, current optimizer "
                f"has {len(current)}). The model architecture likely "
                f"changed since this checkpoint was written."
            )
        for cur, sav in zip(current, saved):
            cur.step = int(sav.get("step", 0))
            # Shape sanity check: the param shapes must match (the
            # current optimizer's buffers were allocated from the
            # current model's params).
            if tuple(sav.get("shape", ())) != tuple(cur.shape):
                raise ValueError(
                    f"CPUAdamW.load_state_dict: shape mismatch at "
                    f"param index {current.index(cur)}: file="
                    f"{tuple(sav.get('shape', ()))} current="
                    f"{tuple(cur.shape)}."
                )
            if sav.get("m") is not None:
                cur.m.copy_(sav["m"])
            if sav.get("exp_avg_sq") is not None:
                cur.exp_avg_sq.copy_(sav["exp_avg_sq"])