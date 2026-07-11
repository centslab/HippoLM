"""Per-mb grad streaming from GPU to CPU.

This module owns the **post-backward** side of
:mod:`src.training.param_offload`: it moves each trainable
parameter's ``.grad`` from the GPU to the matching CPU
accumulator and frees the GPU copy.

Two paths are supported:

  * **Streaming (canonical training path)** — a per-parameter
    post-accumulate-grad hook registered by
    :func:`register_grad_offload_hooks` fires *during* the
    autograd backward and ships each ``.grad`` to the CPU as
    soon as it materialises. The peak GPU grad memory is
    bounded to "at most one param's grad + cast-buffer
    transient" rather than ``sum(grads)``. The accumulated
    entries are dispatched once per microbatch by
    :func:`flush_pending_grads`.

  * **Manual / post-hoc (test paths)** — :func:`accumulate_grads_to_cpu`
    iterates over the optimizer state after ``backward()``
    has returned and copies all grads in one pass. Used by
    callers that have not installed the streaming hooks
    (unit tests that set ``.grad`` directly, etc.).

The module-level state for the streaming queue lives here:

  * :data:`_pending_grads` — list of ``(op, target, src, s)``
    tuples awaiting flush.
  * :data:`_manual_flush_param_ids` — set of ``id(p)`` for params
    that the manual-flush path must handle separately (typically
    tied embeddings whose grad arrives via a custom autograd
    Function that fires *after* the natural-path
    ``accumulate_grad_``; clearing ``p.grad`` in the per-param
    hook would split the tied grad across GPU + CPU).

Three entry points are exposed to the training loop:

  * :func:`register_grad_offload_hooks` — install the streaming
    hooks. Called once per worker after :func:`build_param_groups`.
  * :func:`flush_pending_grads` — sync CUDA and dispatch the
    in-flight entries; called once per microbatch.
  * :func:`flush_manual_flush_params` — copy the manual-flush
    params' grads; called alongside :func:`flush_pending_grads`
    so the tied grad is folded in.

:param_offload.zero_cpu_grad_accum` resets the per-param
    accumulators at cycle start (defensive; the optimizers
    reset their own accumulators inside ``step()``).
"""
from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn

from ._state import _ParamState, _accumulator_target


# Module-level queue for in-flight D2H transfers issued by the
# hooks. Each entry is ``("add", target_tensor, src_cpu)`` —
# ``src_cpu`` is added in place into ``target_tensor``
# (``s.m`` for AdamW, ``s.accum`` for quantized muon, or
# ``s.mom_buf`` for fp* muon). The previous dequant-add-
# requant cycles for int8 / mxfp8 muon ran on every microbatch
# (the per-mb CPU bottleneck at 8+ seconds per mb); they have
# moved to :meth:`CPUMuon.step` (one requantize per param per
# step, amortized over the accumulation cycle). See
# :attr:`_ParamState.accum` for the full design rationale.
# :func:`flush_pending_grads` syncs CUDA and applies the
# accumulated operations.
_pending_grads: list = []

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

    The cast target follows the accumulator's storage dtype
    (``s.m.dtype`` for AdamW, ``s.mom_buf.dtype`` for fp* Muon,
    BF16 for int8 Muon). See :func:`_accumulator_target` for the
    resolution rules.

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
            # NVFP4 mode-3: no leaf Parameter exists, so no hook
            # can be registered. The post-backward path consumes
            # ``module._latest_grad_w`` directly via
            # :func:`accumulate_grads_to_cpu` (see the matching
            # block in that function for the dispatch logic). The
            # training loop must call
            # :func:`accumulate_grads_to_cpu` (or
            # :func:`flush_pending_grads`) after every microbatch
            # backward — the per-param hook path is irrelevant
            # for these modules.
            if s.nvfp4_module is not None:
                s._optimizer = opt
                continue
            p = s.param
            if id(p) in seen:
                continue
            seen.add(id(p))
            if id(p) in _manual_flush_param_ids:
                # Skip the per-param hook; flush_pending_grads
                # handles this param via the manual path.
                continue
            # Attach the optimizer to the state object so the
            # mxfp8 flush dispatch can reach the helper without
            # a separate id→state map.
            s._optimizer = opt
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

    For int8 muon the target is the bf16 ``s.accum`` buffer; the
    per-mb work is a cheap CPU bf16 add (no dequant-add-requant
    cycle). The expensive quantization happens once per optimizer
    step inside :meth:`CPUMuon.step`.

    For mxfp8 muon the target is ``s.mom_buf`` (the cycle-sum is
    the mxfp8 storage itself), but the flush calls the fused C++
    :func:`_mxfp8_apply_grad` instead of a plain ``.add_()`` —
    that op is what saves the 2 bytes/elt of an extra accum
    buffer and is the entire point of the mxfp8 storage format.
    """
    p = s.param
    target, target_dtype, is_mxfp8 = _accumulator_target(s)

    def hook(g: torch.Tensor | None) -> None:
        # ``g`` is the leaf param per the PyTorch API contract.
        # We need the grad, which is on ``p.grad`` at this
        # point. (See the docstring above for the why.)
        grad = p.grad
        if grad is None:
            # The param had no grad this backward (e.g. it was
            # in a no-grad branch). Nothing to offload.
            return None
        # Cast on the GPU (so the DMA carries the cast dtype,
        # not the param's training dtype), flatten to match the
        # 1-D accumulator shape, then issue the async D2H. The
        # .detach() prevents the resulting CPU tensor from
        # carrying autograd metadata (we never want to backprop
        # through a grad-DMA).
        src = (
            grad.detach()
            .to(target_dtype)
            .reshape(-1)
            .to("cpu", non_blocking=True)
        )
        op = "mxfp8_accum" if is_mxfp8 else "add"
        _pending_grads.append((op, target, src, s))
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
    pending: list = []
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
            target, cast_dtype, is_mxfp8 = _accumulator_target(s)
            src_cpu = (
                g.detach()
                .to(cast_dtype)
                .reshape(-1)
                .to("cpu", non_blocking=True)
            )
            op = "mxfp8_accum" if is_mxfp8 else "add"
            pending.append((op, target, src_cpu, s))
            p.grad = None
    if not pending:
        return
    torch.cuda.synchronize()
    # Dispatch by op tag. mxfp8 entries go through the fused
    # kernel via ``CPUMuon._mxfp8_apply_grad`` (saves the 2
    # bytes/elt of a separate bf16 ``accum``). int8 / fp* /
    # AdamW are plain ``.add_()``.
    mxfp8_pending = []
    plain_adds = []
    for entry in pending:
        if entry[0] == "mxfp8_accum":
            mxfp8_pending.append(entry)
        else:
            plain_adds.append((entry[1], entry[2]))
    for target, src in plain_adds:
        target.add_(src)
    if mxfp8_pending:
        for opt in optimizers:
            for s in opt.state.values():
                s._optimizer = opt
                for entry in mxfp8_pending:
                    if entry[1] is s.mom_buf:
                        opt._mxfp8_apply_grad(s, entry[2])
                        break


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
    # Entries come in two flavors:
    #   - ``("add", target, src, s)`` for AdamW / fp* Muon /
    #     int8 Muon (the cheap ``.add_()``).
    #   - ``("mxfp8_accum", mom_buf, src_bf16, s)`` for mxfp8
    #     Muon (dispatched through the fused C++ kernel via
    #     ``CPUMuon._mxfp8_apply_grad``; this op is the raison
    #     d'être of the mxfp8 storage format — saves the 2
    #     bytes/elt of a separate bf16 ``accum`` buffer).
    # Dispatch by the op tag; fall through to ``.add_()`` for
    # the plain add case.
    plain_adds = []
    mxfp8_apps = []
    for entry in _pending_grads:
        op = entry[0]
        if op == "mxfp8_accum":
            mxfp8_apps.append(entry)
        else:
            _, target, src, _ = entry
            plain_adds.append((target, src))
    for target, src in plain_adds:
        target.add_(src)
    if mxfp8_apps:
        for entry in mxfp8_apps:
            _, _, src, s = entry
            opt = getattr(s, "_optimizer", None)
            if opt is None:
                # Defensive: no optimizer attached (e.g. test
                # path that bypassed register_grad_offload_hooks).
                # Fall back to plain bf16 add; the next step
                # will dequant+requant the (wrong) bf16 storage
                # as int8 instead of mxfp8. Logged once via the
                # assertion.
                raise RuntimeError(
                    "mxfp8 flush: param's _ParamState has no "
                    "_optimizer attached; was register_grad_offload_hooks "
                    "called?"
                )
            opt._mxfp8_apply_grad(s, src)
    _pending_grads.clear()


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

    The accumulator is ``s.m`` (AdamW), ``s.accum`` (quantized
    muon: int8 / mxfp8), or ``s.mom_buf`` (fp* muon). The cast
    target follows the accumulator's dtype — see
    :func:`_accumulator_target`. For quantized muon the cast
    target is BF16 and the per-mb path is a cheap CPU bf16 add
    into ``s.accum`` (no dequant-add-requant per mb; the
    requantize happens once per optimizer step inside
    :meth:`CPUMuon.step`).

    The async ``.to("cpu")`` is issued first (with a flattened
    view of the grad, so the resulting ``src_cpu`` matches the
    1-D accumulator tensor), then we sync the current CUDA device
    (or ``sync_device`` if given) before performing the in-place
    CPU add. This avoids the race where the accumulator add runs
    on the CPU before the DMA copy populates ``src``.

    The cast happens BEFORE the DMA so the transfer itself
    matches the cast dtype.
    """
    pending: list = []  # (target, src) — always a plain add now
    for opt in optimizers:
        for s in opt.state.values():
            # NVFP4 mode-3: there is no Parameter.grad (no leaf in the
            # autograd graph). The autograd Function stashed grad_w on
            # ``module._latest_grad_w`` at backward time. We D2H that
            # tensor instead and add it into the right accumulator.
            if s.nvfp4_module is not None:
                module = s.nvfp4_module
                grad_w = module._latest_grad_w
                if grad_w is None:
                    continue
                # Muon mode-3 — the accumulator target follows the
                # same rule as legacy Muon: int8 (non-mxfp8) uses a
                # separate bf16 ``s.accum``; mxfp8 and bf16/fp32
                # storage use the merged-accumulator design
                # (``s.mom_buf``). The cast target matches the
                # accumulator's dtype.
                if s.kind == "muon_nvfp4":
                    if s.accum is not None:
                        # int8 muon: accumulate into bf16 ``accum``,
                        # the cheap CPU path that avoids the
                        # dequant-add-requant cycle per mb.
                        target = s.accum
                        cast_dtype = target.dtype
                    else:
                        # mxfp8 or fp* muon: merged accumulator on
                        # ``s.mom_buf``. The fused per-mb kernel
                        # path (mxfp8) is wired up in the legacy
                        # code via the ``s.param.grad`` route; for
                        # mode-3 we land here too (just a plain
                        # bf16→storage-dtype add — the requantize
                        # happens once at step end).
                        target = s.mom_buf
                        cast_dtype = target.dtype
                    src_cpu = (
                        grad_w.detach()
                        .to(cast_dtype)
                        .reshape(-1)
                        .to("cpu", non_blocking=True)
                    )
                    pending.append(("add", target, src_cpu))
                else:
                    # AdamW mode-3 — same as before.
                    target = s.m
                    src_cpu = (
                        grad_w.detach()
                        .to(target.dtype)
                        .reshape(-1)
                        .to("cpu", non_blocking=True)
                    )
                    pending.append(("add", target, src_cpu))
                module._latest_grad_w = None  # consumed
                continue
            g = s.param.grad
            if g is None:
                continue
            target, cast_dtype, is_mxfp8 = _accumulator_target(s)
            # Cast on the GPU first (so DMA carries the cast
            # dtype), flatten to match the 1-D accumulator, then
            # issue the async D2H.
            src_cpu = (
                g.detach()
                .to(cast_dtype)
                .reshape(-1)
                .to("cpu", non_blocking=True)
            )
            # Mark the queue entry with the op kind so the flush
            # can dispatch mxfp8 entries through the fused kernel
            # instead of a plain ``.add_()``.
            op = "mxfp8_accum" if is_mxfp8 else "add"
            pending.append((op, target, src_cpu))
            # Free GPU grad immediately so the GPU memory is
            # available for the next micro-batch's forward.
            s.param.grad = None

    if sync_device is not None:
        torch.cuda.synchronize(sync_device)
    else:
        torch.cuda.synchronize()

    if not pending:
        return
    # Group entries by op kind; needs the optimizer to dispatch
    # mxfp8 through ``CPUMuon._mxfp8_apply_grad``.
    mxfp8_pending = []
    add_pending = []
    for entry in pending:
        op = entry[0]
        if op == "mxfp8_accum":
            mxfp8_pending.append(entry)
        else:
            add_pending.append((entry[1], entry[2]))
    for target, src in add_pending:
        target.add_(src)
    if mxfp8_pending:
        for opt in optimizers:
            for s in opt.state.values():
                s._optimizer = opt  # belt-and-suspenders for the manual path
                for entry in mxfp8_pending:
                    if entry[1] is s.mom_buf:
                        opt._mxfp8_apply_grad(s, entry[2])
                        break


def zero_cpu_grad_accum(optimizers: List) -> None:
    """Reset CPU accumulators (call this AFTER step() to start the
    next accumulation cycle).

    For AdamW, zero ``s.m`` (the merged accumulator + first
    moment). For Muon, zero ``s.accum`` when set (int8 muon
    only; mxfp8 muon no longer has a separate accum), or zero
    ``s.mom_buf`` for the merged-accumulator design (fp* muon
    AND mxfp8 muon). The optimizer's own ``step()`` already
    zeros the accumulator it consumes; this function is a
    defensive zero for the case where ``found_inf`` is detected
    and the optimizers are NOT stepped — we still want to clear
    the accumulated grads before the next cycle.
    """
    for opt in optimizers:
        for s in opt.state.values():
            if s.kind == "adamw":
                s.m.zero_()
            elif s.accum is not None:
                # int8 muon: zero the separate bf16 accumulator.
                # ``mom_buf`` is irrelevant for next-cycle
                # accumulation (it gets requantized from
                # ``accum`` at step end of the next cycle), so
                # we leave it alone.
                s.accum.zero_()
            else:
                # fp* muon AND mxfp8 muon: zero ``mom_buf``
                # (merged accumulator). For mxfp8, the kernel
                # writes both mom_buf and mom_scale every mb,
                # so leaving mom_scale from the prior cycle is
                # harmless (the kernel overwrites it).
                # int8 zeros the int8 storage without touching
                # the scale; the scale was last set by the
                # requant on the most recent cycle's step,
                # which is the right starting point for the
                # next cycle's accumulation.
                s.mom_buf.zero_()