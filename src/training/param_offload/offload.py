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

  * :data:`_pending_grads` — list of ``(target, src)`` tuples
    awaiting flush.
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

History
-------
The pre-2026-07-12 design had two queue op tags —
``("add", target, src)`` and ``("mxfp8_accum", mom_buf, src)``
— with the latter routed through a fused C++ kernel for the
mxfp8 storage format. After mxfp8 storage was removed
(2026-07-12; long-training instability), the queue reverts to
plain ``.add_()`` entries: every per-mb grad is folded into
its accumulator with a CPU tensor add. The pre-removal code
lives on the ``archive/int8-mxfp8-muon`` branch.

On 2026-07-15 the "merged-accumulator" / mu=1 trick was
deleted: the per-mb grad now folds into ``s.grad`` (a dedicated
BF16 pinned accumulator) instead of ``s.m`` / ``s.mom_buf``,
which are gone. The optimizer's momentum state (β1 EMA for
AdamW, SGD momentum for Muon) lives in ``s.exp_avg`` and is
updated in place by ``step()``. See
:mod:`src.training.param_offload._state` for the new field
layout and the rationale.
"""
from __future__ import annotations

import queue
import threading
from typing import Any, List, Optional

import torch
import torch.nn as nn

from ._state import _ParamState, _accumulator_target
from .cpu_fused import fused_add_into_many, fused_zero_many


# Module-level queue for in-flight D2H transfers issued by the
# hooks. Each entry is ``(target_tensor, src_cpu)`` — ``src_cpu``
# is added in place into ``target_tensor`` (``s.grad`` for both
# AdamW and Muon, post-2026-07-15; both are floating dtypes, so
# a plain ``.add_()`` is the right op).
#
# In sync mode (no async worker started — tests / manual path)
# :func:`flush_pending_grads` drains this list. In async mode
# (production, via :func:`_start_cpu_add_worker`) the hooks push
# to the worker queue instead, and this list stays empty.
_pending_grads: list = []


# ---- Async CPU-add worker (Priority 2: per-chunk D2H overlap) ----
#
# When started via :func:`_start_cpu_add_worker`, the per-param
# streaming hook and the manual-flush path push ``(event, target,
# src)`` entries into ``_cpu_add_queue`` instead of
# :data:`_pending_grads`. The worker thread pops entries, syncs
# the per-entry CUDA event (waits only for THAT DMA — not for
# unrelated GPU work), then applies the CPU add via the BF16
# fused kernel :func:`fused_add_into_many` or a plain
# ``.add_()`` for non-BF16 dtypes.
#
# Per-entry event sync is the key trick: a global
# ``cuda.synchronize()`` would block the training thread's
# next-chunk fwd until the worker finishes the current chunk's
# adds, defeating the overlap. Per-entry events let the main
# thread launch chunk i+1's fwd immediately while the worker
# drains chunk i's DMA-pending src tensors in the background.
#
# The worker is OPTIONAL — when never started, the legacy sync
# paths in this module work unchanged (the streaming-hook list,
# :func:`flush_pending_grads`, :func:`accumulate_grads_to_cpu`,
# :func:`flush_manual_flush_params`). The training loop starts
# the worker in :func:`_setup_worker` and stops it in
# :func:`_teardown_worker`. Tests that don't start the worker
# continue to use the sync path.
_cpu_add_queue: Optional["queue.Queue[Any]"] = None
_cpu_add_worker_thread: Optional[threading.Thread] = None
_cpu_add_worker_shutdown: Optional[threading.Event] = None
_cpu_add_lock: Optional[threading.Lock] = None
_cpu_add_inflight: int = 0
_cpu_add_drained: Optional[threading.Event] = None
# Sentinel pushed onto the queue to break the worker out of
# its ``queue.get`` block during :func:`_stop_cpu_add_worker`.
_SHUTDOWN_SENTINEL: Any = object()


def _start_cpu_add_worker() -> None:
    """Start the async CPU-add worker thread.

    After this call, the per-param streaming hook and the
    manual-flush / NVFP4 stash paths push
    ``(event, target, src)`` entries to an internal queue
    instead of :data:`_pending_grads`. The worker drains the
    queue in the background, syncs per-entry events, and applies
    the CPU adds.

    Idempotent: a second call is a no-op while the worker is
    already running. Safe to call from any thread.
    """
    global _cpu_add_queue, _cpu_add_worker_thread
    global _cpu_add_worker_shutdown, _cpu_add_lock
    global _cpu_add_inflight, _cpu_add_drained
    if _cpu_add_worker_thread is not None:
        return
    _cpu_add_queue = queue.Queue()
    _cpu_add_worker_shutdown = threading.Event()
    _cpu_add_lock = threading.Lock()
    _cpu_add_drained = threading.Event()
    _cpu_add_drained.set()  # empty at start
    _cpu_add_inflight = 0
    _cpu_add_worker_thread = threading.Thread(
        target=_cpu_add_worker_loop,
        name="cpu-add-worker",
        daemon=True,
    )
    _cpu_add_worker_thread.start()


def _stop_cpu_add_worker(timeout: float = 30.0) -> None:
    """Stop the worker thread.

    First drains the queue (blocks until empty or ``timeout``
    elapses), then pushes the shutdown sentinel and joins the
    thread. Idempotent and safe to call when the worker is not
    running. Safe to call from any thread (only the main thread
    should call this in practice — the training worker per-rank
    thread).

    The drain-then-sentinel ordering avoids the race where a
    shutdown sentinel arrives before a still-pending entry: the
    drain ensures all entries enqueued before this call have
    been processed, and the sentinel only signals "exit" once
    the queue is known-empty.
    """
    global _cpu_add_worker_thread, _cpu_add_queue
    global _cpu_add_worker_shutdown, _cpu_add_lock
    global _cpu_add_drained, _cpu_add_inflight
    if _cpu_add_worker_thread is None:
        return
    # Drain first so any entries enqueued before teardown are
    # fully applied (callers expect accumulators to be
    # consistent when _teardown_worker returns).
    _drain_cpu_add_queue(timeout=timeout)
    # Queue is empty (drain returned). Push sentinel; worker
    # picks it up and exits.
    _cpu_add_queue.put(_SHUTDOWN_SENTINEL)
    _cpu_add_worker_thread.join(timeout=5.0)
    _cpu_add_worker_thread = None
    _cpu_add_queue = None
    _cpu_add_worker_shutdown = None
    _cpu_add_lock = None
    _cpu_add_drained = None
    _cpu_add_inflight = 0


def _enqueue_cpu_add(event: torch.cuda.Event, target: torch.Tensor, src: torch.Tensor) -> None:
    """Push one ``(event, target, src)`` to the worker queue and
    bump the inflight counter. No-op when the worker is not
    running — caller should fall back to appending to
    :data:`_pending_grads` and the sync :func:`flush_pending_grads`.
    """
    global _cpu_add_inflight
    if _cpu_add_queue is None or _cpu_add_lock is None or _cpu_add_drained is None:
        return
    with _cpu_add_lock:
        _cpu_add_inflight += 1
        _cpu_add_drained.clear()
    _cpu_add_queue.put((event, target, src))


def _drain_cpu_add_queue(timeout: Optional[float] = None) -> bool:
    """Block until the worker has applied all queued CPU adds.

    Returns ``True`` if drained, ``False`` if timed out. No-op
    (always returns ``True``) when the worker is not running.
    Used at the end of each training step (before
    :func:`fused_inf_nan_count_bf16` / grad-norm / optimizer
    step) to guarantee the per-chunk async adds have landed
    before the accumulators are read.
    """
    if _cpu_add_drained is None:
        return True
    return _cpu_add_drained.wait(timeout=timeout)


def _cpu_add_async_enabled() -> bool:
    """``True`` iff the async CPU-add worker is running.

    Used by the training loop to decide whether the end-of-step
    :func:`_drain_cpu_add_queue` call is needed (no-op in sync
    mode).
    """
    return _cpu_add_queue is not None


def _cpu_add_worker_loop() -> None:
    """Worker thread body.

    Pops ``(event, target, src)`` entries in batches and applies
    the CPU adds via :func:`_apply_cpu_add_batch`. The outer
    ``get(timeout=0.5)`` lets the worker check the shutdown
    signal during idle periods; the inner ``get_nowait()`` loop
    drains whatever is ready without blocking so each batch
    applies a contiguous slice of the queue in one fused call.

    The shutdown sentinel (``_SHUTDOWN_SENTINEL``) is handled
    in both the outer and inner get paths so the worker exits
    promptly when :func:`_stop_cpu_add_worker` signals it.
    """
    batch: list = []
    while True:
        try:
            item = _cpu_add_queue.get(timeout=0.5)
        except queue.Empty:
            # Idle: flush any half-built batch and re-check
            # shutdown. The 0.5 s timeout bounds the worst-case
            # shutdown latency without busy-waiting.
            if batch:
                _apply_cpu_add_batch(batch)
                _decrement_inflight(len(batch))
                batch = []
            if _cpu_add_worker_shutdown is not None and _cpu_add_worker_shutdown.is_set():
                return
            continue
        if item is _SHUTDOWN_SENTINEL:
            if batch:
                _apply_cpu_add_batch(batch)
                _decrement_inflight(len(batch))
            return
        batch.append(item)
        # Drain whatever else is ready without blocking. Within
        # a single batch, entries are FIFO from the main
        # thread's enqueue order, which is the same as the
        # stream order of their recorded events. This means a
        # single fused call covers the whole batch's worth of
        # CPU adds.
        while True:
            try:
                nxt = _cpu_add_queue.get_nowait()
            except queue.Empty:
                break
            if nxt is _SHUTDOWN_SENTINEL:
                _apply_cpu_add_batch(batch)
                _decrement_inflight(len(batch))
                return
            batch.append(nxt)
        _apply_cpu_add_batch(batch)
        _decrement_inflight(len(batch))
        batch = []


def _apply_cpu_add_batch(batch: list) -> None:
    """Apply a batch of ``(event, target, src)`` entries.

    Syncs each entry's event (per-entry, so a global
    ``cuda.synchronize`` doesn't block the training thread on
    unrelated GPU work), then dispatches the adds: BF16 entries
    go through one :func:`fused_add_into_many` call (one OMP
    parallel-for across all BF16 entries — DeepSpeed
    CPUAdam-style); non-BF16 entries fall back to per-tensor
    ``.add_()``.

    Per-entry event sync is cheap: events are recorded on the
    same default stream in chronological order, so the only
    event that actually blocks is the first not-yet-fired event
    in the batch; the rest return immediately because the
    stream's event-firing chain has already covered them.
    """
    if not batch:
        return
    # Sync events in order. Within a single chunk's entries
    # (enqueued back-to-back during backward) all events share
    # a contiguous slice of the stream, so syncing each one in
    # sequence is effectively one wait at the start of the batch.
    for event, _t, _s in batch:
        event.synchronize()
    # Partition by dtype. fused_add_into_many is BF16-only: the
    # promote-add-narrow inner kernel hard-codes BF16 bit layout
    # (sign:1, exp:8, mantissa:7). FP16 / FP32 dtypes would
    # silently corrupt data — see cpu_fused.py's
    # fused_add_into_many BF16-only assertion. Mixed-dtype
    # batches route BF16 through the fused path and the rest
    # through plain ``.add_()``.
    bf16_tgts: list = []
    bf16_srcs: list = []
    other_pairs: list = []
    for _e, target, src in batch:
        if target.dtype == torch.bfloat16 and src.dtype == torch.bfloat16:
            bf16_tgts.append(target)
            bf16_srcs.append(src)
        else:
            other_pairs.append((target, src))
    if bf16_tgts:
        fused_add_into_many(bf16_tgts, bf16_srcs)
    for target, src in other_pairs:
        target.add_(src)


def _decrement_inflight(n: int) -> None:
    """Decrement the inflight counter; signal ``_cpu_add_drained``
    when it reaches zero so :func:`_drain_cpu_add_queue` wakes
    up the training thread.
    """
    if _cpu_add_lock is None or _cpu_add_drained is None:
        return
    with _cpu_add_lock:
        global _cpu_add_inflight
        _cpu_add_inflight -= n
        if _cpu_add_inflight <= 0:
            _cpu_add_inflight = 0
            _cpu_add_drained.set()

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


# --------------------------------------------------------------------------- #
# Offload callback for NVFP4 mode-3 (zero-copy from backward into worker).     #
# --------------------------------------------------------------------------- #
# See :meth:`NVFP4Linear._stash_grad_w`. The optimizer (CPUMuon /
# CPUAdamW) installs a closure built here on the module, so the
# BF16 grad_w → cast → D2H → event-record → worker-enqueue pipeline
# runs INSIDE the autograd backward call instead of waiting for
# the post-backward :func:`accumulate_grads_to_cpu` sweep.
# Cost on the dev shape: ~22ms/chunk (was the dominant per-chunk
# GPU compute-stream idle source at the chunk boundary — see
# auto-memory ``project_chunk_boundary_bubble_source.md``).
def _make_nvfp4_offload_cb(
    module: Any,
    target: torch.Tensor,
    cast_dtype: torch.dtype,
):
    """Build a closure that does cast + D2H + worker-enqueue for one
    NVFP4 module. Called from the custom autograd Function's
    backward via :meth:`NVFP4Linear._stash_grad_w`.

    The closure is single-purpose: it captures ``target`` (the
    CPU accumulator, ``s.grad`` for both AdamW and Muon
    post-2026-07-15) and the cast dtype, and pulls ``grad_w``
    from the backward call.

    Idempotent w.r.t. the async worker: each call enqueues one
    entry; the worker drains in FIFO order with per-entry
    ``event.wait()``.

    Worker-status aware: if the async worker isn't running
    (tests, unit-test paths without :func:`_start_cpu_add_worker`),
    the closure falls back to stashing ``grad_w`` on
    ``module._latest_grad_w`` so the legacy
    :func:`accumulate_grads_to_cpu` path picks it up
    post-backward. This keeps the unit tests that directly call
    ``_stash_grad_w`` working unchanged.
    """
    def _cb(grad_w: torch.Tensor) -> None:
        if _cpu_add_queue is None:
            # Sync mode (no async worker — tests / manual
            # path): stash on module for the post-backward
            # sweep. The legacy ``accumulate_grads_to_cpu`` will
            # read ``module._latest_grad_w`` and do the D2H + add.
            module._latest_grad_w = grad_w.detach()
            return
        # Async mode (production): cast on the GPU so the DMA
        # carries the cast dtype, then flatten to match the 1-D
        # accumulator, then issue the async D2H. The .detach()
        # drops autograd metadata — we never want to backprop
        # through a grad-DMA.
        src = (
            grad_w.detach()
            .to(cast_dtype)
            .reshape(-1)
            .to("cpu", non_blocking=True)
        )
        # Record an event on the current stream AFTER the DMA is
        # enqueued. The worker uses this event to wait only for
        # THIS src's transfer to land, not for unrelated GPU work.
        # This is what enables the per-chunk overlap with the next
        # chunk's fwd — a global ``cuda.synchronize()`` would
        # block the main thread on the worker's batch and defeat
        # the overlap.
        event = torch.cuda.current_stream().record_event()
        _enqueue_cpu_add(event, target, src)
    return _cb


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
    (``s.grad.dtype`` for both AdamW and Muon,
    post-2026-07-15). See :func:`_accumulator_target` for the
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
                continue
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
    target, target_dtype = _accumulator_target(s)

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
        if _cpu_add_queue is not None:
            # Async mode: record an event on the current stream
            # AFTER the DMA is enqueued. The worker uses this
            # event to wait only for THIS src's transfer to
            # land, not for unrelated GPU work. This is what
            # enables the per-chunk overlap with the next
            # chunk's fwd — a global ``cuda.synchronize()``
            # would block the main thread on the worker's
            # batch and defeat the overlap.
            event = torch.cuda.current_stream().record_event()
            _enqueue_cpu_add(event, target, src)
        else:
            # Sync mode (legacy): append to the module-level
            # list. Drained by :func:`flush_pending_grads` at
            # the end of each microbatch.
            _pending_grads.append((target, src))
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
    has_async = _cpu_add_queue is not None
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
            target, cast_dtype = _accumulator_target(s)
            src_cpu = (
                g.detach()
                .to(cast_dtype)
                .reshape(-1)
                .to("cpu", non_blocking=True)
            )
            if has_async:
                # Async mode: record event + push to worker
                # queue. Worker drains in the background; the
                # main thread returns immediately and proceeds
                # to the next chunk's fwd.
                event = torch.cuda.current_stream().record_event()
                _enqueue_cpu_add(event, target, src_cpu)
            else:
                pending.append((target, src_cpu))
            p.grad = None
    if has_async:
        return
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

    In async mode (the production path with the CPU-add worker
    started via :func:`_start_cpu_add_worker`), this function is
    a no-op: the hooks push to the worker queue instead of
    :data:`_pending_grads`, and the worker drains them in the
    background. The end-of-step :func:`_drain_cpu_add_queue`
    call joins the worker before the optimizer step.
    """
    if _cpu_add_queue is not None:
        # Async mode: hooks push to the worker queue, not the
        # legacy list. Nothing to drain here.
        return
    if not _pending_grads:
        return
    if sync_device is not None:
        torch.cuda.synchronize(sync_device)
    else:
        torch.cuda.synchronize()
    # All entries are now plain ``.add_()`` (the
    # mxfp8-fused-kernel path was removed with mxfp8 storage
    # in 2026-07-12).
    for target, src in _pending_grads:
        target.add_(src)
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

    The accumulator is ``s.grad`` for both AdamW and Muon
    (post-2026-07-15; the previous "merged-accumulator" layout
    where ``s.m`` / ``s.mom_buf`` doubled as the accumulator was
    deleted). The cast target follows the accumulator's dtype —
    see :func:`_accumulator_target`.

    The async ``.to("cpu")`` is issued first (with a flattened
    view of the grad, so the resulting ``src_cpu`` matches the
    1-D accumulator tensor), then we sync the current CUDA device
    (or ``sync_device`` if given) before performing the in-place
    CPU add. This avoids the race where the accumulator add runs
    on the CPU before the DMA copy populates ``src``.

    The cast happens BEFORE the DMA so the transfer itself
    matches the cast dtype.
    """
    pending: list = []
    has_async = _cpu_add_queue is not None
    for opt in optimizers:
        for s in opt.state.values():
            # NVFP4 mode-3: there is no Parameter.grad (no leaf in the
            # autograd graph). Two paths exist:
            #
            # 1. Callback path (production, post-2026-07-15): the
            #    optimizer's :meth:`register_nvfp4_module` installed
            #    ``module._nvfp4_offload_cb``. The custom autograd
            #    Function fired the callback DURING ``backward()``
            #    so the D2H + worker enqueue is already in flight
            #    (or done) by the time we get here. We just skip —
            #    UNLESS the callback fell back to stash (no async
            #    worker running — test paths), in which case the
            #    legacy path picks it up via ``_latest_grad_w``.
            # 2. Legacy stash path (no callback installed, e.g.
            #    older code paths or unit tests that explicitly
            #    clear ``_nvfp4_offload_cb``): ``_stash_grad_w``
            #    stashed grad_w on ``module._latest_grad_w``. We
            #    do the D2H + enqueue here, same as before.
            if s.nvfp4_module is not None:
                module = s.nvfp4_module
                grad_w = module._latest_grad_w
                if grad_w is None:
                    # Callback already did the D2H + enqueue
                    # inside backward() — nothing to do.
                    continue
                # grad_w is stashed (either by legacy path or
                # by the callback's worker-not-running fallback).
                # Process it here.
                # Post-2026-07-15 explicit-accumulator layout:
                # a single ``.add_()`` folds the microbatch grad
                # into ``s.grad`` (both AdamW and Muon — the
                # "merged-accumulator" design where ``s.m`` /
                # ``s.mom_buf`` doubled as the accumulator was
                # deleted; ``s.exp_avg`` is the optimizer's
                # momentum state, updated in place by ``step()``).
                target, cast_dtype = _accumulator_target(s)
                src_cpu = (
                    grad_w.detach()
                    .to(cast_dtype)
                    .reshape(-1)
                    .to("cpu", non_blocking=True)
                )
                if has_async:
                    event = torch.cuda.current_stream().record_event()
                    _enqueue_cpu_add(event, target, src_cpu)
                else:
                    pending.append((target, src_cpu))
                module._latest_grad_w = None  # consumed
                continue
            g = s.param.grad
            if g is None:
                continue
            target, cast_dtype = _accumulator_target(s)
            # Cast on the GPU first (so DMA carries the cast
            # dtype), flatten to match the 1-D accumulator, then
            # issue the async D2H.
            src_cpu = (
                g.detach()
                .to(cast_dtype)
                .reshape(-1)
                .to("cpu", non_blocking=True)
            )
            if has_async:
                event = torch.cuda.current_stream().record_event()
                _enqueue_cpu_add(event, target, src_cpu)
            else:
                pending.append((target, src_cpu))
            # Free GPU grad immediately so the GPU memory is
            # available for the next micro-batch's forward.
            s.param.grad = None

    if has_async:
        # Async mode: worker drains entries in background. The
        # main thread returns immediately so the next chunk's
        # fwd can overlap with the worker's CPU adds.
        return

    if sync_device is not None:
        torch.cuda.synchronize(sync_device)
    else:
        torch.cuda.synchronize()

    if not pending:
        return
    for target, src in pending:
        target.add_(src)


def zero_cpu_grad_accum(optimizers: List) -> None:
    """Reset the per-step CPU grad accumulators (call this AFTER
    step() to start the next accumulation cycle, or when
    ``found_inf`` is detected and the optimizers are NOT
    stepped — we still want to clear the accumulated grads
    before the next cycle).

    For both AdamW and Muon, zero ``s.grad`` (the per-step
    accumulator). The optimizer's own ``step()`` already zeros
    ``s.grad`` for the entries it consumes; this function is a
    defensive sweep for the cycle that wasn't stepped.

    All collected ``s.grad`` tensors are zeroed in ONE fused
    OpenMP call (DeepSpeed CPUAdam-style) instead of N separate
    Python ``.zero_()`` calls — same win as :mod:`.cpu_fused`
    brings to v4's per-layer worker and to
    :meth:`CPUMuon.step`'s end-of-cycle housekeeping.

    Note: this function does NOT touch ``s.exp_avg`` /
    ``s.exp_avg_sq`` — those are the optimizer's EMA / momentum
    state and are preserved across steps. Only the per-step
    ``s.grad`` accumulator is reset here.
    """
    bufs: list = []
    for opt in optimizers:
        for s in opt.state.values():
            # Post-2026-07-15 explicit-accumulator layout:
            # ``s.grad`` is the per-step accumulator for both
            # AdamW and Muon. The optimizer's momentum state
            # (``s.exp_avg`` / ``s.exp_avg_sq``) is NOT reset
            # here — only ``s.grad``.
            bufs.append(s.grad)
    fused_zero_many(bufs)
