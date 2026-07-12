"""Per-layer GPU grad accumulation + worker-thread CPU add.

This module is an OPT-IN alternative to the default CPU-add
streaming path (:mod:`src.training.param_offload.offload`).
Instead of one CPU pinned-source + per-mb ``cuda.synchronize`` +
CPU ``add_`` per param, we accumulate per-param grads on the GPU
into a SHARED ``gpu_buf`` of size = max layer grad, then issue one
async D2H per layer to a per-layer CPU pinned slot, and let a
worker thread do the per-param CPU ``add_`` into ``s.m`` /
``s.accum`` / ``s.mom_buf`` in parallel with the main thread's
continued forward / backward.

Design constraints (see ``auto-memory/project_gpu_offload_v4.md``
for the A/B numbers):

  - ONE shared GPU buffer of size = max layer grad in BF16. Reused
    for every layer, every microbatch. Reset to zero after each
    layer's D2H so the next layer's bwd starts clean.
  - ``register_full_backward_hook`` per ``TPHippoLayer`` fires
    AFTER the layer's bwd has populated every ``.grad``. By then
    all submodule bwds have completed (eager mode).
  - ``p.grad = None`` inside the hook releases the per-param
    ``.grad`` so the GPU caching allocator can reuse the block.
    In production, ``empty_cache_between_mb`` (default True)
    returns the blocks to the OS between mbs, keeping VRAM at
    "model + gpu_buf + mf_buf" between microbatches.
  - ``register_post_accumulate_grad_hook`` per leaf param handles
    the 4 manual-flush params (tied embed, top-level norm,
    ``attn_res.query``, ``attn_res.norm``) that live OUTSIDE any
    ``TPHippoLayer``. These fire DURING autograd, no sync needed.
  - Worker thread drains a queue of ``(kind, info, slot, event)``
    tuples; ``event.synchronize()`` blocks until the GPU's D2H
    finishes, then ``target.add_(pinned_slot[off:off+n])`` folds
    the slice into the per-param accumulator.

VRAM (245M test scale, 5060 Ti 16G):

  - Persistent: ``gpu_buf`` (max layer grad, ~55 MiB BF16) +
    ``mf_buf`` (max MF param grad, ~24 MiB BF16) = ~80 MiB.
  - Peak during bwd: ``~1 layer's .grad`` alive at the per-layer
    hook fire time = +55 MiB over v1. The full 500 MiB peak
    delta reported in earlier prototype numbers was from the
    test's caching-allocator carry-over across the 4 sequential
    modes; production (single model + per-mb empty_cache) keeps
    VRAM at "model + ~80 MiB buffer" between mbs.

Step time wins (245M test scale, 5060 Ti 16G):

  - 33–44% over v1 at every MBS measured (256, 512, 1024, 2048,
    4096). Never loses to v1.
  - Beats v2 at every MBS — v2 LOSES at MBS ≥ 1024 because its
    single per-step D2H blocks the optimizer step. v4's per-mb
    per-layer D2Hs amortize into the per-mb bwd tail.

Out of scope:

  - mxfp8 muon (needs a fused C++ add kernel into E4M3 + E8M0
    pair; the GPU ``add_`` on E4M3 silently breaks). mxfp8
    params are skipped by ``setup_per_layer_gpu_accum`` and
    left to the manual flush path.
  - NVFP4 mode-3 (no leaf param; the FP4 packed buffers live on
    the module, not as a Parameter). NVFP4 entries are also
    skipped and fall through to ``flush_manual_flush_params``.
"""
from __future__ import annotations

import gc
import queue
import threading
from typing import Any, Dict, List, Optional

import torch


# --------------------------------------------------------------------------- #
# Module state.                                                                #
# --------------------------------------------------------------------------- #
# Shared GPU buffer for per-LAYER accumulation. Size = max layer grad in
# BF16. Reused for every layer, every mb. Allocated in setup, freed in
# shutdown.
_GPU_BUF: Optional[torch.Tensor] = None
# Per-LAYER info: ``id(layer_mod) -> {params, sizes, offsets, targets,
# layer_size, cpu_slot}``. ``params`` are leaf params in deterministic
# order (matches ``layer_mod.parameters()`` walk). ``offsets[i]`` is the
# start of param i's slice in the shared ``_GPU_BUF``. ``targets[i]`` is
# the per-param CPU accumulator (``.view(-1)`` of ``s.m`` /
# ``s.accum`` / ``s.mom_buf``).
_LAYER_INFO: Dict[int, Dict[str, Any]] = {}
# Manual-flush GPU buffer (the 4 top-level params not under any layer).
# Single shared buffer of size = max MF param grad, BF16.
_MF_GPU_BUF: Optional[torch.Tensor] = None
# Per-MF info: ``id(param) -> {target, cpu_slot, size}``.
_MF_INFO: Dict[int, Dict[str, Any]] = {}
# Work queue + worker thread. Entries are 4-tuples:
#   ("layer", info, pinned_slot, event)
#   ("mf", info, pinned_slot, event)
# ``info`` is the layer or mf dict; the worker reads the pinned slot,
# adds into target.
_QUEUE: "queue.Queue" = queue.Queue()
_WORKER: Optional[threading.Thread] = None


# --------------------------------------------------------------------------- #
# Worker.                                                                      #
# --------------------------------------------------------------------------- #
def _worker_loop() -> None:
    """Drain the queue, add CPU slot slices into per-param targets.

    Exit on ``None`` sentinel.
    """
    while True:
        item = _QUEUE.get()
        if item is None:
            _QUEUE.task_done()
            return
        kind, info, pinned_slot, event = item
        event.synchronize()
        if kind == "layer":
            for tgt, off, n in zip(
                info["targets"], info["offsets"], info["sizes"],
            ):
                tgt.add_(pinned_slot[off:off + n])
        elif kind == "mf":
            n = info["size"]
            info["target"].add_(pinned_slot[:n])
        else:
            raise RuntimeError(f"per_layer_gpu_accum worker: unknown kind {kind!r}")
        _QUEUE.task_done()


# --------------------------------------------------------------------------- #
# Hooks.                                                                       #
# --------------------------------------------------------------------------- #
def _make_mf_hook(p: torch.nn.Parameter):
    """Build a closure-based ``register_post_accumulate_grad_hook``
    callback for one manual-flush param.

    Fires DURING autograd (no sync needed) when ``p.grad`` is ready.
    Does GPU-add into the shared ``_MF_GPU_BUF``, async D2H into the
    per-param CPU pinned slot, and enqueues ``("mf", info, slot,
    event)`` for the worker. Resets the gpu_buf slice so the next
    microbatch starts clean.
    """
    pid = id(p)

    def hook(param: torch.nn.Parameter) -> None:
        info = _MF_INFO.get(pid)
        if info is None:
            return
        g = param.grad
        if g is None:
            return
        n = info["size"]
        _MF_GPU_BUF[:n].add_(g.view(-1))
        param.grad = None
        cpu_slot = info["cpu_slot"]
        cpu_slot[:n].copy_(_MF_GPU_BUF[:n], non_blocking=True)
        event = torch.cuda.Event(blocking=False)
        event.record()
        _QUEUE.put(("mf", info, cpu_slot, event))
        _MF_GPU_BUF[:n].zero_()

    return hook


def _layer_hook(module, grad_input, grad_output) -> None:
    """Per-``TPHippoLayer`` backward hook. Fires AFTER the layer's
    bwd has populated every ``.grad`` (autograd eager mode guarantees
    all submodule bwds complete before the outer module's bwd
    returns).

    Strategy:
      1. For each param in the layer (in deterministic order
         matching the per-layer info table), if ``p.grad`` is set:
         GPU-add into shared ``_GPU_BUF`` at layer-scoped offset,
         release ``p.grad``, release the local Tensor reference.
      2. Async D2H the accumulated slice to the layer's CPU pinned
         slot. Record event. Enqueue for the worker.
      3. Reset ``_GPU_BUF[:layer_size]`` to zero so the next layer's
         bwd (or next mb) starts clean.
    """
    info = _LAYER_INFO[id(module)]
    layer_size = info["layer_size"]
    buf = _GPU_BUF

    for p, off, n in zip(info["params"], info["offsets"], info["sizes"]):
        g = p.grad
        if g is None:
            continue
        buf[off:off + n].add_(g.view(-1))
        p.grad = None
        # Drop the local reference. PyTorch tensors are refcounted,
        # so the grad tensor is now ready for the allocator to
        # reuse. ``gc.collect()`` forces release of any circular
        # refs that might keep the wrapper alive.
        del g

    cpu_slot = info["cpu_slot"]
    cpu_slot[:layer_size].copy_(buf[:layer_size], non_blocking=True)
    event = torch.cuda.Event(blocking=False)
    event.record()
    _QUEUE.put(("layer", info, cpu_slot, event))

    buf[:layer_size].zero_()
    # Aggressive cleanup: force the allocator to release cached
    # .grad blocks back to the pool. PyTorch's caching allocator
    # holds blocks even after the Python ref is gone; without this
    # collect, N layers' worth of grad blocks accumulate in the
    # pool before the next mb reuses them.
    gc.collect()


# --------------------------------------------------------------------------- #
# Public API.                                                                  #
# --------------------------------------------------------------------------- #
def setup_per_layer_gpu_accum(model, optimizers) -> None:
    """Install per-layer GPU accumulation for this rank.

    Walks ``model.layers_per_device[dev_key]`` (a ``ModuleList`` of
    ``TPHippoLayer``). For each layer, builds the per-layer info
    table (param list, sizes, layer-scoped offsets, per-param
    CPU accumulator targets). Allocates ONE shared ``_GPU_BUF`` of
    size = max layer grad in BF16, plus one CPU pinned slot per
    layer for the D2H target.

    For params that live OUTSIDE any ``TPHippoLayer`` (tied embed,
    top-level norm, ``attn_res.query``, ``attn_res.norm`` —
    ``replicated_per_device.0.*`` namespace), installs a
    ``register_post_accumulate_grad_hook`` per leaf param. These
    fire DURING autograd, no sync needed; the per-param hook does
    GPU-add into a separate small ``_MF_GPU_BUF`` + async D2H +
    queue.

    Starts the worker thread (daemon). Caller is expected to call
    :func:`flush_per_layer_gpu_accum` per microbatch (no-op on the
    main thread) and :func:`transfer_per_layer_gpu_accum_join` at
    step end (waits for the worker to drain).
    """
    global _GPU_BUF, _WORKER, _MF_GPU_BUF

    # Drain any stale queue entries from a prior test run.
    while not _QUEUE.empty():
        try:
            _QUEUE.get_nowait()
            _QUEUE.task_done()
        except queue.Empty:
            break

    # Start the worker thread (one daemon for the run).
    if _WORKER is None or not _WORKER.is_alive():
        _WORKER = threading.Thread(
            target=_worker_loop, daemon=True,
            name="per-layer-gpu-accum-worker",
        )
        _WORKER.start()

    # Build a param-id -> _ParamState lookup.
    state_by_pid: Dict[int, Any] = {}
    for opt in optimizers:
        for s in opt.state.values():
            if s.param is not None:
                state_by_pid[id(s.param)] = s

    # Collect all param ids that live INSIDE any TPHippoLayer (so
    # we can identify manual-flush params as the complement).
    layer_param_ids: set = set()
    for dev_key, dev_layers in model.layers_per_device.items():
        for layer_mod in dev_layers:
            for p in layer_mod.parameters():
                layer_param_ids.add(id(p))

    # Lazy import to avoid a top-level cycle.
    from ._state import _accumulator_target

    def _is_mxfp8_param_state(s) -> bool:
        """Mirror v1's mxfp8 skip rule."""
        _, _, is_mxfp8 = _accumulator_target(s)
        return is_mxfp8

    # Walk layers and build per-layer info.
    _LAYER_INFO.clear()
    max_layer_size = 0
    layer_modules: List[torch.nn.Module] = []
    for dev_key, dev_layers in model.layers_per_device.items():
        for layer_mod in dev_layers:
            params: List[torch.nn.Parameter] = []
            sizes: List[int] = []
            offsets: List[int] = []
            targets: List[torch.Tensor] = []
            offset = 0
            for p in layer_mod.parameters():
                if not p.requires_grad:
                    continue
                s = state_by_pid.get(id(p))
                if s is None:
                    continue
                # Skip NVFP4 mode-3 (no leaf param) and mxfp8
                # (needs fused C++ kernel).
                if getattr(s, "nvfp4_module", None) is not None:
                    continue
                if _is_mxfp8_param_state(s):
                    continue
                target, cast_dtype, _ = _accumulator_target(s)
                n = p.numel()
                # Each per-param target has shape ``(p.numel(),)``.
                # We pack all params in a layer contiguously into
                # ``_GPU_BUF``; the worker's per-param add reads
                # ``pinned_slot[off:off+n]`` (layer-scoped slice)
                # into ``target`` at full length.
                params.append(p)
                sizes.append(n)
                offsets.append(offset)
                targets.append(target.view(-1))
                offset += n
            layer_size = offset
            if layer_size == 0:
                continue
            cpu_slot = torch.empty(
                layer_size, dtype=torch.bfloat16,
            ).pin_memory()
            _LAYER_INFO[id(layer_mod)] = {
                "params": params, "sizes": sizes, "offsets": offsets,
                "targets": targets, "layer_size": layer_size,
                "cpu_slot": cpu_slot,
            }
            layer_modules.append(layer_mod)
            if layer_size > max_layer_size:
                max_layer_size = layer_size

    if max_layer_size > 0:
        _GPU_BUF = torch.zeros(
            max_layer_size, dtype=torch.bfloat16, device="cuda",
        )
    for layer_mod in layer_modules:
        layer_mod.register_full_backward_hook(_layer_hook)

    # Manual-flush params: leaf params NOT in any TPHippoLayer.
    _MF_INFO.clear()
    mf_state_by_pid: Dict[int, Any] = {}
    for opt in optimizers:
        for s in opt.state.values():
            p = s.param
            if p is None or not p.requires_grad:
                continue
            if id(p) in layer_param_ids:
                continue
            if getattr(s, "nvfp4_module", None) is not None:
                continue
            if _is_mxfp8_param_state(s):
                continue
            mf_state_by_pid[id(p)] = s
    if mf_state_by_pid:
        max_mf_size = max(s.param.numel() for s in mf_state_by_pid.values())
        _MF_GPU_BUF = torch.zeros(
            max_mf_size, dtype=torch.bfloat16, device="cuda",
        )
        for s in mf_state_by_pid.values():
            p = s.param
            n = p.numel()
            target, _, _ = _accumulator_target(s)
            cpu_slot = torch.empty(n, dtype=torch.bfloat16).pin_memory()
            _MF_INFO[id(p)] = {
                "target": target.view(-1),
                "cpu_slot": cpu_slot,
                "size": n,
            }
            p.register_post_accumulate_grad_hook(_make_mf_hook(p))


def flush_per_layer_gpu_accum() -> None:
    """Per-mb flush. NO-OP on the main thread — the per-layer hooks
    already issued the D2Hs and the worker drains them in parallel
    with the main thread's continued bwd.

    Kept as a callable to match the v1 hook API (``flush_v1`` /
    ``accumulate_grads_to_cpu``).
    """


def transfer_per_layer_gpu_accum_join() -> None:
    """Step-end barrier. Waits for the worker thread to drain all
    queued D2H + add work. Called once per training step.

    Also calls ``torch.cuda.empty_cache()`` to release cached
    ``.grad`` blocks back to the OS. This is the "VRAM only holds
    the buffer" cleanup the user asked for — between steps the
    GPU holds just ``model + gpu_buf + mf_buf + tiny transient``.
    """
    _QUEUE.join()
    torch.cuda.empty_cache()


def shutdown_per_layer_gpu_accum() -> None:
    """Tear down the worker thread and free GPU/CPU buffers. Safe
    to call multiple times.
    """
    global _WORKER, _GPU_BUF, _MF_GPU_BUF
    if _WORKER is not None and _WORKER.is_alive():
        _QUEUE.put(None)
        _QUEUE.join()
        _WORKER.join(timeout=5.0)
        _WORKER = None
    _GPU_BUF = None
    _MF_GPU_BUF = None
    _LAYER_INFO.clear()
    _MF_INFO.clear()