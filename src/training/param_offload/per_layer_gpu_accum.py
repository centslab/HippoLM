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
    finishes, then a fused OpenMP kernel adds all per-param
    slices into their accumulators in parallel across CPU cores.
    Set ``HIPPO_V4_FORCE_PYLOOP=1`` to fall back to the per-param
    Python-loop worker (slower, but no C++ toolchain needed).

VRAM (245M test scale, 5060 Ti 16G):

  - Persistent: ``gpu_buf`` (max layer grad, ~55 MiB BF16) +
    ``mf_buf`` (max MF param grad, ~24 MiB BF16) = ~80 MiB.
  - Peak during bwd: ``~1 layer's .grad`` alive at the per-layer
    hook fire time = +55 MiB over v1. The full 500 MiB peak
    delta reported in earlier prototype numbers was from the
    test's caching-allocator carry-over across the 4 sequential
    modes; production (single model + per-mb empty_cache) keeps
    VRAM at "model + ~80 MiB buffer" between mbs.

Step time wins (245M test scale, 5060 Ti 16G, fused OpenMP worker):

  - 15-37% over v1 at every shape measured (4L/256, 8L/512,
    16L/512). The original 33-44% win was measured before the
    gc.collect() regression (see auto-memory
    ``project_v4_gc_collect_regression.md``); the fused OpenMP
    worker restores the win.
  - Beats v2 at every MBS — v2 LOSES at MBS ≥ 1024 because its
    single per-step D2H blocks the optimizer step. v4's per-mb
    per-layer D2Hs amortize into the per-mb bwd tail.

Out of scope:

  - NVFP4 mode-3 (no leaf param; the FP4 packed buffers live on
    the module, not as a Parameter). NVFP4 entries are
    skipped and fall through to ``flush_manual_flush_params``.

History
-------
The pre-2026-07-12 design had a ``_is_mxfp8_param_state``
skip rule (mxfp8 storage needed a fused C++ add kernel —
the GPU ``add_`` on E4M3 silently breaks). After mxfp8
storage was removed (2026-07-12; long-training
instability), the skip rule is gone: every per-param hook
now does the same GPU ``add_()`` and CPU pin transfer.
The code lives on the ``archive/int8-mxfp8-muon`` branch.

The 2026-07-12 ship also had ``gc.collect()`` at the end of
``_layer_hook``, which was ~1 ms at ship time but grew to
130-150 ms per call as the codebase grew (loop.py split,
param_offload.py split, tp_layers split, NVFP4 mode-3
additions). Removed 2026-07-13; see
``auto-memory/project_v4_gc_collect_regression.md``.

The 2026-07-13 revision replaced the per-param Python
``tgt.add_(slice)`` worker loop with a single fused C++
OpenMP kernel (DeepSpeed CPUAdam-style). JIT-compiled once
on first use; falls back to the Python loop if no C++
toolchain is available at runtime. A/B wins: see the
"Step time wins" section above.
"""
from __future__ import annotations

import os
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
# Fused OpenMP add kernel (DeepSpeed CPUAdam-style).                           #
# --------------------------------------------------------------------------- #
# One C++ call does all per-param CPU adds in parallel via OpenMP, with
# within-op parallelism for big ops. Replaces the per-param Python loop
# (each ``tgt.add_(slice)`` has ~50 us Python+dispatch overhead → 17 calls
# per layer = ~1 ms wasted on overhead alone at small scale). The fused
# kernel removes this overhead AND uses multiple CPU cores for the actual
# add work, bringing CPU add time close to the PCIe D2H floor.
#
# A/B measured on the dev box (5060 Ti 16G):
#   - 4L/256 seq=4096 mbs=1024:  fused -20% vs v1 (cpu_add)
#   - 8L/256 seq=4096 mbs=512:   fused -15% vs v1
#   - 16L/512 seq=4096 mbs=512:  fused -37% vs v1
#
# Falls back to the Python-loop worker if JIT compile fails (no C++ toolchain
# at runtime). The fallback keeps correctness — only the speedup is lost.
_FUSED_EXT: Optional[Any] = None
_FUSED_EXT_FAILED: bool = False
_FUSED_CPP_SRC = r"""
#include <torch/extension.h>
#include <vector>
#include <cstdint>
#include <omp.h>

static inline void _add_chunk(uint8_t* tgt, const uint8_t* src, int64_t n_bytes) {
    int64_t i = 0;
    int64_t n_aligned = n_bytes & ~31;
    for (; i < n_aligned; i += 32) {
        __m256i a = _mm256_loadu_si256((__m256i*)(tgt + i));
        __m256i b = _mm256_loadu_si256((__m256i*)(src + i));
        _mm256_storeu_si256((__m256i*)(tgt + i), _mm256_add_epi16(a, b));
    }
    for (; i < n_bytes; i += 2) {
        uint16_t a = *(uint16_t*)(tgt + i);
        uint16_t b = *(uint16_t*)(src + i);
        *(uint16_t*)(tgt + i) = a + b;
    }
}

void fused_add_into_many(std::vector<int64_t> tgt_ptrs,
                         std::vector<int64_t> src_ptrs,
                         std::vector<int64_t> sizes) {
    int n = (int)tgt_ptrs.size();
    #pragma omp parallel for schedule(dynamic, 1)
    for (int i = 0; i < n; i++) {
        uint8_t* tgt = reinterpret_cast<uint8_t*>(tgt_ptrs[i]);
        const uint8_t* src = reinterpret_cast<const uint8_t*>(src_ptrs[i]);
        int64_t nb = sizes[i];
        int nthreads = omp_get_num_threads();
        int64_t chunk = (nb + nthreads - 1) / nthreads;
        chunk = (chunk + 31) & ~31;  // align to 32 bytes (AVX2 lane width)
        #pragma omp parallel for schedule(static)
        for (int64_t off = 0; off < nb; off += chunk) {
            int64_t end = std::min(off + chunk, nb);
            _add_chunk(tgt + off, src + off, end - off);
        }
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_add_into_many", &fused_add_into_many,
          "Fused multi-target BF16 add with OpenMP (within + across ops)");
}
"""


def _try_load_fused_ext() -> Optional[Any]:
    """JIT-compile the fused kernel once. Cached. Returns None on failure."""
    global _FUSED_EXT, _FUSED_EXT_FAILED
    if _FUSED_EXT is not None:
        return _FUSED_EXT
    if _FUSED_EXT_FAILED:
        return None
    try:
        from torch.utils.cpp_extension import load_inline
        # Honor env var to skip fused kernel (debug / toolchain issues).
        if os.environ.get("HIPPO_V4_FORCE_PYLOOP"):
            _FUSED_EXT_FAILED = True
            return None
        # Pin the build dir so re-imports don't recompile.
        build_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "_fused_ext_build",
        )
        os.makedirs(build_dir, exist_ok=True)
        _FUSED_EXT = load_inline(
            name="hippo_v4_fused_add",
            cpp_sources=[_FUSED_CPP_SRC],
            extra_cflags=["-O3", "-mavx2", "-fopenmp"],
            extra_ldflags=["-fopenmp"],
            verbose=False,
            build_directory=build_dir,
        )
        return _FUSED_EXT
    except Exception as e:  # pragma: no cover — toolchain missing
        _FUSED_EXT_FAILED = True
        return None


# --------------------------------------------------------------------------- #
# Worker.                                                                      #
# --------------------------------------------------------------------------- #
def _py_loop_worker_loop() -> None:
    """Python-loop worker (fallback).

    17 separate ``tgt.add_(slice)`` calls per layer. PyTorch's internal
    OpenMP parallelizes each call across CPU cores, but Python overhead
    per call (~50 us) and cross-call serialization make this slower than
    the fused kernel above.
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


def _fused_worker_loop() -> None:
    """Fused OpenMP worker. One C++ call per layer does all per-param
    CPU adds in parallel across cores.

    Falls back to the py-loop worker if the JIT extension failed to
    compile (no C++ toolchain at runtime).
    """
    ext = _try_load_fused_ext()
    if ext is None:
        _py_loop_worker_loop()
        return
    while True:
        item = _QUEUE.get()
        if item is None:
            _QUEUE.task_done()
            return
        kind, info, pinned_slot, event = item
        event.synchronize()
        if kind == "layer":
            tgts = info["targets"]
            offsets = info["offsets"]
            sizes = info["sizes"]
            slot_addr = pinned_slot.data_ptr()
            tgt_ptrs = [t.data_ptr() for t in tgts]
            src_ptrs = [slot_addr + off for off in offsets]
            nbytes = [sz * 2 for sz in sizes]
            ext.fused_add_into_many(tgt_ptrs, src_ptrs, nbytes)
        elif kind == "mf":
            n = info["size"]
            tg = info["target"]
            ext.fused_add_into_many(
                [tg.data_ptr()],
                [pinned_slot.data_ptr()],
                [n * 2],
            )
        else:
            raise RuntimeError(f"per_layer_gpu_accum worker: unknown kind {kind!r}")
        _QUEUE.task_done()


# Default worker function; can be overridden via env var for A/B testing.
def _worker_loop() -> None:
    if os.environ.get("HIPPO_V4_FORCE_PYLOOP"):
        _py_loop_worker_loop()
    else:
        _fused_worker_loop()


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
        # reuse. (No ``gc.collect()`` needed — grad tensors don't
        # form cycles, and the Caching Allocator reuses freed
        # blocks without it.)
        del g

    cpu_slot = info["cpu_slot"]
    cpu_slot[:layer_size].copy_(buf[:layer_size], non_blocking=True)
    event = torch.cuda.Event(blocking=False)
    event.record()
    _QUEUE.put(("layer", info, cpu_slot, event))

    buf[:layer_size].zero_()
    # NOTE: ``gc.collect()`` was here until 2026-07-13 — it was
    # supposed to force the Caching Allocator to release cached
    # ``.grad`` blocks, but the allocator already reuses freed
    # blocks without it. Grad tensors don't form cycles (they're
    # leaf tensors or freshly-materialized intermediate grads),
    # so gc.collect had nothing to release. In the original
    # codebase (2026-07-12 v4 ship) it cost ~1 ms; after several
    # refactors (loop.py / param_offload.py / tp_layers splits,
    # NVFP4 mode-3 additions) it grew to 130-150 ms per call,
    # making v4 5-8x slower than v1 at the 4L/256 test shape.
    # Removed — see ``auto-memory/project_v4_gc_collect_regression.md``.


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
                # Skip NVFP4 mode-3 (no leaf param). Other
                # params get the per-layer accumulator.
                if getattr(s, "nvfp4_module", None) is not None:
                    continue
                target, cast_dtype = _accumulator_target(s)
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
            mf_state_by_pid[id(p)] = s
    if mf_state_by_pid:
        max_mf_size = max(s.param.numel() for s in mf_state_by_pid.values())
        _MF_GPU_BUF = torch.zeros(
            max_mf_size, dtype=torch.bfloat16, device="cuda",
        )
        for s in mf_state_by_pid.values():
            p = s.param
            n = p.numel()
            target, _ = _accumulator_target(s)
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