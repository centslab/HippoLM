"""JIT-compiled fused CPU kernels (DeepSpeed CPUAdam-style).

Owns the OpenMP kernel that's used by both:

  * v4's per-layer worker (:mod:`.per_layer_gpu_accum`) for
    fused BF16 ``tgt += slice`` across all per-param accumulators
    in one C++ call.
  * The end-of-cycle zero path in :meth:`CPUMuon.step` and
    :func:`zero_cpu_grad_accum` for fused ``memset`` across all
    per-param ``mom_buf`` / ``s.m`` tensors.

Why one C++ call beats a Python loop:
  PyTorch's per-op internal OMP parallelizes within a single op
  but not across ops. A loop of N ``tgt.add_(slice)`` or N
  ``t.zero_()`` calls thus serializes across ops while
  parallelizing within each op. For our shape (17 params ×
  80 KiB BF16 at 4L/256 test scale) the cross-op Python+dispatch
  overhead alone is ~1 ms / layer — the bottleneck for small
  shapes. The fused kernel does all N ops in ONE C++ call with
  OMP parallel across ops, eliminating the Python overhead and
  enabling cross-op CPU bandwidth.

The same trick DeepSpeed's CPUAdam uses (fused C++ single-pass
with OpenMP ``parallel for`` across cores) applies here for the
much simpler zero/add case — both are bandwidth-bound on pinned
memory and benefit identically from multi-core parallel BW.

JIT-compiled once on first use via
``torch.utils.cpp_extension.load_inline``, cached under
``src/training/param_offload/_fused_ext_build/``. Falls back to
per-op Python loops (the original behavior) if no C++ toolchain
is available at runtime — correctness is preserved either way,
only the speedup is lost.

Set ``HIPPO_FUSED_FORCE_PYLOOP=1`` (or the legacy
``HIPPO_V4_FORCE_PYLOOP=1``) to force the slow Python-loop
fallback for debugging / A/B benchmarking.

A/B numbers (5060 Ti 16G dev box) for the fused add path
(v4 worker — see :mod:`.per_layer_gpu_accum`):

| Shape | v1 (cpu_add) | v4 (py_loop) | v4 (fused OMP) | fused vs v1 |
|---|---|---|---|---|
| 4L/256 seq=4096 mbs=1024 | 602 ms | 984 ms | 513 ms | **-15%** |
| 8L/256 seq=4096 mbs=512 | 1725 ms | 2728 ms | 1466 ms | **-15%** |
| 16L/512 seq=4096 mbs=512 | 4189 ms | 6321 ms | 2627 ms | **-37%** |

The zero path follows the same model — same Python-loop
overhead to remove, same multi-core parallelism to unlock.
"""
from __future__ import annotations

import hashlib
import os
import shutil
from typing import Any, List, Optional

import torch


# --------------------------------------------------------------------------- #
# Module state.                                                                #
# --------------------------------------------------------------------------- #
_EXT: Optional[Any] = None
_EXT_FAILED: bool = False


# --------------------------------------------------------------------------- #
# C++ source for the fused kernel.                                             #
# --------------------------------------------------------------------------- #
# Two entry points:
#
#   fused_add_into_many(tgt_ptrs, src_ptrs, sizes)
#     Adds each src into its tgt, element-wise. Sizes are in bytes.
#     Uses AVX2 ``_mm256_add_epi16`` on 2-byte aligned chunks
#     (BF16 / FP16); falls back to scalar for the tail.
#
#   fused_zero_many(ptrs, sizes)
#     Zeros each tensor via ``std::memset``. Sizes are in bytes.
#     ``memset(0)`` produces the IEEE +0.0 bit pattern for any
#     floating dtype (FP16/BF16/FP32), so it's dtype-agnostic.
#
# Both kernels parallelize across ops via an outer OMP
# ``parallel for`` (the main win for many small ops). The add
# kernel also chunks within each op; since nested OMP is OFF by
# default, the inner chunking runs serial on each outer thread
# in practice — the outer parallel-for already saturates per-op
# BW on this box (~30 GB/s).
_CPP_SRC = r"""
#include <torch/extension.h>
#include <vector>
#include <cstdint>
#include <cstring>
#include <algorithm>
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
    if (n == 0) return;
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

// One parallel for across all ops, no inner chunking. Each
// outer thread does one or more whole-op memsets via dynamic
// scheduling (work-stealing balances uneven op counts). glibc's
// memset dispatches to AVX2 stores internally for sizes >= a
// few hundred bytes, so per-op BW is competitive with PyTorch's
// parallel_for-backed ``zero_()`` while the cross-op Python +
// dispatch overhead (~50 us per op) is eliminated.
void fused_zero_many(std::vector<int64_t> ptrs,
                     std::vector<int64_t> sizes) {
    int n = (int)ptrs.size();
    if (n == 0) return;
    #pragma omp parallel for schedule(dynamic, 1)
    for (int i = 0; i < n; i++) {
        std::memset(reinterpret_cast<uint8_t*>(ptrs[i]), 0, sizes[i]);
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_add_into_many", &fused_add_into_many,
          "Fused multi-target BF16 add with OpenMP (within + across ops)");
    m.def("fused_zero_many", &fused_zero_many,
          "Fused multi-target zero with OpenMP");
}
"""


def _try_load_ext() -> Optional[Any]:
    """JIT-compile the fused kernels once. Cached. Returns None on failure.

    Cache invalidation: hashes the C++ source and stores the hash
    in a sentinel file under the build dir. If the hash doesn't
    match on a subsequent import, the build dir is wiped and the
    extension is recompiled. This is independent of ninja's
    dependency tracking (which load_inline uses internally) — the
    sentinel guarantees correctness even if ninja skips a rebuild
    for whatever reason.
    """
    global _EXT, _EXT_FAILED
    if _EXT is not None:
        return _EXT
    if _EXT_FAILED:
        return None
    # Honor env vars to skip the fused kernel (debug / toolchain
    # issues / A/B benchmarking).
    if (os.environ.get("HIPPO_FUSED_FORCE_PYLOOP")
            or os.environ.get("HIPPO_V4_FORCE_PYLOOP")):
        _EXT_FAILED = True
        return None
    try:
        from torch.utils.cpp_extension import load_inline
        build_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "_fused_ext_build",
        )
        os.makedirs(build_dir, exist_ok=True)

        # Hash-based cache invalidation: if the C++ source changed
        # since the last build, wipe the build dir before
        # recompiling. Without this, load_inline's internal ninja
        # may load a stale .so that doesn't have the new kernels.
        src_hash = hashlib.sha256(_CPP_SRC.encode()).hexdigest()[:16]
        sentinel = os.path.join(build_dir, "src_hash.txt")
        if os.path.exists(sentinel):
            try:
                with open(sentinel) as f:
                    old_hash = f.read().strip()
            except OSError:
                old_hash = ""
            if old_hash != src_hash:
                for entry in os.listdir(build_dir):
                    p = os.path.join(build_dir, entry)
                    if os.path.isfile(p) or os.path.islink(p):
                        os.unlink(p)
                    else:
                        shutil.rmtree(p, ignore_errors=True)
        try:
            with open(sentinel, "w") as f:
                f.write(src_hash)
        except OSError:
            pass  # Best-effort; recompile still works.

        _EXT = load_inline(
            name="hippo_fused_cpu",
            cpp_sources=[_CPP_SRC],
            extra_cflags=["-O3", "-mavx2", "-fopenmp"],
            extra_ldflags=["-fopenmp"],
            verbose=False,
            build_directory=build_dir,
        )
        return _EXT
    except Exception as e:  # pragma: no cover — toolchain missing
        _EXT_FAILED = True
        return None


# --------------------------------------------------------------------------- #
# Public API.                                                                  #
# --------------------------------------------------------------------------- #
def fused_zero_many(tensors: List[torch.Tensor]) -> bool:
    """Zero a list of CPU tensors using the fused OpenMP kernel.

    Returns True if the fused path was used, False if the
    fallback (per-tensor ``.zero_()``) was used. The fallback
    is only taken if the JIT extension failed to compile (no
    C++ toolchain at runtime) — correctness is preserved either
    way, only the speedup is lost.

    Accepts an empty list (returns True, no-op).
    """
    if not tensors:
        return True
    ext = _try_load_ext()
    if ext is None:
        for t in tensors:
            t.zero_()
        return False
    ptrs = [t.data_ptr() for t in tensors]
    sizes = [t.numel() * t.element_size() for t in tensors]
    ext.fused_zero_many(ptrs, sizes)
    return True


def fused_add_into_many(
    tgt_tensors: List[torch.Tensor],
    src_tensors: List[torch.Tensor],
) -> bool:
    """Add each ``src_tensors[i]`` into ``tgt_tensors[i]`` in place
    using the fused OpenMP kernel.

    All tensors must be 2-byte element dtypes (BF16 / FP16) for
    the AVX2 ``_mm256_add_epi16`` path. Returns True if the fused
    path was used, False if the fallback was used.
    """
    if not tgt_tensors:
        return True
    assert len(tgt_tensors) == len(src_tensors)
    ext = _try_load_ext()
    if ext is None:
        for tgt, src in zip(tgt_tensors, src_tensors):
            tgt.add_(src)
        return False
    tgt_ptrs = [t.data_ptr() for t in tgt_tensors]
    src_ptrs = [t.data_ptr() for t in src_tensors]
    sizes = [t.numel() * t.element_size() for t in tgt_tensors]
    ext.fused_add_into_many(tgt_ptrs, src_ptrs, sizes)
    return True