"""JIT loader for the mxfp8 fused per-mb dequant+add+requant kernel.

This module wraps :func:`torch.utils.cpp_extension.load_inline` with
the project conventions (mirroring ``src/models/ops/cuda/kda_fwd/
load_inline.py``).

* CPU only — no CUDA arch detection.
* The kernel is small and self-contained: only the standard C++
  math headers + libtorch. Compiles in ~5-10s cold; cached
  sub-second on re-import.
* Build cache lives under ``$TORCH_EXTENSIONS_DIR/mxfp8_accum`` so
  the .so survives a git pull (the .cpp source is versioned, the
  .so is not).

The compilation step is gated behind
:data:`HIPPOLM_MXFP8_ACCUM_REBUILD=1` for the rare case we want
to force a clean rebuild.
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import torch
from torch.utils.cpp_extension import load_inline


def _load(verbose: bool = False):
    """Compile (or load from cache) the mxfp8_accum CPU extension."""
    here = Path(__file__).resolve().parent
    kernel_cpp = here / "kernel.cpp"
    bindings_cpp = here / "bindings.cpp"

    if not kernel_cpp.exists():
        raise FileNotFoundError(f"mxfp8 kernel not found at {kernel_cpp}")
    if not bindings_cpp.exists():
        raise FileNotFoundError(f"mxfp8 bindings not found at {bindings_cpp}")

    cxx_flags = [
        "-O3",
        "-std=c++17",
        "-fPIC",
        # Don't use -ffast-math: we want bit-exact rounding for
        # the E4M3/E8M0 conversions (round-to-nearest-even for
        # quantize). -ffast-math can rewrite the roundf call.
        "-fno-fast-math",
    ]

    # Build directory under PyTorch's default TORCH_EXTENSIONS_DIR
    # so the cache survives a git pull.
    build_dir = Path(os.environ.get(
        "HIPPOLM_MXFP8_ACCUM_BUILD_DIR",
        str(Path.home() / ".cache" / "torch_extensions" / "mxfp8_accum"),
    ))

    # Force a clean rebuild if HIPPOLM_MXFP8_ACCUM_REBUILD=1.
    if os.environ.get("HIPPOLM_MXFP8_ACCUM_REBUILD") == "1":
        if build_dir.exists():
            shutil.rmtree(build_dir, ignore_errors=True)

    build_dir.mkdir(parents=True, exist_ok=True)
    build_dir = str(build_dir)

    return load_inline(
        name="hippolm_mxfp8_accum",
        cpp_sources=[bindings_cpp.read_text(), kernel_cpp.read_text()],
        # load_inline auto-generates the PYBIND11_MODULE entry
        # for these function names. The C++ functions live in
        # global namespace.
        functions=["fused_mxfp8_dequant_add_requant", "requantize_mxfp8"],
        extra_cflags=cxx_flags,
        extra_include_paths=[str(here)],
        build_directory=build_dir,
        verbose=verbose,
        with_cuda=False,
    )


# Lazy module-level cache so the first import is the compile.
_module = None


def get_module(verbose: bool = False):
    """Return the compiled CPU module (compiled on first call)."""
    global _module
    if _module is None:
        _module = _load(verbose=verbose)
    return _module


if __name__ == "__main__":
    # Smoke test: ``python -m src.training.ops.mxfp8_accum.load_inline``.
    print("Compiling mxfp8_accum CPU extension ...")
    import time
    t0 = time.perf_counter()
    mod = get_module(verbose=True)
    dt = time.perf_counter() - t0
    print(f"OK ({dt:.1f}s).")
    sys.exit(0)
