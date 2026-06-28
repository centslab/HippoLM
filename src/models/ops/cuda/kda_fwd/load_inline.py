"""JIT loader for the KDA forward CUDA kernel.

This module wraps :func:`torch.utils.cpp_extension.load_inline` with
the project conventions:

* bf16 inputs only — fp16 / fp32 fall back to the vendored FLA Triton path.
* Multi-arch support: at first import we detect the visible CUDA
  device(s) and emit one ``-gencode=arch=compute_X,code=sm_X`` flag
  per architecture. The 5060 Ti dev box is ``sm_120`` (Blackwell
  consumer); the production path is also tested on Ampere
  (``sm_80``, ``sm_86``) and Ada (``sm_89``).
* Build cache lives under ``$TORCH_EXTENSIONS_DIR/kda_fwd`` (the
  PyTorch default). Re-imports hit the cache and return in <100ms.

The compilation step is gated behind :data:`HIPPOLM_KDA_FWD_REBUILD=1`
for the rare case we want to force a clean rebuild.
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import torch
from torch.utils.cpp_extension import load_inline


# --------------------------------------------------------------------------- #
# Compute-capability detection                                                #
# --------------------------------------------------------------------------- #
def _detect_arch_flags() -> list[str]:
    """Return the per-arch ``-gencode`` flags for the visible CUDA devices.

    Falls back to ``TORCH_CUDA_ARCH_LIST`` if set (CI override). Returns
    an empty list if CUDA is unavailable (caller raises).
    """
    env = os.environ.get("TORCH_CUDA_ARCH_LIST")
    if env:
        # TORCH_CUDA_ARCH_LIST supports both "8.0;8.6" and "80;86" forms.
        flags = []
        for tok in env.replace(",", ";").split(";"):
            tok = tok.strip()
            if not tok:
                continue
            ver = tok if "." in tok else f"{tok[:-1]}.{tok[-1]}"
            major, minor = ver.split(".")
            cc = f"{major}{minor}"
            flags.append(f"-gencode=arch=compute_{cc},code=sm_{cc}")
        return flags

    if not torch.cuda.is_available():
        return []
    seen: set[str] = set()
    flags: list[str] = []
    for i in range(torch.cuda.device_count()):
        try:
            major, minor = torch.cuda.get_device_capability(i)
        except Exception:
            continue
        cc = f"{major}{minor}"
        if cc in seen:
            continue
        seen.add(cc)
        flags.append(f"-gencode=arch=compute_{cc},code=sm_{cc}")
    return flags


def _cuda_path_flags() -> list[str]:
    """Locate CUDA include + lib dirs from ``CUDA_HOME`` or ``nvcc``."""
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if cuda_home is None:
        nvcc = shutil.which("nvcc")
        if nvcc is not None:
            cuda_home = str(Path(nvcc).resolve().parent.parent)
    if cuda_home is None:
        return []
    cuda_home_p = Path(cuda_home)
    return [
        f"-I{cuda_home_p / 'include'}",
        f"-L{cuda_home_p / 'lib64'}",
    ]


def _load(verbose: bool = False):
    """Compile (or load from cache) the kda_fwd CUDA extension.

    First call: ~20-30s cold compile, then cached. Subsequent calls
    are sub-second cache hits.
    """
    here = Path(__file__).resolve().parent
    kernel_cu = here / "kernel.cu"
    bindings_cpp = here / "bindings.cpp"

    if not kernel_cu.exists():
        raise FileNotFoundError(f"KDA CUDA kernel not found at {kernel_cu}")
    if not bindings_cpp.exists():
        raise FileNotFoundError(f"KDA CUDA bindings not found at {bindings_cpp}")

    arch_flags = _detect_arch_flags()
    if not arch_flags:
        raise RuntimeError("No CUDA devices detected and TORCH_CUDA_ARCH_LIST not set")

    cuda_flags = [
        "-O3",
        "--use_fast_math",
        "-std=c++17",
        "-Xptxas=-v",
        "--expt-relaxed-constexpr",
        "-lineinfo",
    ] + arch_flags

    cxx_flags = [
        "-O3",
        "-std=c++17",
        "-fPIC",
    ]

    # Build directory under PyTorch's default TORCH_EXTENSIONS_DIR
    # so the cache survives a git pull (the source files in this
    # module are versioned, but the compiled .so is not).
    build_dir = Path(os.environ.get(
        "HIPPOLM_KDA_FWD_BUILD_DIR",
        str(Path.home() / ".cache" / "torch_extensions" / "kda_fwd"),
    ))

    # Force a clean rebuild if HIPPOLM_KDA_FWD_REBUILD=1 (escape hatch).
    # Do this BEFORE mkdir so the mkdir below can recreate the dir.
    if os.environ.get("HIPPOLM_KDA_FWD_REBUILD") == "1":
        if build_dir.exists():
            shutil.rmtree(build_dir, ignore_errors=True)

    build_dir.mkdir(parents=True, exist_ok=True)
    build_dir = str(build_dir)

    extra_cuda_cflags = cuda_flags + _cuda_path_flags()
    extra_cflags = cxx_flags

    return load_inline(
        name="hippolm_kda_fwd",
        cpp_sources=[bindings_cpp.read_text()],
        cuda_sources=[kernel_cu.read_text()],
        # load_inline auto-generates the PYBIND11_MODULE entry for
        # these function names. The C++ functions live in global
        # namespace in kernel.cu (no `kda::` qualifier).
        functions=["forward_sub", "delta_h", "chunk_o",
                   "kda_fwd", "kda_fwd_version"],
        extra_cflags=extra_cflags,
        extra_cuda_cflags=extra_cuda_cflags,
        extra_include_paths=[str(here)],
        build_directory=build_dir,
        verbose=verbose,
        with_cuda=True,
    )


# Lazy module-level cache so the first import is the compile
_module = None


def get_module(verbose: bool = False):
    """Return the compiled CUDA module (compiled on first call)."""
    global _module
    if _module is None:
        _module = _load(verbose=verbose)
    return _module


def version_string() -> str:
    """Return the compile-time version string baked into the kernel."""
    try:
        mod = get_module()
        return mod.kda_fwd_version()
    except Exception as e:
        return f"<uncompiled: {e}>"


if __name__ == "__main__":
    # Smoke test: ``python -m src.models.ops.cuda.kda_fwd.load_inline``.
    print("Compiling kda_fwd CUDA extension ...")
    import time
    t0 = time.perf_counter()
    mod = get_module(verbose=True)
    dt = time.perf_counter() - t0
    print(f"OK ({dt:.1f}s). version={mod.kda_fwd_version()!r}")
    sys.exit(0)
