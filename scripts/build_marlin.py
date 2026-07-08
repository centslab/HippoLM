#!/usr/bin/env python3
"""Build the per-arch Marlin FP4 .so files used by the W4A16 NVFP4 FFN path.

What this produces
------------------
For each requested SM arch, two .so files are emitted:

    src/models/ops/cuda/lib/marlin_fp4_kernel_only_sm{sm}.so
    src/models/ops/cuda/lib/marlin_fp4_repack_sm{sm}.so

The first exposes ``marlin::marlin_mm`` (FP4 matmul) and the second exposes
``marlin_repack`` (gptq_marlin_repack_kernel) — both via ``extern "C"`` for
ctypes binding. The NVFP4 module loader in
:file:`src/models/ops/nvfp4_marlin.py` picks the right pair at first
forward based on ``torch.cuda.get_device_capability()``.

Why a custom build (and not JIT)
--------------------------------
vLLM's ``gptq_marlin_repack.cu`` (and several siblings) use the
``torch::stable::Tensor`` API at a level that requires a newer Stable ABI
than the precompiled torch 2.12.0+cu130 ships with. JIT via
``torch.utils.cpp_extension.load`` hits errors like::

    error: class "torch::stable::Tensor" has no member "const_data_ptr"
    error: identifier "TORCH_BOX" is undefined
    error: namespace "torch::stable" has no member "empty"

The :file:`src/models/ops/cuda/marlin_build/sources/` tree sidesteps this by
text-extracting just the kernel + repack functions (no
``STABLE_TORCH_LIBRARY_IMPL`` registration, no ``torch::stable::Tensor``
plumbing) and exposing them as ``extern "C"`` symbols. The vendored vllm
headers have two small patches (see :file:`docs/marlin_build_pipeline.md`
section *Patches applied*) and a namespace wrap (see *Namespace wrap*).

Why per-arch
------------
The per-arch .so files are gitignored (``*.so`` rule in the repo root,
with an explicit whitelist for ``src/models/ops/cuda/lib/`` so a
specific arch's .so can be tracked). Each arch needs its own SASS —
the sm_120 SASS the 5060 Ti ships with is not optimal on A100
(sm_80), RTX 4090 (sm_89), or RTX 5090 (sm_120). Run this script
once per machine. A universal PTX-fallback ``.so`` is on the
roadmap (will be ``--ptx``); today's loader does SASS-first lookup
only.

Usage
-----

::

    # Build for the current device (auto-detect SM):
    python scripts/build_marlin.py

    # Build for specific arches:
    python scripts/build_marlin.py --arch 80,89,120

    # Build for the production triple (V100/A100/5060 Ti, etc.):
    python scripts/build_marlin.py --arch 70,80,86,89,120

    # Custom output dir:
    python scripts/build_marlin.py --arch 120 --output /tmp/marlin_build_out

Build cost
----------
On a 5060 Ti sm_120 build, the matmul + repack together take ~35 min
(roughly 30 min for the matmul, ~5 min for the repack). All FP4 +
FP8 NVFP4 instantiations are included; the actual per-arch compile is
the bottleneck, not the source count. The output .so is ~21 MB
(matmul) + ~1.1 MB (repack). The build is incremental: a re-run for
the same arch reuses the .o files in the build dir.
"""
from __future__ import annotations

import argparse
import os
import shlex
import shutil
import subprocess
import sys
import time
from typing import Iterable


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)  # scripts/build_marlin.py -> <repo>/
SOURCES = os.path.join(REPO, "src", "models", "ops", "cuda", "marlin_build", "sources")
DEFAULT_OUTPUT = os.path.join(REPO, "src", "models", "ops", "cuda", "lib")

# Sub-paths under SOURCES/
MARLIN_INC = os.path.join(SOURCES, "marlin")
CORE_INC = os.path.join(SOURCES, "core")
STABLE_INC = os.path.join(SOURCES, "libtorch_stable")


# ---------------------------------------------------------------------------
# CUDA toolchain
# ---------------------------------------------------------------------------
def _find_nvcc() -> str:
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        nvcc = os.path.join("/usr/local/cuda", "bin", "nvcc")
    if not os.path.exists(nvcc):
        raise FileNotFoundError(
            f"nvcc not found on PATH and not at {nvcc}. "
            f"Install CUDA toolkit (we test on 13.0)."
        )
    return nvcc


def _torch_paths() -> tuple[list[str], list[str], list[str]]:
    """Return (include_dirs, library_dirs, library_names) for torch.

    Uses ``torch.utils.cpp_extension``'s path resolution so we don't
    hard-code ``/usr/local/lib/python3.11/...``. The library list is
    what nvcc needs to resolve ``c10::``, ``at::`` symbols in
    ``marlin_template.h``.
    """
    from torch.utils.cpp_extension import include_paths, library_paths

    inc = list(include_paths())
    lib = list(library_paths())
    libs = ["c10", "c10_cuda", "torch_cpu", "torch_cuda", "torch", "torch_python"]
    return inc, lib, libs


def _cuda_include() -> str:
    inc = os.path.join("/usr/local/cuda", "include")
    if not os.path.exists(inc):
        nvcc = _find_nvcc()
        inc = os.path.join(os.path.dirname(os.path.dirname(nvcc)), "include")
    return inc


def _cuda_lib() -> str:
    lib = os.path.join("/usr/local/cuda", "lib64")
    if not os.path.exists(lib):
        nvcc = _find_nvcc()
        lib = os.path.join(os.path.dirname(os.path.dirname(nvcc)), "lib64")
    return lib


# ---------------------------------------------------------------------------
# Arch list
# ---------------------------------------------------------------------------
def _parse_archs(spec: str) -> list[str]:
    """Parse ``"80,89,120"`` -> ``["80", "89", "120"]`` (sorted, deduped)."""
    archs = sorted({a.strip() for a in spec.split(",") if a.strip()})
    if not archs:
        raise ValueError("no archs specified")
    for a in archs:
        if not a.isdigit() or not (50 <= int(a) <= 130):
            raise ValueError(f"invalid arch: {a!r} (expected numeric like 80/89/120)")
    return archs


def _detect_arch() -> list[str]:
    """Detect the current device's SM arch via nvidia-smi / torch."""
    try:
        import torch
        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability()
            return [f"{major}{minor}"]
    except Exception as e:  # noqa: BLE001 — fall through to nvidia-smi
        print(f"[build] torch detection failed: {e!r}", file=sys.stderr)
    smi = shutil.which("nvidia-smi")
    if smi is None:
        raise RuntimeError(
            "could not detect SM: no CUDA device via torch and no nvidia-smi. "
            "Pass --arch 80,89,120 explicitly."
        )
    out = subprocess.run(
        [smi, "--query-gpu=compute_cap", "--format=csv,noheader"],
        check=True, capture_output=True, text=True,
    )
    archs = sorted({
        line.strip().replace(".", "") for line in out.stdout.splitlines() if line.strip()
    })
    if not archs:
        raise RuntimeError("nvidia-smi returned no compute caps")
    return archs


# ---------------------------------------------------------------------------
# Build a single .so
# ---------------------------------------------------------------------------
def _gencode_flags(arch: str) -> list[str]:
    """nvcc ``-gencode=arch=compute_X,code=sm_X`` for SM ``X``.

    The single ``-gencode=arch=compute_X,code=sm_X`` (no PTX) is what we
    want — emitting a single SASS variant keeps the .so small and
    matches the loader's per-arch selection. PTX is not needed because
    the loader refuses to load a .so whose SM doesn't match the device.
    """
    return [f"-gencode=arch=compute_{arch},code=sm_{arch}"]


_BASE_NVCC_FLAGS = [
    "-O3",
    "--use_fast_math",
    "-std=c++17",
    "--expt-relaxed-constexpr",
    "--expt-extended-lambda",
    "-DUSE_CUDA",
    "-DMARLIN_NAMESPACE_NAME=marlin",
    # Undefine the operator/conversion removal flags so we get full
    # bfloat16 / fp8 / fp4 / half2 support. vllm does the same.
    "-U__CUDA_NO_HALF_OPERATORS__",
    "-U__CUDA_NO_HALF_CONVERSIONS__",
    "-U__CUDA_NO_HALF2_OPERATORS__",
    "-U__CUDA_NO_BFLOAT16_OPERATORS__",
    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
    "-U__CUDA_NO_BFLOAT162_OPERATORS__",
    "-U__CUDA_NO_BFLOAT162_CONVERSIONS__",
    "-U__CUDA_NO_FP8_OPERATORS__",
    "-U__CUDA_NO_FP8_CONVERSIONS__",
    "-U__CUDA_NO_FP4_OPERATORS__",
    "-U__CUDA_NO_FP4_CONVERSIONS__",
]


def _build_one(
    *,
    name: str,
    top_cu: str,
    arch: str,
    output_dir: str,
    build_dir: str,
    nvcc: str,
    inc: list[str],
    lib: list[str],
    libs: list[str],
    cuda_inc: str,
    cuda_lib: str,
    verbose: bool,
) -> str:
    """Compile one ``.cu`` -> ``.so``.

    Returns the path of the produced .so.
    """
    os.makedirs(build_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    obj = os.path.join(build_dir, f"{name}.cuda.o")
    so = os.path.join(output_dir, f"marlin_fp4_{name}_sm{arch}.so")

    # nvcc requires the source .cu to be at file scope (not via
    # -I) so #include "marlin/..." etc. resolve to SOURCES/marlin.
    # We ``-I SOURCES`` so the include paths line up.
    compile_cmd = [
        nvcc,
        *_BASE_NVCC_FLAGS,
        *_gencode_flags(arch),
        "-Xcompiler", "-fPIC",
        "-I", SOURCES,
        "-I", MARLIN_INC,
        "-I", CORE_INC,
        "-I", STABLE_INC,
        "-isystem", cuda_inc,
        *(f"-I{p}" for p in inc),
        "-c", top_cu,
        "-o", obj,
    ]
    if verbose:
        print(f"[nvcc -c] {' '.join(shlex.quote(c) for c in compile_cmd)}")

    t0 = time.perf_counter()
    subprocess.run(compile_cmd, check=True)
    t_compile = time.perf_counter() - t0

    # Link. nvcc is fine for shared-library link too, and is more
    # forgiving about the link order than ld.
    link_cmd = [
        nvcc,
        "-shared",
        "-L", cuda_lib,
        *(f"-L{p}" for p in lib),
        *(f"-l{l}" for l in libs),
        "-lcudart",
        obj,
        "-o", so,
    ]
    if verbose:
        print(f"[nvcc -l] {' '.join(shlex.quote(c) for c in link_cmd)}")

    t0 = time.perf_counter()
    subprocess.run(link_cmd, check=True)
    t_link = time.perf_counter() - t0

    size_mb = os.path.getsize(so) / (1024 * 1024)
    print(
        f"[done] {os.path.basename(so)}  "
        f"compile={t_compile:.1f}s  link={t_link:.1f}s  size={size_mb:.1f} MiB"
    )
    return so


# ---------------------------------------------------------------------------
# Build orchestration
# ---------------------------------------------------------------------------
# Each top-level .cu is a standalone translation unit that #includes
# everything it needs. Two top-level files cover the FP4 path:
#   - ``marlin_kernel_only.cu``  ->  ``marlin_fp4_kernel_only_sm{sm}.so``
#   - ``marlin_repack_adapter.cu`` -> ``marlin_fp4_repack_sm{sm}.so``
TARGETS = [
    ("kernel_only", "marlin_kernel_only.cu"),
    ("repack", "marlin_repack_adapter.cu"),
]


def _ensure_sources_present() -> None:
    """Sanity check that the vendored sources are where we expect."""
    for name, cu in TARGETS:
        p = os.path.join(SOURCES, cu)
        if not os.path.exists(p):
            raise FileNotFoundError(f"missing source: {p}")
    for p in (
        os.path.join(MARLIN_INC, "kernel.h"),
        os.path.join(MARLIN_INC, "marlin.cuh"),
        os.path.join(MARLIN_INC, "marlin_template.h"),
        os.path.join(MARLIN_INC, "gptq_marlin_repack_kernel_only.cu"),
        os.path.join(CORE_INC, "scalar_type.hpp"),
        os.path.join(STABLE_INC, "torch_utils.h"),
    ):
        if not os.path.exists(p):
            raise FileNotFoundError(f"missing vendored header: {p}")
    # Patches must be present (we patched the vllm headers to remove
    # iostream from marlin.cuh and add #pragma once to marlin_template.h).
    with open(os.path.join(MARLIN_INC, "marlin.cuh")) as f:
        cuh = f.read()
    # The original line is ``#include <iostream>``. After the patch
    # there's a comment mentioning the removal but no actual include
    # directive. Check for the include directive form.
    if "#include <iostream>" in cuh:
        raise RuntimeError(
            "marlin.cuh still has #include <iostream> — "
            "the patch to drop it is missing"
        )
    with open(os.path.join(MARLIN_INC, "marlin_template.h")) as f:
        head = f.read(1024)
    if "#pragma once" not in head:
        raise RuntimeError(
            "marlin_template.h missing #pragma once — the patch is missing"
        )


def _build_dir_for(arch: str) -> str:
    return os.path.join(SOURCES, ".build", f"sm{arch}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__.split("\n\n", 1)[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--arch",
        default="auto",
        help=(
            "Comma-separated list of SM arches to build for "
            "(e.g. '80,89,120'). Default: auto-detect from "
            "torch.cuda.get_device_capability() / nvidia-smi."
        ),
    )
    p.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help=f"Output directory for the .so files (default: {DEFAULT_OUTPUT})",
    )
    p.add_argument(
        "--build-dir",
        default=None,
        help=(
            "Where to put intermediate .o files "
            "(default: <sources>/.build/sm{arch}/). Reusing the same "
            "build-dir across arches is fine; they get different filenames."
        ),
    )
    p.add_argument(
        "--verbose", "-v", action="store_true",
        help="Print the full nvcc commands before running them.",
    )
    args = p.parse_args(argv)

    if args.arch == "auto":
        archs = _detect_arch()
        print(f"[build] auto-detected SM: {','.join(archs)}")
    else:
        archs = _parse_archs(args.arch)
        print(f"[build] requested SM: {','.join(archs)}")

    _ensure_sources_present()
    nvcc = _find_nvcc()
    inc, lib, libs = _torch_paths()
    cuda_inc = _cuda_include()
    cuda_lib = _cuda_lib()
    print(f"[build] nvcc: {nvcc}")
    print(f"[build] torch includes: {inc}")
    print(f"[build] torch lib dirs: {lib}")
    print(f"[build] cuda inc/lib: {cuda_inc} | {cuda_lib}")

    built: list[str] = []
    t0 = time.perf_counter()
    for arch in archs:
        build_dir = args.build_dir or _build_dir_for(arch)
        for name, cu in TARGETS:
            so = _build_one(
                name=name,
                top_cu=os.path.join(SOURCES, cu),
                arch=arch,
                output_dir=args.output,
                build_dir=build_dir,
                nvcc=nvcc,
                inc=inc,
                lib=lib,
                libs=libs,
                cuda_inc=cuda_inc,
                cuda_lib=cuda_lib,
                verbose=args.verbose,
            )
            built.append(so)
    total = time.perf_counter() - t0

    print(f"\n[build] DONE  total={total:.1f}s  {len(built)} files")
    for p in built:
        print(f"  {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
