"""Build the project-local CUTLASS-based FP8 GEMM shared library.

Mirrors the 87a example from the vendored CUTLASS at
/hy-tmp/cutlass/examples/87_blackwell_geforce_gemm_blockwise/87a_*.

Output: src/models/ops/cuda/lib/fp8_gemm_cutlass_sm_<arch>.so
"""
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path

CUTLASS_DIR = Path("/hy-tmp/cutlass")
PROJECT_DIR = Path(__file__).resolve().parents[1]
SOURCE = PROJECT_DIR / "src/models/ops/cuda/fp8_gemm_build/sources/cutlass_kernel.cu"
LIB_DIR = PROJECT_DIR / "src/models/ops/cuda/lib"


def arch_tag() -> str:
    import torch
    major, minor = torch.cuda.get_device_capability()
    return f"sm_{major}{minor}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", default=None, help="override arch (e.g. sm_120a)")
    args = parser.parse_args()

    if not CUTLASS_DIR.exists():
        print(f"FATAL: {CUTLASS_DIR} not found", file=sys.stderr)
        sys.exit(1)
    if not SOURCE.exists():
        print(f"FATAL: {SOURCE} not found", file=sys.stderr)
        sys.exit(1)

    LIB_DIR.mkdir(parents=True, exist_ok=True)

    if args.arch:
        arch = args.arch
    else:
        try:
            arch = arch_tag()
        except Exception as e:
            print(f"FATAL: cannot detect arch: {e}", file=sys.stderr)
            sys.exit(1)

    # sm_120 → sm_120a (Blackwell arch-specific). The collective-builder
    # path requires the "a" suffix to enable the SM_120 instructions.
    if arch == "sm_120":
        nvcc_arch = "sm_120a"
    else:
        nvcc_arch = arch

    out = LIB_DIR / f"fp8_gemm_cutlass_{arch}.so"

    cmd = [
        "/usr/local/cuda/bin/nvcc",
        "-std=c++17", "-O3", "-DNDEBUG", "--use_fast_math",
        "--expt-relaxed-constexpr",
        "-DCUTLASS_ENABLE_TENSOR_CORE_MMA=1",
        f"-arch={nvcc_arch}",
        "--shared",
        "-Xcompiler", "-fPIC",
        "-Xcompiler", "-Wno-deprecated-declarations",
        f"-I{CUTLASS_DIR}/include",
        f"-I{CUTLASS_DIR}/tools/util/include",
        f"-I{PROJECT_DIR}/src/models/ops/cuda/fp8_gemm_build/sources",
        str(SOURCE),
        "-o", str(out),
    ]

    print(" ".join(shlex.quote(c) for c in cmd))
    rc = subprocess.call(cmd)
    if rc != 0:
        print(f"FATAL: nvcc exit {rc}", file=sys.stderr)
        sys.exit(rc)
    print(f"OK -> {out}")


if __name__ == "__main__":
    main()
