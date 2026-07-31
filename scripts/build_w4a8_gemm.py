#!/usr/bin/env python3
"""Build the W4A8 native GEMM .so (FP8 A + NVFP4 B → FP8 MMA → BF16).

Different from build_fp8_blockwise_gemm.py in that:
  - Source file is `w4a8_gemm.cuh`, not `fp8_gemm.cuh`
  - No `<BM,BN,BK,...>` tokens to substitute (entry file instantiates
    a single template)
  - Single config: bm=128/BN=128/BK=128, NUM_STAGES=3, CWG=2,
    WARP_M=32, WARP_N=64, DIRECT_STORE=true (matches R10 strategy)
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SOURCES = REPO / "src/models/ops/cuda/fp8_gemm_build/sources"
BUILD = REPO / "src/models/ops/cuda/fp8_gemm_build/build"
LIB = REPO / "src/models/ops/cuda/lib"

# The entry file hard-codes a single (BM, BN, BK, NUM_STAGES, CWG,
# WARP_M, WARP_N, DIRECT_STORE) tuple via the FP8GemmMMA template
# instantiation. To swap tile shapes, edit w4a8_kernel_entry.cu.in
# (no build-config knobs on purpose — the kernel is tightly coupled
# to smem budget: 16 KB FP8 A + 9 KB NVFP4 B = 25 KB/stage, 3 stages
# = 75 KB, fits in 99 KB cap; smaller M dim breaks the BLOCK_TILE
# scale hoist that makes the chain flush free).
ENTRY = "w4a8_kernel_entry.cu.in"
EXPORT_SYMBOL = "w4a8_gemm_run"


def find_nvcc() -> str:
    nvcc = shutil.which("nvcc")
    if nvcc:
        return nvcc
    for candidate in ("/usr/local/cuda/bin/nvcc", "/usr/local/cuda-13.0/bin/nvcc"):
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError("nvcc not found")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", default="sm_120")
    args = parser.parse_args()
    nvcc = find_nvcc()

    BUILD.mkdir(parents=True, exist_ok=True)
    LIB.mkdir(parents=True, exist_ok=True)

    entry_template = (SOURCES / ENTRY).read_text()
    rendered = entry_template
    # The entry file doesn't use <BM,BN,BK,NUM_STAGES,CWG,WARP_M,WARP_N,
    # DIRECT_STORE> tokens — the w4a8_gemm.cuh template instantiation
    # hard-codes them inside the entry. So nothing to substitute.
    source = BUILD / f"w4a8_{args.arch}.cu"
    output = LIB / f"w4a8_gemm_{args.arch}.so"
    source.write_text(rendered)
    cmd = [
        nvcc, "-O3", "--use_fast_math", "-std=c++17", "-arch", args.arch,
        "--shared", "-Xcompiler", "-fPIC", "-I", str(SOURCES), "-lcuda",
        "-o", str(output), str(source),
    ]
    print("[build:w4a8]", " ".join(cmd))
    result = subprocess.run(cmd, text=True, capture_output=True)
    if result.returncode:
        print(result.stderr)
        return result.returncode
    print("[build:w4a8] wrote", output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())