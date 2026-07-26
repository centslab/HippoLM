#!/usr/bin/env python3
"""Build the hand-written FP8 GEMM .so files for sm_120.

Each Round produces one `.so`:
    src/models/ops/cuda/lib/fp8_hand_<round>_sm{NN}.so

exposing ``fp8_hand_<round>_gemm_run`` (extern "C"). The kernel uses
per-MmaTile (1x1) FP32 scales matching CUTLASS 87a semantics.

Round history:
  - r0     : 128x128x128 MmaTile, 4 consumer warps (2x2, 64x64),
             NUM_STAGES=2, no warp specialization. Baseline.
  - r1min  : R0 + ONLY NUM_STAGES=3 (isolate the variable). 50-500x
             slower — closed.
  - r2     : R0 + 8 consumer warps (2x4, 64x32), NUM_STAGES=2. Active.

Auto-discovers all ``fp8_gemm_hand_*.cu`` files in the sources dir
and builds each. Adds a new `.cu` to that dir to register a new round.

Usage:
    python scripts/build_fp8_hand_gemm.py
    python scripts/build_fp8_hand_gemm.py --arch sm_120
    python scripts/build_fp8_hand_gemm.py --only r2
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
HAND_SOURCES = REPO / "src" / "models" / "ops" / "cuda" / "fp8_gemm_hand_build" / "sources"
FP8_SOURCES = REPO / "src" / "models" / "ops" / "cuda" / "fp8_gemm_build" / "sources"
LIB_DIR = REPO / "src" / "models" / "ops" / "cuda" / "lib"


def _find_nvcc() -> str:
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        for cand in ("/usr/local/cuda/bin/nvcc", "/usr/local/cuda-13.0/bin/nvcc"):
            if os.path.exists(cand):
                return cand
        raise FileNotFoundError("nvcc not found")
    return nvcc


def _sm_arch(major: int, minor: int) -> str:
    return f"sm_{major}{minor}"


def _build_one(nvcc: str, cu_path: Path, sm: str) -> bool:
    LIB_DIR.mkdir(parents=True, exist_ok=True)
    # fp8_gemm_hand_r2.cu -> fp8_hand_r2_sm_120.so
    stem = cu_path.stem  # e.g. "fp8_gemm_hand_r2"
    round = stem.removeprefix("fp8_gemm_hand_")  # "r2", "r1min", "r1", "r0"
    so_path = LIB_DIR / f"fp8_hand_{round}_{sm}.so"
    cmd = [
        nvcc,
        "-O3",
        "--use_fast_math",
        "-std=c++17",
        "-arch", sm,
        "--shared",
        "-Xcompiler", "-fPIC",
        "-I", str(HAND_SOURCES),
        "-I", str(FP8_SOURCES),
        "-lcuda",
        "-o", str(so_path),
        str(cu_path),
    ]
    print(f"[build] {round} for {sm} -> {so_path.name}")
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"  FAILED: STDERR:\n{res.stderr}", file=sys.stderr)
        return False
    print(f"  OK: {so_path}")
    return True


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--arch", default=None, help="sm_NN (e.g. sm_120); auto-detect if omitted")
    ap.add_argument("--only", default=None,
                    help="Build only this round (e.g. r2); default: build all")
    args = ap.parse_args()

    if args.arch:
        sm = args.arch
    else:
        try:
            import torch
            major, minor = torch.cuda.get_device_capability()
            sm = _sm_arch(major, minor)
        except Exception as e:
            print(f"Could not auto-detect SM arch: {e}", file=sys.stderr)
            return 1

    nvcc = _find_nvcc()

    if not HAND_SOURCES.exists():
        print(f"Missing sources dir: {HAND_SOURCES}", file=sys.stderr)
        return 1

    # Discover all fp8_gemm_hand_*.cu
    cu_files = sorted(HAND_SOURCES.glob("fp8_gemm_hand_*.cu"))
    if args.only:
        cu_files = [p for p in cu_files if p.stem.endswith(f"_{args.only}")
                    or p.stem == f"fp8_gemm_hand_{args.only}"]
        if not cu_files:
            print(f"No source for round {args.only} in {HAND_SOURCES}", file=sys.stderr)
            return 1

    if not cu_files:
        print(f"No fp8_gemm_hand_*.cu in {HAND_SOURCES}", file=sys.stderr)
        return 1

    n_ok = 0
    for cu_path in cu_files:
        ok = _build_one(nvcc, cu_path, sm)
        if ok:
            n_ok += 1

    print(f"\n[{n_ok}/{len(cu_files)}] built successfully for {sm}")
    return 0 if n_ok == len(cu_files) else 1


if __name__ == "__main__":
    sys.exit(main())