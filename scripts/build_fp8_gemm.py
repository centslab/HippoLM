#!/usr/bin/env python3
"""Build the per-arch FP8 GEMM .so files used by the W4A8 NVFP4 FFN path.

What this produces
------------------
For each requested SM arch and tile configuration, one .so file is
emitted to::

    src/models/ops/cuda/lib/fp8_gemm_sm{sm}_{config}.so

Each .so exposes ``gemm_run`` (extern "C") with the same C ABI as the
NVFP4 Marlin path, so the Python wrapper in
:file:`src/models/ops/cuda/fp8_gemm.py` can bind via ctypes without
JIT. The kernel uses TMA + warp specialization + mbarrier pipeline
(pattern lifted from the sm120_gemm BF16 reference at
:file:`/hy-tmp/sm120_gemm/src/bf16_gemm.cuh`).

Why a custom build (and not JIT)
--------------------------------
Same reason as :file:`scripts/build_marlin.py` — the Marlin-style
``extern "C"`` ctypes binding is much faster to first-forward than
``torch.utils.cpp_extension.load_inline`` (which would trigger a
30+ minute nvcc compile on a 5060 Ti). The .so is prebuilt per arch
and looked up at first forward.

Usage
-----

::

    # Build for the current device (auto-detect SM):
    python scripts/build_fp8_gemm.py

    # Build a specific config:
    python scripts/build_fp8_gemm.py --config bm128_bn128_bk64_s3

    # List known configs:
    python scripts/build_fp8_gemm.py --list
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SOURCES = REPO / "src" / "models" / "ops" / "cuda" / "fp8_gemm_build" / "sources"
LIB_DIR = REPO / "src" / "models" / "ops" / "cuda" / "lib"
BUILD_DIR = REPO / "src" / "models" / "ops" / "cuda" / "fp8_gemm_build" / "build"

# (BM, BN, BK, NUM_STAGES, CWG, WARP_M, WARP_N, DIRECT_STORE). Each
# combo produces one .so. BK is fixed at 128 (one 128B swizzle span
# per smem row — see fp8_gemm.cuh static_assert). smem per stage =
# (BM + BN) * 128 bytes; with DIRECT_STORE the Y_out epilogue buffer
# is dropped so BM=128/BN=128/BK=128 fits NUM_STAGES=3 under the
# 99 KB dynamic-smem cap.
#
# Sweep results at prod FFN shapes on sm_120 (gate_up M=16k K=1536
# N=8192, down M=16k K=4096 N=1536; vs torch._scaled_mm):
#   bm128_bn128_bk128_s3_cwg2_wm32_wn64_ds : 1.01-1.04x  (BEST — prod)
#   bm64_bn64_bk128_s3_cwg1_wm32_wn32_ds   : 0.85-0.95x  (small-M)
# Other configs kept for reference / future tuning but no longer
# picked by the wrapper (see fp8_gemm.py preferred list).
KNOWN_CONFIGS: list[tuple[int, int, int, int, int, int, int, bool]] = [
    # name,                       BM, BN, BK, stages, cwg, wm, wn, ds
    ("bm128_bn128_bk128_s3_cwg2_wm32_wn64_ds",
                                  128, 128, 128, 3, 2, 32, 64, True),   # 96KB → 1.04x cuBLAS (PROD)
    ("bm64_bn64_bk128_s3_cwg1_wm32_wn32_ds",
                                   64,  64, 128, 3, 1, 32, 32, True),   # 48KB → small-M fallback
    # Reference configs (not picked by wrapper; kept for future tuning):
    ("bm128_bn128_bk128_s2_cwg2_wm32_wn64",
                                  128, 128, 128, 2, 2, 32, 64, False),
    ("bm128_bn128_bk128_s2_cwg2_wm64_wn32",
                                  128, 128, 128, 2, 2, 64, 32, False),
    ("bm64_bn64_bk128_s2_cwg1_wm32_wn32",
                                   64,  64, 128, 2, 1, 32, 32, False),
    ("bm128_bn64_bk128_s3_cwg1_wm32_wn64_ds",
                                  128,  64, 128, 3, 1, 32, 64, True),   # 72KB
    ("bm64_bn128_bk128_s3_cwg1_wm32_wn64_ds",
                                   64, 128, 128, 3, 1, 32, 64, True),   # 72KB
]


def _find_nvcc() -> str:
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        for cand in ("/usr/local/cuda/bin/nvcc", "/usr/local/cuda-13.0/bin/nvcc"):
            if os.path.exists(cand):
                return cand
        raise FileNotFoundError(
            "nvcc not found on PATH or at /usr/local/cuda/bin/nvcc. "
            "Install CUDA toolkit 12.8+ (sm_120 requires CUDA 12.8+)."
        )
    return nvcc


def _sm_arch(major: int, minor: int) -> str:
    return f"sm_{major}{minor}"


def _build_one(
    nvcc: str,
    name: str,
    BM: int, BN: int, BK: int, STAGES: int, CWG: int, WM: int, WN: int,
    DS: bool,
    sm: str,
) -> Path:
    out_dir = LIB_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    so_path = out_dir / f"fp8_gemm_{sm}_{name}.so"

    # Substitute the kernel_entry.cu.in template.
    entry_cu = BUILD_DIR / f"kernel_entry_{sm}_{name}.cu"
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    template = (SOURCES / "kernel_entry.cu.in").read_text()
    rendered = (
        template
        .replace("BM", str(BM))
        .replace("BN", str(BN))
        .replace("BK", str(BK))
        .replace("NUM_STAGES", str(STAGES))
        .replace("CWG", str(CWG))
        .replace("WARP_M", str(WM))
        .replace("WARP_N", str(WN))
        .replace("DIRECT_STORE", "true" if DS else "false")
    )
    entry_cu.write_text(rendered)

    cmd = [
        nvcc,
        "-O3",
        "--use_fast_math",
        "-std=c++17",
        "-arch", sm,
        "--shared",
        "-Xcompiler", "-fPIC",
        "-I", str(SOURCES),
        "-lcuda",            # cuTensorMapEncodeTiled (TMA descriptor API)
        "-o", str(so_path),
        str(entry_cu),
    ]
    print(f"[build] {name} for {sm} → {so_path.name}")
    print(f"  cmd: {' '.join(cmd)}")
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"  STDERR:\n{res.stderr}", file=sys.stderr)
        raise RuntimeError(f"nvcc build failed for {name} on {sm}")
    print(f"  ✓ {so_path}")
    return so_path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arch", default=None, help="sm_NN (e.g. sm_120); auto-detect if omitted")
    ap.add_argument("--config", default=None, help="config name to build (default: all known)")
    ap.add_argument("--list", action="store_true", help="list known configs and exit")
    args = ap.parse_args()

    if args.list:
        for cfg in KNOWN_CONFIGS:
            print(cfg[0])
        return 0

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

    configs = KNOWN_CONFIGS
    if args.config:
        configs = [c for c in KNOWN_CONFIGS if c[0] == args.config]
        if not configs:
            print(f"Unknown config: {args.config}", file=sys.stderr)
            print("Available:", [c[0] for c in KNOWN_CONFIGS], file=sys.stderr)
            return 1

    for name, BM, BN, BK, STAGES, CWG, WM, WN, DS in configs:
        try:
            _build_one(nvcc, name, BM, BN, BK, STAGES, CWG, WM, WN, DS, sm)
        except RuntimeError as e:
            print(f"  SKIP {name}: {e}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
