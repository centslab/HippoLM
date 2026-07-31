#!/usr/bin/env python3
"""Build the sm_120 FP8 E4M3 blockwise GEMM.

All variants are built from the same kernel template — selected by entry
file (different export name) and template specialization. The shipped set
is the mainline hybrid (1×128 BF16 A scale + TensorWise FP8 B, F16/F32
acc) plus the R7/R10 blockwise baselines and the legacy 1×32 R0.
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

# Tile shape (BM, BN, BK, NUM_STAGES, CWG, WARP_M, WARP_N, DIRECT_STORE).
# Shipped set (2026-07-31): only the configs the mainline wrappers/tests
# actually load. Sweep-only configs (bm64_bn128_s2, bm128_bn128_s2,
# bm64_bn128_s2_cwg2_wm32_wn32) were removed in the side-branch cleanup —
# their results are recorded in MEMORY.md.
BASE_CONFIGS = {
    "bm64_bn128_bk128_s3_cwg1_wm32_wn64_ds_blockwise": (64, 128, 128, 3, 1, 32, 64, True),
    # R7 (VALIDATED 2026-07-25): same BM=128/BN=128 as R6, NUM_STAGES=3
    # + DIRECT_STORE=true → smem = 3 × (32 KB A + 32 KB B) = 96 KB
    # (DIRECT_STORE frees 32 KB Y_out). Beats R6 at all 6 probe shapes:
    # +2.0% at 4096³ (97.5 vs 95.6 TF), +1.9% at 2048/4096/4096,
    # +0.6-1.0% at the M≥4096/K≥8192 L2-wall shapes. Requires the
    # consumer-block `else` guard in fp8_gemm.cuh (without it the
    # producer WG falls through and OOB-stores via the DS epilogue).
    "bm128_bn128_bk128_s3_cwg2_wm32_wn64_ds_blockwise": (128, 128, 128, 3, 2, 32, 64, True),
}

# (entry_template, export_symbol, dual_accum_flag, name_suffix)
# Shipped set (2026-07-31): only the variants the mainline wrappers/tests
# actually load. The R0-R11 sweep variants (dual, 1x64, 64x64, persistent,
# 128x128, 256x256, ksk128, accum_bsk256/512/1024, 32x32) and the dead-end
# hybrid flavors (a1d, b64, a1x128, rowpair, fp32scale, e8m0_b128) were
# removed in the side-branch cleanup — their results are in MEMORY.md.
ENTRY_VARIANTS = {
    "single": ("blockwise_kernel_entry.cu.in", "blockwise_gemm_run", False, ""),
    "accum_bsk128": ("blockwise_accum_bsk128_kernel_entry.cu.in", "blockwise_accum_bsk128_gemm_run", False, "_accum_bsk128"),
    # R10: F16 in-block accumulator (ACC_RAW_F16). 2x QMMA issue rate on
    # sm_120 → 168 TF at large shapes (1.7-3.1x over F32 R6/R7).
    # Overflow boundary documented in the entry file.
    "accum_bsk128_f16": ("blockwise_accum_bsk128_f16_kernel_entry.cu.in", "blockwise_accum_bsk128_f16_gemm_run", False, "_accum_bsk128_f16"),
    # TensorWise B variant of hybrid_a1x128: b_scale shape [K/128]
    # (scalar per K-block) instead of 2D [N/64, K/128]. Two export
    # symbols (bf16 + e8m0 A scale). Halves B scale tensor HBM and
    # removes n_tile indexing from flush load. **Mainline** (confirmed
    # 2026-07-31): 1×128 BF16 scale FP8 A + TensorWise FP8 B.
    "hybrid_a1x128_twb": ("hybrid_a1x128_twb_kernel_entry.cu.in", "hybrid_a1x128_twb_bf16_gemm_run", False, "_hybrid_a1x128_twb_bf16"),
    # Same as hybrid_a1x128_twb but F32 accumulator (ACC_RAW_F16=false).
    # R10's F16 in-block accumulator overflows for adversarial max-
    # magnitude FP8 inputs (sum of 128 raw products can hit 25.7M vs
    # F16 max 65504). The F32 path is the safe production default for
    # W8A8 — ~58% of R10's speed. See MEMORY.md "W8A8 F16-acc overflow
    # — strategy" (2026-07-31).
    "hybrid_a1x128_twb_f32": ("hybrid_a1x128_twb_f32_kernel_entry.cu.in", "hybrid_a1x128_twb_f32_bf16_gemm_run", False, "_hybrid_a1x128_twb_f32_bf16"),
}


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
    parser.add_argument(
        "--variant", choices=["single", "accum_bsk128", "accum_bsk128_f16", "hybrid_a1x128_twb", "hybrid_a1x128_twb_f32", "all"], default="all",
        help="single=1×32 R0, accum_bsk128=R7 (F32 acc), accum_bsk128_f16=R10 (F16 in-block acc, perf pick), hybrid_a1x128_twb=**mainline** 1×128 BF16 scale FP8 A + TensorWise FP8 B (F16 acc, perf pick), hybrid_a1x128_twb_f32=same layout but F32 acc (overflow-safe W8A8/W4A8 production default — see MEMORY.md W8A8 strategy), all=all shipped variants",
    )
    parser.add_argument("--config", choices=["all", *BASE_CONFIGS], default="all")
    args = parser.parse_args()
    nvcc = find_nvcc()
    selected_configs = BASE_CONFIGS if args.config == "all" else {args.config: BASE_CONFIGS[args.config]}
    selected_variants = ENTRY_VARIANTS if args.variant == "all" else {args.variant: ENTRY_VARIANTS[args.variant]}
    BUILD.mkdir(parents=True, exist_ok=True)
    LIB.mkdir(parents=True, exist_ok=True)
    for var_name, (template_file, export_symbol, dual_accum, suffix) in selected_variants.items():
        template = (SOURCES / template_file).read_text()
        for name, values in selected_configs.items():
            bm, bn, bk, stages, cwg, wm, wn, direct_store = values
            rendered = template
            for token, value in {
                "BM": bm, "BN": bn, "BK": bk, "NUM_STAGES": stages,
                "CWG": cwg, "WARP_M": wm, "WARP_N": wn,
                "DIRECT_STORE": "true" if direct_store else "false",
            }.items():
                rendered = rendered.replace(token, str(value))
            # The DUAL_ACCUM=true / false toggle is a template
            # specialization flag baked in by the entry file (which
            # instantiates FP8GemmMMA<...BLOCKWISE_SCALE=true,
            # DUAL_ACCUM=true/false>), so no per-config token
            # replacement is needed here.
            source = BUILD / f"{var_name}_{args.arch}_{name}.cu"
            output = LIB / f"fp8_blockwise_gemm_{args.arch}_{name}{suffix}.so"
            source.write_text(rendered)
            cmd = [
                nvcc, "-O3", "--use_fast_math", "-std=c++17", "-arch", args.arch,
                "--shared", "-Xcompiler", "-fPIC", "-I", str(SOURCES), "-lcuda",
                "-o", str(output), str(source),
            ]
            print(f"[build:{var_name}]", " ".join(cmd))
            result = subprocess.run(cmd, text=True, capture_output=True)
            if result.returncode:
                print(result.stderr)
                return result.returncode
            print(f"[build:{var_name}] wrote", output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())