#!/usr/bin/env python3
"""Build the sm_120 FP8 E4M3 + BF16 1x32 block-scale GEMM.

Both single-accumulator (R0) and dual-accumulator (R5) variants are built
from the same kernel template — selected by entry file (different export
name) and template specialization (DUAL_ACCUM=true in the R5 entry).
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
# The 64x128 tile leaves enough register/shared-memory headroom for the
# per-K-block partial-accumulator rescale. BK=128 contains four 1x32 blocks.
BASE_CONFIGS = {
    "bm64_bn128_bk128_s3_cwg1_wm32_wn64_ds_blockwise": (64, 128, 128, 3, 1, 32, 64, True),
    "bm64_bn128_bk128_s2_cwg1_wm32_wn64_blockwise": (64, 128, 128, 2, 1, 32, 64, False),
    "bm128_bn128_bk128_s2_cwg2_wm32_wn64_blockwise": (128, 128, 128, 2, 2, 32, 64, False),
    # R7 (VALIDATED 2026-07-25): same BM=128/BN=128 as R6, NUM_STAGES=3
    # + DIRECT_STORE=true → smem = 3 × (32 KB A + 32 KB B) = 96 KB
    # (DIRECT_STORE frees 32 KB Y_out). Beats R6 at all 6 probe shapes:
    # +2.0% at 4096³ (97.5 vs 95.6 TF), +1.9% at 2048/4096/4096,
    # +0.6-1.0% at the M≥4096/K≥8192 L2-wall shapes. Requires the
    # consumer-block `else` guard in fp8_gemm.cuh (without it the
    # producer WG falls through and OOB-stores via the DS epilogue).
    "bm128_bn128_bk128_s3_cwg2_wm32_wn64_ds_blockwise": (128, 128, 128, 3, 2, 32, 64, True),
    # WARP_N=32 config: enables BLOCK_OUT_N=32 (32×32 BLOCK_TILE_SCALE
    # with FP32 scales — see blockwise_32x32_kernel_entry.cu.in). CWG=2
    # WARPS_PER_WG=4 = 8 consumer warps, WARPS_M=BM/WARP_M=2 WARPS_N=4,
    # BM=2*32=64 BN=4*32=128.
    "bm64_bn128_bk128_s2_cwg2_wm32_wn32_blockwise": (64, 128, 128, 2, 2, 32, 32, False),
}

# (entry_template, export_symbol, dual_accum_flag, name_suffix)
ENTRY_VARIANTS = {
    "single": ("blockwise_kernel_entry.cu.in", "blockwise_gemm_run", False, ""),
    "dual": ("dual_accum_kernel_entry.cu.in", "dual_accum_gemm_run", True, "_dual"),
    "1x64": ("blockwise_1x64_kernel_entry.cu.in", "blockwise_1x64_gemm_run", False, "_1x64"),
    "64x64": ("blockwise_64x64_kernel_entry.cu.in", "blockwise_64x64_gemm_run", False, "_64x64"),
    "64x64_persistent": ("blockwise_64x64_persistent_kernel_entry.cu.in", "blockwise_64x64_persistent_gemm_run", False, "_64x64_persistent"),
    "128x128": ("blockwise_128x128_kernel_entry.cu.in", "blockwise_128x128_gemm_run", False, "_128x128"),
    "256x256": ("blockwise_256x256_kernel_entry.cu.in", "blockwise_256x256_gemm_run", False, "_256x256"),
    "64x64_ksk128": ("blockwise_64x64_ksk128_kernel_entry.cu.in", "blockwise_64x64_ksk128_gemm_run", False, "_64x64_ksk128"),
    "accum_bsk128": ("blockwise_accum_bsk128_kernel_entry.cu.in", "blockwise_accum_bsk128_gemm_run", False, "_accum_bsk128"),
    "accum_bsk256": ("blockwise_accum_bsk256_kernel_entry.cu.in", "blockwise_accum_bsk256_gemm_run", False, "_accum_bsk256"),
    "accum_bsk512": ("blockwise_accum_bsk512_kernel_entry.cu.in", "blockwise_accum_bsk512_gemm_run", False, "_accum_bsk512"),
    "accum_bsk1024": ("blockwise_accum_bsk1024_kernel_entry.cu.in", "blockwise_accum_bsk1024_gemm_run", False, "_accum_bsk1024"),
    # R10: F16 in-block accumulator (ACC_RAW_F16). 2x QMMA issue rate on
    # sm_120 → 168-169 TF at large shapes (1.7-3.1x over F32 R6/R7).
    # Overflow boundary documented in the entry file.
    "accum_bsk128_f16": ("blockwise_accum_bsk128_f16_kernel_entry.cu.in", "blockwise_accum_bsk128_f16_gemm_run", False, "_accum_bsk128_f16"),
    # 32×32 BLOCK_TILE_SCALE with FP32 scales (SCALE_FP32=true). Only
    # valid for the bm64_bn128_bk128_s2_cwg2_wm32_wn32 config because
    # the static_asserts require WARP_N >= BLOCK_OUT_N. See
    # blockwise_32x32_kernel_entry.cu.in for the precision rationale.
    "32x32": ("blockwise_32x32_kernel_entry.cu.in", "blockwise_32x32_gemm_run", False, "_32x32"),
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
        "--variant", choices=["single", "dual", "1x64", "64x64", "64x64_persistent", "128x128", "256x256", "64x64_ksk128", "accum_bsk128", "accum_bsk256", "accum_bsk512", "accum_bsk1024", "accum_bsk128_f16", "32x32", "all"], default="all",
        help="single=1×32 R0, dual=1×32 R5 (dual-accumulator), 1x64=R5b-strip (1×64 per-row×per-col), 64x64=R5b-square (per-(m_tile, n_tile), **production pick**), 64x64_persistent=PERSISTENT work-stealing variant of 64x64 (1 CTA/SM, atomic tile counter — **experimental, REJECTED 2026-07-25: 0.96× strided + 74% med_rel**, see memory project_fp8_persistent_rejected), 128x128/256x256=coarser 2D-block sweep points, 64x64_ksk128=64×64 tile with full-BK K-block (LDG floor), 32x32=fine-grained 32×32 BLOCK + FP32 scales (precision-tight, see blockwise_32x32_kernel_entry.cu.in), all=all variants",
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
            # 32x32 entry requires WARP_N ≤ BLOCK_OUT_N (=32) — the
            # kernel static_asserts BLOCK_OUT_N >= WARP_N. Skip
            # configs that wouldn't satisfy it instead of failing the
            # whole build.
            if var_name == "32x32" and wn != 32:
                continue
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