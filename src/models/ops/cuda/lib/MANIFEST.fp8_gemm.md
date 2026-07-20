# FP8 E4M3 GEMM — per-config `.so` manifest

This directory holds **prebuilt FP8 E4M3 GEMM kernels** used as an
opt-in backend for the NVFP4 W4A8 forward (Triton dequant →
fp8 → this GEMM). The `.so` files are committed in-tree (not
gitignored) so a fresh checkout can run a forward step without
paying a ~30 s `nvcc` compile cost per config.

## Why per-config and not JIT

- The kernel is heavily templated (BM, BN, BK, NUM_STAGES, CWG,
  WARP_M, WARP_N, DIRECT_STORE) with explicit instantiations keyed
  on tile shape + warp count. Each combination emits a distinct
  SASS, and the best at prod FFN shapes (gate_up K=1536 N=8192,
  down K=4096 N=1536) is `bm128_bn128_bk128_s3_cwg2_wm32_wn64_ds`.
- Per-arch dispatch is automatic (sm120 for the 5060 Ti dev box).
- The Marlin FP4 .so path is the model; this follows the same
  pattern.

## Files

| File                                          | Purpose                                         | Loader entry   |
| --------------------------------------------- | ----------------------------------------------- | -------------- |
| `fp8_gemm_sm_120_bm128_bn128_bk128_s3_cwg2_wm32_wn64_ds.so` | **PROD** — gate_up + down, 1.01-1.04x cuBLAS | `gemm_run`     |
| `fp8_gemm_sm_120_bm64_bn64_bk128_s3_cwg1_wm32_wn32_ds.so`   | small-M fallback                                | `gemm_run`     |
| `fp8_gemm_sm_120_bm128_bn128_bk128_s2_cwg2_wm32_wn64.so`    | reference (no DIRECT_STORE, s=2)                 | `gemm_run`     |
| `fp8_gemm_sm_120_bm128_bn128_bk128_s2_cwg2_wm64_wn32.so`    | reference                                       | `gemm_run`     |
| `fp8_gemm_sm_120_bm64_bn64_bk128_s2_cwg1_wm32_wn32.so`      | reference                                       | `gemm_run`     |
| `fp8_gemm_sm_120_bm128_bn64_bk128_s3_cwg1_wm32_wn64_ds.so`  | reference                                       | `gemm_run`     |
| `fp8_gemm_sm_120_bm64_bn128_bk128_s3_cwg1_wm32_wn64_ds.so`  | reference                                       | `gemm_run`     |
| `fp8_gemm_sm_120_bm64_bn128_bk128_s2_cwg1_wm32_wn64_ds.so`  | reference (2 blocks/SM, 48 KB)                   | `gemm_run`     |
| `fp8_gemm_sm_120_bm128_bn64_bk128_s2_cwg1_wm32_wn64_ds.so`  | reference (2 blocks/SM, 48 KB)                   | `gemm_run`     |

Naming: `{sm}_{tile-config}` where the tile config encodes the warp
specialization layout. The `_ds` suffix means DIRECT_STORE epilogue
(drops the BM*BN*2 Y_out smem buffer → enables NUM_STAGES=3 for
BM=128/BN=128 under the 99 KB smem cap).

## Build provenance

The build pipeline is in `scripts/build_fp8_gemm.py`. Each committed
`.so` is built by running that script on a specific box with:

```
python scripts/build_fp8_gemm.py --arch sm_120
```

The build auto-detects the current GPU's SM arch if `--arch` is
omitted. To trim the .so set to just the two picked by
`src/models/ops/cuda/fp8_gemm.py` (`preferred` list), build with:

```
python scripts/build_fp8_gemm.py --config bm128_bn128_bk128_s3_cwg2_wm32_wn64_ds
python scripts/build_fp8_gemm.py --config bm64_bn64_bk128_s3_cwg1_wm32_wn32_ds
```

The committed .so set was built with CUDA 13.0 + nvcc, on a 5060 Ti
16G (sm_120) running torch 2.12 / driver 580.x.

## Sweep results (sm_120, prod FFN shapes)

| config                                          | gate_up 4.29 ms cuBLAS | down 2.08 ms cuBLAS |
| ----------------------------------------------- | ---------------------- | ------------------- |
| `bm128_bn128_bk128_s3_cwg2_wm32_wn64_ds` (PROD) | **4.12 ms (1.04x)**    | **2.06 ms (1.01x)** |
| `bm64_bn64_bk128_s3_cwg1_wm32_wn32_ds`          | 5.30 ms (0.81x)        | 2.40 ms (0.87x)     |
| `bm128_bn128_bk128_s2_cwg2_wm32_wn64`           | 4.34 ms (0.99x)        | 2.20 ms (0.95x)     |
| `bm64_bn128_bk128_s2_cwg1_wm32_wn64_ds` (2 blk) | 4.67 ms (0.92x)        | 2.24 ms (0.93x)     |
| `bm128_bn64_bk128_s2_cwg1_wm32_wn64_ds` (2 blk) | 4.66 ms (0.92x)        | 2.25 ms (0.93x)     |

cuBLAS (`torch._scaled_mm`) reaches ~96-100 TFLOPS at these shapes
(24-25% of the 400 TFLOPS fp8 spec on sm_120). The custom kernel
matches cuBLAS at the same shapes; the +1-4% win comes from
avoiding nvjet's small-K tail.

## Bottleneck analysis (2026-07-20, see auto-memory `project_fp8_gemm_bottleneck.md`)

At the prod FFN shape, the kernel reaches ~25% of the 400 TFLOPS
fp8 spec (100 TFLOPS) — this is the same ceiling as cuBLAS CUTLASS
RowWise on sm_120 (forced to CUTLASS path because cuBLASLt
RowWise returns 0 algos on Blackwell consumer, see
`cublaslt_no_rowwise_sm120.md`).

**Occupancy is NOT the bottleneck.** All 2-blocks/SM candidates
(both `bm64_bn128_s2` and `bm128_bn64_s2` at 48 KB smem) measure
0.92-0.95x _scaled_mm — worse than the 1 block/SM PROD. The
launch_bounds (384, 1) is intentional: smem 96 KB at PROD
already forces 1 block/SM, and a 2-block attempt trades
pipeline depth for occupancy (NUM_STAGES=2 vs 3) and loses.

**Per-block compute pipeline is the limit.** The kernel is
launch-bound at ~8 consumer warps/SM doing m16n8k32 E4M3 MMA.
Hitting more of the 400 TF spec likely requires a structural
change (2 producer threads for parallel TMA, m16n8k64 mma if
available on sm_120, or split-K) — all TODO, see
`project_fp8_gemm_bottleneck.md` for the next-up optimization
candidates.

