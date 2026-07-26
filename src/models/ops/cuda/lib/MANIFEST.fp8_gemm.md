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
| `fp8_gemm_cutlass_sm_120.so`                                 | **CUTLASS-based** (sm_120 blockwise path, mirrors official 87a). Per-MmaTile 128x128x128 scales; reference / future option. Not on prod path. | `cutlass_fp8_blockwise_gemm_run` |

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

cuBLAS (`torch._scaled_mm`) reaches ~96-100 TFLOPS at these shapes
— the CUTLASS RowWise ceiling on sm_120 (which is **52% of the
188 TF hardware FP8 dense peak** measured via cuBLASLt nvjet
scalar-mode; NVIDIA vendor dense spec is 209 TF; see
`docs/fp8_gemm_landscape_2026_07_23.md` §1 for the full breakdown).
The custom kernel matches the RowWise CUTLASS path at the same shapes;
the +1-4% win comes from its fixed-shape dispatch and K-tail handling.

## Bottleneck analysis (2026-07-20 + 2026-07-23 re-baseline, see auto-memory `project_fp8_gemm_bottleneck.md`)

At the prod FFN shape, the kernel reaches ~100 TFLOPS — the CUTLASS
RowWise ceiling on sm_120 (NOT the hardware FP8 peak). This matches
cuBLAS CUTLASS RowWise on sm_120 (forced to CUTLASS path because
cuBLASLt RowWise returns 0 algos on Blackwell consumer, see
`cublaslt_no_rowwise_sm120.md`). The 100 TF result holds for
`_scaled_mm`, the PROD TMA kernel, and `_scaled_mm` across all K
dimensions (1536/2048/4096).

**The 100 TF is the CUTLASS rowwise ceiling, not the hardware peak.**
The 97 → 188 TF gap (1.9x headroom) is closable by switching the
prod path to a different scale mode (TensorWise scalar at large M,
or Blockwise 1x128) — gated on numerics being acceptable. The
188 → 209 TF gap is hardware-side (FP8 MMA cycle limit on Blackwell
consumer) and not closable via software.

**Occupancy is NOT the bottleneck.** All 2-blocks/SM candidates
(both `bm64_bn128_s2` and `bm128_bn64_s2` at 48 KB smem) measure
0.92-0.95x _scaled_mm — worse than the 1 block/SM PROD. The
launch_bounds (384, 1) is intentional: smem 96 KB at PROD
already forces 1 block/SM, and a 2-block attempt trades
pipeline depth for occupancy (NUM_STAGES=2 vs 3) and loses.

**At the CUTLASS rowwise ceiling; 3 structural levers all dead-ended
for pushing beyond.** Three potential levers were probed empirically
(2026-07-20):
Split-K (2-pass simulation 0.66-0.98x of 1-pass — kernel
launch + add overhead exceeds any parallelism win at prod
tile counts), `m16n8k64` E4M3 MMA (`ptxas` rejects —
"Incorrect instruction type" — not supported on sm_120), and
2 producer lanes for parallel TMA (TMA unit issues 1 per
warp-cycle anyway, and the `expect_tx` ordering across lanes
is racy → deadlock). All three are dead ends on sm_120 for
this kernel; PROD at 1.04x _scaled_mm = ~100 TFLOPS saturates
the CUTLASS rowwise ceiling. To go further on the same kernel
would need either B200 (sm_100 datacenter, ~3x fp8 throughput
per SM) or a different scale mode (not pursued here). See
auto-memory `project_fp8_gemm_bottleneck.md` for full analysis.

