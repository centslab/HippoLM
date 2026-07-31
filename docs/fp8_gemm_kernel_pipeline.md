

> **⚠️ 2026-07-27 audit**: TF claims in this document were measured with
> the buggy cudaEvent single-event pattern, which under-reports
> ctypes-loaded .so kernel time by ~2.07x on sm_120. **Real numbers**
> (verified via batched cudaEvent + torch.profiler): R10 ~81 TF,
> nvjet TensorWise ~85 TF (real, not buggy), CUTLASS MXFP8 ~85 TF
> in practice (theoretical 112 TF). Cross-kernel ratios preserved.
> NVIDIA vendor specs (188 TF, 209 TF) are unaffected — they come
> from NVIDIA datasheets, not local measurements. See memory
> `feedback_cudaevent_2x_underreport_2026_07_27.md` and the
> `.claude/skills/bench-flops/SKILL.md` skill for the audit + correct
> methodology.
# FP8 E4M3 GEMM kernel pipeline

How the FP8 GEMM custom kernel is built, why we ship prebuilt `.so`
per arch instead of JIT, how the runtime auto-dispatcher picks the
best backend per (arch, M, K, N), and the procedure for adapting to
a new GPU arch. Read this before tuning the FP8 compute path on a
new machine.

**Skill**: [`kda-correctness-sweep`](../.claude/skills/kda-correctness-sweep/SKILL.md) — the
correctness sweep gate applies *before* this pipeline runs.
**Rule**: [`einsum-noncontig-triton`](../.claude/rules/einsum-noncontig-triton.md) — the
prebuilt `.so` does not have this trap, but new kernels you write do.

## What this is

Two pieces:

1. **`src/models/ops/cuda/fp8_gemm.py`** — the binding for one specific
   prebuilt `.so` (the legacy single-config path used by the W4A8
   opt-in `use_custom_gemm=True`).
2. **`src/models/ops/cuda/fp8_gemm_dispatch.py`** — the runtime
   auto-dispatcher that picks the best backend per `(M, K, N)` shape
   on the current arch. Public API:
   `fp8_gemm_auto_dispatch(A, B, a_scale, b_scale, out=...)`. Used
   when migrating the W4A8 / W8A8 production path off `_scaled_mm`.

The kernel itself (`fp8_gemm.cuh`) is a TMA + warp-specialized
`m16n8k32` E4M3 MMA on sm_120 (Blackwell consumer) with multi-stage
`mbarrier` pipeline. The kernel is **at the CUTLASS rowwise ceiling**
at large M (~100 TFLOPS = 53% of the **hardware** FP8 dense peak
~188 TF); see auto-memory `project_fp8_gemm_bottleneck.md` for the
bottleneck analysis, and `docs/fp8_gemm_landscape_2026_07_23.md` §1
for the 209 / 188 / 97 / 50 TF breakdown. The remaining upside is
**at small M**, where the per-call backend choice (BM=64 vs BM=128)
is the load-bearing lever — that's what the dispatcher automates.

**Block-scale scale precision (SCALE_FP32).** The `fp8_gemm.cuh`
blockwise variants (64×64 / 128×128 / 256×256 / `accum_bsk*` /
`32x32`) take per-block scales via the `SCALE_FP32` template param.
The **64×64 R5b² variant** (the blockwise throughput winner at
~91 TF) ships with `SCALE_FP32=true` (FP32 scales) as of
2026-07-25 — see auto-memory `project_fp8_issue_bound.md` §
"64×64 BLOCK → FP32 scales" for the full rationale. Briefly:
QMMA `m16n8k32.f32.e4m3.e4m3` on sm_120 has **FP32-only
accumulator** (no .f16 / .bf16 variant on this arch), so passing
FP32 scales matches the accumulator precision and removes any
ambiguity in the per-block accumulator chain. Cost: ~1% perf
from 2× HBM for scales (which is small enough — scales are ~1/64
the size of the FP8 A/B tiles per K-tile). **Important**: BF16→FP32
scale conversion is **bit-exact** (BF16 mantissa 7 bits ≤ FP32
mantissa 23 bits), so the precision benefit of FP32 vs BF16
scales is theoretical (matches accumulator type), not measured
— on synthetic clamped-Gaussian the med_rel is 0.135% for both.
Larger-block variants (128×128 / 256×256) keep BF16 scales
because blocks cover 2M+ FP8 elements per scale, well past the
BF16 mantissa precision floor.

## Per-arch baseline — what "peak" means on this arch

sm_120 has two FP8 ceilings that matter, not one:

- **Hardware FP8 dense peak** (the actual FP8 ceiling): **~188 TF**
  measured via `torch._scaled_mm` TensorWise (scalar scale) at large
  M; NVIDIA vendor dense spec is 209 TF. The cuBLASLt `nvjet_sm120`
  hand-tuned kernel reaches 88% of vendor.
- **CUTLASS rowwise ceiling** (current prod path): **~97-100 TF** at
  the same shape — the gap to 188 TF is **scale-broadcast overhead**
  (40-46% of CUTLASS rowwise time at prod FFN shapes).

Quick measurement recipe for the *hardware* peak (use this as the
denominator for "% of FP8 peak"):

```
# Baseline: a single M=16384 N=4096 K=4096 FP8 GEMM via _scaled_mm
# with TensorWise (scalar) scale. This hits the nvjet hand-tuned
# kernel = the hardware FP8 dense ceiling on sm_120.
#
# For other archs (best-known measurements):
#   sm_80 (A100):       ~220 TF (vendor dense spec ≈ 312 TF)
#   sm_89 (RTX 4090):   ≈ 165 TF (vendor ≈ 330 TF dense)
#   sm_90 (H100):       ≈ 360 TF
#   sm_100 (B200):      ≈ 660 TF
#   sm_120 (5060 Ti):   ≈ 188 TF (vendor 209 TF dense)
```

For the *CUTLASS rowwise* measurement (current prod path), the
recipe is the same shape but with `_scaled_mm` RowWise or
`_scaled_mm` Blockwise 1x128 (the rowwise-scale modes); that's
~97-100 TF on sm_120.

The earlier versions of this doc (pre-2026-07-23) used 97 TF as
"100% peak" — that was wrong. The 97 TF number is the CUTLASS
rowwise achievable, not the hardware ceiling. The auto-memory
`project_fp8_gemm_bottleneck.md` was re-baselined 2026-07-23 to
reflect this.

## Build pipeline

`scripts/build_fp8_gemm.py` is the build entry point. Per-arch SASS,
one `.so` per `(BM, BN, BK, NUM_STAGES, CWG, WARP_M, WARP_N,
DIRECT_STORE)` specialization:

```bash
# Build for the auto-detected arch.
python scripts/build_fp8_gemm.py

# Force a specific arch (e.g. on a CI farm with a different GPU).
python scripts/build_fp8_gemm.py --arch sm_89

# Build only the two configs the dispatcher currently prefers
# (see fp8_gemm.py::preferred).
python scripts/build_fp8_gemm.py --config bm128_bn128_bk128_s3_cwg2_wm32_wn64_ds
python scripts/build_fp8_gemm.py --config bm64_bn64_bk128_s3_cwg1_wm32_wn32_ds
```

**Why prebuilt, not JIT.** Per `feedback_autotune_production.md`:
production kernels are hardcoded config — the `.so` files ARE the
hardcoded config. The dispatcher just picks between them on first
call. JIT compile at first use would cost ~30 s per shape, which is
intolerable in a training loop. (Triton dispatch via
`torch.compile` is a different lever, not pursued here.)

**Why per-config.** Each `(BM, BN, BK, NUM_STAGES, CWG, WARP_M,
WARP_N, DIRECT_STORE)` combination emits a distinct SASS. The
right config at prod FFN shapes is `bm128_bn128_bk128_s3_cwg2_wm32_
wn64_ds` on sm_120 (per `lib/MANIFEST.fp8_gemm.md`); this is *not*
derivable from theory and must be measured.

## Sweep procedure on a new arch

```bash
# 1. Run the canonical sweep (tracked recipe — ships with the repo at
#    scripts/probes/, no longer lives in test/_tmp).
PYTHONPATH=/hy-tmp/HippoLM python scripts/probes/probe_fp8_small_m_sweep.py
```

What this sweep does and why every line matters:

| Step | Why |
| --- | --- |
| `for M in [64, 128, 256, 512, 1024, 2048, 4096, 8192]` | Covers sub-BM64 inputs, the MBS-shrink trajectory, and the large-M regression anchor. |
| For each `(M, K, N)`: 3 prod shapes (KDA qkv, FFN gate_up, FFN down) | Each shape has a different N tile-utilization behavior; a single shape is not representative. |
| Per cell, benchmark every prebuilt `.so` (skip the ones whose `BM`, `BN`, `BK` don't divide `M`, `K`, `N`) | The kernel hard-asserts (`fp8_gemm.cuh:567-569`), so we filter in Python rather than crash on launch. |
| Median of 10 timed iters per backend (5 warmup) | Matches `feedback_verify_correctness.md` and the regression-guard style in `test_fp8_gemm.py`. |
| Correctness: max-diff vs cuBLAS BF16 reference, isfinite check | Per `feedback_verify_correctness.md` — "CHUNK=32 silently NaN" lesson: every sweep must include the isfinite gate. |

## Interpreting results

**Metric: absolute TFLOPS, not % of FP8 peak.** "% of peak" depends
on what your arch's peak FLOPS is, which is itself a measurement. The
load-bearing comparison is **FP8 vs cuBLAS BF16 at the same M**. The
FP8-over-BF16 ratio is stable at 1.7-2.0x across M from 128 to 8192
on sm_120; this ratio is also stable across archs in the same
generation. If the ratio collapses below 1.5x at any M, the
dispatcher isn't picking the right config — re-sweep with more
candidate `.so` files.

**Pick rule: trust the micro-bench, not a fixed M threshold.** The
auto-memory `project_fp8_dispatch_by_m.md` sweep on sm_120 showed:

- For N=1536 family (KDA qkv, FFN down): BM64 wins below M≈2048
  (the 12 col tiles + small row tiles under-fill the 36 SMs at BM=128).
- For N=4096 family (FFN gate up): BM128 wins almost everywhere
  (32 col tiles already fill the SMs).

This rule is **shape-dependent**, not just M-dependent. The original
draft of the memory entry said "BM64 wins at M<2048" — wrong,
overgeneralized. The dispatcher's first-call micro-bench handles
this correctly. Don't try to encode threshold heuristics in
production code; trust the timing.

**Large-M regression guard.** Any new config you ship must not
regress at M≥4096 vs the existing PROD. The dispatch test
`test/test_fp8_gemm_dispatch.py::test_dispatch_no_regression_at_
large_m` enforces this. If your new config is slower at large M by
more than 5%, it stays out of the dispatcher set — ship it for
narrow-M use only.

## Dead ends on sm_120 (don't re-explore without re-justifying)

Three structural levers were probed 2026-07-20 (see
`project_fp8_gemm_bottleneck.md`); all dead ends:

1. **2 producer lanes** (lane 0 TMA A, lane 1 TMA B in same warp):
   kernel deadlocks at M=128 smoke. `expect_tx` ordering across
   lanes is racy. Don't try.
2. **m16n8k64 E4M3 mma**: `ptxas` rejects on sm_120 ("Incorrect
   instruction type"). The instruction is sm_100+ only (B200
   datacenter). Don't try on Blackwell consumer.
3. **Split-K** (atomic-add across K splits): 0.66-0.98x of 1-pass
   at prod shapes; launch + reduce overhead exceeds any parallelism
   win at prod tile counts. Not prebuilt in any `.so`; future work
   if needed for very small M (where Split-K's parallelism gain
   could outweigh its overhead — unproven).

## Numerics gates (per `feedback_verify_correctness.md`)

Every new dispatched backend must pass before it enters the set:

- `isfinite(y).all()` — non-finite is a hard fail. CHUNK=32-style
  silent NaN history (auto-memory `project_attn_res_bwd.md`)
  applies.
- `median_rel(y, y_ref) < 5%` — FP8 cast noise is 3-4% typical, so
  5% is the safety floor. The dispatch test enforces this via
  `test_dispatch_matches_scaled_mm`.

The reference is `torch._scaled_mm`, not fp32 dequant. The two
should match within fp32 accumulator rounding; differences ≥5%
signal a scale-application bug in the new backend.

## Wiring in code

The dispatcher is opt-in today. To migrate a `_scaled_mm` callsite
to the dispatcher:

```python
# Old:
out = torch._scaled_mm(A_q, B_q.T, a_scale, b_scale, out_dtype=torch.bfloat16)

# New:
from src.models.ops.cuda.fp8_gemm_dispatch import fp8_gemm_auto_dispatch
out = fp8_gemm_auto_dispatch(A_q, B_q, a_scale, b_scale, out=out)
```

Same output dtype, same inputs. The dispatcher swallows the
per-arch discovery + first-call micro-bench; downstream callers see
no API change.

Plumbing targets for the migration (none touched at doc time):
- `src/models/ops/fp8_linear.py` — the FP8 W8A8 fwd+bwd
- `src/models/ops/nvfp4_linear_w4a8.py:884` — the FFN W4A8
  pass-2 GEMM (currently behind `use_custom_gemm=True` opt-in)

First wiring target is the W4A8 FFN pass-2 — it's the per-step
hot-spot, and the dispatcher's `ffn_gate_up` M=1024 pick recovers
~17% of the GEMM cost (per the auto-memory sweep) with zero
correctness risk (the test guards `rel_median < 5%` at that
shape).

## Adding a new config

```bash
# 1. Edit scripts/build_fp8_gemm.py to add the new (BM, BN, ...) tuple
#    to the build matrix (default for that arch).
# 2. Build only the new config:
python scripts/build_fp8_gemm.py --config bm<X>_bn<Y>_...
# 3. Run the sweep; verify at least one (M, K, N) cell picks it.
PYTHONPATH=/hy-tmp/HippoLM python scripts/probes/probe_fp8_small_m_sweep.py
# 4. Re-run the dispatch tests (correctness gate, regression guard):
PYTHONPATH=/hy-tmp/HippoLM pytest test/test_fp8_gemm_dispatch.py -v
# 5. The dispatcher's _list_arch_configs() picks the new .so
#    automatically — no Python edits needed.
```

## What NOT to do

- Don't `fp8_gemm_scaled`-swap as a "perf win" without re-running
  the sweep on the active shape. The legacy
  `fp8_gemm_scaled` returns the first entry of `preferred`, which
  is `bm128_..._ds` on sm_120. Use
  `fp8_gemm_auto_dispatch` instead.
- Don't add a `--tile-config` CLI flag in `train.py`. The
  auto-dispatcher picks per shape; making the user override the
  choice hides correctness regressions and removes the
  micro-bench's self-tuning.
- Don't bump torch / CUDA back to 2.9.x / 12.8 / 570.x "to recover
  V100 compatibility". See `.claude/rules/dont-target-v100.md` —
  V100 was dropped 2026-07-08; reintroducing it would also drop
  W4A16/W8A8 (which require sm_80+).
- Don't JIT-compile a per-arch .so from Python in production. The
  ~30 s nvcc cost on a CUDA kernel update would block training and
  would also leave the kernel untrackable. Build on CI, ship the
  `.so`.
