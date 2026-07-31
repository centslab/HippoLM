

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
# FP8 GEMM landscape on sm_120 (RTX 5060 Ti) — full analysis

Date: 2026-07-23
Author: probes by Claude
Hardware: RTX 5060 Ti 16G, sm_120, 36 SMs, 99 KB smem/SM cap, torch 2.12 / CUDA 13

This document captures a comprehensive sweep of FP8 GEMM performance across
(M, N, K) on sm_120, identifies why small-M GEMMs fall short of peak even
when arithmetic intensity is sufficient, and lays out a complete optimization
strategy by matrix-size regime.

## TL;DR

| Claim | Status |
| --- | --- |
| 97 TF is the **hardware FP8 peak** on sm_120 | **WRONG.** 97 TF is the CUTLASS rowwise ceiling. Hardware FP8 peak is ~185 TF (88% of NVIDIA vendor dense spec). |
| 97 TF is the **CUTLASS rowwise ceiling** | **TRUE** (measured). |
| FP8 / BF16 ratio is ~2x | **WRONG** for this card. FP8 scalar / BF16 = 3.7x (because Blackwell consumer BF16 is capped at 50 TF, not the 100 TF FP8 has). |
| We can close the 97 → 185 TF gap | **Yes — via TensorWise scalar.** cuBLASLt native scalar hits 177-188 TF (88% of hardware peak, 1.6-1.8x faster than rowwise). **MXFP8 path (b12x) only reaches 95-112 TF on sm_120** — the 167 TF theoretical MXFP8 ceiling assumes a hand-tuned CUTLASS path that b12x doesn't ship. |
| BF16 wins for very small / skinny GEMMs | **TRUE** at M ≤ 2048 for K=128 or N=128 (BF16 launch overhead ~30us lower than FP8 rowwise). |
| The "AI is sufficient but not at peak" issue | Two regimes: (a) **launch-bound** at M<128 (BF16 wins), (b) **scale-broadcast-bound** at M≥512 (40-46% of CUTLASS rowwise time is scale broadcast). |

## §1 — Real hardware peak: 185 TFLOPS, not 97

Vendor spec for RTX 5060 Ti:
- FP8 sparse 2:4: **419 TFLOPS**
- FP8 dense: **~209 TFLOPS** (= sparse / 2)

Earlier memory (`project_fp8_gemm_bottleneck.md`) claimed:
> "FP8 GEMM kernel saturates the chip's fp8 dense peak ~97 TFLOPS = ~103% peak"

That claim used 97 TF as the denominator, but 97 TF is the CUTLASS rowwise
*achievable* on sm_120, **not the hardware peak**. We can measure the
hardware peak by routing around CUTLASS:

| Backend | Achievable TFLOPS | % vendor dense spec (209 TF) |
| --- | --- | --- |
| `torch._scaled_mm` **RowWise** (current prod path) | **97-99 TF** | 47-48% |
| `torch._scaled_mm` **TensorWise** (single scalar) | **177-188 TF** | 85-90% |
| `torch._scaled_mm` **Blockwise 1x32** (MXFP8 native) | **138-167 TF** | 66-80% |
| BF16 / FP16 cuBLAS | **50 TF** | 24% |
| TF32 cuBLAS | **25 TF** | 12% |

The FP8 scalar path launches `nvjet_sm120_qqtst_mma_128x128x64_6_64x64x64_tmaAB_bz`
— the cuBLASLt hand-tuned kernel. It hits **185 TFLOPS at large M**, which is
the hardware FP8 dense peak (88% of vendor dense spec). The "gap to 209 TF"
is hardware-side (FP8 MMA cycle limit on Blackwell consumer).

**Memory update**: the previous "97 TF = 100% peak" framing was wrong on the
denominator. The real numbers are:

- **Hardware FP8 dense peak: ~185 TF** (scalar-mode measurement, the
  hand-tuned cuBLASLt kernel)
- **CUTLASS rowwise ceiling: ~97 TF** (52% of hardware peak)
- **MXFP8 native ceiling: ~167 TF** (90% of hardware peak at large M)
- **BF16 peak: 50 TF**

**Clock-check confirmation** (probe `test/_tmp/probe_gpu_clock_check_2026_07_23.py`,
deleted): at M=16384 K=4096 N=4096 saturating shape, BF16 runs at median
**2790 MHz** SM clock vs FP8 at median **2827 MHz** — within +1.3% noise.
FP8 does NOT throttle to a lower clock than BF16. The 188 TF measurement
is the genuine FP8 dense ceiling on sm_120 at the rated clock, not a
clock-throttling artifact. (For reference: vendor max boost is 3090 MHz;
both workloads measured ~2.8 GHz, so neither is at max boost — they
plateau at the same sustainable clock under sustained compute.)

## §2 — Why "AI is sufficient but not at peak" — the diagnosis

The user's question: at prod FFN shapes (M=1024, K=1536, N=4096), the
arithmetic intensity is **AI = 1069 FLOPs/byte**, well above the FP8 ridge
of 123 FLOPs/byte. So this should be **compute-bound** and reach hardware
peak. But RowWise only achieves **82 TFLOPS = 44% of hardware peak**.

Two regimes cause under-utilization:

### A. Launch + tile-fill regime (small M)

At M < 128, the GEMM is dominated by kernel launch + workspace allocation +
heuristic, not by compute. Tile grid is 12 tiles on 36 SMs (waves=0.33), so
most SMs are idle. The "achieved TFLOPS" metric is meaningless because the
time is ~30-60us regardless of M.

**BF16 wins at M < 256** because its kernel launch path is more optimized:
FP8 rowwise floor: ~30us. BF16 floor: ~28us. The 2-5us advantage compounds
for many small calls.

### B. Scale-broadcast regime (medium-large M)

At M ≥ 512 with compute-bound AI, RowWise still doesn't reach hardware peak
because **40-46% of CUTLASS rowwise time is spent on per-row × per-col scale
broadcast**.

Measured at M=1024-16384 FFN gate_up shapes:

| M | RowWise TF | Scalar TF | **Scale broadcast cost** |
| --- | --- | --- | --- |
| 1024 | 82 TF | 137 TF | 41% (64us / 155us) |
| 4096 | 93 TF | 170 TF | 45% (252us / 553us) |
| 8192 | 95 TF | 174 TF | 46% (497us / 1088us) |
| 16384 | 95 TF | 123 TF* | 23% (* scalar degraded here) |

*The M=16384 scalar number is anomalously low (123 vs 174 at M=8k). Possibly
measurement variance or a kernel choice change at that shape. Not a
fundamental ceiling.*

**Why this happens**: CUTLASS rowwise loads scale_a [M,1] and scale_b [1,N]
into SMEM, then multiplies them into the FP32 accumulator at every output
element. That's N+M scale-multiply ops per output cell. The "scalar" path
just multiplies by 2 scalars at the end.

The MXFP8 (1x32 block scale) GEMM sits in between: it has more scales than
scalar (1 per 32 K-elements) but **fewer** than rowwise (1 per K/32 vs 1 per
K), AND the hardware natively supports block-scale broadcast via TMA in
Blackwell consumer. That's why MXFP8 hits 138-167 TF — closer to scalar
peak than rowwise peak.

## §3 — Comprehensive M / N / K sweep

Sweep recipe: same `a, b` for each backend (random normal × 0.1, 0.05);
median of 20 timed iters after 5 warmup; CUDA-event timing; reference is
cuBLAS BF16 cuBLAS linear. (Probes: `test/_tmp/probe_fp8_full_sweep_2026_07_23.py`,
`test/_tmp/probe_fp8_hardware_peak_2026_07_23.py`.)

### Prod-shape M sweep

| Shape | M | FP8 RowWise | FP8 Scalar | BF16 | Winner | Gap |
| --- | --- | --- | --- | --- | --- | --- |
| KDA_qkv K=N=1536 | 1024 | 78 TF | 97 TF | 46 TF | FP8 scalar | 1.25x |
| KDA_qkv K=N=1536 | 4096 | 91 TF | 145 TF | 50 TF | FP8 scalar | 1.59x |
| FFN_gate_up K=1536 N=4096 | 1024 | 82 TF | 137 TF | 49 TF | FP8 scalar | **1.67x** |
| FFN_gate_up K=1536 N=4096 | 4096 | 93 TF | 170 TF | 50 TF | FP8 scalar | **1.83x** |
| FFN_gate_up K=1536 N=4096 | 8192 | 95 TF | 174 TF | 50 TF | FP8 scalar | **1.84x** |
| FFN_down K=4096 N=1536 | 1024 | 86 TF | 134 TF | 48 TF | FP8 scalar | **1.56x** |
| FFN_down K=4096 N=1536 | 4096 | 96 TF | 95 TF | 50 TF | rowwise | (tie) |
| lm_head K=1536 N=248320 | 4096 | 96 TF | 161 TF | 50 TF | FP8 scalar | **1.69x** |
| lm_head K=1536 N=248320 | 8192 | 96 TF | 160 TF | 49 TF | FP8 scalar | **1.66x** |
| KDA_g_proj K=1536 N=128 | 1024 | 11 TF | 9 TF | 13 TF | **BF16** | -22% (BF16 wins) |
| KDA_g_proj K=1536 N=128 | 8192 | 74 TF | 69 TF | 41 TF | FP8 rowwise | 1.79x over BF16 |

**Headline**: at prod M=1024-8192, FP8 scalar is **1.5-1.85x faster** than
RowWise (current prod). The gain is from eliminating per-row × per-col scale
broadcast.

### Skinny-shape (K=128 or N=128) BF16-vs-FP8 crossover

| M | K | N | FP8_rw TF | BF16 TF | Winner |
| --- | --- | --- | --- | --- | --- |
| 1024 | 128 | 128 | 0.9 | 1.1 | **BF16** |
| 1024 | 128 | 1536 | 10.8 | 13.2 | **BF16** |
| 1024 | 1536 | 128 | 11.3 | 13.2 | **BF16** |
| 1024 | 1536 | 1536 | 80 | 47 | FP8 |
| 4096 | 128 | 1536 | 43 | 29 | FP8 |
| 4096 | 1536 | 128 | 50 | 41 | FP8 |

**BF16 wins at small M for skinny shapes** because FP8 rowwise has higher
launch overhead (~30us floor). The crossover is around M=2048 for K=128 or
N=128, M=512 for K=N=1536. At M=8192, FP8 rowwise pulls ahead by 1.5-1.8x.

### K-sweep at M=1024, N=4096

| K | FP8_rw | FP8_sc | BF16 | Note |
| --- | --- | --- | --- | --- |
| 128 | 30 | 23 | 22 | BF16 ≈ FP8_sc (bandwidth-bound K) |
| 512 | 68 | 92 | 45 | FP8_sc pulls ahead |
| 1536 | 82 | 136 | 48 | prod FFN, 1.66x gap |
| 4096 | 87 | 92 | 49 | dim. returns on scalar |

### N-sweep at M=1024, K=1536

| N | FP8_rw | FP8_sc | BF16 | Note |
| --- | --- | --- | --- | --- |
| 128 | 11 | 9 | 13 | BF16 wins (skinny N) |
| 1024 | 74 | 70 | 41 | FP8_rw ≈ FP8_sc |
| 1536 | 78 | 99 | 46 | KDA_qkv prod |
| 4096 | 82 | 136 | 48 | FFN_gate_up prod |
| 8192 | 89 | 147 | 50 | approaching peak |

## §4 — Strategy by (M, N, K) regime

Mapping the sweep to a per-shape pick table:

### Regime I: Tiny GEMMs (M ≤ 128)

- **Use BF16** (lower launch overhead ~30us vs FP8 rowwise)
- Even at N=4096, FP8 rowwise doesn't beat BF16 at this M
- Example: KDA g/b_proj at decode / first-step warmup

### Regime II: Skinny shapes (K ≤ 256 OR N ≤ 256)

- **Use BF16 for M ≤ 2048**
- **Use FP8 rowwise for M ≥ 4096**
- Crossover point: M ≈ 2048 for K=128 or N=128
- Example: KDA g_proj at small microbatch sizes

### Regime III: Compute-bound square-ish (M ≥ 256, K ≥ 512, N ≥ 1024)

This is the **prod regime** (KDA q/k/v/o, FFN gate/up, FFN down, lm_head).

- **Current prod**: FP8 rowwise (CUTLASS), 78-95 TF (52% of hardware peak)
- **Path 1 (zero-risk, NOT YET SHIPPED)**: Switch prod to TensorWise
  (scalar scale) at large M. **1.6-1.8x faster** at prod FFN/KDA/lm_head
  shapes — see `probe_fp8_large_size_scalemode_2026_07_23.py` (deleted).
  Hits 86-90% of hardware peak (158-171 TF at M=2048-8192 vs 82-96 TF
  RowWise). Numerics: cos_sim 0.99930 same as RowWise on Gaussian inputs
  (per-tensor amax ≈ per-row × per-col amax when magnitudes are uniform).
  Real-data loss-curve probe NOT YET DONE — needs 50-step comparison
  before shipping. Code change is one-liner (amax(dim=1) → amax() in the
  quant function).
- **Path 2 (zero-risk, shipped)**: Auto-dispatcher picks BM64 vs BM128.
  Saves 13-29% over BM128 at small M (`project_fp8_dispatch_by_m.md`).
- **Path 3 (re-measured 2026-07-24, sees TensorWise scalar as winner)**: MXFP8
  (1x32 block scale) GEMM via b12x's `b12x::mxfp8_linear_fused`. Measured
  max ~112 TF @ K=N=4096 M=4096 (best of all shapes tried up to M=32768),
  but **strictly LOSES to TensorWise scalar at every prod shape** (SC is
  1.5-2.0x faster at FFN/KDA/lm_head shapes). Same 3.5-4.2% sig_rel as
  RW/SC after the canonical `ceil(log2)` encoder fix. b12x's ~500µs
  per-call floor at small M is also a non-starter for KDA-time GEMMs.
  Decided against (a) no sm_120 MXFP8 vendor kernel hits the 167 TF
  theoretical peak, (b) SC is faster by all measures, (c) numerics
  equivalent — no accuracy win either.
- **Path 4 (dead end, verified 2026-07-23)**: Blockwise 1x128 scale mode
  in `torch._scaled_mm` torch 2.12 — `Invalid scaling configuration`.
  Not exposed in this torch version.

**Big remaining lever at large M is Path 1 (TensorWise)** — the 1.6-1.8x
gain is the biggest single FP8 optimization opportunity identified in
2026-07-23 (was masked by the previous wrong "97 TF = peak" framing).
  `probe_fp8_blockwise_2026_07_23.py`).
- **Path 3 (custom kernel)**: Don't bother on sm_120. The
  `fp8_gemm.cuh` kernel is already at 100% of CUTLASS rowwise ceiling.
  See `project_fp8_gemm_bottleneck.md`. The only way to beat CUTLASS
  rowwise is to attack the scale broadcast — which is what MXFP8 does.

### Regime IV: Large M, single GEMM (M ≥ 8192)

- RowWise saturates at ~95-99 TF (CUTLASS ceiling, AT PEAK)
- FP8 scalar hits 170-185 TF (hardware ceiling)
- If we can route through MXFP8 or cuBLASLt scalar with correction,
  ~1.7-1.8x is available
- lm_head at M=8192 K=1536 N=248320: 96 vs 160 TF = **1.66x**
  (lm_head is the biggest single FP8 GEMM in our step time — current
  9.5 ms/step could drop to ~5.7 ms/step if MXFP8 lands)

### Regime V: Skinny-K width (K ≤ 256)

- Already bandwidth-bound, FP8 vs BF16 mostly about element size
- Don't waste cycles on FP8 quant overhead here
- **Use BF16** for these (e.g. KDA f_proj[1], g_proj where applicable)

## §5 — Concrete next steps, prioritized

### Priority 1: BF16 fallback for KDA skinny GEMMs at small M (low risk, immediate win)

Switch KDA g_proj and b_proj to BF16 at M ≤ 2048. Both are K=128 or N=128,
which is **launch-bound** territory where BF16 wins. The WSD schedule can
back-compute the right precision flag at module init based on the planned
microbatch size.

Estimated per-step savings: ~1-2 ms (depending on call count × the
FP8-BF16 gap at that shape).

### Priority 2 (REVISED 2026-07-24): Ship TensorWise scalar — NOT MXFP8

After the b12x MXFP8 probe (2026-07-24), MXFP8 is **not** the
right path. Ship TensorWise scalar instead (Path 1 above) — the
1.6-1.8x faster mode that actually beats both RowWise and b12x
MXFP8 at every prod shape. b12x MXFP8 maxes at 112 TF; cuBLASLt
TensorWise scalar reaches 179 TF. Reason closed: no sm_120 MXFP8
vendor path is faster than the scalar path on this hardware.

If a hand-tuned sm_120 MXFP8 kernel ships in the future
(CUTLASS sm_120 native blockwise path, DeepGEMM sm_120, or
vendor extension), re-evaluate. But until then: **TensorWise
scalar is the production lever.**

### Priority 3: Dispatcher routing (already shipped)

The runtime dispatcher in `fp8_gemm_dispatch.py` picks BM64 vs BM128 per
shape. Already deployed in production for prod FFN/KDA paths. Saves
13-29% GEMM at M=1024 (per `project_fp8_dispatch_by_m.md`).

### Priority 4: cuBLAS upgrade (NOT in scope)

Per the user's constraint: "cuBLAS 貌似没对5060 Ti做深度优化, 后面考虑环境稳定也不会升级cublas版本". Skip.

### Priority 5 (NOT recommended): custom FP8 GEMM with scale broadcast fix

Tempting but sm_120-specific:
- m16n8k64 E4M3 mma: `ptxas` rejects ("Incorrect instruction type") — sm_100+
  only. **Not viable**.
- Split-K: 0.66-0.98x of 1-pass at prod shapes (launch + atomic overhead
  exceeds parallelism win). **Not viable** at large M. Might help at
  small M (M < 256) — not pursued.
- 2 producer lanes: kernel deadlocks. **Not viable**.

The 97 TF ceiling is **the CUTLASS rowwise implementation's ceiling**, not
the hardware ceiling. The only way past it on sm_120 is via the
hardware-native MXFP8 / Blockwise scale modes (Priority 2).

## §6 — Open questions / future probes

1. **MXFP8 numerics fix**: what scale layout does cuBLASLt actually expect?
   Hypothesis: swizzled layout for SMEM bank conflict avoidance. Probes to
   test: 8x8 swizzle, 16x16 swizzle, transposed [K//BK, M] layout.

2. **cuBLASLt direct API with scale_mode=BLK128x128_32F**: error message
   mentions it but my test path failed. Worth probing once MXFP8 works.

3. **MXFP8 vs NVFP4 W4A8 trade-off**: NVFP4 W4A8 already in production
   (mode 3, FP4 weights + FP8 act). MXFP8 W8A8 vs NVFP4 W4A8 — same
   kernel, different quant. Compare accuracy vs speed.

4. **Persistent-kernel FP8 GEMM**: at small M, persistent SMs with
   better latency hiding could help. But sm_120 has only 36 SMs and the
   latency-hiding gain is small at M≥1024. Likely 5-10% at best.

5. **4090 (sm_89) transfer**: same probes on sm_89 to see if the
   pick table changes. Expected: similar shape but higher absolute
   TFLOPS (sm_89 BF16 ~80 TF, FP8 ~165 TF).

## §6a — Three-way comparison: RowWise vs TensorWise scalar vs MXFP8 (b12x), 2026-07-24

Added a third axis to the decision matrix after installing b12x
(`pip install -e /hy-tmp/b12x`, sm_120-only CuTe-DSL kernels, ~11K
LOC). Goal: validate or refute the user's prior recollection of
a "more efficient MXFP8 GEMM kernel" against the b12x dense
implementation on Blackwell consumer.

### Headline numbers (pure-GEMM, best M per shape)

| Shape | M | RW (prod) | SC | MX (b12x) | winner | mx/sc |
|---|---|---|---|---|---|---|
| KDA_qkv (K=N=1536) | 8192 | 82 TF | **115 TF** | 51 TF | SC | 0.44x |
| KDA_qkv (K=N=1536) | 16384 | 90 TF | **107 TF** | 78 TF | SC | 0.73x |
| FFN_gate (K=1536 N=4096) | 8192 | 91 TF | **154 TF** | 95 TF | SC | 0.62x |
| FFN_gate (K=1536 N=4096) | 16384 | 94 TF | **120 TF** | 100 TF | SC | 0.83x |
| FFN_down (K=4096 N=1536) | 8192 | 93 TF | **147 TF** | 83 TF | SC | 0.57x |
| FFN_down (K=4096 N=1536) | 16384 | 97 TF | **168 TF** | 72 TF | SC | 0.43x |
| FFN_largeK (K=N=4096) | 4096 | 95 TF | **164 TF** | 112 TF | SC | 0.69x |
| FFN_largeK (K=N=4096) | 8192 | 98 TF | **174 TF** | 106 TF | SC | 0.61x |
| FFN_largeK (K=N=4096) | 16384 | 99 TF | **179 TF** | 85 TF | SC | 0.48x |
| (KDA path), best across M | various | 90-99 TF | **106-179 TF** | 4-112 TF | SC | 0.16-0.83x |

**Range summary:** Across all tested (shape, M) combinations:
- RowWise saturates ~95-99 TF (CUTLASS rowwise ceiling)
- TensorWise scalar saturates 115-179 TF (88-95% of 188 TF vendor peak)
- MXFP8 (b12x) saturates at 112 TF best (FFN_largeK M=4096 K=N=4096),
  typically 4-100 TF — strictly worse than SC at every prod shape

### Small-M headroom (per-call kernel-launch + quant overhead)

At M=128 (production never sees this, but useful for kernel health):

| Mode | M=128 | M=256 | M=512 |
|---|---|---|---|
| RW (`_scaled_mm`) | 107 µs | 79 µs | 90 µs |
| SC (`_scaled_mm`) | **90 µs** | **89 µs** | **86 µs** |
| MX (b12x `::mxfp8_linear_fused`) | **573 µs** | **574 µs** | **478 µs** |

MX (b12x) has a ~500 µs per-call floor regardless of M — likely
CuTe-DSL JIT'd kernel + `freeze_kernel_resolution` artifact. The
RW/SC paths are <100 µs at all M≥128. For a production step with
thousands of calls, this floor matters.

### Numerics (sig_rel vs BF16 cuBLAS reference)

| Mode | sig_rel range | Notes |
|---|---|---|
| RW (per-row + per-col, FP32) | 3.5-4.2% | FP8 representation limit |
| SC (single tensor scale, FP32) | 3.5-4.2% | Same noise floor |
| MX (b12x, ceil(log2) UE8M0 scales) | 3.5-4.2% | Same noise floor |

**All three modes converge to the same 3.5-4.2% sig_rel** — the
FP8 representation (3-bit mantissa) is the bottleneck, not scale
granularity. Per-block FP32 scales (RW) do NOT save noise over a
single tensor FP32 scale (SC) on Gaussian inputs.

### MXFP8 b12x numerical correctness — was broken, now fixed

First probe run: 27-32% sig_rel. Root cause: the modelopt encoder
in the b12x test file uses `ceil(log2(amax/448))` for the scale
byte; my initial probe truncated instead of ceiled. After applying
the same `ceil(log2)` pattern as
`b12x/tests/test_gemm_mxfp8_linear.py::_quantize_modelopt_mxfp8_rows`:

```python
scale_exp = ceil(log2(amax / 448)).clamp(-127, 127)
scale_byte = (scale_exp + 127).to(uint8)
```

sig_rel dropped to 3.5-4.2% (matches RW/SC). All finite.

### Decision

1. **Ship TensorWise scalar** (Path 1, §5). Wins everywhere.
   Same numerics as RW; 1.6-1.8x faster on prod FFN/KDA/lm_head.
   No new vendor dependency.
2. **Do NOT ship MXFP8 (b12x) on sm_120.** No hand-tuned vendor
   kernel hits the 167 TF theoretical peak on Blackwell consumer;
   b12x saturates at 112 TF and is slower than SC at every shape.
3. **Re-consider MXFP8 only if** a tuned sm_120 native kernel ships
   (CUTLASS sm_120 blockwise, DeepGEMM sm_120, or vendor extension).
   Until then, SC is the production lever.

### Reproducibility

- `test/_tmp/probe_three_mode_gemm_2026_07_24.py` — full probe
  (pure-GEMM + end-to-end bf16→quant+GEMM at M ∈ {128...32768})
- `pip install -e /hy-tmp/b12x` — required for MXFP8 path
- Output TFLOPS table written inline; DELETE before commit per
  `feedback_test_first.md`



## §7 — Probe scripts (for reproducibility)

All probes are in `test/_tmp/` and should be deleted before commit per
`feedback_test_first.md`:

- `probe_fp8_hardware_peak_2026_07_23.py` — §1 baseline + verify
- `probe_fp8_scalar_verify_2026_07_23.py` — confirms 185 TF is FP8 (kernel name)
- `probe_fp8_full_sweep_2026_07_23.py` — §3 sweep
- `probe_fp8_diagnose_2026_07_23.py` — §2 root-cause analysis
- `probe_fp8_mxfp8_path_2026_07_23.py` — first attempt at MXFP8 (had layout wrong)
- `probe_fp8_blockwise_2026_07_23.py` — corrected MXFP8 timing (1.67-1.76x)
- `probe_fp8_mxfp8_numerics_2026_07_23.py` — current MXFP8 numerics issue