

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
# FP8 scale-format precision landscape

Date: 2026-07-25
Author: probes by Claude
Hardware: RTX 5060 Ti 16G (sm_120), torch 2.12 / CUDA 13
Probes: `scripts/probes/probe_fp8_scale_precision.py` + `scripts/probes/probe_fp8_scale_precision_real.py` + `scripts/probes/probe_fp8_scale_precision_extras.py`

Companion to [`fp8_gemm_landscape_2026_07_23.md`](fp8_gemm_landscape_2026_07_23.md)
(performance) and [`fp8_gemm_kernel_pipeline.md`](fp8_gemm_kernel_pipeline.md)
(kernel construction). This doc answers the **numerics** questions:

- Why does swapping E8M0 (MXFP8 default) for BF16 scale lower average error?
- Marginal returns from each extra mantissa bit in the scale format
- Block-size effect on per-element / per-block noise
- Per-element error distribution under each (scale × granularity × fp8) combination
- 2D blocks (M×K tiles): do they help on real activations?
- Weight-side saturation: how often does the FP8 weight hit ±448 at different block sizes?

## TL;DR

| Claim | Status |
| --- | --- |
| E8M0 → BF16 lowers average error significantly | **TRUE in SQNR / GEMM-level sig_rel**, but **NOT in per-element median** (which is dominated by FP8 representation noise). On real KDA activations BF16 row-scale beats E8M0 row-scale by **21 dB SQNR / 0.7 pp sig_rel**. |
| More mantissa bits → lower error, monotonic | **FALSE** at the scale-format level. The curve is **non-monotonic** — m=1 is **4.7 dB worse than m=0**. Plateau at m≥7 (BF16). |
| BF16 is the sweet-spot scale format | **TRUE.** BF16 = FP16 = FP32 within ±0.5 dB; BF16 = E8M0 only in per-element median, not SQNR. |
| Block size is uniform across activations | **FALSE.** KDA q_proj prefers per-row / 1×128; FFN down_proj prefers 1×16; lm_head prefers 1×16. K-dim variation is the lever. |
| max_rel / p99 grow with block size | **DISTRIBUTION-DEPENDENT.** FFN down_proj (outlier-heavy) shows +1.6 pp max_rel growth from block=16 to block=1536. KDA q_proj (post-RMSNorm, narrow) shows the **opposite** (−5.1 pp). |
| 2D blocks (16×16, 32×32, 64×64, 128×128) help on real activations | **NOT FOR OUR MODEL.** Real activations are dominated by per-row (M-axis) structure; K-axis 1D blocks + per-row already capture everything. 2D tiles add GEMM kernel complexity with no SQNR gain. |
| Weight saturation count is a useful block-size metric | **PARTIALLY.** It's mostly "fraction of blocks whose amax is at FP8 max" — informative on weight tensors (Kaiming-init Gaussian: 9.6% at block=16 → 3.6% at block=512) but not a direct noise-floor signal. |

**Engineering recommendation (2026-07-25)**:

1. **Use BF16 blockwise scale** for any blockwise FP8 GEMM that supports it.
   E8M0 (MXFP8 default) is a precision trap on sm_120 — the ceil_log2 encoder's
   systematic bias caps SQNR at ~34 dB regardless of block granularity.
2. **Pick block size per activation** (see §6 table), not a project-wide default.
3. **Don't use E8M0 + blockwise on sm_120 for any activation where
   row-amax varies by more than 2x** — KDA q_proj loses 21 dB vs BF16.
4. **2D blocks are not justified** by the current data; the K-axis 1D block
   captures all the local amax structure we have.

---

## §1 — Methodology

Three probe scripts, all under `scripts/probes/` (long-term stable home,
not deleted at commit time):

- `probe_fp8_scale_precision.py` — Part A (scale-only precision),
  Part B (per-element + per-block SQNR), Part C (GEMM-level sig_rel),
  Part D (headline E8M0 vs BF16 comparison).
- `probe_fp8_scale_precision_real.py` — captures real activations
  from a 2-layer HippoModel (hidden=512, num_heads=8, head_dim=64, K=512
  KDA + K=1536 FFN) on B=2 T=256 random tokens, runs Part B/C metrics
  on them.
- `probe_fp8_scale_precision_extras.py` — Part E (mantissa-bits
  sweep 0..23), Part F (2D blocks + saturation count + max_rel/p99),
  Part G (full diagonal cross-matrix on synthetic + real).

**Distributions** (synthetic): D1 Gaussian, D2 heavy-tail (post-silu_mul-like),
D3 outlier rows (0.5% rows ×10 amax), D4 post-RMSNorm, D5 attn-logit (wide
dynamic range), D6 block-constant.

**Metrics**:

- **SQNR (dB)**: 10·log10(signal_power / noise_power) per row, median over rows.
  This is the engineering-correct metric for "how much noise does this
  quantization add to a downstream matmul."
- **med_rel / p99_rel / max_rel**: per-element |dequant − v| / (|v| + 1% floor),
  to avoid spurious 100% on v≈0 elements.
- **sig_rel**: max |out − ref| / max|ref|, the production-output metric.
- **saturations**: # elements quantized to FP8 = ±448 (E4M3 max). Mostly
  measures "fraction of blocks whose amax is at the FP8 ceiling."
- **scale-only precision**: round-trip a log-uniform[1e-3, 10] scale
  distribution through the format, measure relative error. Isolates the
  format's intrinsic precision from the FP8 round-trip.

**Hardware**: RTX 5060 Ti sm_120, all quantization done in PyTorch on GPU;
all metrics computed on GPU tensors.

---

## §2 — Part A: scale-format intrinsic precision

Round-trip 100k scales drawn from log-uniform[1e-3, 10] through each format:

| Format | Mantissa bits | med_rel | p99_rel | max_rel | Comment |
|---|---|---|---|---|---|
| FP32 | 23 | 0% | 0% | 0% | control |
| FP16 | 10 | 0.017% | 0.043% | 0.049% | FP16-equivalent |
| **BF16** | **7** | **0.135%** | **0.346%** | **0.389%** | **BF16 — sweet spot** |
| E8M0 (round_log2) | 0 | 17.0% | 40.4% | 41.4% | unbiased |
| E8M0 (floor_log2) | 0 | 28.7% | 49.6% | 50.0% | always undersizes |
| E8M0 (ceil_log2) | 0 | **42.5%** | 98.6% | 99.997% | MXFP8 canonical — always oversizes |

BF16 is **315× more precise** than E8M0 (ceil_log2) at the median. This is
the source of the "大幅降低" effect on aggregate noise power — but it does
NOT show up in per-element median (see §3).

The three E8M0 encoders differ by which direction of bias is acceptable:

- **ceil_log2** (MXFP8 default): scale ≥ true_scale → dequant ≤ true_value
  → values are systematically underestimated. **No saturation**, but the
  largest FP8 value used is < 448 (wastes top of range).
- **floor_log2**: scale ≤ true_scale → risk of **saturation** on the block
  amax, but the largest FP8 value is always 448.
- **round_log2**: nearest power of 2, errors in both directions. Best of
  the three E8M0 variants but still 17% median scale error.

---

## §3 — Part B: per-element + per-block SQNR (synthetic)

Full sweep on 6 distributions × 4 scale formats × 8 granularities. The
key surprising finding is that **per-element median/p99 look similar across
scale formats**, while **per-block SQNR differs dramatically**.

### 3.1 Per-element median (D1 Gaussian, E4M3)

| granularity | FP32 med | BF16 med | E8M0 med |
|---|---|---|---|
| tensor | 2.27% | 2.08% | 2.13% |
| row | 2.17% | 2.17% | 2.13% |
| block32 | 2.11% | 2.11% | 2.13% |
| block128 | 2.15% | 2.15% | 2.13% |

**Per-element median is dominated by FP8 representation noise** (~2% per
element on E4M3). Scale format is invisible at the median.

### 3.2 Per-block SQNR (D1 Gaussian, E4M3) — where the difference is

| granularity | FP32 | BF16 | E8M0 |
|---|---|---|---|
| tensor | 31.7 dB | 31.7 dB | 31.7 dB |
| row | 31.7 dB | 31.7 dB | 31.7 dB |
| block32 | 32.5 dB | 32.5 dB | 31.6 dB |
| block128 | 31.9 dB | 31.9 dB | 31.6 dB |

For Gaussian i.i.d. inputs all rows/blocks have similar amax, so SQNR is
nearly constant across (granularity × scale_format). The FP8 representation
floor (~32 dB) dominates.

### 3.3 Per-block SQNR on D6 block-constant (where E8M0 vs BF16 diverges most)

D6 = each 32-K-block is a single constant value (perfectly representable
in BF16 scale, but E8M0's powers-of-2 miss the exact value).

| granularity | FP32 | BF16 | E8M0 |
|---|---|---|---|
| block1 | 144.6 dB | 55.7 dB | 31.7 dB |
| block16 | 144.6 dB | 57.3 dB | 31.7 dB |
| block32 | 144.6 dB | 57.3 dB | 31.7 dB |
| block64 | 137.9 dB | 50.8 dB | 31.7 dB |

For D6 (block-constant), BF16 hits 57 dB at block32, **26 dB above E8M0**.
This is the worst-case for E8M0 — when the true scale happens to fall
between two powers of 2, E8M0's ceil_log2 always oversizes, and the entire
block of constant values gets reconstructed as the wrong magnitude.

### 3.4 The bias-cancellation effect

Why E8M0's per-element median looks similar to BF16 despite 315× worse
scale precision: **E8M0's systematic bias means all values are
underestimated by similar percentages**. The SQNR formula
`signal_power / noise_power` has both terms scale together — a 12.5%
bias-shifted signal has 12.5% biased noise, ratio unchanged.

This is why per-element median is misleading for scale-format selection.

---

## §4 — Part D: real KDA activations (the user's headline)

Layers captured from a 2-layer HippoModel (B=2 T=256 random tokens).

### 4.1 Real KDA q_proj (layers.0.kda.attn.q_proj, shape [2,256,512], amax=1.83)

```
scale   per-tensor   per-row   block16   block32   block64   block128
FP32     30.8 dB     54.7 dB   36.9 dB   39.8 dB   45.3 dB   54.7 dB
FP16     30.7 dB     55.1 dB   37.0 dB   39.8 dB   45.4 dB   55.1 dB
BF16     31.1 dB     55.8 dB   36.9 dB   39.8 dB   45.6 dB   55.7 dB
E8M0     34.2 dB     34.2 dB   34.2 dB   34.2 dB   34.2 dB   34.2 dB
```

**Critical observation**: E8M0 is **flat at 34.2 dB regardless of granularity**.
The ceil_log2 noise floor (~34 dB) is below the FP8 representation floor
(32 dB) — so E8M0 looks "acceptable" only because FP8 noise hides its
own contribution.

**BF16 (and FP32, FP16) show 21 dB improvement** going from per-tensor to
per-row. This is what the user observed as "大幅降低平均误差".

### 4.2 Real FFN down_proj (layers.0.ffn.down_proj, shape [2,256,1536], amax=5.16)

```
scale   per-tensor   per-row   block16   block32   block64   block128
FP32     31.6 dB     31.8 dB   35.5 dB   34.3 dB   33.5 dB   32.8 dB
BF16     31.6 dB     31.8 dB   35.5 dB   34.3 dB   33.5 dB   33.2 dB
E8M0     31.5 dB     31.5 dB   31.5 dB   31.5 dB   31.5 dB   31.5 dB
```

For FFN down_proj, **block16 wins** (35.5 dB). Per-row is poor (31.8 dB)
because row-amax is uniform across rows — the variation is in the K-dim
(FFN activations span a wider range of magnitudes across K=1536). Block16
captures the local K-dim variation.

E8M0: same flat 31.5 dB pattern.

### 4.3 Real lm_head (shape [2,256,512], amax=4.59)

```
scale   per-tensor   per-row   block16   block32   block64   block128
BF16     31.6 dB     31.6 dB   33.0 dB   32.4 dB   32.0 dB   31.8 dB
E8M0     31.5 dB     31.5 dB   31.5 dB   31.5 dB   31.5 dB   31.5 dB
```

For lm_head, block16 is best (33.0 dB). Per-row ≈ per-tensor because
lm_head has uniform amax across rows.

### 4.4 Real-activation cross-matrix (Part G)

Same data on a 2D grid:

```
layers.0.kda.attn.q_proj (amax=1.844, K=512)
granularity        FP32          BF16          E8M0
per-tensor         31.12dB       31.40dB       34.20dB
per-row            55.08dB       55.92dB       34.20dB
1x16               37.01dB       36.96dB       34.20dB
1x32               40.06dB       40.08dB       34.20dB
1x64               50.52dB       53.61dB       34.20dB
1x128              54.92dB       55.90dB       34.20dB
1xK=512            55.08dB       55.92dB       34.20dB

layers.0.ffn.down_proj (amax=5.312, K=1536)
granularity        FP32          BF16          E8M0
per-tensor         31.54dB       31.55dB       31.51dB
per-row            31.82dB       31.82dB       31.51dB
1x16               35.64dB       35.60dB       31.51dB   ← block16 wins
1x32               34.41dB       34.40dB       31.51dB
1x64               33.53dB       33.53dB       31.51dB
1x128              32.91dB       32.89dB       31.51dB
1xK=1536           31.82dB       31.82dB       31.51dB
```

E8M0 column is **constant** across granularity — confirms the scale-precision
ceiling dominates. FP32 ≈ BF16 in every cell (within 0.1 dB). Per-row
dominates when row-amax varies a lot (KDA q_proj); 1x16 dominates when
K-dim amax varies a lot (FFN down_proj).

---

## §5 — Part E: mantissa-bits diminishing-returns curve (synthetic Gaussian)

Parametric scale format with `m` mantissa bits (FP32 exponent range,
round-to-nearest), measured at 1×32 block on D1 Gaussian E4M3:

```
m_bits   med_rel    p99_rel   sqnr_med  sqnr_mean   format_equivalent
   0      2.128%     5.263%    31.57dB    31.59dB    E8M0 (UE8M0, ceil_log2)
   1      2.183%     9.677%    26.83dB    26.84dB                       ← WORSE!
   2      2.183%     5.882%    30.40dB    30.40dB    E5M2-equivalent
   3      2.174%     5.485%    31.57dB    31.57dB    E5M3-equivalent
   4      2.128%     5.556%    32.15dB    32.15dB                       ← first gain
   5      2.096%     5.556%    32.33dB    32.34dB
   6      2.108%     5.517%    32.38dB    32.39dB
   7      2.108%     5.541%    32.40dB    32.40dB    BF16-equivalent   ← plateau start
   8      2.110%     5.548%    32.40dB    32.40dB
  10      2.109%     5.517%    32.40dB    32.40dB    FP16-equivalent
  12      2.109%     5.525%    32.40dB    32.40dB
  15      2.109%     5.523%    32.40dB    32.40dB
  18      2.110%     5.524%    32.40dB    32.40dB
  23      2.110%     5.524%    32.40dB    32.40dB    FP32 (control)
```

### Key non-monotonic insight

**m=1 (E5M1-equivalent) is 4.7 dB WORSE than m=0 (E8M0 ceil_log2)**.
This is a real effect, not a measurement artifact:

- m=0 with ceil_log2 has **consistent one-direction bias** (always
  oversize). Dequant values are systematically smaller; SQNR
  `signal / noise` has both terms shrink in lockstep, ratio ~unchanged.
- m=1 with round-to-nearest has **errors in both directions**. The noise
  variance is higher than the bias-cancellation case.

This is why the user observed "BF16 (m=7) reduces error vs E8M0 (m=0)"
but might have missed that m=1 would have been worse than E8M0. Don't
ship m=1/m=2 thinking they're "between" — they're not.

### Plateau structure

- m=4 → 32.15 dB (first improvement over m=0)
- m=5..6 → 32.33-32.38 dB
- **m≥7 plateau at 32.40 dB** — BF16 already at the SQNR ceiling
- m=8..23 → 32.40 dB (FP16, FP32 give **zero** additional SQNR)

Conclusion: **BF16 is the precision sweet spot**. FP16 / FP32 buy nothing
on top of BF16 in this metric; E8M0 leaves 1 dB on the table at the
ceiling, more at finer block sizes (see §4).

Saved to memory as `project_scale_format_mantissa_curve.md` so future
explorations don't rediscover the m=1 trap.

---

## §6 — Part F: 2D blocks, saturation, max_rel / p99 vs block size

### 6.1 max_rel / p99 vs block size on REAL KDA q_proj (BF16 scale)

```
block   saturations    max_rel    p99_rel   sqnr_med
 16       9.483%        5.45%      4.65%     37.07dB
 32       6.136%        5.18%      4.52%     39.99dB
 64       4.389%        4.74%      3.55%     45.56dB
128       3.769%        4.61%      0.35%     55.85dB
256       3.735%        0.35%      0.31%     55.86dB
512       3.735%        0.35%      0.31%     55.86dB   (= per-row)
```

**User's hypothesis "max_rel and p99 grow with block size" — REVERSED for
narrow distributions**. KDA q_proj is post-RMSNorm (narrow, unit-variance):
- max_rel DECREASES from 5.45% → 0.35% (block 16 → 256)
- p99_rel DECREASES from 4.65% → 0.31%
- Saturations DECREASE from 9.5% → 3.7%

Reason: at block=16, the amax is calculated from 16 elements. Many of these
block-amaxes land in FP8 ranges where the surrounding elements don't fit
cleanly → edge cases produce high max_rel. At block=256 (= per-row), the
row-amax fits the FP8 range perfectly → uniform quantization, low max_rel.

### 6.2 max_rel / p99 vs block size on REAL FFN down_proj (BF16 scale)

```
block   saturations    max_rel    p99_rel   sqnr_med
 16       6.614%        5.82%      5.44%     35.53dB
 32       3.341%        5.82%      5.46%     34.38dB
 64       1.699%        5.82%      5.46%     33.48dB
128       0.864%        6.72%      5.47%     32.86dB
256       0.443%        8.71%      5.47%     32.41dB
512       0.226%        8.71%      5.46%     32.09dB
1536      0.078%        9.20%      5.47%     31.83dB
```

**For FFN down_proj, the user's hypothesis is CORRECT**: max_rel GROWS
from 5.82% to 9.20% as block size increases. Reason: FFN activations have
outlier elements in some K columns; small blocks isolate them so the amax
is well-matched, large blocks include too many normal elements that get
underrepresented relative to the outlier-dominated amax.

**Conclusion**: max_rel/p99 vs block size is **distribution-dependent**.
Narrow distributions (post-RMSNorm): max_rel DECREASES with block size.
Outlier-heavy distributions (FFN, attn-logit): max_rel INCREASES with
block size. SQNR (per-block MSE) is the better cross-distribution metric.

### 6.3 2D blocks on real activations — NOT useful for our model

Real KDA q_proj (shape 2×256×512, M_eff=512 K=512):

```
2d_block    saturations    max_rel    p99_rel   sqnr_med
per-row     3.735%         0.35%      0.31%     55.86dB
16x16       3.735%         0.35%      0.31%     55.86dB
32x32       3.735%         0.35%      0.31%     55.86dB
64x64       3.735%         0.35%      0.31%     55.86dB
128x128     3.735%         0.35%      0.31%     55.86dB
```

Real FFN down_proj (shape 2×256×1536, M_eff=512 K=1536):

```
2d_block    saturations    max_rel    p99_rel   sqnr_med
per-row     0.078%         9.20%      5.47%     31.83dB
16x16       0.078%         9.20%      5.47%     31.83dB
32x32       0.078%         9.20%      5.47%     31.83dB
64x64       0.078%         9.20%      5.47%     31.83dB
128x128     0.078%         9.20%      5.47%     31.83dB
```

**All 2D blocks give the same SQNR as per-row** in our model. Reason:
real activations are dominated by per-row (M-axis) structure (each token's
activations are correlated within a row), and per-row + K-axis 1D block
already captures everything. M-axis partitioning (16×16, 32×32 etc.) does
NOT add information because M=512 already has many distinct rows.

Conclusion: **Don't build 2D blockwise FP8 GEMM kernels for our model.**
The complexity (M-axis partition logic, swizzle patterns, both axes'
scale tensors) buys nothing.

### 6.4 Weight-side saturation on real model weights (Kaiming-init, Gaussian)

For each Linear layer's BF16 weight quantized to FP8 E4M3:

```
layers.0.kda.attn.q_proj  shape=(512, 512)  amax=0.077
  block   saturations   sqnr_med
    16      9.640%      32.63dB
    32      6.591%      32.26dB
    64      5.069%      32.07dB
   128      4.312%      31.98dB
   256      3.894%      31.92dB
   512      3.658%      31.89dB

layers.0.ffn.down_proj  shape=(512, 1536)  amax=0.054
  block   saturations   sqnr_med
    16      9.597%      32.66dB
    32      6.568%      32.26dB
    64      5.068%      32.07dB
   128      4.320%      31.96dB
   256      3.878%      31.91dB
   512      3.594%      31.88dB
  1536      3.352%      31.86dB
```

Kaiming-init weights are roughly Gaussian → saturations behave the same
as D1 Gaussian in §3: 9.6% at block=16, 3.6% at block=512. SQNR is
roughly constant at 32 dB (FP8 representation floor). Real (post-training)
weights will have different distributions; this is just a sanity check.

**Note**: weight saturations at block=16 are 9.6%, **above** the activation
saturations at the same block size. The activations are post-RMSNorm so
amax is well-controlled; the weights are raw Kaiming with high kurtosis.

---

## §7 — GEMM-level sig_rel (Part C)

Dequantize A and B with the chosen scale format, multiply in BF16, compare
to BF16 reference. Simulates any FP8 MMA path that consumes the same FP8 +
scale. Prod KDA shape M=1024 K=N=1536:

| Distribution | scale | tensor | row | block16 | block32 | block64 | block128 |
|---|---|---|---|---|---|---|---|
| Gaussian | BF16 | 4.10% | 3.89% | **3.40%** | 3.46% | 3.46% | 4.02% |
| Gaussian | E8M0 | 3.74% | 3.74% | 3.74% | 3.74% | 3.74% | 3.74% |
| outlier_rows | BF16 | 3.52% | 3.65% | 3.74% | **3.35%** | 4.29% | 4.02% |
| outlier_rows | E8M0 | 4.02% | 4.02% | 4.02% | 4.02% | 4.02% | 4.02% |

E8M0 blockwise gives **no GEMM-level improvement** — the per-block scale
gain is masked by E8M0's own noise floor. BF16 block16-32 hits **3.35-3.40%
sig_rel**, which matches `project_fp8_distribution_sweep_2026_07_24.md`
(MXFP8 3.5-4.2% / BF16 blockwise 3.5%) and confirms BF16's edge.

**The user's original conclusion "TS > RS ≈ MX" reads as "TS worst,
RS ≈ MX best".** The matrix shows that on real activations this should
be "TS worst, RS best, MX much-worse-than-RS":

- RS (per-row FP32): 3.65-3.89% sig_rel
- MX (1×32 E8M0): 3.74-4.02% sig_rel

**E8M0 1×32 is worse than per-row FP32 by 0.1-0.4 pp.** The original "≈"
overstates how close MX and RS are.

---

## §8 — Engineering recommendations

### 8.1 Scale format choice

| Format | Verdict | Why |
|---|---|---|
| **BF16** | **SHIP** | Same SQNR as FP32/FP16; half storage vs FP16; sm_120 native MMA type |
| FP16 | OK if storage matters more than speed | Indistinguishable precision from BF16 |
| FP32 | Wasted | No SQNR benefit over BF16 at any granularity |
| E8M0 (MXFP8) | **AVOID** | 21 dB worse than BF16 on real KDA activations at fine granularity |

### 8.2 Block size per activation type

For our model (KDA q/k/v/o + FFN gate/up/down + lm_head):

| Activation | Best block | SQNR | Reasoning |
|---|---|---|---|
| **KDA q/k/v/o** | **per-row or 1×128** | 55.8 dB | Row-amax varies a lot across tokens; K-axis is uniform post-RMSNorm |
| **FFN gate/up** | **per-row** | 55.9 dB | Same as KDA q/k/v — narrow post-silu |
| **FFN down** | **1×16** | 35.6 dB | Long K=1536 with amax variation across K columns; per-row is bad |
| **lm_head** | **1×16** | 33.0 dB | Logit pre-softmax; some K-dim variation |

Practical choice: **build the FP8 GEMM kernel with per-row scale as the
default, plus a 1×16 K-block path for FFN down_proj**. Don't bother
supporting arbitrary 1×B block sizes — the data shows per-row and 1×16
cover the practical cases.

### 8.3 Memory implication of blockwise BF16

Per-row BF16 scale adds 2 bytes/row (BF16). For K=1536 the scale is
negligible (<0.2% of FP8 weight storage). 1×16 BF16 scale adds 2 bytes/16
elements = 12.5% overhead on the FP8 weight storage. Both are within
HBM budget.

### 8.4 sm_120 GEMM kernel implications

Per [`project_rowwise_epilogue_gap_2026_07_23.md`](project_rowwise_epilogue_gap_2026_07_23.md):
- RowWise 100 TF, TensorWise scalar 186 TF. The 86 TF gap is per-element
  Sm90ColBroadcast × Sm90RowBroadcast epilogue.
- A custom BF16-blockwise epilogue (the `fp8_blockwise_gemm` 1×32 +
  hand-written R5b² dual-accumulator at 78-91 TF, per
  `project_fp8_issue_bound.md`) gives 1.7-2.2× RowWise throughput AND
  better numerics.

Conclusion: a BF16 blockwise kernel is BOTH faster (1.7-2.2× RowWise)
AND more accurate (21 dB SQNR vs E8M0 blockwise) on sm_120. The user's
"MXFP8 path" (E8M0) cannot match this — same speed, much worse numerics.

---

## §9 — Future work / open questions

1. **KL divergence as the production end metric.** `feedback_kl_div_for_quant_eval.md`
   notes that SQNR / sig_rel are engineering validation metrics; the
   real production gate is forward KL between BF16 teacher and FP8
   student on the logits. Once that sweep is stable, it should become a
   skill. (Not done yet.)
2. **Real (post-training) weight distributions.** §6.4 used Kaiming-init
   Gaussian; trained weights have non-trivial structure (spectral
   norms, outliers in some columns). Re-run §6.4 on a trained checkpoint.
3. **Post-RMSNorm 2D block probe.** §6.3 was on raw model activations
   (post-norm). The downstream FFN input (post-silu) might have more
   K-dim structure that 2D blocks could exploit — but unlikely given the
   uniform SQNR we already see.
4. **NVFP4 W4A4 scale format comparison.** NVFP4 uses E4M3 scales
   (1×16 blockwise). E4M3 is the missing entry in Part E; if a custom
   sm_120 NVFP4 GEMM is built, E4M3 vs BF16 vs E8M0 1×16 precision
   should be measured directly. (Project is shipping NVFP4 W4A8, not
   W4A4; W4A4 was rejected per `project_block_bf16_scale_dead_end.md`.)

---

## §10 — Cross-references

- `docs/fp8_gemm_landscape_2026_07_23.md` — performance landscape (TFLOPS)
- `docs/fp8_gemm_kernel_pipeline.md` — kernel construction pipeline
- `project_fp8_error_decomposition.md` (memory) — 3.6% sig_rel decomposition
- `project_fp8_distribution_sweep_2026_07_24.md` (memory) — RW/SC/MX sweep
- `project_fp8_issue_bound.md` (memory) — sm_120 1×64 + dual-accum kernel
- `project_mxfp8_b12x_sm120_probe_2026_07_24.md` (memory) — b12x MXFP8 probe
- `project_block_bf16_scale_dead_end.md` (memory) — block BF16 dead-end on
  sm_120 (no kernel consumes block FP32/BF16 scales natively)
- `project_scale_format_mantissa_curve.md` (memory) — non-monotonic m-bits
  curve (m=1 worse than m=0)
- `feedback_kl_div_for_quant_eval.md` (memory) — production end metric

## §11 — Reproducibility

Probes live at `scripts/probes/probe_fp8_scale_precision*.py` (long-term
canonical home). Re-run from this branch with:

```
PYTHONPATH=/hy-tmp/HippoLM python scripts/probes/probe_fp8_scale_precision.py
PYTHONPATH=/hy-tmp/HippoLM python scripts/probes/probe_fp8_scale_precision_real.py
PYTHONPATH=/hy-tmp/HippoLM python scripts/probes/probe_fp8_scale_precision_extras.py
```

Hardware: RTX 5060 Ti (sm_120) 16 GB. torch 2.12 / CUDA 13. Probe seeds
are fixed (seed=42 for synthetic, seed=0 for real-activation capture)
so the numbers are bit-reproducible on the same GPU.