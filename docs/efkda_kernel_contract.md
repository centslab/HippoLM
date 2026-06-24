# EFKDA kernel contract & debugging notes

This document captures the contract that every chunkwise linear-attention
kernel in this repo must satisfy, and the debugging history of the
EFKDA PyTorch reference that motivates the contract. It is intended
to be the first place the next kernel author (Triton EFKDA, future
GDN3, future RWKV-anything) reads before touching a chunkwise
recurrence.

## The kernel

`src/models/ops/_vendored/fla/ops/kda/chunk_efla_naive.py::efla_chunk_kda`

is the production PyTorch reference. It is *correct-but-slow*; the
Triton kernel is the speed path. The contract below is what both
implementations must hold.

## Contract (mandatory)

1. **Tensor layout**
   - Input: `q, k, v, g, beta` all of shape `[B, T, H, K_or_V]`.
   - Internally transposed to `[B, H, T, K_or_V]`. Tensors are
     up-cast to float32 inside the kernel for the (I+T) solve's
     stability; output is cast back to the input dtype.

2. **cu_seqlens (varlen)**
   - When supplied, the caller has already flattened
     `[B, T, hidden]` → `[1, B*T, hidden]`. The kernel sees `B == 1`
     and processes one flat sequence.
   - The recurrent state `h` MUST be reset to zero at every chunk
     that starts a new doc. Without this, `short_conv` resets the
     Q/K/V state at the boundary but the recurrence carries the
     previous doc's `h_start` into a chunk whose Q/K/V are FRESH.
     The resulting `p = (k * exp(g_cum))^T h_start` is stale, the
     `(I+T)^{-1}` solve produces an incorrect `v_new`, and the chain
     rule back through the stale `p` amplifies across chunks until
     grads are inf.
   - **Pre-condition:** cu_seqlens offsets must be aligned to
     `chunk_size`. This is exactly what `pack_chunk_aligned`
     (in `src/training/data/collate.py`) produces.

3. **Per-chunk checkpoint (VRAM)**
   - The kernel's per-chunk intermediates include `[B, H, L, L, K]`
     rank-5 tensors (the `decay_inner`/`decay_q` matrix). At
     base.yml prod dims (B=4, T=4096, L=32, H=8) all 64 chunks'
     saved-for-backward tensors are alive in the autograd graph at
     once if the chunkwise forward is a single Python-level call —
     this exceeds 16 GB. Wrap each chunk's forward in
     `torch.utils.checkpoint.checkpoint` with `use_reentrant=False`
     so intermediates are freed after the chunk and recomputed on
     backward. The recurrent state `h` is a tensor input/output of
     the chunk, so autograd connects `h_new` of chunk `i` to
     `h_start` of chunk `i+1` and the gradient flows correctly.

4. **Numerical stability of the (I+T) solve**
   - The chunkwise recurrence is
     `(I + T) v_new = (v - p)` with `T` a strict-lower-triangular
     `[L, L]` matrix per `(B, H)`. The entries of `(I+T)^{-1}` grow
     exponentially with `L` (for L=64 with α=0.5 the (L-1, 0) entry
     can hit 1e+18). Forward-substitution's VJP is bounded; the
     explicit inverse's VJP is not. Use
     `torch.linalg.solve_triangular(., upper=False)`, never
     `torch.linalg.inv(I + T) @ ...`.

5. **fp32 NaN guard on the upper triangle**
   - In the `T` matrix construction, `diff_g = g_cum[t] - g_cum[t']`
     for `t' > t` is positive (g decay is monotone). `exp(diff_g)`
     overflows fp32 for `diff_g > 89`. The upper-triangle positions
     are masked to 0 in the OUTPUT (`torch.tril(., diagonal=-1)`),
     but the backward through `exp()` still multiplies the
     (masked-out) 0 gradient by `inf`, giving nan. **Mask the
     `diff_g` to 0 BEFORE `exp()`** with `torch.where(strict_lower_mask, diff_g, 0)`.

## Why these matter: the EFKDA debugging journey

This is the chain of failures that motivated the contract above.
Reading it should make clear *why* each rule is non-negotiable.

### Failure mode 1: VRAM blow-up

**Symptom:** `torch.cuda.OutOfMemoryError` at base.yml prod dims
(B=4, T=4096, L=32, H=8, K=128) on the first microbatch backward.

**Root cause:** The naive implementation runs all 64 chunks in one
Python-level forward. The autograd graph then holds all 64 chunks'
`[B, H, L, L, K]` saved-for-backward tensors alive at once during
backward. At the prod dims, that's 4.5 GB per layer × 4 layers per
block = 18 GB, which blows the 16 GB budget.

**Fix:** Per-chunk `torch.utils.checkpoint.checkpoint`. Intermediates
are freed after each chunk and recomputed on backward. Backward peak
drops to ~9 GB.

### Failure mode 2: fp32 NaN at T > 1024 with small g

**Symptom:** Per-parameter grads land in `[0, 1e-3]` for T ≤ 1024,
but cross to inf at T = 2048 in synthetic tests. Always at a
specific tensor (typically `q_proj.weight`).

**Root cause:** Two compounding issues:
1. `decay_inner = (g_cum[t] - g_cum[t']).exp()` for `t' > t` overflows
   fp32 (`diff_g` is positive in the upper triangle because g decay
   is monotone; the upper-triangle mask is applied to the OUTPUT but
   not to the input). Backward through `exp()` then propagates
   `0 * inf = nan`.
2. The explicit `(I+T)^{-1}` inverse has exponentially-large
   entries; its VJP amplifies them back into the inputs.

**Fix:** Mask `diff_g` to 0 before `exp()`; use
`torch.linalg.solve_triangular(., upper=False)` for the (I+T) solve.

### Failure mode 3: cu_seqlens state-leak (the subtle one)

**Symptom:** Direct kernel test (random q/k/v + packer cu_seqlens)
passes. Standalone EFKDA layer test passes. 4-layer micro-test
**with the same cu_seqlens** produces 60/72 inf grads. Actual
`scripts/train.py` with `--tp_size=1 --batch_size=2 --num_layers=4`
produces finite loss and finite grad_norm. The residual non-finites
in the micro-test (1-9 per 1024x1024 tensor, max_abs ~1e-3) are bf16
quantization noise, not the original bug.

**Investigation:** Direct kernel test with the model-shaped q/k/v
(l2norm'd, realistic `g = -A_log.exp() * softplus(...)`, T=24576,
384 chunks) IS finite. The micro-test failure is reproducible only
with the *combination* of (a) the EFKDA wrapper reshaping
`[B, T, hidden]` → `[1, B*T, hidden]` so the kernel sees one flat
sequence and (b) the packer producing cu_seqlens with offsets at
SPECIFIC chunk-aligned positions (T=4096, ~30 doc starts).

**Root cause:** The kernel did not honor cu_seqlens. The wrapper
flattens `[B, T, hidden]` to `[1, B*T, hidden]` for the kernel
input, but `short_conv` (which is depthwise per-channel) does
honor cu_seqlens via `causal_conv1d`'s state-reset at boundaries.
So at every doc boundary: q/k/v are FRESH (short_conv state was
reset), but the kernel's `h_start` is the previous doc's accumulated
state. The first chunk after a boundary computes
`p = (k * exp(g_cum))^T h_start` from a stale h, the (I+T) solve
returns a `v_new` that mismatches v against an unrelated state,
the chain rule back through the stale `p` amplifies across chunks.

**Why the pattern is "specific boundary positions":** With chunk_size=64
and packer cu_seqlens at e.g. [0, 1408, 2688, 4096, ...], the doc
starts land at chunks [22, 42, 64, ...]. The error compounds
multiplicatively through the recurrence — at ~30 doc starts, the
amplification factor hits fp32 overflow. T ≤ 2048 (≤ 192 chunks)
doesn't accumulate enough starts to overflow, which is why the bug
only appears at T ≥ 4096.

**Fix:** Pre-compute `_doc_start_chunks: set[int]` from the
`cu_seqlens` offsets (one CPU sync at kernel entry, then O(1) set
lookup per chunk). Reset `h = torch.zeros(...)` at every chunk that
starts a new doc. Assert `cu_seqlens[i] % chunk_size == 0` (the
packer is required to produce chunk-aligned offsets; without this,
a doc boundary inside a chunk would need a partial-chunk solve that
the kernel does not support).

## What the Triton kernel must do differently

The PyTorch ref is the correctness oracle. The Triton kernel must:

- Hold the same contract on cu_seqlens state-reset (otherwise
  re-introducing Fix 3's failure mode at 50× the throughput).
- Hold the same `decay_inner` fp32-NaN guard (the kernel will see
  the same monotone-positive diff_g in the upper triangle; masking
  the INPUT is the only safe place).
- Use forward-substitution for the (I+T) solve (or a closed-form
  equivalent, e.g. WY representation with bounded VJP).
- Pre-compute `_doc_start_chunks` from `cu_seqlens` once on the
  Python side, pass it as a kernel arg.
- Verify against the PyTorch ref: forward (ULP) and backward (fp32
  finite) at the test shapes in `test/_tmp/test_efkda_triton_correctness.py`.

### Failure mode 4 (excluded): adamw_v=bf16 → grad norm NaN/Inf

**Symptom (claimed):** Long-sequence training (T ≥ 4096) with
GDN2/EFKDA produces grad_norm → inf after a few steps. User memory
attributed this to `adamw_v` being stored as bf16 (commit 81095c3
"AdamW v should be BF16, not FP32").

**Investigation:** Reverted model to GDN2-stable code (parent of
4777a3e), ran 4 controlled smoke tests at B=4, T=4096, num_layers=8,
num_blocks=2, on the 5060 Ti 16G:

| Setup | grad_accum | adamw_v | grad_norm (steps 1-3) | loss (steps 1-3) | NaN? |
|---|---|---|---|---|---|
| #1 | 2  | bf16 | 507 → 686 → 896     | 11.94 → 12.02 → 12.10 | none |
| #2 | 2  | fp32 | 515 → 670 → 885     | 11.94 → **−1e33** → −8e31 | loss explodes |
| #3 | 16 | bf16 | 73 → 212 → 216      | 11.93 → 12.31 → 12.29  | none |
| #4 | 16 | fp32 | 74 → 203 → 212      | 11.93 → **3.7e33** → **nan** → nan | loss NaN at step 2 |

**Conclusion: bf16 v is MORE stable than fp32 v at T=4096 in the
current code path.** The original failure was most likely the FP16
v bug from `f99f5b6` (FP16 v underflow at v ≈ 1e-8 → factor=m/eps
explosion), not a BF16 v issue. The bf16 7-bit mantissa drift
introduces "noise" in the AdamW factor that, counterintuitively,
prevents the optimizer from finding degenerate directions where the
lm_head matmul produces huge logits.

**Action: keep `adamw_v: bf16` in `configs/base.yml`.** The diagnostic
that produced the table above is in `test/_tmp/test_adamw_v_bf16_bug.py`
(throwaway, delete before commit per the test-first pattern). The
H2 scenario in that script — long-horizon bf16 v EMA with
mean_rel_err ≈ 1.0 on the AdamW factor — is the numerical reason
bf16 v "works": the precision noise caps how aggressively AdamW
can push any single parameter.

### Failure mode 5 (open): FusedLinearCE loss NaN under aggressive AdamW

**Symptom (observed):** When `adamw_v` was switched to fp32 (#2 and
#4 above), the optimizer steps were all finite (grad_norm 74-885,
post-opt pmax = 6.906, `all finite` in the post-opt diag) but the
NEXT forward's loss output was already −1e33 / +3e33 / NaN. The
NaN appears in `mb_loss` (the FusedLinearCE output), not in any
parameter gradient.

**Working hypothesis:** the precise fp32 v AdamW step can drive the
lm_head weights in a direction that produces one logit >> others
for some specific token in the next forward. The chunked online
softmax in FusedLinearCE then does its fp32 `m_i = max(m_{i-1},
max(chunk))` and `lse` subtraction in bf16 chunks, which loses
precision when one chunk's max dominates. The result is a loss
value of order ±1e33, which then rounds to NaN when the next
microbatch's grad hits bf16's dynamic range.

**Open questions (for the next investigation pass):**
1. Does FusedLinearCE cast `lse` back to fp16 / bf16 between chunks,
   or does it keep fp32 throughout?
2. Does the lm_head matmul run under `torch.amp.autocast` (bf16) or
   in the param's natural dtype (fp16)?
3. Is the tied-embed `dw` accumulation in
   `src/models/ops/_vendored/fla/modules/fused_linear_cross_entropy.py`
   doing any bf16 sum that could overflow with the boosted lm_head
   rows?
4. Is the "no analytical solution → long-sequence error accumulation"
   hypothesis from the original triage still credible, given that
   the EFKDA PyTorch ref passes standalone at T=24576 with
   realistic g?

### Failure mode 6 (revised): catastrophic loss oscillation at TP=2, both v dtypes

**This section supersedes the "bf16 v is MORE stable" claim in
Failure mode 4 above.** A 20-step reproduction at GDN2-stable code
(`gdn2-repro` branch, commit `892315b`) with the smoke-test dims
(B=4, T=4096, num_layers=8, num_blocks=2, lr=0.01, grad_accum=2,
`--use_dummy_data`, `--tp_sim --tp_size=2`) shows BOTH
`adamw_v=bf16` and `adamw_v=fp32` produce the same catastrophic
loss oscillation. From the production training log:

| step | fp32 v loss | bf16 v loss | fp32 grad_norm | bf16 grad_norm |
|------|-------------|-------------|----------------|----------------|
| 1    | 11.94       | 11.94       | 513            | 505            |
| 2    | **−1.57e24** | **−9.13e33** | 699            | 722            |
| 3    | 12.10       | 12.10       | 884            | 880            |
| 4    | **+5.10e26** | 12.10       | 873            | 833            |
| 5    | 12.09       | 12.08       | 778            | 801            |
| 6    | 12.09       | **−1.92e28** | 741            | 804            |
| ...  |             |             |                |                |
| 10   | **−4.41e29** | 12.23       | 1136           | 1065           |
| 11   | 12.32       | **−1.57e7**  | 1156           | 1092           |
| 15   | NaN         | NaN         | NaN            | NaN            |

**Key observations:**
- The catastrophic losses are **alternating**, not monotonically
  escalating. Normal steps (~12.1) interleave with explosions
  (1e24 to 1e33). This pattern is consistent across both v dtypes.
- Explosion magnitude varies (1e7 to 1e33) and is determined by
  which specific microbatch hits the pathological logit, not by
  v precision.
- grad_norm grows steadily (505→1326 by step 14) before going NaN.
  The growing grad_norm matches the smoke-test signature.
- At **TP=1** the same dims produce NO explosions within 4 steps
  (loss stays 12.6–12.7, grad_norm 4–6). TP=2 is required to
  reproduce.

**Revised interpretation:** the original −2e33 failure is NOT
specific to `adamw_v=fp32`. Both v dtypes hit the same
catastrophic loss oscillation at TP=2. The earlier conclusion
("bf16 v is more stable") was an artifact of the short smoke
test that exited at the first explosion and happened to be
running the bf16 path; running both paths for ≥10 steps shows
they fail at comparable rates. **The bf16-v preference is
indeterminate on the failure rate itself**, though bf16 v still
wins on the storage cost (2 bytes/elt vs 4).

**Open hypothesis (failure mode 5 still open):** at TP=2 the
all-reduced gradients may push a specific lm_head row into a
post-AdamW state where one token's logit dominates. The chunked
online softmax in FusedLinearCE then overflows fp32 lse.
Both v dtypes can hit this path; which one arrives first
depends on microbatch ordering rather than dtype.

**Implication for the next debug pass:**
- The previous "keep adamw_v=bf16" recommendation stands as a
  non-fixing precaution (smaller state, no failure-rate benefit
  at this scale).
- The real fix is downstream of the v-dtype choice:
  - Per-microbatch loss check: skip the optimizer step if any
    microbatch produces a non-finite loss in the current window.
  - Reduce `learning_rate` from 0.01 to e.g. 0.005 or 0.002.
  - Aggressive grad clipping at `max_grad_norm=0.5` or lower
    (current 1.0 is not enough at TP=2, B=4, T=4096).
  - Investigate whether the pathological logit is on a specific
    token (e.g. a single high-frequency token in the dummy
    uniform distribution).

The diagnostics in `test/_tmp/` (loss-isolation ablations,
init-sensitivity sweep) all show bf16 v ≡ fp32 v at smaller
scale; the divergence only manifests at TP=2, full production
dims.

## Diagnostic infrastructure

These tools live under `test/_tmp/` and are auto-enabled when
debugging EFKDA training (not on the production path):

- `test/_tmp/find_inf.py` — locates the first tensor with inf/nan
  grad, with hook-level detail.
- `test/_tmp/mem_dump.py` — VRAM peak per stage (forward, backward,
  optimizer). Used to validate Fix 1.
- `test/_tmp/bisect_inf.py` — halves T until the failure stops
  reproducing (used to identify Fix 3's T=2048 → T=4096 cliff).
- `test/_tmp/test_kernel_long.py` — direct kernel test at the
  failing T, with model-shaped inputs. Used to disambiguate
  "kernel bug" vs "wrapper bug".

These are throwaway debug tools and follow the
`feedback_test_first.md` rule (delete before commit, never on the
production code path).
