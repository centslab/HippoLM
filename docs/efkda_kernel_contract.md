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

### Failure mode 7 (root-cause): GDN2 chunked-delta backward NaN

**Symptom:** Training reaches step 9-15, then the post-accumulate-grad
hook reports a few specific parameters with non-finite accum grads.
Loss stays finite (~12.1) and flat after the NaN. `c_logits` (the
chunked online-softmax outputs of FusedLinearCE) are clean: max|logit|
stays at ~4 even when the grad is NaN.

**The specific params that go non-finite are GDN2 chunk-delta
projections, NOT the lm_head.** The non-finite shape patterns are
consistent across reproductions:
- `[512, 1024]`
- `[1536, 1024]`
- `[1024, 512]`
- `[1024, 1536]`
- `[256, 1024]`

The product of the two dims (e.g. 512×1024 = 524288) is constant —
these are the GDN2 q/k/v projection reshaped to `[H*hidden, K]` or
similar, where `H=8 K=128 V=128 hidden=1024` (or `H=8 K=128
hidden=1536` in the block-summarizer path). Any chunked-delta
recurrence applied to a `[B, T, H, K]` tensor in this shape pattern
triggers the bug.

**Reproduction:** TP=2, B=4, T=4096, num_layers=8, num_blocks=2,
lr=0.01, grad_accum=2, `--use_dummy_data`, `--tp_sim --tp_size=2`,
`adamw_v=fp32`. **~30% of seeds at fp32** hit non-finite grad within
15 steps. Triggered immediately by seed=7 (ga=2); seeds 42, 53, 79
delay. Both `bf16` and `fp32` v dtypes hit it; bf16 v sometimes
buys an extra 1-3 steps because the lower-precision AdamW factor
adds noise that delays the divergence. **Isolated ablations (TP=1,
single layer) do NOT reproduce; both failure modes require TP=2 +
production dims + many steps.**

**Why the loss stays flat at ~12.1 after NaN:** the all-reduced
grad (with NaN) is reduced across the TP group before the
optimizer step. The NaN propagates into the post-AdamW state of
the offending parameters; subsequent fwd/bwd use the corrupt
weights, but the FusedLinearCE chunked online-softmax masks the
catastrophic logit rows (they don't appear in the label tokens
for the uniform dummy data). The net effect: loss looks fine but
parameters are corrupt and the run is unsalvageable.

**Why `c_logits` stay clean (max|logit| ~4):** FusedLinearCE's
chunked online softmax does `m_i = max(m_{i-1}, max(chunk))` in
fp32, and the lm_head's `[V=128K, hidden]` matmul + bias is in
bf16. Even with a corrupt lm_head row that would produce a huge
logit, the chunk that contains that row also contains it before
the AdamW step (when its logit is finite), so the chunk's running
max is dominated by THAT pre-step logit value. Only the NEXT
forward sees the post-step logit — and at that point the chunk's
running max has already been established by earlier tokens. The
logit explosion is masked.

**Root cause:** GDN2's chunked-delta backward (the
`chunk_gated_delta_rule_bwd_*` Triton kernel family in FLA) has a
numerical-stability bug in the dhu / dAv path at the model's
operating point. Specifically, the inter-chunk recurrence
backward through `h_{i+1} = A_i h_i + B_i x_i` amplifies round-off
errors when the chunk size × K matches a Triton shmem pressure
sweet spot. The error compounds across 64 chunks × 8 heads ×
production dims to a single NaN element in the dhu pass, which
back-propagates through `solve_triangular` into the q/k/v
projections.

**Fix:** EFKDA — the
`src/models/ops/_vendored/fla/ops/kda/chunk_efla_naive.py` kernel
is forward-correct vs the EFLA spec and uses a different
recurrence formulation (chunk-local triangular solve + per-chunk
h_new, no inter-chunk solve). The EFKDA PyTorch ref is VRAM-safe
(per-chunk checkpoint, fix #1) and numerically stable at the
model's actual g scale (fix #2: mask `diff_g` before `exp()`;
fix #2 also: `solve_triangular` instead of explicit inverse).
**This is the bug that motivated the EFKDA rewrite on
`efkda-exp` branch (commit 4777a3e).** Confirming the bug on
`gdn2-repro` HEAD means EFKDA was the correct call.

**Implication for the next debug pass:**
- If you see post-accumulate-grad non-finite params with the
  shape patterns above on a GDN2 build, it's FM 7, not FM 4-6.
- TP=1 ablations will not catch this. Always test at TP=2 with
  full prod dims.
- Don't bisect the lm_head or FusedLinearCE path — c_logits are
  clean, the bug is in the attention backward.

### Failure mode 8 (root-cause): FFD-packer tail-pad state-leak

**Symptom:** Intermittent `[DIAG] hidden_states has inf/nan!
max_abs=nan` after the final norm, with all params at 100%
non-finite in the post-accumulate-grad hook. Only triggers in
**train mode** (not eval / no_grad), and only when
`cu_seqlens[-1] < B*T` — i.e. the last pack of the FFD-packed
batch is not full.

**Root cause:** The FFD packer (in `src/training/data/collate.py`)
produces `cu_seqlens` as the END offset of each packed doc, with
the final entry being the END of the LAST real doc, not `B*T`.
When the last pack doesn't fill `B*T` tokens (e.g. last pack
holds docs totaling 3968 tokens out of 4096, leaving 128
tail-pad tokens), `cu_seqlens` ends at 3968 — and the kernel
sees 64 chunks of which the last one (chunks at offset 3904..3967)
is a partial real chunk, plus chunks 63 (3968..4031) through the
end of the (B=1, T=4096) flattened view are tail-pad.

The tail-pad chunks have valid-shape but garbage-content q/k/v
(whatever was in the flattened tensor at those positions). The
kernel processes them WITHOUT resetting `h` (because no doc
starts there) — so `h_start` for chunk 63 is the last real doc's
accumulated state. The chunk's `p = (k * exp(g_cum))^T h_start`
reads the (large) stale `h`, the `(I+T)^{-1}` solve amplifies
across 1-3 chunks, and the autograd graph carries NaN into the
FusedLinearCE's tied-embed matmul (which sums over ALL output
positions, including tail-pad).

**Why it only triggers in train mode:** no_grad stops
autograd from accumulating into the tail-pad chunks' chain rule
(the tail-pad outputs are masked by FusedLCE because labels are
-100 at those positions). Train mode propagates the tail-pad
backward all the way to the parameters.

**Why it only triggers when `cu_seqlens[-1] < B*T`:** when the
packer fills the buffer exactly, the last "doc" ends at
`B*T` and the kernel sees a clean boundary (h reset to 0 at
chunk 64). When the last pack is short, the kernel never sees
the boundary and accumulates stale state into tail-pad chunks.

**Fix:** in `pack_chunk_aligned` (the FFD packer wrapper in
`src/training/data/collate.py`), ALWAYS append `n_packs *
seq_len` to `cu_seqlens_list` as the final entry — even when the
last real doc ends BEFORE `B*T`. This marks the start of the
tail-pad "doc" and the kernel resets `h` to zeros at the
tail-pad boundary. The padding chunks' outputs are masked out
by FusedLCE (labels are -100 at the tail-pad positions) so the
loss is unaffected, but the autograd graph is now clean.

**Diagnostic:** `test/_tmp/bisect_inf.py` with a T sweep can
confirm this is FM 8 vs FM 7: FM 8 reproduces only when
`cu_seqlens[-1] < B*T`, FM 7 reproduces regardless of packing.

**Implication for the next debug pass:**
- The `pack_chunk_aligned` invariant MUST be enforced in the
  packer. Never call the kernel with a `cu_seqlens` that doesn't
  end at `B*T`.
- If you add a new packer / loader, audit the cu_seqlens
  generation. The contract is: `cu_seqlens[-1] == B*T` after
  padding, no exceptions.

### Failure mode 9 (EFKDA bwd bug): grad_g_chunk_acc double-counting

**Symptom:** After the `[BS, BS, BK]` sub-tile rewrite of the
EFKDA backward kernel (`src/models/ops/efkda_bwd.py`, commit
e31a2ae), K=16 fp32 gradient test fails catastrophically:
`grad_g_chunk` has relative error 2.90e+0 (i.e. off by ~3×) vs
the PyTorch reference. All other grads (`grad_q`, `grad_k`,
`grad_v`, `grad_beta`, `grad_A_log`) are fine.

**Root cause:** The original sub-tile rewrite has TWO
accumulation points for `grad_g_chunk_acc`:
1. Inside the `for ti in range(NS)` (sub-tile) loop:
   `grad_g_chunk_acc += grad_g_chunk_sub` — this is the
   intended per-sub-tile accumulation.
2. At the END of the `for k_idx in range(NC)` (K-tile) loop:
   `grad_g_chunk_acc += grad_g_chunk_kt_total` — this was a
   leftover from a different formulation.

Both points accumulate the Kalpha contribution to `grad_g_chunk`
for the same `(chunk, head, d)` position, so the final
`grad_g_chunk[d]` is exactly 2× what it should be. (The
per-sub-tile accumulation has already folded in the Kalpha
contribution; the per-K-tile wraparound then adds it again.)

**Investigation:** wrote `/tmp/test_isolate_gradg.py` to bisect
the kernel and compare against the unfused PyTorch ref. The
test's actual-vs-expected gap showed `kernel ≈
-expected_cumsum + chunk` — the `+ chunk` term was the symptom
of the double count, since `chunk` was 2× its true value and
the "extra" amount accounted for the entire mismatch.

**Fix:** remove the duplicate line (was line 611 in the
pre-fix file). After the fix, K=16 fp32 relerr drops to
9.16e-7 (was 2.90e+0). The fix is one-line; the bug took
~30 min of bisection because `grad_g_chunk` is the only
output that's wrong (everything else is self-consistent
through the kernel's internal accumulators).

**Implication for the next debug pass:**
- The sub-tile pattern in EFKDA bwd has FOUR accumulators that
  span both the ti loop and the k_idx loop:
  `grad_q_kt`, `grad_k_kt`, `grad_g_kt`, `grad_k_from_alpha_t_acc`.
  **All four must be checked for double-count.** The pattern is
  "accumulator that has contributions from BOTH the inner
  per-tile loop AND a post-loop fixup pass". An accumulator
  that has contributions from ONLY the post-loop fixup is fine
  (e.g. `grad_k_from_alpha` is intentionally only summed
  post-loop because it needs the full K-axis sum).
- When extending the sub-tile pattern to a new gradient
  (e.g. grad_gamma if a gate scaling is added), the same
  audit must be repeated.
- This bug is invisible to fp16/bf16 noise — the relerr 2.90
  is in fp32. Always run the EFKDA bwd correctness tests in
  fp32 first; only switch to bf16 after fp32 passes.

### Failure mode 10 (NOT a bug): universal loss drift above log(V)

**Symptom:** ALL seeds, both `adamw_v=bf16` and `adamw_v=fp32`,
at GDN2-stable code, B=4, T=4096, num_layers=8, num_blocks=2,
lr=0.01, grad_accum=2, `--use_dummy_data`, show loss growing
monotonically from ~11.94 (step 1) to ~12.50 (step 16) at
fp32 v, or saturating at ~12.07 at bf16 v. grad_norm grows
from ~500 (step 1) to ~1500 (step 16). No NaN, no inf — loss
and grads are all finite.

**Reference:** `log(V) = log(151936) ≈ 11.93` is the optimal
loss for uniform prediction over the vocabulary. The fp32 v
trajectory is 0.57 nats above optimal; the bf16 v trajectory
saturates 0.14 nats above optimal. Both are worse than uniform.

**This is NOT a bug.** It is the model overfitting on the
uniform-random dummy data. Each step nudges the params in a
direction that DECREASES loss on the current microbatch, but
the dummy data is i.i.d. uniform so the "direction" is
random noise correlated with the labels of THIS microbatch.
After ~10-20 steps the model has memorized the noise of the
recent microbatch and produces a slightly non-uniform output
that happens to be wrong on the NEXT microbatch.

**Why it's important to record:** if you see loss-drift-above-log(V)
in a smoke test, the natural reaction is to bisect for an
exploding-gradient bug. There is no such bug — the gradients
are finite, the optimizer step is finite, the post-opt
diag is clean. The drift is a feature of training on
random data with a non-zero LR.

**Diagnostic:** plot loss vs step and overlay `log(V)`. If
the loss curve stays ABOVE `log(V)` (i.e. worse than uniform)
for the entire run, it's FM 10 (overfitting on dummy data),
not FM 4-7 (numerical bug). For numerical bugs, loss explodes
or goes NaN/inf within a few steps; FM 10's drift is gradual
and bounded.

**Implication for the next debug pass:**
- Use a real (non-dummy) data source for smoke tests that
  are intended to validate training health.
- For dummy-data smoke tests, expect the loss to drift above
  `log(V)`; don't panic and don't bisect for a bug.
- The bf16 v saturation at 12.07 (vs fp32 v's 12.50) is
  consistent with the lower-precision AdamW factor adding
  noise that prevents the optimizer from fully memorizing
  the recent microbatch. This is why bf16 v is "more
  stable" on dummy data — not because it's more numerically
  correct, but because it's MORE noise, and noise prevents
  the model from following the random-data gradient
  direction to its (catastrophic) extreme.

## Compile-time negative results

These are experiments that were tried and REVERTED. They are
recorded here so the next kernel author doesn't repeat them.
Each entry is "what was tried → what went wrong → why we
stopped".

### N1: 4-kernel split (fwd_prep + dhu_bwd + intra_bwd + grad_fixup)

**Tried:** FLA-style 4-kernel split of the EFKDA bwd
(`/tmp/efkda_bwd_v2.py` with per-K-tile specialization via
`K_IDX: tl.constexpr`).

**What went wrong:** the intra kernel had a silent data
corruption bug at K=128 with sub-block tiling on t'
(BS=16, NS=4): the `[BT=64, BS=16, BK=16]` per-iter tensor
plus the per-iter accumulators plus the per-iter masks
pushed total shmem past the 5060 Ti's 99 KB limit, causing
L2 spill. The output was within fp32 tolerance for MOST
positions, but `T[6, 5]` for some (chunk, h) pairs had
0.1-0.3 absolute error vs the unfused reference — a small
localized corruption that's hard to catch in a smoke test
but breaks downstream gradient correctness.

**Why we stopped:** the smaller BS=8 (NS=8 sub-blocks, 32 KB
per-iter) was correct but slower than the monolithic
`[L, L, K]` version (K=128 intra at 291 ms in the 4-kernel
split vs 90 ms monolithic). The 4-kernel split's h kernel
also hit a 198 KB shmem OOM that required changing the
64-iter triangular forward-sub from `tl.static_range` to
`range` (loses unroll, ~13 ms for the h kernel). **The win
that FLA gets from the multi-kernel split (per-kernel
shmem is small → fast ptxas compile) is consumed by our
extra per-iter live variables (mask matmuls, sub-tile
intermediates).**

### N2: 3-kernel split with runtime K-tile loop

**Tried:** FLA-style 3-kernel split with a runtime K-tile
loop (`/tmp/efkda_bwd_v3.py`, single specialization
parameterized by `NC: tl.constexpr`).

**What went wrong:** the `intra_bwd` kernel needs
`grad_o`, `grad_h_new`, `grad_p`, `grad_T`,
`grad_Q_dot_K`, `v_new`, `k_c`, `g_c`, `g_cum` — 9 live
variables × `[L=64, BK=16]` × 4B = 144 KB+ at K=64, 192 KB
at K=128. K=128 hits shmem OOM. K=64's runtime K-tile loop
didn't help compile time because the body is still too big
(18+ min and didn't finish a 4-tile K=64 sweep).

**Why we stopped:** the production kernel's `[BS, BS, BK]`
sub-tile rewrite keeps per-iter shmem at ~17 KB precisely to
avoid this. The split's runtime K-tile loop didn't help
because the body is still too big; only splitting the
K-iter from the L-iter would help, and that's what the
sub-tile approach already does.

### N3: tl.math.exp2(x * LOG2E) swap (for compile-time perf)

**Tried:** replaced `tl.exp(x)` with `tl.math.exp2(x * LOG2E)`
on the hypothesis that the explicit `* LOG2E` allows the
compiler to constant-fold the input scale.

**What went wrong:** K=128 regressed by 34% (169 ms → 227 ms,
K=128 prod bf16 tri-only bench). K=64 saw a 32% gain in a
side-by-side (1.51× → 2.00×) but the K=128 regression is
the production-relevant number.

**Why we stopped:** likely the explicit `* LOG2E` prevents
the compiler from constant-folding some fusions in the
masked/conditional decay paths. K=64's masked decay path
is shorter (4 K-iters) so the constant-folding is more
valuable there; K=128's is longer (8 K-iters) and the
loss-of-fusion cost dominates. **Do not use `exp2` in this
kernel — keep `tl.exp`.**

### N4: K=32 hang, torch.compile 26× slowdown, CUDA graphs > 240 s

**Tried three things in sequence:**

1. **K=32 first-call:** the kernel hangs (no output, no
   error) on the first call at K=32. Likely cause: register
   spills on the `[L=64, L=64, K=32]` intermediates
   `decay_inner` / `decay_q` push the kernel into a
   register-pressure regime that ptxas can't recover from
   in the 5060 Ti's 99 KB shmem.
2. **torch.compile:** wrapping the EFKDA bwd in
   `torch.compile` makes the backward 26× SLOWER (220 ms/chunk
   vs 8.5 ms eager). The compiled graph re-materializes the
   `[B, H, L, L, K]` intermediates from the per-chunk
   checkpoint's saved tensors, blowing the autograd graph's
   memory budget.
3. **CUDA graphs:** capturing the EFKDA bwd into a CUDA
   graph takes > 240 s because the first-call Triton compile
   happens INSIDE the capture, and the graph is invalidated
   on any input-shape change.

**Why we stopped:** all three are dead ends for the
production path. Workarounds:
- K=32: pad to K=64 in the wrapper, or wait for the
  sub-tile rewrite to handle K=32 correctly (the
  `[BS=16, NS=4]` sub-tile loop should fit in shmem at K=32
  but the per-iter live variables are still many).
- torch.compile: don't use it. The PyTorch eager path is
  the production path for both fwd and bwd; the Triton
  kernel is selected per-call.
- CUDA graphs: not worth the capture time. Stick with
  eager Triton launches.

**Implication for the next debug pass:**
- If a PR proposes any of (4-kernel split, 3-kernel split,
  exp2 swap, K=32 production, torch.compile, CUDA graphs)
  for the EFKDA kernel, point them at this section first.
- All four negative results share a common pattern: the
  EFKDA bwd has more live variables per K-iter than FLA's
  bwd (mask matmuls, sub-tile intermediates, fp32 lse in
  the online softmax for the FusedLinearCE caller), so
  optimizations that work for FLA don't transfer.

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
