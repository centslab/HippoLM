# FastKDA development history — 25 commits, 5 weeks

**Branch:** `feature/cuda-kernel-optim` (merged into `main`) +
`feature/fastkda-r9-dev` (Round-9 + deep-numerics, dev-only)
**Period:** 2026-05 (Round-1) → 2026-06-30 (Round-9 + deep-numerics)
**Hardware:** single RTX 5060 Ti 16G
**Stack:** CUDA 12.8 + Triton 3.5.1 + PyTorch 2.9.1

The full fwd-bwd pipeline at production shape (B=1 T=16384 H=12 K=V=128):

| Phase      | FLA fwd (ms) | FastKDA fwd (ms) | Speedup | Numerical vs FLA |
|------------|--------------|------------------|---------|------------------|
| Round-1    | 5.13         | 66 (chunkwise CUDA, naive) | 0.08x | cos=1.0 (correct but slow) |
| Round-2    | 5.13         | 32 (4 opts applied) | 0.16x | cos=1.0 |
| Round-3    | 5.13         | (Triton recurrence + V-split) | — | partial |
| Round-5    | 5.13         | 3.14 (reg-resident h_prev) | **1.63x** | cos=0.92 WRONG |
| Round-9    | 5.38         | 5.15 (applies (I-L)^-1) | **1.04x** | cos=0.97 PARTIAL |

The story is: 5 weeks of optimization, then 1 week of discovering that the
fastest version was numerically broken, then a fix that cost 50% of the
speedup. The deep-numerics investigation at the end pinned the residual
error to two specific algorithmic gaps versus FLA. Going forward (per
project direction), the focus shifts to **optimizing FLA's KDA fwd
directly** rather than competing port.

## Timeline of commits

(Only `feature/cuda-kernel-optim` + `feature/fastkda-r9-dev` shown.
Pre-merge cleanup commit `cddf3b2` not counted.)

### Round-1: first CUDA KDA forward (a8d101c)

Goal: a chunkwise CUDA implementation of KDA forward as a perf baseline.

- One CUDA kernel per stage: `forward_sub`, `delta_h`, `wy_transform`,
  `chunk_o`. Each launches its own grid.
- Used cuBLAS for the 10-pair intra bmm (causal 16x16x128 across chunks).
- Result: 66 ms at prod shape. **5.13x slower than FLA.**
- All outputs cos=1.0 vs FLA. Correctness was never the problem in Round-1.

What we learned:
- Per-stage launches have fixed launch overhead × N_stages. Compounding.
- cuBLAS for the 10-pair causal bmm is overkill — Triton can do this in
  one kernel.

### Round-2: opt #1–#4 (a42345a → aa6e506)

Goal: shrink the Round-1 implementation via targeted micro-opts.

- **opt #2** — `delta_h` launch_bounds + remove 2 redundant `__syncthreads`.
  Saved ~0.5 ms.
- **opt #3** — Wrapper dead-code + redundant `.contiguous()` removal.
  Saved ~0.3 ms.
- **opt #4** — `delta_h` switched to WMMA TF32 tensor cores (the single
  biggest win). Saved ~6 ms.
- Total: 66 → 32 ms (Round-2 end). Still 0.16x of FLA, but the path was
  clear: the WMMA win said "tensor cores matter, more fusion matters".

### Round-2A: chunk_o → Triton (f4693a7 → 0cc09a6)

- R2A — chunk_o (largest single bottleneck at ~9 ms) ported to Triton.
  Initial port had a missing `o_p` pointer in `_chunk_o_kernel` — fixed
  in `0cc09a6`.
- Total: 32 → ~22 ms.

What we learned:
- Triton's bf16 mma codegen beats hand-written WMMA at K=128.
- One missing kernel arg was invisible in the perf numbers but produced
  wrong outputs; a numerical check would have caught it before the
  commit, but we relied on smoke-only at this stage.

### Round-3: Triton recurrence kernel + V-split (9696bf7)

The structural change. Replaced `delta_h` + `wy_transform` + `chunk_o`
chain with a single Triton recurrence kernel that processes all chunks
sequentially inside one program.

Key design:
- Grid: `(H, V // BLOCK_V)`. With `BLOCK_V=16`, grid = `(H, 8)` = 96
  programs on 36 SMs (full saturation).
- Each program runs `for c in 0..NC-1`: load `h_prev` from gmem (or
  initial_state for c=0), compute chunk contributions, store `h[c]`.
- V-split is exact (the KDA recurrence decomposes per V column).

Why V-split works:
- `v[:, :, :, v_block]` and `h[:, :, :, v_block]` are independent across
  `v_block`.
- `h_prev` (= `[K, BLOCK_V]` fp32) is small enough to stay in registers.
  No spills.

The mandatory Python for-loop in the kernel is the cost of
sequential-chunk correctness — Triton's launch order isn't guaranteed, so
chunk `c` could read `h[c-1]` before the writer program finishes.
Solved by "one program per (head, v-block) processes ALL chunks".

### Round-4: register-resident h_prev (02f158a)

The biggest single perf jump in the whole project. Discovery: with V=128
and `BLOCK_V=16`, `h_prev` (= `[K=128, BLOCK_V=16]` fp32 = 8 KB) fits
comfortably in registers across all 8 V-slices. Round-3 was re-reading
`h_prev` from gmem every chunk (8 KB × 1024 chunks = 8 MB extra HBM
per head).

- Register-resident `h_prev` survives the inner chunk loop.
- Adaptive `num_stages` based on shared-memory budget.

Result: 22 ms → ~9 ms.

What we learned:
- The V-split decision from Round-3 had a hidden constraint: BLOCK_V
  must be small enough that `h_prev` × sizeof(fp32) × K fits in the
  register budget.
- 8 KB is the sweet spot for 5060 Ti (per-SM register file is 256 KB,
  occupancy is fine).

### Round-5: cos=0.92 — the bug we didn't know we had (3d648b9)

Round-5 numbers were *beautiful*:

```
Round-5 (broken):    prep 0.99 + rec 2.15 = 3.14 ms (1.63x vs FLA 5.13 ms)
                                                — cos=0.92 WRONG
```

But `cos=0.92` is a HUGE red flag that was waved away because the perf
was so good. The recurrence at this point was:

```python
o_chunk = Mqk[c] @ V_chunk + q_decayed[c] @ h_prev
h_new   = h_prev * exp(g_total[c]) + k_restored[c]^T @ V_chunk
```

Note: prepare computed `INV` (the (I-L)^-1 forward sub) and stored it to
gmem, but the recurrence IGNORED it. This is the "naive" form — it
captures the cross-chunk state contribution but misses the intra-chunk
(I-L)^-1 contribution. cos=0.92 was the symptom.

The fix would have been easy to spot at Round-5 if we'd:
1. Run `chunk_kda` reference vs FastKDA and looked at the **mean / median
   relative error**, not just cosine similarity.
2. Tested multiple random seeds (cos is fragile to cancellation effects).

See [Round-9](#round-9-the-fix) for the actual fix, and [deep-numerics](#deep-numerics-investigation)
for the systematic error analysis that came after.

### Round-6: bwd analysis + V-split tension (b774c06 → 957f17f)

Forward was at 1.63x of FLA (cos=0.92). The bwd was still FLA-vendored
at 13.5 ms (2.6x fwd) — 73% of step time.

- Stage breakdown: intra (32%) + wy_dqkg (28%) = 60% of bwd.
- Data movement analysis: every bwd subkernel is bandwidth-bound.
- V-split/V-loop tension documented: BLOCK_V=64 has better V-loop reuse
  but worse register pressure; BLOCK_V=16 has the inverse.

Decision: don't write a Triton bwd at this point. FLA's bwd is already
bandwidth-bound and rewriting it in our style would yield marginal
gains at significant engineering cost. Move forward with FLA bwd.

### Round-7: CHUNK=16 is the algorithm's hard limit (38a2d9e)

Tried CHUNK=32. Got NaN. Root cause: `exp(g_cumsum)` overflow at
g_cumsum ≈ 88 (bf16 max ≈ 3.4e38, but cumulative sums in linear-attention
gates easily exceed this for T=1024 with the production `lower_bound=-5`).

- CHUNK=16 keeps `g_cumsum[i] - g_cumsum[j]` ≤ 16 × g_max per chunk.
- With g_max ≈ 5 (typical after sigmoid + lower_bound=-5), max
  exponent is ≈ 80, bf16-safe.
- CHUNK=32 doubles the exponent budget to ≈ 160, borderline; CHUNK=64
  definitively overflows.

Pinned: CHUNK=16 forever. Documented the math.

### Round-8: per-chunk rescaling + fp32 k_inv experiment (118e05a)

Tried to push perf by computing things in higher precision and casting
down at the last moment. Result: marginal gains (0.99 → 0.97 ms in
prepare) at complexity cost. Adopted fp32 `k_inv` storage as a stable
choice; rejected per-chunk rescaling as too brittle.

### Round-9: the fix (66bc532 was the wire-up commit; the kernel fix
lives on `feature/fastkda-r9-dev` ee74f21)

After Round-5 was caught (5 weeks after deployment), the fix landed:

1. **Prepare:** precompute `Mqk_eff = Mqk @ INV` and `K_pre = INV @ k_restored`.
   Drop `inv_ptr` from the recurrence workspace (no longer stored). The
   two extra prepare bmms cost <5%.
2. **Recurrence:** add per-chunk `v_residual = V - k_decayed @ h_prev`
   subtraction + `* beta` multiplication. The recurrence now matches
   FlashKDA's reference (line-by-line) and FLA's chunk_kda path.
3. Wrapper signature: gained `k_decayed` and `beta` args.

```
Round-5 (broken):    prep 0.99 + rec 2.15 = 3.14 ms (1.63x) — cos=0.92 WRONG
Round-9 (correct):   prep 0.98 + rec 4.18 = 5.15 ms (1.04x) — cos=0.97 OK
```

50% of the prior speedup was spent on correctness.

### Deep-numerics investigation (2026-06-30, on `feature/fastkda-r9-dev`)

User pushed back: "嗯数值检验你还是看一眼相对误差和绝对误差吧，只看cos不保险"
("for numerical validation look at relative and absolute error too,
cos alone isn't reliable").

This investigation uncovered that `cos=0.97` (Round-9) is also misleading.

**The picture:**

- FLA's `h` vs per-token fp64 ground truth: `cos=0.9997`, **`med_rel=2.0%`** (bf16 noise floor).
  FLA is correct.
- FastKDA's `h` vs the same ground truth: `cos=0.99`, **`med_rel=8.5%`** (4x noise floor).
  FastKDA is OFF.
- FLA's `o` vs FastKDA's `o`: `cos=0.97`, `med_rel=15-17%` (8x noise floor). Systematic.

So Round-9 reached "looks similar from afar" but is **systematically
divergent** from FLA on a per-element basis. The cos metric is hiding
real error.

**Root causes pinpointed:**

1. **`INV` computation differs.**
   FastKDA `prepare.py:222-227`: iterative `(I-L) @ (I+L)^k` accumulation
   (NOT a true Neumann series; the kernel comment is misleading).
   FLA `chunk_intra.py:286-325`: exact block-partitioned forward sub via
   4 `SOLVE_TRIL_DOT_PRECISION` matmul chains. Avoids the
   iterative-convergence error that the comment claim doesn't actually
   deliver.
   Replaying prepare + recurrence in PyTorch with the FLA-style exact
   INV brings med_rel from 7-11% down to ~2% on tested shapes.

2. **`v_xc` (cross-chunk v subtraction) differs.**
   FastKDA: `v_xc = k_decayed @ h_prev`.
   FLA: `b_v = v - w @ h_prev.T` where `w = INV @ (k * beta * exp)`.
   These are algebraically different formulations:
   - The per-token unrolled GDR uses raw `k @ h_prev` (no decay, no INV).
   - FastKDA's `k_decayed @ h_prev` has g-decay on k but no beta.
   - FLA's uses w which has both beta and g-decay AND INV pre-applied.

   The fix is to switch to FLA's `w` form. The matmul count doesn't grow
   (load `w` instead of `k_dec`).

**The decision:** the user chose to abandon the numerical fix attempt
because the cost (likely 5.7-7.0 ms — losing the 1.04x parity entirely)
is too high for a port that will be replaced by FLA optimization work
next. The investigation scripts are preserved on
`feature/fastkda-r9-dev` in `test/_tmp/` for re-runs.

## Bwd / ratio state (unchanged throughout)

- Bwd 13.5 ms (2.6x fwd), FLA vendored kernels. 73% of step time.
- `docs/fast_kda_bottleneck_analysis.md` for the per-subkernel breakdown.

## Performance data archive

All timings measured on RTX 5060 Ti 16G at prod shape (B=1 T=16384
H=12 K=V=128). Median over 50 iters back-to-back via CUDA events.

| Round | prep (ms) | rec (ms) | fwd total | vs FLA | numerical |
|-------|-----------|----------|-----------|--------|-----------|
| 1     | 19.4 (intra) + 4.4 (delta_h) + 4.7 (wy) + 0.65 (fwd_sub) + 3 (chunk_o) + cuBLAS = 66 ms (loose) | — | 66 | 0.08x | cos=1.0 |
| 2     | (CUDA path) | — | 32 | 0.16x | cos=1.0 |
| 2A    | (CUDA path) | — | 22 | 0.23x | cos=1.0 |
| 3     | (Triton path) | — | ~9 | 0.57x | cos=1.0 |
| 4     | 0.99 | 9.0 (recurrence re-reads h_prev) | ~10 | 0.51x | cos=1.0 |
| 5     | 0.99 | 2.15 | 3.14 | 1.63x | cos=0.92 WRONG |
| 6-8   | (V-split / CHUNK experiments) | | | | |
| 9     | 0.98 | 4.18 | 5.15 | 1.04x | cos=0.97 / med_rel=15-17% PARTIAL |

## What to take forward (next phase: optimize FLA's KDA fwd)

The user's instruction was clear: don't keep trying to fix the FastKDA
port. Optimize FLA's KDA fwd directly. The key learnings to apply:

1. **V-split is the right structural move.** FLA doesn't V-split in
   chunk_kda (it uses BLOCK_V=64 typically). Going to BLOCK_V=16
   doubles the grid (H, 8) → full SM saturation. `h_prev` register
   residence is feasible (8 KB per program).
2. **The matmul chain matters more than the matmul count.** Round-9
   gained correctness by adding one matmul but lost 50% perf because
   the 4-deep dependency chain serializes the pipeline. FLA's
   structure has the same shape — there's room to break the chain.
3. **CHUNK=16 is fixed forever** (Round-7's overflow finding).
4. **Numerical validation: always include med_rel alongside cos.**
   This is now baked into the project's verification rules.
5. **Pre-compute `(I-L)^-1` ONCE per chunk in the prepare pass.**
   The recurrence should consume `Mqk_eff` and `K_pre` directly. FLA
   currently computes `(I-L)^-1` inside the fused inter+solve kernel
   each call; it's a candidate for being hoisted to a separate
   kernel-call boundary if the chain can be restructured.
6. **`v_xc` should use FLA's `w` form** (the deep-numerics finding).
   This is free perf-wise (same matmul count) and aligns the recurrence
   with FLA's.

## Files & references

- `src/models/ops/_triton/fast_kda/prepare.py` — Kernel 1
- `src/models/ops/_triton/fast_kda/recurrence.py` — Kernel 2
- `src/models/ops/_vendored/fla/ops/kda/backends/fastkda.py` — Backend
  wiring (`FastKDABackend`)
- `src/models/ops/kda.py` — Production wrapper (`chunk_kda` calls
  `FastKDABackend` if enabled)
- `bench/fast_kda_recurrence_bench.py` — Per-stage recurrence timing
- `bench/fast_kda_prepare_bench.py` — Per-stage prepare timing (added
  Round-9)
- `docs/fast_kda_bottleneck_analysis.md` — Roofline + data-movement
  analysis (Round-6)
- `docs/kda_kernel_structure.md` — FLA call chain (the
  post-FastKDA-deprecation reference)
- `docs/efkda_debugging_history.md` — Historical EFKDA fixes
  (pre-KDA era)
- Branch `feature/fastkda-r9-dev` — Round-9 + deep-numerics
  investigation, dev-only

## Verification commands

For any future contributor who wants to reproduce:

```bash
# Stage breakdown at prod shape
python bench/fast_kda_recurrence_bench.py

# End-to-end FLA vs FastKDA fwd (needs both kernels compiled)
python bench/fast_kda_prepare_bench.py

# Numerical — MUST include med_rel alongside cos (per deep-numerics)
python test/_tmp/test_round9_h_vs_o.py  # on feature/fastkda-r9-dev

# Smoke test on the prod training stack
python scripts/train.py --config configs/base.yml --use_dummy_data \
  --max_steps 4 --tp_sim --tp_size 2 --gradient_accumulation_steps 2 \
  --batch_size 2 --seq_len 512 --num_layers 4 --num_blocks 2
```