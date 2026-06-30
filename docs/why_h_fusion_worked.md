# Why did Lever H fusion work when the prior 3 fusions failed?

**Date:** 2026-06-30
**Question:** intra_solve 10-pair bmm loop fused into 1 Triton kernel saves 18 ms
at prod. But R1 intra-bmm-batching, R2 intra+wy fusion, R2 precomputed r/c all
failed (NO ROI, reverted). What changed?

Cross-references:
- `bench/kda_fwd_bench.py` (the prior fusion summary text)
- `docs/chunkwise_kda_opportunities.md` (chunkwise KDA algorithmic ceiling)

## TL;DR

| Fusion | Result | Why |
|---|---|---|
| H (intra_solve → 1 Triton kernel) | **+18 ms win** | 20 launches → 1, intermediates in regs |
| R1 (intra bmm batching) | NO ROI, reverted | Concat to fewer bmms adds memory pressure |
| R2 (intra+wy fusion) | NO ROI, reverted | Forced to wait for forward_sub; can't fuse |
| R2 (intra precomputed r/c) | NO ROI, reverted | Per-(s_i, anchor) precompute adds memory |

The pattern that won: **N independent micro-kernels, each launch-bound (not
compute-bound), share the same inputs, with element-wise intermediates.**

The pattern that lost: any of these four — (a) trying to materialize a shared
intermediate across heterogeneous pairs, (b) fusion across a sequential
dependency, (c) precomputation that costs more HBM than it saves, (d) trying
to fuse across chunk boundaries when each chunk is a separate matmul.

## 1. Why Lever H worked

The 10-pair loop was 19.4 ms at prod. The fused Triton kernel is 1.2 ms.
Sources of the win:

### 1a. Launch overhead reduction (biggest win, ~10 ms)

Each cuBLAS bmm at this shape has ~5 ms of CPU launch overhead vs ~0.1 ms
of actual GPU compute. The per-pair bmm is tiny:
- [N=3072, M=16, K=128] × [N=3072, K=128, N=16] = 805 MFLOPs
- At 165 TFLOPs bf16 mma peak (RTX 5060 Ti): 0.005 ms compute
- At 5 ms launch overhead: 1000x compute-bound

20 cuBLAS launches × ~5 ms overhead = ~100 ms of pipeline stalls (overlapped
with compute, but still ~10 ms wall).

1 Triton launch with 122,880 programs = ~1 ms launch overhead. Save: ~9-10 ms.

### 1b. Intermediate elimination (small win, ~2-3 ms)

The Python loop allocated 30 small tensors per call:
- 10 q*r outputs of [N, BC=16, K=128] bf16 = ~10 MB each
- 10 k*c outputs of [N, BC=16, K=128] bf16 = ~10 MB each
- 10 k_row*r outputs of [N, BC=16, K=128] bf16 = ~10 MB each
- Plus bmm outputs, scale, etc.

Total intermediate writes: ~600 MB per loop. At 448 GB/s = 1.3 ms of HBM
write traffic, plus the allocator overhead (similar magnitude).

In Triton, these stay in registers. No allocation, no HBM traffic. Save:
~2-3 ms.

### 1c. The pair independence condition (why it's fusable)

Each pair (s_i, s_j) reads from DIFFERENT slices of q, k, g_per but uses
THE SAME KERNEL PATTERN. There's NO cross-pair data flow — each pair writes
to its own [BC, BC] sub-block of A_qk and A_kk. So 10 pairs can be computed
by 10 different programs without any cross-program synchronization.

This is the structural condition that makes fusion easy. If pair N+1
needed to read pair N's output, we'd need a barrier or a memory round-trip
between them — that would defeat the fusion.

### 1d. Shared input reloaded (free L2 win)

All 10 pairs for a given (chunk, hv) read from different slices of the SAME
q, k, g_per tensors. With one kernel and grid (10 pairs × num_chunks*HV):
- Programs sharing pid_n (chunk, hv) read overlapping input slices
- The L2 cache (32 MB on 5060 Ti) catches these — q[k,v,g] for one
  (chunk, hv) is only ~24 KB total, easily fits in L2

Across pid_n for the same pid_pair, the slice data is different but the
overall access pattern is regular. Total HBM: ~1.2 GB instead of the naive
3 GB that 20 separate launches would incur.

## 2. Why the prior fusions failed

### 2a. R1 (intra bmm batching) — `opt#1` in the bench summary

**What was tried:** Combine the 10 pairs into fewer (e.g., 1 or 2) bigger
bmms by concatenating the (q*r, k*c) inputs along the M or N axis.

**Why it failed:**
- The 10 pairs use different slices of q/k with different r/c scales.
  To concatenate, you have to MATERIALIZE the q*r and k*c slices first.
- Materialization costs: 10 pairs × 3 tensors = 30 device-to-device copies
  of [num_chunks*HV, BC, K] each. At prod this is ~600 MB of write traffic
  + 600 MB read = 1.2 GB extra HBM. At 448 GB/s = 2.7 ms overhead.
- The bigger bmm is still cuBLAS — same launch overhead per launch,
  same kernel selection overhead.
- Plus 2x VRAM for the materialized intermediates.

**Lesson:** Concatenation-based "fusion" only helps when the input
layouts are already compatible (e.g., all use the same q slice). The intra
loop's pairs have different slices — concatenation adds memory cost that
exceeds the launch-overhead saving.

### 2b. R2 (intra_solve + wy_transform fusion) — `opt#5` in the bench summary

**What was tried:** Fuse the 10-pair bmm loop with the wy transform (u = A^-1 @ v
* beta, w = A^-1 @ k * beta * exp2(g)) into one kernel.

**Why it failed:**
- wy_transform reads A_inv, which is the output of forward_sub.
- forward_sub is an in-place matrix inverse (sequential within rows).
- The dependency chain is: intra_solve → forward_sub → wy_transform.
- You CAN'T fuse across this chain because forward_sub needs to complete
  for the whole A_inv before wy_transform can use it.
- Any "fusion" attempt either (a) leaves forward_sub as a separate kernel
  and doesn't help, or (b) duplicates forward_sub inside the wy kernel
  and is way more expensive than the original.

**Lesson:** Sequential dependencies defeat fusion. The intra_solve →
forward_sub → wy chain has 2 hard barriers (memory round-trips for A_inv)
that can't be elided.

### 2c. R2 (intra precomputed r/c) — `opt#6` in the bench summary

**What was tried:** Precompute the r and c blocks once per (s_i, anchor_type)
pair, then use them in the 10-pair loop.

**Why it failed:**
- For s_i, the r block depends on whether the pair is diag (anchor =
  middle) or off-diag (anchor = row start). So r for s_i has 2 variants.
- Precomputing 4 s_i × 2 anchor_type = 8 r blocks = 8 × [num_chunks*HV,
  BC, K] fp32 = 64 MB at prod. Plus 8 c blocks = 64 MB.
- Total precompute HBM: 128 MB read + 128 MB write = 256 MB. At 448 GB/s
  = 0.57 ms.
- The savings: skip 30 element-wise ops (q*r, k*c, k_row*r) inside the
  loop. Each is [num_chunks*HV, BC, K] = 24 MB. 30 × 24 = 720 MB. So
  the savings is 720 MB / 448 GB/s = 1.6 ms.
- Net: 0.57 - 1.6 = +1.0 ms LOSS. Plus the launch overhead for the
  precompute kernel.

**Lesson:** Precomputation helps only when the materialized data is reused
many times. Here, each r/c block is used once per pair — so the precompute
cost is paid back 1x at best.

### 2d. What's the meta-pattern?

The failed fusions all tried to ADD work (concat, precompute) to reduce
launch overhead. The work they added cost more than the launch overhead
saved. The H fusion only ELIMINATED work (no precompute, no materialization)
and reduced launch overhead.

**Rule of thumb:** Fusion is a win iff the work eliminated (in HBM bytes +
launch overhead) exceeds the work added (extra reads, extra intermediates,
extra synchronization).

## 3. The H-pattern: when does it generalize?

H-fusion applies when ALL these hold:
1. **N independent compute units**, no cross-unit data flow.
2. Each unit does **element-wise + small matmul** (intermediates fit in
   registers).
3. Units share the **same input slices** or the input layout is regular
   enough that L2 reuses across programs.
4. Each unit is **launch-bound, not compute-bound** (small matmul + small
   element-wise = small compute, but launch overhead dominates).
5. **No sequential dependency** between the units and downstream
   consumers (the output of the fused kernel is the START of the next
   stage, not the MIDDLE).

## 4. Where the H-pattern applies elsewhere in the KDA pipeline

### 4a. wy_transform (4.7 ms at prod) — NEXT TARGET

```python
# Current: 2 cuBLAS bmms + 4 element-wise + 4 .to() casts
beta_v = (v_per_stacked * beta_stacked.unsqueeze(-1)).contiguous()         # ~0.5 ms
u = torch.bmm(A_kk_fp32, beta_v.to(torch.float32)).to(torch.bfloat16)      # ~2.3 ms
g_cum_exp2 = g_per_stacked.exp2()                                          # ~0.3 ms
k_with_beta_g = (k_per_stacked * beta_stacked.unsqueeze(-1) * g_cum_exp2.to(torch.bfloat16)).contiguous()  # ~0.5 ms
w = torch.bmm(A_kk_fp32, k_with_beta_g.to(torch.float32)).to(torch.bfloat16)  # ~2.3 ms
# Total: 2 launches (u, w), 2 intermediates (beta_v, k_with_beta_g)
```

Fused pattern: ONE Triton kernel that loads A_kk_fp32 once into shmem and
produces both u and w. Grid: (max(V,K)//BV, num_chunks*HV). Each program:
- Loads A_kk [BT=64, BT=64] fp32 into shmem
- Computes beta_v in registers, does bmm(A_kk, beta_v) → u slice
- Computes k_with_beta_g in registers, does bmm(A_kk, k_with_beta_g) → w slice
- Writes u and w

Estimated savings:
- 1 launch elimination: ~2 ms (half of the 4.7 ms)
- Intermediate elimination (beta_v, k_with_beta_g): ~1 ms
- Cast elimination (.to(float32) before, .to(bfloat16) after): ~0.5 ms
- **Total: ~2-3 ms at prod**

### 4b. forward_sub + wy_transform fusion (5.4 ms combined)

**Attempted 2026-06-30 as Lever J. RESULT: LOSS (-0.42 ms at prod).**

Tried to inline forward_sub into the wy_transform kernel via tl.static_range(64)
loop with 2D reductions. Input changed from POST-inverse A_inv to PRE-inverse
A = I + A_kk_unscaled.

Measured at prod (N=3072, BT=64, V=K=128):
- Old: _forward_sub(CUDA, 0.65 ms) + wy_fused_transform(Triton, 0.9 ms) = 1.77 ms
- New: wy_fused_transform_with_inv (Triton fused forward_sub + wy) = 2.19 ms
- Delta: **-0.42 ms LOSS**

Numerical correctness: cos=0.99999x, med_rel<0.5% (PASS at all 3 shapes).

**Why it failed:** The custom CUDA forward_sub is highly optimized for the
sequential-within-row access pattern. It processes 64 rows in parallel (one
row per thread), each doing simple cumulative updates. Triton's general
2D-reduction-via-`tl.where`-masks adds ~1.3 ms of work that the CUDA kernel
avoids. The fusion saves the launch overhead (~0.65 ms) but adds ~1.3 ms of
Triton-side forward_sub work, net -0.65 ms vs separate kernels.

**Lesson:** Fusion is NOT a win when the replaced kernel is highly optimized
for its specific access pattern. forward_sub's pattern is: 64 sequential rows,
each with simple cumulative updates — perfectly matched to CUDA threads.
Triton's general-purpose 2D reductions add overhead the specialized kernel
doesn't have.

**Tried-and-reverted. Files kept:** `src/models/ops/cuda/kda_fwd/triton_wy_transform_with_inv.py`
and `test/_tmp/test_wy_with_inv.py` for reference (delete before commit).

### 4c. post (beta/mask/eye = 0.7 ms) — REMOVE, not fuse

The mask multiplication is redundant:
```python
A_kk_unscaled = A_kk * beta_stacked.unsqueeze(-1)  # A_kk is already lower-tri (zero upper)
A_kk_unscaled = A_kk_unscaled * mask.unsqueeze(0)   # mask zeros the diagonal AND off-diag-on-mask
A_kk_fp32 = A_kk_unscaled + eye                     # sets diagonal to 1
```

For the lower-tri (where the kernel wrote): `A_kk * beta * mask * 0 → 0` then `+ 1`.
For the upper-tri (where the kernel didn't write, A_kk is already 0): `0 * mask * 0 = 0` then `+ 0`.
For the diagonal (kernel wrote, mask is 0): `A_kk * beta * 0 = 0` then `+ 1`.

So mask multiplication is unnecessary. Removing it saves the mask allocation
+ 2 element-wise ops = ~0.7 ms.

**But this is not a fusion — it's a removal. Different category.**

### 4d. setup (3.2 ms) — hard to fuse

The setup does (g.float() * scale).cumsum() and 5 .contiguous() calls
after transpose. The cumsum is sequential within a chunk — can't parallelize.
The transposes are necessary because the downstream kernels expect
contiguous [num_chunks, HV, BT, K] layouts.

Could fuse the .contiguous() into the intra_solve kernel via strided loads:
- Save 5 .contiguous() calls = ~1.4 ms
- Lose some HBM efficiency (strided loads are slower than coalesced)
- Net: uncertain, probably small win

## 5. Summary

**H worked** because all 4 conditions held simultaneously: independent units
+ element-wise intermediates + shared inputs + launch-bound.

**H-pattern next target:** wy_transform fusion (4a). Estimated ~2-3 ms
savings at prod. Lower ROI than H but a clean application of the same
pattern. Worth attempting.

**Don't try:** any fusion that crosses forward_sub (the chain has hard
sequential dependencies), any precomputation that materializes more than
it saves, any concat that requires materializing heterogeneous inputs.

## 6. Plan

1. Attempt wy_transform fusion (4a). Expected ~2-3 ms savings.
2. If 4a wins, attempt forward_sub + wy_transform fusion (4b). +0.65 ms.
3. Remove redundant mask multiplication (4c). +0.7 ms.
4. Total expected additional savings: ~3-4 ms (brings 15.6 ms → ~12 ms).

If wy_transform fusion fails, the meta-lesson generalizes: the H-pattern
is rare (only intra_solve + wy_transform qualify in the current pipeline),
and the wrapper is near its algorithmic ceiling.