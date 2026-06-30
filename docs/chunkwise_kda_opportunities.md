# Chunkwise KDA — when does it have opportunity?

**Date:** 2026-06-30
**Scope:** answer two related questions, motivated by the fact that
the current chunkwise path (FastKDA Round-9 + FLA) has hit HBM peak.

1. Why is the "chunkwise-parallel form avoiding matrix inverse"
   infeasible? (Already verified — but now we want the complete
   argument so we can recognize the boundary conditions.)
2. Under what conditions does chunkwise KDA have opportunity to
   break the HBM ceiling?

Cross-references:
- `docs/fast_kda_bottleneck_analysis.md` Round-8 ("Why parallel scan
  doesn't win", "Why CHUNK=16 is the algorithm's hard limit").
- `docs/efkda_debugging_history.md` (the 3-kernel split dead-end,
  the per-tile loop that ran 18+ min at K=64 and OOM'd at K=128).
- `docs/triton_kernel_playbook.md` (decision tree for kernel design).

## TL;DR

The closed-form-with-inverse form of chunkwise KDA is **the natural
form for tensor cores**. The "matrix inverse" is a math trick that
turns a sequential recurrence into one big parallel matmul per
chunk. Avoiding it costs mma granularity (the dominant reason) and
gets you exp overflow (which is form-independent) for free.

Chunkwise has opportunity to break the HBM ceiling only when:
- **CHUNK > 16** (architectural: precision-tolerant gate or fp64)
- **K > 128** (architectural: AI grows ~linearly)
- **Hardware offers wgmma + TMA + cluster** (Hopper+ / Blackwell)
- **Algorithmic refactor: decouple decay accumulation from k·v
  product** (allows each piece to be optimized independently)

None of these are "easy wins". The current path is at its
algorithmic ceiling.

## 1. The math, restated

KDA is a delta-rule linear attention with per-head log-space decay:

```
h_t = exp(g_t) · h_{t-1} + β_t · (k_t v_t^T − k_t k_t^T h_{t-1})
o_t = q_t^T h_t
```

Three ways to compute this within a chunk of M tokens:

### 1a. Sequential (the natural recurrence)

```
for t in 0..M-1:
    h_t = exp(g_t) · h_{t-1} + β_t · (k_t v_t^T − k_t k_t^T h_{t-1})
    o_t = q_t^T h_t
```

M dependent steps. **Inherently sequential** — can't use TensorCores
on the recurrence itself. Only the per-step `k_t v_t^T` outer
product could use a TensorCore, but it's [K, 1] @ [1, V] = a rank-1
update, way below the [16, 16, 16] mma granularity.

### 1b. Closed-form-with-inverse (the parallel reformulation)

```
A = I + L     where L[m,n] = exp(g_cum[m] − g_cum[n]) · β[m] · k[m] · k[n]^T  (m > n, else 0)
            = lower-triangular, [M, M]
A · w = β · k · exp(g_cum)   ⟹   w = A^{-1} · β · k · exp(g_cum)
A · u = v                      ⟹   u = A^{-1} · v
h_M = w^T h_prev + u           (close the chunk; h_prev is the chunk-boundary state)
o[m] = q[m]^T h_M · scale + Σ_n A[m,n] · v_new[n]
```

One big matmul `[M, K] @ [K, M]` for L, one triangular forward-sub
(M² flops, basically free), two matmuls `[M, M] @ [M, K]` and `[M, M]
@ [M, V]` for w and u. **All TensorCore-friendly**: the matmul
sizes are `[16, 128] @ [128, V_tile]`, perfect m16n8k16 fit.

### 1c. Parallel scan (Blelloch)

```
Operator:  (h_a, α_a) ⊕ (h_b, α_b) = (h_b · α_a + h_a, α_a · α_b)

Step 1: pair adjacent tokens, compute partial recurrences
        (h_{2i}, h_{2i+1}, α_{2i}·α_{2i+1}) for i in 0..M/2-1
Step 2: pair partials, halve again
...
Step log M: combine the whole chunk.
```

At each step the matmul size halves: [M, K] @ [K, V], then
[M/2, K] @ [K, V], then [M/4, K] @ [K, V], ..., finally
[1, K] @ [K, V]. The last two steps are <50% mma utilization.

## 2. Why (1b) wins over (1a) and (1c)

### 2a. Sequential vs closed-form

The closed-form trades M sequential steps for one big matmul + a
free inverse. The matmul is the win because:

- TensorCore mma needs operands ≥ 16 in M and N, ≥ 16 in K. The
  per-step `k_t v_t^T` outer product is [K, 1] @ [1, V] — completely
  wasted as a TensorCore op. It runs on CUDA cores at fp32
  accumulator speed, which is ~30× slower than TensorCore flops.
- The closed-form `[M, K] @ [K, M]` with M=16, K=128 fits the
  m16n8k16 instruction exactly (16 rows × 8 cols per warp,
  16-deep K reduction). One matmul, fully utilized.
- The closed-form `A^{-1}` via triangular forward sub is O(M²/2)
  flops — for M=16 that's 128 flops per row, total 2K flops per
  chunk. **Two thousand flops is 5 ns of work at 419 TFLOPs.** It's
  invisible next to the matmul.

Sequential loses by a factor of ~M (the number of dependent steps)
on the recurrence work. That's ~16× slower on the recurrence. The
only way sequential could win is if M is large enough that the
TensorCore under-utilization in closed-form exceeds the sequential
dependency cost — but the closed-form is fully utilized at M=16+, so
this never happens.

### 2b. Parallel scan vs closed-form

Same matmul work in aggregate, but the scan form's matmul sizes
shrink geometrically. At each step:

| step | matmul size (m, K, V) | mma util |
|-----:|----------------------:|---------:|
| 1    | (16, 128, 128)        | 100%     |
| 2    | (8, 128, 128)         | 50%      |
| 3    | (4, 128, 128)         | 25%      |
| 4    | (2, 128, 128)         | 12.5%    |
| 5    | (1, 128, 128)         | 6.25%    |

The closed-form has 1 matmul at 100% util. The scan has 4 matmuls
at geometrically-decreasing util. The scan form is **strictly worse
on the recurrence work** at M=16.

If M were larger (say M=64), the scan form would do 6 steps with
the first few at 100%, so the average is much better — but M=64
suffers from the exp overflow (see §2c) and from the L matrix
filling 8 KB per chunk in bf16 (precision issues with `||L|| < 1`).

### 2c. The exp overflow at M > 16 is form-independent

This is the subtle one. Whether you compute the cumulative decay
sequentially, via scan, or via the closed-form, the **dynamic
range** of the cumulative product is the same:

```
r[s] = exp(Σ_{t≤s} g_activated[t])    ∈ (0, 1]      safe
1/r[s] = exp(−Σ_{t≤s} g_activated[t]) can be ≫ 1     ← can overflow bf16/fp32
```

The closed-form needs `exp(−g_cumsum)` for `k_inv`, which is what
overflows. The scan form needs `exp(−Σ g)` for the backward product
of α, which is the same overflow. **The form doesn't change the
arithmetic.** You'd need:
- fp64 storage (threshold ~284, but 2× slower than fp32 mma)
- per-step rescaling (subtract max-so-far before exp) — equivalent
  to capping M at 16 in practice
- a different gate formulation that doesn't use `exp(g_cumsum)` —
  architectural

So **the closed-form is the natural form, but the chunk size is
capped at 16 by the gate, regardless of form.**

### 2d. Backward complexity

The closed-form bwd uses `(I − L)^{-1}` as a black box:

```
dL^{-1} = −L^{-1} · dL · L^{-1}
```

Direct application of chain rule on the inverse. ~5 lines of
autograd.

The scan form bwd requires deriving the gradient through the
log-step scan structure. Each scan step has its own forward (h_b ·
α_a + h_a, α_a · α_b), so each step needs its own bwd rule
chain-ruled back through both `h_a`, `h_b`, `α_a`, `α_b`. The
total is log(M) × M dependencies to derive. **Weeks of work for a
kernel that's already at 99% HBM.**

This is a soft reason, but a real one: the engineering cost is high
and the perf upside is zero at M=16.

## 3. When does chunkwise have opportunity?

The current chunkwise path is at:
- **88-99% HBM peak** (memory-bound; compute is essentially free)
- **M = CHUNK = 16** (precision cap from bf16 exp overflow)
- **AI = 6.9 FLOPs/B at K=V=128** (ridge is 935 FLOPs/B; 130×
  below compute-bound)

To break the HBM ceiling, you need to **either reduce HBM traffic
or increase AI enough to hit the ridge**. The closed-form at M=16,
K=128 has done neither — it sits firmly on the floor.

### Opportunity 1: M > 16 (precision-tolerant gate)

The CHUNK=16 ceiling is set by `exp(-g_cumsum)` overflow in bf16.
The threshold is `|g_cumsum| > 88` (ln(3.4e38)). With `lower_bound
= -5` and the standard sigmoid gate, M=32 produces `|g_cumsum| ≈
103` → overflow.

To raise M, you need one of:
- **Per-step fp32 rescaling**: subtract the running max from
  `g_cumsum` before exp. Costs ~5% perf. Equivalent to capping M
  at 16 anyway (the rescale happens at every token).
- **fp64 storage** of `k_inv`: ~2× slower bmm. Eats most of the M
  win.
- **Different gate formulation** that doesn't produce unbounded
  cumulative sums (architectural).

Each doubling of M roughly doubles AI (`AI(M) ≈ M(4K + 8M) / (4M
+ 12K)` at K=128, M ≫ 1). M=32 → AI = 13.5 (2× better). Still 70×
below ridge. M=64 → AI = 27. M=128 → AI = 52. None of these reach
the ridge.

**Verdict: M can grow but the ceiling on M (without architectural
change) is ~32, and even then we're still 70× below compute-bound.
Modest wins possible.**

### Opportunity 2: K > 128 (architectural)

AI grows linearly with K. At M=16:
- K=128: AI = 6.9
- K=256: AI = 12.6
- K=512: AI = 23.8
- K=1024: AI = 46.5
- K=2000: AI = 90 (still below ridge)

To hit the ridge (935) at M=16, need K ≈ 18,000. Implausible for
LM heads.

**Verdict: K growth helps but doesn't reach the ridge alone. Need
to combine with M growth.**

### Opportunity 3: M × K compound

Both levers compound. At M=64, K=512: AI ≈ 110. M=128, K=1024: AI
≈ 280. Still 3× below ridge. **M=128, K=4096**: AI ≈ 900. Ridge.
But M=128 needs precision-tolerant gate (per Opportunity 1).

**Verdict: hitting compute-bound requires both M and K to grow.
Architectural.**

### Opportunity 4: Hardware — wgmma + TMA + cluster (Hopper+/Blackwell)

Hopper (sm_90) and Blackwell (sm_100, sm_120) have:
- **wgmma** (warp-group mma): instructions are larger, e.g.
  m64n128k16. One warp-group instruction does 4× the work of an
  Ampere mma. Effectively raises the "useful matmul" threshold by
  4× in K.
- **TMA** (Tensor Memory Accelerator): async DMA, can overlap
  loads with compute, eliminating launch latency from the critical
  path. Triton's `tl.make_tensor_descriptor` wraps TMA but, per
  FastKDA Round-5 Path C, is currently slower than plain `tl.load`
  on Triton 3.5.1 / sm_120.
- **Cluster launches** (sm_90+): thread blocks can share data via
  distributed shared memory. Enables a persistent kernel pattern
  where multiple programs share workspace without HBM roundtrip.

**Verdict: real wins, but mostly on hardware we don't target.**
The 5060 Ti (sm_120) reports `check_shared_mem('ada') == True` but
not `'ampere'`; it doesn't have wgmma at full Hopper capacity. The
wgmma path would matter on H100 / B200.

### Opportunity 5: Algorithmic refactor — decouple decay from k·v

The current chunkwise algorithm interleaves decay computation
(g_cumsum → r/c scales → A matrix) with the k·v outer product.
This is what makes the per-chunk compute light: each chunk is
mostly exp() + small matmul + small inverse.

A 2-pass formulation could:
- **Pass 1**: compute and store the per-chunk A matrix (decay only)
- **Pass 2**: compute the k·v product using the precomputed A

Each pass becomes more uniform → more TensorCore-friendly. But:
- Adds an extra HBM roundtrip (A matrix saved and re-loaded)
- The original interleaving is what makes the closed-form
  one-shot; separating them is the same compute but more I/O

**Verdict: probably a wash. Would need careful measurement.**

### Opportunity 6: Persistent kernel with workspace sharing (already done)

FastKDA Round-4 already explored this: one program per head,
sequential chunk loop, register-resident `h_prev`. Hit 94% of HBM
peak.

The next step would be: one program per head, sequential chunk
loop, AND shared-memory-resident workspace tensors across chunks.
But the workspace tensors differ per chunk (q, g, A_qk, v_new all
change), so there's no natural sharing. **This lever is exhausted.**

## 4. So what's actually achievable?

Given the analysis above, here are the realistic paths forward,
ordered by ROI:

| Path | Potential gain | Cost | Risk |
|------|----------------|------|------|
| **B: bf16 h in chunk_o (already verified)** | 0.23 ms at prod (~22% of chunk_o, ~4% of fwd pipeline) | 1-line kernel change | low (cos=1.0, max_rel=11% in tail) |
| **F: per-step fp32 rescaling to lift CHUNK to 32** | ~10-20% on recurrence | 5% perf hit on rescale | medium (NaN risk if A_log not converged) |
| **G: TMA loads on sm_120** | ~10-15% on prepare | weeks to debug | high (TMA failed in Round-5) |
| **H: re-write intra_solve as single fused Triton kernel** | ~10-15 ms (intra_solve is 19.4 ms currently) | multi-day kernel effort | medium |
| **I: K grows to 256+** | linear in AI | architectural | depends on model design |
| **J: persistent kernel + cluster launches** | ~30% on recurrence | Hopper/Blackwell-only | not on 5060 Ti |

The cheapest win is **B** (already verified, can land). The biggest
real win is **H** (intra_solve is the biggest bottleneck, ~19 ms,
no algorithmic change needed). Beyond that, the ceiling is
algorithmic.

## 5. The general principle

For any "sequential recurrence + per-token decay" kernel (KDA,
RetNet, Mamba, etc.) running on Ampere-class tensor cores:

- **Closed-form-with-inverse** is the natural form at M=16-32.
  Below M=16, you don't have enough parallelism per chunk.
  Above M=32, you hit exp overflow.
- **Sequential within chunk** loses to closed-form by ~M× on
  TensorCore utilization.
- **Parallel scan** loses on geometrically-decreasing mma
  granularity.
- **FFT-based** requires shift-invariance that KDA's gate breaks.

The way out is not a different formulation. The way out is:
- Larger K (architectural)
- Tolerance to larger M (architectural, or hardware fp64)
- Better hardware (wgmma + TMA + cluster)

The chunkwise algorithm is at its algorithmic ceiling. We've
extracted everything the form allows.

## Files / references

- `docs/fast_kda_bottleneck_analysis.md` Round-8 — full
  parallel-scan analysis + per-step rescaling experiment
- `docs/efkda_debugging_history.md` — K=128 compile cost, 3-kernel
  split dead-end, the historical attempts that failed
- `test/_tmp/test_chunk_o_bf16_h.py` — Lever B verification
  (ship-it, 1.28× faster, cos unchanged)
- `docs/kda_kernel_structure.md` Round-9 — the SAFE_GATE / BK=128 /
  dhu ns=2 optimizations, the FLA-side headroom we already ate

Per CLAUDE.md, this file documents research findings for the
project. The test/_tmp/ files should be deleted before any commit
that lands Lever B (or any other change).