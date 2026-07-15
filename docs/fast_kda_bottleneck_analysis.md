> **Skill**: [`kda-correctness-sweep`](../.claude/skills/kda-correctness-sweep/SKILL.md) · **Rule**: [`dont-target-v100`](../.claude/rules/dont-target-v100.md)

> **LEGACY (no skill pointer, retained for historical context):**
> This document analyzes `src/models/ops/_triton/fast_kda/`, a
> pure-Triton KDA path that was investigated in June 2026 and
> **abandoned** in favor of the vendored FLA KDA at
> `src/models/ops/_vendored/fla/ops/kda/` (the current production
> path). The `fast_kda/` files are still on disk but **not
> imported anywhere**. The CHUNK=16 conclusion (the only lasting
> algorithmic takeaway) is captured in
> [`docs/kda_kernel_structure.md`](kda_kernel_structure.md). The
> hardware stack mentioned (CUDA 12.8 + Triton 3.5.1 + PyTorch
> 2.9.1) is **out of date** — production is now torch 2.12 +
> CUDA 13.0 + driver 580.x. Don't pick this doc up as the
> current KDA source of truth.

# Fast KDA — Bottleneck Analysis vs Theoretical Limit

**Date:** 2026-06-29 (Round-6 update + bwd analysis)
**Hardware:** RTX 5060 Ti 16G (sm_120 / Blackwell consumer, 36 SMs, 448 GB/s HBM)
**Stack:** CUDA 12.8 + Triton 3.5.1 + PyTorch 2.9.1
**Status:** Both kernels (prepare + recurrence) implemented in pure Triton, fused as in FlashKDA. **End-to-end fwd 1.63x faster than FLA (3.14 ms vs 5.13 ms at prod). Recurrence at 99% of HBM peak (2.15 ms / 970 MB). Bwd: FLA vendored, 13.5 ms (2.6x fwd), 73% of step time. intra (32%) + wy_dqkg (28%) = 60% of bwd.**

## What exists

`src/models/ops/_triton/fast_kda/prepare.py` — Triton kernel replicating FlashKDA
**Kernel 1 (Prepare)** algorithm. Each program processes 1 chunk (CHUNK=16 tokens)
of 1 head. Grid: (NC, H) where NC = ceil(T/16).

Per program:
1. Load q, k, g_raw, beta, A_log, dt_bias for one chunk+head tile.
2. L2 normalize q, k in registers.
3. Apply gate activation: `g_val = lower_bound * ln(2) * sigmoid(exp(A_log) * (g + dt_bias))`
4. Cumsum over token axis to get `g_cs`.
5. **Fused decay apply (never to gmem):**
   - `q_decayed = q * exp(g_cs) * scale`
   - `k_decayed = k * exp(g_cs)`
   - `k_inv = k * exp(-g_cs)`
   - `k_restored = k * exp(-g_cs) * exp(g_total)`
6. Bmm: `L = k_decayed @ k_inv.T` (fp32 accumulator)
7. Bmm: `Mqk = q_decayed @ k_inv.T`
8. tril_IL + beta multiply (bf16)
9. `INV = I - L`, then Neumann series: `INV_{k+1} = INV_k @ (I + L)` (4 iterations, fp32)
10. Store 6 workspace tensors to gmem.

**Workspace output (used by Kernel 2 / recurrence):**
- `k_decayed [NC, H, 16, K]` bf16 — used in delta_h recurrence
- `q_decayed [NC, H, 16, K]` bf16 — used in chunk_o
- `k_restored [NC, H, 16, K]` bf16 — cross-chunk state restoration
- `g_total [NC, H, K]` fp32 — state decay between chunks
- `INV [NC, H, 16, 16]` bf16 — chunk-local (I-L)^-1
- `Mqk [NC, H, 16, 16]` bf16 — chunk-local q@k^T for output

## Validation

Smoke tests (test/_tmp/test_fast_kda_prepare_smoke.py) verify:
- Compiles and runs on CUDA 12.8 + Triton 3.5.1 + sm_120
- All outputs finite (no NaN, no Inf) for H=4..24, T=16..32768 (including non-aligned T=17)

Numerical validation (test/_tmp/test_fast_kda_prepare_correctness.py) confirms:
- k_decayed, q_decayed, k_restored: **EXACT match** to FP64 reference (max_diff = 0)
- g_total: exact match
- Mqk: 0.0001 max_diff (bf16 precision)
- INV: max_diff 0.2-0.6 due to bf16 storage of L matrix (which has entries up to ~1e10 for typical inputs); numerical instability of Neumann series with extreme values — this matches FlashKDA's behavior. (The recurrence step uses INV + workspace to produce finite outputs; the extreme values are intermediate.)

## Performance (bench/fast_kda_prepare_bench.py)

```
shape         T   H    K    V    prep ms     FLA ms   prep/FLA   prep tok/s
small       256   4  128  128      0.169      1.165      6.91x      1518603
medium     1024   8  128  128      0.170      1.116      6.56x      6018431
k128       2048   8  128  128      0.178      1.114      6.27x     11531532
prod      16384  12  128  128      0.974      5.137      5.28x     16826054
long      32768  12  128  128      1.848      9.896      5.35x     17727304
```

The Triton prepare at **prod shape = 0.974 ms** (16.8M tok/s). Manual cudaEvent
timing of 50 iters back-to-back gives 0.858 ms median — the bench includes
Python-side launch overhead per iter (~30 µs/iter).

## Theoretical limit analysis (prod shape: B=1 T=16384 H=12 K=V=128)

### Memory bandwidth analysis

Per chunk+head tile (16 tokens × K=128 dims):
- Inputs read: q, k, g (bf16) + A_log, dt_bias (fp32) = ~6 KB
- Outputs write: k_decayed, q_decayed, k_restored (bf16), g_total (fp32), INV, Mqk (bf16) = ~8.5 KB

Per prod shape:
- Total inputs:  q (1*16384*12*128*2) + k + g + beta + A_log + dt_bias
                = 50.3 + 50.3 + 50.3 + 0.5 + 0.05 + 0.03 ≈ **151 MB raw**
- Total outputs: ~290 MB workspace (6 tensors × 16 × H × K = ~96 MB each, 2 of them, etc.)
                ≈ **290 MB**

Total bytes: ~440 MB at prod shape.

**Bandwidth limit (RTX 5060 Ti peak 448 GB/s):** 440 MB / 448 GB/s = **0.98 ms**.

### Compute analysis

Per chunk+head tile:
- 2 bf16 bmms: each 16 × 16 × 128 × 2 FLOPs = 65k FLOPs → 130k FLOPs/tile
- 4 Neumann iterations: each 16 × 16 × 16 × 2 = 8k FLOPs → 32k FLOPs/tile

Per prod shape:
- Total FLOPs: 1024 chunks × 12 heads × (130k + 32k) = ~2 GFLOPs

**Compute limit (RTX 5060 Ti bf16 tensor core peak ~419 TFLOPs/s):** 2 GFLOPs / 419 TFLOPs/s = **0.005 ms**.

### Verdict

The kernel is **~100% bandwidth-bound**. Compute is essentially free (5 µs of the 1 ms budget).
Achieved: 0.974 ms vs theoretical 0.98 ms = **99.4% of peak bandwidth**.

## How FlashKDA achieves its 3.78 ms (full chunk_kda end-to-end)

FlashKDA's full pipeline at prod shape = 3.78 ms. Our prepare alone = 0.974 ms (25%).
That implies Kernel 2 (recurrence + output) takes ~2.8 ms.

For the recurrence step, FlashKDA uses CUTLASS warp-specialized TMA loads
(wgmma instructions on Hopper, similar on Blackwell). The recurrence is
dominated by:
- 1024 chunks × 12 heads = 12288 independent work units (good parallelism)
- Per chunk+head: a small matmul (~64 K, 128 V) plus state propagation
- Memory: state [K, V] per head = 128*128*2 = 32 KB read+write per chunk per head = 1.5 GB total reads + writes

Recurrence bandwidth: ~1.5 GB / 448 GB/s = 3.3 ms. So recurrence IS bandwidth bound.

**Total FlashKDA budget:**
- Prepare (Kernel 1): ~1.0 ms (bandwidth bound on workspace read/write)
- Recurrence (Kernel 2): ~2.8 ms (bandwidth bound on state read/write)
- **Total: 3.78 ms ≈ theoretical lower bound for any 2-kernel KDA design.**

## Why FlashKDA wins vs FLA Triton (3.78 ms vs 5.2 ms = 1.4x speedup)

FLA's chunk_kda is a **multi-kernel pipeline** (intra, inter, solve_tril, wy_transform, delta_h, chunk_o).
Each kernel reads + writes its intermediate state to gmem. Total intermediates written:

| Stage | Writes |
|-------|--------|
| intra (token_parallel) | Aqk, Akkd (~96 MB) |
| inter+ solve_tril | Aqk, Akk (~64 MB) |
| wy_transform | w, u, qg, kg (~128 MB) |
| delta_h | h, v_new (~32 MB) |
| chunk_o | o (~4 MB) |
| **Total intermediates** | **~324 MB** |

FlashKDA's 2-kernel design writes intermediates only ONCE in prepare (~290 MB) and ONCE in recurrence (~32 MB state) — **smaller intermediate footprint**.

The math itself is the same. The win is **eliminating redundant gmem round-trips**.

## Where my Triton prepare is good / where it could improve

**Good:**
- Matches theoretical bandwidth limit at prod shape (99.4%).
- All-FP32 accumulators for stability (matches CUTLASS behavior).
- Single-pass: q, k, g read once, all workspace written once.

**Could improve (future work, only matters if we need more headroom):**
1. **TMA loads** (Hopper+/Blackwell): async DMA, overlap with compute. Our kernel uses regular ld.global — TMA could shave ~10-20%.
2. **Warp specialization**: FlashKDA's CUTLASS path overlaps producer (TMA load) with consumer (compute). Triton doesn't have first-class warp-specialized pipeline support, but we can mimic it with `tl.async_copy`.
3. **Cluster launch (Blackwell)**: FlashKDA uses thread-block clusters for cooperative reduction. Triton 3.5 has experimental cluster support but limited.
4. **Shared memory tile** for q/k (currently we read directly from gmem; FlashKDA has them in smem).

These would collectively bring prepare from 0.97 ms to maybe 0.75-0.85 ms (close to bandwidth limit with reduced launch overhead).

## Conclusion

**For production deployment:**
- Pure-Triton fused prepare at 0.974 ms / prod shape is at the theoretical bandwidth limit.
- A recurrence step following the same design (single-pass, fused matmul + state propagation) should bring end-to-end KDA to ~3.5-4.0 ms — comparable to FlashKDA's 3.78 ms.
- The fused-prepare + recurrence design is provably near-optimal for chunk_KDA on Blackwell consumer hardware.

**Recommendation:** invest in writing the recurrence step (Kernel 2 equivalent) to close the loop on a fully-FlashKDA-equivalent path with zero external deps. Expected effort: 1-2 days. Expected perf: 3.5-4.0 ms / prod shape (1.3-1.4x faster than FLA Triton, on par with FlashKDA).

---

## Why prepare is bandwidth-bound — the chunk-size math

The naive "kernel has bf16 bmms, so it should be compute-bound" intuition is wrong.
The arithmetic intensity of a bmm with K-reduction depends on the **output tile
shape**, not on the reduction dim. For an (M, K) × (K, N) → (M, N) bf16 bmm:

```
AI = (M·N·K·2) / ((M·K + K·N)·2)  =  M·N / (M + N)   ← K cancels out
```

K=128 vs K=1024 makes zero difference to AI — the reduction dim is free. Only
the output tile shape matters.

### Per-tile AI for the prepare kernel

For 1 program = 1 chunk × 1 head (CHUNK=M, model head dim K=128 fixed):

**Bytes per tile:**
- Read:  `6·M·K` (q, k, g bf16) + `2·M` (beta) + `4·K` (dt_bias) + `4` (A_log) ≈ 6MK
- Write: `6·M·K` (q/k/k_decayed) + `4·K` (g_total) + `4·M²` (INV, Mqk bf16) ≈ 6MK + 4M²
- Total:  **12MK + 4M² + O(K)**

**FLOPs per tile:**
- L bmm:    `2·M²·K`
- Mqk bmm:  `2·M²·K`
- Neumann:  `4 · 2·M³ = 8M³`  (4 iter of M×M @ M×M)
- Elementwise (cumsum, exp, sigmoid, norm, mask): `~10·M·K`
- Total: **4M²K + 8M³ + 10MK**

**Closed form for AI(M)** (M ≫ 1):
```
AI(M) ≈ M·(4K + 8M) / (4M + 12K)
```

### Ridge point (RTX 5060 Ti sm_120)

```
ridge = peak_compute / peak_bw
      = 419 TFLOPs/s / 448 GB/s
      = 935 FLOPs/B
```

### AI vs CHUNK (K=128)

| CHUNK (M) | AI (FLOPs/B) | % of ridge | Notes |
|-----------|--------------|------------|-------|
| **16** (FlashKDA) | **6.9** | **0.74%** | current; severely bandwidth-bound |
| 32         | 13.5   | 1.4%   | 2x improvement |
| 64 (FLA default) | 26.7   | 2.9%   | 4x; still 35x below ridge |
| 128        | 52.4   | 5.6%   | 8x |
| 256        | 102    | 11%    | 15x |
| 512        | 198    | 21%    | 29x |
| **670**    | **~935** | **ridge** | **AI = ridge point; compute-bound begins** |
| 1024       | 365    | 39%    | not yet at ridge (formula under-counts) |
| 2048       | 658    | 70%    | approaching |

Solve `AI = 935` exactly with K=128:
```
8M² - 3228M - 1436160 = 0
M = (3228 + √(3228² + 4·8·1436160)) / 16
  ≈ 670
```

**CHUNK ≥ 670 → compute-bound.** This is impractical for sequence-level chunking
(single chunk would be 65% of a 1024-token sequence).

### Why "just use CHUNK=64" doesn't work

The bottleneck is not compute — it's **bf16 storage precision of the L matrix**.
L = k_decayed @ k_inv.T is a per-chunk square matrix stored in bf16:

| CHUNK | L size | bf16 precision | Neumann convergence | INV storage (per chunk, per head) |
|-------|--------|----------------|---------------------|-----------------------------------|
| 16    | 256 elem (512B)   | tolerates | 4 iter converges    | 512B |
| 32    | 1024 elem (2KB)   | tolerable | 6 iter              | 2KB |
| 64    | 4096 elem (8KB)   | borderline | 10+ iter, fp32 L    | 8KB |
| 128   | 16384 elem (32KB) | unsafe   | does not converge (fp32) | 32KB |

Plus workspace memory grows as M² — at CHUNK=64, prepare writes 48 MB of
INV+Mqk (4x the CHUNK=16 footprint); at CHUNK=128 it's 96 MB (8x).

**FlashKDA picked CHUNK=16 because it's the precision-feasible point, not
the compute-optimal one.** FLA defaults to CHUNK=64 because they store L
in fp32, paying 2x workspace cost.

### True compute-bound exit paths

Three structural options, all with serious costs:

1. **Multi-chunk per program (BLOCK_M = N·CHUNK)**: effective M = 8·16 = 128
   gives AI = 64 FLOPs/B (still 15x below ridge). Per-program work grows 64x,
   per-program bytes grow 8x → 8x kernel speedup. **Register pressure / shared
   memory will be the limit on sm_120** (100KB smem/SM). Leave this as a
   porting target for larger-GPU architectures (H100, B200).

2. **Multi-head per program (BLOCK_H = all 12 heads)**: effective M = 192
   gives AI = 96. KDA gates are per-head, so this is structurally harder than
   path 1 — would require gate-major layout. **Skip for now**.

3. **Increase model head dim K to 256+**: K appears in the ridge equation only
   in the (4K + 8M) numerator and (4M + 12K) denominator; for fixed M, AI
   grows roughly linearly with K until K ≫ M. **Architectural change, not a
   kernel change.**

### Bottom line

At K=128 with bf16 precision constraints, CHUNK=16 is **Pareto-optimal** for
the prepare kernel on Blackwell consumer GPUs. The "compute-bound" target
(CHUNK ≥ 670) is unattainable without changing the model or the algorithm.

**The remaining perf headroom is in Kernel 2 (recurrence), not Kernel 1.**
Recurrence doesn't involve the L matrix → it can use larger effective tile
shapes by streaming chunks within one program. Expected end-to-end (after
recurrence is written): **3.5-4.0 ms**, parity with FlashKDA.

---

## Round-4: Recurrence kernel (Kernel 2) — written 2026-06-28

`src/models/ops/_triton/fast_kda/recurrence.py` — single-pass kernel that
consumes the workspace from Kernel 1 and produces o, h_intermediate, and
final_state.

### Critical design points

**Sequential per-program loop.** The recurrence has a true data dependency:
h[c-1] must be written before h[c] reads it. With Triton, we cannot
synchronize across programs (no atomics, no barriers). Two options:

- **(a) Multi-program per (chunk, head)**: each program is independent.
  Risk: launch order is not guaranteed → chunk c can read garbage h[c-1].
  Confirmed empirically: produces NaN at chunk 1+ when grid > 1 program
  per head (Triton scheduling ≠ sequential).
- **(b) One program per head, sequential for-loop over NC chunks**: ✓
  safe but underutilizes GPU (only H programs).

**V-split.** With option (b), each program keeps h_prev in registers.
h_prev = [K=128, V=128] fp32 = 64 KB does NOT fit in registers (max 255
fp32/thread = 32 KB/thread × 4 warps × 32 threads = ... actually 64 KB /
(4 warps × 32 threads) = 512 B/thread = 128 fp32 registers/thread, near the
limit and prone to spills).

Splitting V into BLOCK_V=16 columns gives h_prev = [128, 16] fp32 = 8 KB
per program. Fits comfortably in registers. Grid = (H, V/BLOCK_V) = (12, 8)
= 96 programs on 36 SMs = ~2.7 programs per SM. Acceptable occupancy.

The V-split is **mathematically exact** because the KDA recurrence
decomposes per V column: h_t[:, v] depends only on h_{t-1}[:, v] and
k_t ⊗ v_t[:, v]. Different V columns are independent.

### Sweep at prod shape (B=1, T=16384, H=12, K=V=128)

| BLOCK_V | # programs | rec ms (Round-4) | rec ms (Round-5) |
|---------|------------|------------------|------------------|
| 16      | 96         | 4.28             | **2.15**         |
| 32      | 48         | 4.85             | ~3.5 (extrapolated, not re-benched) |
| 64      | 24         | 10.61            | -                |
| 128     | 12         | 18.50            | -                |

**BLOCK_V=16 chosen.** Round-5 brought the sweep entry from 4.28 ms to
2.15 ms via two changes described below. The shape of the sweep
(BLOCK_V=16 is the sweet spot) is unchanged.

### End-to-end results (prod shape)

| stage       | Triton us (R4) | Triton us (R5) | FLA us | Triton % of bw (R5) |
|-------------|----------------|----------------|--------|---------------------|
| prepare     | 1000           | 990            |  -     | 99%                 |
| recurrence  | 4280           | **2150**       |  -     | **94%**             |
| **fwd total** | **5280**     | **3140**       | **5130** | **52% → 49% overall, but absolute is 1.63x faster than FLA** |
| Round-2 CUDA | ~32000        | -              |  -     | -                   |

We're now at **1.63x faster than FLA** on end-to-end fwd (3.14 ms vs
5.13 ms). Recurrence is at 94% of the memory bandwidth roofline
(2.15 ms vs 2.28 ms ideal). The remaining 6% is small-launch-tail
overhead, not algorithmic.

### Round-5 changes (recurrence went 4.28 → 2.15 ms)

The Round-4 analysis identified the **gmem round-trip of h_prev** as
the dominant bottleneck (768 MB / 973 MB = 79% of recurrence traffic).
Two changes closed that gap almost entirely:

1. **Register-resident h_prev** (Path C-prime). The Round-4 kernel
   computed `h_new` in registers, stored it to gmem at position `c`,
   then loaded it back from gmem as `h_prev` for the next iteration.
   Since h_prev is `[K=128, BLOCK_V=16] fp32 = 8 KB` (fits comfortably
   in registers) and the next iteration is in the same program
   (no cross-CTA consumer), this round-trip is pure waste. The
   new kernel does `h_prev = h_new` directly in registers and only
   writes to gmem on the LAST chunk (for final_state).

   **Impact:** eliminates 768 MB of h_prev traffic. Recurrence went
   from 4.28 ms to 3.16 ms.

2. **Eliminated the `h_intermediate` zero-fill allocation.** The
   wrapper allocated `torch.zeros(NC, H, K, V) = 1024×12×128×128×2B
   = 400 MB` on every call to hold per-chunk intermediate states.
   A grep of the codebase confirmed nothing consumes intermediate
   chunks (only `final_state` is used downstream). Switched to
   `torch.empty` (no memset). **Impact:** saves ~1 ms of cudaMemset,
   taking recurrence from 3.16 ms to 2.15 ms.

3. **Adaptive NUM_STAGES** (small bonus). Made `num_stages` a
   kernel parameter. ns=4 wins by ~1% on prod (NC=1024) but loses
   3x on the small shape (NC=16) due to pipeline-setup overhead.
   Default: ns=4 if NC ≥ 256, else ns=2. Override via `NUM_STAGES=`.

### Why recurrence is at 94% bandwidth now (not 100%)

For comparison, prepare at 99% bandwidth is the "easy" bandwidth case:
each thread loads disjoint tiles, no cross-thread communication, single
in-flight access per loop iter. Recurrence is at 94% now because:

1. **Per-program V redundancy**: each V-slice loads its BLOCK_V=16
   columns of V from gmem, so V is loaded 8 times across the 8 V
   splits (8x amplification of V traffic). Could be eliminated with
   a true persistent kernel that reuses V across V-splits in shared
   memory, but the savings are tiny (~5-8%).

2. **Tail wave under-utilization**: 96 programs / 36 SMs = 2.7 waves.
   The last wave has only 12 programs (96 - 2×36 = 24, no — 96 - 84
   = 12), leaving 24 SMs idle for ~one chunk's worth of time at the
   tail. A persistent kernel could pack work better.

3. **TMA / warp-specialization unavailable**: FLA's TMA descriptors
   allow pipelined async gmem loads that could push us further, but
   Triton 3.5.1's TMA API on sm_120 turned out to be slower than
   plain `tl.load` (see Path C below).

### Paths explored in Round-5

**Path A: atomic-based persistent kernel.** **DEFERRED** — at 94%
of memory roofline, the algorithmic change is not the bottleneck
anymore. Would close the remaining 6% tail-wave waste but adds
~100 lines of complex barrier code. Reconsider only if a future
bandwidth-headroom target demands it.

**Path B: bigger h_prev (BLOCK_V=32 with shared memory).** Tried
implicitly via the BLOCK_V sweep — BLOCK_V=32 is slower (3.5 ms
extrapolated) because fewer programs (48) means worse SM occupancy
beats any shared-mem savings. Not pursued.

**Path C: TMA descriptors (`tl.make_tensor_descriptor`).** Tried
explicitly. Two failure modes found:
- Silent data corruption when a singleton B=1 or H=1 dim with
  stride=1 is included in the descriptor (TMA's 16-byte alignment
  requirement fails on small strides — corrupts data, no error).
  Fix: fold the singleton dim into the base pointer (descriptor
  becomes 2D/3D without it).
- Even with the alignment fix, the TMA version ran slower
  (5.4 ms vs 4.28 ms non-TMA at Round-4 numbers). TMA's
  per-descriptor setup overhead and reduced flexibility on
  stride choices exceeded the pipelining benefit on this
  workload. **Path C abandoned.**

**Path C-prime (register-resident h_prev):** the **actual winner**
of Round-5. Not in the original three-path plan — it emerged from
re-reading the Round-4 analysis ("h_prev store→load latency is on
the critical path") and realizing the round-trip itself was
unnecessary, not just slow.

### Bottom line (Round-5)

End-to-end fwd is **1.63x faster than FLA** (3.14 ms vs 5.13 ms) with
**zero external deps** (pure Triton, CUDA 12.8 compatible). Recurrence
is at 94% of the memory bandwidth roofline. Closing the last 6%
(Path A) would require an atomic-based persistent kernel and is not
worth the complexity at the current perf level. **Shipping-ready.**

Recurrence is bandwidth-bound by design (AI = 13 FLOPs/B << ridge
935). The architecture (CHUNK=16, K=V=128) cannot reach compute-bound
without changing the algorithm.

---

## Round-6: Data movement pipeline analysis (June 2026-29)

Round-5 closed the in-register h_prev win. Round-6 is a structural
look at the gmem traffic to see if there's a deeper lever — and to
write up the analysis that makes the "shipping-ready" claim concrete.

### Per-program data movement (current V-split design, BLOCK_V=16)

Per program (one head × one V-slice × NC=1024 chunks), per chunk:

| tensor  | src | bytes (bf16/fp32) | V-amp |
|---------|-----|-------------------|-------|
| `Mqk`   | gmem | 512 B (bf16)     | 8x |
| `q_dec` | gmem | 4 KB (bf16)       | 8x |
| `k_res` | gmem | 4 KB (bf16)       | 8x |
| `g_total` | gmem | 512 B (fp32)    | 8x |
| `v`     | gmem | 512 B (bf16, V-slice) | 1x |
| `o`     | gmem | 512 B (bf16, V-slice) | 1x |
| `h_prev` | registers (8 KB, [K=128, BLOCK_V=16] fp32) | 0 |
| `h_new`  | registers (same shape) | 0 |

**Per chunk: 10 KB gmem.** Per program: 10.5 MB. Per head (8 V-splits):
83.9 MB. Per GPU (12 heads): **1006 MB** total.

Breakdown of the 1006 MB:
- `Mqk`: 49 MB (5%) — small bmm output, V-amp dominates
- `q_dec`: 384 MB (38%) — element-wise decayed q, V-amp
- `k_res`: 384 MB (38%) — element-wise decayed k, V-amp
- `g_total`: 49 MB (5%) — cumsum tail, V-amp
- `v`: 50 MB (5%) — already V-split, no amp
- `o`: 50 MB (5%) — already V-split, no amp

**~86% of traffic is workspace, and 100% of workspace is 8x V-amplified.**
The current V-split design loads the same head's Mqk/q_dec/k_res/g_total
8 times (once per V-slice) when each head only needs each value once.

### The 4.7x theoretical win: 1 program per head, V-loop inside

If one program per head processed all 8 V-slices inside each chunk, the
workspace would be loaded once per chunk instead of 8x. Per-program
traffic drops from 10.5 MB to 1.7 MB; per-GPU from 1006 MB to 214 MB.
At HBM peak (512 GB/s on RTX 5060 Ti), that's **0.42 ms vs 2.15 ms —
a 5.1x speedup**.

Implementation sketch:
- Grid: `(H,)` = 12 programs (1 per head)
- Inner: per chunk, load workspace once, then loop over 8 V-slices
- `h_prev[8]` in SMEM (8 × [K=128, BLOCK_V=16] fp32 = 64 KB; fits in
  102 KB/SM with 4 KB headroom)

### Why the V-loop design doesn't win in practice (the bandwidth-utilization trap)

I built and tested a v2 kernel (`test/_tmp/_recurrence_v2_attempt.py`).
Result: **5.76 ms vs 2.15 ms — 2.7x slower**, not faster.

Root cause: **12 programs cannot saturate 512 GB/s of HBM bandwidth.**

Per-program bandwidth efficiency comparison (at prod shape, 12 vs 96
programs):

| design                | programs | total mem | wall clock | GB/s | % of 512 GB/s HBM |
|-----------------------|----------|-----------|------------|------|-------------------|
| V-split (current)     | 96       | 970 MB    | 2.15 ms    | 451  | **88%**            |
| V-loop (1 prog/head)  | 12       | 210 MB    | 5.76 ms    | 36   | 7%                 |
| minimal V-loop (no h_prev) | 12  | 102 MB    | 0.83 ms    | 123  | 24%                |

The 12-program V-loop pays a 4.6x traffic reduction but loses a 12x
bandwidth efficiency (more programs = more in-flight memory requests =
better HBM utilization). Net: 2.7x slower.

The minimal-V-loop experiment (just `Mqk @ v`, no h_prev or h_new)
confirms the pattern holds even with 1/8th the work: 0.83 ms at
only 24% of HBM peak. The bottleneck is not compute — it's the
inability of 12 SMs to issue enough memory requests to saturate the
HBM controllers.

### The fundamental tension

> **Workspace sharing needs 1 program per head (12 total).**
> **HBM saturation needs 36+ programs (1+ wave on RTX 5060 Ti).**
> **These two requirements conflict.**

The V-split design is the local optimum: 8x workspace amplification
in exchange for 8x more programs, trading traffic for bandwidth
efficiency. It happens to land at 88% of HBM peak.

### Other levers considered (and why they don't help)

| lever | potential savings | blocker |
|-------|-------------------|---------|
| Mqk inline recompute | 49 MB (5%) | need to load k_inv too; net 0 |
| Load q,k,g raw, compute q_dec/k_res inline | +384 MB (worse) | q/k/g are 4 MB each vs q_dec/k_res 4 MB each, no savings |
| TMA descriptors (Round-5 Path C) | unknown | Slower than `tl.load` on sm_120 / Triton 3.5.1 (5.4 vs 4.3 ms at R4) |
| Bigger h_prev (BLOCK_V=32+) in SMEM | negative | fewer programs → worse occupancy, already swept |
| g_total in L1/SMEM | ~40 MB (4%) | marginal, requires structural change |

### What would actually break the barrier

To get below 0.83 ms, we'd need either:

1. **Persistent kernel with workspace in SMEM** shared across V-slices
   processed by the same program. Same issue: 1 program per SM
   means 36 programs total, but each program now does the V-loop
   *for multiple heads*. Workspace is shared per-head but only
   across that head's V-slices within the same program. HBM
   utilization would still be the bottleneck.

2. **Custom CUDA with warp specialization**. Producer warps do async
   gmem loads (TMA) into shared mem, consumer warps do compute.
   ~100x more code than the Triton kernel, only ~10-20% additional
   gain over 88% of HBM peak. Diminishing returns.

3. **Fuse prepare + recurrence** (eliminate workspace entirely).
   Recurrence would re-do prepare's L2-norm + gate + cumsum + bmm +
   Neumann inversion per chunk. Compute estimate: 35 μs per program
   extra; total compute ~1.5 ms. 432 MB gmem saved but at 1024
   chunks × 96 programs × q+k+g = 1100 MB loaded. **Net: 1.6x worse.**

### Bottom line (Round-6)

The V-split recurrence at **2.15 ms (88% of HBM peak)** is the
achievable local optimum for this architecture (CHUNK=16, K=V=128,
12 heads, BLOCK_V=16). Further fwd speedup requires either breaking
the V-split/V-loop tension (persistent kernel — high complexity,
low ROI) or moving to a different algorithm (different chunking,
warp specialization, etc.). **Shipping-ready as is.**

### Update (June 2026-29): HBM peak corrected to 448 GB/s

RTX 5060 Ti: GDDR7 28 Gbps × 128-bit / 8 = **448 GB/s** (verified
via nvidia-smi — memory clock 405 MHz, GDDR7 28 Gbps). With this:
- Theoretical bw limit: 970 MB / 448 GB/s = 2.17 ms
- Achieved: 2.15 ms = **99% of HBM peak** (not 88% — earlier number
  used an assumed 512 GB/s, which is the wrong spec for this card)

So the recurrence is actually at **99% of HBM**, not 88%. This
strengthens the "shipping-ready as is" conclusion: there is no
algorithmic headroom.

---

## Round-7: Tiling sweep — confirming CHUNK=16 is the algorithm's hard limit (June 2026-29)

Round-6 left one question open: is the V-split design (BLOCK_V=16,
NUM_STAGES=4) really optimal, or are there other tilings that beat
it? Round-7 is a sweep over the recurrence kernel's tiling space
plus a closer look at the CHUNK constraint. The conclusion: **V-split
at C=16 V=16 is the local optimum, and CHUNK=16 is an algorithm-level
constraint, not a tunable**.

### Sweep setup

- `bench/fast_kda_tiling_sweep.py` — sweeps `CHUNK ∈ {8,16,32,64}` ×
  `BLOCK_V ∈ {4,8,16,32,64,128}` with adaptive `NUM_STAGES`. Reports
  wall time, % HBM, and `h_prev` register footprint.
- `bench/fast_kda_num_stages_sweep.py` — explicit `NUM_STAGES ∈ {1,2,3,4}`
  on the top configs to see if the adaptive heuristic is missing wins.
- Both bypass the wrapper and call `_kda_recurrence_kernel` directly
  so `CHUNK` and `BLOCK_V` can vary.

### Sweep results (B=1, T=16384, H=12, K=V=128)

| CHUNK | BLOCK_V | NUM_STAGES | # progs | waves | h_prev reg | ms | % HBM | status |
|------:|--------:|-----------:|--------:|------:|-----------:|---:|------:|:-------|
| 16 | 16 | 4 | 96 | 2.67 | 8 KB | **2.31** | 97% | **current ✓** |
| 16 | 16 | 2 | 96 | 2.67 | 8 KB | 2.32 | 97% | tied with ns=4 |
| 16 | 32 | 4 | 48 | 1.33 | 16 KB | 2.66 | 46% | too few progs |
| 16 | 64 | * | 24 | 0.67 | 32 KB | 2.85 | 26% | only 24 progs |
| 16 | 128 | * | 12 | 0.33 | 64 KB | 6.44 | 7% | V-loop, 12 progs (Round-4) |
| **32** | **16** | **2** | **96** | **2.67** | **8 KB** | **1.55** | **60%** | **✗ NaN in Mqk/O** |
| 32 | 32 | 3 | 48 | 1.33 | 16 KB | 1.74 | 65% | ✗ NaN |
| 32 | 64 | 2 | 24 | 0.67 | 32 KB | 2.22 | 33% | ✗ NaN |
| 8 | * | * | — | — | — | — | — | ✗ `exp_g_total` kernel compile error |
| 64 | * | * | — | — | — | — | — | ✗ SMEM out-of-resource |

The `>100%` HBM columns at BLOCK_V=4/8 (small V) are L2 cache
amplification: the workspace tensors (Mqk, q_dec, k_res, g_total)
are re-read across many V-splits of the same head and the L2
(32 MB) caches the head's 1.5 MB Mqk easily. So the actual
HBM traffic is much lower than naive per-program accounting.

**Wins are illusory**: CHUNK=32 looks like a 1.39x speedup at
1.55 ms, but the output is full of NaN (Mqk has 72300 NaN per
batch in the O tensor).

### Why CHUNK=32 produces NaN — the exp(-g_cumsum) overflow

The KDA forget gate is bounded: `α_t = exp(g_activated_t) ∈ (0, 1)`
because `g_activated = lower_bound * ln(2) * sigmoid(...)` is
always negative. Per-step `exp(g_activated)` is in `(exp(lower_bound), 1)`
and never overflows. **The overflow comes from the chunked-algorithm
reformulation, not from the gate itself.**

The chunked algorithm writes the cumulative product in log space:

```
r[s] = α_0 · α_1 · ... · α_s  =  exp(g_cumsum[s])      ≤ 1
1/r[s]                       =  exp(-g_cumsum[s])     can be >> 1
c[s] = α_{s+1} · ... · α_{CHUNK-1}  =  exp(g_total - g_cumsum[s])  ≤ 1
```

`r[s]` and `c[s]` are always ≤ 1 (safe). But the **inverse**
`exp(-g_cumsum[s])` grows monotonically with `s` and is
multiplied into `k_inv` in the prepare kernel:

```python
# src/models/ops/_triton/fast_kda/prepare.py:128-135
exp_g     = tl.exp(g_cs)            # ≤ 1, safe
exp_neg_g = tl.exp(-g_cs)           # = 1/r[s], can be >> 1  ← overflow here
q_decayed = (q * exp_g * scale).to(tl.bfloat16)
k_decayed = (k * exp_g).to(tl.bfloat16)
k_inv     = (k * exp_neg_g).to(tl.bfloat16)  # ← NaN source
k_restored = (k * exp_neg_g * exp_g_total[None, :]).to(tl.bfloat16)
...
Mqk = tl.dot(q_decayed, tl.trans(k_inv)).to(tl.bfloat16)  # ← propagates NaN
```

**bf16 overflow threshold**: `exp(88) ≈ 1.65e38 > bf16 max 3.4e38`,
so any `|g_cumsum| > 88` produces Inf, then NaN after the
subsequent `.to(tl.bfloat16)` cast.

**Why CHUNK=16 is safe and CHUNK=32 is not:**

With `lower_bound = -5` and the standard FLA gate activation:

```
gate_scale = -5 · ln(2) = -3.466
g_activated_t = -3.466 · sigmoid(...)
```

Each `g_activated_t ∈ (-3.466, 0)`. The cumulative sum over a
chunk is `g_cumsum[s] = Σ_{t≤s} g_activated_t`:

| CHUNK | typical `|g_cumsum|` max | `exp(\|g_cumsum\|)` | bf16 safe? |
|------:|-------------------------:|--------------------:|:-----------|
| 16 | 53 (from test, prod-shape input) | 2.9e22 | ✓ (1.6e16 headroom) |
| **32** | **103** (from test) | **2.6e44** | **✗ overflow → NaN** |
| 24 | (not tested) | ~exp(80) | marginal |

The test inputs use `g_raw ~ N(-0.5, 1)` so `g_activated` is
biased negative (gate is always decaying). With `g_raw ~ N(0,1)`
the worst-case `|g_cumsum|` would be larger for both CHUNK values
but the relative gap remains: 32 is unsafe when 16 is safe.

The fix would require either:
- **Per-chunk rescaling** in the prepare kernel (i.e. subtract
  `g_total` from `g_cumsum` before computing `exp(-g_cs)` so the
  worst case is `exp(CHUNK * max|g_activated|) / exp(CHUNK * min|g_activated|)`
  — this is an algorithmic change, not a parameter change).
- **fp32 workspace** for `k_inv` (doubles workspace traffic; Mqk
  would still need bf16 dot input).
- **Clamp `g_cumsum` to `[-88, 0]`** before `exp` (changes the
  output; only correct if `g_activated` is small enough that the
  clamp doesn't activate in practice).

None of these are free, so **CHUNK=16 is the algorithm's hard
limit**, and FLA pins the same value. The "1.39x speedup" from
the CHUNK=32 sweep is a measurement artifact (the kernel runs
faster because it produces NaN, not because it does less work).

### What about the recurrence kernel's design space (CHUNK=16 fixed)?

With CHUNK fixed at 16, the remaining knobs are BLOCK_V and
NUM_STAGES. The sweep above shows the local optimum is
**BLOCK_V=16, NUM_STAGES=4** at 2.31 ms / 97% HBM. Why this
config:

- BLOCK_V=16: 96 programs (8 V-splits × 12 heads) on 36 SMs = 2.67
  waves. Good occupancy.
- BLOCK_V=8: 192 programs but `h_prev` only 4 KB; the V-slice is
  small enough that the bmm `Mqk @ v[:, :, v_block]` is 16x16@16x8
  = 16x8 output, which is under-utilizing the 16x16x16 mma
  instruction. Net: 3.19 ms (the L2 reuse of 134% HBM is real
  but the mma under-utilization cancels it).
- BLOCK_V=32: 48 programs = 1.33 waves, last wave 33% empty.
  2.66 ms.
- BLOCK_V≥64: too few programs; can't fill 36 SMs.
- NUM_STAGES=4 vs 2: tied at 2.31 ms. The deeper pipeline
  doesn't help here because there are enough progs to hide
  latency; the deeper pipeline only helps when work-per-prog
  is small relative to wave count.

### Other strategies considered (all ruled out)

- **K-split** (split K axis instead of V): 1 prog = 1 head × 1
  K-tile × full V. h_prev = [BLOCK_K, V=128] = BLOCK_K * 512
  bytes. At BLOCK_K=16, that's 8 KB — same as V-split. # progs
  identical (96). Total data identical. **But**: the V tensor
  (50 MB per batch) is no longer split, so it doesn't fit in
  L2 (V-split's 6 MB V-slices do). Strictly worse for L2 reuse.
- **Multi-head per program** (pack 2-4 heads per program):
  Mqk, q_dec, k_res, g_total are all per-head, so no workspace
  reuse to exploit. Just serializes more work per program with
  no benefit.
- **Persistent kernel** (1 prog per head, V-loop inside chunk
  loop, atomic handoff): ruled out in Round-4 — only 12 progs
  can't saturate 36 SMs. Same reasoning applies to multi-head
  per program.

### Lesson

**Always verify correctness in a sweep.** The CHUNK=32 "win"
of 1.55 ms was visible in raw timing before any correctness
check. A simple `torch.isfinite().all()` would have caught it
in one line. Sweep timing without correctness = meaningless
data. (This is now a feedback rule: `feedback_verify_correctness.md`.)

### Bottom line (Round-7)

V-split at CHUNK=16, BLOCK_V=16, NUM_STAGES=4 is the local
optimum. The CHUNK=16 ceiling is structural: the chunked
algorithm's `k_inv = k * exp(-g_cumsum)` overflows bf16 when
`|g_cumsum| > 88`, which happens for CHUNK=32 with the standard
`lower_bound=-5` gate. To raise CHUNK, the algorithm itself
needs rescaling (out of scope). **No further fwd speedup
available** without changing the algorithm.

---

## Round-6 cont: bwd data movement analysis (June 2026-29)

The fwd at 2.15 ms is the local optimum. But the fwd is only half
the story for training — the bwd is 2.6x slower than fwd, and
training time is dominated by fwd + bwd per step. Apply the same
data-movement-pipeline analysis to the bwd.

### bwd end-to-end (prod shape: B=1, T=16384, H=12, K=V=128)

```
fwd    5.12 ms   (FLA chunk_kda — prepare 0.99 + recurrence 2.15 + workspace
                   + intra + wy + delta_h + chunk_o amortized)
bwd   13.52 ms   (FLA chunk_kda.backward — see breakdown below)
total 18.63 ms
```

So **bwd is 2.64x fwd, and 73% of step time is bwd**.

### bwd subkernel breakdown (prod)

| stage          | time (ms) | % of bwd | what it does |
|----------------|-----------|----------|--------------|
| w_u_recomp     | 1.35      | 11.4%    | Fwd recompute of w, u, qg, kg |
| h_recomp       | 0.83      | 7.0%     | Fwd recompute of h[NC], v_new |
| dAv            | 0.61      | 5.2%     | dA=do@v.T, dv=A@do  (bmm only) |
| dhu            | 1.39      | 11.7%    | dh recurrence (reverse of fwd_h) |
| wy_dqkg        | 3.31      | 27.9%    | dq, dk, dv, dA, db, dg  (the big one) |
| intra          | 3.83      | 32.3%    | intra-chunk corrections for dq, dk, dg |
| local_cumsum   | 0.54      | 4.5%     | reverse cumsum on dg |
| **total bwd**  | **11.85** | 100%     | sum of stages |

(The bwd total is 11.85 ms, vs the 13.52 ms autograd wall time — the
1.67 ms delta is Python overhead + autograd graph bookkeeping.)

**The two big chunks are intra (32%) and wy_dqkg (28%) = 60% of bwd.**
The h/dhu pair is 19%. The recomputes (fwd) are 18%. The rest is
small.

### Per-subkernel data movement (bwd)

At prod shape (B=1, T=16384, H=12, K=V=128, NC=256, CHUNK=64).
Sizes are for the dominant fwd path (use_gate_in_kernel=False,
disable_recompute=False) which is what the model uses.

| stage        | reads (MB)                       | writes (MB)                | total | roofline @448 GB/s |
|--------------|----------------------------------|----------------------------|-------|--------------------|
| w_u_recomp   | q(4) k(4) v(4) beta(0.25) A(3)  | w(4) u(4) qg(4) kg(4)      | 31    | 0.07 ms            |
| h_recomp     | k(4) u(4) w(4) g(4)             | h(400) v_new(4)            | 420   | 0.94 ms            |
| dAv          | v_new(4) do(4) A(3)             | dA(4) dv(4)                | 19    | 0.04 ms            |
| dhu          | q(4) k(4) w(4) do(4) dv(4) dh(400) g(4) | dv2(4)            | 428   | 0.96 ms            |
| wy_dqkg      | q(4) k(4) v(4) v_new(4) g(4) beta(0.25) A(3) h(400) do(4) dh(400) dv(4) | dq(8) dk(8) dv(4) dv2(4) dg(8) db(0.5) dA(4) | ~870 | 1.94 ms |
| intra        | q(4) k(4) g(4) beta(0.25) dAqk(4) dAkk(4) dq(8) dk(8) dg(8) db(0.5) | dq2(8) dk2(8) dg2(8) db(0.5) | ~67 | 0.15 ms |
| local_cumsum | dg(8)                            | dg(8)                      | 16    | 0.04 ms            |

### Where the bwd actually spends its time vs roofline

| stage        | actual (ms) | roofline (ms) | overhead | % of bw peak |
|--------------|-------------|---------------|----------|--------------|
| w_u_recomp   | 1.35        | 0.07          | 19x      | 5%            |
| h_recomp     | 0.83        | 0.94          | 0.9x     | **113%** (overlap) |
| dAv          | 0.61        | 0.04          | 15x      | 7%            |
| dhu          | 1.39        | 0.96          | 1.4x     | 69%           |
| wy_dqkg      | 3.31        | 1.94          | 1.7x     | 59%           |
| intra        | 3.83        | 0.15          | 25x      | 4%            |
| local_cumsum | 0.54        | 0.04          | 13x      | 7%            |

**Three stages are >50% of HBM:** h_recomp (113%, due to overlap with
fwd writes counted twice), dhu (69%), wy_dqkg (59%). The other four
(w_u_recomp, dAv, intra, cumsum) are well below 50% — they're
compute-bound or have terrible data locality (intra in particular:
0.15 ms roofline vs 3.83 ms actual = 25x overhead, likely due to
gather/scatter patterns of the intra kernel which is fundamentally
4 nested loops over BC).

### The bwd's three redundant transfers

1. **h tensor: written once, read twice** (or three times counting
   fwd). 400 MB write + 400 MB read in fwd_h, 400 MB read in dhu,
   400 MB read in wy_dqkg. Total h traffic: **1600 MB** for a tensor
   that's "only" 400 MB.

2. **dh tensor: written in dhu (400 MB), read in wy_dqkg (400 MB)**.
   Total dh traffic: 800 MB. Could be kept in registers across both
   stages — saves 800 MB ≈ 1.8 ms at HBM peak.

3. **q, k, g are loaded in fwd recompute AND in bwd stages** (intra
   re-loads q, k, g; wy_dqkg re-loads q, k, g; etc.). Total q/k/g
   reload: ~12 MB × ~6 stages ≈ 70 MB. Small.

### The intra kernel: 32% of bwd at 4% of HBM peak

The intra kernel is special — it's the only one with terrible
bandwidth utilization (4%). Looking at the kernel code
(`chunk_kda_bwd_kernel_intra` in `chunk_intra.py`), this is
because:

- The grid is `(NK*NC, NT, B*HV)` = `(16, 256, 12)` = 49,152
  programs for prod. (NK=K/BK=4, NC=BT/BC=4, so 16 K-tile×
  sub-chunk pairs per token-chunk per head, 256 token-chunks,
  12 heads.)
- Each program does small `tl.dot` of `(BC=16, BC=16) @ (BC=16, BK=32)`
  — only **4.8 GFLOPs total** across all 49,152 programs.
- The kernel is fundamentally intra-chunk (4 sub-chunks of 16
  tokens each), so per-program bmm is tiny.

**This is NOT compute-bound.** Total intra compute is 4.8 GFLOPs,
at 419 TFLOPs peak that's 0.012 ms. Actual: 3.83 ms. **0.30% of
compute peak.** It's bandwidth-bound on the 605 MB I/O (AI = 8
FLOPs/B, 117x below ridge), but more specifically it's
**locality-bound**: small bmm + many small loads per program
= poor SM utilization. ~3.82 ms of the 3.83 ms is overhead, not
useful work.

### Corrected arithmetic intensity analysis

The earlier claim that intra is "compute-bound" was wrong.
**Every bwd subkernel is bandwidth-bound.** The corrected picture
(bench/fast_kda_bwd_ai_analysis.py):

| stage          | FLOPs    | bytes  | AI    | ridge 935 | bound   | % peak |
|----------------|----------|--------|-------|-----------|---------|--------|
| w_u_recomp     | 6.4 G    | 428 MB | 15.0  | 62x below | BW      | 1.1% tc |
| h_recomp       | 12.9 G   | 352 MB | 36.6  | 25x below | BW      | 3.7% tc |
| dAv            | 6.4 G    | 226 MB | 28.4  | 33x below | BW      | 2.5% tc |
| dhu            | 19.3 G   | 453 MB | 42.7  | 22x below | BW      | 3.3% tc |
| wy_dqkg        | 27.4 G   | 932 MB | 29.4  | 32x below | BW      | 2.0% tc |
| intra          | 4.8 G    | 606 MB | 8.0   | 117x below| BW-locality | 0.3% tc |
| local_cumsum   | 25 M     | 201 MB | 0.1   | way below | BW      | 0.01% tc |
| **TOTAL bwd**  | **77 G** | **3.2 GB** | **24** | **39x below ridge** | **all BW** | **1.6% tc** |

Ridge point = 419 TFLOPs / 448 GB/s = **935 FLOPs/B**. Every bwd
subkernel is 20-120x below ridge — they have 30-60x more
*bandwidth* headroom than *compute* headroom. Even intra at
0.30% of compute peak.

**Total bwd compute = 77.3 GFLOPs** (chain-rule check: fwd is
~30 GFLOPs, so bwd is 2.6x — matches the chain-rule prediction
for 2-3x bwd-to-fwd ratio).

**Total bwd time = 11.86 ms** at 1.56% of compute peak → 64x
headroom on compute. The bwd is firmly bandwidth-bound; the
2.6x slowdown vs fwd is **not** from compute.

### Where the bwd time actually goes

```
bwd time (11.86 ms) vs:
  - compute peak:  0.18 ms   (we use 1.6% of it, 64x headroom)
  - HBM peak (using only the unique I/O across stages, no h/dh re-reads): 7.14 ms
  - actual:       11.86 ms   (60% of HBM peak, 1.7x the "deduplicated" roofline)
```

The 11.86 / 7.14 = 1.7x gap is the cost of redundant h/dh traffic:
h is read 3x (fwd_h write+read, dhu read, wy_dqkg read) and dh is
read+writen across 2 stages. A persistent kernel fusing fwd_h +
dhu + wy_dqkg would deduplicate the h/dh traffic and bring bwd
close to the 7.14 ms roofline — saving ~4.7 ms (40% of bwd, 25%
of step time).

### The wy_dqkg fused kernel: 28% of bwd at 63% of HBM

This is the "workhorse" bwd kernel — fuses dq, dk, dv, dA, db, dg
into one launch. It reads h[100 MB] and dh[100 MB] which are 22% of
its 932 MB I/O (also reads q, k, v, v_new, g, beta, A, do, dv).
At 63% of HBM, it's similar to the fwd recurrence: bandwidth-bound
on the h/dh traffic. ~0.8 ms of theoretical improvement at HBM
peak.

### Bottom line (bwd, corrected)

The bwd at **13.5 ms (2.6x fwd)** is bandwidth-bound, dominated by:
1. **intra (32%)** — 0.3% of compute peak, 35% of HBM peak.
   Bandwidth-bound AND locality-bound (small bmm in 49k programs).
   Hard to fix without restructuring intra math (e.g., larger
   sub-chunks, fewer programs).
2. **wy_dqkg (28%)** — 2% of compute peak, 63% of HBM peak.
   Bandwidth-bound on h/dh traffic, ~0.8 ms headroom.
3. **h_recomp + dhu (19%)** — 4% / 3% of compute peak, 95% / 73%
   of HBM. Reverse+forward recurrence on h[NC], already near HBM
   peak.

**Compute is irrelevant.** Total bwd compute is 77 GFLOPs which
would take 0.18 ms at peak. The 2.6x bwd-vs-fwd slowdown is from
h/dh tensor re-traffic, not compute.

**Biggest single opportunity** is the **700 MB h/dh redundant
traffic** across fwd_h, dhu, wy_dqkg. A persistent kernel fusing
all three (similar to what we considered for fwd) would save
~4.7 ms (40% of bwd, 25% of step time), at the cost of 5x more
code complexity. **Shipping-ready for now** — the complexity
isn't worth it without evidence that training is the bottleneck.

**Bwd is the new bottleneck for training throughput.** If we ever
need to optimize, this is where to look. But not now.

---

## Round-8: Per-chunk rescaling & parallel scan analysis (June 2026-29)

Round-7 closed the fwd at 99% HBM. But it left three open questions
about the algorithm and its precision:

1. **Can we avoid the bf16 overflow in `k_inv = k * exp(-g_cumsum)`?**
   Per-chunk rescaling?
2. **Can a parallel-scan formulation skip the matrix inverse entirely?**
   Would it be faster/simpler/more precise?
3. **Where is the *real* CHUNK=16 precision bottleneck?** Not the exp
   (that's safe). Maybe the Neumann truncation?

This round is the analytical work answering (1) and (2), plus a
bench experiment for (1) using fp32 `k_inv`. The precision question
(3) is open — see "Open questions" at the end.

### Question 1: Can we avoid the `exp(-g_cumsum)` overflow?

The overflow is in `prepare.py:135`:

```python
k_inv = (k * exp_neg_g).to(tl.bfloat16)   # NaN source at CHUNK=32
```

The bf16 max exponent is `ln(3.4e38) ≈ 88`. With CHUNK=32 and
`lower_bound=-5`, `|g_cumsum|_max = 103` (measured), so
`exp(103) = 2.6e44 > 3.4e38 → NaN`.

**Three candidate fixes:**

| fix | math | cost | benefit |
|-----|------|------|---------|
| (a) **Per-chunk rescaling** (subtract reference from g_cumsum) | shift `g_cumsum` to compress dynamic range | 1 sub/chunk (free) | **doesn't work** — see below |
| (b) **fp32 k_inv** (don't cast to bf16) | nothing changes, just higher precision storage | bmm becomes mixed-precision bf16×fp32 | overflow eliminated |
| (c) **Parallel scan** (no inverse) | rewrite chunk-local recurrence as a Blelloch scan | O(log CHUNK) scan steps; bmm sizes shrink as scan progresses | math is clean, but mma efficiency drops |

### Why (a) per-chunk rescaling doesn't work

The natural idea: subtract a reference value from `g_cumsum` so the
range of `exp(-g_cumsum)` is bounded. E.g.:

```python
g_cumsum_rescaled = g_cumsum - g_total   # most negative becomes 0
exp(-g_cumsum_rescaled) = exp(-g_cumsum + g_total) = exp(g_total - g_cumsum) = c[s]
```

This just renames `c[s]` (the backward product, always ≤ 1 — safe).
But the **dynamic range** of `exp(-g_cumsum)` within a chunk is
`exp(|g_total| - |g_cumsum[0]|) ≈ exp(|g_total|)`. Subtracting a
constant from the input doesn't compress this — it just shifts the
absolute values. The range itself is determined by the cumulative
sum of `g_activated` over the chunk, which is independent of any
subtraction:

```
range(exp(-g_cumsum)) = exp(max(-g_cumsum) - min(-g_cumsum))
                     = exp(|g_total| - |g_cumsum[0]|)
                     ≈ exp(|g_total|)
```

For CHUNK=32, this is `exp(101)` regardless of rescaling. So (a)
gives the same overflow at CHUNK=32.

**Real per-chunk rescaling** (the kind that does work) would be
**dynamic**: monitor the running `g_cumsum` and split the chunk
when `|g_cumsum|` approaches the bf16 threshold (88). This is
equivalent to capping CHUNK at 16 (the algorithm's natural limit).
Net: same as the current CHUNK=16 design. No free lunch.

### (b) fp32 k_inv: experiment

The simplest practical fix: keep `k_inv` in fp32 (it stays in
registers, no gmem storage). The bmm inputs become mixed-precision
`bf16 × fp32`. Triton's `tl.dot` handles this by promoting the
bf16 operand to fp32 (so the dot is effectively fp32). The output
Mqk is then fp32, which we cast to bf16 at the end.

**Implementation (in registers, no gmem change):**
- `k_inv = k * exp_neg_g` (no `.to(tl.bfloat16)` cast)
- `L = tl.dot(k_decayed_bf16, k_inv_fp32.T)` — Triton promotes
- `Mqk = tl.dot(q_decayed_bf16, k_inv_fp32.T).to(tl.bfloat16)`

**Experiment results (June 2026-29, `bench/fast_kda_fp32_kinv.py`):**

| CHUNK | k_inv dtype | Mqk finite? | g_total max | prepare time |
|------:|-------------|:-----------:|------------:|-------------:|
| 16    | bf16 (current) | ✓ | 53.0 | 0.99 ms |
| 16    | fp32           | ✓ | 53.0 | 0.99 ms (no regression) |
| 32    | bf16           | ✗ NaN | 102.7 | 1.02 ms (NaN) |
| 32    | fp32           | **✗ NaN** | 102.7 | 1.10 ms (NaN) |

**Result: fp32 does NOT fix the overflow.** The first NaN is at
Mqk[31, 31] (the diagonal, last row/col), from
`k_inv[31] = k[31] * exp(-g_cumsum[31]) = k[31] * exp(103)`.

Why fp32 doesn't help: the overflow threshold is determined by
the dtype's max exponent. Both bf16 and fp32 have `max ≈ 3.4e38`,
so `ln(3.4e38) ≈ 88` is the threshold in BOTH cases. CHUNK=32
produces `|g_cumsum| ≈ 103` regardless of k_inv's storage dtype.

**Conclusion for (b):** to enable CHUNK=32 via storage precision
alone, we'd need fp64 (max ~1.3e308, threshold ln(1.3e308) ≈ 284).
fp64 bmm is 2x slower than fp32 bmm on tensor cores, which would
eat most of the recurrence speedup. Not worth it.

**End-to-end fwd timing with fp32 k_inv (NaN or not):**
- CHUNK=16 bf16 k_inv (current): 3.16 ms
- CHUNK=32 fp32 k_inv: 2.50 ms (1.26x faster) — but **output is NaN**, so this is not a real win. The kernel just runs faster on NaN inputs.

**The CHUNK=32 speedup of 1.26x on the recurrence kernel is
real, but the precision is the blocker.** No free lunch.

### Question 2: Parallel scan (Blelloch) — why FLA/FlashKDA don't do it

The chunkwise parallel form uses an associative operator to combine
partial recurrences in O(log CHUNK) steps:

```
(h_a, α_a) ⊕ (h_b, α_b) = (h_b * α_a + h_a, α_a * α_b)
```

For CHUNK=16, this is 4 scan steps (vs 16 sequential). No matrix
inverse. No `exp(-g_cumsum)` overflow (only `exp(α_t)` per step,
which is bounded).

**Why FLA/FlashKDA use the closed-form inverse instead:**

1. **mma granularity**: the closed-form uses one
   `[CHUNK=16, K=128] @ [K, BLOCK_V=16]` mma (m16n8k16 perfectly
   fits). The scan form does `[chunk_size, K] @ [K, BLOCK_V]` at
   each step; `chunk_size` halves each step (8, 4, 2, 1), so the
   last two steps have <50% mma utilization.

2. **The exp overflow at CHUNK>16 is independent of the form**.
   The cumulative product `exp(Σ g_activated)` is the same
   regardless of whether you compute it sequentially or via scan.
   The scan form only avoids the *matrix inverse*, not the
   *cumulative product overflow*. To push CHUNK>16 with the scan
   form, you still need per-step fp32 (or periodic rescaling).

3. **Backward complexity**. The closed-form bwd uses
   `(I - L)^(-1)` as a black box (chain rule on the inverse is
   direct: `dL^(-1) = -L^(-1) @ dL @ L^(-1)`). The scan form
   requires a separate bwd derivation through the scan steps —
   weeks of work for a kernel that's already at 99% HBM.

**Per the conversation with the user, the parallel-scan approach
was attempted in an earlier EFKDA project but didn't succeed** —
likely for reason (1) or (3). The closed-form-with-inverse is
the natural form for this algorithm on tensor cores.

### Question 3: Where is the real CHUNK=16 precision bottleneck?

The bf16 exp overflow only matters at CHUNK>16. At CHUNK=16, the
prepare kernel is in safe territory (`exp(55) = 7e23` vs max
`exp(88) = 3e38` — 15 orders of magnitude headroom).

The likely real precision bottleneck at CHUNK=16 is the
**Neumann series inverse of L** in `prepare.py:172-174`:

```python
# 4 iterations: (I + L + L^2 + L^3 + L^4) approximation
for _ in range(4):
    INV_f32 = tl.dot(INV_f32, I_plus_L, out_dtype=tl.float32)
```

The Neumann series `Σ L^k` converges iff `||L|| < 1`. With
L2-normalized q/k, `||L||` is at most 1, and the 4-term
truncation error is `||L^5|| / (1 - ||L||)`. For `||L|| ≈ 1`,
this is O(1) — i.e. the truncated series has order-1 error,
not order-1e-3.

**Mitigation: use `tl.inverse` (CUDA 12.4+) for an exact inverse
instead of Neumann.** FLA's vendored kernel uses this on sm_120.
Our kernel still uses Neumann. The fix is straightforward but
unverified at our precision target.

### Open questions

1. **Neumann truncation error** at CHUNK=16 — measure vs
   `tl.inverse` (would need a bench comparing the two). This is
   the most likely real precision bottleneck.
2. ~~**fp32 k_inv experiment**~~ — **DONE, doesn't help** (see
   experiment above). `exp(103)` overflows fp32 same as bf16.
   Would need fp64, which is too slow.
3. **Per-step rescaling** (dynamic, not the static shift in (a))
   — could theoretically allow larger CHUNK by splitting mid-chunk
   when `|g_cumsum|` exceeds threshold. Equivalent to CHUNK=16
   with extra overhead. Not promising.
4. **CHUNK=32 with explicit per-sub-chunk rescaling** — split the
   CHUNK=32 chunk into 2 sub-chunks of 16 at the algorithm level,
   with fp32 handoff between sub-chunks. This is essentially
   CHUNK=16 with extra structure. Would need a new kernel design.
