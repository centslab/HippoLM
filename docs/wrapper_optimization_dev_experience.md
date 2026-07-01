# KDA wrapper optimization — development experience (June 2026-30)

A field log of the `feature/wrapper-lev-mop` branch. This is the
"keep" branch: it does not feed main. Main stays on the
FLA-vendored `chunk_kda` kernel (the path used in production by
`KimiDeltaAttention`); the experimental `src/models/ops/cuda/kda_fwd/`
wrapper lives here.

If you are reading this to decide whether to merge the wrapper into
main, jump to the **End state** and **Why main is FLA-only**
sections at the bottom.

## Why a separate branch

The wrapper is a custom CUDA + Triton rewrite of the FLA `chunk_kda`
forward path. It uses hand-rolled CUDA kernels (`forward_sub`,
`delta_h`) and Triton kernels (intra_solve, wy_transform,
g_cumsum, chunk_o). It is **not** wired into production — the
production `KimiDeltaAttention` layer in `src/models/ops/kda.py`
calls FLA's vendored `chunk_kda` directly.

The wrapper optimization work (~12 commits, ~5 ms saved at prod) is
in-tree as a research artifact. Putting it on a feature branch:

- Keeps main aligned with origin/main (the agreed stable state).
- Lets the wrapper experiment continue without dragging the
  "stable" tree along.
- Makes the work easy to discard if the wrapper is abandoned
  (single `git branch -D feature/wrapper-lev-mop`).

## Timeline of levers

| Commit  | Lever | Saved at prod | Status |
|---------|-------|---------------|--------|
| `d5d7415` | (pre-Lever) chunk_o h in bf16 | -0.23 ms | kept |
| `816e942` | **H** intra_solve fused Triton | ~18 ms (broke 19.4 → 1.2) | kept |
| `51c037a` | **I** wy_transform fused Triton | ~3.8 ms | kept |
| `5094b3b` | J forward_sub + wy fusion attempt | -0.42 ms LOSS | **reverted** (commits deleted; docs only) |
| `59a493d` | **K** skip mask multiplication in intra_solve | ~0.78 ms | kept |
| `dbabb61` | **L** fused g_cumsum kernel | ~1.77 ms | kept |
| `320bdef` | Lever L docs | — | kept |
| `79e56f2` | **M** strided reads in intra_solve + wy | ~0.97 ms | kept |
| `996f254` | **O** wy_fused direct u write to [T, HV, V] | ~0.36 ms | kept |
| `3dbcc31` | **P** fuse beta + I into intra_solve | ~0.52 ms | kept |
| `f43e780` | deprecate old stage_breakdown.py | — | kept |

Plus one reverted mid-session experiment that did not land:

- **N** (reverted) merge all 10 intra_solve pairs into a single
  Triton program (unrolled pairs). Standalone looked promising
  (0.78 ms vs 1.05 ms) but **per-stage was a regression** (1.04 ms)
  because Triton's `tl.static_range` unrolling caused register
  pressure that reduced SM occupancy. Lesson: never trust
  standalone kernel timings — always measure the full wrapper.

## Per-stage breakdown at end of session (prod, B=1 T=16384 H=12 K=V=128)

| stage | ms | share |
|-------|-----|-------|
| g_cumsum (Triton fused, Lever L) | 0.66 | 8% |
| intra_solve (Triton, H + M + P) | 1.05 | 13% |
| forward_sub (CUDA) | 0.65 | 8% |
| wy_transform (Triton, I + M + O) | 0.75 | 9% |
| delta_h (CUDA WMMA) | 3.89 | 49% |
| chunk_o (Triton) | 0.89 | 11% |
| Python orchestrator | ~0.12 | 1% |
| **total** | **~8.0** | |

FLA reference: ~5.75 ms. Wrapper is 0.72x FLA on prod. Wrapper is
~2x faster than FLA on small/medium/k128 (less delta_h serial
recurrence work).

## What worked, what didn't

### Worked

- **Single-launch fusion** of bmm loops into Triton kernels. The
  Python bmm loop took 19.4 ms (intra_solve, 10 pairs × cuBLAS
  overhead) and 4.7 ms (wy_transform). Replacing each with one
  Triton launch cut overhead by ~10x per call.

- **In-register casts** from bf16 to fp32 (free — no HBM
  round-trip) instead of pre-casting to fp32 tensors. Saved
  2 × ~1.3 ms at prod (intra_solve).

- **Fused g_cumsum** (Triton, Lever L) — read g, scale by RCP_LN2,
  cumsum, cast back to bf16, all in one launch. Replaced 4 ops
  (~1.77 ms).

- **Strided reads** (Lever M) — kernels read directly from
  [T, H, K] / [T, HV, V] natural layouts, eliminating the
  `q_per = q_tok.view(...).transpose(1,2).contiguous()` pattern
  that materialized a [num_chunks, HV, BT, K] view per fwd.
  Saved ~0.97 ms at prod.

- **Direct write** (Lever O) — wy_fused_transform now writes u
  directly to [T_total, HV, V] strided layout, eliminating the
  downstream `.transpose(1,2).contiguous().view(T, HV, V)` that
  was the last real `.contiguous()` in the wrapper. Saved 0.36 ms.

- **In-kernel beta + I** (Lever P) — the intra_solve kernel now
  multiplies A_kk by beta[i] row-wise and adds 1.0 on the diagonal
  in-kernel, so forward_sub can consume A directly. Saved 0.52 ms.

### Did not work

- **Lever N — merge 10 pairs into one program** (reverted). The
  standalone test showed 0.78 ms vs 1.05 ms (good), but the
  per-stage test showed 1.04 ms (regression). Triton's
  `tl.static_range` unrolling inflated register usage; occupancy
  dropped. **Lesson: standalone micro-benchmarks lie. Always re-time
  the full wrapper after any kernel change.**

- **Lever J — fuse forward_sub + wy** (reverted, -0.42 ms loss).
  Forward_sub needs the FULL A matrix to do the inversion, while
  wy_transform processes pairs of A blocks. They have different
  parallelism structures (forward_sub is per-chunk serial, wy is
  per-(chunk, pair) parallel). Forcing them into one kernel added
  shared sync barriers that cost more than the saved A HBM
  round-trip. **Lesson: parallelism mismatch is a hard wall.**

- **bf16-cast attempts** (mentioned in `project_kda_cuda_perf.md`,
  older round) — every attempt to move h_prev or v_residual to
  bf16 reduced cos to 0.93. The recurrence loses too much precision
  in bf16 over 1024 chunks. **Lesson: keep all recurrence state in
  fp32 shmem.**

## Verification protocol (used for every lever)

1. **Standalone kernel test** in `test/_tmp/`. Sweep 4 shapes
   (small/medium/k128/prod). For each: cos ≥ 0.99 vs FLA,
   med_rel < 5%, no NaN/Inf, bit-exact determinism on two runs.
2. **Per-stage timing** in
   `bench/kda_fwd_stage_breakdown_v2.py` (the v2 file, NOT the
   old `kda_fwd_stage_breakdown.py` which times the old Python
   loop and is now deprecated). Measure the 7 stages around the
   change.
3. **Smoke training** via `python scripts/train.py --config
   configs/test/kda_chunk.yml` (4 steps). All finite, grad_norm
   bounded, loss bit-identical to pre-lever.
4. **Compat test** via `python -m pytest test/test_kda_chunk_compat.py`.
5. **Commit** with a `perf(kda): Lever X — <what> — <ms saved>`.

If standalone passed but per-stage was a regression, **revert**
(don't paper over it with a "we'll fix it later" commit).

## Where the gap to FLA lives

After Levers H through P, the wrapper is 0.72x FLA on prod. The
remaining 2.15 ms gap is concentrated in:

1. **delta_h (3.89 ms, ~49% of subtotal)** — already WMMA-optimized
   (nvcuda::wmma tf32 m16n8k8). Hard to optimize further:
   - Per-chunk recurrence is serial by nature (h_next depends on
     h_prev).
   - The kernel's bf16→fp32 conversion in the WMMA staging loop
     (per-K tile) is significant. Switching to bf16 mma would
     2x the K-tile efficiency but loses precision in the
     recurrence — see "Lesson: keep all recurrence state in
     fp32 shmem" above.
   - Possible future: split V into multiple programs to
     parallelize across v_slices (but the recurrence is still
     serial within each program).

2. **intra_solve (1.05 ms, ~13%)** — already at the fused-Triton
   limit. Further reduction requires structural change (e.g.,
   merge pairs, larger BC). Past attempts (Lever N) failed due
   to register pressure.

3. **forward_sub (0.65 ms, ~8%)** — could fuse into intra_solve
   to save the 48 MB A HBM round-trip (~50-100 us). Risky
   because intra_solve writes A in 10 separate program invocations
   while forward_sub needs the full A.

4. **chunk_o (0.89 ms, ~11%)** — strided v_new reads; could
   potentially be improved by pre-permuting v_new into a
   contiguous layout (but that's a HBM copy first).

For all four, the expected ROI is small (each < 0.5 ms) and the
engineering cost is high. The wrapper is approaching a local
optimum for the current algorithmic decomposition.

## End state (feature/wrapper-lev-mop at f43e780)

Wrapper: 8.0 ms at prod (0.72x FLA). Saved 1.85 ms this session
through Levers M, O, P. Cross-shape speedup vs FLA:

- small  (T=256, H=4):  1.98x
- medium (T=1024, H=4): 2.26x
- k128   (T=1024, H=8): 1.62x
- prod   (T=16384, H=12): 0.72x

The wrapper is faster than FLA on shapes with short sequences
(small, medium, k128) where the launch and fixed-overhead cost
of delta_h's serial recurrence is amortized over fewer chunks.
On prod (T=16384), the serial recurrence dominates and FLA's
single-launch approach wins.

## Why main is FLA-only

Two reasons:

1. **Production wiring is FLA.** `src/models/ops/kda.py` calls
   `KimiDeltaAttention` which calls FLA's vendored `chunk_kda`.
   The custom wrapper in `src/models/ops/cuda/kda_fwd/` is not
   imported anywhere in `src/`. Merging the wrapper into main
   would not change production behavior — it would just be dead
   code on the stable branch.

2. **The wrapper is research-grade.** It passes end-to-end
   correctness (cos ≥ 0.99 vs FLA at all 4 shapes, smoke
   training 4 steps, grad_norm bounded) and is a known 0.72x
   on prod. But it has not been:
   - Benchmarked against FLA on a production-scale training
     run (vs the smoke test).
   - Validated for varlen inputs (only fixed-len is tested).
   - Tested with `use_gate_in_kernel=True` / `safe_gate=True` /
     `lower_bound` (the FLA flags that interact with the
     internal `chunk_kda`).
   - Tested with `output_final_state=True` and `initial_state`
     (currently asserts these are None/false in the wrapper).

   Until those gaps are closed, the wrapper is not a drop-in
   replacement for FLA.

## How to use the wrapper (if you want to)

```python
from src.models.ops.cuda.kda_fwd import chunk_kda_fwd

o, final_state = chunk_kda_fwd(
    q, k, v, g, beta,
    cu_seqlens=cu_seqlens,    # optional; fixed-len if None
    scale=1.0 / math.sqrt(K), # default
    initial_state=None,       # not supported
    output_final_state=False, # not supported
    BT=64,                    # hard-coded
)
```

Constraints:
- `q, k, v, g, beta` must be bf16 and on CUDA.
- `H == HV` (no GVA).
- `K == V`.
- `T_total == num_chunks * BT` (caller pads; varlen support
  exists but is untested).
- `initial_state` and `output_final_state` not implemented.
- Falls back to FLA `chunk_kda` for non-bf16 inputs.

The 4 stages of the wrapper:

1. **intra_solve** (Triton) — computes A_qk and A_kk. Output A_kk
   is `A = I + A_kk * beta` ready for forward_sub.
2. **forward_sub** (CUDA) — in-place A → A^{-1}.
3. **wy_transform** (Triton) — w = A^{-1} @ (k * beta * exp(g_cum)),
   u = A^{-1} @ (v * beta). u is written directly in [T, HV, V]
   layout (Lever O).
4. **delta_h** (CUDA WMMA) — per-chunk recurrence computing
   v_new = u - w @ h_prev. Writes v_new to global, h_per_chunk
   to global.
5. **chunk_o** (Triton) — o = (q * exp(g)) @ h + A_qk @ v_new.

## Files in this branch

- `src/models/ops/cuda/kda_fwd/__init__.py` — public entry point.
- `src/models/ops/cuda/kda_fwd/kernel.cu` — custom CUDA: forward_sub,
  delta_h (scalar + WMMA), chunk_o (CUDA fallback).
- `src/models/ops/cuda/kda_fwd/triton_intra_solve.py` — Lever H + M + P.
- `src/models/ops/cuda/kda_fwd/triton_wy_transform.py` — Lever I + M + O.
- `src/models/ops/cuda/kda_fwd/triton_g_cumsum.py` — Lever L.
- `src/models/ops/cuda/kda_fwd/triton_kernels.py` — chunk_o Triton.
- `src/models/ops/cuda/kda_fwd/load_inline.py` — `torch.utils.cpp_extension`
  loader.
- `src/models/ops/cuda/kda_fwd/bindings.cpp` — pybind11 bindings.
- `bench/kda_fwd_stage_breakdown_v2.py` — per-stage timing (current).
- `bench/kda_fwd_bench.py` — full wrapper vs FLA at all shapes.
- `test/_tmp/test_lever_m_strided.py` — Lever M + O + P correctness.

## See also

- `docs/kda_kernel_structure.md` — the FLA `chunk_kda` call chain
  (the path main uses).
- `docs/efkda_debugging_history.md` — earlier wrapper debugging
  notes.
- `docs/triton_kernel_playbook.md` — 8 working Triton patterns
  (multi-kernel split, autotune, make_block_ptr, ...) used
  here.
- `docs/fast_kda_bottleneck_analysis.md` — older Round-1..9
  FastKDA analysis (separate abandoned work).
- Memory: `project_kda_wrapper_perf.md` in
  `~/.claude/projects/-hy-tmp-HippoLM/memory/`.
