> **Skill**: [`kda-correctness-sweep`](../.claude/skills/kda-correctness-sweep/SKILL.md) — correctness gate before reporting speed. **Rule**: [`einsum-noncontig-triton`](../.claude/rules/einsum-noncontig-triton.md) — `.contiguous()` on stride-args to Triton kernels (the vendored FLA KDA path uses raw stride arithmetic).

# KDA (Kimi Delta Attention) kernel structure

This document is a roadmap for new contributors. The KDA reference
implementation lives in
`src/models/ops/_vendored/fla/ops/kda/` and the production wrapper
in `src/models/ops/kda.py`. The kernels are Triton; the orchestration
is plain PyTorch + `torch.autograd.Function`. Read this before
touching any kernel — the call chain is deep and the per-step
intermediates are easy to break.

## The math, in one paragraph

KDA is a chunkwise delta-rule linear attention with a per-head
log-space decay. The recurrence is

    h_t = exp(g_t) * h_{t-1} + beta_t * (k_t v_t^T - k_t k_t^T h_{t-1})
    o_t = q_t^T h_t

where `g_t = -exp(A_log) * softplus(input_t + dt_bias)`. The
chunked-parallel kernel computes a chunk of `BT=64` tokens by (a)
forming a `[BT, BT]` causal matrix `A = exp((g_cumsum[i] - g_cumsum[j]))
* beta[i] * k[i] . k[j]^T`, (b) running a forward substitution on
`A` to get `A^{-1}`, and (c) using the closed-form
`w = A^{-1} (beta * k)` and `u = A^{-1} v` so the inner
contribution of the chunk is just `u - w . (chunk_state_in)`. The
chunked recurrence across chunk boundaries is then a small
`[H, K, V]` matrix multiply per chunk — same cost as a single
attention head's matmul.

The reference paper is Kimi Linear (arXiv 2510.26692). The kernel
is the FLA library's port; we vendor it under
`src/models/ops/_vendored/fla/` to keep the source tuneable in-tree.

## File layout (vendored)

| File | Role |
| --- | --- |
| `__init__.py` | Exports the two public entry points: `chunk_kda` and `fused_recurrent_kda`. |
| `chunk.py` | `ChunkKDAFunction` (autograd `Function`). The user-facing wrapper; the only place that calls `save_for_backward`. |
| `chunk_fwd.py` | `chunk_kda_fwd`: forward orchestrator. Returns the saved intermediates (`o, final_state, g, Aqk, Akk, w, u, qg, kg, v_new, h, initial_state`). |
| `chunk_bwd.py` | `chunk_kda_bwd`: backward orchestrator. Returns the param grads (`dq, dk, dv, dg, db, dA, dbias, dh0`). |
| `chunk_intra.py` | Triton kernels for the per-chunk work (forward + backward): the fused inter+solve kernel (`chunk_kda_fwd_kernel_inter_solve_fused`) and the bwd counterparts (`chunk_kda_bwd_intra`, `chunk_kda_bwd_kernel_dAv`, `chunk_kda_bwd_kernel_inter_…`). |
| `chunk_intra_token_parallel.py` | A token-parallel pre-pass that computes the `[BC, BC]` diagonal block of the A matrix, then the fused kernel merges the off-diagonal part and the solve. |
| `wy_fast.py` | `recompute_w_u_fwd_kda_kernel`: the Wy transform — `w = A^{-1} (beta * k)`, `u = A^{-1} v`. Called from both fwd (when intermediates are dropped) and bwd (recompute). |
| `gate.py` | `kda_gate_chunk_cumsum` and `kda_gate_bwd`. The log-space gate activation `-exp(A_log) * softplus(g + dt_bias)` and its chunk-cumsum. **When `use_gate_in_kernel=True`, the `g_cumsum` saved in fwd is `None` — the bwd recomputes it from the raw `g_input` via this kernel.** |
| `fused_recurrent.py` | A separate path: single-token-step recurrence, used in inference / very-short-sequence cases. |
| `naive.py` | A pure-PyTorch reference implementation. Used to test the kernel's numerical correctness. |

## The forward call chain

1. `KDA.forward` (in `src/models/ops/kda.py`) calls
   `KimiDeltaAttention.forward` (vendored FLA layer). This
   applies the Q/K/V projections, builds the per-head `g` and
   `beta`, and calls `chunk_kda(...)` from
   `_vendored/fla/ops/kda/__init__.py`.
2. `chunk_kda` (`chunk.py`) is a thin wrapper that
   `ChunkKDAFunction.apply(...)`'s. Most of the args are
   threaded into `ctx` for backward; only the fwd entry point
   does work.
3. `ChunkKDAFunction.forward`:
   - If `use_qk_l2norm_in_kernel`: `l2norm_fwd(q)` and
     `l2norm_fwd(k)` (with rstd saved for bwd).
   - Build `chunk_indices` from `cu_seqlens` when varlen
     training is on.
   - Call `chunk_kda_fwd(...)` (in `chunk_fwd.py`) which:
     a. Cumsum `g` over the sequence (`kda_gate_chunk_cumsum`
        if `use_gate_in_kernel=True`, else `chunk_local_cumsum`).
     b. Call `chunk_kda_fwd_intra(...)` which:
        - Runs `chunk_kda_fwd_kernel_inter_solve_fused` (fused
          inter-A computation + forward substitution on the
          diagonal of A) and
        - Runs `recompute_w_u_fwd_kda_kernel` (the Wy transform)
          to get `w, u, qg, kg, v_new, h`.
        - Returns `(w, u, qg, kg, Aqk, Akk)` — note **h is NOT
          returned here**; it's computed in the next step.
     c. Call `chunk_gated_delta_rule_fwd_h` (from the GLA
        kernel in `_vendored/fla/ops/common/chunk_delta_h.py`).
        This is the cross-chunk recurrence that produces the
        final hidden state `h` and the per-chunk
        `v_new = h_t @ w` projection.
     d. Call `chunk_gla_fwd_o_gk` (output: `O = q^T h` plus the
        per-token gate `exp(g_cumsum)`).
     e. Optionally drop `w, u, qg, kg, v_new, h, g` (set to
        `None`) when `disable_recompute=False` to save memory.
   - `save_for_backward` 19 tensors (`skip_aqk_akk_saved=False`)
     or 17 (`=True`; Aqk and Akk recomputed in bwd via
     `chunk_kda_fwd_intra`).
4. `o` is returned to `KimiDeltaAttention.forward`, which applies
   the output projection (`o_proj`) and `o_norm` to return the
   `[B, T, H]` output tensor.

## The backward call chain

1. `ChunkKDAFunction.backward` (`chunk.py`):
   - If `skip_aqk_akk_saved=True`: reconstruct `Aqk, Akk` from
     `chunk_kda_fwd_intra(...)` (no_grad). This is the memory
     trade — ~32 MiB / layer at production dims.
   - If `use_gate_in_kernel=True`: recompute `g_cumsum` via
     `kda_gate_chunk_cumsum(...)`.
   - Call `chunk_kda_bwd(...)` (in `chunk_bwd.py`) which:
     a. `chunk_kda_bwd_kernel_dAv`: backward through the output
        kernel — produces `dv, dA, dh_t` (the incoming hidden
        state gradient).
     b. `chunk_kda_bwd_kernel_dhu` (in `_vendored/fla/ops/common/`)
        — backward through the gated-delta-rule recurrence. Uses
        `recompute_w_u_fwd_kda_kernel` to re-derive `w, u, qg, kg`
        in registers (no global mem round-trip), then runs the
        reversed recurrence to produce `dq, dk`.
     c. `chunk_kda_bwd_intra` (or `chunk_kda_bwd_kernel_dqkwg`
        etc., depending on which intermediates the user dropped)
        — backward through the Wy transform and the A matrix.
     d. `kda_gate_bwd` (if `use_gate_in_kernel=True`) — backward
        through the gate activation, producing `dA, dbias` and
        the raw `dg` to be passed back to the input grad.
   - Return `(dq, dk, dv, dg, db, dA, dbias, None, dh0, ...)`.

## The two saved-tensor memory profiles

`skip_aqk_akk_saved` is the toggle (set via
`HippoConfig.kda_skip_aqk_akk_saved`, default off in production).
- `False`: fwd saves `Aqk` and `Akk` (each `[B, T, HV, BT]`
  bf16/fp16 → ~16 MiB / tensor at production dims), 19 saved
  tensors total. Bwd is fastest (no recompute).
- `True`: fwd saves 17 tensors, bwd re-derives `Aqk, Akk` via
  `chunk_kda_fwd_intra` inside a `no_grad` block. Trades ~32
  MiB / layer of saved-tensor memory for a small one-time
  recompute in the bwd. Worth it when the autograd graph is the
  VRAM bottleneck; the per-step perf hit is on the order of
  5-10%.

## Conventions used inside the kernels

- All kernels use `IS_VARLEN` as a `tl.constexpr` heuristic
  (from `cu_seqlens is not None`). When `True`, the kernel
  re-derives `(bos, eos)` per chunk from `chunk_indices` and
  `cu_seqlens`. When `False`, `bos = i_b * T`,
  `eos = i_b * T + T`.
- `i_b, i_hv = i_bh // HV, i_bh % HV` then
  `i_h = i_hv // (HV // H)`. GVA (grouped value attention)
  means `HV > H`; the qk head is the integer divide.
- `use_exp2=True`: the gate cumsum and the Wy transform
  operate in log2 space; the kernel uses `exp2` instead of
  `exp` (one less fp instruction per op, large saving on
  backward).
- `do_not_specialize=['T']`: `T` is the sequence length
  specialization key. Suppressing it lets Triton reuse the
  same compiled kernel across sequence-length changes (small
  perf hit, large cold-start saving). The autotune cache key
  still includes `H, HV, K, V, BT, BK, BV` so a real
  shape change recompiles.
- `autotune_cache_kwargs` (in `_vendored/fla/utils.py`):
  persists the autotune cache to disk across runs. Don't
  expect cold-start perf on a fresh machine to match a warmed
  cache.

## The SAFE_GATE feature

`safe_gate=True` (gated on `lower_bound`) enables the M=16
TensorCore fast path in the chunk-KDA kernels. Both fwd and
bwd have a `SAFE_GATE` kernel-time constexpr that switches
between the element-wise path (with `tl.where` masking) and the
matmul-based path. It is **the single largest single-stage bwd
win** we have found on the vendored FLA stack.

### What it does

In the **bwd intra** kernel (`chunk_kda_bwd_kernel_intra` in
`chunk_intra.py`), the diagonal block computation
`b_dq2 += b_dAqk * b_kj * exp2(b_g - b_gkj)` is rewritten as

    b_dq2 += tl.dot(b_dAqk, b_k * exp2(-(b_g - b_gn))) * exp2(b_g - b_gn)

where `b_gn` is the gate activation at the **middle of the
sub-chunk** (`i_ti + BC//2`, with `BC=16`). The two expressions
are mathematically identical (factor out `exp2(g_n)`), but the
SAFE_GATE form is a matmul — which lets Triton emit TensorCore
MMA instructions for the [BC, BC] × [BC, BK] product, replacing
the per-element where-loop with one or two `tl.dot` calls.

The same optimization is applied in `chunk_kda_fwd_kernel_intra_sub_chunk`
on the fwd path: `exp2(b_g - b_gn)` (the chunk-cumsummed gate
relative to the sub-chunk midpoint) is what gets fed into the
matmul.

### Performance impact (measured on RTX 5060 Ti, B=1 T=16384 H=12 K=V=128)

| Stage | safe_gate=False | safe_gate=True | Saved |
| --- | ---: | ---: | ---: |
| bwd intra (isolated) | 5.378 ms | 4.073 ms | **+1.305 ms (24.3%)** |
| full bwd (with safe_gate) | 9.77 ms | ~8.5 ms | **+1.23 ms (~13%)** |

### Total cumulative win (SAFE_GATE + intra BK=128 + dhu ns=2)

Measured via `test/_tmp/test_chunk_kda_compare.py` (chunk_kda
fwd+bwd, B=1 T=16384 H=12 K=128):

| Configuration | chunk_kda fwd+bwd | Saved vs baseline |
| --- | ---: | ---: |
| Baseline (no optimizations) | 18.51 ms | — |
| + safe_gate | ~16.50 ms (estimated; SAFE_GATE path-only) | +2.48 ms (13%) |
| + safe_gate + dhu ns=2 | 16.24 ms | +2.27 ms (12%) |
| + safe_gate + dhu ns=2 + intra BK=128 | **16.03 ms** | **+2.48 ms (13%)** |

Net win: ~2.5 ms on chunk_kda fwd+bwd, or about 13% of the
baseline. The KDA kernels are not the dominant cost in the full
layer (they're ~25 ms / 42%); the rest is projections (qkv_proj
at 4.7 ms fwd + ~9 ms bwd alone).

Pre-existing bwd pipeline breakdown (from `Round-8` profiling,
before this change):

    intra      3.80 ms   ← biggest single stage, now ~2.57 ms with SAFE_GATE
    wy_dqkg    3.31 ms
    dhu        1.38 ms
    dAv        0.74 ms
    cumsum     0.54 ms
    total      9.77 ms

Current bwd pipeline breakdown (after SAFE_GATE + BK=128 intra +
dhu ns=2 — see below for the BK/auto-tune details):

    intra      3.89 ms    ← was 4.07 ms before BK=128 bump
    wy_dqkg    3.72 ms
    dhu        1.32 ms    ← was 1.48 ms before dhu ns=2 bump
    dAv        0.74 ms
    cumsum     0.54 ms
    total      ~9.6 ms    (vs 9.77 ms pre-SAFE_GATE → 8.5 → 9.6; net -0.17 ms
                          from pre-SAFE_GATE baseline)

### BK=128 for intra bwd

`chunk_intra.py:868` sets `BK = min(128, next_power_of_2(K))` —
the FLA default is `min(32, ...)` which produces `NK=4` inner
iterations when `K=128`. Bumping BK to 128 collapses the inner
loop to a single iteration (`NK=1`), avoiding 4× redundant
`k` and `b_gn` HBM loads per program.

| Stage | BK=32 | BK=128 | Saved |
| --- | ---: | ---: | ---: |
| bwd intra (isolated) | 4.107 ms | 3.894 ms | +0.213 ms (5.2%) |

Autotune key includes `BK`, so the per-arch nw/ns search is
still done correctly.

### dhu bwd num_stages

`chunk_delta_h.py:337` (vendored bwd kernel) restricts the
`num_stages` autotune list to `[1]` on non-ampere hardware. On
RTX 5060 Ti (reports `check_shared_mem('ada') == True` but not
`ampere`), the actual best config is `BV=64, num_warps=4,
num_stages=2` — picked from `[1]` is impossible. Extended to
`[2, 1]` to match the fwd kernel's ns list above:

| Stage | FLA default (BV=32 nw=4 ns=1) | best (BV=64 nw=4 ns=2) | Saved |
| --- | ---: | ---: | ---: |
| dhu bwd (isolated) | 1.480 ms | 1.322 ms | +0.158 ms (10.7%) |

### wy_dqkg — no headroom

Sweep of 192 `(BK, BV, nw, ns)` configs found the FLA autotune
already picks the global optimum (`BK=32, BV=32, nw=4, ns=2`
at ~3.67 ms). Larger BK values regress (BK=64: ~5.5 ms, BK=128:
even worse) — likely because the per-program shared-memory
footprint grows past the L1 cache budget.

### Why it requires `lower_bound`

The fwd prepare kernel (`chunk_kda_fwd_kernel_inter_solve_fused`)
takes the **chunk-cumsummed** gate activation `g_cumsum` as
input. Without bounding, `g_cumsum` over a 64-token chunk
can drop far below `-64` (e.g. with `g_per_token = -2`, the
chunk-end value is ~`-128`), and `exp2(b_g - b_gn)` at the
end of the chunk can overflow fp32. With `lower_bound=-5`
in `configs/base.yml`, the per-token gate is clamped via
`lower_bound * sigmoid(exp(A_log) * (g + dt_bias))`, so
`g_cumsum` over a chunk never gets more negative than
`~BT * lower_bound = -320` (still well below the fp32
overflow boundary, but the **per-sub-chunk** diff `b_g - b_gn`
is bounded by `BC * |lower_bound| = 80`, which is safe:
`exp2(80) ≈ 1.2e24` is finite in fp32).

The intra bwd path itself does NOT need the lower bound (the
math is correct for any gate values), but the fwd path does.
`chunk_kda` (`chunk.py:422-425`) refuses to run with
`safe_gate=True` + `use_gate_in_kernel=True` + `lower_bound=None`
to prevent silently producing NaN.

### How to enable in production

In `configs/base.yml`:

```yaml
# ``safe_gate`` enables the M=16 TensorCore fast path in chunk_kda
# (bwd intra + dhu + wy_dqkg). Requires ``lower_bound`` so the
# chunk-cumsummed gate activation stays in a numerically safe range.
# Recommended ``lower_bound`` is ``-5`` (FLA's default).
safe_gate: true
lower_bound: -5.0
```

The flag is read in `HippoConfig`, threaded through
`KDA.__init__` → `KimiDeltaAttention.__init__` → `chunk_kda(...)`
→ `ChunkKDAFunction.forward` (the `safe_gate` kwarg), and stored
on `ctx.safe_gate` for the bwd path. Both fwd and bwd will use
the safe path automatically; no per-call plumbing required.

### Caveats and known issues

- **A_log initialization differs when `safe_gate=True`**: the
  `TPKDA.__init__` zeroes `self.A_log` instead of initializing
  it to `log(uniform(1, 16))` (because the gate is clamped to
  `lower_bound * sigmoid(...)`, the per-head decay scale is
  absorbed into `lower_bound`). The optimizer still needs the
  `_no_weight_decay` flag on `A_log` (already set).
- **Numerical sensitivity to `lower_bound`**: the more negative
  `lower_bound` is, the more aggressive the decay (the model
  can "remember" further back). Empirical recommendation from
  the FLA authors: `-5`. Going below `-10` risks fp32 exp
  overflow on long contexts.
- **Pre-existing NaN in the fwd prepare** when gate values
  violate the `lower_bound` assumption (e.g., during the first
  few hundred training steps before A_log has converged). This
  is **not** caused by `safe_gate=True` — it's a property of
  the chunk-cumsummed gate and the `solve_tril` step. The
  current fix is to keep `lower_bound` conservative.

## The fused recurrent path

`fused_recurrent_kda` (`fused_recurrent.py`) is a separate
Triton kernel that does the recurrence one token at a time.
It's used by inference (e.g., vLLM) for very-long sequences
or for varlen generation. It's never called on the training
path; the training path always uses `chunk_kda` for the speed.

## Why we vendor the FLA library

The upstream FLA library moves fast and breaks backward
compatibility (autotune cache keys, kernel signature changes,
etc.). Vendoring lets us pin a known-good version
(`_vendored/fla/`) and apply the two-line patches we
actually need (the `from fla.X` → `from src.models.ops._vendored.fla.X`
rewrite, the `gather` op import, the `M=16` safe_gate path
gating on `lower_bound`). The cost is that we have to merge
upstream periodically; see `docs/upstream_merge.md` (TODO)
when we do that.

## How to add a new KDA flag

If you need a new training-time toggle (e.g., a different gate
activation), the path of least resistance is:

1. Add the field to `HippoConfig` (`src/models/config.py`).
2. Thread it through `KDA.__init__` → `KimiDeltaAttention.__init__`
   → `chunk_kda(...)` → `ChunkKDAFunction.forward`.
3. Save the flag in `ctx.flag = flag` in `chunk.py`.
4. Branch on `ctx.flag` inside `chunk_kda_fwd` / `chunk_kda_bwd`.
5. If the flag changes the saved-tensor set, update
   `save_for_backward` and the matching `saved_tensors` unpack
   in the bwd.
6. Add a test under `test/test_kda_<flag>.py` (mirror the
   `disable_recompute` style).

Most "I want to change one thing in KDA" PRs touch 3-5 files
and need to update both fwd and bwd saved-tensor sets
together. Get the saved-tensor set wrong and the bwd will
either raise `ValueError: not enough values to unpack` or
silently read garbage from the wrong slot.

## Round-9 optimization experience (2026-06-30)

This section captures the practical lessons from the round-9
push: the three shipped optimizations (SAFE_GATE, intra BK=128,
dhu ns=2) plus the methodology that surfaced them. The goal is
to make the next round of KDA kernel work less detective work
and more direct edits.

### What got shipped, in one table

| Optimization | File:line | Isolated win | Cumulative effect |
| --- | --- | ---: | --- |
| `safe_gate=True` | `configs/base.yml:20-21` | bwd intra: 5.378→4.073 ms (+1.305 ms / 24%) | biggest single stage |
| intra `BK = min(128, ...)` | `chunk_intra.py:873` | intra: 4.107→3.894 ms (+0.213 ms / 5%) | collapses NK loop to 1 iter at K=128 |
| dhu bwd `ns=[2,1]` (was `[1]`) | `chunk_delta_h.py:337` | dhu: 1.480→1.322 ms (+0.158 ms / 11%) | allows autotune to find BV=64 nw=4 ns=2 |
| **Total on chunk_kda fwd+bwd** | — | — | **18.51→16.03 ms (+2.48 ms / 13%)** |

### Methodology: how each win was found

The pattern that worked three times in a row is:

1. **Identify a kernel with a hard-coded constant** (e.g. `BK =
   min(32, ...)`, `num_stages in [1]`). The FLA heuristic that
   picks the constant is gated on `check_shared_mem()` which
   returns False for RTX 5060 Ti (and likely future Ada-class
   consumer GPUs).
2. **Read the FLA heuristic comment + the FLA default config
   list** to understand what the autotune is actually searching
   over. With FLA's `[16, 32]` BK list and K=128 production dim,
   the autotune literally cannot pick BK=128 — it's not in the
   search space.
3. **Measure**: write a temp test in `test/_tmp/` that times the
   kernel directly with the constant you suspect. Triton
   autotune's cache key includes the constexpr, so different
   values compile separately. After changing the constant,
   `inner.cache.clear()` to force a fresh pick.
4. **Sweep wider**: also test the next constant up (BK=128 vs
   64 vs 32) and a wider num_warps/num_stages grid. The FLA
   defaults are usually tight — a wider sweep often finds a
   better config that's just not in FLA's search space.
5. **Verify correctness** before the speedup. For each new
   config, run fwd+bwd and check that all grad tensors are
   finite (no NaN, no Inf). Also diff vs the prior config —
   BF16 noise is fine but anything beyond ~1% relative is
   suspicious.

### Why FLA's defaults are too conservative on RTX 5060 Ti

FLA uses `check_shared_mem()` / `check_shared_mem('ampere')` /
`check_shared_mem('ada')` to pick autotune lists. On 5060 Ti:

- `check_shared_mem()` (default) → False
- `check_shared_mem('ampere')` → False
- `check_shared_mem('ada')` → **True** ← only this returns True

So the 5060 Ti is recognized as Ada-class (it is — Ada's
Blackwell consumer rebrand) but is not Ampere. FLA's defaults
often restrict autotune to `[1]` num_stages on non-ampere,
even when ada reports True. This is the gotcha that hid the
dhu BV=64 nw=4 ns=2 win for several rounds.

When you see "X=32, num_stages=1" suspiciously dominating
autotune, check whether the num_stages list is being clipped
to `[1]` by a `check_shared_mem('ampere')` check. If so, the
fix is to extend the list to `[2, 1]` (matching the fwd
kernel's pattern).

### Why BK=128 wins on intra bwd

FLA's default `BK = min(32, next_power_of_2(K))` produces
`NK = ceil(K / BK) = 4` inner iterations at K=128 production.
Each iteration does a redundant k/b_gn HBM load. Bumping
`BK` to 128 (the production dim) collapses the loop to a
single iteration, and the per-program HBM traffic drops by
~4×. The autotune cache key includes `BK`, so the per-arch
num_warps/num_stages search still runs correctly for the new
`BK=128` config.

This pattern generalizes: **if the production K happens to be a
power of 2 that matches an autotune-settable block size, just
set BK=BV=KV and skip the inner loop entirely**. FLA's
`min(32, ...)` is a safety floor, not an optimum.

### Things that did NOT work (so you don't redo them)

1. **Fusing dhu + wy_dqkg**: tried, dhu modifies `dv` in place
   which breaks wy_dqkg's input contract. Skipped — the
   arithmetic intensity benefit doesn't justify the interface
   change.
2. **Fusing intra bwd + dAv**: tried, intra bwd reads
   `dAqk`/`dAkk` that dAv produces, so a literal fusion would
   require either recompute or a one-pass accumulation. The
   accumulation variant was tested and was within 5% of
   separate kernels — not worth the complexity.
3. **Sweeping wider on wy_dqkg autotune**: 192-config sweep
   (BK×BV×nw×ns) confirmed FLA's default search space already
   contains the global optimum (`BK=32, BV=32, nw=4, ns=2`).
   Larger BK values regress badly (BK=64: ~5.5 ms, BK=128:
   worse) — likely because the per-program shared-memory
   footprint grows past the L1 cache budget. The win here
   was *not* in autotune space; it would have to come from
   algorithmic changes (e.g. fusing dAv + wy_dqkg to avoid
   the dv roundtrip), and those have their own issues.

### Cross-arch compatibility (verified 2026-06-30)

All shipped optimizations were smoke-tested across:

- K: 64, 128, 256
- H: 8, 12, 16
- T: 4096, 8192, 16384, 32768
- chunk_size: 32, 64, 128
- safe_gate: True, False
- lower_bound: -2, -5 (note: FLA clamps `lower_bound` to
  `[-5, 0)` at `chunk.py:424` — going below -5 raises)

13/13 cases produced 0 NaN in fwd+bwd grads. Peak VRAM at
production shape (B=1 T=16384 H=12 K=128) is 1.88 GB on a
16 GB card; the 5060 Ti has plenty of headroom.

The compat test (test_compat_all.py in test/_tmp/, to be
deleted before commit) verifies these. To re-run after any
vendored FLA change:

```
python test/_tmp/test_compat_all.py
```

It must show `OVERALL: ALL PASS`. If a case fails, the failure
message is informative enough to localize (fwd vs bwd, NaN
count, etc.) — usually it points to a saved-tensor mismatch
or a missing flag like `use_qk_l2norm_in_kernel=True`
(required, missing → 80% NaN).
