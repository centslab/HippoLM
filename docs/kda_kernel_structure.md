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
