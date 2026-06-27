# EFKDA debugging history

This document captures the full debugging history of the EFKDA
(EFLA-form KDA) kernel — what was tried, what failed, and the
final stable state. KDA is in production; the EFKDA branch
(`efkda-exp`) was the experimental fork that fed into the
production KDA. Reading this before working on the KDA Triton
forward / backward saves a week of re-derivation.

## Why this matters

The four "fixes" below (VRAM, fp32 NaN, cu_seqlens state-leak,
tail-pad state-leak) are the difference between a kernel that
runs and a kernel that produces infinite gradients. Anyone
touching the KDA forward / backward (vendored
`src/models/ops/_vendored/fla/ops/kda/`) needs to know about all
four — the first three hide inside the matmuls, the fourth hides
in the packer, and the failure mode of each is "loss is fine for
many steps, then NaN on step N".

## Fix 1 — VRAM (per-chunk checkpoint)

Each chunk's forward is wrapped in
`torch.utils.checkpoint.checkpoint(...)` so per-chunk
intermediates (the `[B, H, L, L, K]` rank-5 tensors) are freed
after the chunk's forward and recomputed on backward. Without
this, all 64 chunks' saved-for-backward tensors are alive at
once, blowing 16 GB at the base config (B=4, T=4096, L=32,
H=8, K=128).

- Forward peak: ~5 GB. Backward peak: ~9 GB at the above dims.
- Fits in 16 GB.

## Fix 2 — fp32 NaN in upper triangle

The kernel's `exp()` of a strictly-decreasing difference was
overflowing fp32 in the upper-triangle positions (decay is
monotone, so `diff_g > 0` there, but the masked-out positions
were 0). The 0 * inf = NaN in the VJP.

- Fix: `torch.where(strict_lower_mask, diff_g, 0)` BEFORE `exp()`.
- Replaced the explicit `(I+T)^{-1}` matmul with
  `torch.linalg.solve_triangular(., upper=False)`. The
  lower-triangular inverse of a dense lower-triangular matrix has
  entries that grow exponentially with L; forward-substitution's
  VJP is bounded.

## Fix 3 — cu_seqlens state-leak (the actual root cause of the second inf-grad bug)

The kernel previously did NOT honor `cu_seqlens` — it processed
the flattened sequence as one chunked recurrence, but the wrapper
reshapes `[B, T, hidden]` → `[1, B*T, hidden]` and `short_conv`
resets state at doc boundaries. The kernel kept the previous
doc's accumulated `h_start` into a chunk whose q/k/v are FRESH,
producing a stale `p = (k * exp(g_cum))^T h_start` and an
`(I+T)^{-1}` solve that mismatches `v` against an unrelated
state. With the packer cu_seqlens pattern (T=4096, ~30 doc
starts at specific chunk-aligned positions), the error amplified
across 384 chunks until grads were inf.

- Fix: with `chunk_size=64` and `pack_chunk_aligned` (offsets are
  multiples of `chunk_size`), the fix is a single
  `h = torch.zeros(...)` at every chunk that starts a new doc —
  pre-computed as `_doc_start_chunks: set[int]` once on the
  Python side (one CPU sync at kernel entry, then O(1) set lookup
  per chunk).
- See the kernel docstring for the full contract.

## Fix 4 — tail-pad state-leak (the THIRD inf-grad bug, found 2026-06-16)

The FFD packer produces `cu_seqlens` ending at the END of the
LAST doc, not at `B*T`. If the last pack isn't full (e.g. last
pack only holds docs totaling 3968 tokens out of 4096, leaving
128 tail-pad tokens), the kernel processes the tail-pad chunks
with stale `h` from the last real doc. The tail-pad's
`p = (k * g_cum.exp()).T @ h` reads the (large) stale `h`, the
`(I+T)^{-1}` solve amplifies, and the autograd graph carries NaN
into the FusedLCE.

- Fix: in `pack_chunk_aligned` (`src/training/data/collate.py`),
  ALWAYS append `n_packs * seq_len` to `cu_seqlens_list` as the
  final entry — even when the last real doc ends before `B*T`.
  This marks the start of the tail-pad "doc" and the kernel
  resets `h` there. The padding chunks' outputs are masked out
  by FusedLCE (labels are -100) so the loss is unaffected, but
  the autograd graph is now clean.
- Symptom was: `[DIAG] hidden_states has inf/nan! max_abs=nan`
  after the final norm, with all params at 100% non-finite in
  the post-accumulate-grad hook. This bug only triggered in
  train mode (not eval / no_grad) and only when
  `cu_seqlens[-1] < B*T`.

## Why the PyTorch ref is stable at the model's operating point

The model feeds `g = -A_log.exp() * softplus(f + dt_bias)` into
the kernel. With `A_log ∈ [log 1, log 16]` and `dt_bias` from
log-uniform [0.001, 0.1], g typically lands in (-20, -0.5) with
mean ~-8. The strong decay (g_cum at T=4096 ≈ -13000) suppresses
long-range interaction terms, and the `(I+T)^{-1}` solve stays
well-conditioned. This is the dominant stability factor — not the
chunk size or the solve method.

- Tests with non-realistic g (g~0.5) show the expected
  exponential blow-up (T=1024 → inf); the model's actual input
  is well within the kernel's stable operating regime.

## Production wiring (since 2026-06-17)

The KDA kernel is the production attention. The vendored FLA
library's chunk_kda is the active implementation. EFKDA was
absorbed into the KDA vendored tree as the kernel contract; the
separate `efkda-exp` branch is no longer the production path.

The KDA Triton bwd is HARDWARE-BLOCKED on the 5060 Ti (99KB
shmem limit). The path to <2s/microbatch is the `[BS, BS, BK]`
sub-tile approach in the bwd prep kernel (production
`src/models/ops/_vendored/fla/ops/kda/chunk_intra.py`); the fwd
stays monolithic.

## K=128 compile-time cost is a real bottleneck

The Triton kernel's K=128 specialization triggers a `ptxas`
compile. With the K-tiling rewrite (BK=16, NC=8 tiles for the
`[L, L, K]` decay tensors; intermediates are `[64, 64, 16]` per
tile instead of `[64, 64, 128]`) and `num_warps=8` for K≥128,
the fresh compile is ~3.1 min on the dev box (down from 15+ min
pre-tiling).

- The first training step after a fresh Triton import will block
  on this compile; subsequent steps hit the cache (~3.2 ms).
- Plan training runs accordingly (or warm the cache via the bench
  first).
- For fast iteration during model code changes, the production
  path has no "ref" toggle — KDA is the only kernel. The PyTorch
  ref is the correctness oracle in tests, not a runtime switch.

## 3-kernel split attempt (DID NOT LAND)

Tried the FLA-style multi-kernel split (fwd_prep / dhu_bwd /
intra_bwd / grad_fixup as separate Triton kernels) to fix K=128
compile timeout. Two variants tested:

- **v2 (per-K-tile specialization via K_IDX constexpr):** K=16
  correct, K=32 correct, K=64 ~3 min × 4 specializations = 12+
  min total, K=128 OOM/slow.
- **v3 (runtime K-tile loop with NC constexpr, single
  specialization):** K=16 correct (1.15 ms/call, relerr ~1e-6),
  K=64 18+ min and didn't finish, K=128 shmem OOM.

Why they failed at K=128: the 3-kernel split re-introduced
full-tile loads (`[L, BK]` per K-iter) in `intra_bwd` for grad_o,
grad_h_new, grad_p, grad_T, grad_Q_dot_K, v_new, k_c, g_c, g_cum
— 9 live variables × `[L=64, BK=16]` × 4B = 144 KB+ at K=64,
192 KB at K=128. The `[BS, BS, BK]` sub-tile rewrite keeps
per-iter shmem at ~17 KB precisely to avoid this. The split's v3
runtime K-tile loop didn't help because the body is still too
big; only splitting the K-iter from the L-iter would help, and
that's what the sub-tile approach already does.

**Conclusion:** the 3-kernel split is a dead end for K=128. The
sub-tile rewrite is the correct path forward. The FLA library
solves this differently (multi-kernel split with smaller per-
kernel work), but FLA's bwd body is much simpler (no per-ti
mask matmuls) — the FLA pattern doesn't apply directly here.

## Test/bench infrastructure (auto-enables when the Triton kernel lands)

- `test/test_efkda_triton_correctness.py` (was `_tmp`, now
  promoted) — 8 test cases (fp32/bf16 fwd+bwd, g_scale stress,
  initial_state, chunk_size sweep, realistic shape).
- `test/test_efkda_triton_bench.py` (was `_tmp`, now promoted) —
  12-config sweep at fp32/bf16, `--quick` for fast iteration.

The test refactor to triton-only mode (2026-06-17) defaulted to
running the Triton kernel twice for determinism + once for bwd,
verifies finiteness, and is ~2x faster than the full
PyTorch-ref-comparison path.
