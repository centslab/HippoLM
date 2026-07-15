# BlockAttnRes kernel structure

> **Skill**: [`step-perf-remeasure`](../.claude/skills/step-perf-remeasure/SKILL.md) — re-measure step time after any change to the AttnRes path. **Rule**: [`einsum-noncontig-triton`](../.claude/rules/einsum-noncontig-triton.md) — the kernel reads strides as raw `ptr + offset` arithmetic; call `.contiguous()` on every `stride_*` arg.

This document is the design note for the BlockAttnRes path
(shipped as of 2026-07-14). It is the sibling of
[`docs/kda_kernel_structure.md`](kda_kernel_structure.md).
Read both before touching either — the call chain runs through
the model-level residual aggregation, not just the kernel.

## What BlockAttnRes is

BlockAttnRes is the block-level attention-over-residual that
replaces the standard residual connection between adjacent
blocks. Each block boundary emits a hidden state; at the
boundary, a learned per-head query computes a softmax weight
**per head** over the residual sources (the boundary states of
all previous blocks), and the output is the weighted sum. This
sits between the intra-block (standard residual stack) and the
inter-block (learned routing via per-head attention).

The reference for the math is
[`docs/kda_kernel_structure.md`](kda_kernel_structure.md) and
the in-source `BlockAttnRes.forward` /
`_FusedAttnResFn.forward` in `src/models/ops/attn_res.py`.
This document covers the kernel-level design and the design
choices that aren't obvious from the source.

## File layout

| File | Role |
| --- | --- |
| `src/models/ops/attn_res.py` | `BlockAttnRes` (nn.Module) + `_FusedAttnResFn` (autograd `Function`). The orchestration; the only place that calls `save_for_backward`. |
| `src/models/ops/attn_res_triton.py` | forward + backward Triton kernels. Forward: `_online_attnres_kernel` (single-kernel online softmax). Backward: `_attnres_bwd_dv_dqw_kernel` (per (b, t) → `dV` + `dqw_partial` fp32) + `_attnres_bwd_dq_dw_kernel` (per (h, dh) → `dq` + `dw`); two-kernel direct derivation, no re-run. |

The Triton forward is **one kernel** that does the full
RMSNorm + per-head logit + online softmax + weighted-sum in
one pass. The Triton backward is **two kernels**: one
per-(b,t) that produces `dV` (full hidden `D`) + `dqw_partial`
(fp32), and one per-(h, dh) that reduces over `(b, t)` to
`dq` + `dw`.

## Forward: the single-kernel online-softmax design

`_online_attnres_kernel`: one program per `(b, t)` token. Loops
over `N` residual sources (constexpr → unrolled). Per source:
reads `V[n, b, t, :]` **once** as a padded `[BLOCK_HF, BLOCK_DH]`
full-head tile, computes `rstd` over the full hidden, computes
the per-head logit `q · RMSNorm(v)`, and folds both the softmax
normaliser and the weighted-sum accumulator in registers
(online softmax). Writes `out[B, T, H, D_h]` bf16 at the end.
Hardcoded `num_warps=2` (best at prod shape; **production
discipline = no autotune**).

The online-softmax design reads `V` once instead of twice and
keeps the running max / running sum / running weighted-sum in
registers, eliminating the intermediate logits HBM round-trip
the earlier 2-kernel design had.

### Per-call cost at prod shape (N=8, B=1, T=16384, D=1536, H=12, D_h=128) on 5060 Ti

| Path | Time | Speedup vs ref | HBM |
|---|---:|---:|---:|
| Reference (PyTorch 5-op) | ~26 ms | 1.0× | ~3500 MB |
| 2-kernel fusion (RMSNorm-dot + softmax + weighted-sum) | ~2.1 ms | 12× | ~825 MB |
| **Single online kernel (shipped)** | **~1.17 ms** | **22×** | **~430 MB** |

The single-kernel path is also **1.8× faster** than the
2-kernel path that shipped earlier the same day.

### Why FLA's `fused_attnres` is not a drop-in

FLA's `fused_attnres` is **single-head**: its query is `[D]`
and produces ONE scalar logit per residual source per token
(softmax over the residual-source dim gives a single weight
applied to the whole hidden vector). HippoLM's BlockAttnRes is
**multi-head**: query is `[num_heads, head_dim]`, **one
softmax weight per head** over the `N` sources. So FLA's
kernel would change the model's math. We kept HippoLM's
per-head semantics and borrowed FLA's **design**: single
kernel, V read once, online softmax (running max / acc / o in
registers), each V tile used for both the logit and the
weighted sum.

## Backward: two-kernel direct derivation

The shipped `_FusedAttnResFn.backward` does **not** re-run the
forward math under `torch.enable_grad()` (which is what the
PyTorch reference does and which costs O(N · T · D · H) in
PyTorch ops). Instead, it uses two direct-derivation Triton
kernels modeled on FLA's `fused_attnres` design, adapted for
multi-head:

1. **`_attnres_bwd_dv_dqw_kernel`** — per `(b, t)`. Produces
   `dV` (full hidden `D`, bf16) + `dqw_partial` `[B, T, H_LOCAL,
   D_h]` (fp32). Uses the forward-saved `logit[N, B, T, H_LOCAL]`
   (bf16) and `lse[B, T, H_LOCAL]` (fp32) — no extra HBM pass
   for saves.
2. **`_attnres_bwd_dq_dw_kernel`** — per `(h, dh)`. Reduces
   `dqw_partial` over `(b, t)` → `dq` `[H_LOCAL, D_h]` + `dw`
   (local slice of full-hidden `d_norm_weight`).

The forward kernel was extended to write `logit` and `lse` in
the same online-softmax pass; `out` is saved for
`delta = <do, out>` and replaces the `V_full_g` clone (~384
MB/call) the PyTorch re-run path needed. Net HWM per call:
-380 MB (clone dropped) + 50 MB (out saved) + 3 MB (logit)
+ 0.8 MB (lse) ≈ **-330 MB**.

### Backward cost at prod shape

| Path | Time | Speedup vs ref |
|---|---:|---:|
| PyTorch re-run (`torch.enable_grad` over fwd math) | ~136 ms | 1.0× |
| **Two-kernel Triton bwd (shipped)** | **~4.1 ms** | **~33×** |

Total per-step savings from the Triton bwd: **~3-3.5 s out of
33 s ≈ 10% step time** (see
[`project_step_breakdown_2026_07_14.md`](../.claude/skills/step-perf-remeasure/SKILL.md#see-also)
for the step-time context).

## TP > 1: full-head space, local-slice output

Both kernels work in **full-head space**: read the full-hidden
row for the RMSNorm reduction, compute / write only the local
head range `[HS, HS + H_LOCAL)` (query loaded into the local
rows of the full-head tile, others masked). This is correct at
world > 1.

### The latent TP > 1 forward bug fixed in this rev

The OLD 2-kernel path **compile-failed at world > 1**:
`tl.reshape(k_full_D, [next_pow2(hpp), D_h])` had mismatched
element counts (e.g. 2048 → 6·128 = 768 at hpp = 6). The
exception was caught by `fused_attn_res_forward`'s try/except
and silently fell back to PyTorch. So the smoke test
(`tp_size = 2`) never actually ran the Triton path before
this rev; production (`base.yml tp_sim = false → world = 1`)
did. The single kernel runs the fast path at **both** world = 1
and world = 2.

### The latent TP > 1 backward bug fixed in the same rev

Once the forward succeeds at world > 1, the autograd Function's
backward runs and previously did `K.view(N, B, T, H, D_h)` with
`H = hpp` (local) but `K` over full `D` — a shape error. Fixed
to slice `K` to the local head range
(`K.view(N, B, T, D // D_h, D_h)[..., head_start:head_start + H, :]`,
`head_start = slice_start // D_h`), mirroring the PyTorch
fallback. Guarded by
`test_fused_tp_slice_forward_backward[rank=0, 1]` in
`test/test_attn_res_triton.py`.

## Fallback path

When Triton fails (CUDA missing, non-bf16, kernel crash), falls
through to a PyTorch implementation that mirrors the reference
math (norm over full `V`, slice `K` to local heads, einsums).
The fallback uses `tensor.contiguous()` on every stride-arg
input per the [`einsum-noncontig-triton`](../.claude/rules/einsum-noncontig-triton.md)
rule.

## Tests

`test/test_attn_res_triton.py`:
- `test_fused_backward_matches_reference` — 3 shapes including
  prod-ish (8 / 1 / 4096 / 12 / 128); asserts rel < 5% vs
  PyTorch autograd ref.
- `test_fused_tp_slice_forward_backward[rank=0, 1]` — TP = 2
  forward + backward correctness.

The probes (`test/_tmp/probe_attnres_fwd.py`,
`test/_tmp/probe_attnres_bwd.py`) were deleted per the
test-first rule.

## How to extend

- **Any new AttnRes variant** should start from this
  single-kernel online-softmax design. Read `V` once; keep the
  softmax online in registers; work in full-head space so
  TP > 1 stays correct.
- **Don't reintroduce the 2-kernel path** (logits HBM round-trip
  + double `V` read) or the 5-op PyTorch path.
- **When comparing against FLA**, note the single-head vs
  multi-head semantic gap — FLA's kernel is not a drop-in.

## History (superseded, retained for context)

### Variants A / B / C — superseded 2026-07-14 by single-kernel fusion

Per auto-memory `project_attn_res_variants.md` (benchmarked at
2026-07-13, deleted probe per test-first rule):

| Variant | Cost | Verdict |
|---|---:|---|
| A: stack + einsum (current pre-fusion prod) | 1.69 ms/call | baseline |
| B: pre-alloc `[N, B, T, D]` buffer + `copy_` each block | wash | torch caching allocator already amortizes realloc; no end-to-end saving |
| C: zero-query fast path | 0.65 ms/call | 1-step illusion — `query` gets a gradient on first bwd pass and is non-zero forever |
| D: partial-copy across calls (hypothetical) | -0.78 ms/step | < 1% step time, requires API changes to `src/models/tp_model/model.py` lines 639-688 |

The real win was Triton fusion (this document), not any of A /
B / C / D.

## See also

- [`docs/kda_kernel_structure.md`](kda_kernel_structure.md) —
  sibling kernel design doc.
- [`docs/triton_kernel_playbook.md`](triton_kernel_playbook.md)
  §1 (multi-kernel split) — the multi-kernel-split philosophy
  applies to the **backward** (two kernels); the forward is
  intentionally one kernel for the online-softmax design.
- [`docs/vram_debugging.md`](vram_debugging.md) §3.2 — AttnRes
  VRAM breakdown (the "saved" cost of the N boundary stacks).
- Auto-memory `project_attn_res_triton.md` — original ship
  writeup (forward).
- Auto-memory `project_attn_res_bwd.md` — original ship
  writeup (backward).
- Auto-memory `project_attn_res_variants.md` — A / B / C / D
  microbench writeup (now history).