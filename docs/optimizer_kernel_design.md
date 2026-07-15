# Optimizer-side kernel fusion design

> **Skill**: [`step-perf-remeasure`](../.claude/skills/step-perf-remeasure/SKILL.md) — re-measure step time after any change to the opt-phase path. **Rule**: [`saved-tensors-not-hwm`](../.claude/rules/saved-tensors-not-hwm.md) — perf claims must cite `max_memory_allocated()`, not `saved_tensors`.

This document is the design note for the **CPU-side fused-kernel
work** in `src/training/param_offload/`. It is the sibling of
[`docs/optimizer_layout.md`](optimizer_layout.md) — that doc
covers *which* params go to which optimizer; this one covers
*how* the per-step opt phase is implemented. Read both before
touching either.

## What this design is optimizing for

The opt phase (`infnan_ms + gradnorm_ms + muon_ms + adamw_ms +
zero_ms + repack_ms` in the `[mb-time]` log) was 53% of step
wall time at 2026-07-13 (see
[`project_opt_phase_breakdown.md`](../.claude/skills/step-perf-remeasure/SKILL.md#see-also));
it shrank to ~15% at 2026-07-14 after the fusion work below
shipped (see `project_step_breakdown_2026_07_14.md`). The
remaining 80% is `compute_ms` (the model itself). The opt-side
fusion work is **done in steady state** — future per-step wins
are dominated by the model kernel side (KDA + AttnRes + FFN),
not the opt phase.

The fusion work converged on **four recurring patterns** that
all live in the opt-phase hot path. Each shipped fix is one
instance of a pattern; the patterns are what to copy when
adding a new fused op.

## Pattern 1: DeepSpeed-CPUAdam-style fused C++/OpenMP kernels

The template: PyTorch's per-op internal OMP parallelizes within
a single op but **not across ops**. A Python loop of `N`
`tgt.add_(slice)` calls serializes across ops while
parallelizing within each — at the per-param BF16 pinned
distribution the cross-op dispatch overhead alone is measurable.
The fix is one C++ call with OMP `parallel for` across ops.

### Kernels shipped

| Kernel | File | Role | When fired |
|---|---|---|---|
| `fused_add_into_many` | `src/training/param_offload/cpu_fused.py` | BF16 `tgt.add_(src)` across N tensors, scalar/AVX-2/AVX-512 paths | async worker drain (post-2026-07-13); was per-mb before v4 retired |
| `fused_zero_many` | same | BF16 `tgt.zero_()` across N tensors | end-of-step `zero_cpu_grad_accum` + `CPUMuon.step` end-of-cycle |
| `fused_l2_norm_sq_bf16` | same | OMP-parallel BF16 → FP32 shift-left + square + accumulate into single FP64 scalar | `_compute_and_clip_grad_norm` end-of-step |
| `fused_scale_many_bf16` | same | scalar × BF16 tensor across N tensors, scalar/AVX-2/AVX-512 paths | `_compute_and_clip_grad_norm` clip (replaces per-param `_scale_accum`) |
| `fused_adam_step_bf16` | same | AdamW step (β1 EMA + β2 EMA + bias-corrected denom + step) across N tensors in one OMP pass | `CPUAdamW.step()` end-of-cycle |

All five are JIT-compiled via
`torch.utils.cpp_extension.load_inline` on first use, cached
under `_fused_ext_build/`. Each falls back to per-op Python
loops if no C++ toolchain at runtime (correctness preserved,
only speed lost). Set `HIPPO_FUSED_FORCE_PYLOOP=1` to force
the fallback for debugging.

### Why one C++ call beats a Python loop

PyTorch's per-op internal OMP parallelizes within a single op
but not across ops. A loop of N `tgt.add_(slice)` calls
serializes across ops while parallelizing within each — at the
per-param BF16 pinned distribution the cross-op Python+dispatch
overhead alone is ~1 ms/layer. The fused kernel does all N ops
in ONE C++ call with OMP parallel across ops. Same trick
DeepSpeed's CPUAdam uses (fused C++ single-pass with OpenMP
`parallel for` across cores). The bandwidth-bound nature of
pinned-memory zero/add/scale benefits identically from
multi-core parallel BW.

### Numerical correctness

**BF16 SIMD add must never use integer add on the BF16 bit
pattern.** Integer add is only correct for same-exponent BF16
values; for mixed exponents it produces wrong results. The
correct pattern is FP32 promote + add + RNE-narrow (see auto-
memory `feedback_bf16_int_add_wrong.md`). All five kernels
above follow this pattern.

## Pattern 2: async CPU-add worker (overlap D2H with fwd)

Before 2026-07-13, `accumulate_grads_to_cpu` did a global
`cuda.synchronize()` to wait for per-chunk D2H, blocking the
main thread on the next chunk's fwd. The async worker decouples
them: a daemon thread drains a queue of `(event, target, src)`
entries in the background, where each entry has its own
per-stream CUDA event so the worker only waits for **that
specific** DMA — not unrelated GPU work.

- Started by `_start_cpu_add_worker()` in `_setup_worker`.
- Stopped by `_stop_cpu_add_worker()` in `_teardown_worker`
  (always runs via `finally` clause in `_run_training_loop`).
- End of step: `_drain_cpu_add_queue()` joins the worker so the
  per-param accumulators are consistent before inf/nan,
  grad-norm, optimizer step.

The legacy sync mode is preserved for unit tests (e.g.
`test_grad_offload_hook.py`); public API of `flush_pending_grads`
/ `accumulate_grads_to_cpu` / `flush_manual_flush_params`
unchanged.

### Per-entry event sync matters

A global `cuda.synchronize()` in the worker would block the
main thread's next-chunk fwd until the worker finishes —
defeats the overlap. Per-entry events let the main thread
launch chunk `i+1`'s fwd immediately while the worker drains
chunk `i`.

### Measurement at base.yml prod shape (32 layers, NVFP4 mode-3)

| Component | Pre-worker | Post-worker |
|---|---:|---:|
| compute_ms | 26.6 s | 26.6 s |
| drain_ms | n/a | 0.3 ms (worker kept up) |
| gradnorm_ms | 2.3 s | 1.1 s |
| muon_ms | 7.3 s | 2.7 s |
| adamw_ms | 2.1 s | 1.4 s |
| **total** | **39.4 s** | **32.3 s** |

The drain_ms itself is tiny because the worker keeps up; the
real savings come from `gradnorm_ms` halving (cache-locality:
worker touches the accumulators immediately after the per-
chunk push) and `muon_ms` dropping by ~4.6 s (GPU copy engine
now overlaps D2H with the next chunk's fwd NS).

### Latent bug fixed incidentally

Commit `0473912 feat(nvfp4): ship mode-3 (no BF16 master)`
removed the per-chunk `flush_pending_grads` call from
`_run_training_loop` and added a comment claiming
`accumulate_grads_to_cpu` drains `_pending_grads`. The drain
was **never actually wired up** in `accumulate_grads_to_cpu` —
the streaming-hook list was leaked in production. Smoke test
passed because it only checks `model_state_dict is non-empty`
and `loss is finite`, not that grads actually update the
model. The async worker path incidentally fixes this: hooks
push to the worker queue, the worker drains, end-of-step
`_drain_cpu_add_queue()` joins.

## Pattern 3: in-backward D2H callback (overlap with bwd kernels)

When a custom autograd `Function` produces a non-leaf grad that
the optimizer wants to consume off GPU, the naive path is:
1. `Function.backward()` returns the grad
2. `accumulate_grads_to_cpu` loop drains it after `backward()`
   returns
3. CPU work sits on the critical path between chunks

The in-backward pattern (shipped 2026-07-15 for NVFP4 mode-3):
install a callback on the module at `register_*_module` time.
The callback fires **inside** the Function's backward (during
`_stash_grad_w`), while autograd is still walking the graph
backward. Cast, D2H, record event, enqueue to the async
CPU-add worker — all happening in the bwd critical path so
subsequent layers' bwd kernels overlap with the D2H PCIe
traffic.

### Worker-status-aware fallback

Unit tests for the custom Function drive `_stash_grad_w` then
call `accumulate_grads_to_cpu([opt])` directly — no async
worker running. Without the fallback the callback would enqueue
to a dead worker and the test's CPU buffer would never see the
grad. Solution: when the worker is `None`, the callback falls
back to stashing on `module._latest_grad_w` exactly like the
original code. Then `accumulate_grads_to_cpu` skip condition
becomes "stash is None", not "callback is set". Result: zero
test changes, prod path gets the full speedup.

### Measurement at dev shape (4 layers, n_chunks=8)

| Metric | Before | After |
|---|---:|---:|
| `accumulate_grads_to_cpu` median / chunk | 23.52 ms | 0.11 ms |
| `accumulate_grads_to_cpu` total / step | 209.5 ms | 0.82 ms |
| Step time mean | 1229 ms | 963 ms |
| Step time delta | — | **-21.7%** |
| Peak resident (HWM) | 4.27 GB | 4.27 GB |
| Smoke test (4/4 steps finite) | ✓ | ✓ |

## Pattern 4: Triton fused kernels for hot per-op paths

When a single PyTorch op dominates a sub-component and the
op's HBM traffic is wasteful, fuse it into a Triton kernel.
Two shipped examples:

### `quantize_pack` (NVFP4 E2M1 pack)

The PyTorch path materializes a `[numel, 8]` float32 distance
matrix (~4× the input size) then runs `argmin` over 8 E2M1
magnitudes. Memory-bound. The Triton kernel reads bf16 +
block_scale_raw once, fuses divide + 7 compares + sign + nibble
pack into one streaming pass with no intermediate. Per-chunk
wall-clock on 5060 Ti (4096×1536 bf16): **4.4 ms → 0.57 ms
(7.7×)**; (4096, 4096) → 9×. Per-step savings: ~730 ms
(~1.2% of step).

**Byte-exact via `tl.div_rn` + strict `>` at every cascade
boundary** (matches PyTorch argmin's first-minimum tie-
breaking; Triton's default `/` uses `__fdividef` fast-math
which is ~2 ULP error and fails at boundary ties like
abs_x = 2.5).

**Did not fuse absmax into the same kernel** — the simple
kernel already reads bf16 once; fusing absmax would save ~50%
HBM but adds a small 16-element reduction. Current 30 GB/s is
~3% of HBM peak on the 5060 Ti; not justified.

See `src/models/ops/nvfp4_quant_triton.py`.

## Why the per-step grad accumulator must stay on CPU

This is the load-bearing constraint that the entire design
rests on. The user phrased it as: "不可避免的原因是因为为了
训练稳定必须保留 grad clip". The grad-norm clip MUST be
preserved for training stability (no equivalent substitute is
acceptable), and the cheapest substrate that makes the clip
pipeline feasible at production scale (624 M params, ~2.5 GB
of CPU grad state, 1 TP all-reduce per step) is a CPU-side
grad accumulator. Three reasons:

1. **Single fused pass for the L2 norm.** The end-of-step
   `_compute_and_clip_grad_norm` walks *all* per-param `s.grad`
   tensors in one OMP `parallel for` via
   `fused_l2_norm_sq_bf16`. If the accumulator lived on the GPU
   we'd need a per-param `.pow(2).sum().item()` loop that
   `.item()`-syncs the GPU once per param — at ~624 M params
   that's ~5 s/step of pure sync stalls.

2. **TP all-reduce on a scalar, not per-tensor.** The clip
   threshold `max_grad_norm` is a property of the *global*
   gradient. Doing the reduction once on a 1-element FP64 tensor
   is one NCCL/gloo call; doing it per-param would be `n_params`
   collectives.

3. **In-place clip after the reduction.** Once we have the
   global L2 norm, the clip coefficient is
   `max_norm / (total_norm + eps)` and we apply it to every
   per-param `s.grad` in place via `fused_scale_many_bf16` —
   one OMP pass across all `s.grad` tensors (same fused-
   bandwidth trick). Doing this on the GPU would require
   `n_params` separate kernel launches (no fused kernel exists
   for the in-place scale of N heterogeneous tensors on the GPU).

The full rationale lives in
[`docs/optimizer_layout.md`](optimizer_layout.md) →
*Why the per-step grad accumulator must live on CPU pinned
memory*. Don't propose "move the accumulator to GPU for free
H2D savings" without re-reading those three points first.

## Explicit-accumulator layout (post-2026-07-15)

The per-step grad accumulator (`s.grad`) and the optimizer's
momentum / EMA state (`s.exp_avg`) are **separate, distinct
CPU pinned buffers**, not merged. The previous "merged-
accumulator / mu = 1" trick conflated two semantically distinct
roles (per-step accumulation vs cross-step smoothing) and is
deleted. See `docs/optimizer_layout.md` §*Per-step state
layout (post-2026-07-15)* for the buffer identity table and
per-param VRAM impact.

## Per-step state (current production, post-2026-07-15)

| Buffer | Optimizer | Role | Lifecycle |
|---|---|---|---|
| `s.grad` | both | per-step grad accumulator | zeroed at end of every `step()` (or by `zero_cpu_grad_accum` on `found_inf`) |
| `s.exp_avg` | AdamW | first-moment EMA `β1·prev + (1-β1)·grad` | preserved across steps |
| `s.exp_avg_sq` | AdamW | second-moment EMA `β2·prev + (1-β2)·exp_avg²` | preserved across steps |
| `s.exp_avg` | Muon | SGD momentum `β·prev + grad` (Keller Jordan ref, no `(1-β)` factor) | preserved across steps |

## Files

| File | Role |
|---|---|
| `src/training/param_offload/_state.py` | `_ParamState` dataclass + per-state helpers (post-2026-07-15 buffer identity) |
| `src/training/param_offload/offload.py` | streaming hook + async worker + `accumulate_grads_to_cpu` (post-2026-07-15 callback install + worker-status-aware fallback) |
| `src/training/param_offload/cpu_fused.py` | all 5 JIT kernels (`fused_add_into_many`, `fused_zero_many`, `fused_l2_norm_sq_bf16`, `fused_scale_many_bf16`, `fused_adam_step_bf16`) |
| `src/training/param_offload/adamw.py` | `CPUAdamW.step()` |
| `src/training/param_offload/muon.py` | `CPUMuon.step()` (NS on GPU + CPU fused add) |
| `src/training/param_offload/param_offload.py` | `build_param_groups` (1-D → AdamW, 2-D → Muon, 3-D → AdamW; routing in `docs/optimizer_layout.md`) |
| `src/models/ops/nvfp4_quant_triton.py` | `_quantize_pack` Triton kernel |
| `src/training/loop/grad_norm.py` | `_compute_and_clip_grad_norm` + the rationale module docstring |

## How to extend

When adding a new opt-phase fused kernel:

1. **First identify which Pattern (1-4) above fits.** Don't
   invent a new template — copy the matching pattern.
2. **Measure before + after with the
   [`step-perf-remeasure`](../.claude/skills/step-perf-remeasure/SKILL.md)
   skill.** The opt-phase is no longer the elephant (compute
   is), so a fused kernel that wins 50 ms on opt_ms = 0.15%
   step time. Pick targets where the win is ≥ 1% step time.
3. **BF16 SIMD correctness:** FP32 promote + add + RNE-narrow
   only (auto-memory `feedback_bf16_int_add_wrong.md`).
4. **HWM invariant:** use the HWM probe per
   [`saved-tensors-not-hwm`](../.claude/rules/saved-tensors-not-hwm.md)
   — a fused kernel that wins time but loses HWM is a FAIL.
5. **JIT fallback:** every fused kernel must have a
   per-op Python fallback for environments without a C++
   toolchain. Correctness preserved, only speed lost.

When adding a new custom autograd `Function` that produces a
non-leaf grad the optimizer wants to consume off GPU, follow
**Pattern 3** (in-backward D2H callback) instead of accumulating
in Python after `backward()` returns. The Python work sits on
the critical path between chunks; the in-backward callback
overlaps with subsequent bwd kernels.

## See also

- [`docs/optimizer_layout.md`](optimizer_layout.md) — *which*
  params go to which optimizer (1-D → AdamW, 2-D → Muon, 3-D
  → AdamW) + the buffer identity table + the grad-acc CPU
  rationale.
- [`docs/vram_debugging.md`](vram_debugging.md) — VRAM
  breakdown methodology (HWM probe).
- [`docs/nvfp4_ffn_design.md`](nvfp4_ffn_design.md) — NVFP4
  W4A16 FFN design (Pattern 4's `quantize_pack` lives here).
- Auto-memory: `project_cpu_add_jit.md`, `project_async_cpu_add.md`,
  `project_inback_d2h.md`, `project_triton_quant_pack.md`,
  `project_fused_scale_clip.md`, `project_explicit_accum_layout.md`
  — individual ship writeups (this doc is the consolidated view).
- Auto-memory `project_step_breakdown_2026_07_14.md` — the
  re-measure that shrank opt phase from 53% to 15%.
- Auto-memory `project_opt_phase_breakdown.md` — pre-fusion
  baseline (infnan / gradnorm / muon / adamw / zero split).
- Auto-memory `feedback_bf16_int_add_wrong.md` — why BF16
  SIMD add must FP32-promote, never integer-add the bit
  pattern.