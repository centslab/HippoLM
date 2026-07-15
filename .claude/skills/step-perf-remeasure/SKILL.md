---
name: step-perf-remeasure
description: Re-measure per-component step time after any perf change (fused kernel, layout refactor, async worker, autograd callback). Use when the user asks "is the change faster", "where's the bottleneck now", "drill into X ms", or before/after any opt-phase fused kernel work. Gate: per-component delta in `[mb-time]` log + HWM unchanged via `max_memory_allocated()` (per `saved-tensors-not-hwm` rule).
---

# Step Perf Re-measure

Per-component step-time breakdown via the `[mb-time]` log in
`src/training/loop/run.py` (gated by `--mb_timing N`). Has been
the standard re-measure loop for the opt-phase work over
2026-07-13 → 2026-07-15 — see `project_step_breakdown*.md`,
`project_opt_phase_breakdown.md`, `project_per_tensor_offload.md`,
`project_chunk_boundary_bubble_source.md` in auto-memory for the
worked examples. The skill itself is just the command + the
PASS/FAIL gates; the interpretation is in the existing memory
entries.

## Command

```
python scripts/train.py --config configs/base.yml --use_dummy_data \
  --max_steps 2 --mb_timing 2 \
  --seq_len 65536 --micro_batch_size 16384 --gradient_accumulation_steps 2
```

Run from the repo root (`/hy-tmp/HippoLM/`). `--mb_timing 2`
prints the per-component breakdown for steps 0 and 1; bump the
trailing integer for more samples. Add `--num_layers 32
--num_blocks 8` to match the base.yml default if you overrode
the smoke shape (the smoke-test skill uses smaller defaults —
those numbers do **not** extrapolate to production).

## Components the log emits

`src/training/loop/run.py:319-455` prints one line per step:

| Component | What it covers |
|---|---|
| `data_ms` | data wait + H2D for the chunk |
| `compute_ms` | fwd+bwd interleaved (the model itself) |
| `drain_ms` | post-chunk `accumulate_grads_to_cpu` + `flush_manual_flush_params` |
| `infnan_ms` | CPU `isfinite+sum` over per-param accumulators (post `project_fused_gil_release` should be ~0) |
| `gradnorm_ms` | `_compute_and_clip_grad_norm` (per-param fused scale clip since 2026-07-14) |
| `muon_ms` | Muon step (NS on GPU + CPU BF16 add) |
| `adamw_ms` | AdamW step (`fused_adam_step_bf16` AVX-2/AVX-512 path) |
| `zero_ms` | `zero_cpu_grad_accum` per-param zero |
| `repack_ms` | NVFP4 `repack_nvfp4_weights` over 64 modules |
| `scaler_ms` | AMP scaler bookkeeping (negligible) |

`compute_ms` scales linearly with `n_chunks`; everything else is
per-step constant. Read **total** at the production `n_chunks`
(the base.yml default is `n_chunks=16`), not the small-shape
extrapolation — see `project_step_breakdown_2026_07_14.md` for
the projection math.

## PASS / FAIL gates

A re-measure **PASSES** iff **all three** of these hold:

1. **The target component moved in the expected direction** —
   the change's goal is achieved in the `[mb-time]` log. (If
   the target moved but another component regressed by ≥10%,
   that's a PARTIAL; the change net-loses if the regression
   exceeds the gain.)
2. **HWM unchanged** — measured via
   `torch.cuda.max_memory_allocated()` per the
   [`saved-tensors-not-hwm`](../rules/saved-tensors-not-hwm.md)
   rule. The 5060 Ti 16 GB ceiling still applies (see the
   [`run-smoke-test`](../skills/run-smoke-test/SKILL.md)
   skill). A VRAM regression in a perf change is an automatic
   FAIL even if step time improved.
3. **Step loss still finite** — `step_loss=...` last line is
   finite. A non-finite loss after a perf change means the
   fused kernel or layout refactor broke numerics; revert and
   run the [`kda-correctness-sweep`](../skills/kda-correctness-sweep/SKILL.md)
   skill.

If (2) fails: do not commit. Profile with a focused probe
under `test/_tmp/` (per the test-first rule in
[`CONTRIBUTING.md`](../../CONTRIBUTING.md)) and reduce.

## Reading the output

Key landmarks in the log:

- `setup_mem done` → initial alloc baseline; compare against
  `torch.cuda.max_memory_allocated()`.
- `step N: data=Xms compute=Yms ... total=Zms` (one line per
  step). Compare `Yms` against the last-known baseline for the
  same shape.
- `state_dict has N entries` → from the smoke-test path; if
  you see it here, the re-measure is exercising the full
  optimizer step too.
- `step_loss=...` → finite value confirms bwd path also ran.

**The compute_ms vs opt_ms ratio matters more than absolute
numbers.** At base.yml prod (`n_chunks=16`) post-2026-07-14
the split is ~80% compute / ~20% opt (see
`project_step_breakdown_2026_07_14.md`). A change that lowers
opt_ms by 50% is a 10% step win; a change that lowers compute_ms
by 10% is also a 10% step win. Don't chase opt-side wins when
the compute phase is the elephant.

## Drill-down (when the target component is still too coarse)

If the target component in `[mb-time]` is itself too coarse to
localize the win (e.g. `muon_ms` is high but the cost is in
NS vs CPU add vs H2D), write a focused probe in `test/_tmp/`
that splits that component further. Patterns:

- `torch.cuda.Event` start/stop around the suspected sub-op.
- Per-tensor Python timing on a list comprehension that calls
  `.item()` once per param (catches `.item()` sync stalls).
- `torch.cuda.synchronize()` only at probe boundaries, never on
  the critical path.

Delete the probe before commit per the test-first rule.

## This skill vs `run-smoke-test`

- **`run-smoke-test`** — runs the full stack end-to-end on one
  device, validates the change doesn't break correctness or
  VRAM. **Use before commit.**
- **`step-perf-remeasure` (this skill)** — measures per-step
  wall time of a specific perf change, validates the change
  delivered the expected step-time win without regressing
  HWM. **Use after any opt-side change, before claiming a win.**
- Both are needed; neither replaces the other.

## See also

- `project_step_breakdown_2026_07_14.md` — worked example:
  baseline numbers + per-component table for the 2026-07-14
  post-fusion state.
- `project_opt_phase_breakdown.md` — opt-phase drill-down
  (infnan / gradnorm / muon / adamw / zero / repack split).
- `project_chunk_boundary_bubble_source.md` — example of
  probe-driven drill-down (per-chunk Python overhead via
  `test/_tmp/probe_per_chunk_python.py`).