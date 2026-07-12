---
name: run-smoke-test
description: Run the canonical smoke test for any change that touches the training loop, optimizer layout, data path, or model code. Use when the user asks to "smoke test", "smoke-run", "verify the change runs", "run the canonical smoke test", or any time a model / loop / opt / data-path commit is about to happen. PASS/FAIL gate: step log shows non-empty checkpoint state_dict AND peak resident ≤ 5060 Ti 16 GB ceiling.
---

# Run Smoke Test

The canonical end-to-end check for model/loop/optimizer/data-path
changes. Runs a 4-step training pass on a single GPU using TP
simulation mode (exercises all sharded forward / all-reduce paths
on one device), against a 5060 Ti 16 GB ceiling.

The command lives here as a single source of truth — auto-memory
`project_hardware.md` carries the same command for cross-conversation
recall, but **this file is what an agent should run**.

## Command

```
python scripts/train.py --config configs/base.yml --use_dummy_data \
  --max_steps 4 --tp_sim --tp_size 2 --gradient_accumulation_steps 2 \
  --batch_size 2 --seq_len 512 --num_layers 4 --num_blocks 2
```

Run from the repo root (`/hy-tmp/HippoLM/`).

## PASS / FAIL gates

A run **PASSES** iff **all three** of these hold:

1. **No exceptions** during `setup` / `run` / `teardown` (run
   completes all 4 steps).
2. **Checkpoint is non-empty** — the step log shows
   `state_dict has N entries` for `N > 0`. A `state_dict = {}`
   line (or an empty `meta` block in the saved file) is a FAIL
   even if no exception fired; this catches state-dict-graph
   regressions that produce silent zero-grad output (the
   `TPHippoModel` `nn.ModuleDict` fix of 2026-06, see auto-memory
   `project_tp_state_dict_bug.md`).
3. **Peak resident ≤ 16 GB** on the 5060 Ti — measured via
   `torch.cuda.max_memory_allocated()`. The 16 GB ceiling is the
   load-bearing constraint for any new GPU-state-adding change.

If (3) fails, do not commit even if (1)/(2) pass — it means the
change pushes a new state tensor onto HWM. Profile with
`test/_tmp/test_benchmark_training.py` and reduce.

## Reading the output

Key lines to find (one per step):

- `setup_mem done` (or whichever marker `src/training/diagnostics.py`
  emits for peak-allocation log) → compare against last-known
  baseline before your change.
- `state_dict has N entries` → that's (2).
- `step_loss=...` (last printed line) → finite value confirms
  bwd path also ran (not just fwd).

If the run dies before step 4, dump the full traceback — most
issues that touch this stack surface as one of:
`OOM` (reduce `--batch_size` or `--num_layers`), `NaN loss`
(promote to `feedback_verify_correctness.md` + run
`kda-correctness-sweep`), or `state_dict empty` (likely a TP
sharding graph regression — see auto-memory
`project_tp_state_dict_bug.md`).

## Smoke vs integration vs unit

- **Smoke (this skill)** — runs the full stack end-to-end on one
  device, ~few minutes. Use before any model/loop/opt/data-path
  commit.
- **Unit** — pytest in `test/`. Run for specific functions
  (optimizer, streamer, packer, grad norm, WSD schedule).
- **Integration** — `configs/test/*.yml` scenarios run via
  `scripts/train.py --config configs/test/<scenario>.yml`.
  Longer than smoke; use when a `configs/test/*` scenario
  exercises the touched code path.
