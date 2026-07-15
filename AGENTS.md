# HippoLM Agent Instructions

This file is the project orientation for AI coding agents (Claude
Code, etc.). For development workflow conventions, see
[`CONTRIBUTING.md`](CONTRIBUTING.md). Historical incidents and
one-off debugging notes live in the project's auto-memory.

## Overview

HippoLM is a research project exploring LLM training and inference
on a custom architecture. The architecture combines a fast linear
attention variant (KDA — Kimi Delta Attention) with a learned
residual-aggregation scheme over block-level representations.

**KDA is the stable attention mechanism in production** (since
June 2026). The earlier GDN2 (Gated DeltaNet 2) implementation
has been fully removed. See
[`docs/kda_kernel_structure.md`](docs/kda_kernel_structure.md)
for the kernel-level design and the auto-memory for the historical
EFKDA debugging notes that informed the production pin.

The current design is intentionally experimental — see *Notes*
at the bottom. Treat any specific architectural claim in the code
with skepticism; the project description below is intentionally
vague to discourage coupling downstream work to today's design.

## Architecture (informal)

The model is a transformer-style stack with a global linear
attention (KDA, replacing standard softmax attention) and a
block-level softmax attention replacing the standard residual
connection between adjacent layers. The combined effect is a
hybrid: intra-block behaves like a standard residual stack,
inter-block routes via a learned attention over block summaries.
RMSNorm is used throughout (no LayerNorm). No positional encoding
by design.

Configuration is driven by `:class:src.models.config.HippoConfig` —
that is the canonical place to read current dimensions, head
counts, block sizes, and precision flags.

## Training

- **Hardware target**: single RTX 4090 (sm_89, production training)
  or single 5060 Ti 16G (sm_120, dev box). V100 (sm_70) was
  dropped on 2026-07-08 — Marlin FP4 + W4A16 NVFP4 FFN both
  require sm_80+ BF16 MMA, which V100 lacks. For the smoke-test
  command and the 16 GB ceiling constraint, see the auto-memory
  `project_hardware.md`. (See also `.claude/rules/dont-target-v100.md`.)
- Training is data-parallel-style across the TP group: forward
  and backward on GPU, optimizer state on CPU pinned memory
  (BF16 for both AdamW and Muon; int8/mxfp8 Muon storage was
  removed on 2026-07-12).
- Streaming data path: ModelScope (Aliyun CDN) preferred, with a
  local parquet cache and a HF mirror fallback. The fallback
  activates only on real network errors. See
  [`docs/cache_routing.md`](docs/cache_routing.md).
- Tokenizer lives at `src/tokenizer/` (vendored; not a download).
- The CLI entry point is `scripts/train.py`. Multiple test
  configurations live under `configs/test/` (each one a standalone
  e2e scenario — no separate `--profile` flag). The yml overlay at
  `configs/base.yml` is the canonical run configuration; command
  line flags exist for overrides only. See
  `configs/test/README.md`.
- Tensor parallelism is exercised through the Megatron-style
  column/row parallel pattern. A single-GPU "TP simulation" mode
  (`--tp_sim`) exists for development on smaller boxes; the
  sharding math is identical to the real path, only the transport
  changes (NCCL → gloo, N processes pinned to one device).

## Evaluation

Primary metric and eval library are intentionally left vague here;
check `scripts/eval_server.py` for the current contract.

## CUDA Kernels

Custom kernels (when present) live under `src/models/ops/cuda/`.
A kernel must have a numerical-correctness test and a benchmark
before it lands. Triton patterns live in
[`docs/triton_kernel_playbook.md`](docs/triton_kernel_playbook.md);
Marlin FP4 build and runtime notes live in
[`docs/marlin_build_pipeline.md`](docs/marlin_build_pipeline.md).
For the correctness gate when sweeping kernel parameters, see
[`.claude/skills/kda-correctness-sweep/SKILL.md`](.claude/skills/kda-correctness-sweep/SKILL.md).

## Development Roadmap

The roadmap is intentionally not pinned to version numbers.
Current focus areas (in approximate order of priority):

- Architecture iteration on the attention + block aggregation
  hybrid (KDA + AttnRes at block boundaries).
- CUDA kernel work for the KDA op where the Triton reference
  has a perf gap.
- Scaling training to longer contexts and more tokens.

Anything else (multimodal, inference backend, production release)
is speculative and uncommitted.

## Directory Structure

```
HippoLM/
├── src/
│   ├── models/             # Model code (config, layers, TP). See src/models/ for layout.
│   ├── training/           # Per-rank loop, optimizers, ckpt, data, diagnostics, env
│   ├── tokenizer/          # Vendored tokenizer files
│   └── inference/          # Reserved path (do not delete; not yet implemented)
├── test/                   # Regression tests (focus: optimizers, data, loop utilities)
├── docs/                   # Architecture + kernel design docs (one .md per topic)
├── scripts/                # train.py / cli.py / eval_server.py
├── configs/
│   ├── base.yml            # Canonical run configuration
│   └── test/               # One .yml per e2e test scenario
├── .claude/
│   ├── skills/             # Repo-local task workflows (indexed below)
│   └── rules/              # Auto-injected scoped coding rules (indexed below)
└── output/                 # Smoke runs write here (gitignored)
```

The model sub-package layout is documented in source — `ls` of
`src/models/` and `src/training/` is the right way to find current
files.

## Skills and rules

Workflow files in `.claude/skills/` (manually triggered by
description match) and `.claude/rules/` (auto-injected when their
`paths:` glob matches a file being edited):

| Trigger | Resource | Purpose |
|---|---|---|
| About to commit a model / loop / optimizer / data-path change | [`.claude/skills/run-smoke-test/SKILL.md`](.claude/skills/run-smoke-test/SKILL.md) | Smoke-test command + PASS/FAIL gate |
| Sweeping a Triton / CUDA / Marlin kernel config for perf | [`.claude/skills/kda-correctness-sweep/SKILL.md`](.claude/skills/kda-correctness-sweep/SKILL.md) | Correctness gate before reporting speed |
| After a perf change to fused kernel / layout / async worker / autograd callback | [`.claude/skills/step-perf-remeasure/SKILL.md`](.claude/skills/step-perf-remeasure/SKILL.md) | Per-component `[mb-time]` re-measure + HWM gate |
| Editing `src/models/ops/cuda/**` or `configs/**` | [`.claude/rules/dont-target-v100.md`](.claude/rules/dont-target-v100.md) | sm_70 dropped 2026-07-08; don't reintroduce V100 paths |
| Proposing a memory-saving change to a kernel or training path | [`.claude/rules/saved-tensors-not-hwm.md`](.claude/rules/saved-tensors-not-hwm.md) | `saved_tensors` from autograd hooks ≠ peak resident; use the HWM probe |
| Editing `src/models/ops/**` (passing `stride_*` tensors to Triton kernels) | [`.claude/rules/einsum-noncontig-triton.md`](.claude/rules/einsum-noncontig-triton.md) | `torch.einsum` output stride ≠ obvious; `.contiguous()` before stride-indexed Triton |
| Deleting a `_legacy` / shim / re-export file | [`.claude/rules/shim-deletion-protocol.md`](.claude/rules/shim-deletion-protocol.md) | Grep absolute + relative + smoke-import before deleting |

For development workflow rules (test-first, smoke-test, autotune
discipline, shim protocol, commits), see
[`CONTRIBUTING.md`](CONTRIBUTING.md).

## Notes

- This is a research project; expect frequent pivots. The
  architecture and training recipe are not stable contracts.
- `src/inference/` is reserved for the future inference backend.
  It is intentionally empty today; do not remove the path.
- Tests focus on optimizers (Muon and AdamW across precisions),
  streaming data plumbing (parquet cache eviction, packer
  alignment), and training-loop utilities (WSD LR schedule,
  grad-norm clip). Model-layer tests are intentionally sparse
  because the model code iterates quickly; correctness is validated
  end-to-end via the smoke-test step log and the
  `configs/test/*.yml` scenarios.
