# HippoLM Project Design

## Overview

HippoLM is a research project exploring LLM training and inference
on a custom architecture. The architecture combines a fast linear
attention variant (KDA — Kimi Delta Attention) with a learned
residual-aggregation scheme over block-level representations.

**KDA is the stable attention mechanism in production** (since
June 2026). The earlier GDN2 (Gated DeltaNet 2) implementation
has been fully removed — it was an experimental predecessor that
the project standardised away from after the KDA reference
implementation reached production-grade numerical stability. See
:mod:`docs.kda_kernel_structure` for the kernel-level design and
the project's auto-memory for the historical EFKDA debugging
notes that informed the production pin.

The current design is intentionally experimental — see *Notes*
at the bottom. The codebase is expected to change often, including
the core model code, optimizer layout, and training schedule.
Treat any specific architectural claim in the code with
skepticism; the project description below is intentionally vague
to discourage coupling downstream work to today's design.

## Architecture (informal)

The model is a transformer-style stack with a global linear
attention (KDA, replacing standard softmax attention) and a
block-level softmax attention replacing the standard residual
connection between adjacent layers. The combined effect is a
hybrid: intra-block behaves like a standard residual stack,
inter-block routes via a learned attention over block summaries.
RMSNorm is used throughout (no LayerNorm). No positional encoding
by design.

Configuration is driven by :class:`src.models.config.HippoConfig` —
that is the canonical place to read current dimensions, head
counts, block sizes, and precision flags.

## Training

- Hardware target: 8x V100-class 16G GPUs or single 5060 Ti 16G
  GPU, tight VRAM budget — most optimization work targets the
  boundary, not the center.
- Training is data-parallel-style across the TP group: forward
  and backward on GPU, optimizer state on CPU pinned memory
  (BF16 for AdamW, int8-quantized momentum for Muon).
- Streaming data path: ModelScope (Aliyun CDN) preferred, with
  a local parquet cache and a HF mirror fallback. The fallback
  activates only on real network errors.
- Tokenizer lives at ``src/tokenizer/`` (vendored; not a download).
- The CLI entry point is ``scripts/train.py``. Multiple test
  configurations live under ``configs/test/`` (each one a
  standalone e2e scenario — no separate ``--profile`` flag).
  The yml overlay at ``configs/base.yml`` is the canonical run
  configuration; command line flags exist for overrides only.
- Tensor parallelism is exercised through the Megatron-style
  column/row parallel pattern. A single-GPU "TP simulation" mode
  (``--tp_sim``) exists for development on smaller boxes; the
  sharding math is identical to the real path, only the transport
  changes (NCCL → gloo, N processes pinned to one device).

## Evaluation

- Primary metric and eval library are intentionally left vague
  here; check :mod:`scripts.eval` (or whatever lives there at
  the time of reading) for the current contract.

## CUDA Kernels

Custom kernels (when present) live under ``src/models/ops/cuda/``.
A kernel must have a numerical-correctness test and a benchmark
before it lands.

## Development Roadmap

The roadmap is intentionally not pinned to version numbers. The
project's research nature means feature priorities are revisited
often, and the multimodal extensions (vision, audio, embodiment)
make any semver-based contract meaningless — they would freeze
research directions into shipping promises.

Current focus areas (in approximate order of priority):

- Architecture iteration on the attention + block aggregation
  hybrid (KDA + AttnRes at block boundaries).
- CUDA kernel work for the KDA op where the Triton reference
  has a perf gap (see :mod:`docs.kda_kernel_structure` for
  the call chain and the project's auto-memory for the
  performance target history).
- Scaling training to longer contexts and more tokens.

Anything else (multimodal, inference backend, production
release) is speculative and uncommitted.

## Directory Structure

```
HippoLM/
├── src/
│   ├── models/                  # Model code (config, layers, TP)
│   │   ├── config.py            # HippoConfig (single source of truth for dims)
│   │   ├── model.py             # HippoModel (the plain single-GPU model)
│   │   ├── tp_model/            # TPHippoModel package (split 2026-06)
│   │   │   ├── embed.py         # TPShardedEmbed (vocab-parallel embed)
│   │   │   ├── swiglu.py        # TPSwiGLU (column-row parallel FFN)
│   │   │   ├── kda.py           # TPKDA (TP-sharded Kimi Delta Attention)
│   │   │   ├── lm_head.py       # TPFusedLceLoss (training) + TPLmHead (inference)
│   │   │   ├── layer.py         # TPHippoLayer (one residual block)
│   │   │   └── model.py         # TPHippoModel (the TP orchestrator)
│   │   └── ops/
│   │       ├── _vendored/       # Vendored flash-linear-attention (KDA + utils)
│   │       └── cuda/            # Custom CUDA / Triton kernels (see *CUDA Kernels* above)
│   ├── training/                # Training utilities
│   │   ├── param_offload.py     # Canonical optimizers (CPU offload)
│   │   ├── checkpoint.py        # Save / load
│   │   ├── diagnostics.py       # Per-step telemetry
│   │   ├── env.py               # HF / NCCL / datasets env pinning
│   │   ├── launch/              # GPU selection, TP simulation
│   │   ├── data/                # Streaming dataset, prefetch, sources
│   │   ├── loop.py              # Per-rank training worker (setup/run/teardown)
│   │   └── tokenizer.py         # Local Qwen3.5 tokenizer loader
│   ├── tokenizer/               # Vendored tokenizer files
│   └── inference/               # Reserved path for the inference backend
│                                # (not yet implemented; do not delete)
├── test/                        # Tests (focus on optimizers, streaming
│                                #  data, training-loop utilities; model-
│                                #  layer tests are sparse on purpose)
├── docs/                        # Architecture + kernel design docs
├── scripts/
│   ├── train.py                 # Thin shell: parse args -> call train
│   ├── cli.py                   # argparse setup + YAML overlay
│   └── eval_server.py           # OpenAI-compatible eval backend
├── configs/
│   ├── base.yml                 # Canonical run configuration
│   └── test/                    # One .yml per e2e test scenario
└── output/                      # Smoke runs write here (gitignored)
```

## Notes

- This is a research project; expect frequent pivots. The
  architecture and training recipe are not stable contracts.
- ``src/inference/`` is reserved for the future inference
  backend. It is intentionally empty today; do not remove the
  path.
- Tests focus on optimizers (Muon and AdamW across precisions),
  streaming data plumbing (parquet cache eviction, packer
  alignment), and training-loop utilities (WSD LR schedule,
  grad-norm clip). Model-layer tests are intentionally sparse
  because the model code iterates quickly; correctness is
  validated end-to-end via the smoke-test step log and the
  ``configs/test/*.yml`` scenarios.
- "All changes need to be tested in production" is the rule
  for any refactor that touches the training loop, the
  optimizer layout, the data path, or the model. Use the
  smoke-test command in the project's hardware-target memory
  entry and confirm checkpoint state_dict is non-empty after
  the change.
