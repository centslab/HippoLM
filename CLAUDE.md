# HippoLM Project Design

## Overview

HippoLM is a research project exploring LLM training and inference
on a custom architecture. The architecture combines a fast linear
attention variant with a learned residual-aggregation scheme over
block-level representations.

The current design is intentionally experimental — see *Notes* at
the bottom. The codebase is expected to change often, including the
core model code, optimizer layout, and training schedule. Treat any
specific architectural claim in the code with skepticism; the project
description below is intentionally vague to discourage coupling
downstream work to today's design.

## Architecture (informal)

The model is a transformer-style stack with a global linear
attention replacing standard softmax attention, and a block-level
softmax attention replacing the standard residual connection
between adjacent layers. The combined effect is a hybrid: intra-block
behaves like a standard residual stack, inter-block routes via a
learned attention over block summaries. RMSNorm is used throughout
(no LayerNorm). No positional encoding by design.

Configuration is driven by :class:`src.models.config.HippoConfig` —
that is the canonical place to read current dimensions, head counts,
block sizes, and precision flags.

## Training

- Hardware target: 8x V100-class 16G GPUs or single 5060 Ti 16G
  GPU, tight VRAM budget — mostoptimization work targets
  the boundary, not the center.
- Training is data-parallel-style across the TP group: forward
  and backward on GPU, optimizer state on CPU pinned memory
  (BF16 for AdamW, int8-quantized momentum for Muon).
- Streaming data path: ModelScope (Aliyun CDN) preferred, with
  a local parquet cache and a HF mirror fallback. The fallback
  activates only on real network errors.
- Tokenizer lives at ``src/tokenizer/`` (vendored; not a download).
- The CLI entry point is ``scripts/train.py``. The yml overlay at
  ``configs/base.yml`` is the canonical run configuration; command
  line flags exist for overrides only.
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

The roadmap is intentionally short and high-level — the project's
research nature means feature priorities are revisited often.

- v0.0.x: foundation (architecture correctness, end-to-end training)
- v0.1.x: small-model training + evaluation
- v0.2.x: custom CUDA kernels
- v0.3.x: scale up
- v1.0.0: production release
- v2.0.0+: multimodal extensions (vision, audio, embodiment)
  — all speculative, design not finalized

## Directory Structure

```
HippoLM/
├── src/
│   ├── models/                  # Model code (config, layers, TP)
│   │   ├── config.py            # HippoConfig (the single source of truth for dims)
│   │   ├── *.py                 # Attention, residual-aggregation, FFN, etc.
│   │   └── ops/
│   │       └── cuda/            # Custom CUDA kernels (see *CUDA Kernels* above)
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
├── test/                        # Tests
├── scripts/
│   ├── train.py                 # Thin shell: parse args -> call train
│   └── cli.py                   # argparse setup + YAML overlay
└── configs/
    └── base.yml                 # Canonical run configuration
```

## Notes

- This is a research project; expect frequent pivots. The
  architecture and training recipe are not stable contracts.
- ``src/inference/`` is reserved for the future inference
  backend. It is intentionally empty today; do not remove the
  path.
- Tests focus on optimizers (Muon and AdamW across precisions).
  Model-layer tests are intentionally sparse because the model
  code iterates quickly; correctness is validated end-to-end
  via the smoke-test step log.
