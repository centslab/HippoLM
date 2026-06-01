# HippoLM Project Design

## Overview

HippoLM is a production-grade LLM training and inference project based on a novel architecture combining **Kimi Delta Attention (KDA)** with **Block Attention Residuals (Block AttnRes)** from [arXiv:2603.15031](https://arxiv.org/abs/2603.15031).

KDA is a fast delta-rule linear attention from [flash-linear-attention](https://github.com/sustcsonglin/flash-linear-attention) that supports chunked parallel prefill and fused recurrent decoding. Block AttnRes replaces standard residual connections with learned, input-dependent softmax attention over block-level representations.

### Why KDA

The previous attempt used a custom Key-Value Delta Linear Attention (kvDLA) with explicit per-slot K/V state maintained per head. **kvDLA failed** because the gated delta rule with per-slot state cannot be decomposed into a WY-product form. Without that factorization, the activation memory cannot be reduced via chunked prefill, and parallel prefill across the sequence length is not possible. KDA from flash-linear-attention provides exactly the missing capability: a WY-style chunked computation that supports both training (chunk mode) and inference (fused recurrent mode), with reduced activation memory.

## Architecture

### Core Parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| `vocab_size` | 248,320 | Qwen3.5 tokenizer |
| `hidden_size` | 1024 | Model dimension |
| `num_heads` | 16 | Number of attention heads |
| `head_dim` | 64 | Per-head dimension |
| `num_kv` | 64 | Number of value heads for grouped value attention |
| `num_layers` | 32 | Total transformer layers |
| `num_blocks` | 8 | Block AttnRes blocks |
| `intermediate_size` | 2736 | SwiGLU intermediate dim = `hidden_size * 8 / 3`, rounded to multiple of 8 |
| `rms_norm_eps` | 1e-6 | RMSNorm epsilon |
| `max_seq_len` | -1 | Config value `-1` means unlimited context in code |
| `tie_word_embeddings` | True | Share token embedding and lm_head weights |
| `use_bias` | False | No bias in any linear layer |

### Kimi Delta Attention (KDA)

Wrapper around `fla.layers.kda.KimiDeltaAttention` (see [Kimi Linear paper](https://arxiv.org/abs/2510.26692)).

- Per-key-dim gating: forget gate has shape `[B, T, H, K]` (vector per head), compared to GDN's scalar per-head gate.
- Supports `chunk` mode for training (parallel prefill) and `fused_recurrent` mode for inference.
- Each layer has approximately `hidden_size * key_dim * 2 + hidden_size * value_dim * 3` parameters, plus a low-rank `f_proj` bottleneck.
- No short convolution by default in this project (global attention, no local inductive bias).
- Wrapper at `src/models/kda.py` provides a simple `[B, T, D] -> [B, T, D]` interface.

### Block Attention Residuals (Block AttnRes)

From Kimi Team's [Attention Residuals](https://arxiv.org/abs/2603.15031) paper.

- 32 layers partitioned into 8 blocks of 4 layers each
- Within each block: standard residual accumulation (`partial_block`)
- Across blocks: softmax attention over block-level representations
- Each sub-layer (KDA and FFN) has its own pseudo-query `w_l` for AttnRes computation
- `w_l` initialized to 0 (ensures uniform attention weights at start, matching paper)
- RMSNorm applied to keys in attention computation

**Block representation:**
```
b_n = sum_{j in Block_n} (KDA_out_j + FFN_out_j)
```

**Layer input via AttnRes:**
```
h_l = sum_i softmax( w_l^T * RMSNorm(v_i) ) * v_i
where v_i in {b_0, b_1, ..., b_{n-1}, partial_block}
```

### FFN: SwiGLU

Standard SwiGLU architecture:
```
gate = x * W_gate
up   = x * W_up
out  = (gate * Swish(up)) * W_down
```

### Normalization: RMSNorm

No standard LayerNorm. RMSNorm throughout.

### Position Encoding

No positional encoding. Architecture is NoPE (No Positional Encoding), following Kimi Linear / MLA design.

## Training

### Hardware
- 8x NVIDIA V100 16GB
- Tight VRAM budget - optimizations required

### Optimizations
- DeepSpeed-like CPU optimizer offload (AdamW on CPU)
- Automatic Mixed Precision (AMP) with FP16
- Gradients stored in FP16
- Tensor Parallelism (TP)
- Gradient checkpointing: save every `num_layers / num_blocks = 4` layers
- Gradient accumulation
- Auto GPU selection: only GPUs with >= 10GB free VRAM

### Datasets (streaming from hf-mirror.com)

Set `HF_ENDPOINT=https://hf-mirror.com` to stream from the Chinese HF mirror.

- **Pretraining:** `openbmb/Ultra-FineWeb-L3` (text field)
- **SFT:** `openbmb/UltraData-SFT-2605` (messages field, applied via chat template)

Both are streamed via `datasets.load_dataset(..., streaming=True)`. The local SFT dataset is not available — must be streamed.

### Tokenizer

Local copy of Qwen3.5 tokenizer at `src/tokenizer/`. Files copied from `/home/wlx/Qwen3.5-9B`:
- `tokenizer.json`, `tokenizer_config.json`
- `vocab.json`, `merges.txt`
- `preprocessor_config.json`, `chat_template.jinja`
- `config.json`, `configuration.json`

### Hyperparameters
- Epochs: 1
- Max steps: 1000 (for v0.0.0 validation)

## Evaluation

- Primary metric: MMLU only
- Library: `evalscope`

## CUDA Kernels

- Located in `src/models/ops/cuda/`
- Must have numerical correctness tests
- Must have performance benchmarks

## Development Roadmap

### v0.0.0: Foundation
- Basic LLM architecture (KDA + Block AttnRes)
- Code correctness verified
- Gradients flow correctly
- Training runs end-to-end

### v0.0.1: Memory Optimization
- Reduce training VRAM footprint

### v0.0.2: Small Model Training
- Train small parameter model
- Run eval scripts

### v0.0.3: CUDA Kernels
- Optimize inference and training with custom kernels

### v0.0.4: MTP (Multi-Token Prediction)

### v0.0.5: Scale Up
- Train larger parameter model

### v1.0.0: Production Release

### v2.0.0: HippoVLM
- Vision modality support
- Wavelet transform on images
- Patch embedding: 32x32x3 blocks
- `<image>` token handling
- MTP with diffusion/CNN decoder

### v2.1.0: Audio Modality

### v3.0.0: HippoVLM-embody
- Motor and sensor modalities

## Directory Structure

```
HippoLM/
├── src/
│   ├── models/
│   │   ├── ops/
│   │   │   └── cuda/          # CUDA kernels
│   │   ├── config.py          # HippoConfig
│   │   ├── norms.py           # RMSNorm
│   │   ├── kda.py             # Kimi Delta Attention wrapper
│   │   ├── block_attn_res.py  # Block Attention Residuals
│   │   ├── model.py           # HippoModel
│   │   └── activation.py      # SwiGLU FFN
│   ├── training/
│   │   └── cpu_adamw.py       # CPU offloaded AdamW
│   ├── tokenizer/             # Qwen3.5 tokenizer files
│   └── inference/
├── test/                      # All tests
├── scripts/
│   └── train.py               # Training script
└── configs/
    └── base_config.py         # Default configuration
```

## Notes

- Architecture failure probability is high - design may change at any time
- This is a research project; expect frequent pivots
- No position encoding by design
- Block AttnRes pseudo-queries MUST be initialized to zero (paper requirement)
- KDA requires CUDA tensors (uses Triton kernels) — tests and training must run on GPU
- KDA chunk mode is the only mode supported for training (fused_recurrent is inference-only)
