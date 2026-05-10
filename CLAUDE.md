# HippoLM Project Design

## Overview

HippoLM is a production-grade LLM training and inference project based on a novel architecture combining **Gated DeltaNet-inspired key-value linear attention (kvDLA)** with **Block Attention Residuals (Block AttnRes)** from [arXiv:2603.15031](https://arxiv.org/abs/2603.15031).

The architecture replaces the state matrix in Gated DeltaNet with explicit key-value pairs and replaces standard residual connections with learned, input-dependent depth-wise attention over block-level representations.

## Architecture

### Core Parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| `vocab_size` | 248,320 | Qwen3.5 tokenizer |
| `hidden_size` | 1024 | Model dimension |
| `num_heads` | 16 | Number of attention heads |
| `head_dim` | 64 | Per-head dimension |
| `num_kv` | 64 | Number of KV pairs (training). Equal to `head_dim`. Inference can vary. |
| `num_layers` | 32 | Total transformer layers |
| `num_blocks` | 8 | Block AttnRes blocks |
| `intermediate_size` | 2736 | SwiGLU intermediate dim = `hidden_size * 8 / 3`, rounded to multiple of 8 |
| `rms_norm_eps` | 1e-6 | RMSNorm epsilon |
| `max_seq_len` | -1 | Config value `-1` means unlimited context in code |
| `tie_word_embeddings` | True | Share token embedding and lm_head weights |
| `use_bias` | False | No bias in any linear layer |

### Key-Value Delta Linear Attention (kvDLA)

Replaces traditional self-attention. Maintains `num_kv` explicit key-value pairs per head instead of a state matrix.

**Per token computation:**

```
k_t = x_t * W_K                          # [B, num_heads, head_dim]
v_t = x_t * W_V                          # [B, num_heads, head_dim]

v_hat_t = sum_i softmax( K_i^T k_t / sqrt(H) )_i * V_i

e_t = v_t - v_hat_t

g_i = sigmoid( alpha * (K_i^T k_t) + beta * (V_i^T e_t) + gamma )

V_i <- V_i + eta_v * g_i * e_t
K_i <- K_i + eta_k * g_i * (k_t - K_i)
```

**Parameters:**
- `alpha`, `beta`, `gamma`: trainable scalars. Initialized: `alpha=0.1`, `beta=0.1`, `gamma=0`
- `eta_v`, `eta_k`: trainable vectors, shape `[num_heads, head_dim]`. Initialized to `0.01`
- `K`, `V`: state variables (not model parameters). Initialized `N(0, 0.01^2)`
- `H` = `head_dim` = 64

### Block Attention Residuals (Block AttnRes)

From Kimi Team's [Attention Residuals](https://arxiv.org/abs/2603.15031) paper.

- 32 layers partitioned into 8 blocks of 4 layers each
- Within each block: standard residual accumulation (`partial_block`)
- Across blocks: softmax attention over block-level representations
- Each sub-layer (kvDLA and FFN) has its own pseudo-query `w_l` for AttnRes computation
- `w_l` initialized to 0 (ensures uniform attention weights at start, matching paper)
- RMSNorm applied to keys in attention computation

**Block representation:**
```
b_n = sum_{j in Block_n} (kvDLA_out_j + FFN_out_j)
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

### Hyperparameters
- Epochs: 1
- Max steps: 1000 (for v0.0.0 validation)
- Dataset: `/mnt/6138/datasets/xpengx/SlimPajama-627B`
- Tokenizer: Qwen3.5 from `/home/wlx/Qwen3.5-9B` (to be copied into project)

## Evaluation

- Primary metric: MMLU only
- Library: `evalscope`

## CUDA Kernels

- Located in `src/models/ops/cuda/`
- Must have numerical correctness tests
- Must have performance benchmarks

## Development Roadmap

### v0.0.0: Foundation
- Basic LLM architecture
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
│   │   ├── kv_dla.py          # Key-Value Delta Linear Attention
│   │   ├── block_attn_res.py  # Block Attention Residuals
│   │   ├── layers.py          # HippoLayer
│   │   └── model.py           # HippoModel
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
