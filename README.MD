# HippoLM

**生产级 LLM 训练与推理框架，基于 Kimi Delta Attention (KDA) + Block Attention Residuals (Block AttnRes) 架构。**

[![Python: 3.12+](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch: 2.9.1+](https://img.shields.io/badge/PyTorch-2.9.1+-red.svg)](https://pytorch.org/)

---

## 目录

- [项目概述](#项目概述)
- [核心架构](#核心架构)
- [核心技术详解](#核心技术详解)
- [目录结构](#目录结构)
- [依赖与环境](#依赖与环境)
- [快速开始](#快速开始)
- [训练详情](#训练详情)
- [推理](#推理)
- [GPU 架构迁移指南](#gpu-架构迁移指南)
- [开发路线图](#开发路线图)
- [参考资料](#参考资料)

---

## 项目概述

HippoLM 是一个基于新型架构组合的生产级 LLM 训练/推理项目，核心技术来自：

1. **Kimi Delta Attention (KDA)** — 来自 [flash-linear-attention](https://github.com/sustcsonglin/flash-linear-attention)，支持 chunked parallel prefill（训练）和 fused recurrent mode（推理）
2. **Block Attention Residuals (Block AttnRes)** — 来自 Kimi 团队论文 [arXiv:2603.15031](https://arxiv.org/abs/2603.15031)，用学习到的 softmax 注意力替代固定残差连接

### 为什么选择这个架构？

上一个尝试使用的 **kvDLA**（自定义 KV Delta Linear Attention）因为无法分解成 WY-product 形式而失败——这意味着无法通过 chunked prefill 减少激活内存，也无法实现序列长度的并行 prefill。

KDA 提供了缺失的能力：WY 风格的 chunked 计算，同时支持训练（chunk 模式）和推理（fused_recurrent 模式），且激活内存大幅降低。

### 主要特性

| 特性 | 说明 |
|------|------|
| KDA 线性注意力 | 高效 delta-rule 注意力，O(L) 内存复杂度 |
| Block AttnRes | 解决 PreNorm 稀释问题，梯度分布更均匀 |
| SwiGLU FFN | 标准高效的前馈网络结构 |
| RMSNorm | 全程无 LayerNorm，纯 RMSNorm |
| NoPE | 无位置编码，跟随 Kimi Linear/MLA 设计 |
| Tensor Parallelism | Megatron-style TP，column-row 并行 |
| CPU Optimizer Offload | AdamW 状态卸载到 CPU，节省 GPU 显存 |
| AMP FP16 | 混合精度训练 |
| Gradient Checkpointing | 每 4 层保存，节省激活内存 |

---

## 核心架构

### 整体结构

```
Token Embeddings
      │
      ▼
┌─────────────────────────────────┐
│         Block 0 (4 layers)      │
│  Layer 0: AttnRes → KDA → FFN   │
│  Layer 1: AttnRes → KDA → FFN   │
│  Layer 2: AttnRes → KDA → FFN   │
│  Layer 3: AttnRes → KDA → FFN   │────► block_0 representation
└─────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────┐
│         Block 1 (4 layers)      │
│  Layer 4: AttnRes → KDA → FFN   │
│  Layer 5: AttnRes → KDA → FFN   │
│  Layer 6: AttnRes → KDA → FFN   │
│  Layer 7: AttnRes → KDA → FFN   │────► block_1 representation
└─────────────────────────────────┘
      │
      ▼
         ...
      │
      ▼
┌─────────────────────────────────┐
│         Block 7 (4 layers)      │
│  Layer 28-31: ...               │────► block_7 representation
└─────────────────────────────────┘
      │
      ▼
   RMSNorm → LM Head → Output
```

### 32 层分为 8 个 Block，每 Block 4 层：
- **Block 内**：标准残差累加（partial_block）
- **Block 间**：softmax 注意力聚合 block 级表示

### 主要参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `vocab_size` | 248,320 | Qwen3.5 tokenizer |
| `hidden_size` | 1024 | 模型维度 |
| `num_heads` | 16 | 注意力头数 |
| `head_dim` | 64 | 每头维度 |
| `num_kv` | 64 | GQA value 头数 |
| `num_layers` | 32 | 总层数 |
| `num_blocks` | 8 | Block AttnRes block 数 |
| `intermediate_size` | 2736 | SwiGLU 中间维度 |
| `rms_norm_eps` | 1e-6 | RMSNorm epsilon |
| `tie_word_embeddings` | True | 共享 embedding 和 lm_head 权重 |
| `use_bias` | False | 所有线性层无 bias |

---

## 核心技术详解

### 1. Kimi Delta Attention (KDA)

KDA 是一种快速的 delta-rule 线性注意力，源自 [Kimi Linear paper](https://arxiv.org/abs/2510.26692)。

**核心特性：**
- Per-key-dim gating：遗忘门形状为 `[B, T, H, K]`（每头一个向量），不同于 GDN 的标量门
- 支持两种模式：
  - `chunk` 模式：训练用（并行 prefill）
  - `fused_recurrent` 模式：推理用（递归模式）

**数学形式：**
```
h_l = sum_i alpha_i->l * v_i
其中 alpha_i->l = softmax(w_l^T * RMSNorm(k_i))
```

### 2. Block Attention Residuals (Block AttnRes)

来自 [arXiv:2603.15031](https://arxiv.org/abs/2603.15031)，解决 PreNorm 稀释问题。

**核心思想：**
- 标准残差连接用固定单位权重累加所有层输出，导致隐藏状态幅度随深度增长 O(L)，各层贡献被稀释
- Block AttnRes 用 **学习到的 softmax 注意力** 替代固定累加，让每层可以选择性地聚合前面的表示

**Block 表示：**
```
b_n = sum_{j in Block_n} (KDA_out_j + FFN_out_j)
```

**层输入（AttnRes）：**
```
h_l = sum_i softmax(w_l^T * RMSNorm(v_i)) * v_i
其中 v_i 在 {b_0, b_1, ..., b_{n-1}, partial_block} 中
```

**关键设计：**
- 每层有一个可学习的 pseudo-query `w_l`
- `w_l` **必须初始化为 0**（确保训练初期注意力权重均匀）
- RMSNorm 应用于 keys，防止大幅度输出主导注意力权重

### 3. SwiGLU FFN

标准 SwiGLU 结构：
```
gate = Swish(x @ W_gate)
up   = x @ W_up
out  = (gate * up) @ W_down
```

### 4. RMSNorm

全程使用 RMSNorm，无标准 LayerNorm。公式：
```
RMSNorm(x) = x / RMS(x) * weight
where RMS(x) = sqrt(mean(x^2) + eps)
```

---

## 目录结构

```
HippoLM/
├── CLAUDE.md                  # 项目设计文档
├── README.md                  # 本文件
├── requirements.txt           # Python 依赖
├── .env                       # 环境变量（可选）
│
├── configs/
│   ├── base_config.py         # HippoConfig 配置类
│   └── base.yml               # 默认训练配置
│
├── scripts/
│   └── train.py               # 训练脚本（v0.0.1）
│
├── src/
│   ├── models/
│   │   ├── config.py          # 配置兼容层
│   │   ├── norms.py           # RMSNorm
│   │   ├── kda.py             # KDA 包装器（兼容层）
│   │   ├── activation.py      # SwiGLU
│   │   ├── block_attn_res.py  # Block AttnRes（兼容层）
│   │   ├── model.py           # HippoModel 主模型
│   │   ├── tp_layers.py       # Tensor Parallel 层
│   │   ├── tp_model.py        # TP 模型包装器
│   │   └── ops/
│   │       ├── __init__.py    # ops 导出
│   │       ├── kda.py         # KDA 实现
│   │       ├── attn_res.py    # Block AttnRes 实现
│   │       └── _vendored/     # flash-linear-attention 库
│   │           └── fla/
│   │               ├── layers/     # 各种注意力层（KDA, GLA, Mamba...）
│   │               ├── ops/        # CUDA/Triton 算子
│   │               └── modules/   # 融合模块
│   │
│   ├── training/
│   │   ├── cpu_adamw.py       # CPU offload AdamW
│   │   └── param_offload.py   # 参数 offload
│   │
│   ├── tokenizer/             # Qwen3.5 tokenizer 文件
│   │   ├── tokenizer.json
│   │   ├── tokenizer_config.json
│   │   ├── vocab.json
│   │   ├── merges.txt
│   │   ├── config.json
│   │   ├── configuration.json
│   │   ├── preprocessor_config.json
│   │   └── chat_template.jinja
│   │
│   └── inference/            # 推理相关（待实现）
│
├── test/                      # 测试
│   ├── test_model.py
│   ├── test_memory.py
│   ├── diag_tp_loss.py
│   └── diag_nan_source.py
│
└── output/                    # 训练输出目录
```

---

## 依赖与环境

### 基础依赖

```txt
# Core ML
torch==2.9.1+cu128
transformers==5.5.4
numpy>=1.24.0
einops==0.8.2
flash-linear-attention

# Data
datasets>=2.14.0
dill>=0.4.0

# Training
accelerate>=0.20.0
tqdm>=4.65.0

# Evaluation
evalscope==1.6.0

# Utilities
pyyaml>=6.0
psutil>=5.9.0
```

### 硬件要求

| 配置 | 要求 |
|------|------|
| GPU | NVIDIA V100 / A100 / H100 或兼容 CUDA 的 GPU |
| 显存 | 至少 10GB（用于自动 GPU 选择） |
| 内存 | 建议 64GB+ |
| 驱动 | CUDA 12.8+ |

### NCCL 要求

NCCL 用于 Tensor Parallelism 通信。建议使用与 PyTorch 配套的 NCCL 版本。

### 环境变量

```bash
# 数据集镜像（国内加速）
HF_ENDPOINT=https://hf-mirror.com

# HuggingFace token（可选，用于私有模型）
# HF_TOKEN=your_token_here

# NCCL 设置（可选）
NCCL_P2P_LEVEL=NVL    # 优先使用 NVLink
NCCL_IB_DISABLE=1     # 禁用 IB

# 缓存设置
HF_HOME=~/.cache/huggingface
```

---

## 快速开始

### 1. 克隆与安装

```bash
git clone https://github.com/centslab/HippoLM.git
cd HippoLM
pip install -r requirements.txt
```

### 2. 配置

编辑 `configs/base.yml` 或创建 `.env` 文件：

```bash
# .env 示例
HF_ENDPOINT=https://hf-mirror.com
HF_TOKEN=your_token_here
```

### 3. 运行训练

```bash
# 标准训练
python scripts/train.py

# 指定配置
python scripts/train.py --config configs/base.yml

# 指定 GPU
python scripts/train.py --gpus 0,1,2,3

# 指定步数
python scripts/train.py --max_steps 10000
```

### 4. 检查训练状态

训练日志输出到 `output/` 目录，包括：
- `train.log` — 训练日志
- `checkpoint_*` — 检查点文件

---

## 训练详情

### 训练脚本特性（v0.0.1）

| 特性 | 说明 |
|------|------|
| **自动 GPU 选择** | 只选择显存 >= 10GB 的 GPU，优先选择连续 GPU |
| **Megatron-style TP** | column-row 并行 FFN，column 并行 lm_head |
| **Muon + AdamW** | 2D 线性权重用 Muotton，norm/lm_head/embed 用 AdamW |
| **CPU Optimizer Offload** | 优化器状态 m/v 卸载到 pinned memory |
| **无 GPU 梯度存储** | micro-batch 梯度流式传输到 CPU accumulator |
| **Background Prefetch** | 后台线程预取数据 |
| **双数据源** | 优先 ModelScope（阿里云 CDN），失败则回退 HF mirror |
| **Qwen3.5 Tokenizer** | 本地 tokenizer |
| **max_step 限制** | 支持步数限制而非 epoch 限制 |

### 数据集

| 阶段 | 数据集 | 字段 |
|------|--------|------|
| 预训练 | `openbmb/Ultra-FineWeb-L3` | `text` |
| SFT | `openbmb/UltraData-SFT-2605` | `messages`（应用 chat template） |

两者都通过 `datasets.load_dataset(..., streaming=True)` 流式加载。

### 数据流

```
Raw Text → Tokenizer → input_ids + labels
                              │
                              ▼
                      DataLoader (streaming)
                              │
                              ▼
              ┌───────────────────────────────┐
              │     TP Training Loop          │
              │  ┌─────────────────────────┐  │
              │  │  Micro-batch forward    │  │
              │  │  (no grad on GPU)       │  │
              │  └─────────────────────────┘  │
              │               │               │
              │               ▼               │
              │  ┌─────────────────────────┐  │
              │  │  Stream grad to CPU     │  │
              │  │  Free GPU activation    │  │
              │  └─────────────────────────┘  │
              │               │               │
              │               ▼               │
              │  ┌─────────────────────────┐  │
              │  │  Optimizer step (CPU)   │  │
              │  │  after N micro-batches  │  │
              │  └─────────────────────────┘  │
              └───────────────────────────────┘
```

### 关键超参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `batch_size` | 2 | micro-batch 大小 |
| `seq_len` | 512 | 序列长度 |
| `learning_rate` | 1e-4 | 学习率 |
| `weight_decay` | 0.01 | 权重衰减 |
| `gradient_accumulation_steps` | 4 | 梯度累积步数 |
| `fp16` | true | 混合精度 |
| `kda_mode` | chunk | KDA 模式 |
| `use_short_conv` | true | 短卷积（本地感应偏置） |
| `num_layers` | 32 | 层数 |
| `num_blocks` | 8 | block 数 |

---

## 推理

### 模型生成

```python
import torch
from src.models.model import HippoModel
from src.models.config import HippoConfig
from src.tokenizer import get_tokenizer

# 加载配置和模型
config = HippoConfig()
model = HippoModel(config)
model.load_state_dict(torch.load("output/checkpoint_last.pt"))
model.eval()

# 加载 tokenizer
tokenizer = get_tokenizer("src/tokenizer")

# 生成
input_ids = tokenizer("Hello, world!")["input_ids"]
input_ids = torch.tensor([input_ids])

output_ids = model.generate(
    input_ids,
    max_new_tokens=100,
    temperature=0.8,
    top_k=50
)

print(tokenizer.decode(output_ids[0]))
```

### KDA 推理模式

推理时将 `kda_mode` 切换为 `fused_recurrent`：

```python
config = HippoConfig(kda_mode="fused_recurrent")
model = HippoModel(config)
```

---

## GPU 架构迁移指南

### 当前支持状态

| 架构 | Compute Capability | GPU 示例 | 支持状态 |
|------|------------------|---------|----------|
| Volta | sm_70 | V100 | ✅ 支持 |
| Ampere | sm_80 | A100 | ✅ 支持 |
| Hopper | sm_90 | H100 | ✅ 支持 |

### 迁移新框架难度评估

**极小。** 主要原因：

1. **Triton 内核是架构透明的**
   - 项目大量使用 Triton JIT 内核（来自 vendored `flash-linear-attention`）
   - Triton 会自动为目标架构生成对应代码
   - sm_70 / sm_80 / sm_90 迁移无需代码改动

2. **PyTorch 分布式是架构无关的**
   - Tensor Parallelism 使用标准 `torch.distributed` + NCCL
   - NCCL 支持多架构，只需确保 NCCL 版本兼容

3. **Checkpoints 完全兼容**
   - PyTorch state_dict，跨架构直接可用

### 自定义 CUDA 内核（未来 v0.0.3）

| 方案 | 说明 | 迁移成本 |
|------|------|---------|
| **Triton** | 自动适配所有架构 | 无 |
| **CUDA + PTX JIT** | 编译成 virtual ISA，运行时 JIT | 极小 |
| **预编译 cubin** | 每个架构单独编译 | 高（需多架构测试） |

**建议**：使用 Triton 编写自定义 kernel，一份代码架构透明。

### 迁移步骤

1. 安装目标机器的 GPU 驱动
2. 安装对应 CUDA 版本的 PyTorch
3. 确保 NCCL 版本兼容
4. 复制 checkpoints 和代码
5. 验证训练几个 step 检查 loss 正常

---

## 开发路线图

| 版本 | 目标 | 状态 |
|------|------|------|
| **v0.0.0** | 基础架构（KDA + Block AttnRes），梯度流动正确，训练端到端 | ✅ 完成 |
| **v0.0.1** | 内存优化，降低训练 VRAM 占用 | 🔄 当前 |
| **v0.0.2** | 小模型训练 + MMLU 评测 | 📋 待开始 |
| **v0.0.3** | CUDA Kernels 优化推理和训练 | 📋 待开始 |
| **v0.0.4** | MTP（Multi-Token Prediction） | 📋 待开始 |
| **v0.0.5** | 放大模型规模 | 📋 待开始 |
| **v1.0.0** | 正式发布 | 📋 待开始 |
| **v2.0.0** | HippoVLM — 视觉模态，小波变换，32x32x3 patch embedding | 📋 规划中 |
| **v2.1.0** | 音频模态支持 | 📋 规划中 |
| **v3.0.0** | HippoVLM-embody — 具身智能，motor + sensor 模态 | 📋 规划中 |

---

## 参考资料

### 论文

1. **Kimi Linear** — [arXiv:2510.26692](https://arxiv.org/abs/2510.26692)
   > Yu Zhang et al. *Kimi Linear: An Expressive, Efficient Attention Architecture*

2. **Attention Residuals** — [arXiv:2603.15031](https://arxiv.org/abs/2603.15031)
   > Kimi Team. *Attention Residuals: Learning Depth-wise Softmax Aggregation for Modern LLMs*

3. **flash-linear-attention** — [GitHub](https://github.com/sustcsonglin/flash-linear-attention)
   > Songlin Yang et al. *Flash-Linear-Attention*

### 架构设计决策

| 决策 | 原因 |
|------|------|
| 使用 KDA 而非 kvDLA | KDA 可分解为 WY-product，支持 chunked prefill |
| Block AttnRes pseudo-query 初始化为 0 | 确保训练初期注意力权重均匀，防止训练波动 |
| NoPE（无位置编码） | 跟随 Kimi Linear/MLA 设计 |
| RMSNorm 而非 LayerNorm | Kimi Linear 验证有效 |
| SwiGLU FFN | 标准高效的前馈结构 |

### 相关技术

| 技术 | 用途 |
|------|------|
| Triton | GPU kernel 编写（架构透明） |
| NCCL | 多 GPU 通信 |
| Muon | 2D 线性权重优化器 |
| AdamW | 1D 参数优化器 |
| AMP | 混合精度训练 |
| Gradient Checkpointing | 节省激活内存 |

---

## 注意事项

- ⚠️ **架构失败概率较高** — 设计可能随时更改
- ⚠️ 这是科研项目 — 预期频繁 pivot
- ⚠️ 无位置编码（NoPE）— 架构层面决定
- ⚠️ KDA 需要 CUDA tensor（使用 Triton kernel）— 测试和训练必须在 GPU 上运行
- ⚠️ KDA chunk 模式是训练唯一支持的模式（fused_recurrent 仅推理用）

---

## 致谢

- [flash-linear-attention](https://github.com/sustcsonglin/flash-linear-attention) — 核心注意力实现
- [Kimi Team](https://github.com/MoonshotAI) — KDA 和 Block AttnRes 论文
- [Qwen](https://github.com/QwenLM) — Tokenizer 和架构参考
