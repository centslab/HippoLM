# HippoLM Memory

Historical incidents, one-off debugging notes, and architectural
decisions that were once carried across sessions by Claude Code's
auto-memory system. Re-populate entries here as discoveries are
re-made or rediscovered.

## Hardware

- **Target GPUs**: RTX 4090 (sm_89, production) / 5060 Ti 16G (sm_120, dev).
- **VRAM ceiling**: 16 GB (5060 Ti). Smoke test must verify
  `torch.cuda.max_memory_allocated() <= 16 GB`.
- **V100 dropped** 2026-07-08: Marlin FP4 + NVFP4 W4A16 FFN require
  sm_80+ BF16 MMA. See `.claude/rules/dont-target-v100.md`.
- **sm_75 (Turing) and sm_100 (Blackwell datacenter)** also out of scope.
- **CUDA toolkit pin**: 12.8 (sm_120 ISA intro). No bumping for V100 compat.
- **Smoke-test command** (canonical):
  ```
  python scripts/train.py --config configs/base.yml --use_dummy_data \
    --max_steps 4 --tp_sim --tp_size 2 --gradient_accumulation_steps 2 \
    --batch_size 2 --seq_len 512 --num_layers 4 --num_blocks 2
  ```

## KDA / Attention

### EFKDA → KDA transition

EFKDA (the earlier GDN2-era kernel) was fully replaced by KDA in
June 2026. The production KDA kernel uses a chunk-wise formulation
with `chunk_fwd`, `chunk_bwd`, and `prepare` sub-kernels. The
Triton reference implementation lives in
`src/models/ops/_vendored/fla/ops/kda/`; the CUDA port is in
`src/models/ops/cuda/kda_fwd/`.

Key debugging history (see `docs/kda_kernel_structure.md`):
- CHUNK is algorithm-level, not a tiling parameter.
- `q_decayed = q * exp(g_cumsum)` overflows bf16 when
  `g_cumsum > ~88` (CHUNK=32 failure mode).
- Fix: rework the gating recurrence or cast to fp32 for the
  cumulative sum.

### AttnRes bwd non-contiguous incident (2026-07-14)

`torch.einsum` output is non-contiguous. Passing it to a Triton
kernel that indexes via `stride_*` args silently reads wrong data.
Fix: `.contiguous()` before any stride-indexed Triton kernel call.
`tl.make_block_ptr` with explicit `order=` sidesteps the trap.

### Roofline discipline

Prior KDA bwd speedup claims underestimated FLOPs (used 2*M*K*N
instead of proper counting). Every `tl.dot` = `2 * M * K * N`.
Always verify FLOPs counting + AI vs ridge before reporting
speedup. See `.claude/skills/kda-correctness-sweep/SKILL.md`.

## Optimizer

### Muon

- BF16 for both AdamW and Muon state (CPU pinned).
- int8/mxfp8 Muon storage removed 2026-07-12 (no HWM benefit).
- Muon momentum in BF16: need to verify that the byte math works
  correctly (BF16 has 7-bit mantissa, updates must not underflow).
- Per-tensor offload was attempted but did not move HWM, reverted.

### Saved tensors ≠ HWM

Multiple optimization attempts (NVFP4 FFN, Marlin cache drop, etc.)
reported "saved_tensors dropped N MiB" but none moved actual HWM.
Use `torch.cuda.max_memory_allocated()` for the real peak.
See `.claude/rules/saved-tensors-not-hwm.md`.

## Training Loop

### TP state_dict bug

`TPHippoModel` uses `nn.ModuleDict` for per-tensor-parallel
wrappers; state_dict was silently missing TP-sharded params.
Fix: ensure `ModuleDict` keys match the checkpoint convention or
override `state_dict()` to aggregate.

### Opt-N REVERTED

Several optimizer-phase refactors were committed then reverted
because the HWM didn't move or correctness regressed. Check git
log for commits prefixed `revert:` in the opt phase area (~June
2026).

## Data Pipeline

### Cache routing

ModelScope (Aliyun CDN) is preferred. Local parquet cache with HF
mirror fallback. Fallback activates only on real network errors
(not DNS / timeout). See `docs/cache_routing.md`.

### Packer alignment

Data packer must align to the chunk boundary. Misalignment causes
silent token dropout in the streaming prefetcher.

## Coding Conventions

- **English identifiers only** (Chinese allowed in commit messages
  and conversation). Enforced by review.
- **No backward-compat shims**. Delete old code outright. See
  `.claude/rules/shim-deletion-protocol.md`.
- **No `torch.compile(model, mode="reduce-overhead")`**. Not
  compatible with the custom autograd functions.
- **Lazy imports**: use `PEP 562` (`__getattr__` at module level)
  for optional dependencies, not top-level try/except.

## Evaluation

Primary metric and eval library: check `scripts/eval_server.py`
for the current contract.

## FP8 GEMM Accumulation Precision

### cuBLASLt nvjet 和 R10 都用 F16-acc，不是 F32-acc

cuBLASLt nvjet 的 `nvjet_sm120_qqtst_mma_128x128x64_6_64x64x64_tmaAB_bz_TNNN` 使用 `QMMA.16832.F16.E4M3.E4M3`（F16 累加器），**不是** F32-acc。R10 使用同样的 `mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16` 指令。

sm_120（Blackwell consumer）上两种 MMA 变体实测：
- F32-acc（R7）：~47 TF
- F16-acc（R10）：~81 TF，nvjet：~85 TF（均为 2026-07-27 audit 校正后真实数字）

F16-acc 快 ~1.74x 的原因：每个 MMA slot 只需要 2 个寄存器（packed F16，`uint32_t` 装 2×`__half`），F32-acc 需要 4 个。寄存器压力减半，occupancy 翻倍。

R10 原理：每 128 K-element block 内用 QMMA 累加到 packed F16，block 结束时 `__half22float2()` 转 FP32 并 FMA 乘 block scale 后累加到主 FP32 accumulator，F16 accum 归零。最终数值精度由 FP32 主 accumulator 和 FP32 scale 保证。

"cublasLt 用 F32-acc 到 80+ TF" 的说法是误解——可能把输出端 **per-tensor FP32 scale**（TensorWise 标量模式）误当作累加器精度了。scale 精度和 MMA 累加器精度是两回事。

详见 `.claude/skills/bench-flops/SKILL.md`（audit 记录）和 `docs/fp8_gemm_kernel_pipeline.md`。

## FP8 GEMM 理论峰值与实测分析

### 修正后的理论极限公式（2026-07-28 实测）

Blackwell consumer (sm_120, RTX 5060 Ti) 的 FP8 F16-acc 极限：

```
Peak TFLOPS = 8192 * 36 * t * freq_GHz / 1000
  8192 = FLOPs per m16n8k32 (2 * 16 * 32 * 8)
  36   = SM 数量
  t    = MMA throughput (MMA/cycle/SM)
  freq = GPU 实际工作频率 (GHz)
```

**各层级的 t 值（实测 @ 2797 MHz）：**

| 层级 | t (MMA/cycle/SM) | TFLOPS | %理论极限 | 说明 |
|------|:-:|:-:|:-:|------|
| 理论峰值 (NVIDIA spec) | 0.250 | 206.2 | 100% | 512*144*freq 的等价形式 |
| 纯 MMA 微基准 (8 warps, 500k iter batched) | 0.116 | 95.4 | 46.3% | 硬件实测上限：4 TC × 16 cycle latency 导致 ~54% pipeline 效率损失 |
| R10 accum F16-acc (BK=128, S=3) | 0.101 | 83.6 | 40.5% | 含 TMA load + scale + barrier 开销 |
| nvjet TensorWise | 0.103 | ~85 | ~41% | cuBLASLt 最优路径 |

**关键分析：**

- tcgen05 不占 warp slot，允许 warp 连续发 8 次 mma 到不同 accumulator（内层 ni 循环），但对**极限峰值无影响**——tensor core pipeline 完成率固定为 1 mma 每 ~16 cycles 每 sub-partition。
- 理论到实测的 ~54% gap（0.250→0.116）来自 pipeline depth：4 tensor core per SM，每个需 ~16 个 cycle 完成一次 mma，8 warps 的指令发射不够消除流水线气泡。
- R10 达到纯 MMA 上限的 ~87%（0.101/0.116），剩余 ~13% 为 TMA/scale/barrier 不可重叠开销。
- **最大杠杆仍然是 TensorWise scalar 路径**（`_scaled_mm` TensorWise），但实测 nvjet 也仅 ~85 TF，提升不大。

### R11 (BK=64, NUM_STAGES=6, ACC_RAW_F16) 实验结论

R11 尝试匹配 nvjet 的 pipeline（BK=64, S=6），结果：

| 配置 | 正确性 | 性能 @ 4096^3 | 说明 |
|------|--------|:-:|------|
| SWIZZLE_128B + BK=64 | ❌ NaN | — | 128B XOR 溢出 64B 行边界 |
| SWIZZLE_64B + BK=64 | ❌ 数值错误 | — | SWIZZLE_64B 的 4-bank XOR 与 ldmatrix `.aligned` 的 8-bank XOR 不兼容 |
| SWIZZLE_NONE + BK=64 | ✅ 正确 | 59.0 TF (≈71% R10) | 无 swizzle，但 bank conflict 导致 perf 下降 |

SWIZZLE_64B 在 CUDA 13.0 中可用（`CU_TENSOR_MAP_SWIZZLE_64B`，要求 boxDim[0] <= 64），但与 ldmatrix `.aligned` 的固定 8-bank XOR 不兼容。正确路径是 SWIZZLE_NONE + 无软件 XOR，但性能不如 R10。

**R10 (BK=128, S=3) 仍是 sm_120 上 blockwise FP8 GEMM 的最优配置。**

### R10 性能优化天花板

R10 的 {pipeline, tile, scale} 配置已接近最优：

| 优化方向 | 尝试 | 结果 |
|----------|------|------|
| BLOCK_OUT=128x128 + F16-acc | 构建并测试 | 与 64x64 完全一致（~1ms 误差），scale 开销仅 ~0.015% 总 FLOP |
| NUM_STAGES=4 | smem 不足 (96KB cap) | 不可行 |
| BK=64, S=6 (R11) | SWIZZLE_NONE | 59 TF，比 R10 慢 29% |
| WARP_N=32 | 未测试 | 潜在 ~1-2%，可尝试 |
| TensorWise scalar | 未实现 | nvjet 实测 ~85 TF，提升有限 |

## FP8 GEMM 优化全景（R1 → R10）2026-07-28 收尾

**R10 已经是 blockwise FP8 GEMM 在 sm_120 上的最终解。** 整个 7 轮 sweep 的脉络：

### 各轮的核心突破

| 轮次 | 关键改动 | 实测 @ 4096^3 (post-audit) | 杠杆来源 |
|------|----------|----------------------------|----------|
| R0/R1 | 基础 TMA + ldmatrix 路径 | ~30 TF | 起点 |
| R2/R3 | warp-specialization (producer/consumer) | ~50 TF | pipeline 隐藏 TMA 延迟 |
| R4 | 多 lane 并行 TMA | ~60 TF | producer 内部 |
| R5/R5b | blockwise 2D scale (64x64) | ~70 TF | 消除 per-element scale broadcast |
| R6 | native MMA accumulation (BLOCK_ACCUM=true) | ~95 TF | 消除 sub-MMA FMA flush |
| R7 | NUM_STAGES=3 + DIRECT_STORE=true | **97.5 TF** | smem 翻倍 + 跳过 TMA store |
| R10 | F16 in-block accumulator (ACC_RAW_F16) | **83.6 TF (real)** | 寄存器压力减半 → occupancy 翻倍 |
| (R11) | 尝试 BK=64/S=6 匹配 nvjet pipeline | 59 TF (FAILED) | 被 swizzle/ldmatrix 兼容性阻断 |

### R10 为何是终点

**R10 已经是 ~95% of nvjet（83.6 vs ~85 TF）**。剩下的 ~5% 是 blockwise scale 模式固有的开销，**不是 kernel 优化能拿到的**：

1. **tcgen05 不占 warp slot，但不影响极限峰值**：warp 可以在 16 cycle/mma 期间继续发 ldmatrix/arithmetic，但 tensor core pipeline 完成率固定为 1 mma/16 cycle/TC。极限峰值 0.25 mma/cycle/SM 来自硬件，warp 调度器再优化也不能突破。
2. **R10 已经达到纯 MMA 峰值的 87%**（0.101/0.116 = t_R10/t_pureMMA）。剩余 13% 是 TMA/scale/barrier 不可重叠的固定开销。
3. **8 warps per CTA 已经是最优的 latency hiding 配置**：
   - R10 (WARP_M=32, WARP_N=64, CWG=2): **83.5 TF** ← production
   - nvjet warp tile (WARP_M=64, WARP_N=64, CWG=1): **71.3 TF** (slower! 4 warps 不够)
4. **BLOCK_SCALE_K 增大变慢**：bsk256/512/1024 都比 bsk128 慢 17-20%。

### R11 失败总结（不要重蹈覆辙）

**目标**：用 BK=64 + NUM_STAGES=6 匹配 nvjet 的 pipeline 深度，看能不能拿到 ~5% 加速。

**3 个失败路径：**

| 路径 | 现象 | 根因 |
|------|------|------|
| SWIZZLE_128B + BK=64 | 全 NaN | 128B XOR 跨 64B 行边界，2 行数据被 XOR 混洗 |
| SWIZZLE_64B + BK=64 | finite 但数值错（med\|diff\|=89%） | sm_120 ldmatrix `.aligned` 硬编码 8-bank XOR，与 4-bank SWIZZLE_64B 不匹配 |
| SWIZZLE_NONE + BK=64 | random 数据下仍有 med\|diff\|=39% | 看起来 ldmatrix 边界条件不满足，结构化测试（整数/小数值）能通过，random float 漏出来 |

**根本限制**：sm_120 ldmatrix `.aligned` 假设 SMEM 用 128B 段 swizzle。要支持 64B 行就要么改 ldmatrix 的版本（不灵活）、要么放弃 swizzle（perf 损失 ~30%）、要么把 smem 行 pad 到 128B（smem 翻倍超 99KB cap）。**R11 在当前 framework 下不可行。**

### nvjet "1.7x 加速" 是误判

`docs/fp8_gemm_landscape_2026_07_23.md` 中的 "TensorWise scalar 1.6-1.8x faster" 来自**有 bug 的 cudaEvent 单次计时**（under-report 2.07x）。post-audit 真实数字：nvjet TensorWise ~85 TF，R10 blockwise ~83.6 TF——**差距 1.5%，不是 1.7x**。**R10 没有 ROI 改进空间了。**

### 代码现状（commit 时的状态）

- `fp8_gemm.cuh`：
  - **R10 路径（BK=128）行为完全不变**：`swizzle_smem_offset` 的 BK>=128 分支保持原 128B XOR；TMA 仍用 `SWIZZLE_128B`。
  - 静态断言放宽到 `BK == 64 || BK == 128`（为未来实验留口子）。
  - BK=64 路径：TMA 走 `SWIZZLE_NONE` fallback，软件 `swizzle_smem_offset` 也走 no-swizzle 分支（两者一致，data path 不会因不匹配崩溃，但 R11 仍有未解的随机数据正确性 bug）。
- `blockwise_accum_bsk128_f16_kernel_entry.cu.in`：加了一条 doc comment 标记 2026-07-27 audit（real 数字）。
- `MANIFEST.fp8_gemm.md`：已记录 sweep 状态。
- 没有新增功能、没有改变 R10 的 prod .so 行为。

### 此后遇到 FP8 GEMM 优化时

1. **不要**再尝试 BK=64 + SWIZZLE_64B 路径（已知 ldmatrix 不兼容）。
2. **不要**用单次 cudaEvent 测 ctypes 加载的 .so kernel（under-report 2.07x）。要用 batched 计时或 `torch.profiler`。
3. **不要**给 R10 加新变体（w64x64、bsk256/512/1024 都已测试且更慢）。
4. **真正的杠杆**是改变 scale 模式（TensorWise scalar/RowWise vs Blockwise），但用户已确认 "没意义也不会上"。
5. 看 `docs/fp8_gemm_kernel_pipeline.md` § Dead ends on sm_120（Split-K、2 producer lanes、m16n8k64 MMA 都已 dead end）。

