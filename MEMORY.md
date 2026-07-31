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

### 更正（2026-07-31 逆向实证）：cuBLASLt nvjet 是纯 F32-acc，没有 F16-acc 路径，也无需"防溢出策略"

之前的条目（2026-07-27 audit）声称 nvjet 用 `QMMA.16832.F16.E4M3.E4M3`（F16 累加器）。**该结论是错的**，源于误读 PTX 内核名里的寄存器存储类型后缀。2026-07-31 全量逆向 cuBLASLt 13.1.1.3（5449 个 cubin，含 sm_120×1596 / sm_90×1597 / sm_89×247 / sm_100×1188 全部反汇编）结果：

1. **sm_120 全部 FP8 MMA 只有三种**，全是 F32 累加：
   `QMMA.16832.F32.E4M3.E4M3` / `.E5M2.E4M3` / `.E4M3.E5M2`。**任何 arch 上都没有 `QMMA.*.F16.*`**（sm_89 同样只有 F32）。
2. **运行时 heuristic 日志**：`computeType=COMPUTE_32F`，algoId=67、customOption=18 恒定；`nvjet_sm120_qqtst_mma_128x128x64_6_64x64x64_tmaAB_bz_TNNN`（4096³，798 us）tile=128x128、stages=64x3、255 regs、4 warps/CTA、50176B smem。
3. **PTX 源码**（`libcublasLt.so.281.sm_120.ptx` 等）：nvjet 的 GEMM 是 `sm89_xmma_gemm_e4m3f16_e4m3f32_f32_tn_n_tilesize128x128x64_stage3_...` 一族，指令为 `mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32` → `QMMA.16832.F32`。名字里的 `e4m3f16_e4m3f32` 是 **A/B 在寄存器里的存储格式**（f16 = 2 个 FP8 打包进一个 16-bit 寄存器，f32 = 每 32-bit 寄存器装 1 个），第三个字段 `_f32` 才是累加器类型——**全族都是 f32**。
4. **SASS 结构**：K 循环体里只有 QMMA + LDSM + LDGSTS + DEPBAR + BAR.SYNC，**MMA 之间零 FFMA**；scale 是 TensorWise 标量，K 循环结束后一次性应用（`FMUL R135, R136, R135` / `FMUL R132, R135, R132`，3 个 FMUL 搞定 alpha×scale_a×scale_b）。384 个 FFMA 全在 epilogue。
5. **对抗性数据实测（4096³ 全 FP8 max=448）**：nvjet 输出 = 8.2208e8（448²×4096），与 FP32 参考**逐位一致、无 NaN/Inf**；同一数据跑 R10 hybrid（F16-acc）输出 **inf**。F16-acc 在 K>~1800 的 chain 上必然溢出（见 W8A8 F16-acc overflow 条目），nvjet 不受影响是因为根本没有 F16-acc。

**结论**：nvjet 防溢出的"策略"就是**用 F32 累加 + TensorWise 标量 scale 在末尾一次性应用**——最保守的做法，没有 F16-acc 也就不存在溢出风险。R10 的 F16-acc（`mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16`，见 `fp8_gemm.cuh:539`）是项目自己的优化，不是 nvjet 的做法；"R10 和 nvjet 用同一条指令"是把 R10 的指令错安在 nvjet 头上。nvjet 在 sm_120 比 R10 快（178 vs 170 TF @ 4096³）靠的是 TMA + pipeline 深度 + TensorWise 无 blockwise scale 开销，不是 F16-acc。

旧条目保留的仍正确部分：
- F16-acc（R10）比 F32-acc（R7）快 ~1.74x 的寄存器原因分析仍成立（2 vs 4 regs/MMA slot）。
- R10 的 blockwise F16-acc 原理描述（每 128 K-block QMMA 累加 packed F16 → `__half22float2()` 转 FP32 → FMA 乘 scale）仍成立。
- "scale 精度和 MMA 累加器精度是两回事"仍成立——但之前把输出端 per-tensor FP32 scale 误当 F16-acc 的那句也一起更正了：nvjet 的 scale 就是 TensorWise 标量，累加器就是 F32，两者一致。

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
| BLOCK_OUT=128x128 + F16-acc | 构建并测试（CUPTI + 批量 Event，4096³ 5 次重复） | **0.978–0.982× R10**（~2% 慢），修正 pre-audit 的 "完全一致" 说法 |
| NUM_STAGES=4 | smem 不足 (96KB cap) | 不可行 |
| BK=64, S=6 (R11) | SWIZZLE_NONE | 59 TF，比 R10 慢 29% |
| WARP_N=32 | 未测试 | 潜在 ~1-2%，可尝试 |
| TensorWise scalar | 未实现 | nvjet 实测 ~85 TF，提升有限 |
| Interleaved 2×BK=64 (K-loop 重写) | 仅分析，未实现 | 理论可行，预期 <2% 收益，见下文 |

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

### Interleaved 2×BK=64 K-loop 重写（理论可行，未实现，2026-07-30 收尾）

R11 三条失败路径（SWIZZLE_128B / SWIZZLE_64B / SWIZZLE_NONE 配 BK=64）都死在同一个根本约束上：**sm_120 ldmatrix `.aligned` 硬编码 8-bank XOR**（PTX 规范：`.aligned` 隐含 `addr ^ (row & 7) << 4`），没有 non-aligned 变体可选。这意味着 smem 必须按 **128B 段**组织（每段 8 个 16B sub-segment），XOR 索引用 `row & 7`。SWIZZLE_64B 把每行切成 64B 段（4 个 16B sub-segment），XOR 用 `row & 3`——两个 4-bank XOR 跟 8-bank ldmatrix 在段边界上冲突。

唯一能让 BK=64 + ldmatrix 共存的"特殊 layout"是 **interleave 2 个 BK=64 行进一个 128B super-row**：

| 部件 | 当前 R10 (BK=128) | Interleaved (BK=64, super-row=128B) |
|------|-------------------|--------------------------------------|
| TMA boxDim | `(128, BM)` | `(128, BM)` ← 同样拉 128B/次，但承载 2 行 64B |
| smem 行宽 | 128B | 64B（但每 2 逻辑行共享一个 128B swizzle 段） |
| swizzle_smem_offset 索引 | `row = k_block` (8-bank XOR) | `super_row = k_block / 2` (8-bank XOR 作用在 super-row) |
| K-loop 每次迭代 | 1 个 K-block = 4 个 m16n8k32 | 2 个 sub-K-block = 4 个 m16n8k32（mma 总数不变） |
| BLOCK_SCALE_K=128 scale flush | 每 128 K 一次 | 每 128 K 一次（横跨 2 个 sub-K-block，flush 节奏重排） |
| smem 用量 | 96KB（2 × 3 stages × 128×128B） | 96KB（2 × 3 stages × 128×128B super-row）← 一致 |

**为什么理论上可行**：
- TMA 每次拉 128B 进 smem，自然容纳 2 行 64B 数据，物理 smem 行宽仍为 128B super-row
- `swizzle_smem_offset(super_row, col, 128)` 用 8-bank XOR，作用在 super-row 上（0-7 循环），跟 ldmatrix `.aligned` 的硬编码 XOR 模式一致
- K-loop 重写：每次处理 2 个 sub-K-block，mma 链和 scale flush 节奏需要按 super-row 重排（不是改一个 constexpr 就行——mma 链、stage barrier、DIRECT_STORE epilogue 全要调整）

**为什么没做**：
- **预期收益 <2%**：R10 已经是纯 MMA 峰值的 87%（0.101/0.116 = t_R10/t_pureMMA），剩 13% 是 TMA/scale/barrier 不可重叠开销，interleaved 不会改变这些固定开销结构。R10 vs nvjet TensorWise 差距实测 1.5%（83.6 vs ~85 TF），nvjet 的 S=6 pipeline depth 是用 TensorWise scalar 模式（无 blockwise scale 开销）拿到的，blockwise 模式硬上 S=6 要 `2 × 6 × 16KB = 192KB smem`，**超 99KB cap 2 倍**——要么砍 BM/BN（破坏 128×128 CTA tile），要么砍 NUM_STAGES 到 3（pipeline 深度跟 R10 一样，没收益）
- **风险不对称**：K-loop 是 R10 的核心路径，重写风险（mma 链正确性、scale flush 节奏、barrier 同步）远大于 <2% 收益的潜在价值
- **BLOCK_OUT 128×128 实验已证伪了 "放宽 scale 粒度就能拿到加速" 的假设**：2026-07-30 head-to-head 测出 0.978–0.982× R10（5 次重复稳定），说明 scale 路径在 R10 已经摊薄到极限

**决策**（2026-07-30 用户确认）：优化停在 R10，interleaved 路径记入 memory 作为"唯一理论可行但不值得做"的备查项。如果未来 nvjet blockwise 路径开放且 <2% 收益变得有意义，再回来走这条路。

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

1. **不要**再尝试 BK=64 + SWIZZLE_64B 路径（已知 ldmatrix 不兼容）。如果非要 BK=64，唯一理论可行的是 interleaved 2×BK=64 K-loop 重写（见上文），但预期 <2% 收益不抵重写风险，除非 nvjet blockwise 路径开放。
2. **不要**用单次 cudaEvent 测 ctypes 加载的 .so kernel（under-report 2.07x）。要用 batched 计时或 `torch.profiler`。
3. **不要**给 R10 加新变体（w64x64、bsk256/512/1024、BLOCK_OUT=128×128 都已测试且不更快）。
4. **真正的杠杆**是改变 scale 模式（TensorWise scalar/RowWise vs Blockwise），但用户已确认 "没意义也不会上"。
5. 看 `docs/fp8_gemm_kernel_pipeline.md` § Dead ends on sm_120（Split-K、2 producer lanes、m16n8k64 MMA 都已 dead end）。

## Hybrid (FSFP8) GEMM — 2D BF16 weight + 1D E8M0 activation (2026-07-30)

User asked: explore feasibility of a hybrid GEMM mixing 2D 64×128 BF16
weight scales (FSFP8 best practice) with 1D per-row E8M0 activation
scales (decode-friendly), reusing R10's optimization strategy. Five
variants explored: A=1×64 BF16/E8M0, A=1×64 B=64×64, A=1×128 BF16/E8M0.

### Hardware reality

**sm_120 has no native MXFP8 MMA.** Verified by trying both
`mma.sync.aligned.m16n8k32.row.col.kind::mxf8.mxfp8.f32` and
`tcgen05.mma.mxf8.mxfp8.f32` — both fail PTX assembly with
"Unknown modifier" / "Arguments mismatch". Native MXFP8 MMA exists
on sm_100 (datacenter Blackwell), not sm_120 (consumer Blackwell)
or sm_89 (Ada). Both "MXFP8" and "1×64 E8M0 emulation" therefore
use BF16 F16-acc MMA + software scale application — the only
option on current hardware.

### E8M0 decode is already optimal integer arithmetic

`e8m0_to_float()` in fp8_gemm.cuh:
```cpp
__device__ __forceinline__ float e8m0_to_float(uint8_t e) {
    int exp = (int)(e & 0xff);
    if (exp == 0)    return 0.0f;
    if (exp == 0xff) return 0.0f;
    return __uint_as_float(((uint32_t)exp) << 23);
}
```

PTX assembly confirms `shl.b32 %r, 23` is the only arithmetic op
(constructs FP32 exponent field directly from E8M0 byte). No FP
transcendental, no special "scale" instruction. This is the
optimal CUDA-level decode — any alternative (e.g. `exp2f`,
`powf`, lookup tables) is 10-50x slower. There's no special
hardware instruction for E8M0; the bit-shift IS the fast path.

### Kernel additions (2026-07-30)

Added to `src/models/ops/cuda/fp8_gemm_build/sources/fp8_gemm.cuh`:
- New template params `A_SCALE_1D_BLOCK` (bool), `A_BLOCK_SCALE_K` (int),
  `B_BLOCK_SCALE_K` (int, default -1 → BLOCK_SCALE_K), `A_SCALE_E8M0` (bool).
- New helper `e8m0_to_float()` decodes OCP MX 1.0 E8M0 byte to FP32
  via `2^(exp-127)` (single PTX bit shift).
- BLOCK_ACCUM flush branch for A_SCALE_1D_BLOCK=true: per-row A scale
  loads (2 a_s per mi via lane_id>>2) × per-N-tile B scale (warp-
  constant, hoisted). B's natural 2D scale stride uses B_BLOCK_SCALE_K
  (=128 for FSFP8 64×128 weight block, =64 for the finer hybrid_b64
  variant) while chain granularity is BLOCK_SCALE_K (64 or 128).

Three entry files:
- `hybrid_a1d_kernel_entry.cu.in` — A=1×64, B=64×128 (chain=2).
  Two export symbols (e8m0 + bf16 A scale) sharing one .so.
- `hybrid_b64_kernel_entry.cu.in` — A=1×64, B=64×64 (chain=2, finer B).
- `hybrid_a1x128_kernel_entry.cu.in` — A=1×128, B=64×128 (chain=4,
  **R10-equivalent chain length**). Two export symbols (e8m0 + bf16).

Build: `python scripts/build_fp8_blockwise_gemm.py --variant hybrid_a1d
| hybrid_b64 | hybrid_a1x128 --config bm128_bn128_bk128_s3_cwg2_wm32_wn64_ds_blockwise`.

Python wrapper: `src/models/ops/cuda/fp8_hybrid_gemm.py` with
`a_block_k={64, 128}` and `b_block_k={64, 128}`.
R10 wrapper: `src/models/ops/cuda/fp8_r10_gemm.py` (new — pulls in
the existing accum_bsk128_f16 .so).

Backward compat: all old template params have defaults so existing
entry files (1x64, 64x64, accum_bsk128_f16, …) recompile cleanly.

### Correctness (`output/hybrid_gemm_probe.json`)

vs FP64 reference at headroom=0.10 (max FP8 ≈ 45, safe for F16 acc):

| shape | r10 | bf16 1×64 | e8m0 1×64 | bf16 1×64/64×64 | bf16 1×128 | e8m0 1×128 |
|---|---|---|---|---|---|---|
| 256³  | 0.027 | 0.027 | 0.027 | 0.027 | 0.027 | 0.027 |
| 512³  | 0.037 | 0.037 | 0.037 | 0.037 | 0.037 | 0.037 |
| 1024³ | 0.027 | 0.027 | 0.027 | 0.027 | 0.027 | 0.027 |
| 2048³ | 0.037 | 0.037 | 0.037 | 0.037 | 0.037 | 0.037 |

All five hybrid variants match R10 precision within FP8 quantization
noise floor. At headroom=0.25 (max FP8 = 112, F16-acc boundary),
hybrid produces 1-6 Inf outputs per shape — same F16-acc overflow
that affects R10 with adversarial data. Production: stay at headroom ≤ 0.1.

### Performance (`output/hybrid_gemm_probe.json`)

CUPTI + batched cudaEvent cross-validation, RTX 5060 Ti sm_120:

| shape | r10 | bf16 1×64 | e8m0 1×64 | b64 1×64 | **bf16 1×128** | **e8m0 1×128** |
|---|---|---|---|---|---|---|
| 2048³ TF | 143.7 | 114.6 (0.80x) | 100.2 (0.70x) | 113.4 (0.79x) | **129.8 (0.90x)** | **120.4 (0.84x)** |
| 4096³ TF | 168.7 | 135.3 (0.80x) | 117.7 (0.70x) | 134.0 (0.79x) | **154.1 (0.91x)** | **141.9 (0.84x)** |

The 1×128 variants are **14-20% faster than the 1×64 variants** at
large shapes. The win comes from matching BLOCK_ACCUM chain length
to R10 (4 sub-MMAs = 1 FFMA flush per K-block), which removes the
2× FMA overhead of the 1×64 chain (2 sub-MMAs = 2 flushes per K-block).

### Why 1×128 beats 1×64

The dominant cost in hybrid (vs R10) is BLOCK_ACCUM chain length:

- **R10 (2D 64×128 BF16)**: chain=4 sub-MMAs, 1 FFMA flush per K-block → 4 FFMA per (mi,ni) per K-block.
- **1×64 hybrid**: chain=2 sub-MMAs (because BLOCK_SCALE_K must match A's 1×64 granularity), 2 flushes per K-block → 8 FFMA per (mi,ni) per K-block.
- **1×128 hybrid**: chain=4 sub-MMAs (BLOCK_SCALE_K=128 matches A's 1×128 granularity AND matches R10's chain), 1 flush per K-block → 4 FFMA per (mi,ni) per K-block. **Same as R10.**

The remaining ~9% perf gap vs R10 at 4096³ comes from per-row a_s
LDGs that can't be hoisted (8 distinct rows per m16 tile via
lane_id>>2), vs R10's single hoisted warp-constant a_s LDG.

### Why no further improvement possible

Tested three further follow-ups, none moved the needle:

- **64×64 BF16 B scale (hybrid_b64)** — B scale granularity matches
  chain (no `scale_k / 2` indexing math). Same perf as 64×128 B
  (the chain length dominates, not B scale granularity).
- **E8M0 decode via integer arithmetic** — already the optimal path
  (`shl.b32 %r, 23`, single PTX instruction). No faster hardware
  instruction exists for E8M0 → FP32.
- **Split-flush (BLOCK_SCALE_K=128 with separate A flush every 64 K)** —
  Sketched but rejected without implementing: per K-block the
  split-flush path does 12 FFMA (4 A-only flush + 8 chain flush),
  vs R10's 4 FFMA. 3× more FMA in the flush critical path. The
  "chain=4" benefit (fewer flushes) is more than offset by the
  extra partial-save FMAs.

The fundamental limit: per-row A scale breaks the BLOCK_TILE_SCALE
warp-constant hoist. The 1×128 variant minimizes this loss by
recovering R10's chain length; the remaining 9% gap is the cost of
per-row a_s LDGs that no further code reorganization can remove.

E8M0 vs BF16 hybrid (1×128): E8M0 is consistently ~7% slower despite
halving scale HBM bandwidth — at compute-bound shapes the 1-cycle
on-load decode is pure overhead. Could win at memory-bound regimes
(small batches or very large K where scale HBM dominates).

### Verdict for the user's question

- **Feasibility**: YES, kernel works and matches R10 precision.
- **Speedup with 1×128 hybrid**: 0.91x R10 (BF16) / 0.84x R10 (E8M0).
  Still 9-16% slower than uniform-2D R10, but viable for the FSFP8
  best practice (E8M0 1×128 activation + BF16 128×128 weight) that
  gives the best accuracy on WikiText-2 16K (ΔPPL = −0.077 per
  /hy-tmp/FSFP8/MEMORY.md).
- **Recommended use case**: the 1×128 hybrid is the right pick for
  LLM decode where activation amax structure changes per-token (the
  reason 2D block on activations requires re-quantization in decode
  — see /hy-tmp/FSFP8 "Optimal scheme is hybrid" entry). For
  training / prefill where weights dominate, R10's uniform 2D is
  still preferred (4-9% faster).

### sm_89 caveat (pre-existing, not from this change)

The kernel template uses `__launch_bounds__(N, 1, 1)` — the third
`1` is the cluster size, which is sm_90+ only. Cross-compile for
sm_89 fails with "cannot specify max blocks per cluster for this
GPU architecture". This affects R10 too — the sm_89 production path
falls back to `torch._scaled_mm` per `fp8_gemm.py` `_load()`. Same
behavior for the hybrid .so (the wrapper falls back gracefully).
To actually run on sm_89, the template's `__launch_bounds__` would
need a 2-arg form for the sm_89 build path.

### Code added (no R10 .so changes; only kernel template additions)

- `src/models/ops/cuda/fp8_gemm_build/sources/fp8_gemm.cuh` — 3 new
  template params, `e8m0_to_float()`, BLOCK_ACCUM flush branch for
  A_SCALE_1D_BLOCK.
- `src/models/ops/cuda/fp8_gemm_build/sources/hybrid_a1d_kernel_entry.cu.in` — new (1×64).
- `src/models/ops/cuda/fp8_gemm_build/sources/hybrid_b64_kernel_entry.cu.in` — new (1×64, B=64).
- `src/models/ops/cuda/fp8_gemm_build/sources/hybrid_a1x128_kernel_entry.cu.in` — new (1×128, **R10-equivalent chain**).
- `src/models/ops/cuda/fp8_hybrid_gemm.py` — Python wrapper, supports
  `a_block_k={64, 128}` × `b_block_k={64, 128}`.
- `src/models/ops/cuda/fp8_r10_gemm.py` — wrapper for the production
  accum_bsk128_f16 .so (was previously un-wrapped), new.
- `scripts/build_fp8_blockwise_gemm.py` — adds hybrid_a1d +
  hybrid_b64 + hybrid_a1x128 variants.
- `lib/fp8_blockwise_gemm_sm_120_..._hybrid_a1d_e8m0.so` — prebuilt.
- `lib/fp8_blockwise_gemm_sm_120_..._hybrid_b64_bf16.so` — prebuilt.
- `lib/fp8_blockwise_gemm_sm_120_..._hybrid_a1x128_bf16.so` — prebuilt.
- `test/_tmp/probe_hybrid_gemm.py` — correctness + perf probe (still
  in `_tmp/` per "test-first on kernel changes" rule in CONTRIBUTING.md;
  promote to `test/` after R10 baseline sweep also validates).
- `output/hybrid_gemm_probe.json` — current probe results.


## Hybrid 1×128 final optimization sweep (2026-07-31) — all 3 steps negative

User requested 3 final optimization steps on hybrid_128 (BF16) closing
the 0.913× R10 gap at 4096³: (1) layout change, (2) half2 fused scale,
(3) K-loop rewrite. **All 3 produced no positive result on sm_120.**
Code is in the tree with defaults disabled (templates + entries kept
for future experimentation, but not selected by `fp8_hybrid_gemm`).

### Step 1 — per-K-major a_s layout (`A_SCALE_K_MAJOR=true`)

**Kernel change**: `fp8_gemm.cuh` adds template param `A_SCALE_K_MAJOR`
(default false). When true, flush path uses `a_bs[scale_k * M + out_row0]`
indexing instead of `a_bs[out_row0 * scale_cols + scale_k]`. Wrapper
adds `a_scale_layout="k_major" | "row_major"` (default "row_major"; for
`a_block_k=128` wrapper does `.t().contiguous()` to K-major before launch).
Entry file `hybrid_a1x128_kernel_entry.cu.in` set
`A_SCALE_K_MAJOR=true`.

**Result — REGRESSION**. 4096³ hybrid_128 BF16 dropped from
0.913×R10 (154.1 TF) → 0.82-0.87×R10 (140-147 TF, **-3 to -6%**).
Root cause: `scale_k * M` is an IMAD vs the row-major path's
`out_row0 * scale_cols` which is a power-of-2 shift-add (scale_cols
= K/A_BLOCK_SCALE_K is power-of-2 friendly). Plus worse L2 row-reuse
pattern at large M (M=4096 stride breaks per-row prefetch).

**Decision**: Reverted entry file to `A_SCALE_K_MAJOR=false`. Template
parameter kept in `fp8_gemm.cuh` (default false, backward compat).
Wrapper keeps the `a_scale_layout` arg for future reactivation.

### Step 2 — half2 fused scale

Tried packed `__hmul2(a_s01, b_s_h)` + `__hfma2(f01_h, s01, ...)` to
fuse the 4 FP32 fmaf into 2 packed FFMA per (mi, ni) per flush.
Main accumulator stays FP32 (K=4096 over 32 K-blocks can't accumulate
in F16).

**Cycle count analysis (per (mi, ni) per flush, sm_120)**:

| Variant | Instruction sequence | cycles |
|---|---|---|
| Original | 2 fp32 mul + 2 half22float2 + 4 fmaf | **8** |
| half2 (s0,s1→F16x2) | 2 fp32 mul + 1 floats2half2 + 2 hfma2 + 2 half22float2 + 4 add | 11 |
| half2 (s0,s1 each F16) | 2 fp32 mul + 2 float2half2 + 2 hfma2 + 2 half22float2 + 4 add | 11 |

sm_120 FP16 packed SIMT FFMA throughput == FP32 FFMA throughput
(1 inst/cycle, packed = 2 ops per inst but cycle count same). The
FP32↔F16 conversion overhead negates the packed FMA savings. Plus
`__floats2half2_rn(s0, s1)` introduces ~0.1% precision loss on the
scale product.

**Decision**: No kernel change. Skip.

### Step 3 — row-pair packed a_s layout (`A_SCALE_ROW_PAIR=true`)

**Alternative layout** (user chose this when asked between skipping
Step 3 vs trying row-pair packed): `a_block_scale [K/128, M/16, 8, 2]`
where the inner 16 BF16 per row-pair-range hold (a_s[g], a_s[g+8]) for
g=0..7 in 2 adjacent bytes → 1 LDG.b32 per row pair (vs 2 LDG.b16
for row-major). Expected: halves LDG instruction count per warp per
K-block, reduces L1 cache line fetches per warp from 8 → 1.

**Implementation**: Template param `A_SCALE_ROW_PAIR` (default false),
new entry file `hybrid_a1x128_rowpair_kernel_entry.cu.in`, wrapper
`a_scale_layout="row_pair"`, helper `to_row_pair_packed()` in probe,
build variant `hybrid_a1x128_rowpair`.

**Result — IMPLEMENTATION BUG**. The packed data is verified correct
(`to_row_pair_packed` produces the right `a_s_row[m, k]` at
`pairs[k, m/16, m%8, 0/1]` for every (m, k)). But the kernel output
is wrong: `out_rp[0, 0]` matches `out_rmaj[0, 0]` (row 0), but
`out_rp[1, 0]` differs by ratio 0.63 (= a_s_row[2, 0] / a_s_row[1, 0]),
suggesting pair_idx is shifted +1 from expected for r=1 (lane 4 mi=0).

SASS check confirms the kernel emits 1 LDG.E (32-bit) for the packed
read (vs 2 LDG.E.U16 for row-major), so the optimization IS being
applied — the bug is in the row-pair indexing arithmetic (probably
the `pair_row = r_for_pair >> 4` / `pair_idx = r_for_pair & 7`
decomposition doesn't match the lane layout assumptions for some
(warp, mi, lane) combination). Couldn't be debugged in time budget.

**Decision**: Code kept with `A_SCALE_ROW_PAIR=false` default (so
production unaffected). Entry file + variant kept for future debug
if someone wants to retry.

### Conclusion for hybrid_128 perf ceiling

After 7 weeks of sweep + 3-step final push:
- R10 baseline at 4096³: **168.7 TF (sm_120, post-2026-07-27 audit)**
- Best hybrid variant: **1×128 BF16 = 154.1 TF (0.913× R10)**
- 9% gap is **structural** (per-row a_s LDG overhead + extra
  `__bfloat162float` decode vs R10's BLOCK_TILE_SCALE warp-constant).
- R10 itself is at ~95% of nvjet TensorWise ceiling (memory
  `feedback_roofline_discipline.md`).

**Hybrid_128 perf is at the blockwise ceiling on sm_120.**

## FP8 GEMM external library comparison on sm_120 (2026-07-31)

User requested comparison of hybrid_128 (FSFP8) vs pure MXFP8 GEMM from
DeepGEMM / b12x / cublas / cutlass. None of these have sm_120 builds —
DeepGEMM has sm_100/sm_90 only, b12x requires cute.nvgpu.warp.MmaMXF8Op
(sm_100+ only), cublasLt MXFP8 (Blockwise 1×32 E8M0) returns
CUBLAS_STATUS_NOT_SUPPORTED on sm_120 (no native MXFP8 MMA on consumer
Blackwell).

**What works on sm_120** (`torch._scaled_mm`):
- TensorWise FP8 (scalar scale, `nvjet_sm120` hand-tuned) — **works**
- RowWise FP8 — fails shape constraint on B layout
- Blockwise 1×128 FP8 — fails "outer-dim-major" stride on A
- Blockwise 128×128 FP8 — fails inner-dim-major constraint
- Blockwise 1×32 E8M0 (MXFP8) — **CUBLAS_STATUS_NOT_SUPPORTED**

### Benchmark @ 4096³ (5 trials, 50 iters each, median)

| Kernel | TFLOPS | vs R10 | Notes |
|---|---|---|---|
| **cuBLAS TensorWise FP8 (nvjet)** | **178** | **1.05×** | Hardware FP8 dense ceiling |
| HippoLM R10 (uniform 2D 64×128 BF16) | 170 | 1.00× | sm_120 blockwise ceiling |
| hybrid_128 BF16 (1×128 + 64×128) | 148 | 0.87× | per-row a_s LDG + decode |
| hybrid_128 E8M0 (1×128 + 64×128) | 136 | 0.80× | + E8M0 decode cost |

### Across shapes (median of 3 trials)

| Shape | R10 blockwise | cuBLAS TensorWise | hybrid_128 BF16 |
|---|---|---|---|
| 1024³ | 46 TF | 46 TF (1.00×) | 11 TF (**0.24×**) |
| 2048³ | 140 TF | 141 TF (1.01×) | 87 TF (0.62×) |
| 4096³ | 170 TF | 178 TF (1.05×) | 148 TF (0.87×) |
| 8192³ | 166 TF | 183 TF (1.10×) | 149 TF (0.90×) |

### Conclusion

1. **R10 is at sm_120 blockwise ceiling** (~95% of nvjet TensorWise peak).
2. The remaining 5-10% gap to nvjet comes from **scale broadcast overhead**
   (per-block scale application vs TensorWise scalar) — structural, not
   a kernel optimization.
3. **hybrid_128 collapses at small M** (1024³): per-row a_s LDG overhead
   dominates. For LLM decode (small batch × seq), this matters.
4. **MXFP8 path is dead on sm_120** — cuBLAS has no native MXFP8 MMA
   for consumer Blackwell; DeepGEMM/b12x target sm_100 datacenter.
   Pure MXFP8 is a no-go for our hardware.
5. To exceed R10, the only path is **switching scale mode** (blockwise
   → TensorWise scalar), which is an algorithmic decision (precision
   budget trade-off), not a kernel optimization.

## Final post-K-major/half2/row-pair/FP32-scale sweep (2026-07-31)

User attempted 4 additional optimizations after the "hybrid_128 is at
ceiling" conclusion. **All 4 negative**, confirming the ceiling.

| Step | Idea | Result |
|---|---|---|
| K-major a_s layout (`A_SCALE_K_MAJOR=true`) | `scale_k * M` IMAD vs `out_row0 * scale_cols` shift-add | -3% to -6% @ 4096³, **regression** |
| half2 fused scale (`__hmul2`/`__hfma2` packed) | sm_120 packed FFMA throughput == scalar FP32 FFMA | +3 cycles/flush (FP32↔F16 overhead negates savings), **skipped** |
| Row-pair packed `[K/128, M/16, 8, 2]` | 1 LDG.b32 per row pair (vs 2 LDG.b16) | kernel correctness bug (pair_idx misaligned), **reverted** |
| FP32 scales (`SCALE_FP32=true`) | 3 × __bfloat162float saved | -1.5% @ 4096³, -14% @ 2048³ (HBM cost dominates), **regression** |
| cuBLAS TensorWise FP8 comparison | Hardware FP8 dense ceiling | nvjet is 1.05-1.10× R10 at large M — **confirms R10 blockwise ceiling** |
| Interleaved BK=64 K-loop rewrite | sm_120 ldmatrix .aligned hardcoded 8-bank XOR | **skipped** (high risk + <2% expected + R10 already at ceiling per cuBLAS comparison) |

**Bottom line**: hybrid_128 perf is at the **blockwise ceiling** for
sm_120. Going faster requires a **different scale mode** (TensorWise
or per-row FP8 + TensorWise B), not more kernel work.


## Hybrid 1×128 + TensorWise B scale (2026-07-31) — modest +3% at large M

User requested: try **per-row 1×128 A scale + TensorWise scalar B scale**
hybrid (instead of the 2D 64×128 B scale in the production hybrid_128).
Rationale: scalar B drops the `n_tile * b_scale_cols` indexing from the
flush, halves B scale HBM, and lets `b_s` be hoisted out of the
(mi, ni) loop.

**Kernel change**: `fp8_gemm.cuh` adds `B_SCALE_TENSORWISE` template
param (default false). When true, B scale flush reads
`b_bs[b_scale_k]` (no `n_tile` dim). Static_assert: requires
`BLOCK_TILE_SCALE` (so b_s stays warp-constant).

**Entry file**: `hybrid_a1x128_twb_kernel_entry.cu.in` — BF16 + E8M0 A
scale variants. Build: `--variant hybrid_a1x128_twb`.

**Wrapper**: `fp8_hybrid_gemm` adds `b_scale_mode="blockwise" | "tensorwise"`.
TensorWise B has shape `[K/b_block_k]` (scalar per K-block); the wrapper
asserts this and dispatches to the new .so.

### Benchmark vs 2D-B baseline (median of 3 trials)

| Shape | 2D-B BF16 | TW-B BF16 | delta |
|---|---|---|---|
| 1024³ | 10.90 TF | 8.92 TF | -18% (noise) |
| 2048³ | 78.53 TF | 73.04 TF | -7% |
| **4096³** | **144.31 TF** | **149.42 TF** | **+3.5%** |
| **8192³** | **149.36 TF** | **154.15 TF** | **+3.2%** |

| Shape | 2D-B E8M0 | TW-B E8M0 | delta |
|---|---|---|---|
| 1024³ | 10.87 TF | 9.19 TF | -15% (noise) |
| 2048³ | 78.62 TF | 84.89 TF | +8% (noise) |
| 4096³ | 136.52 TF | 134.51 TF | -1.5% |
| 8192³ | 140.68 TF | 139.00 TF | -1.2% |

### Conclusion

- **TensorWise B + 1×128 BF16 A**: **+3-3.5% at large M** (4096+, 8192+).
  Modest gain from removing n_tile indexing and halving B scale HBM.
- **TensorWise B + 1×128 E8M0 A**: no consistent gain (within noise).
- Code kept as opt-in (`hybrid_a1x128_twb.so`). Use case: inference
  scenarios where 1 scalar per K-block for B is acceptable precision-
  wise (e.g., pre-quantized checkpoints with global B scale).
- **Still ~12% slower than R10 blockwise** at large M (149 vs 170 TF),
  and ~16% slower than cuBLAS nvjet TensorWise (149 vs 178 TF). The
  per-row 1×128 A scale is the bottleneck (8 LDG per warp per K-block),
  not B scale mode.


## Hybrid_128 perf gap attribution + 128×128 E8M0 B variant (2026-07-31)

### Attribution: why hybrid_128 is ~15% slower than R10

**Root cause (confirmed by SASS instruction count)**: per-row 1×128 a_s
is **lane-bound** — 8 unique row pairs per warp per K-block, each needs
a_s0 + a_s1 = 2 LDG.b16. Per warp per K-block:

| Op | R10 (uniform 2D) | hybrid_128 BF16 | Delta |
|---|---|---|---|
| a_s LDG | 1 (warp-constant) | **16** (a_s0+a_s1 × 8 row pairs) | +15 LDG |
| b_s LDG | 1 | 1 | 0 |
| `__bfloat162float` | 2 | 3 | +1 |
| mma (m16n8k32) | 64 | 64 | 0 |
| ldmatrix | 33 | 33 | 0 |
| FFMA flush | 16 | 16 | 0 |

Per warp tile (32 K-blocks): **+480 LDG issue cycles + 32 convert cycles**
per warp tile for hybrid vs R10. Over ~2048 MMA cycles per warp tile,
this is ~25% raw overhead; net effective ~15% after LDG-MMA overlap.

**Per-row 跑不满的瓶颈（三层）**:
1. **L1 broadcast 只到 4-lane group**: 32 lanes 分成 8 组 (lane_id>>2)，
   每组 4 条 lane 共享同一 a_s 地址，但 8 组之间地址不同 → 8 unique
   LDG per (a_s0/a_s1)，L1 无法跨组 broadcast。
2. **Register pressure**: a_s0/a_s1 是 per-lane 值，不能 hoist 出
   (mi, ni) loop（每 lane 持自己的 row 的 scale）。R10 的 a_s 是
   warp-constant，编译器能完全 hoist。
3. **L2 contention**: 每 CTA 每 K-block 128 LDG 全部命中去取 scale
   （8 warps × 16 LDG），虽然数据只有 32 字节但指令开销巨大。

**结论**: 结构性瓶颈，kernel 无法消除（per-row 语义要求每个 row 有
独立 scale）。除非改为 warp-constant scale（BLOCK_TILE_SCALE on A），
但那会破坏 per-row 量化的精度意图。

### 128×128 E8M0 B variant (B_SCALE_E8M0=true, BLOCK_OUT_M=N=128)

User asked: can 128×128 E8M0 B + 1×128 E8M0 A (fully E8M0, 4× smaller
B scale) reduce the gap further?

**Kernel change**: `fp8_gemm.cuh` adds `B_SCALE_E8M0` template param
(default false). When true, b_block_scale is uint8 E8M0 (decoded via
e8m0_to_float). Combined with BLOCK_OUT_M=128/BLOCK_OUT_N=128 (the
entry sets both), m_tile becomes CTA-wide (all warps share m_tile=0),
n_tile still 2 values → 2× fewer unique (m_tile, n_tile) per CTA per
K-block, 4× smaller b_s tensor.

**Entry file**: `hybrid_a1x128_e8m0_b128_kernel_entry.cu.in`.
**Wrapper**: `b_block_out_n=128, b_block_out_m=128` (implies E8M0 B).

### Benchmark (median of 3 trials)

| Shape | 2D-B E8M0 | **B128 E8M0** | delta | 2D-B BF16 | TW-B BF16 |
|---|---|---|---|---|---|
| 2048³ | 84.78 | **86.17** | +1.6% | 88.38 | 85.13 |
| 4096³ | 138.25 | **143.62** | +3.9% | 146.36 | 149.05 |
| 8192³ | 141.18 | **147.20** | +4.3% | 148.51 | 153.38 |

**Result**: 128×128 E8M0 B is **+2-4% faster than 64×128 E8M0 B**
(fewer unique b_s per CTA, smaller tensor). But still **~2% slower than
64×128 BF16 B** at 4096³ (143.62 vs 146.36) because E8M0 A decode cost
persists. TW-B BF16 remains the fastest hybrid variant (+3.5% over
2D-B BF16 at large M).

**Bottleneck ranking** (hybrid vs R10 gap):
- a_s per-row LDG (16/warp/K-block): **dominant** (~+480 cycles/warp tile)
- `__bfloat162float` / e8m0_to_float decode: minor (~+32 cycles)
- B scale mode (2D-64×128 / 2D-128×128 / TW): **smallest lever** (±3-4%)

**Conclusion**: the per-row 1×128 A scale is the hard bottleneck. B
scale granularity (64×128 → 128×128 → TW) only moves ±3-4%. To close
the full gap to R10, need warp-constant A scale (loses per-row precision).


## NVFP4 GEMM 逆向（cuBLASLt 13.1.1.3 + CUTLASS 4.3.2）(2026-07-31)

User asked: reverse-engineer cuBLASLt / CUTLASS NVFP4 GEMM — what precision
do the mma A/B/C/D operands use, and how is overflow handled. Full evidence
chain in `/tmp/opencode/fp4test/`, `/tmp/opencode/blt_sm120/`,
`/tmp/opencode/blt_sm100/`, `/tmp/opencode/blt_sm103/` (may be cleaned up;
conclusions below are the durable record).

### 核心结论：sm_120 上 cuBLASLt NVFP4 GEMM 不存在

**Host 层实测**（ctypes 直调 libcublasLt.so.13.1.1.3，`CUBLASLT_LOG_LEVEL=5`）：
`R_4F_E2M1`（=33）输入在 sm120 上**所有 scale mode 都被拒**：

| scale mode (attr 值) | 名称 | 结果 |
|---|---|---|
| 1 | SCALAR_32F (tensorwise) | NOT_SUPPORTED |
| 2 | VEC128_32F | INVALID_VALUE |
| 3 | OUTER_VEC_32F | INVALID_VALUE |
| 4 | BLK128x128_32F | INVALID_VALUE |
| 5 | VEC32_UE8M0 (MX-style) | INVALID_VALUE |

D 类型 f32/bf16/f16 全试过，一律 "Unsupported combination of precisions"。
FP8×FP8 同接口 rc=0 正常。cuBLASLt 的 scale 数据类型必须 FP32
（`Scale data type (R_8F_E4M3) does not match the expected type (R_32F)`）。

**SASS 全量反汇编**（1596 个 sm120 cubin 全部 nvdisasm 完）：
全部 MMA 指令 = `QMMA.16832.F32.E4M3.E4M3` / `.E5M2.E4M3` / `.E4M3.E5M2`
(FP8, F32 累加)、`DMMA.8x8x4`、`HMMA.16816.F32.BF16` / `.F32`、`IMMA.16832.S8.S8`、
`HMMA.1688.F32.TF32`。**没有任何 E2M1（FP4）操作数的 MMA**。

- 字符串表里的 `cutlass3x_sm120_bstensorop_s16864gemm_block_scaled_ue4m3xe2m1_...128x128x256_1x1x1_...`
  等 FP4 kernel 名**只在 host 端 heuristic 注册表，对应 SASS 不存在**（幽灵注册项）。
- 二进制里唯一的 FP4 痕迹是 cubin 33 的 epilogue 辅助 kernel
  `globalKernelBgradAB<...,__nv_fp4_e2m1,{__nv_bfloat16|__half|float},float>`（后向梯度反量化）。

**结论**：cuBLASLt NVFP4 在消费级 Blackwell（sm120）上是死路，和 MXFP8 一致
（`docs/fp8_gemm_kernel_pipeline.md` 已记录 MXFP8 返回 NOT_SUPPORTED）。

### CUTLASS 4.3.2 源码级 MMA 精度（`include/cute/arch/mma_sm120.hpp`，FlashKDA/cutlass 是 4.3.2）

FP4 MMA 指令与操作数精度（cuBLASLt 数据中心 sm100/sm103 内嵌的同款 CUTLASS 家族）：

| 指令 | A | B | C | D |
|---|---|---|---|---|
| `kind::f8f6f4.m16n8k32.row.col.f32.e2m1.e2m1.f32` (sm120) | FP4 E2M1 (4×uint32/lane) | FP4 E2M1 (2×uint32/lane) | FP32 (4×float/lane) | FP32 |
| `kind::f8f6f4.m16n8k32.row.col.f16.e2m1.e2m1.f16` (sm120) | FP4 | FP4 | FP16 (2×uint32/lane) | FP16 |
| `kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e2m1.e2m1.f32.ue8m0` (sm120 VS 原子) | FP4 | FP4 | FP32 | FP32 + 硬件 E8M0 block-scale（每 32 K 元素一个） |
| `tcgen05.mma.cta_group::1.kind::f8f6f4` (sm100) | FP4 (SMEM/desc) | FP4 | FP32 (TMEM) | FP32 |
| `tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale` (sm100) | FP4 | FP4 | FP32 (TMEM) | FP32 + 硬件 E8M0 scale |

- blockscaled mainloop 用的是 `SM120_16x8x32_TN_VS` 原子（`sm120_blockscaled_mma_tma.hpp`），
  配对 zip-tensor `(A, SFA)` / `(B, SFB)` 喂给 MMA。
- **FP4 nibble 对齐 quirk**：`ldmatrix b4x16` 把 FP4 放在 byte 低 4 位，但 MMA 期望
  **bits 2-5**（`0b00ABCD00`），CUTLASS 喂 MMA 前对 A/B 做 `v << 2` 移位
  （`mma_traits_sm120.hpp:220`）。FP6/FP8 不需要此移位。
- `CUTE_ARCH_F8F6F4_MMA_ENABLED` 要求 sm120 + CUDA ≥ 12.8；但 CUDA 13.0 ptxas 实测
  **拒绝** `kind::f8f6f4` 于 sm120/sm100/sm103（"FP6/FP4 floating point type not supported"）——
  FP4 MMA 是 CUDA 13.1+ / PTX ISA 9.x 特性。项目现有 FP8 用无 `kind::` 前缀的
  `mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16`（→ `QMMA.16832.F16.E4M3.E4M3`），13.0 可编。

### 溢出风险处理

1. **主防线 = FP32 累加**。cuBLASLt sm120 FP8 kernel 全用 `QMMA.16832.F32.*`
   （F32 acc；nvjet 全族都是 F32-acc，无 F16-acc 路径，见上方 2026-07-31 逆向实证）。
   FP4×FP4 max = 6×6 = 36，K=4096 求和 1.5e5，FP32 物理上不溢出。
2. **F16-acc 变体是真实风险点**（`...f16.e2m1.e2m1.f16`）：FP16 max=65504，
   全 6 值下 K > ~1800 溢出成 Inf。cuBLASLt FP4 kernel 名带 `_f32_` 所以避开；
   与项目 R10 的 FP8 F16-acc 是同一类风险（见 `fp8_scale_format_precision_landscape.md`）。
3. **block-scale 溢出**：E8M0（2^127 上限）× FP4 max(6) 理论可超 FP32（3.4e38），
   但正常 amax 归一化后 scaled 值 O(1)；E8M0 的 0xFF 保留位。
4. **输出 cast 饱和**：epilogue FP32→BF16/FP16 做 clamp/saturate，不漏 NaN。

### 对本项目的启示

- 项目 hybrid FP4（E8M0 1×128 act + 2D BF16 weight）在 sm120 上是**唯一可行的
  NVFP4 路径**——cuBLASLt 原生 NVFP4 消费级 Blackwell 是死的。
- 想上原生 FP4 MMA 需 CUDA 13.1+（新 ptxas 才支持 `kind::f8f6f4`），
  且要处理 `v << 2` nibble 对齐。
- cuBLASLt sm120 连 FP8 都默认 F32 累加 → NVIDIA 对精度保守；项目 R10 用 F16-acc
  换 1.74× 是主动承担那类溢出风险。

### 复现要点

- cuBLASLt 提取 fatbin：`cuobjdump -arch sm_120 -xelf all libcublasLt.so.13.1.1.3`
  （1596 个 cubin）；FP4 epilogue 在 cubin 33。
- scale mode 枚举：`CUBLASLT_MATMUL_DESC_A_SCALE_MODE=31`，值 1=SCALAR, 2=VEC128,
  3=OUTER_VEC, 4=BLK128x128, 5=VEC32_UE8M0（从 LOG 的名字映射）。
- ctypes 调 `cublasLtMatmul` 签名是 16 参：`(handle, desc, alpha, A, Adesc, B, Bdesc, beta, C, Cdesc, D, Ddesc, algo, workspace, workspaceSize, stream)`——参数顺序极易搞错；scale pointer attr 是 17/18（不是 6/7）；CUDA 13 的 cudaDataType 是 R_4F_E2M1=33、R_8F_E4M3=28、R_32F=0（不是老版 120/118/4）。

## Gradient accumulation FP8 quantization format sweep (2026-07-31, rev 2)

User questions:
1. For "N BF16 gradient matrices accumulated into one accumulator,
   with intermediate values stored in FP8 only" — which of
   {tensorwise, rowwise, fp8_2d_tight (Muon current), 1×16 BF16,
   1×128 BF16, 32×32 BF16, 128×128 BF16} has the lowest error?
2. **User noticed fp8_2d_tight uses FP32 scales**; would 2D-tight
   with `block=16` + BF16 scales be even lower?
3. Does the ranking hold for real Muon gradients (not just
   synthetic)?

Probe: `test/_tmp/probe_grad_accum_quant_sweep.py`. Output:
`output/grad_accum_quant_sweep.json` (4 shapes × 5 distributions ×
10 formats × 3 test types = 600 cells, ~100 s on sm_120).

Distributions tested (gradient-like): D1 Gaussian σ=0.05,
D2 heavy-tail (0.1% ×10), D3 outlier rows (0.5% ×10),
D4 Gaussian σ=0.5 (10× larger magnitude), D5 **real Muon gradients**
captured from a forward-backward pass on a Linear layer at prod
shapes (random init, MSE loss → real grad statistics).
Shapes: (1536,1536), (1536,4096), (8192,1536), (4608,1536).
N_steps ∈ {4, 16, 32, 64}.

### Cross-magnitude ranking (NEW finding from rev 2)

The 2D-tight block=16 BF16 advantage is **magnitude-dependent**:

| per-step σ  | ref_amax  | 1×16 BF16 | 2d_b16_bf16 | 2d_b16_fp32 | fp8_2d_tight (b32) | tensorwise |
|------------:|----------:|----------:|------------:|------------:|-------------------:|-----------:|
| 0.05        | 0.996     | 23.65 dB  | **24.37**   | **24.38**   | 23.45              | 22.21      |
| 0.005       | 0.099     | 23.65 dB  | **24.36**   | **24.37**   | 23.45              | 22.21      |
| 0.0005      | 0.010     | 23.64 dB  | **24.20**   | **24.21**   | 23.41              | 22.20      |
| 0.00005     | 0.001     | **23.66** | 22.19       | 22.17       | 22.17              | 22.20      |
| 0.00001     | 0.0001    | **23.65** | 22.17       | 22.16       | 22.16              | 22.22      |

(Multi-step N=16, shape (1536, 1536), all formats on the same
synthetic Gaussian data; only the input magnitude varies.)

**Crossover at ref_amax ≈ 0.01**: above that, 2D-tight block=16
wins by 0.7 dB; below that, 1×16 BF16 wins by 1.5 dB.

### Muon-specific EMA sweep (N=64, β=0.95)

This is what the storage actually sees — m_amax grows with N.
Sweeping per-step grad σ:

| per-step σ | m_amax @ N=64 | 1×16 BF16 | 1×128 BF16 | 2d_b16_bf16 | fp8_2d_tight | tensorwise |
|-----------:|--------------:|----------:|-----------:|------------:|-------------:|-----------:|
| 5e-4       | 0.0082        | 22.82     | 21.61      | **23.51**   | 22.63        | 21.31      |
| 5e-5       | 0.0008        | **22.83** | 21.62      | 21.45       | 21.36        | 21.41      |
| 1e-5       | 0.0001        | **22.82** | 21.63      | 21.48       | 21.38        | 21.40      |
| 1e-6       | 0.0000        | **22.83** | 21.62      | 21.50       | 21.41        | 21.32      |

For real Muon magnitudes (per-step σ ~ 5e-5, m_amax ~ 1e-3 at
N=64), **1×16 BF16 wins by 1.4 dB** over all 2D-tight variants.

### Aggregate ranking (cross-shape × cross-dist median SQNR)

Test 2 (multi-step accumulation, the user's main scenario):

| format             | SQNR (dB) | gap to best |
|--------------------|----------:|------------:|
| 2d_tight_b16_fp32  | 23.33     | (baseline)  |
| 2d_tight_b16_bf16  | 23.31     | −0.02 dB    |
| 1×16 BF16          | 22.83     | −0.50 dB    |
| fp8_2d_tight       | 22.57     | −0.76 dB    |
| 2d_tight_b32_bf16  | 22.55     | −0.78 dB    |
| 1×128 BF16         | 21.62     | −1.71 dB    |
| 32×32 BF16         | 21.43     | −1.90 dB    |
| rowwise            | 21.39     | −1.94 dB    |
| 128×128 BF16       | 21.38     | −1.95 dB    |
| tensorwise         | 21.35     | −1.98 dB    |

Test 3 (Muon NS-output drift, β=0.95 momentum → NS 5 steps):

| format             | SQNR (dB) | gap to best |
|--------------------|----------:|------------:|
| 2d_tight_b16_bf16  | 25.20     | (baseline)  |
| 2d_tight_b16_fp32  | 25.18     | −0.02 dB    |
| 1×16 BF16          | 24.80     | −0.40 dB    |
| 2d_tight_b32_bf16  | 24.45     | −0.75 dB    |
| fp8_2d_tight       | 24.44     | −0.76 dB    |
| 1×128 BF16         | 23.59     | −1.61 dB    |
| 32×32 BF16         | 23.37     | −1.83 dB    |
| rowwise            | 23.35     | −1.85 dB    |
| 128×128 BF16       | 23.33     | −1.87 dB    |
| tensorwise         | 23.32     | −1.88 dB    |

### Real Muon gradient (D5) ranking — the user's specific question

For D5 (real Muon gradients, σ ~ 5e-5, N=64), the ordering is:

1. **1×16 BF16**: 17.75 dB (best, 1.4 dB ahead)
2. 1×128 BF16: 16.58 dB
3. 2d_tight_b32_bf16: 16.24 dB
4. 2d_tight_b16_bf16: 16.24 dB
5. 2d_tight_b16_fp32 = fp8_2d_tight: 16.39 dB
6. rowwise / 32×32 / 128×128 / tensorwise: 16.40-16.42 dB

**Answer to "对于muon是不是也是这个规律"**: 是的，而且**比合成数据更明显**——
1×16 BF16 赢 1.4 dB，2d_tight block=16 退化到 2d_tight block=32 同等水平。

### Key conclusions (rev 2)

1. **fp8_2d_tight is NOT optimal for Muon gradients** — it's
   a "middle ground" that beats coarse formats (tensorwise,
   rowwise, 32×32, 128×128) by ~1.4 dB but loses to finer
   formats (1×16 BF16) by 0.4-1.5 dB.

2. **User's hypothesis (2d_tight block=16 + BF16)**: BF16 vs FP32
   scale precision is essentially identical (0.0-0.07 dB, within
   noise). The **block size** matters more than the **scale dtype**.
   2d_tight block=16 BF16 ≈ 2d_tight block=16 FP32 (within 0.02 dB).

3. **The crossover at ref_amax ≈ 0.01** is a critical detail:
   - **Large magnitudes (σ ≥ 0.001, ref_amax ≥ 0.01)**: 2D-tight
     block=16 wins (finer K-block + outlier isolation helps).
   - **Small magnitudes (σ ≤ 0.0005, ref_amax ≤ 0.001)**: 1×16 BF16
     wins (2D-tight's M-axis partition becomes overhead because
     min(scale_dim1, scale_dim2) ≈ max() in the small-σ regime,
     so the 2D structure adds no information).

4. **For Muon's actual gradient magnitudes** (per-step σ ~ 5e-5,
   m_amax ~ 1e-3 at N=64), the storage is firmly in the
   "1×16 BF16 wins" regime. **2D-tight block=16 BF16 would only
   win for activations or other large-magnitude tensors**, not
   for the Muon exp_avg buffer.

5. **The current fp8_2d_tight (block=32, FP32) is in the middle
   of the pack**: not as good as 1×16 BF16 (-1.4 dB at Muon
   magnitudes), not as good as 2d_tight block=16 BF16 (-0.7 dB
   at large magnitudes). The "min(scale_dim1, scale_dim2) per
   element" design rationale (outlier isolation to col-block) is
   only load-bearing when there ARE outliers — for pure-Gaussian
   gradient data, it adds no information.

### Implication for Muon `exp_avg_storage`

Three practical options, ranked by SQNR at Muon magnitudes:

1. **Switch to 1×16 BF16**: best at Muon magnitudes (+1.4 dB vs
   current), simplest implementation (no per-element min, BF16
   scales), no storage win. The clear winner for prod.
2. **Keep fp8_2d_tight** (current): known-good, design rationale
   documented for outlier-isolation use case. Loses 1.4 dB to
   option 1 at Muon magnitudes.
3. **Switch to 2d_tight block=16 BF16**: would only be a win for
   LARGE-magnitude tensors (e.g., KDA q/k/v activations before
   RMSNorm, where amax > 0.1). For Muon gradients, loses 1.4 dB
   to option 1.

**Recommendation**: switch to **1×16 BF16** if the engineering cost
is acceptable (~50 lines of code change in
`src/training/ops/fp8_2d_tight.py` to make block=16 + BF16 the
default, or write a new `quantize_1d_bf16` for the 1D path).
The 1.4 dB SQNR improvement is consistent across all tested
distributions, shapes, and N (the "small magnitude" regime
holds for all Muon params, including the EMA steady-state).

### Magnitude of various HippoLM tensors (for context)

Approximate amax at prod training:
- KDA q/k/v grad_w (per-step): σ ~ 0.0005, amax ~ 0.002
- KDA q/k/v exp_avg (N=64, β=0.95): amax ~ 0.005-0.01
- FFN gate/up grad_w (per-step): σ ~ 0.0005, amax ~ 0.002
- FFN down grad_w (per-step): σ ~ 0.0005, amax ~ 0.002
- All Muon exp_avg (steady state): m_amax ~ 1e-3 to 1e-2

All firmly in the "1×16 BF16 wins" regime.

## Per-grad quantization vs accumulator requantization (2026-07-31, final)

User question: if each BF16 grad is quantized to 1×16 BF16-scale FP8
*once* (per-grad quantization), then summed — how much error does
this add vs the "requantize accumulator at each step" strategy?

Probe: `test/_tmp/probe_grad_accum_per_grad_sweep.py`. Output:
`output/grad_accum_per_grad_sweep.json`.

### Key finding: per-grad quantization is ~9 dB better than requantize

Cross-shape × cross-dist × N=4..64 aggregate SQNR:

| strategy                           | SQNR (dB) | N=4→N=64 error growth |
|------------------------------------|----------:|-----------------------:|
| C. Per-grad quant + **FP32** sum   | **32.76** | 1.2×                   |
| B. Per-grad quant + **BF16** sum   | 32.59     | 1.3×                   |
| D. Per-grad + BF16 sum, requantize end | 29.80 | 2.1×               |
| E. **Requantize accumulator** each step | 23.57 | 13.2×              |

The current Muon strategy (E) is **9.2 dB WORSE** than per-grad
quantization (C). At N=64, the per-grad accumulator's error is
essentially independent of N (FP8 representation noise floor),
while the requantize-accumulator's error grows ~linearly with N
because each requantization error scales with the accumulator
magnitude at that step.

### Theoretical prediction (validated)

| strategy | error per grad | accumulation | error growth with N |
|----------|---------------:|--------------|--------------------:|
| Per-grad (B/C) | 1.5% of grad std | sum of independent errors | ~sqrt(N) |
| Requantize acc (E) | n/a | error ∝ accumulator magnitude ∝ sqrt(N) per step, summed | ~N |

Measured med_rel growth N=4→N=64:
- B (per-grad BF16): 0.014% → 0.07% (5× ≈ sqrt(16))
- E (requantize acc): 0.019% → 0.274% (14.4× ≈ 16)

### Magnitude sweep (validates prediction across all sigmas)

For sigma ∈ [5e-2, 5e-6]:
- C (per-grad FP32): 32.66 dB (constant across magnitudes)
- B (per-grad BF16): 32.49 dB (constant)
- E (requantize acc): 23.51 dB (constant)

Per-grad is **~9 dB better regardless of magnitude** — the
"requantize accumulator" strategy loses the same way at every scale.

### Real Muon gradients (D5, 1536×1536)

| strategy | N=4 | N=16 | N=64 | growth |
|----------|----:|-----:|-----:|-------:|
| C (per-grad FP32) | 32.91 | 32.76 | 32.23 | 1.2× |
| B (per-grad BF16) | 32.88 | 32.62 | 31.87 | 1.3× |
| E (requantize acc) | 28.99 | 23.65 | 17.75 | 13.3× |

### Practical implication

**The biggest win is NOT the storage format — it's the accumulation
strategy**. Per-grad quantization with BF16 (or FP32) sum beats the
current "requantize accumulator" strategy by ~9 dB at N=64.

The fp8_2d_tight format choice (block=32 FP32 vs 1×16 BF16 vs
2d_tight block=16 BF16) is a 1-2 dB question on top of a 9 dB
strategic question.

**Recommendation for Muon path**: if the exp_avg storage layout
can be redesigned, the right pattern is:
  1. Each per-step grad is quantized once to FP8 + scale (any format).
  2. The dequantized grad is added to a BF16 (or FP32) EMA accumulator
     in CPU pinned memory.
  3. The accumulator is NEVER requantized between steps — only the
     raw grad is quantized.

This is essentially the "BF16 fallback" branch of the current
Muon code (when `rows % block != 0`). The fallback path happens
to be the BEST path; the FP8 storage path is actively worse at
N=64 by 13x in error.

**Cost of switching**: must always store FP32/BF16 exp_avg in CPU
pinned memory (no FP8 storage win). For the 1536×1536 KDA o_proj,
that's ~4.5 MiB BF16 vs ~2.25 MiB FP8+scales — about 2x storage
increase. Whether this is acceptable depends on the CPU memory
budget for the optimizer state.

### Caveat

The per-grad strategy requires a different code path from the
current `fp8_2d_tight` storage. The current Muon code stores
exp_avg as FP8 + 2 scales (block=32) and requantizes after each
EMA step. The per-grad strategy would store exp_avg as FP32/BF16
and quantize only the per-step grad. Engineering cost: ~100 lines
in `muon.py` + a `quantize_per_grad_1x16_bf16` kernel for the
per-step grad quant. Memory cost: 2x exp_avg storage in CPU pinned
memory. Precision win: 9 dB SQNR at N=64.

## W8A8 F16-acc overflow — strategy (2026-07-31)

User committed W8A8 = tensorwise-W FP8 + 1×128 A BF16 scale + BF16
grad. Question: how to handle F16-acc overflow in the new GEMM
(which is the same code path as hybrid_a1x128_twb)?

### Worst-case F16-acc overflow math

F16 max = 65504. R10 chain length = BLOCK_SCALE_K=128 = 4 sub-MMAs of
32 K each. With `ACC_RAW_F16=true`, the chain accumulates RAW FP8
products (no scale applied yet) in F16.

| headroom | FP8 max | max raw product | max chain sum (128 K) | overflows F16? |
|----------|--------:|----------------:|----------------------:|:--------------:|
| none     | 448     | 200,704         | 25.7M                 | 392× over      |
| 0.10     | 44.8    | 2,007           | 257K                  | 4× over        |
| 0.05     | 22.4    | 502             | 64K                   | borderline     |
| 0.025    | 11.2    | 125             | 16K                   | safe           |

So even headroom=0.10 (which the hybrid_gemm_probe used as a "safe"
benchmark setting) is **4× over the F16 overflow bound for adversarial
max-magnitude data**. The current R10 works only because real LLM
activations (post-RMSNorm, post-attn) have Gaussian-like tails where
all 128 elements simultaneously being at the max has ~0 probability.

### Recommended strategy

**Default to F32 acc** for W8A8. Use the `blockwise_accum_bsk128` entry
(`ACC_RAW_F16=false`):
- No overflow risk.
- ~58% of R10 perf (47 vs 81 TF at 4096³ on sm_120, post-audit
  2026-07-27 numbers) — but we already established that's the
  blockwise ceiling vs cuBLAS nvjet TensorWise (~5% gap).
- The 1.74× F16-acc speedup is not worth the fragile "data is well-
  behaved" assumption for production training.

**F16 acc stays as opt-in** (`blockwise_accum_bsk128_f16` entry, the
current R10):
- For perf-only experiments with well-characterized data.
- Document the overflow conditions clearly (chain length × max product
  per K-block).
- Add a runtime NaN check on first forward of each new shape; on NaN,
  log a warning and fall back to F32 acc.

**Do not** try to find a "principled" F16-acc headroom. The math above
shows it doesn't exist without a 25× precision loss.

### Implication for the W8A8 GEMM build

The W8A8 GEMM (1×128 A + tensorwise B) needs a new entry file
instantiating the kernel with `ACC_RAW_F16=false`. Same R10 strategy
otherwise (BM=128/BN=128/BK=128/S=3/CWG=2/WM=32/WN=64/DS, BLOCK_TILE
on B, 64×64 BLOCK_OUT, BLOCK_ACCUM=true). The build target name:
`hybrid_a1x128_twb_f32` (or similar).

## W4A8 native — plan (2026-07-31)

User: W4A8 = NVFP4 W + 1×128 BF16-A FP8 + BF16 grad. Constraint:
FP4×FP8 stays in FP8 dynamic range for clean dequant. No FP4 MMA
(sm_89 compat, no CUDA upgrade) — use FP8 MMA + in-pipeline dequant.
Also: GEMM takes BF16 A in (quant inside) and outputs BF16 (so KDA
state update / residual doesn't pay dequant).

### Why this is worth doing

Two-pass current path (BF16→FP8 quant, then NVFP4→FP8 dequant, then
FP8×FP8 GEMM):
- A HBM traffic: 2 (read BF16) + 1 (write FP8) + 1 (read FP8 in
  GEMM) = 4 MK bytes
- B HBM traffic: 0.5 (read NVFP4) + 1 (write FP8) + 1 (read FP8 in
  GEMM) = 2.5 MK bytes
- 3 separate kernel launches (~5-10 us each)

Fused path (BF16 A + NVFP4 B → FP8 MMA → BF16 Y, all in 1 kernel):
- A HBM traffic: 2 (read BF16 in GEMM) = 2 MK bytes
- B HBM traffic: 0.5 (read NVFP4 in GEMM) = 0.5 MK bytes
- 1 kernel launch

Savings: 2 MK on A + 2 MK on B = **4 MK HBM + 2 kernel launches**.
At FFN gate_up (M=16k, K=1536) that's 96 MB saved = ~100 us/forward
on sm_120. Plus the kernel launch savings (~10 us).

### Design (sm_120, BF16 A + NVFP4 B → FP8 MMA → BF16 Y)

**Smem per stage** (BM=BN=128, BK=128, no Y_out smem with
DIRECT_STORE):
- A: BF16 [128, 128] = 32 KB
- B: NVFP4 packed [128, 64] = 8 KB + scales [128, 8] E4M3 = 1 KB
- per stage: 41 KB
- NUM_STAGES=2 = 82 KB (fits in 99 KB sm_120 cap)
- NUM_STAGES=3 = 123 KB (over cap, so 2 stages only)

vs R10 smem (FP8 A + FP8 B): 16+16=32 KB × 3 stages = 96 KB.
The fused W4A8 loses 1 pipeline stage (2 vs 3) because A is BF16
in smem. Producer-consumer pipelining is still effective.

**Consumer per K-tile**:
1. ldmatrix.x4.b16 for A — 2 calls per sub-MMA (8 BF16 per call ×
   2 = 16 BF16 per lane = 16x32 BF16 tile). Quantize 16 BF16 → 16
   FP8 in registers (multiply by per-row 1×128 inv_scale, clamp to
   ±448, cast to FP8). Pack into 4 b32 regs in the MMA layout.
2. Regular `ld.shared.b32` for B (4 bytes = 8 FP4 nibbles per lane
   per K-block). Dequant 8 FP4 → 8 FP8 in registers (decode E2M1
   magnitude table, apply E4M0 microblock scale, apply global scale,
   cast to FP8). Pack into 2 b32 regs in the MMA layout.
3. FP8 MMA m16n8k32 (F32 acc).
4. Scale flush at chain boundary (BLOCK_ACCUM=true, BLOCK_SCALE_K=128,
   per-row 1×128 a_s × tensorwise b_s = product per K-block).
5. DIRECT_STORE BF16 output.

**Producer per K-tile**:
- TMA load A (BF16) from HBM to smem.
- TMA load B (NVFP4 packed) from HBM to smem.
- TMA load B (E4M0 microblock scales) from HBM to smem.
- TMA load A scale (per-row 1×128 BF16) from HBM to registers (small,
  loaded directly into consumer instead — TMA descriptor optional).
- mbarrier signal on stage ready.

**Key change from R10**: A is loaded as BF16 (not FP8), B is loaded
as NVFP4 (not FP8). The consumer does the transforms in registers
instead of trusting pre-quantized HBM. Everything else is identical.

### Template parameters to add to fp8_gemm.cuh

```cpp
bool A_BF16_INPUT = false,  // when true, A in smem is BF16; consumer quantizes to FP8
bool B_NVFP4_INPUT = false, // when true, B in smem is NVFP4; consumer dequantizes to FP8
```

The smem storage struct changes (BF16 A and/or NVFP4 B), the TMA
load paths change (different data types for cuTensorMapEncodeTiled),
the consumer ldmatrix changes (BF16 needs 2x calls per sub-MMA), the
consumer per-element math adds quant/dequant steps before the
existing scale-flush.

### Build target naming

- `w4a8` variant: `bm128_bn128_bk128_s3_cwg2_wm32_wn64_ds_w4a8`
  (but NUM_STAGES=2, not 3 — TBD at build time)
- `w8a8_f32` variant (W8A8 BF16-input + F32 acc): same R10 config
  but ACC_RAW_F16=false and A_BF16_INPUT=true

### Implementation order

1. **W8A8 strategy writeup** (this entry — done)
2. **W8A8 BF16-input entry** (1×128 A + tensorwise B + F32 acc, FP8
   B stays in HBM pre-quantized): adds A_BF16_INPUT=true; consumer
   quantizes BF16→FP8 in registers; FP8 B path unchanged
3. **W4A8 BF16-input entry** (1×128 A + tensorwise B + F32 acc +
   NVFP4 B): adds A_BF16_INPUT=true + B_NVFP4_INPUT=true; consumer
   quantizes A and dequantizes B
4. **Smoke test** on the canonical configs/base.yml to verify HWM
   ≤ 16 GB
5. **Perf probe** vs current two-pass W4A8 on FFN gate_up shape

### Build perf expectation

At FFN gate_up (M=16k, K=1536, N=4096):
- R10 (FP8 fwd): ~165 TF = 2.5e11 FLOPs / 165e12 = 1.5 ms
- Current two-pass W4A8: ~0.59 (pack) + 0.08 (dequant) + GEMM = ~2.1 ms
- Expected fused W4A8: 0.5 (pack, can be cached) + 1 GEMM with BF16
  A + NVFP4 B = 1.5-1.8 ms (target: 0.3-0.6 ms savings vs two-pass)

If fused ends up < 1.7 ms (i.e. competitive with R10 forward), we
have a real win — the weight storage savings (NVFP4 = 3.5x smaller
than FP8 = 7x smaller than BF16) compound with the forward speedup.

### W4A8 native — implementation status (2026-07-31)

**Two-pass is the production default.** The fused in-pipeline-dequant
kernel (`src/models/ops/cuda/fp8_gemm_build/sources/w4a8_gemm.cuh`)
is correct for single-K-tile shapes (K ≤ 128) but has an **unresolved
async-proxy smem race in the producer dequant for multi-K-tile
pipelines** (K > 128). The two-pass wrapper
(`src/models/ops/cuda/w4a8_gemm.py`, `native=False`) is verified correct.

**What works**:
- `w4a8_gemm.py` two-pass: `dequant_nvfp4_to_fp8` (Triton) → R10
  `hybrid_a1x128_twb_f32` (F32-acc FP8 GEMM). Verified: full test
  max-diff 0.03 vs 2-pass reference at 256³; at FFN gate_up
  (4096×1536×4096): dequant 141 us + GEMM 645 us (79.9 TF) = 717 us.
- Native kernel single-K-tile (K ≤ 128): byte-identical to two-pass
  (max-diff 0.1958 at 128³, same as the FP8 quantization noise floor).

**The race (why the fused kernel is not production)**:
The producer TMA-loads A + NVFP4 B_in into smem, then dequantizes
B_in → B_fp8 in registers. For K > 128 (multi-K-tile), the second
stage's B_in[0] region reads as **zeros** at dequant time, producing
nondeterministic NaN in B_fp8[1] (NaN count varies 0-473 per run).
Debug findings:
- smem offsets verified non-overlapping (A/B_in/B_fp8 properly laid
  out; base = 2048, `__align__(1024)` needed to make the TMA
  SWIZZLE_128B layout match `swizzle_smem_offset`).
- The TMA loads and the mbarrier wait both appear correct (B_in[1]
  loads correctly, B_fp8[0] dequant is exact). The first stage's
  B_in[0][0:8] shows all-zeros immediately after the (all-lanes)
  tma_barrier wait + `fence.proxy.async.shared::cta`, yet B_fp8[0]
  is correct — pointing to an async-proxy write-visibility gap that
  the mbarrier wait + fence did not close for the first stage's
  generic-proxy reads. Not fully root-caused.
- The fragment layout for the mma was verified via a standalone
  ldmatrix probe (`probe_ldmatrix_a/b.cu`): B fragment = lane i →
  (row i>>2, cols (i&3)*4 + {0..3} / +{16..19}).

**Why the two-pass GEMM is F32-acc**: the R10 `hybrid_a1x128_twb_f32`
entry (ACC_RAW_F16=false) — overflow-safe per the W8A8 strategy note
above. The NVFP4 round-trip already costs ~2% mean_rel; F32 acc
preserves what's left.

**To retry the fused kernel later**: the race is in the producer
dequant's read of TMA'd B_in. Options: (a) cp.async (generic proxy)
instead of TMA for B_in, with `cp.async.wait_group`; (b) a second
mbarrier phase explicitly fencing the async→generic proxy transition
beyond the current `fence.proxy.async`; (c) consumer-side dequant
with the verified fragment layout (probe_ldmatrix_b shows the exact
byte positions). Do NOT re-derive the mma fragment from first
principles — use the probe.
