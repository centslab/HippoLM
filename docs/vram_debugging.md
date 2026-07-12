> **Skill**: [`run-smoke-test`](../.claude/skills/run-smoke-test/SKILL.md) · **Rule**: [`saved-tensors-not-hwm`](../.claude/rules/saved-tensors-not-hwm.md) — VRAM 改动以 `max_memory_allocated()` 为准。

# VRAM Debugging Methodology

This document captures the methodology, tools, and per-component
breakdown for `python scripts/train.py --config configs/base.yml`
(H=1536, L=32, num_blocks=8, V=248320, B=1, T=262144, **NVFP4 mode-3
on**: `ffn_nvfp4: true` + `ffn_nvfp4_marlin: true` +
`ffn_nvfp4_no_bf16_master: true`, 5060 Ti 16G target).

**Audited 2026-07-06.** Earlier "RMSNorm.saved = 1.78 GiB" attribution
in `/tmp/vram_report.md` was a misclassified bucket — the real content
was the KDA FP32 `g_cumsum` saves misattributed by the frame-based
classifier. See §6 "Audit findings" below and
`project_opt5_rmsnorm_residual_skip.md` in auto-memory for the full
evidence chain.

**Re-audited 2026-07-10.** The opt-5/2 BF16-only state was superseded
by NVFP4 mode-3 (FP4 packed buffers ARE the source of truth; no BF16
master on CPU/GPU). See §10 "Re-audit 2026-07-10: NVFP4 mode-3 baseline"
for the corrected breakdown. §2 / §3.5 / §3.10 / §5.7 retain the
opt-5/2-era numbers; §10 is the live production baseline.

**Re-audited 2026-07-11 (rev 3).** ckpt structure revised to honor
the "2 rounds of recomputation per block" design intent. Each
non-last block is split into **2 sub-blocks of 2 layers each**, each
sub-block wrapped in its own `torch.utils.checkpoint.checkpoint`
call. bwd walks: re-fwd sub-block 0 → bwd L0,L1 → release → re-fwd
sub-block 1 → bwd L2,L3 → release. **Peak during non-last block
bwd = 2 layers' saves alive** (2368 MiB KDA-style dynamic state at
T=16384). Last block: per-layer ckpt for layers 28-30 + un-ckpt'd
layer 31 (unchanged). Total: 17 ckpt() invocations per fwd (was 10
at block-level rev 2, was 31 at sbs=1 rev 1). Peak: alloc 9984
MiB / reserved 10982 MiB / driver 11210 MiB at base.yml on 5060
Ti. The "sub_block" inner loop is reintroduced (with a hardcoded
`_sub_block_layers = 2` constant); the `sub_block_size` attribute
remains removed.

## When to reach for this

Use these tools when:
- An OOM happens during training (need to find the offending tensor).
- Bumping `seq_len` or `micro_batch_size` and need to know what fits.
- A new optimization is claimed to save X MiB and you need to verify
  empirically before merging.
- `nvitop` or `nvidia-smi` shows a peak that doesn't match your mental
  model of the code.

## Tool stack (all under `tools/vram_profile/`)

| Tool | Module | Purpose |
|------|--------|---------|
| **profile** | `tools.vram_profile.profile` | Run the production training loop with `torch.cuda.memory._record_memory_history` enabled; dump per-phase snapshots to a pickle + per-phase JSON log. |
| **analyze** | `tools.vram_profile.analyze` | Offline read of the pickle; size-bracket histogram + innermost-frame frequency table (1-deep, 2-deep, 3-deep). |
| **classify** | `tools.vram_profile.classify` | Offline read; bucket by coarse component purpose (model weights / saved tensors / last-block cache). **Has known misclassification bug — see §6.** |
| **saved_tensor_probe** | `tools.vram_profile.saved_tensor_probe` | Small-scale (H=512, T=2048) probe that hooks `FunctionCtx.save_for_backward` to capture every saved tensor at forward time with dtype+shape+bytes. |
| **saved_tensor_probe_base** | `tools.vram_profile.saved_tensor_probe_base` | Same hook at production H=1536, T=2048 (the probe that won't fit full T=16384 in 16 GB). |
| **vram_peak** | `tools.vram_profile.vram_peak` | A/B harness: run the model twice (with/without a candidate change) and compare `torch.cuda.max_memory_allocated()` across 3 scales. The empirical verifier — never trust the classifier without it. |
| **poll_prod** | `tools.vram_profile.poll_prod` | High-frequency (default 200 Hz) background poll of `mem_get_info()` during a live run. Catches transient peaks that synchronous snapshots miss. |
| **rmsnorm_math_check** | `tools.vram_profile.rmsnorm_math_check` | Verifies that the y-based RMSNorm backward is bit-equivalent to the x-based one (the mathematical premise behind opt-5/1). |

### Why this set?

`profile` writes a snapshot of the allocator state at every chunk
boundary (fwd_in / fwd_out / bwd_in / bwd_out / flush_out). These
snapshots miss the **transient peaks** between fwd_end and bwd_end,
when both the fwd saved-tensors AND the bwd's newly-allocated grad
buffers coexist. `poll_prod` catches that, but only at the coarse
`mem_get_info()` granularity — it cannot tell you which tensor caused
the spike.

The right workflow:
1. **`saved_tensor_probe_base`** at the relevant dim scale to get
   the actual list of saved tensors per Function (dtype+shape+bytes).
2. **`profile`** to find the steady-state peak and per-component breakdown.
3. **`analyze`** + **`classify`** to attribute ownership to specific
   code paths (verify against the `saved_tensor_probe` output).
4. **`vram_peak`** to A/B-verify any optimization candidate empirically.
5. **`poll_prod`** if the steady-state peak doesn't match an external
   observation (e.g. `nvitop` reads higher than your snapshot's
   `max_driver`).

## Reproducing the standard profile

```bash
cd /hy-tmp/HippoLM

# 1. Production profile (2 steps, 16 chunks each, T=262144).
python -m tools.vram_profile.profile \
    --steps 2 --n-chunks 16 --chunk-size 16384 \
    --dump /tmp/vram_prod.pickle \
    --log /tmp/vram_prod.log

# 2. Size + frame histogram of peak snapshots.
python -m tools.vram_profile.analyze /tmp/vram_prod.pickle | head -200

# 3. Component-purpose bucketing (see §6 caveat).
python -m tools.vram_profile.classify /tmp/vram_prod.pickle --focus peak

# 4. Interactive MemoryViz (timeline view in browser).
python -m torch.cuda._memory_viz /tmp/vram_prod.pickle

# 5. High-frequency poll (catches transient peaks).
python -m tools.vram_profile.poll_prod --steps 1 --poll-hz 500
```

The `tools/vram_profile/profile.py` script's `init_tp` defaults to
`backend="gloo"` (single-GPU dev). For production-accurate driver
peak estimates, edit the `init_tp(...)` call to use `backend="nccl"`
(see "Caveats" below).

## Reading the output

`profile.py` writes a `<log>.json` with one record per phase boundary:

```json
{
  "label": "step0.chunk2.bwd_out",
  "alloc_cur": 3794255872,        // bytes live at this snapshot
  "alloc_peak": 9909092352,       // torch's allocated_bytes.all.peak (since reset)
  "reserved_cur": 12998656,       // torch's reserved_bytes.all.current
  "reserved_peak": 13002342400,   // torch's reserved_bytes.all.peak
  "driver_used": 13257945088,     // mem_get_info: bytes held by CUDA driver
  "driver_total": 16619929600,    // total GPU memory
  "inactive_split": 778567680,    // torch's inactive_split_bytes.all.current
  "num_alloc": 0,                 // num_alloc_retries (failed allocs)
  "num_ooms": 0,                  // OOM count
  "t": 1234.567                   // seconds since script start
}
```

Key fields:
- `alloc_cur` — sum of all live tensors. **This is the actual peak VRAM**
  for the active chunk's phase.
- `reserved_cur` — what torch's caching allocator has reserved from the
  driver. `reserved_cur - alloc_cur` = the **slack** (freed segments the
  allocator is holding onto for future re-allocation).
- `driver_used` — what the CUDA driver has actually allocated.
  `driver_used - reserved_cur` = overhead from non-torch pools
  (cuBLAS workspaces, CUDA context).
- `inactive_split` — segments the allocator has carved off for a
  specific expected re-alloc; subset of the slack.

## Per-component breakdown (audited 2026-07-06)

All numbers below are at production dims (H=1536, T=16384, n_layers=32,
n_blocks=8, TP=1, **`ffn_nvfp4: false`**). Bucket source frames come
from `tools.vram_profile.classify`'s manual re-bucketing by
**innermost-frame filename**, plus the empirical probes
(`saved_tensor_probe_base` + `vram_peak`).

| Bucket | Live size | % | Source frame | What it is |
|--------|----------:|--:|--------------|------------|
| **(no-frames) = model weights** | **2.452 GiB** | 25.4% | n/a | embed (727.5 MiB) + 32 × KDA+norms (~627 MiB) + 32 × FFN BF16 master (1.13 GiB). See §3.10 walk. NVFP4 buffers are GONE since opt-5/2. |
| **KDA.intermediates** | **2.135 GiB** | 22.1% | `chunk.py:102` `ChunkKDAFunction.forward` | Per-layer saves: 9 BF16 [B,T,H,K] @ 48 MiB + 2 BF16 [B,T,H,BT] @ 24 MiB + 1 BF16 5D h @ 96 MiB + **1 FP32 `g_cumsum` @ 96 MiB** + tiny ≈ **672 MiB / layer**. Two sub-blocks' worth alive during ckpt bwd + layer 31 un-checkpointed. **The 96 MiB FP32 was the "fp32 RMSNorm residual_out" misclassification of earlier drafts** (see §6). |
| **FLCE dw** | **0.776 GiB** | 8.0% | `fused_linear_cross_entropy.py:433` | `dw = torch.zeros_like(weight, ...)` — embed gradient `[V, H]` bf16 (728 MiB) + `dx` `[N, H]` (48 MiB). **Label is misleading**: the bigger 728 MiB block is `dw`, not `dx` (line 427's `dx` is only 48 MiB). Cleaned by per-chunk `empty_cache`. |
| **AttnRes.saved** | **1.008 GiB** | 10.4% | `attn_res.py:245 forward` | 6 boundary stacks `[N, B, T, H]` where N ∈ {1..6}: 48+96+144+192+240+288 = 1008 MiB. |
| **RMSNorm.saved** *(corrected)* | **0.193 GiB** | 2.0% | `layernorm.py:770` + `fused_norm_gate.py:670` | 2 × LayerNormFunction (attn + mlp of layer 31) × 48 MiB BF16 + 1 × LayerNormGatedFunction (o_norm of layer 31) × 96 MiB BF16. **BF16 references, NOT fp32 casts.** See §3.3. |
| **Checkpoint.input** | **0.672 GiB** | 6.9% | `torch/utils/checkpoint.py` (CheckpointFunction) | 14 sub-block inputs (blocks 0-6 × 2 sub-blocks/block) × 48 MiB BF16. Inner Functions DON'T call save_for_backward during the ckpt-wrapped fwd (wrapped fn runs in `no_grad()`). |
| **KDA.wy_recompute** | **0.348 GiB** | 3.6% | `wy_fast.py:267/269 recompute_w_u_fwd` | Layer 31's recompute workspace (KDA's w/u + the recompute tensor). |
| **AttnRes.einsum** | **0.338 GiB** | 3.5% | `functional.py:373 einsum` | AttnRes's softmax @ V scores @ V intermediates (per-block summaries). |
| **FFN.bf16_matmul** *(was FFN.nvfp4_matmul pre-opt-5/2)* | **0.256 GiB** | 2.6% | `tp_layers.py:294 forward` (F.linear) | Layer 31's `gate_up_proj` output `[16384, 8192]` bf16 = 256 MiB. NOT NVFP4-dequant workspace — those buffers don't exist on GPU since opt-5/2. |
| **Linear.TP_allreduce** | **0.240 GiB** | 2.5% | `tp_layers.py:348 forward` | TP all-reduce buffers (96 MiB × 3 layers). |
| **FFN.silu / swiglu** | **0.256 GiB** | 2.6% | `functional.py:2371 silu` + `swiglu.py:67` | Layer 31's silu(gate) + silu*up = 128 MiB each. |
| **KDA.gate_cumsum** *(last-block)* | **0.120 GiB** | 1.2% | `gate.py:443 kda_gate_chunk_cumsum` | Layer 31's g_cumsum cumsum intermediate (96 MiB FP32, the bigger view-tracked one). |
| **Layer.residual_save** | **0.096 GiB** | 1.0% | `layer.py:95 forward` | Layer 31's input + 1 other (the un-checkpointed last layer's residual). |
| **KDA.chunk_delta_h + l2norm + chunk_kda_fwd** | **0.121 GiB** | 1.3% | `chunk_delta_h.py:700` + `l2norm.py:157` + `chunk.py:898` | Layer 31's per-sub-chunk delta_h + L2-norm + per-chunk KDA small state. |
| **KDA.l2norm + (other)** | **0.020 GiB** | 0.2% | various | softmax intermediates + tiny buffers. |
| **Total attributed** | **8.918 GiB** (66.9% of 9.673) | | | |

The 0.755 GiB gap to 9.673 GiB peak is the **transient allocator
fragmentation** between the synchronous fwd_out snapshot and the
deepest point in fwd. High-freq poll confirms a transient peak at
10.715 GiB (chunk 14), absorbed by the reserved pool slack.

### Why this isn't a memory leak

The per-chunk peak is bounded (chunks 2..15 all hit the same 9.673 GiB).
The KDA states are **detached between chunks** (truncated BPTT). Per
chunk end: `flush_pending_grads` + `flush_manual_flush_params` +
`torch.cuda.empty_cache()` returns the slack to the driver, so the
between-chunks alloc floor is ~2.5 GiB (model weights only).

## §3 Evidence chains

All formulas assume production config: B=1, T_chunk=16384, H=1536,
V=248320, I=4096, n_layers=32, n_blocks=8, NC=4, TP=1, num_heads=12,
head_dim=128, expand_v=1.0, head_v_dim=128. KDA / RMSNorm saves
**verified by** `tools.vram_profile.saved_tensor_probe_base` at
H=1536, T=2048 (T=16384 doesn't fit in 16 GB).

### §3.1 Embed (`src/models/tp_model/embed.py`)

**Output shape** `[B, T, H]` bf16:
```
16384 × 1536 × 2 = 48 MiB
```

**Per-layer parameter** (tied to lm_head):
```
V × H × 2 = 248320 × 1536 × 2 = 727.5 MiB
```

**Saved tensors** (`_ShardedEmbedLookup.forward`, line 62):
`ctx.save_for_backward(input_ids, local_ids_safe, mask)` — all small
(`[B, T]` int64 / int64 / bool, <1 MiB total). The replicated `[B, T, H]`
output is the autograd graph node consumed by the first layer.

### §3.2 AttnRes (`src/models/ops/attn_res.py`)

**Per-block boundary input** (saved by `attn_res.py:245 forward`):
```python
V = torch.stack(blocks, dim=0)  # [N, B, T, H]
```
where N is the number of completed block representations at the
boundary. Per boundary stack-size at base:

```
boundary 1: N=1 →  48 MiB
boundary 2: N=2 →  96 MiB
boundary 3: N=3 → 144 MiB
boundary 4: N=4 → 192 MiB
boundary 5: N=5 → 240 MiB
boundary 6: N=6 → 288 MiB
```
**Live at fwd_out of the deepest chunk point**: 6 of the 7 boundaries
have live inputs at peak (the last one is consumed as the input to
block 8). Sum = 1008 MiB = **1.008 GiB** (the §2 entry).

### §3.3 RMSNorm / FusedRMSNormGated (`src/models/norms.py` + vendored `layernorm.py` / `fused_norm_gate.py`)

**The "fp32 residual_out" attribution of earlier drafts is wrong.**
When this project calls RMSNorm with `residual=None` (the case for
`attn_norm`, `mlp_norm`, `final_norm`, and `o_norm`), and
`residual_in_fp32=False`:

`fused_norm_gate.py:466-469`:
```python
if residual is not None or (residual_dtype is not None and residual_dtype != x.dtype):
    residual_out = torch.empty(T, D, device=x.device, dtype=residual_dtype)
else:
    residual_out = None
```
`fused_norm_gate.py:524`:
```python
return y, mean, rstd, residual_out if residual_out is not None else x
```

`layernorm.py:563-566`: identical structure.

So in `LayerNormGatedFunction.forward` (`fused_norm_gate.py:658-670`),
when `residual_out = None` (the `o_norm` call path), the
`save_for_backward(residual_out, g, weight, bias, mean, rstd)` saves
the BF16 input tensor `x` (which is the `residual_out` slot's
substitution) plus `g`. **There is no FP32 cast save.** The
empirical probe at H=1536, T=2048 confirms:

```
LayerNormGatedFunctionBackward (2 calls, 12 MiB per call)
  saved:
    BF16 [T*HV, head_v_dim] = [24576, 128]  = 6 MiB  ← residual_out = x BF16 ref
    BF16 [T*HPP*head_v_dim] = [2048, 1536] = 6 MiB  ← g (the gate input)
    BF16 [head_v_dim] = [128]                          ← norm weight
    FP32 [T*HV] = [24576]            = 96 KiB         ← rstd
```

At production (T=16384, 8×): 1 × 48 MiB (o_x ref) + 1 × 48 MiB
(g ref) = **96 MiB per o_norm call**.

**Per-layer RMSNorm/Gated total**: `48 (attn) + 48 (mlp) + 96 (o_norm) = 192 MiB`.
**Live at peak**: only layer 31 (un-ckpt'd) contributes the inner
Function saves; ckpt'd layers' inner saves are discarded by the ckpt
wrapper. So peak RMSNorm/Gated ≈ **192 MiB**, NOT 1.78 GiB.

### §3.4 KDA (`src/models/tp_model/kda.py`)

Per-layer parameters (`kda.py:__init__`):
```
qkv_proj.weight : [2*key+value, H] = [(2*12*128 + 12*128), 1536]
                = [4608, 1536] bf16 = 13.5 MiB
o_proj.weight   : [H, value]      = [1536, 1536]     = 4.5 MiB
f_proj1.weight  : [gate, d_v]     = [1536, 128]      = 0.375 MiB
g_proj1.weight  : [value, d_v]    = [1536, 128]      = 0.375 MiB
g_proj1.bias    : [value]         = [1536]           = 3 KiB
fg_first.weight : [2*d_v, H]      = [256, 1536]      = 0.75 MiB
b_proj.weight   : [n_v_heads, H]  = [12, 1536]       = 36 KiB
A_log, dt_bias  : fp32, ~hpp each ≈ 48 B + ~gate/world B ≈ 0
conv1d (q/k/v)  : [C, 1, 4] bf16  ~ 0
o_norm.weight   : [head_v_dim]    = [128]            = 256 B
```
Per-layer parameter total ≈ 19.6 MiB. × 32 layers = **~627 MiB**.

**Forward pass saves** — measured by
`tools.vram_profile.saved_tensor_probe_base` at H=1536, T=2048,
n_layers=2 (84 MiB per call):

`ChunkKDAFunction.forward` (`chunk.py:96-106`, called from `chunk_kda`
at `chunk.py:63` which unpacks `chunk_kda_fwd` at `chunk_fwd.py:135`):

| Tensor | Shape (T=2048) | Size | At T=16384 (×8) |
|--------|----------------|-----:|----------------:|
| `q`           | BF16 [1, 2048, 12, 128] | 6 MiB | 48 MiB |
| `q_rstd`      | FP32 [1, 2048, 12] | 96 KiB | 0.4 MiB |
| `k`           | BF16 [1, 2048, 12, 128] | 6 MiB | 48 MiB |
| `k_rstd`      | FP32 [1, 2048, 12] | 96 KiB | 0.4 MiB |
| `v`           | BF16 [1, 2048, 12, 128] | 6 MiB | 48 MiB |
| **`g_cumsum`** | **FP32 [1, 2048, 12, 128]** | **12 MiB** | **96 MiB** |
| `g_input`     | BF16 [1, 2048, 12, 128] | 6 MiB | 48 MiB |
| `beta`        | BF16 [1, 2048, 12] | 48 KiB | 0.4 MiB |
| `A_log`       | BF16 [12] | 24 B | 48 B |
| `dt_bias`     | BF16 [1536] | 3 KB | 6 KiB |
| `Aqk`         | BF16 [1, 2048, 12, 64] | 3 MiB | 24 MiB |
| `Akk`         | BF16 [1, 2048, 12, 64] | 3 MiB | 24 MiB |
| `w`           | BF16 [1, 2048, 12, 128] | 6 MiB | 48 MiB |
| `u`           | BF16 [1, 2048, 12, 128] | 6 MiB | 48 MiB |
| `qg`          | BF16 [1, 2048, 12, 128] | 6 MiB | 48 MiB |
| `kg`          | BF16 [1, 2048, 12, 128] | 6 MiB | 48 MiB |
| `v_new`       | BF16 [1, 2048, 12, 128] | 6 MiB | 48 MiB |
| `h`           | BF16 [1, 32, 12, 128, 128] | 12 MiB | 96 MiB (scales w/ num_chunks) |
| `initial_state`, `cu_seqlens`, `chunk_indices` | None/tiny | ~0 | ~0 |

**Per-KDA-layer save total at production**: 9 × 48 (BF16 [T,H,K]) +
2 × 24 (BF16 [T,H,BT]) + 96 (h 5D) + **96 (FP32 g_cumsum)** + tiny ≈
**672 MiB**.

**Live count at peak**: only layer 31 (un-ckpt'd) has alive inner
Function saves. Ckpt'd layers' inner saves = 0 during ckpt'd fwd
(wrapped fn runs in `no_grad()`; inner Functions don't call
`save_for_backward`).

### §3.5 FFN / SwiGLU (`src/models/tp_model/swiglu.py`) — opt-5/2 era

> **2026-07-10:** This section describes the opt-5/2 BF16-only state.
> Current production runs NVFP4 mode-3 — see §10 for the live breakdown.

Per-layer parameters (BF16 only since opt-5/2 — NVFP4 buffers were
dropped on 2026-07-06, then re-introduced as mode-3 on 2026-07-10):
- `gate_up_proj.weight` = `[2*I, H]` = `[8192, 1536]` bf16 = 24 MiB
- `down_proj.weight` = `[H, I]` = `[1536, 4096]` bf16 = 12 MiB
- **Per-layer FFN parameter total**: 36 MiB. × 32 layers = **1.13 GiB**

There are NO `packed_weight` or `scales` buffers on GPU in opt-5/2 mode
(`swiglu.py:42-48` guard forces plain `ColumnParallelLinear` /
`RowParallelLinear`). NVFP4 mode-3 reintroduces them — see §10.

**Forward pass saves** (per F.linear's built-in autograd Function):

For `gate_up_proj`:
- input `x` (residual stream) = `[B*T, H]` bf16 = 48 MiB
- `weight` = `[8192, 1536]` bf16 = 24 MiB (parameter, always alive)
- output `gu` = `[B*T, 2*I]` bf16 = **256 MiB**

For `down_proj`:
- input `silu_out` = `[16384, I]` bf16 = 128 MiB (same tensor as `silu*up`)
- `weight` = `[1536, 4096]` bf16 = 12 MiB (parameter)
- output = `[B*T, H]` bf16 = 48 MiB

**Per-layer FFN saved total (un-checkpointed, layer 31)**:
- `gu` (gate_up output): 256 MiB
- `silu(gate)`: 128 MiB
- `silu*up` = down input: 128 MiB (same tensor as `silu_out`)
- **Total: ~512 MiB**

### §3.6 FLCE / FusedLinearCE (`src/models/tp_model/lm_head.py`)

`_TiedFusedLCEFunction.forward` saves:
- `dx` = `[N, H]` bf16 = 48 MiB (where N = B*(T-1) = 16384)
- `dw` = `[V, H]` bf16 = `248320 × 1536 × 2 = 727.5 MiB`
- **Total per chunk: ~775 MiB**, cleaned by per-chunk `empty_cache`.

### §3.7 Layer residual save (`src/models/tp_model/layer.py`)

`TPHippoLayer.forward` (`layer.py:91-97`):
```python
kda_out, final_state = self.kda(
    self.attn_norm(x), cu_seqlens=cu_seqlens, initial_state=initial_state,
)
x = x + kda_out                  # line 95
x = x + self.ffn(self.mlp_norm(x))  # line 96
return x, final_state
```

For ckpt-wrapped layers, the per-layer input is held via the
CheckpointFunction's save_for_backward, not via any per-layer
residual save (the wrapped fn runs in `no_grad()`). For layer 31
(un-ckpt'd), the per-layer input IS held. Peak residual.save =
**96 MiB** (layer 31's input + 1 other held for the next sub-block).

### §3.8 Sub-block / per-layer ckpt input (`src/models/tp_model/model.py`)

Non-last block sub-block forward (`model.py:466`, hardcoded
`_sub_block_layers = 2`):
```python
x, sub_final = torch.utils.checkpoint.checkpoint(
    self._block_forward, sub_layers, x, cu_seqlens,
    sub_states, True,                             # return_states=True
    use_reentrant=True, preserve_rng_state=False,
)
```

Last block per-layer ckpt (`model.py:504`):
```python
x, final_state = torch.utils.checkpoint.checkpoint(
    layer, x, cu_seqlens, init,
    use_reentrant=True, preserve_rng_state=False,
)
```

With `use_reentrant=True`, the inner `_CheckpointFunction` saves the
**wrapped function's positional args**. The wrapped function itself
runs in `torch.no_grad()`, so its inner custom Functions
(`LayerNormFunction`, `LayerNormGatedFunction`, `ChunkKDAFunction`,
F.linear, silu, etc.) DO NOT call `save_for_backward` during the
ckpt-wrapped fwd. Per ckpt input:
- `x` = `[B, T, H]` bf16 = 48 MiB (held by autograd)
- `cu_seqlens` (small, a few KB)
- `sub_states` = `_sub_block_layers × [1, hpp, K, V]` fp32 = 3 MiB
  (per sub-block); or 1.5 MiB per per-layer ckpt

For 14 sub-block ckpts (blocks 0-6, 2 sub-blocks each, 2 layers
per sub-block): 14 × 48 MiB = **672 MiB**. For 3 last-block
per-layer ckpts (layers 28-30): 3 × 48 = 144 MiB. For layer 31
un-ckpt'd: 0 from this path. Total ckpt input overhead = 816 MiB.

### §3.9 Block-boundary AttnRes saves

Per §3.2: 6 boundary stacks `[N, B, T, H]` bf16 → **1008 MiB** at
peak fwd_out (the 7th is consumed as input to block 8).

### §3.10 Model weights (the "(no-frames)" 2.452 GiB bucket) — opt-5/2 era

> **2026-07-10:** This section describes the opt-5/2 BF16-only state.
> See §10 for the NVFP4 mode-3 baseline.

Allocated before `torch.cuda.memory._record_memory_history` was
enabled, hence no frames. Walked via
`model.named_parameters_per_device(0)` + `named_buffers()` (no FFN
buffers since opt-5/2):

| Component | Per layer | Tensors | Total | Formula |
|-----------|----------:|--------:|------:|---------|
| `embed_tokens.weight` (tied) | 727.5 MiB | 1 | 727.5 MiB | V*H*2 |
| KDA + norms | ~19.6 MiB | 32 | ~627 MiB | qkv(13.5)+o(4.5)+fg_first(0.75)+f/g_proj1(0.375 ea)+b(0.036)+tiny |
| FFN BF16 master (no NVFP4 buffers) | 36 MiB | 32 | 1.13 GiB | gate_up(24)+down(12) |
| **Total** | | | **~2.49 GiB** | matches the 2.452 GiB `(no-frames)` bucket within 40 MiB slack |

### §3.11 Last layer (layer 31) total contribution

Layer 31 is the only **fully un-checkpointed** layer (block 7, final).

| Layer 31 tensor | Size | Bucket |
|-----------------|-----:|--------|
| KDA ChunkKDAFunction saves (q, k, v, g_input, g_cumsum FP32, Aqk, Akk, w, u, qg, kg, v_new, h) | 672 MiB | KDA.intermediates |
| KDA recompute_w_u (`wy_fast.py:269/267`) | 348 MiB | KDA.wy_recompute |
| KDA gate_cumsum (`gate.py:443`) | 120 MiB | KDA.gate_cumsum |
| KDA chunk_delta_h + l2norm + chunk_kda_fwd | 121 MiB | KDA.chunk_delta_h + others |
| attn_norm + mlp_norm saves | 96 MiB | RMSNorm.saved |
| o_norm saves | 96 MiB | RMSNorm.saved |
| FFN `gu` + silu + silu*up | 512 MiB | FFN.* |
| Layer input (residual.save) | 48 MiB | Layer.residual_save |
| **Layer 31 grand total** | **~1.42 GiB** | ~15% of peak |

## §4 Identifying the layer-31 contribution

Per-layer KDA saves are 672 MiB; layer 31's full un-checkpointed state
is ~1.42 GiB. To identify layer-31-specific segments in a snapshot:

1. Check `src/models/tp_model/model.py` for the sub-block / per-layer
   checkpoint logic. Confirm that block 7's last layer is NOT wrapped.
2. Find segments with innermost frames in `wy_fast.py:269/267`,
   `gate.py:443`, `chunk_delta_h.py:700`, `tp_layers.py:294`,
   `swiglu.py:67`. These are the saved-tensor frames for layer 31.
3. Sum: KDA ~672 MiB + FFN ~512 MiB + residual ~48 MiB + RMSNorm
   ~192 MiB = **~1.42 GiB** (15% of peak).

If you don't see ~1.42 GiB for these buckets combined, one of the
following is wrong:
- The layer-31 forwarding path is checkpoint-wrapped (check the
  source again).
- Some segments are getting misclassified (run the manual re-bucketing
  procedure in §6 to confirm).

## §5 Caveats and gotchas

### §5.1 Synchronous snapshots miss transient peaks

The `profile.py` script captures snapshots at fwd_out / bwd_out /
flush_out. The **transient** between fwd_end and bwd_end can be
1-2 GiB higher than the fwd_out snapshot, because the fwd saved-tensor
set is still alive while the bwd's newly-allocated grad buffers are
being created.

If an external observer (`nvitop`, `nvidia-smi`) reports a higher peak
than your snapshot, run `poll_prod` at 100+ Hz to find it.

### §5.2 gloo vs NCCL backend

`profile.py` defaults to `init_tp(backend="gloo")` for single-GPU dev.
The NCCL backend (production) adds **~510 MiB** of CUDA context
(comm buffers, NCCL streams, internal state) that gloo skips.

To accurately measure production peak driver, edit `profile.py` to
use `backend="nccl"`. This requires either NCCL + a multi-GPU box, or
the `--tp_sim` launcher (`src/training/launch/tp_sim.py`) which
simulates TP=N on a single device.

### §5.3 The `(no-frames)` bucket

PyTorch's `torch.cuda.memory._record_memory_history` only records
frames for allocations that happen **after** it's enabled. Allocations
before enablement (typically the model build + optimizer init) appear
as `(no frames)` segments.

To attribute those, walk `model.named_parameters_per_device(0)` after
construction but before the recorder is enabled. The breakdown
(post-opt-5/2):

| Identity | Size | Evidence |
|----------|-----:|----------|
| `embed_tokens.weight` | 727.5 MiB | V×H×2 = 248320×1536×2 |
| KDA params (qkv + o + f + fg_first + b + conv1d + o_norm + dt_bias) | ~627 MiB | 32 × sum of per-layer params |
| FFN BF16 master (gate_up + down) | 1.13 GiB | 32 × (24 + 12) MiB |
| FFN NVFP4 buffers | **0 GiB** | dropped 2026-07-06 (opt-5/2) |
| **Total** | **~2.49 GiB** | matches the 2.452 GiB `(no-frames)` bucket within 40 MiB slack |

### §5.4 Per-chunk peak is NOT cumulative

The KDA states are **detached between chunks** (truncated BPTT, see
`project_chunked_kda_oom.md`). So the per-chunk peak is bounded to
one chunk + the persistent model weights + the optimizer-step residual.

If a per-chunk measurement shows growing alloc across chunks (e.g.
chunk 0 < chunk 14), it's NOT saved-tensor growth — it's:
- cuBLAS workspace growth (lazily allocated on first large-matmul use).
- torch allocator metadata accumulation.
- NCCL per-step comm bookkeeping.

The growth is bounded (~90 MiB at production), and is fully absorbed
by the reserved pool slack. It does NOT grow the driver ask.

### §5.5 `empty_cache` doesn't free live tensors

`torch.cuda.empty_cache()` only returns **freed segments** to the driver.
Live allocations are unaffected. To free live tensors, you must:
- Drop references (e.g., `del out` after the autograd graph is severed).
- Call `torch.autograd.backward()` to consume the saved-tensors.
- Call `register_grad_offload_hooks` to D2H-copy the .grad tensors.

If your "peak" doesn't drop after `empty_cache`, you're looking at
live tensors that need one of the above.

### §5.6 The FLCE `dw` is **mislabeled** in the classifier

`fused_linear_cross_entropy.py:433` is `dw = torch.zeros_like(weight, ...)`
which is the embed gradient `[V, H]` bf16 = **728 MiB** at base config.
The classifier labels this bucket `fused_lce.dx`, but it's actually
`dw` (line 427's `dx` is only 48 MiB). Don't be misled by the name.

The `dw` lifecycle:
1. Allocated at line 433 during FLCE fwd (728 MiB on GPU).
2. Lives through bwd.
3. Consumed by FLCE's chunked bwd kernel (partial `add_` writes).
4. After bwd: flushed via `flush_manual_flush_params` to the CPU
   pinned accumulator (the manual_flush path for the tied embed).
5. `empty_cache` returns the pool to the driver.

The `dw` is NOT a peak driver (728 MiB = 7.5% of peak), and IS
cleaned by per-chunk `empty_cache`. The earlier FLCE chunking attempt
(`project_flce_dw_chunking.md`) confirmed `HWM unchanged` because
the saved-tensors budget dominates and the cache is recycled.

### §5.7 NVFP4 state (re-audited 2026-07-10)

**Current production state (NVFP4 mode-3, ON):**
- `configs/base.yml` has `ffn_nvfp4: true`, `ffn_nvfp4_marlin: true`,
  `ffn_nvfp4_no_bf16_master: true`.
- NVFP4 `packed_weight` + `scales` + Marlin `_scales_for_kernel` are
  the source of truth on GPU (324 MiB + 36 MiB ≈ 360 MiB total — see §10).
- **No BF16 master**: the BF16 view is materialized only inside the
  optimizer's per-chunk stream path (~8 MiB peak surface).
- Marlin FP4 fwd uses ctypes; bwd is **BF16 dequant + cuBLAS**
  (`_MarlinNvFp4Matmul.backward`, line 1022-1027) — the Marlin bwd
  kernel `_marlin_bwd_grad_x` is wired but **unused** in production
  (NaN/Inf at FFN scale, see `project_marlin_bwd_blocked.md`).
- Marlin `repacked` int32 buffer (~24 MiB / module) is **recomputed
  each forward** — not cached — see `project_marlin_cache_drop.md`
  for the 24 MiB/module × 64 = 1.5 GiB unreleasable-cache history.

**Re-enable path**: modes are mutually exclusive in the yml. The
mode-2 path (BF16 master + packed derived) is wired but not the
default. Mode-1 (BF16-only, opt-5/2) is reachable by setting
`ffn_nvfp4: false` in a child yml.

## §6 Audit findings (2026-07-06)

The earlier `/tmp/vram_report.md` (rev 4) had a misclassified bucket
called `RMSNorm.saved = 1.78 GiB`. The content of that bucket was
actually the KDA FP32 `g_cumsum` saves (96 MiB / layer from
`gate.py:443` via `chunk.py:102`), not RMSNorm residual_out casts.

### Root cause

`tools.vram_profile.classify`'s `is_residual_input` is too greedy: it
matches ANY frame in the stack containing `layer.py + forward`. KDA
intermediates (g_cumsum, Aqk/Akk, w/u/qg/kg/v_new) called from inside
the layer's forward get misattributed. The greedy match caught
`chunk.py:102 ChunkKDAFunction.forward` (which is inside
`TPHippoLayer.forward`) and labelled its saves as
`rms_norm.saved`.

A secondary misattribution claimed the RMSNorm saves were fp32 casts
("fp32 residual_out" sized at 96 MiB per call). **There is no fp32
cast** — `fused_norm_gate.py:466-469` shows `residual_out = None` when
`residual=None, residual_in_fp32=False` (this project's setup), and
the function returns `x` (BF16 input ref) per `:524`. The 1.78 GiB
"fp32 cast" of earlier drafts was the KDA FP32 `g_cumsum` (also
96 MiB per layer, also `[B, T, H, K]` FP32) — same shape, different
source.

### Empirical verification

`tools.vram_profile.vram_peak` was used to test the opt-5/1
hypothesis (drop the rmsnorm `x` save) across 3 scales at H=1536:

| hidden | T | L | peak (with x save) | peak (without x save) | delta |
|--------|---|---|---:|---:|---:|
| 1536 | 4096 | 8 | 1568.35 MiB | 1568.85 MiB | -0.50 MiB (noise) |
| 1536 | 4096 | 24 | 3562.38 MiB | 3562.38 MiB | +0.00 MiB |
| 1536 | 8192 | 8 | 2131.67 MiB | 2131.67 MiB | +0.00 MiB |

**Delta is exactly 0 across three independent scales.** The saved
`x` in fla's `LayerNormFunction` is a BF16 reference to the layer's
input tensor; that same tensor is held by the layer fwd chain (the
residual add `x = x + sub` consumes it on the next line of
`layer.py:95`). Removing the save doesn't free the storage because
there's a second reference.

See `project_opt5_rmsnorm_residual_skip.md` for the full evidence
chain (RMSNorm math verified by
`tools.vram_profile.rmsnorm_math_check` + empirical probe + opt-5/1
plumbing revert).

### Manual re-bucketing procedure (the fix)

```python
import pickle, collections
with open("/tmp/vram_prod.pickle", "rb") as f:
    snap = pickle.load(f)
# Pick the largest snapshot
peaks = snap["peak_snapshots"]
sizes = [(sum(b["size"] for seg in d.get("segments") or []
              for b in seg.get("blocks", [])
              if b.get("state") == "active_allocated"),
          label, d) for label, d in peaks]
sizes.sort(key=lambda x: -x[0])
peak_label, peak_dict = sizes[0][1], sizes[0][2]

# Walk segments, bucket by innermost-frame filename only
buckets = collections.defaultdict(int)
for seg in peak_dict.get("segments") or []:
    active = [b for b in seg.get("blocks", []) if b.get("state") == "active_allocated"]
    if not active: continue
    seg_sz = sum(b["size"] for b in active)
    frames = active[0].get("frames", [])
    inner = frames[0].get("filename", "?").split("/")[-1] if frames else "(no frames)"
    buckets[inner] += seg_sz

# Sorted
for k, v in sorted(buckets.items(), key=lambda x: -x[1]):
    print(f"  {v/1024**2:>8.1f} MiB  {k}")
```

Cross-reference the `saved_tensor_probe_base` output to confirm what
each innermost frame's function actually saves (don't trust the
classifier's label — the label is heuristic).

### Lesson

**The frame-based classifier is a heuristic.** For any optimization
claim (or any non-trivial analysis), validate via:
1. `saved_tensor_probe_base` to get the actual saved-tensor list per
   Function.
2. `vram_peak` A/B to verify the empirical delta across 2-3 scales.
3. Source-line check on the Function's `save_for_backward` to confirm
   the dtype/shape.

## §7 Quick diagnostic recipes

### §7.1 "What just OOMed?"

```bash
# 1. Run with smaller chunk size to fit, then examine the
#    peak snapshot's live tensors.
python -m tools.vram_profile.profile --steps 1 --chunk-size 8192 \
    --dump /tmp/vram_oom.pickle

# 2. Find the largest 20 active blocks.
python -c "
import pickle
with open('/tmp/vram_oom.pickle', 'rb') as f: snap = pickle.load(f)
biggest = sorted(
    (b for seg in snap['end'].get('segments', [])
     for b in seg.get('blocks', [])
     if b.get('state') == 'active_allocated'),
    key=lambda b: -b['size'])[:20]
for b in biggest:
    fr = b.get('frames', [{}])[0]
    print(f'{b[\"size\"]/1024**2:>7.1f} MiB  {fr.get(\"filename\",\"?\").split(\"/\")[-1]}:{fr.get(\"line\",\"?\")} {fr.get(\"name\",\"\")}')
"
```

### §7.2 "Is my optimization actually saving VRAM?"

Step 1: probe the actual saves that will be removed/changed.
```bash
python -m tools.vram_profile.saved_tensor_probe_base
```

Step 2: A/B with `vram_peak` (or `profile` for prod-scale).
```bash
# Run before the change
python -m tools.vram_profile.profile --steps 1 --dump /tmp/before.pickle
# Apply change
# Run after the change
python -m tools.vram_profile.profile --steps 1 --dump /tmp/after.pickle
# Diff
python -c "
import json
for path in ['/tmp/before.log.json', '/tmp/after.log.json']:
    with open(path) as f: t = json.load(f)
    print(f'{path}: peak_driver={t[\"max_driver\"]/1024**2:.1f} MiB  '
          f'peak_reserved={t[\"max_reserved\"]/1024**2:.1f} MiB  '
          f'peak_allocated={t[\"max_allocated\"]/1024**2:.1f} MiB')
"
```

**Acceptance criterion**: peak_allocated AND peak_driver must both
decrease by the claimed amount. If only one drops (typically just
`peak_allocated` because the allocator recycled freed segments), the
saving is a "soft" win — the cache pool grew slack but the driver
ask didn't change. This is the recurring pattern across the 2026-07
opt revert history (Opt-1/2/3 all reverted for the same reason).

### §7.3 "Why does nvitop see more than my snapshot?"

Three candidates:
1. **Transient peak** (see §5.1): run `poll_prod`.
2. **NCCL init overhead** (see §5.2): edit `profile.py` to use
   `backend="nccl"`.
3. **Different environment** (different CUDA, different library
   versions): compare `mem_get_info()`'s `total` to confirm it's the
   same device.

## §8 Cross-references

- `docs/gradient_checkpointing.md` — the sub-block / per-layer
  checkpoint structure used in the model (explains why some saves
  are released and others persist).
- `docs/kda_kernel_structure.md` — the KDA kernel call chain;
  useful for attributing saved tensors to specific kernel frames.
- `project_chunked_kda_oom.md` — why the KDA states are detached
  between chunks.
- `project_opt5_bf16_only_ffn.md` — opt-5/2 shipped 2026-07-06;
  FFN is BF16-only, NVFP4 buffers gone.
- `project_opt5_rmsnorm_residual_skip.md` — opt-5/1 REVERTED
  2026-07-06; the empirical probe that proved dropping the rmsnorm
  `x` save is 0 MiB peak delta.
- `project_nvfp4_no_bf16_master.md` — NVFP4 mode-3 design +
  optimizer streaming chunk contract.
- `project_marlin_cache_drop.md` — why `repacked` is recomputed
  every fwd instead of cached (24 MiB/mod × 64 = 1.5 GiB history).
- `project_marlin_bwd_blocked.md` — why the Marlin FP4 bwd
  kernel is unused in prod (BF16 cuBLAS path instead).
- `project_nvfp4_mode3_unwired.md` — register_nvfp4_module
  wiring gap that produced the 1152 MiB stash leak (now fixed;
  verified 2026-07-10).

## §9 Version history

- 2026-07-11 (rev 3): ckpt structure revised again to honor the
  user's **"2 rounds of recomputation per block"** design intent.
  Each non-last block is split into **2 sub-blocks of 2 layers each**,
  each sub-block wrapped in its own ckpt (so bwd has 2 rounds of
  re-fwd + release per block). The rev-2 block-level design (1 ckpt
  per block, 4 layers wrapped) was wrong because `use_reentrant=True`
  keeps ALL inner Functions' saves alive during the re-fwd → peak
  4 × 1184 = 4736 MiB. The rev-1 per-layer design (31 ckpts) was
  also wrong: peak 1 layer but fwd_out +672 MiB and +~600 ms/step
  bwd cost. §10.2 / §10.3.B / §10.4 / §10.5 / §3.8 updated. Peak:
  9984 MiB alloc / 10982 reserved / 11210 driver (same as the
  legacy `sub_block_size=2` baseline — this rev-3 IS sbs=2 with
  hardcoded `_sub_block_layers = 2`). Total: 17 ckpt() invocations
  per fwd (14 sub-block + 3 last-block per-layer). Peak delta from
  fwd_out during bwd = 1518 MiB ≈ 1.3 layers alive (polling average
  between 2-layer re-fwd peak and 1-layer inner-bwd phase).
- 2026-07-11 (rev 2): block-level ckpt for non-last blocks +
  per-layer ckpt for last block. WRONG design — `use_reentrant=True`
  keeps 4 layers' saves alive during block bwd (peak delta 4797
  MiB ≈ 4 layers). Reverted same day after the user clarified the
  intent: "2 rounds of recomputation".
- 2026-07-11 (rev 1, reverted): per-layer ckpt everywhere
  (`sub_block_size=1`, 31 ckpts). WRONG design — per-layer meant
  +672 MiB at fwd_out and +~600 ms/step bwd cost for only a 1184
  MiB peak reduction vs the 2-rounds design. Reverted same day.
- 2026-07-10: NVFP4 mode-3 baseline re-audit (§10). Header / §3.5 /
  §3.10 / §5.7 marked as opt-5/2 era with pointers to §10. Peak
  alloc 9984 MiB / reserved 10982 MiB / `mem_get_info` 11210 MiB at
  `test/_tmp/probe_vram_nvfp4.py` (sample-hz=500). Mode-3 stash
  verified clean (64 modules in opt state, 0 leaked `_latest_grad_w`).
  2026-07-10 follow-up: §10.3 static-table slack (~70 MiB) removed;
  exact per-module walk = 1714.1 MiB at TP=1, ≈ 857 MiB per rank at
  TP=2 (base.yml prod).
- 2026-07-06: Methodology + corrected breakdown (RMSNorm misclassification
  resolved). Tools formalized at `tools/vram_profile/` (profile, analyze,
  classify, saved_tensor_probe, saved_tensor_probe_base, vram_peak,
  poll_prod, rmsnorm_math_check).

## §10 Re-audit 2026-07-10: NVFP4 mode-3 baseline

Production now runs with NVFP4 mode-3 (the opt-5/2 BF16-only path
was reverted). This section supersedes §2 / §3.5 / §3.10 / §5.7 for
the **live production baseline**.

### §10.1 Probe configuration

`test/_tmp/probe_vram_nvfp4.py` at `configs/base.yml` + 5060 Ti 16G
+ `sample-hz=500` over 3 micro-batches (`chunk_size=16384`).

```
NVFP4 mode-3 audit (post-init):
  Total NVFP4 modules:                64
  Modules with BF16 master Param:      0   (mode-3: no BF16 master)
  Persistent FP4 (packed+scales+gs):  324.0 MiB   (= 603,979,776 elements × 0.5625 B/elem)
  Marlin caches (_scales_for_kernel):  36.0 MiB   (= 0.75 MiB/mod × 64, persistent=False registered buffers)
  Stashed grad_w (transient, mode-3):   0.0 MiB   (post-init, before any backward)

Mode-3 wiring audit (post-step):
  NVFP4 modules in any optimizer state:  64   (register_nvfp4_module called ✓)
  NVFP4 modules with _latest_grad_w:      0   (stash drained by accumulate_grads_to_cpu ✓)
```

### §10.2 Peak measurements

| Metric | Value | Sampled at | Δ from rev 2 (block-level) | Δ from rev 1 (sbs=1) | Δ from sbs=2 (legacy) |
|--------|------:|------------|---------------------------:|---------------------:|----------------------:|
| `torch.cuda.memory_allocated` peak | **9984.4 MiB** | during step 1 backward walk | −2943 MiB | −420 MiB | 0 MiB |
| `torch.cuda.memory_reserved` peak  | **10982.0 MiB** | same | −2384 MiB | −672 MiB | 0 MiB |
| `torch.cuda.mem_get_info used`     | **11210.0 MiB** | same | −2384 MiB | −672 MiB | 0 MiB |
| `nvitop` baseline (user-reported)  | ~11834 MiB      | 1 Hz sampling, catches transient peaks | −2266 MiB | −466 MiB | 0 MiB |

**Step boundaries** (`torch.cuda.memory_allocated`):

| Phase | alloc (MiB) | (was @ rev 2) | (was @ sbs=1) | (was @ sbs=2) |
|-------|------------:|--------------:|--------------:|--------------:|
| pre-fwd (post-flush) | 1783.5 | 1783.5 | 1783.5 | 1719.2 |
| post-fwd (layer 31 un-ckpt'd done) | **8466.4** | 8130.4 | 9170.4 | 8466.4 |
| post-bwd (post-flush_pending_grads + flush_manual + empty_cache) | 1783.5 | 1783.5 | 1783.5 | 1783.5 |
| post-step (optimizer.step + repack) | 1783.5 | 1783.5 | 1783.5 | 1783.5 |

The 9984 MiB sampled peak is reached **during the bwd walk** while
KDA backward rebuilds intermediates through a non-last block's
sub-block ckpt (2 layers wrapped). Peak delta from fwd_out during
bwd = **9984 - 8466 = 1518 MiB ≈ 1.3 layers** (1184 MiB / layer
at T=16384) — the polling average over re-fwd (2 layers alive
transiently) + inner bwd (1 layer alive). Same value as the
legacy `sub_block_size=2` baseline (this rev-3 design IS
sub_block_size=2, just hardcoded with a `_sub_block_layers = 2`
constant instead of a sweepable attribute). The
`flush_pending_grads + flush_manual_flush_params + empty_cache`
sequence at end-of-chunk collapses 8466 → 1783 in a single
synchronize.

### §10.3 Corrected per-component breakdown

**A. Static persistent GPU state (~1714 MiB at TP=1)**

Walked via `test/_tmp/probe_vram_nvfp4.py`'s per-module-class
aggregation (probes both `named_parameters` and `named_buffers`,
sums every Tensor that lives on cuda). **Probe runs at TP=1**
(`init_tp(world_size=1)`), so weights are stored full per device.
Production at TP=2 splits ColumnParallel out / RowParallel in halves
→ per-rank static ≈ 857 MiB; total across ranks = 1714 MiB.

| Component | MiB | Module count | Note |
|-----------|----:|-------------:|------|
| `TPShardedEmbed.weight` (tied) | 727.5 | 1 | V×H×2 = 248320×1536×2 |
| `ColumnParallelLinear` (KDA qkv/f_proj1/g_proj1/b_proj + AttnRes) | 457.2 | 128 | per-mod 13.5 MiB for qkv, ~0.4 MiB for f/g/b; at TP=1 stored full |
| `RowParallelLinear` (KDA o_proj + AttnRes o_proj) | 144.0 | 32 | [H, H] BF16 = 4.5 MiB/mod at TP=1 |
| `NVFP4ColumnParallelLinear` (gate_up packed+scales+cache) | 240.0 | 32 | packed 6 + scales 0.75 + Marlin cache 0.75 = 7.5 MiB/mod |
| `NVFP4RowParallelLinear` (down packed+scales+cache) | 120.0 | 32 | packed 3 + scales 0.38 + Marlin cache 0.38 = 3.75 MiB/mod |
| `Linear` (BlockAttnRes's non-Column projections, etc.) | 24.0 | 32 | small per-module |
| `ShortConvolution` (q/k/v short conv, 3 calls/layer) | 1.1 | 96 | conv_size=4 depthwise |
| `RMSNorm` × 66 + `TPKDA` × 32 (A_log/dt_bias FP32) + `FusedRMSNormGated` × 32 + `BlockAttnRes` × 1 | 0.3 | ~130 | tiny weights |
| **Static total** | **1714.1** | | probe-step "alloc before fwd" measured 1719.2 MiB (5 MiB CUDA init overhead, not a tensor) |

The earlier `~1783 MiB` row had a fictitious `misc unclassified ~70`
slack line; the per-module-class walk sums to **exactly 1714.1 MiB**.
No slack needed.

**NVFP4 detail** (the user's earlier correction verified):
- Total elements: 1536 × 4096 × 3 × 32 = 603,979,776
- Bytes/element: 4/8 (FP4 packed) + 1/16 (per-16 FP8 scale) = 0.5625
- Total NVFP4 weight bytes: 339,738,624 B = **324 MiB** (not the
  earlier-estimated 605 MiB which double-counted per-layer vs
  per-module)
- Marlin `_scales_for_kernel` (per-128 FP8, derived from
  `_process_global_scale` + Marlin permute, persistent=False
  registered buffer): 0.75 MiB/mod × 64 = **36 MiB**
- `_global_scale_adj` (4 B fp32 scalar): negligible

**B. Dynamic state at fwd_out (8466 MiB, +6747 over static)**

Per-layer saves at production T=16384 (verified by
`save_for_backward` hook at T=2048, 8× scale to T=16384):

| Function (per layer) | At T=2048 | At T=16384 |
|----------------------|----------:|-----------:|
| ChunkKDAFunction (q, k, v, g_input, g_cumsum FP32, Aqk, Akk, w, u, qg, kg, v_new, h) | 84 MiB | 672 MiB |
| MarlinNVFP4 bwd saves (gate_up `_NVFP4MarlinNoLeafMatmul` + down `_MarlinNvFp4RowMatmulNoLeafImpl`) | 22 MiB | 176 MiB |
| CausalConv1dFunction (q/k/v short conv, 3 calls/layer) | 18 MiB | 144 MiB |
| LayerNormGatedFunction (o_norm) | 12 MiB | 96 MiB |
| LayerNormFunction (attn_norm + mlp_norm, 2 calls/layer) | 12 MiB | 96 MiB |
| **Per-layer save total** | **148 MiB** | **1184 MiB** |

**Peak live layers at any moment:**
- Non-last blocks (sub-block ckpt, 2 layers per sub-block since
  2026-07-11 rev 3): **2 layers' saves alive during sub-block bwd
  (transient)**. Each sub-block is wrapped in its own
  `torch.utils.checkpoint.checkpoint(use_reentrant=True)` call; the
  re-fwd during bwd generates both layers' inner Functions'
  saved_tensors simultaneously, then the inner bwd walks layer by
  layer and releases each layer's saves as that layer's bwd
  completes. Polling catches a peak delta from fwd_out of
  **~1518 MiB** (between 1 and 2 layers' worth, the polling average
  of re-fwd peak + inner-bwd peak). At the moment of re-fwd
  completion, 2 × 1184 = 2368 MiB alive transiently; during inner
  bwd, 1 layer at a time.
  - 2026-07-11 rev 2 used block-level ckpt (1 ckpt per block, 4
    layers wrapped) — peak delta 4797 MiB ≈ 4 layers. The user
    identified this as wrong: their design intent was "2 rounds of
    recomputation", meaning 2 ckpts per block (2 sub-blocks), not 1
    ckpt per block.
  - 2026-07-11 rev 1 used per-layer ckpt (`sub_block_size=1`) —
    peak delta 1234 MiB ≈ 1 layer. The user also identified this
    as wrong: per-layer everywhere meant 31 ckpts and a
    +~600 ms/step bwd cost for only a 1184 MiB peak reduction vs
    the 2-rounds design.
- Last block (per-layer ckpt for layers 28-30, un-ckpt for layer 31):
  **1 layer's saves alive** (either layer 31 un-ckpt'd OR a
  ckpt'd layer's recomputation).

**Peak KDA-style dynamic state = 2 layers × 1184 = 2368 MiB** during
a non-last sub-block bwd (transient). Add 14 sub-block ckpt inputs
(each 48 MiB = 672 MiB) + 3 last-block ckpt inputs (144 MiB) + 8
block embeddings (384 MiB) + AttnRes intermediates (252 MiB) +
residual chain outputs (~768 MiB across layers) ≈ 4600-5100 MiB of
dynamic state at fwd peak.

**C. Gradients at chunk boundaries (~0 MiB)**

Streaming D2H via `accumulate_grads_to_cpu` +
`flush_pending_grads + flush_manual_flush_params +
torch.cuda.empty_cache` per chunk keeps `.grad` buffers empty at
fwd_out. The transient `_latest_grad_w` stash (24 MiB gate_up + 12
MiB down × 32 layers × 2 modules = **1152 MiB**) is drained at
end-of-chunk.

**D. Marlin transient (~24 MiB peak)**

Marlin `repacked` int32 buffer (`_repack_for_marlin`): **24 MiB per
gate_up module, recomputed every fwd**. Single module alive at a time
(sequential model execution) → peak **~24 MiB**. Bwd path uses BF16
dequant + cuBLAS, not the Marlin bwd kernel (`_marlin_bwd_grad_x` is
wired but unused — NaN/Inf at FFN scale).

### §10.4 What was wrong before this audit

1. **NVFP4 weight size**: I estimated 605 MiB packed (was the
   per-layer × per-module confusion); actual is **324 MiB**. The
   user's calculation (0.5625 B/elem × 603,979,776 elem) is the
   correct compact formula.
2. **KDA live count** (re-audited 2026-07-11, three times): doc
   said "last block un-checkpointed (4 layers)" — that referred to
   the BLOCK-level structure (last block has no block-level ckpt,
   runs layer-by-layer). Per-layer, only 1 layer is un-ckpt'd
   (layer 31) and 3 are individually ckpt'd (layers 28-30). The
   design intent is **2 rounds of recomputation per non-last
   block**: each block is split into 2 sub-blocks (2 layers each),
   each wrapped in its own ckpt, giving peak = 2 layers' saves
   alive during sub-block bwd. rev-1 (per-layer everywhere,
   `sub_block_size=1`, peak delta 1234 MiB) was wrong because
   per-layer meant 31 ckpts and +~600 ms/step bwd cost for only a
   1184 MiB peak reduction vs the 2-rounds design. rev-2
   (block-level ckpt, 1 ckpt per block, peak delta 4797 MiB ≈ 4
   layers) was also wrong because `use_reentrant=True` keeps all 4
   layers' saves alive during re-fwd, not the intended 1 layer.
   rev-3 (current) honors the design intent: 17 ckpt() calls per
   fwd, peak delta 1518 MiB ≈ 2 layers alive during sub-block
   bwd.
3. **Gradient streaming E bucket**: doc implicitly assumed `.grad`
   stays on GPU; in fact `accumulate_grads_to_cpu + empty_cache`
   per chunk keeps chunk-boundary E = 0.
4. **Marlin repacked bucket**: doc attributed ~768 MiB to repacked
   buffers; actual transient peak is **24 MiB** (single module, fwd
   only — bwd doesn't use the Marlin kernel).

### §10.5 Where the ~5400 MiB unaccounted residual lives

After subtracting attributed buckets from the 9984 MiB peak, the
residual ~5400 MiB is:
- **CausalConv1d + Marlin bwd saves** for the 2 layers alive during
  sub-block bwd: 2 × (144 + 176) = 640 MiB (not in the
  KDA.intermediates bucket)
- **Residual chain** outputs across 14 sub-block ckpt inputs + 3
  last-block ckpt inputs + layer 31 un-ckpt'd = ~18 layer outputs
  alive at fwd_out (each 48 MiB) = ~864 MiB
- **Block embeddings list** (8 entries for AttnRes input): 384 MiB
- **Autograd graph nodes + edges** in the chunked BPTT link: a few
  GiB of metadata (not tensor data; allocator bookkeeping)
- **CUDA caching allocator overhead** (alloc↔reserved gap): ~1000 MiB
- **Driver / context**: ~300 MiB

Empirical verification path: `test/_tmp/probe_vram_nvfp4.py` at
`sample-hz=500` captures this directly; high-freq `poll_prod`
confirms the 9984 MiB is reproducible.