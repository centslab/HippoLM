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
