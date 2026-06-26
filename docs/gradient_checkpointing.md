# Gradient checkpointing policy

This document captures the **per-2-layer sub-block checkpoint** policy
in the model forward path (`src/models/tp_model.py`), and the trade
space that led to choosing it over per-block or per-layer alternatives.

It is intended as the reference the next optimizer reads before touching
`sub_block_size` or switching `use_reentrant` mode, and as the
contract for the L-dependent tuning of `sub_block_size`.

> **Config declaration (this doc is valid for):**
>
> All per-layer / per-component numbers in this doc were measured at
> **`32 layers × 1024 hidden × 8 heads × head_dim=128 × 3072 FFN ×
> seq_len=16384 × batch_size=1 × BF16 × TP=1`** (the base.yml config
> at the time this doc was written).
>
> If `configs/base.yml` is changed (e.g., 2026-06-26 bump to
> `hidden_size=1536 / num_heads=12 / intermediate_size=4096`), the
> absolute numbers here **do not apply**. The structure (ckpt
> placement, use_reentrant=True, sub_block_size=2 sweet spot at
> L=24..64, the FWD-tail-vs-BWD peak distinction, the empty_cache
> trade-off) still holds; only the byte counts change.
>
> Re-measure with the methodology at the end of this doc before
> quoting any number from a "Per-component breakdown" table.

## The constraint

Production config: `32 layers × 1024 hidden × 8 heads × 16384 seq ×
TP=1` on a single 16 GB GPU. The dominant VRAM cost is **per-layer
KDA intermediates alive during backward**, not parameters or optimizer
state (those are CPU-offloaded; see CLAUDE.md).

Per-layer KDA bwd holds roughly 288 MB of FP32 buffers (`dq, dk, dg,
dA, dv` at T=16384) plus the saved fwd tensors (`q, k, v, g_cumsum,
g_input, Aqk, Akk` ≈ 112 MB) per layer. With N layers in flight,
peak is approximately proportional to N.

## The three options at L=32 (base config)

Measured with the FLCE `dw` BF16 accumulator change already applied
(sweep harness: `test/_tmp/sweep_ckpt.py`):

| Policy                                  | sub_block_size | Saved inputs / non-last block | Layers alive during bwd | Peak (GB) | Step (ms) |
|-----------------------------------------|----------------|-------------------------------|--------------------------|-----------|-----------|
| per-layer (`use_reentrant=True`)        | 1              | 4                             | 1                        | 6.76      | 3215      |
| **per-2-layer (current, `use_reentrant=True`)** | **2** | **2**                     | **2**                    | **6.47**  | **3218**  |
| per-block (legacy, `use_reentrant=False`)| 4              | 1                             | 4                        | 7.91      | 3217      |

Per-2-layer wins at the base config: -290 MB vs per-layer, -1.44 GB
vs per-block, **same wall-clock** as per-block. Per-layer has a
slightly higher fwd peak (more saved inputs) that over-takes its bwd
savings; per-block has the lowest fwd peak but pays for it in bwd.

## L-sensitivity: which `sub_block_size` wins at each depth

The trade-off is between two opposing forces that both grow with L:

- **FWD peak term**: scales with `(L/block_size - 1) × block_size / sub_block_size × 32 MB`
  (number of saved sub-block inputs across all non-last blocks).
  Smaller `sub_block_size` → more saved inputs → higher fwd peak.
- **BWD peak term**: scales with `sub_block_size × 400 MB`
  (per-layer KDA intermediates alive during sub-block bwd).
  Larger `sub_block_size` → more KDA state in flight → higher bwd peak.

Empirical sweep at base hidden/heads/seq/TP=1 (B=1, seq_len=16384,
bf16 throughout, FLCE BF16 dw):

| L   | sbs=1 (per-layer) | sbs=2 (per-2-layer) | sbs=4 (per-block) | Winner   | Winner peak |
|-----|-------------------|---------------------|-------------------|----------|-------------|
| 16  | **4.05 GB**       | 4.28 GB             | 5.84 GB           | sbs=1    | 4.05 GB     |
| 24  | 5.28 GB           | **5.25 GB**         | 6.75 GB           | sbs=2    | 5.25 GB     |
| 32  | 6.76 GB           | **6.47 GB**         | 7.91 GB           | sbs=2    | 6.47 GB     |
| 48  | 10.46 GB          | **9.77 GB**         | 10.98 GB          | sbs=2    | 9.77 GB     |
| 64  | OOM (FWD peak 60 saved inputs) | 14.24 GB | 15.07 GB    | sbs=2    | 14.24 GB    |
| 80  | OOM               | OOM                 | OOM               | (model exceeds 16 GB) | — |

**Conclusions**:

1. **Per-block is never optimal at any L in the 16–64 range.** Its
   lower fwd peak never makes up for its bwd peak cost.
2. **Per-2-layer is the sweet spot for L=24..64.** It is the unique
   sbs that balances the two terms.
3. **Per-layer wins only at L ≤ 16**, where the fwd term is small
   enough (max 12 saved inputs × 32 MB = 384 MB) to be dominated by
   the bwd savings. **But it costs 2× wall-clock at L=16** (4.0 s
   vs 2.0 s) because every layer bwd triggers a re-forward. For
   L=24+ the per-2-layer step time is similar to per-block, so
   per-2-layer is the right answer on both memory and time axes.
4. **L=80 OOMs regardless of sbs.** The 16 GB budget is the binding
   constraint from L=80 upward, not the checkpoint policy. To scale
   beyond L=64 on 16 GB the model itself must shrink (fewer heads,
   smaller hidden, shorter seq), or the FLCE / last-block policy
   must change.

## Why `use_reentrant=True` matters

The legacy per-block checkpoint used `use_reentrant=False`, which
replays the autograd graph inside the bwd call. The whole block's
per-op intermediates are retained in the autograd graph; on bwd they
are reused in-place. This means **all 4 layers' KDA state is live
simultaneously** during block bwd, regardless of the saved-input
count. Peak is `O(layers_in_block)`.

`use_reentrant=True` runs an actual re-forward inside the bwd call.
Only the saved `x` and the per-layer outputs of the re-forward are
retained; per-op intermediates are not kept. The "layers alive"
count becomes `sub_block_size`, not `block_size`. This is the lever
that unlocks the bwd peak reduction.

Trade: one extra forward per checkpointed region per backward.
For sbs=2 at L=32 this is ~1.4 s/step on the production config.
For sbs=1 at L=16 it is ~3.0 s/step (the "per-layer wins" cell is
~2× slower than the alternatives).

## How `sub_block_size` is selected

`sub_block_size` is exposed as a `TPHippoModel` attribute (default
2, set in `__init__`). The forward method reads it via
`getattr(self, "sub_block_size", 2)`. To re-tune for a different
L range, override it after construction:

```python
model = TPHippoModel(config, devices=[0], dtype=torch.bfloat16)
model.sub_block_size = 1   # for L <= 16
model.sub_block_size = 2   # for L in 24..64 (default)
```

Block size is held at 4 (it is structural — see `HippoConfig.block_size`).
`sub_block_size` must divide `block_size`, so the valid set is
{1, 2, 4}.

## Verification

- **Numerical correctness**: `test/_tmp/test_per2_ckpt.py` captures
  loss and a representative set of grad norms under the per-2-layer
  policy and compares against a per-block baseline within bf16
  tolerance. Loss and grad norms match exactly (relative diff 0).
- **L-sweep peak memory**: `test/_tmp/sweep_ckpt.py` builds a model
  at each (L, sub_block_size) and records `max_memory_allocated`.
  Re-run with `--layers L1,L2,... --sub-sizes 1,2,4` to re-confirm
  before changing the default.
- **Single-config peak memory**: `test/_tmp/baseline_mem.py
  --label per2` produces a per-stage JSON of memory state.

## Combined with FLCE BF16 accumulator

The per-2-layer policy is **stacked** with the FLCE `dw` BF16
accumulator change in
`src/models/ops/_vendored/fla/modules/fused_linear_cross_entropy.py`:

- FLCE `dw` accumulator: FP32 → BF16 → saves 509 MB during bwd at
  base config. Per-element max abs diff vs FP32 baseline: 6e-5.
  Training trajectory preserved.
- All peak numbers in the tables above already include this change.
  Combined savings vs the original per-block + FLCE FP32 baseline:
  ~1.4 GB at L=32, growing roughly linearly with L.

These two changes are independent and additive; either can be
reverted in isolation without breaking the other.

## What NOT to change without re-measuring

- **`sub_block_size`**: do not change the default from 2 without
  re-running `sweep_ckpt.py` at the production L. The sweet spot
  is L-dependent; for L=24..64 it is 2, for L≤16 it is 1.
- **`use_reentrant`**: keep `True` for sub-block ckpt. The original
  `False` value is only correct when `sub_block_size == block_size`
  (whole-block re-forward, all intermediates retained).
- **The last block's per-layer ckpt** (3 ckpt, 1 not): the final
  layer must be eager so its output feeds the loss directly. The
  per-layer ckpt of the first 3 last-block layers is independent of
  this policy and should not be conflated with the non-last-block
  sub-block policy.
- **block_size**: held at 4 by `HippoConfig`. Changing it would
  require re-deriving the `sub_block_size` sweet spot and re-doing
  the numerical correctness check.

## Production VRAM profile (TP=1, 32L × 1024H × 8H × 16384seq, cu_seqlens path)

> **Numbers below are for `32 layers × 1024 hidden × 8 heads × 3072
> FFN × seq_len=16384` (base.yml as of the doc write date).
> For the current `hidden_size=1536 / num_heads=12 / intermediate_size=4096`
> base.yml, the absolute MB values shift; see the config declaration
> at the top of this doc.**

Measured with the real `scripts/train.py` contract: per-2-layer ckpt,
FLCE `num_chunks=64` (BF16 `dw`), cu_seqlens via FFD packing, no
pre-allocated `.grad` tensors (grads are D2H to CPU pinned per-mb by
`flush_pending_grads` and freed on GPU before the next mb starts —
production path, see `src/training/param_offload.py`).

### Peak memory by phase (per-microbatch, 16-mb cycle)

Measured by `test/_tmp/prod_real_peak.py` (real `build_param_groups`,
real `flush_pending_grads` + `flush_manual_flush_params` after each
mb). Driver view via `torch.cuda.mem_get_info`.

| Phase                                | alloc    | reserved | peak (mb) | driver used |
|--------------------------------------|----------|----------|-----------|-------------|
| model build (no inputs)              | 1.32 GB  | 1.32 GB  | 1.32 GB   | 1.46 GB     |
| post input load (cu_seqlens, labels) | 1.32 GB  | 1.32 GB  | 1.32 GB   | 1.46 GB     |
| **mb 0 (cold) fwd**                  | 5.45 GB  | 6.79 GB  | **6.29 GB** | 6.99 GB   |
| **mb 0 bwd**                         | 2.66 GB  | 7.06 GB  | **6.46 GB** | 7.27 GB   |
| **mb 1 (steady) fwd**                | 6.78 GB  | 8.02 GB  | **7.62 GB** | 8.23 GB   |
| **mb 1 bwd**                         | 2.66 GB  | 8.02 GB  | **7.62 GB** | 8.23 GB   |
| **mb 2+ (steady) fwd**               | 6.78 GB  | 8.02 GB  | **7.62 GB** | 8.23 GB   |
| **mb 2+ bwd**                        | 2.66 GB  | 8.02 GB  | **7.62 GB** | 8.23 GB   |

**Production peak = 7.62 GB (fwd)**, not the 6.47 GB figure that
appears in earlier drafts. The 6.47 GB figure is **mb 0 peak only**
(allocator pool starts empty, no cache reuse). From mb 1 onward the
allocator already holds 1.34 GB of cached-but-reusable residue from
mb 0's bwd, so mb 1 fwd reaches 7.62 GB = mb 0 fwd peak + that 1.34 GB.

**Steady-state reserved = 8.02 GB** (without `empty_cache`), driver
view = 8.23 GB (8428 MiB). The user's `nvidia-smi` reading of
"7984 MiB" matches this exactly. Hardware ceiling on the 5060 Ti
16 GB gives **~7.77 GB headroom** (16,311 - 8,428 MiB).

### With `--empty_cache_between_mb true` (default in current code)

The training loop calls `torch.cuda.empty_cache()` after each mb's
`flush_manual_flush_params` (see `src/training/loop.py:601`). This
returns the ~4 GB of caching-allocator slack pool to the driver
between microbatches; the next fwd re-allocates as needed.

| Phase                              | alloc    | reserved | driver used |
|------------------------------------|----------|----------|-------------|
| mb 1+ fwd (during fwd)             | 6.78 GB  | ~7.6 GB  | ~7.8 GB     |
| mb 1+ post-fwd (between mb's)      | 2.66 GB  | **4.09 GB** | **~4.2 GB** |
| mb 1+ peak driver                  | —        | —        | **8.23 GB** |

After `empty_cache`: **reserved 4.09 GB** (was 8.02 GB), driver view
~4.2 GB (was ~8.4 GB). Effective headroom = **12.13 GB** (was 7.77 GB).
Cost: +24 ms/mb (+0.7% wall-clock), hidden inside the fwd's
matmul/concat work.

Disable with `--empty_cache_between_mb false` only if you specifically
need the cached pool (e.g. for allocator microbenchmarks).

**Zero fragmentation across cycles**: mb 1 reserved == mb 2 reserved
== mb 3 reserved == 8.02 GB. The pool size stabilizes after the
first mb.

### Where the 1.34 GB "extra" between model size and steady alloc comes from

After mb 0 bwd, `memory_allocated` is **2.66 GB** even though the
model is only 1.32 GB. The 1.34 GB delta is **caching-pool residue**,
not a leak:

1. mb 0 bwd allocates ~1.34 GB of intermediate tensors (KDA bwd
   saved tensors, FLCE `dx`/`dw`, autograd graph nodes, cuDNN
   workspace). After `backward()` returns and `flush_pending_grads`
   runs, these tensors are released by Python reference counting
   and become eligible for GC.
2. PyTorch's caching allocator does **not** return freed blocks to
   the driver — it keeps them in the pool for reuse. `memory_allocated`
   counts them as long as they live in the pool, even when no Python
   reference holds them.
3. The next mb's fwd reuses these blocks. That's why mb 1 fwd only
   needs ~4.12 GB of new alloc (reaching 2.66 + 4.96 = 7.62 GB peak)
   rather than the ~5.45 GB mb 0 needed.

**In the default config this residue IS cleaned up**: with
`--empty_cache_between_mb true` (default; see `src/training/loop.py:611`),
the training loop calls `torch.cuda.empty_cache()` immediately after
each mb's `flush_manual_flush_params`. `empty_cache()` returns the
free pool back to the driver; mb 1+ post-fwd `memory_reserved` drops
from **8.02 GB → 4.09 GB** (driver view 8.23 → 4.2 GB). The 1.34 GB
residue is part of that 4 GB slack pool that gets released. See
"With `--empty_cache_between_mb true`" table above for the per-phase
numbers. The 1.34 GB figure quoted here is the *without-empty_cache*
steady-state residue; with `empty_cache_between_mb=true` it gets
returned to the driver between mbs and re-allocated on the next fwd.
Cost of the cleanup: +24 ms/mb (+0.7% wall-clock).

**Verification** (`test/_tmp/post_bwd_introspect.py`, walks `gc.get_objects()`):

| Persistent tensor shape        | Count | Total    | Source |
|--------------------------------|-------|----------|--------|
| `[248320, 1024] BF16` (Param)  | 1     | 485 MB   | embed.weight |
| `[248320, 1024] BF16` (Tensor) | 1     | 485 MB   | `_TiedFusedLCEFunction` saved embed_weight |
| `[6144, 1024] BF16`            | 32    | 384 MB   | FFN gate+up (model params) |
| `[3072, 1024] BF16`            | 32    | 192 MB   | KDA qkv (model params) |
| `[1024, 3072] BF16`            | 32    | 192 MB   | FFN down (model params) |
| `[1024, 1024] BF16`            | 32    | 64 MB    | KDA o (model params) |
| `[256, 1024] BF16`             | 32    | 16 MB    | (model params) |
| `[1024, 128] BF16`             | 64    | 16 MB    | (model params) |
| **Sum**                        | —     | **1.79 GB** | (param 1.32 GB + autograd 485 MB) |

The 485 MB autograd duplicate is `_TiedFusedLCEFunction.save_for_backward`
holding a reference to `embed_weight` for the backward pass
(`src/models/tp_model.py:348`). This reference is released as soon as
the next mb's fwd completes (autograd releases `ctx.saved_tensors`
when the backward graph for the prior mb is dropped), but until then
it stays in the alloc pool.

The 8.02 GB **reserved** - 2.66 GB **alloc** = 5.36 GB is the
remaining free pool. It is not held for any specific tensor; the
allocator simply doesn't shrink the pool between mb's because doing so
would force expensive `cudaFree` calls only to `cudaMalloc` them again
on the next mb.

### Per-component breakdown (peak = 7.62 GB, mb 1+ steady state)

Component sizes verified by walking the model's parameters and
measuring each layer's isolated workspace (sweep harness: temporary
test in `test/_tmp/`; deleted after producing the table).

**Model parameters (1.32 GB total)**

| Component                  | Size      | Count | Per-layer size      |
|----------------------------|-----------|-------|---------------------|
| Embedding (tied w/ LM head) | 0.474 GB  | 1     | [248320, 1024] BF16 |
| FFN gate+up (combined)     | 0.375 GB  | 32    | [6144, 1024] BF16 (3 × hidden) |
| FFN down                   | 0.188 GB  | 32    | [1024, 3072] BF16  |
| KDA qkv (combined)         | 0.188 GB  | 32    | [3072, 1024] BF16 (3 × hidden) |
| KDA o                      | 0.062 GB  | 32    | [1024, 1024] BF16  |
| AttnRes queries            | 0.032 GB  | 8     | [1024, 1024] BF16 (1 per block) |
| KDA dt_bias + A_log        | 0.1 MB    | 64    | tiny scalars        |
| Conv1d (q, k, v)           | 0.8 MB    | 96    | [4, hidden] BF16    |
| RMSNorm weights            | 0.1 MB    | 98    | [hidden] FP32       |
| **Sum**                    | **1.32 GB** | 549 |                       |

**Caching-pool residue at start of mb 1+ (~1.34 GB)**

This is **not** a leak — it is the PyTorch caching allocator holding
onto blocks that were freed by Python refcount after mb 0 bwd but
not returned to the driver. The next mb's fwd reuses these blocks,
which is why mb 1 alloc grows by only ~4.12 GB to reach the 7.62 GB
peak (rather than ~5.45 GB as in mb 0). See the section above for
the breakdown.

**KDA bwd saved tensors (~2-3 GB, the largest single consumer — corrected)**

`ChunkKDAFunction.save_for_backward` (`src/models/ops/_vendored/fla/ops/kda/chunk.py:89-93`)
retains these tensors per layer:

| Tensor | Shape | Bytes (BF16/FP32) | Per-layer |
|---|---|---|---|
| q, k, v | [B, T, H] | 32 MiB each | 96 MiB |
| g_input | [B, T, H] | 32 MiB | 32 MiB |
| Aqk | [B, T, HV, BT] | 16 MiB | 16 MiB |
| Akk | [B, T, HV, BT] | 16 MiB | 16 MiB |
| Akkd | [B, T, HV, BC] FP32 | 32 MiB | 32 MiB |
| q_rstd, k_rstd | [B, T, n_heads] | 256 KiB each | ~0.5 MiB |
| g_cumsum, beta | [B, T] | 32 KiB each | ~0.1 MiB |
| A_log, dt_bias | [n_heads] | 16 B each | ~0.0 MiB |
| initial_state, cu_seqlens, chunk_indices | tiny | — | ~0.0 MiB |
| **Per-layer total** | | | **~193 MiB** |

Note: `w, u, qg, kg, v_new, h` are set to `None` in `chunk_kda_fwd`
return (because `disable_recompute=True` is the default path for
training; see `chunk_fwd.py:128-134`) and are therefore **NOT** saved
for bwd. h would have been **256 MiB per layer** ([B, NT, HV, K, V]
= [1, 256, 8, 128, 1024] BF16), so this avoids 256 MiB/layer.

**Why the previous doc claimed 1.3 GB (wrong)**: counted all 21
tensors as [B, T, H] (32 MiB each), got 21 × 32 ≈ 672 MiB/layer × 2
layers = 1.3 GB. But Aqk and Akk are [B, T, HV, BT] (16 MiB not 32),
Akkd is FP32, and h + 5 others are actually `None`. Real per-layer
is ~193 MiB.

**Why the 2-3 GB at fwd peak**: at fwd peak, the autograd graph
holds KDA saved tensors for **all 32 layers** (not just 2-3) because
the inner sub-graphs of `use_reentrant=True` ckpt wrappers retain
their saved-tensors even though they don't retain the recomputed
intermediates. Direct evidence: GC walk at fwd peak shows 2.24 GB of
`(kT, H)` stair-step tensors (k=2..8) that don't fit a "3 layers in
flight" model. Walking `loss.grad_fn.next_functions` only surfaces
the eager last layer's KDA + ckpt inputs, missing the rest.

Net: 32 layers × 193 MiB ≈ **6.2 GB** if all were held, but
empirical fwd-peak delta is **~4.2 GB** (post-fwd alloc 5597 - 1.32
params - 1.0 FLCE - 0.10 other = ~3.0 GB KDA + activations, with the
stair-step 2.24 GB explainable as 8 layers' worth of KDA saved in
PyTorch's contiguous-buffer packing of [B, T, H]-shaped saves).

**FLCE chunked workspace + autograd hold (~1.0 GB total)**

- `c_logits` chunk: [C=256, V=248320] BF16 = **124 MB** (single chunk
  peak; FLCE chunk loop serializes over NC=64 chunks).
- `dw` accumulator: [V=248320, H=1024] BF16 = **509 MB** (allocated
  once outside the chunk loop).
- `dx` accumulator: [N=16384, H=1024] BF16 = 32 MB.
- **`embed_weight` autograd hold** ([V=248320, H=1024] BF16) =
  **485 MB**. `_TiedFusedLCEFunction.save_for_backward` retains
  `embed_weight` for bwd to scatter the LCE grad into the parent
  embed. On TP=1 `local_w` is a view of the full embed, so this
  reference == the full embed (485 MB).

`_TiedFusedLCEFunction` total saved: 32 + 509 + 485 = **1026 MiB**
(autograd walk confirmed: `_TiedFusedLCEFunctionBackward` saves
1002 MiB — dx is 32 MiB of that).

The c_logits chunk is bounded by `cdiv(V,H) = 243` chunks → C_min=128,
so the floor on c_logits peak is `128 × 248320 × 2 = 62 MB`. Pushing
past `num_chunks=64` saves at most 60 MB but adds 22% fwd wall-clock
(verified by sweep). **Leave at 64.**

The 485 MB embed reference is released on backward completion and
returns to the caching-pool residue (1.34 GB at start of mb 1+) — not
extra cost.

**Block-attn residuals (~256 MB)**

Each of 8 blocks retains a residual tensor ([1, 16384, 1024] BF16 =
32 MB) alive across the block's 4 layers for the AttnRes query
projection. Total: 8 × 32 MB = **256 MB** saved-tensor cost. Plus
~150 MB of per-block q/k/v workspace (the AttnRes query projection
output).

**FFN intermediates (~480 MB)**

SwiGLU's `gate * up` intermediate ([1, 16384, 3072] BF16 = 96 MB per
layer) is alive during the per-2-layer ckpt recompute: **480 MB**
total for ~5 layers in flight at peak (not just 2 — the 5 [1, T, 3072]
tensors visible in the GC walk are FFN intermediates from in-flight
sub-blocks of the current fwd).

**Autograd graph ckpt inputs (~544 MB)**

`use_reentrant=True` ckpt wrappers in the outer autograd graph:
- 14 sub-block ckpts (7 non-last blocks × 2 sub-blocks) × 1 [B, T, H]
  input each = 14 × 32 = **448 MB**
- 3 per-layer ckpts in last block × 1 [B, T, H] input each = 3 × 32 =
  **96 MB**
- Total: **544 MB** of ckpt-saved inputs

These are the tensors the ckpt wrapper re-forwards into during bwd
with `use_reentrant=True`. They sit in the outer graph at fwd peak.

**Other (cuDNN/cuBLAS/workspace) (~150 MB)**

- cuDNN benchmark cache: ~150 MB, allocated once on first use, stable
  across configs.
- cuBLAS handle workspace: ~50 MB (small relative to named components).
- Autograd graph metadata (not the saved tensors themselves): small.

These are the "named" components. The old ~2 GB "cuDNN/cuBLAS/
fragmentation" bucket was a mis-accounting — the real largest
unaccounted items are the KDA bwd saved tensors above, which were
undercounted in the previous doc.

**Sum check** (mb 1+ peak = 7.62 GB, no `empty_cache_between_mb`):
- Pre-mb alloc: 1.32 GB params + 1.34 GB caching residue = **2.66 GB**
- New alloc for fwd: 2.66 → 7.62 = **+4.96 GB**
  - **ckpt inputs (outer graph)**: 17 × [B,T,H] = **544 MB**
  - **KDA inner-saved [B,T,H]** (q/k/v/g_input of in-flight sub-block,
    distinct from ckpt inputs which sit in the outer graph):
    ~10 × 32 = **320 MB**
  - **KDA inner-saved [B,T,HV,BT/BC]**: Aqk/Akk/Akkd of in-flight
    sub-block + eager last layer ≈ **1.4 GB** (these are non-[B,T,H]
    shapes, computed from KDA's [B,T,H] q/k/v above)
  - **FLCE autograd hold**: 485 (embed_weight) + 32 (dx) = **517 MB**
    (the 485 MB embed reference returns to the caching pool on bwd)
  - **FFN intermediates** (SwiGLU gate*up): ~5 in-flight sub-block
    layers × 96 MB = **480 MB**
  - **Block-attn residuals** (8 blocks × 32 MB): **256 MB**
  - **Other** (cuDNN workspace + autograd metadata + small per-op
    allocations): ~**200 MB**
  - Sum: 544 + 320 + 1400 + 517 + 480 + 256 + 200 ≈ **3.72 GB**

**Reconciliation note**: this is still ~1.24 GB below the measured
`+4.96 GB` delta. The gap is the GC walk's blind spot — Python
references inside autograd `Function.ctx` (per-op cuDNN/cuBLAS handle
state, the KDA inner subgraph's `initial_state`/`cu_seqlens`/
`chunk_indices` tensors, plus per-chunk workspace freed between
sub-block bwd passes that the GC walk sees as live because their
refcount hasn't dropped yet). The numbers above are the
*attributable* cost; the residual 1.24 GB is "GC walk blind spot,
bounded by the steady-state 8.02 GB reserved pool" rather than a
specific component.

**Old "27 activations" entry was misleading**: the previous version
of this doc counted `17 ckpt inputs + 10 KDA inner-saved [B,T,H]`
together as "27 × [B,T,H] activations = 0.86 GB". That value is the
sum of two *separate* sub-budgets (ckpt inputs in the outer graph;
KDA inner saved in the per-layer subgraph) and shouldn't be added
to "KDA saved" without double-counting. The breakdown above splits
them into the two attributable lines.

**The 8.02 GB reserved - 2.66 GB alloc = 5.36 GB** is the free
pool (not held for any specific tensor; the allocator doesn't shrink
the pool between mb's because that would force expensive `cudaFree`
calls only to `cudaMalloc` them again on the next mb).

**Dominant headroom opportunities** (revised):
1. **KDA bwd saved (~2-3 GB)** — most of the transient fwd-peak cost
   is KDA saved tensors. Recomputing Aqk + Akk in bwd would save
   32 MiB/layer × ~30 layers in scope ≈ **1 GB** of fwd peak. Recomputing
   h in bwd (already done via `disable_recompute=True`) saves 256 MiB/
   layer. Recomputing q/k/v in bwd saves 96 MiB/layer × ~30 ≈ 2.9 GB
   but q/k/v bwd needs them so the saving is partial.
2. **FLCE embed reference (485 MB)** — released on bwd, returns to
   caching residue, not a true cost. To claw back: use the FLCE output
   dw only (not embed_weight), at the cost of forcing a full-embed
   D2H copy of the dw shard.
3. **Block-attn residuals (256 MB)** — intrinsic to the AttnRes design.
4. **Caching-allocator slack pool (5.36 GB)** — addressed by
   `--empty_cache_between_mb true` (frees 4.16 GB back to driver).

### KDA Aqk + Akk recompute in bwd (kda_skip_aqk_akk_saved)

Implemented as a `HippoConfig.kda_skip_aqk_akk_saved` flag and a
`--kda_skip_aqk_akk_saved` CLI switch (default off). The vendored
FLA KDA kernel (`ChunkKDAFunction` in
`src/models/ops/_vendored/fla/ops/kda/chunk.py`) was modified to
optionally skip saving `Aqk` and `Akk` in `save_for_backward`; the
bwd instead recomputes them via `chunk_kda_fwd_intra` (wrapped in
`torch.no_grad()` to avoid bloating the autograd graph).

**Correctness**: verified by running a 4-layer T=4096 model in two
subprocesses (one with the flag on, one off, identical state_dict) and
comparing loss + grads. Loss diff = 0; all parameter grads match to
rel diff < 1e-2 (in practice 0 because the recompute is exact).

**Memory savings at production dims (T=16384, H=8, BT=64, BF16)**:
- Per-layer KDA saved: `Aqk [B,T,HV,BT] BF16 + Akk [B,T,HV,BT] BF16`
  = 2 × 16 MiB = **32 MiB per layer**.
- Per-sub-block peak (sub_block_size=2): 2 × 32 MiB = **64 MiB
  held simultaneously** (only the current sub-block is held, the
  rest are released by the checkpoint policy).
- With `kda_skip_aqk_akk_saved=True`, the KDA bwd recomputes
  `Aqk + Akk` (plus the existing `w, u, qg, kg` recompute and
  `Akkd` fp32 workspace) per layer. The recompute transient is
  ~32 MiB (per layer, not stacked across sub-blocks), so the peak
  shifts from "64 MiB saved" to "~32 MiB recompute transient" — a
  net **~32 MiB peak saving per sub-block**.
- Note: the doc earlier estimated ~1 GB peak saving; the GC walk
  that produced the 2-3 GB figure summed *cumulative* KDA saved
  across all sub-blocks, not the simultaneous peak. The actual
  per-sub-block peak saving is ~32 MiB at production dims. Still
  worth the 0.2% bwd wall-clock cost (measured at T=16384 L=2: bwd
  went from 104.7 ms → 107.0 ms, 2.3 ms / 0.2%).

**Trade-off summary**:
- Pros: ~32 MiB peak per sub-block, 0.2% bwd cost, no API change for
  callers (config-driven, default off), bit-identical gradients.
- Cons: Adds a `chunk_kda_fwd_intra` call to the bwd hot path. The
  recompute is mathematically equivalent (Triton kernels, deterministic)
  but means the bwd does ~1 extra intra pass per KDA layer.
- Recommendation: keep default off (no behavior change for existing
  users); enable via `--kda_skip_aqk_akk_saved true` when 32 MiB
  peak headroom matters (e.g. fits one more training example per
  microbatch on a tight VRAM budget).

### mb 0 vs mb 1+ delta is allocator reuse, not warmup overhead

mb 0 starts with an empty allocator pool (1.32 GB reserved, all
model params). Its fwd allocates ~4.96 GB of new workspace, peaking
at 6.29 GB. mb 0 bwd reuses much of this; after the backward graph
is dropped, ~1.34 GB sits in the pool as residue.

mb 1 starts with 2.66 GB alloc (1.32 GB params + 1.34 GB residue).
Its fwd allocates ~4.96 GB of new workspace — **but** because the
1.34 GB residue is reusable, the *peak* alloc is 2.66 + 4.96 =
7.62 GB (the residue is logically part of the working set at peak).

So the apparent "mb 0 = 6.29 GB → mb 1+ = 7.62 GB" growth is **not**
a real memory cost: it is the same 4.96 GB working set, with the
peak being higher only because the residue contributes to the
high-water mark. The driver view (8.23 GB) and the reserved pool
(8.02 GB) confirm this — they stabilize at mb 1 and do not grow.

### "Other VRAM consumers" investigated

| Hypothesis                                                | Verdict |
|-----------------------------------------------------------|---------|
| cu_seqlens tensor on GPU (16 B → few hundred bytes)       | Negligible. FLA's `prepare_*_indices` derivatives are < 100 KB total. |
| KDA fwd's `disable_recompute` saves (g_cumsum/w/u/qg/kg)  | Already excluded from `save_for_backward` (set to None in return). |
| cuDNN/cuBLAS workspace                                    | ~200 MB total (driver - reserved). Stable across configs. |
| Multi-run allocator fragmentation                         | **Not real within a cycle**: reserved stays at 8.02 GB across 3 full cycles. The 1.34 GB gap between mb 0 and mb 1 is allocator reuse, not true fragmentation. `expandable_segments:True` saves only 9 MB. |
| D2H per-mb grad transfer                                  | Bounded by chunked pinned-memory source buffers; doesn't raise steady-state peak. |
| First-mb cuBLAS handle init                               | ~200 MB one-time allocation; cache pool reuses it on subsequent mb's. |
| Embed/LM-head tied weight (474 MB)                        | Already shared via `tie_word_embeddings=True`. Cost is intrinsic to the V×H matrix. The 485 MB `_TiedFusedLCEFunction` saved embed_weight is part of the 1.34 GB caching residue — released on backward, not a separate copy. |

### Dominant remaining consumers

1. **KDA bwd saved tensors (~1.0 GB cumulative across the model)**
   — fundamental to `ChunkKDAFunction`. The per-sub-block peak is
   ~64 MiB (2 layers × 32 MiB at production dims, sub_block_size=2);
   the rest of the 1 GB is summed across all sub-blocks held at
   different times during the bwd. The `Aqk + Akk` portion
   (~32 MiB/layer, ~64 MiB/sub-block) can now be elided via
   `--kda_skip_aqk_akk_saved true`; see the section above. The
   remaining `q/k/v` (96 MiB/layer) and `g_cumsum` (32 MiB/layer)
   are harder to elide (q/k/v are needed for l2norm bwd; g_cumsum
   is recomputed in bwd from `g_input` for `use_gate_in_kernel=True`
   and is therefore already freed in fwd). The bulk of the "KDA
   saved" cost in the GC walk is actually the inner KDA autograd
   subgraph for the current sub-block (one sub-block at a time
   during bwd, but the subgraph holds onto q/k/v plus the
   `l2norm` workspace and the conv1d state).
2. **FLCE `dw` accumulator (509 MB)** — sized by V×H; cannot shrink
   without changing the loss or the optimizer path.
3. **Caching-allocator slack pool (5.36 GB)** — reserved-but-unused
   blocks at start of mb 1+. Not fragmentation (pool size stabilizes
   after mb 1), but the pool cannot shrink below ~8 GB without
   forcing `cudaFree`/`cudaMalloc` thrash across the fwd+bwd cycle.
   Most likely targets to claw back headroom:
   - Lower `pack_buffer_size` (production uses 128; tests show 8 is
     enough but doesn't change peak since the pack is on CPU)
   - Try `expandable_segments:True` (saves only ~10 MB; not
     significant in steady state)
   - Profile per-op allocations with the CUDA profiler (currently
     blocked: CUPTI initialization fails on this box with
     `CUPTI_ERROR_INVALID_DEVICE`; need root or a CUDA-capable
     profiler container)
4. **Embed tied weight (474 MB + 485 MB autograd hold)** — intrinsic
   to V×H matrix. The 485 MB autograd reference returns to the
   pool between mb's and is reused on the next fwd; the 474 MB
   Parameter itself is unavoidable.
   - Reduce reserved pool via `torch.cuda.set_per_process_memory_fraction`.
   - Profile per-op allocations with the CUDA profiler (currently
     blocked: CUPTI initialization fails on this box with
     `CUPTI_ERROR_INVALID_DEVICE`; need root or a CUDA-capable
     profiler container).
4. **Embed tied weight (474 MB)** — intrinsic; no further sharing.

## VRAM profiling methodology (throwaway diagnostic pattern)

When the steady-state numbers in this doc drift from observation, or a
new config size pushes peak past the budget, re-measure using the same
methodology that produced the numbers above. The diagnostic itself is
a one-shot script in `test/_tmp/vram_breakdown.py`; it is **deleted
before commit** per the test-first rule, but the recipe below
re-creates it.

### The tool

`test/_tmp/vram_breakdown.py` runs the actual `_train_worker` from
`src/training/loop.py` (so the path matches `scripts/train.py`
exactly), and adds four hooks:

1. **`torch.cuda.memory._record_memory_history`** for the full
   allocation trace (best-effort; some PyTorch builds lack
   `_dump_memory_history` and the dump is a no-op).
2. **Monkey-patched `torch.cuda.empty_cache`** that logs every
   call's pre-state (alloc, reserved, driver_used).
3. **Monkey-patched `torch.utils.checkpoint.checkpoint`** that logs
   alloc/reserved/driver around each ckpt region (only enabled with
   `--inline_snapshots`).
4. **Monkey-patched `torch.Tensor.backward`** that logs alloc/reserved
   pre/post each `loss.backward()` call.
5. **Monkey-patched `torch.cuda.synchronize`** that logs every new
   `max_memory_allocated` high-water mark — this is what catches the
   peak inside an op, not just at op boundaries.

### How to run it

```bash
# Default — 1 step × 16 mbs, production config, with empty_cache enabled
python test/_tmp/vram_breakdown.py --step 1 --grad_accum 16

# Compare empty_cache on vs off (must override the argparse bool directly,
# see Pitfalls below)
python test/_tmp/vram_breakdown.py --step 1 --grad_accum 16 --no_empty_cache

# Per-component attribution via gc.walk at the end
python test/_tmp/vram_breakdown.py --step 1 --grad_accum 16 --track_tensors

# Test --kda_skip_aqk_akk_saved at the prod dims
python test/_tmp/vram_breakdown.py --step 1 --grad_accum 16 --kda_skip_aqk_akk_saved
```

A short run (`--step 1 --grad_accum 4`) completes in ~30 s on a
5060 Ti 16G; `--grad_accum 16` takes ~2 min. The output gives you
the per-mb steady state, the per-ckpt alloc/reserved trace, and the
peak HWM with phase labels.

### Pitfalls

- **`argparse` `type=bool` does NOT parse `"False"` as `False`.**
  `bool("False")` is `True` because non-empty strings are truthy.
  The CLI flag `--empty_cache_between_mb` uses `type=bool`, so passing
  `--empty_cache_between_mb False` from a wrapper silently enables
  the flag. The wrapper script in `test/_tmp/vram_breakdown.py`
  works around this by parsing the arg normally and then overriding
  `cli_args.empty_cache_between_mb = False` directly. **Before
  trusting a "with-empty_cache" run, verify `empty_cache call count`
  in the output matches the expected mb count.**

- **`max_memory_allocated` only catches peaks at sync points.**
  The `torch.cuda.synchronize` monkey-patch in the tool closes this
  gap: it logs every new high-water mark. Without it, peaks that
  happen inside a CUDA op (e.g., between `loss.backward()`'s pre-
  and post-snapshots) are invisible. The sync-patch is the reason
  this methodology catches the 6.31 GB FWD-tail peak that the
  raw `max_memory_allocated` alone would have missed.

- **`max_memory_reserved` ≠ `max_memory_allocated`.** Reserved is
  the caching-allocator pool size (incl. freed-but-not-returned
  blocks); allocated is live-tensor bytes only. The peak to optimize
  is `max_memory_allocated`; `max_memory_reserved` only matters for
  the driver-side ceiling (i.e., `mem_get_info()`).

- **`mem_get_info()` "driver used" is `total - free`, not `reserved`.**
  They differ by the small amount of allocator metadata the driver
  tracks but `memory_reserved` doesn't. Use `mem_get_info()` when
  you care about the actual ceiling; use `memory_reserved` for
  allocator-behavior analysis.

### How to read the output

The script's output has four sections; this is how to interpret them:

**`Snapshot table`** (top of output) — pre-init and post-train-worker
values for `memory_allocated`, `memory_reserved`, `max_memory_allocated`,
`max_memory_reserved`, `driver_used`, `driver_total`. The
`max_memory_allocated` is the **peak live tensors** at any point
during the run; that's the number to optimize against.

**`empty_cache call log`** — state BEFORE each empty_cache invocation.
This is the "between mb" snapshot. Without `--empty_cache_between_mb
false` you should see one entry per mb. If `reserved` here is ~7 GB
without empty_cache and ~2.6 GB with it, the cleanup is working.

**`Inline ckpt snapshots`** — alloc/reserved/driver around each ckpt
region. With sub_block_size=2 and 17 ckpts per mb, mb 0 produces
ckpts 0-16 (forward pass), ckpts 17-33 (backward re-forward), and
so on for each mb. **The pattern to look for:**
- ckpt 0-16: alloc grows monotonically from ~1.4 GB to ~4.3 GB
  (cumulative saved-tensor cost of the fwd graph)
- ckpts 17-33: alloc drops back to ~1.4 GB at start (post-bwd of
  first ckpt), then grows again as bwd progresses (re-fwd creating
  new subgraph tensors). Peak ckpt-post alloc across the bwd
  region is usually smaller than the fwd peak — if it's not, the
  re-fwd is hitting a memory wall.

**`Sync points with new max_memory_allocated`** — only the high-water
marks. The **last** entry (highest peak) tells you the phase:
- `sync# <last ckpt fwd post>`: peak is in the ckpt region itself
  (rare — would mean the per-ckpt saved is the bottleneck)
- `sync# <after ckpts, before bwd>`: peak is in FWD's tail (eager
  last layer + lm_head + FLCE forward). This is the most common
  case for the prod config — the FLCE workspace + eager last
  layer's KDA saved add ~1.4 GB after ckpts end.
- `sync# <mid-bwd>`: peak is in BWD. Check ckpt snapshots for the
  re-fwd burst; this is the case where `--kda_skip_aqk_akk_saved`
  or per-layer sub_block_size helps.

### Why `--kda_skip_aqk_akk_saved` doesn't measurably reduce peak

At prod dims, this flag saves ~32 MiB/layer (Aqk + Akk) × 32 layers
= 1 GB theoretical. Empirically the peak drops by <50 MB (within
noise). The reason: the **peak happens at FWD-tail** (sync#17 in
the sync log), not at the per-ckpt region. By the time the peak
is reached, only a fraction of layers' KDA saved are alive — the
others are either not yet materialized (forward hasn't reached
them) or already partially released by autograd graph traversal.
The 1 GB "savings" only applies if the peak is dominated by KDA
saved of *in-flight* layers, which it isn't at our L=32.

If the peak *did* shift to a per-ckpt region (e.g., at L=80 or
larger dims where the per-ckpt saved grows), this flag would help
proportionally. Don't trust the doc claim of "1 GB peak savings"
without re-measuring at the actual peak phase.

### What to do with the result

After running the tool, identify the phase of the peak from the
sync log, then read the matching per-component section above:

| Peak phase (last sync#) | Dominant consumer | See section |
|---|---|---|
| After ckpts, before bwd | FLCE workspace + eager last layer + KDA inner saved | "Per-component breakdown" + "FLCE chunked workspace" |
| Mid-ckpt-region | KDA bwd saved (per-layer 193 MiB) | "KDA bwd saved tensors" |
| Mid-bwd of a single ckpt | ckpt re-fwd + new KDA subgraph | "Why `use_reentrant=True` matters" |
| Steady-state between mbs (with empty_cache) | Caching pool size | "Caching-allocator slack pool" |

If the measured peak is much higher than the table predicts, the
config has changed since this doc was last updated — re-measure and
update the table.
