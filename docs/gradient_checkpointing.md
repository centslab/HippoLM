# Gradient checkpointing policy

This document captures the **per-2-layer sub-block checkpoint** policy
in the model forward path (`src/models/tp_model.py`), and the trade
space that led to choosing it over per-block or per-layer alternatives.

It is intended as the reference the next optimizer reads before touching
`sub_block_size` or switching `use_reentrant` mode, and as the
contract for the L-dependent tuning of `sub_block_size`.

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
