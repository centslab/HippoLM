> **Skill**: [`step-perf-remeasure`](../.claude/skills/step-perf-remeasure/SKILL.md) — re-measure step time after any perf change to the loop path.

# Why `empty_cache_between_mb` is off (2026-07-12)

`configs/base.yml` sets `empty_cache_between_mb: false`. This
document captures the reasoning and the HWM evidence so the next
person to re-enable it (out of habit or "just in case") has the
numbers at hand.

## Background

`torch.cuda.empty_cache()` returns *cached but unused* blocks
from PyTorch's caching allocator back to the CUDA driver. The
next `cudaMalloc`-equivalent allocation has to re-pull memory from
the driver (slower than reusing a cached block).

In the chunked-training loop (`src/training/loop/run.py`), the
call fires once per micro-batch after `accumulate_grads_to_cpu`
and `flush_manual_flush_params`. With `n_chunks = 16` per step
(base.yml defaults: `seq_len=262144 / micro_batch_size=16384`), that's
**16 cudaFree/cudaMalloc cycles per step**.

## Why the instinct to leave it on is wrong

The flag was originally added to keep the VRAM `nvidia-smi` readout
from showing a high "allocated" number — i.e. for *display* purposes,
not for correctness. The caching allocator would happily reuse the
blocks for the next chunk's forward anyway, but `nvidia-smi`
keeps reporting them as "in use" until they're returned.

That display concern is misleading. Two facts:

1. **VRAM HWM is unaffected.** `torch.cuda.max_memory_allocated()`
   (the number that matters against the 5060 Ti 16 GB ceiling per
   `.claude/rules/saved-tensors-not-hwm.md`) tracks the peak of
   CUDA-side allocations. `empty_cache()` returns blocks to the
   driver but does *not* retroactively lower the recorded peak.
   The HWM is the same with the flag on or off.

2. **The allocator already does the right thing.** Between chunks,
   the freed `p.grad` blocks (cleared by the post-accumulate-grad
   hook in `src/training/param_offload/offload.py`) are added to
   the cache. The next chunk's forward allocates from this cache
   without going back to the driver.

So `empty_cache()` is paying a real per-call cost (driver round-trip)
for no measurable benefit.

## Measured impact (2026-07-12, 5060 Ti 16G)

Benchmark: `test/_tmp/bench_step.py` with seq_len=4096,
micro_batch_size=1024, hidden_size=512, num_layers=8, num_blocks=4,
4 chunks/step, 10 measured steps.

| Config | mean_ms | median_ms | min_ms | max_ms | HWM (MiB) |
|---|---|---|---|---|---|
| `empty_cache=true` (pre-change) | 1958.6 | 1945.6 | 1880.0 | 2096.9 | 223 |
| `empty_cache=false` (post-change) | **1870.5** | 1875.9 | 1795.8 | 1991.8 | **223** |

−4.5% mean, no HWM change. The HWM equality is the load-bearing
result — see the next section.

## HWM measurement details

The `torch.cuda.max_memory_allocated()` value is captured via:

```python
torch.cuda.reset_peak_memory_stats()
# ... warm-up step + 10 measured steps ...
hwm_mb = torch.cuda.max_memory_allocated() / 1024**2
```

`reset_peak_memory_stats()` runs *after* model construction and
*before* the warm-up step, so the HWM captures the steady-state
training peak (warm-up + measured steps). Both configs report
223 MiB — the empty_cache path neither raises nor lowers it.

Why this matters: in 2026-07 the team evaluated five NVFP4
optimization attempts (see `MEMORY.md` Opt-1..5 entries) and
discovered that `saved_tensors` and PyTorch caching-allocator
"allocated" numbers are *not* the HWM (rule:
`.claude/rules/saved-tensors-not-hwm.md`). The proper HWM probe is
`torch.cuda.max_memory_allocated()`, and it doesn't move with
`empty_cache()`.

## Production extrapolation

At production scale (4090, base.yml, 16 chunks/step) the per-chunk
`cudaFree/cudaMalloc` overhead is the same constant cost per call,
but applied 16× per step. Expected gain in the 5-10% range; HWM
expected unchanged. Validate on the production box before claiming
the gain lands there.

## Re-enabling the flag (when would it help?)

Almost never. Two scenarios where it might matter:

1. **Inter-process CUDA isolation.** If a side process wants the
   GPU to look "empty" (e.g. another model serving on the same box),
   `empty_cache_between_mb=true` would let `nvidia-smi` show a
   lower number between chunks. But that's a scheduling decision,
   not a training-loop decision.

2. **Severe memory pressure across very long chunks.** If
   `n_chunks=1` and the chunk is so large that the cache can't hold
   it, `empty_cache` may not help anyway (the alloc still has to
   pull fresh memory). In practice this never fires for HippoLM
   (the chunked path always has `n_chunks > 1`).

## What changed

`configs/base.yml`:

```yaml
# before
empty_cache_between_mb: true

# after
empty_cache_between_mb: false
```

No code change. No `src/training/loop/run.py` modification. The
`if getattr(args, "empty_cache_between_mb", True)` guard stays so
the flag still works for anyone who sets it from a CLI override or
a per-experiment yml.