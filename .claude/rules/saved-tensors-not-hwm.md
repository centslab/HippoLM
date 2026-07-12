---
paths:
  - "src/models/ops/**"
  - "src/training/**"
  - "configs/**"
---

# `saved_tensors` is NOT the HWM

When a perf / VRAM change claims a memory win, the "what
`saved_tensors` shows" number is **not** the peak resident memory.
Use `torch.cuda.max_memory_allocated()` for the actual HWM —
that's the only number that matters against the 5060 Ti 16 GB
ceiling enforced by
[`.claude/skills/run-smoke-test/SKILL.md`](../skills/run-smoke-test/SKILL.md).

## The trap

`saved_tensors` from a `saved_tensors_hooks` pack function
reflects the **aggregate size of tensors held by autograd for
backward**, which doesn't include:

- Working buffers (matmul outputs reused later in the same pass).
- `torch.empty()` scratch the kernel allocates but doesn't save.
- Optimizer state (Muon BF16 momentum buffers, AdamW CPU-offload
  shadow copies — Muon int8-quant buffers removed 2026-07-12).
- Cached reloads (e.g. Marlin FP4 `_cached_repack` output — see
  auto-memory `project_marlin_cache_drop.md`; cache was *deleted*
  because it wasn't on HWM anyway, just wasted time).

The five NVFP4 / FFN optimization attempts in 2026-07-05 to
2026-07-10 (Opt-1 pre-RMSNorm fusion, Opt-2 KDA q/k/v recompute,
Opt-3 FLCE dw chunking, Opt-4 NVFP4 SMEM dequant, Opt-5/2 BF16-only
FFN) **all** reported "saved_tensors dropped N MiB" but **none**
moved the actual HWM. See auto-memory `project_nvfp4_mode3.md` and
the `Opt-N REVERTED` entries under `MEMORY.md` for the timeline.

## How to actually measure HWM

```python
torch.cuda.reset_peak_memory_stats()
# ... the change being evaluated (forward + backward + opt step) ...
peak_gb = torch.cuda.max_memory_allocated() / 1024**3
```

Run inside the smoke-test command from
[`.claude/skills/run-smoke-test/SKILL.md`](../skills/run-smoke-test/SKILL.md)
so the measurement matches production (TP sim exercises the
sharded path; dummy data is fine for HWM).

## When this rule fires

If a perf claim cites "saved_tensors dropped N MiB" or
"max-saved-tensors" without a `max_memory_allocated()` number,
ask for the HWM number. If it's missing, assume the win is
illusion. If it exists and the HWM actually went down, the
optimization is real — promote the assertion to the docs and a
memory entry.
