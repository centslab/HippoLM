---
name: kda-correctness-sweep
description: Run a correctness sweep before reporting a KDA / Triton / CUDA / Marlin kernel config sweep as "faster". Use when the user asks to "sweep a kernel", "tune BK/BV/CHUNK", "compare kernels", "autotune", "find the fastest config", or any time a kernel-speedup claim is being made. Gate: isfinite + max-diff vs reference + roofline (FLOPs / AI / ridge) — three checks before timing is meaningful.
---

# KDA Correctness Sweep

A speed sweep is not done until three checks pass. The 2026-06-29
CHUNK=32 sweep found a 1.55 ms "winner" that produced 72300 NaN
+ 76308 Inf per batch in `O`, and a full `NaN` matrix in `Mqk`
(see auto-memory `feedback_verify_correctness.md`). It was
meaningless speed.

## Phase 1 — Correctness gate

Run the candidate config, then **both** of these before any
timing measurement is meaningful:

```python
out = kernel_fn(...)
ref = reference_fn(...)             # FP64 reference, or pre-existing prod config
torch.testing.assert_close(out, ref, atol=..., rtol=...)  # operator-appropriate tolerance
assert torch.isfinite(out).all(), "kernel produced non-finite output"
```

The tolerance defaults are too loose for KDA — see
[`docs/triton_kernel_playbook.md`](../../docs/triton_kernel_playbook.md)
for the per-op ranges. KDA `O` typically takes `atol=5e-3` against
FP64 reference at `seq_len=2048`.

If isfinite fails: do not proceed to timing. The "speed win" is
a real number but a meaningless one — bail out and report.

## Phase 2 — Reference for the math

The CHUNK=32 failure root cause: `q_decayed = q * exp(g_cumsum)`
overflows bf16 when `g_cumsum > ~88`. With CHUNK=32 the cumsum is
twice the CHUNK=16 span, pushing `g_total` into `-90` to `-100`
→ `exp()` → NaN.

The FlashKDA algorithm is fundamentally tuned for CHUNK=16 (FLA
also uses CHUNK=16). CHUNK is **algorithm-level**, not tiling.
Larger CHUNK requires algorithmic rescaling, not just a
`constexpr` change. See
[`docs/kda_kernel_structure.md`](../../docs/kda_kernel_structure.md)
for the chunk recurrence.

This isn't the only algorithm-level knob. Safe-gate mode and
`lower_bound` are also algorithm-level — see the gate math in
`docs/kda_kernel_structure.md` and the `safe_gate` config in
`configs/base.yml`.

## Phase 3 — Roofline before claiming compute-bound

Count properly before claiming "compute-bound" or "memory-bound".
On 2026-06-29 the KDA bwd intra was miscalled "compute-bound"
because someone used only HBM% as the proxy (see auto-memory
`feedback_roofline_discipline.md`).

The check:

1. `flops = sum(2 * M * K * N for tl.dot in kernel)`
2. `bytes = sum(read + write bytes)` (bf16=2 B, fp32=4 B)
3. `ai = flops / bytes`
4. `ridge = peak_tflops_bf16 / peak_bw_GBs` (4090: ~165 / ~1000 = 0.165; 5060 Ti: ~75 / 560 = 0.134)
5. If `ai >> ridge` → compute-bound, optimize for `tl.dot`.
6. If `ai << ridge` → bandwidth-bound, optimize for SMEM coalescing / L2 reuse.
7. If `ai ≈ ridge` and kernel is still slow → latency / occupancy-bound.
8. Report the actual % of **both** peaks, not just HBM%.

Common KDA bwd pitfalls: forgetting the `2` in `tl.dot`; counting
only one `bmm` when there are multiple per program; counting
fp32 workspace as "write traffic" when it's reused; missing the
diagonal vs off-diagonal `bmm` split; missing the per-K-tile tail
`bmm` after the V-loop.

## Phase 4 — Report shape

A sweep report must include:

- timing (ms or µs as appropriate);
- isfinite + max-diff vs reference (**always**, even on "obvious"
  parameter changes);
- # programs / SM occupancy (for tl.kernel dispatch);
- h_prev register footprint (recurrence kernels: `K * BLOCK_V *
  4 B` is the per-program limit; K=128 → BLOCK_V ≤ 32 to fit);
- AI + ridge, with the kernel's actual % of both peaks;
- dense + varlen workloads if applicable (varlen catches
  edge-padding bugs that dense misses).

Do **not** report timing without the rest. The 2026-06-29 CHUNK=32
sweep is the worked example of why.
