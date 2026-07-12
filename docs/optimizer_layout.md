> **Skill**: [`run-smoke-test`](../.claude/skills/run-smoke-test/SKILL.md) — 任何 optimizer layout 改动 commit 前必过。

# Optimizer layout: which params go to AdamW, which to Muon

The per-param optimizer split lives in
`build_param_groups` (`src/training/param_offload.py`). The
routing rule is:

| Param shape | Optimizer | Examples |
| --- | --- | --- |
| `ndim < 2` (1-D) | AdamW | norms, biases, A_log, dt_bias, o_norm weight |
| `ndim == 2` (matmul) | Muon | q/k/v/o projections, gate_up_proj, down_proj, b_proj, fg_first |
| `ndim == 3` (conv1d) | AdamW | q_conv1d, k_conv1d, v_conv1d weights (depthwise) |

## Why depthwise conv1d goes to AdamW, not Muon

The depthwise conv1d weights (`*.q_conv1d.weight`,
`*.k_conv1d.weight`, `*.v_conv1d.weight`, plus the `p.ndim == 3`
backstop in `build_param_groups`) are routed to **AdamW**, not
Muon. Two reasons:

1. `CPUMuon._newton_schulz` does `g.t()` which only works on
   2-D matrices — 3-D conv weights are not Muon-eligible
   anyway.
2. Per established KDA practice: conv weights are AdamW-managed
   in the KDA reference; don't try to push them into Muon via
   reshape, and don't unify the conv path with the rest of the
   model params in a single Muon group.

## Why 1-D params go to AdamW

1-D params (norm weights, biases, A_log, dt_bias) are the "scale
and shift" parameters of the model. Muon's Newton-Schulz
iteration orthogonalizes a 2-D weight matrix via
`G @ (aG + bG @ G.T @ G + ...)`, which is fundamentally a 2-D
operation — the VJP doesn't reduce cleanly to a 1-D vector.
AdamW handles them correctly via per-element momentum and
variance.

## Why 2-D matmul params go to Muon

Muon's Newton-Schulz gives the orthogonalized momentum a
"rotation-only" update, which empirically converges faster than
the per-element scaling AdamW does for matmul-shaped params
(better-conditioned optimizer trajectory, fewer epochs to a
given validation loss). This is the original Muon paper's
contribution and is what makes the memory-cheap int8-quantized
momentum still work in the precision-tight regime.

## Param routing that needs special handling

These are params that need explicit routing beyond the ndim
rule, and the `no_weight_decay` / `_no_weight_decay` flag they
carry:

- `A_log` (per-head log-decay): 1-D, goes to AdamW, but marked
  `no_weight_decay` (the initialisation log-uniform over
  `[log 1, log 16]` is already calibrated; further decay is
  driven by the gradient signal alone).
- `dt_bias` (per-head log-time-constant offset): 1-D, goes to
  AdamW, also `no_weight_decay` (same reasoning as A_log).
- `o_norm.weight` (per-head RMSNorm on the value dim): 1-D,
  goes to AdamW, with weight decay (it's a normal norm weight).
- The embed weight (`embed_tokens.weight`, `[vp, H]`, 2-D):
  goes to Muon in the production routing. Some configs use
  AdamW for the embed (when the embed is tied and the optimizer
  has a special "no_muon_on_embed" rule). Check the config.

## When to add a new param

If you add a new sub-layer, the rule of thumb:

1. If the param is a 2-D matrix that participates in a matmul
   (`x @ W` or `W @ x`), it goes to Muon.
2. If the param is a 1-D scale/shift (norm weight, bias, decay
   parameter), it goes to AdamW.
3. If the param is a 3-D conv weight, it goes to AdamW. Don't
   reshape to 2-D to push it into Muon — the gradient has a
   different structure (per-channel vs per-element) and the
   Newton-Schulz orthogonalization doesn't apply.
4. If the param doesn't fit any of the above, ask before
   routing it. Don't guess — the wrong optimizer group means
   the model trains but the loss landscape is different from
   the reference, and a "stable" run is actually a different
   model.

## Production wiring

`build_param_groups` (in `src/training/param_offload.py`)
collects params by walking `model.parameters()` and dispatching
on `p.ndim` + a small set of name-based overrides for the
`no_weight_decay` flags. The function returns two lists — one
for AdamW, one for Muon — and the training loop passes them to
the per-device optimizer factories in `src/training/loop.py`.

The test that pins this routing is
`test/test_param_routing.py` (was `_tmp` during the routing
debug, now promoted). It covers:

- 1-D params → AdamW
- 2-D params → Muon
- 3-D conv1d → AdamW
- A_log, dt_bias, o_norm → AdamW with no_weight_decay
- Embed weight (when 2-D) → Muon
- A mixed-model fixture that exercises the ndim-2 / ndim-1 /
  ndim-3 cases together

## Muon accumulator layout: bf16 merged vs separate `accum` for quantized storage

Muon's per-cycle grad accumulator layout depends on the
configured `precision.muon_momentum` storage dtype. The split
was introduced in 2026-07-02 to fix a per-microbatch CPU sync
bottleneck in the merged-accumulator design when the storage
dtype was quantized (int8 / mxfp8).

### The bottleneck (pre-fix)

The pre-fix design had `s.mom_buf` doubling as the per-mb
grad accumulator (`mu=1` accumulation — `mom_buf` resets to
zero at the end of every `step()`). For fp* storage this was
fine: each microbatch's bf16 grad was just added to
`s.mom_buf` (CPU bf16 add, ~30 ms at 8-layer smoke).

For quantized storage (int8 + BF16 per-row scale, mxfp8 +
E8M0 per-block scale) this merged design forced a full
dequant-add-requant cycle on the CPU pinned momentum buffer
for **every** microbatch: dequant `mom_buf`/`mom_scale` →
FP32, add the new bf16 grad, requantize back to int8/E4M3 +
scale, copy back. At 8-layer smoke this was ~7-8 seconds per
microbatch; at the production 32-layer scale ~32 seconds per
microbatch — the dominant cost in `flush_pending_grads`. With
`gradient_accumulation_steps=16` a single optimizer step
spent ~8 minutes on CPU sync alone, blocking every forward.

### The fix (2026-07-02)

A separate `s.accum` field was added to `_ParamState` —
BF16, pinned, `numel=n` — for Muon params with quantized
storage. The per-mb hot path becomes `s.accum.add_(grad_bf16)`
— a single CPU bf16 add at ~30 ms (independent of scale).
The expensive dequant-add-requant now runs **once per
optimizer step** (amortized over all microbatches in the
cycle) inside `CPUMuon.step`, which reads `s.accum` as the
cycle's grad sum, does Newton-Schulz, and requantizes the
result back into `s.mom_buf` / `s.mom_scale` for state_dict
observability and cycle continuity.

fp* muon keeps the merged-accumulator design: `s.mom_buf`
**is** the accumulator (no separate `s.accum`). Only the
quantized path pays the extra 2 bytes/elt for the separate
`accum` buffer.

### Per-storage-dtype layout

| `muon_momentum` dtype | accumulator | momentum storage | per-mb work |
| --- | --- | --- | --- |
| `bf16` / `fp16` / `fp32` | `s.mom_buf` (merged) | `s.mom_buf` (same tensor) | CPU fp add (cheap) |
| `int8` (per-row BF16 scale) | `s.accum` (bf16) | `s.mom_buf` (int8) + `s.mom_scale` (bf16) | CPU bf16 add to `accum` (cheap); dequant-add-requant happens once at `step()` |
| `mxfp8` (E4M3 + per-block E8M0 scale) | `s.accum` (bf16) | `s.mom_buf` (E4M3) + `s.mom_scale` (E8M0) | CPU bf16 add to `accum` (cheap); dequant-add-requant happens once at `step()` |

### API surface

- `_accumulator_target(s)` returns `(target, cast_dtype)`:
  - AdamW: `(s.m, s.m.dtype)`
  - Muon, quantized: `(s.accum, torch.bfloat16)`
  - Muon, fp*: `(s.mom_buf, s.mom_buf.dtype)`
- `register_grad_offload_hooks` always emits
  `("add", target, src)` for muon (no more `mxfp8_muon` /
  `int8_muon` pending entry kinds).
- `flush_pending_grads`, `flush_manual_flush_params`, and
  `accumulate_grads_to_cpu` only handle the plain `("add", ...)`
  triple. The dequant-add-requant cycle has moved to
  `_quantize_accum_to_mom_buf`, called from
  `CPUMuon.step` at cycle end (one requantize per param per
  step).
- `_scale_accum(s, coef)` (renamed from `_scale_mxfp8_mom_buf`)
  scales the right accumulator per param kind: `s.m` for
  AdamW, `s.accum` for quantized muon, `s.mom_buf` for fp*
  muon. The previous FP8 dequant-mul-requant round-trip is
  gone — the accumulator is always a floating-point dtype.
- `state_dict` / `load_state_dict` save and restore `s.accum`
  for mid-cycle resume.

### Memory cost

The separate `s.accum` is 2 bytes/elt (BF16) for every
quantized-muon param. Cost vs the merged design:
- int8 muon: 1 byte/elt (int8 mom_buf) + 2 bytes/elt (bf16
  per-row scale) + 2 bytes/elt (bf16 accum) = 5 bytes/elt
  vs 3 bytes/elt before. About 67% more CPU memory.
- mxfp8 muon: 1 byte/elt (E4M3 mom_buf) + 0.03 bytes/elt
  (E8M0 per-block scale, 1 byte per 32) + 2 bytes/elt (bf16
  accum) = ~3.03 bytes/elt vs ~1.03 bytes/elt before.
  About 3x more CPU memory.
- fp* muon: unchanged. No separate `s.accum`.

The memory cost is paid back by ~10x faster per-mb sync:
~30 ms per mb for the separate-accumulator path vs ~7000 ms
for the merged-accumulator + per-mb dequant-add-requant
cycle. The cost also includes one requantize per param per
optimizer step (amortized over `gradient_accumulation_steps`
microbatches), so the net per-step CPU sync cost drops
~3-4x at production scale.

### Regression test

`test/_tmp/test_muon_separate_accum.py` pins the design:

1. `test_mxfp8_per_mb_flush_is_fast` — at 8-layer smoke
   scale, the per-mb flush for mxfp8 muon must be <2 s
   (pre-fix: ~7 s).
2. `test_int8_per_mb_flush_is_fast` — same budget for int8.
3. `test_mxfp8_final_param_close_to_bf16` — the mxfp8 path's
   post-step param must match the bf16-muon post-step param
   to within quantization noise (<5% rel diff). Pins that the
   redesign changed only the per-mb path, not the algorithm.
4. `test_bf16_momentum_unchanged` — bf16 muon must still use
   the merged-accumulator design (no separate `s.accum`).
   `_accumulator_target` returns `(s.mom_buf, bfloat16)` for
   bf16 muon.

If any of these regress, the redesign has been broken or
reverted — the next time a quantized storage config is
shipped, the per-mb CPU sync cliff will return.
