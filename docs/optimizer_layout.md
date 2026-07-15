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
contribution.

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

## Per-step state layout (post-2026-07-15)

As of 2026-07-15 both optimizers use an **explicit-accumulator
layout**: the per-mb grad accumulator and the optimizer's
momentum / EMA state are separate, distinct tensors. The
previous "merged-accumulator" design (where `s.m` for AdamW
and `s.mom_buf` for Muon doubled as both the grad accumulator
and the optimizer's first-moment / momentum state, with β
effectively equal to 1 because the accumulator was reset at
every step) was deleted — the optimization it represented
(`m ← m + grad_accum` being exactly the SGD momentum update
with β=1) is not free of cost: it prevents β1 < 1 EMA smoothing
across steps for AdamW and SGD momentum smoothing for Muon.
Both are real, well-known stabilizers for the respective
optimizers and were restored in this refactor.

### AdamW (`CPUAdamW`)

Per trainable param, three CPU pinned BF16 buffers:

| Field | Role | Lifecycle |
| --- | --- | --- |
| `s.grad` | per-step grad accumulator (sum of microbatch grads) | zeroed at end of every `step()` (or by `zero_cpu_grad_accum` when `found_inf` skips the step) |
| `s.exp_avg` | first moment EMA: `β1·prev + (1-β1)·grad` | preserved across steps; zeroed at init |
| `s.exp_avg_sq` | second moment EMA: `β2·prev + (1-β2)·exp_avg²` | preserved across steps; zeroed at init |

The fused C++ kernel `fused_adam_step_bf16` reads `m` (= our
`exp_avg` after the in-place β1 EMA done in Python) and squares
it for the v update; v bias-correction uses `step` (no m
bias-correction — same as the legacy behavior). The factor is
written FP32 then cast to the param's training dtype for the
GPU apply.

Storage dtype for `grad` and `exp_avg` is
`precision.adamw_m.dtype` (BF16 by default); for
`exp_avg_sq` it is `precision.adamw_v.dtype` (BF16 by default).
The two are independently configurable.

### Muon (`CPUMuon`)

Per trainable param, two CPU pinned buffers at the configured
`precision.muon_momentum` storage dtype (BF16 / FP16 / FP32,
production default BF16):

| Field | Role | Lifecycle |
| --- | --- | --- |
| `s.grad` | per-step grad accumulator (sum of microbatch grads) | zeroed at end of every `step()` (or by `zero_cpu_grad_accum`) |
| `s.exp_avg` | SGD momentum: `β·prev + grad` (Keller Jordan's Muon reference formulation, no (1-β) factor); with Nesterov the NS input is `grad + β·exp_avg` | preserved across steps; zeroed at init |

The step computes the EMA on CPU (in the storage dtype), the
Nesterov correction on CPU (a `grad.add(exp_avg, alpha=β)` if
Nesterov, otherwise just `exp_avg`), H2Ds + casts to FP32 for
NS, then applies the update. The CPU side is the source of
truth across steps — no D2H of the post-step momentum is needed.

Quantized storage formats (`int8` + per-row BF16 scale,
`mxfp8` + per-block E8M0 scale) were removed on 2026-07-12
after long-training runs showed quantization error compounding
over many gradient-accumulation steps and destabilizing
optimization; pre-removal code lives on the
`archive/int8-mxfp8-muon` branch.

### Why the per-step grad accumulator must live on CPU pinned memory

This is **not a missed optimization** — it's a deliberate
design constraint driven by training stability requirements
that the codebase cannot give up. The single, non-negotiable
constraint is that **gradient clipping (`max_grad_norm`) must
work reliably**, and the cheapest substrate that makes the
clip pipeline feasible at production scale is a CPU-side grad
accumulator. Three reasons:

1. **Single fused pass for the L2 norm.** The end-of-step
   `_compute_and_clip_grad_norm` walks *all* per-param `s.grad`
   tensors in one OMP `parallel for` via
   `fused_l2_norm_sq_bf16` (in `cpu_fused.py`). The kernel
   promotes BF16 → FP32 via shift-left-16, squares in FP32,
   and accumulates `Σ ||g||²` into a single FP64 scalar. That
   scalar is what gets TP-all-reduced. If the accumulator lived
   on the GPU, the fused kernel wouldn't apply (it's a CPU
   kernel) and we'd need a per-param `.pow(2).sum().item()`
   loop that `.item()`-syncs the GPU once per param — at ~624 M
   params that's ~5 s/step of pure sync stalls. Empirically
   measured at base.yml shape, the CPU fused norm saves ~7.7
   s/step vs the legacy Python loop (see
   `project_fused_scale_clip.md` in auto-memory for the
   gradnorm-clip half of the same win).

2. **TP all-reduce on a scalar, not per-tensor.** The clip
   threshold `max_grad_norm` is a property of the *global*
   gradient (sum across every param), not per-param. Doing
   the reduction once on a 1-element FP64 tensor is one NCCL
   or gloo call. Doing it per-param would be `n_params`
   collectives and the TP topology doesn't benefit (the
   per-param grads are already partitioned by TP rank via
   the optimizer construction).

3. **In-place clip after the reduction.** Once we have the
   global L2 norm, the clip coefficient is
   `max_norm / (total_norm + eps)` and we apply it to every
   per-param `s.grad` in place via `fused_scale_many_bf16`
   (one OMP pass across all `s.grad` tensors — same fused-
   bandwidth trick). Doing this on the GPU would require
   `n_params` separate kernel launches (no fused kernel
   exists for the in-place scale of N heterogeneous tensors
   on the GPU; we'd fall back to per-tensor `.mul_()`).

For training stability, gradient clipping is the most reliable
guard against loss spikes from a noisy outlier batch. Removing
it would require replacing it with something equally robust
(an adaptive LR schedule that scales on observed gradient
norm statistics — feasible but a much larger refactor and
strictly less general than the simple global clip we have).
The CPU accumulator is the minimum-cost substrate that makes
all three of the above feasible at production scale (624 M
params, ~2.5 GB of CPU grad state, 1 TP all-reduce per step).

**Conclusion**: the `s.grad` accumulator is *unavoidable* as a
CPU pinned buffer in this design. If a future contributor
proposes "move the accumulator to GPU for free H2D savings",
they should re-read the above three points first; the design
isn't a missed optimization, it's a deliberate substrate
choice driven by the hard requirement to keep grad clip working
without per-param overhead.

### History: removed quantized storage paths

Quantized muon storage (`int8` + per-row BF16 scale, `mxfp8` +
per-block E8M0 scale) was implemented in 2026-06 with a
separate `s.accum` BF16 buffer that avoided the
per-mb dequant-add-requant cycle. Long-training runs in
2026-07 showed quantization error compounding over many
gradient-accumulation steps and destabilizing optimization;
both paths were removed on 2026-07-12. The state fields
(`s.mom_scale`, `s.accum`, `s.mxfp8_block_size`) and the
`mxfp8_accum` fused C++ kernel were deleted along with them.

The pre-removal code lives on the `archive/int8-mxfp8-muon`
branch (kept for reference but not maintained). The pre-fix
analysis is preserved in `docs/mxfp8_3_bugs.md` on that branch.

Current supported dtypes are FP32 / FP16 / BF16 only. The
`TensorPrecision(dtype=...)` constructor rejects any other
dtype at config-parse time
(`test_precision_config_rejects_int8_dtype`,
`test_precision_config_rejects_mxfp8_dtype`).