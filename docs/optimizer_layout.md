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
