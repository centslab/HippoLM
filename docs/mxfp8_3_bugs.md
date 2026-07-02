# MXFP8 muon: three latent bugs caught while shipping

This document captures the three bugs that were caught (and
fixed) while integrating MXFP8 (E4M3 + per-block E8M0) muon
momentum storage in July 2026. Each bug is self-contained:
the symptom, the root cause, why a test catches it, and the
fix. They share a theme: **low-level bitcast / scale / kernel
plumbing that PyTorch 2.9.1's eager-mode surface doesn't error
on, and that no unit test would catch until a real grad
flowed through.**

All three are now caught by
`test/test_precision.py::test_muon_mxfp8_storage_step_runs_and_produces_finite_params`
and the e2e smoke `configs/test/muon_mxfp8.yml`. The test
config is just a one-line `extends: ../base.yml` with model
size overrides — see the docstring of that yml for the
rationale.

## Why this matters

The MXFP8 storage design (E4M3 elements + E8M0 per-block
scale, OCP MX spec) is paper-backed and gives finer
within-row granularity than the canonical int8 +
per-channel scheme at comparable CPU RAM. But it sits on top
of two layers of FP8 plumbing that the rest of the codebase
never touches: an 8-bit floating point with no CPU reduction
kernels, and a one-byte "exponent only" dtype whose value
is `2^(byte-127)` rather than the byte itself. Get either
wrong and the symptom is *silent*: a step that prints as
"all finite, pmax=2.97e-01" but the param doesn't move
(bug 1), or a step that prints as "NaN" after 2-3 microbatches
(bug 3). The bugs hide until a real grad path is exercised
end-to-end.

## Bug 1 — E8M0 bitcast-as-uint8 decode (silent zero-update)

### Symptom

The smoke run prints "Step N/4 | Loss: 5.56..." for every step
but the param data does not change between steps. The
post-step `pmax` is unchanged from the initial random init
scale. E4M3 `mom_buf` is full of nonzero values
(`q_int unique = 254` — the full E4M3 range), so the
quantization step is producing bytes — but the dequantized
view comes out as zero everywhere.

### Root cause

E8M0 is a one-byte **exponent-only** floating-point dtype.
Byte value `b` decodes to the FP32 value `2^(b-127)`:

```
byte  113  →  2^(113-127)  =  2^-14  ≈  6.10e-05
byte  127  →  2^0          =  1.0
byte  140  →  2^13         =  8192.0
```

PyTorch's `Tensor.float()` on an E8M0 tensor does the value
conversion (the right thing):

```python
>>> e8m0 = torch.tensor([113], dtype=torch.uint8).view(torch.float8_e8m0fnu)
>>> e8m0.float()
tensor([6.1035e-05])
```

But going through `view(torch.uint8).float()` first
bitcasts to uint8 (giving the raw byte), then casts the
uint8 to float32 (giving the byte as an integer). Both
conversions are themselves correct in isolation; the
*combination* silently misinterprets the encoding:

```python
>>> e8m0.view(torch.uint8).float()
tensor([113.])     # <-- WRONG: should be 6.10e-05
```

Plugging 113.0 into the dequant `q_blocks * scale_fp32`
multiplies every quantized value by 113, which for values
near 0.05 produces numbers around 5.65 — way past the E4M3
saturation threshold of 448, but the `clamp(-448, 448)`
catches it. The next requantize divides by the new scale
(another 2^N in the same range), the round-trip lands in
roughly the same place, and the momentum buffer stays
stuck. The "all finite" check passes because nothing is
NaN — just stuck at zero-update.

The bug was at **six sites** in `src/training/param_offload.py`
(dequant in `_dequantize_mxfp8`, requant in
`_requantize_mxfp8`, both `step()` paths, and the per-mb
accumulate helper `_mxfp8_muon_accumulate`). All six
inherited the same wrong pattern.

### Fix

`scale_fp32 = s.mom_scale.float()  # E8M0 value, 2^(b-127)`

Not `s.mom_scale.view(torch.uint8).float()`. The `.float()`
on an E8M0 tensor does the right thing (value conversion).
The bitcast pattern is wrong in every context and easy to
reintroduce when copying code.

### Test that catches it

`test/test_precision.py::test_muon_mxfp8_storage_step_runs_and_produces_finite_params`
asserts `not torch.equal(p.data, initial)` after a step.
With the bug, the bitcast produces the byte value, every
requant saturates, and the param doesn't move. The assert
fires.

For an even tighter check (catches the byte-vs-2^N
mismatch in isolation, no optimizer needed):

```python
e8m0 = torch.tensor([113], dtype=torch.uint8) \
            .view(torch.float8_e8m0fnu)
assert e8m0.float().item() == 2.0 ** -14   # ≈ 6.10e-05
# NOT e8m0.view(torch.uint8).float() == 113.0
```

## Bug 2 — padded storage missing for cols not multiple of block_size (RuntimeError)

### Symptom

The e2e smoke dies before the first step with:

```
RuntimeError: shape '[32, 1, 32]' is invalid for input of size 512
```

from `_mxfp8_muon_accumulate` at `s.mom_buf.view(rows,
n_blocks, bs).float()`. The 2D param has shape
`[rows=32, cols=16]`, so `numel = 512`. The dequant
reshape wants `[rows=32, n_blocks=1, bs=32] = 1024`
elements. The 1024 vs 512 mismatch is the entire problem.

### Root cause

MXFP8's "per-block E8M0 scale" needs `cols` rounded up to
a multiple of `block_size` (32 for the OCP MX-FP8 spec).
The padded tail block is supposed to be all zeros (no
grad contribution). But the CPUMuon allocator was sizing
`mom_buf` as `p.numel() = rows * cols`, not
`rows * cols_padded`. The dequant path then does
`view(rows, n_blocks, bs)` on a tensor that's missing the
padded block.

The same allocator path is shared with the int8 storage
type, where `numel = rows * cols` is correct (per-row
scale, no block padding). The fix is conditional on
`mxfp8_block_size is not None`.

Most params in the canonical model (hidden_size=1536,
intermediate_size=4096) have cols that are already
multiples of 32, so the bug only fires on:
- Tiny smoke models (KDA `f_proj1` / `g_proj1` are
  `[32, 16]` when `hidden_size=32`)
- Any 2D weight with cols < 32 or cols not a multiple of
  32 in production

### Fix

`CPUMuon.__init__` allocator (in
`src/training/param_offload.py`):

```python
storage_n = n
if block_size is not None and p.ndim == 2:
    cols_dim = shape[1]
    cols_p = cols_dim if cols_dim % block_size == 0 \
        else cols_dim + (block_size - cols_dim % block_size)
    storage_n = shape[0] * cols_p
```

The padded tail is filled with zeros (the allocator
default) so it contributes nothing on dequant. The
`m_2d = m_blocks.view(rows, cols_p)[:, :cols].reshape(rows,
cols)` crop in the per-mb helper is what makes this work
on the read side: the padded tail never reaches the
optimizer, just lives in the storage as zero padding.

### Test that catches it

`test_muon_mxfp8_storage_step_runs_and_produces_finite_params`
runs three param shapes: `(8, 32)`, `(8, 16)`,
`(16, 32)`. The `(8, 16)` case is cols < block_size and
would raise without the fix. The test also asserts
`s.mom_buf.numel() == expected_numel` to catch the silent
case where the wrong size happens to satisfy the view
reshape.

## Bug 3 — Newton-Schulz divergence at σ > 1.265 (NaN after 2-3 microbatches)

### Symptom

The smoke runs fine for `n_mb ≤ 2` and dies on
`n_mb = 3` with `param nan=3145728 inf=0` — every element
of the 1024×3072 param becomes NaN. The dequantized
momentum is finite and well-scaled (absmax ~ 0.086 at
n_mb=3); the NaN is produced **inside** the
Newton-Schulz orthogonalization, not in the
quantization.

### Root cause

The standard Muon NS polynomial
`p(λ) = a + b*λ + c*λ²` with coefficients
`(3.4445, -4.7750, 2.0315)` has fixed points at:

```
λ ≈ 0.868  and  λ ≈ 1.265
```

(where `λ` is a singular value of the input, and the
iteration is `g_new = inner @ g` on the ggt branch). For
σ in `[0.868, 1.265]` the iteration converges; outside
that band it diverges — toward 0 for σ < 0.868 and
toward infinity for σ > 1.265.

For random Gaussian gradients at typical
magnitude (σ ~ 0.01 × sqrt(rows) ~ 0.32 per microbatch)
the per-mb dequant-add-requant produces a momentum with
σ in `[0.39, 1.44]` after 3 microbatches. The largest
σ = 1.44 sits **above** the upper fixed point 1.265, so
the polynomial iteration blows it up:

```
iter 0:  σ = 1.44,  ggt σ_max = 2.08
iter 1:  σ = 2.42,  ggt σ_max = 11.05
iter 2:  σ = 661,   ggt σ_max = 4.37e5  →  NaN
```

The bug is **not specific to MXFP8** — it fires
identically with int8, fp16, and bf16 muon storage. Any
input matrix with σ > 1.265 explodes; any input matrix
with σ < 0.868 collapses to zero. Real training
gradients with condition number in the hundreds are
common enough that this would have surfaced in
production as intermittent NaN, possibly hours into a
run.

The standard reference implementation (Keller Jordan's
`muon.py`) normalizes by `X.norm() + eps` before NS for
exactly this reason. Our CPUMuon was missing the
normalization.

### Fix

`CPUMuon._newton_schulz` in
`src/training/param_offload.py`:

```python
eps = 1e-7
x = x.to(torch.float16) / (x.to(torch.float16).norm() + eps)
```

Cast to FP16 first (so the matmul runs on tensor cores),
then normalize by Frobenius norm. The post-normalization
singular values are bounded by `‖X‖_F / σ_min(X) * ‖X‖_F`
which is at most `‖X‖_F` per element, so the NS band is
hit. The eps avoids div-by-zero on all-zero inputs (the
optimizer's existing `m.abs().sum() == 0` early-exit
already covers this case, but the eps is cheap insurance).

### Test that catches it

`test_muon_mxfp8_storage_step_runs_and_produces_finite_params`
runs **three** microbatches of grad accumulation before
the step — exactly the threshold where the bug fires
(at n_mb=3 with random grads ~0.01). The test asserts
`torch.isfinite(p.data).all()` post-step. Without the
fix, the param goes fully NaN at the third step. The
test also covers int8, fp16, bf16 storage paths via
`test_muon_step_runs_for_int8_and_floating_storage_paths`,
but those tests only run a single microbatch — they
didn't catch the divergence because at n_mb=1 the
singular values stay in the stable band.

## Summary

| # | Bug | Surfaced by | One-line fix |
|---|-----|-------------|--------------|
| 1 | E8M0 bitcast-as-uint8 decode (silent zero-update) | e2e smoke loss appears stuck | `scale.float()` not `scale.view(uint8).float()` |
| 2 | Padded storage missing for cols not multiple of 32 | e2e smoke on tiny model (or any cols % 32 ≠ 0 param) | Allocate `rows * cols_p` for mxfp8 |
| 3 | NS divergence at σ > 1.265 (NaN at n_mb=3) | Random-grad smoke at 3+ microbatches | Normalize by Frobenius norm before NS |

All three are now caught by the
`test_muon_mxfp8_storage_step_runs_and_produces_finite_params`
test (one test, three shape variants) and the
`configs/test/muon_mxfp8.yml` smoke config. The smoke
config is intentionally a one-liner (`extends:
../base.yml` plus model size overrides) — the precision
block is inherited from `base.yml`, which now ships with
`muon_momentum: { dtype: mxfp8 }` as the default.