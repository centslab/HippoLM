
> **⚠️ 2026-07-27 audit**: TF claims in this file may have used the
> buggy cudaEvent single-event timing pattern (2.07x under-report on
> sm_120 for ctypes-loaded .so kernels). Real numbers via batched
> cudaEvent / torch.profiler are roughly half the absolute TF values.
> See memory `feedback_cudaevent_2x_underreport_2026_07_27.md` and
> skill `.claude/skills/bench-flops/SKILL.md`.
> **Rules**: [`saved-tensors-not-hwm`](../.claude/rules/saved-tensors-not-hwm.md) · [`dont-target-v100`](../.claude/rules/dont-target-v100.md) — NVFP4 FFN 走 sm_80+ BF16 MMA；VRAM 改动以 `max_memory_allocated()` 而非 `saved_tensors` 为准。**Skill**: [`step-perf-remeasure`](../.claude/skills/step-perf-remeasure/SKILL.md) — re-measure step time after any change to the FFN path. **Doc**: [`optimizer_kernel_design.md`](optimizer_kernel_design.md) — Triton pack + Marlin + in-backward D2H design patterns.

# W4A16 NVFP4 FFN — design notes (mode-3 production state, 2026-07-15)

**Status: production (NVFP4 mode-3).** Replaces the BF16 master +
packed FP4 view design that was the original intent. The current
state is "FP4 packed IS the source of truth" — there is no BF16
master on GPU. See `project_nvfp4_mode3.md` in auto-memory for
the full migration history (5 latent bugs + the 2026-07-10
wiring-gap fix that flipped this from "research path" to
"production default"). Three related flags land this in
`base.yml`:

  - `ffn_nvfp4: true`
  - `ffn_nvfp4_marlin: true`  (the W4A16 fused-dequant matmul)
  - `ffn_nvfp4_no_bf16_master: true`  (mode-3)

## What it is

A production path that stores the FFN weights in NVIDIA's NVFP4
packed format (E2M1 + FP8-e4m3fn 1x16 microblock scales) **as the
only on-device representation**. The Marlin kernel (see
[`docs/marlin_build_pipeline.md`](marlin_build_pipeline.md))
performs the W4A16 fused-dequant matmul at ~3.5× cuBLAS BF16 at
FFN gate_up / down shapes on sm_120 (47-49 TFLOPS, ≈ 0.93× of
the cuBLAS BF16 peak on torch 2.12 — was 3.5× on torch 2.9.1).
Activations stay in BF16 throughout.

Memory footprint at prod shape (32 layers × 3 FFN modules):
~324 MiB packed FP4 + 0.5 MiB scales × 32 layers ≈ 10.4 GiB
cumulative, but only 1 module's worth resident per fwd
(~325 MiB peak).

## Why not a true FP4 matmul?

torch 2.12's `torch._scaled_mm` NVFP4 path (blockwise 1x16
scaling) still requires **both** A and B to be FP4-packed
(W4A4 only) — torch 2.9.1's behavior is unchanged. A pure BF16
× FP4 mixed-precision GEMM is not exposed via the public API.
The forward therefore dequantizes the FP4 weight to BF16 inside
the Marlin kernel (register dequant + `cp.async` double-buffered
prefetch) before issuing the BF16 tensor-core MMA. See the
Marlin doc for the per-arch + PTX-fallback build.

The dequant-on-fwd path keeps the autograd contract identical to
a plain `F.linear`: the FP4 packed weight is what the optimizer
indirectly updates (via STE on the BF16 working copy the
optimizer step writes); the gradient flows through it via the
custom Function's STE for the quantization noise.

## When to use

Three flags, all default-on in `configs/base.yml`:

| Flag | Default | What it controls |
|---|---|---|
| `ffn_nvfp4` | `True` | Use NVFP4 packed weights for FFN linears |
| `ffn_nvfp4_marlin` | `True` | Use the Marlin W4A16 fused-dequant matmul (vs dequant+cuBLAS path); requires `ffn_nvfp4=True` |
| `ffn_nvfp4_no_bf16_master` | `True` | Don't keep a BF16 master on GPU/CPU; the FP4 packed buffers ARE the source of truth. Requires `ffn_nvfp4=True`. Mode-3. |

Set per-flag via yml or via the loop's `getattr(args, ...)`
override. The trio is independent — `ffn_nvfp4=False` short-
circuits the other two.

## State contract (mode-3)

`NVFP4Linear` (and the TP variants `NVFP4ColumnParallelLinear` /
`NVFP4RowParallelLinear`) hold **two** on-device buffers in mode-3
(no BF16 master):

  - `packed_weight`   : uint8 packed FP4 — **source of truth**.
                         What the optimizer step indirectly
                         updates (via the BF16 working copy held
                         in optimizer state, then quantized back
                         via `repack_nvfp4_weights` after each step).
  - `scales`          : fp8_e4m3fn per-16-element block scales
                         (derived from the BF16 master at repack
                         time).

Pre-mode-3, an additional BF16 `weight` buffer was held on GPU
(~324 MiB per layer at prod). Mode-3 deletes that buffer; the
FP4 packed + scales are now the on-device source of truth. The
optimizer's BF16 momentum / EMA state is on CPU pinned memory
(see [`docs/optimizer_layout.md`](optimizer_layout.md)).

The training loop calls `repack_nvfp4_weights(model)` after each
optimizer step (via `src/training/loop/run.py`). This is a
no-op when `ffn_nvfp4=False` (the walker finds 0 NVFP4 modules
and returns 0). With mode-3, repack reads the BF16 working
copy from the optimizer state, quantizes via `quantize_pack`
(Triton kernel — see below), and writes both `packed_weight` +
`scales` in one pass.

### In-backward D2H (shipped 2026-07-15)

The NVFP4 `_NVFP4NoLeafMatmul` autograd Function stashes `grad_w`
on `module._latest_grad_w` during backward. A callback installed
at `register_nvfp4_module` time fires inside `_stash_grad_w` (in
autograd backward), casts to BF16, does async D2H, records a
stream event, and enqueues to the async CPU-add worker — all in
the bwd critical path so subsequent layers' bwd kernels overlap
with the D2H PCIe traffic. See
[`docs/optimizer_kernel_design.md`](optimizer_kernel_design.md)
Pattern 3 for the design + `project_inback_d2h.md` in auto-
memory for the ship writeup.

Worker-status-aware fallback: when no async worker is running
(unit-test mode), the callback falls back to stashing on
`module._latest_grad_w` exactly like the original code.
`accumulate_grads_to_cpu` skip condition becomes "stash is None"
(not "callback is set"), so test path is unaffected.

Result at dev shape (4 layers, n_chunks=8):
`accumulate_grads_to_cpu` median per chunk 23.52 ms → 0.11 ms;
step time -21.7%; HWM unchanged.

## Block size

Fixed at 16 (the NVFP4 spec). Smaller blocks waste scale storage;
larger blocks lose precision. Triton pack kernel
(`_quantize_pack` in `src/models/ops/nvfp4_quant_triton.py`)
operates on 16-element microblocks.

## Triton quantize kernel (shipped 2026-07-14)

Replaces the pure-PyTorch `[numel, 8] + argmin` quantize path
with a single Triton kernel — cascade of strict-greater compares
against the 7 E2M1 midpoints. Per-chunk wall-clock on 5060 Ti
(4096×1536 bf16): **4.4 ms → 0.57 ms (7.7×)**; (4096, 4096) → 9×.
Per-step savings: ~730 ms (~1.2% of the 32 s/step budget).

**Byte-exact via `tl.div_rn` + strict `>` at every cascade
boundary** (matches PyTorch argmin's first-minimum tie-breaking;
Triton's default `/` uses `__fdividef` fast-math which is ~2 ULP
error and fails at boundary ties like `abs_x = 2.5`).

Implementation: `src/models/ops/nvfp4_quant_triton.py`. The
dispatch is `quantize_nvfp4_with_global_scale` →
`_quantize_pack` → Triton kernel; falls back to PyTorch on any
failure. Regression test `test/test_nvfp4_quant_triton.py` (29
cases: byte-exact diff vs reference for 9 shapes × {direct
kernel call, public integration, quant/dequant round-trip} +
non-contig + boundary cases).

## Files

  - `src/models/ops/nvfp4.py`              — pure-PyTorch quantize/dequantize
                                             reference (correctness-first)
  - `src/models/ops/nvfp4_linear.py`       — `NVFP4Linear` + autograd Function
                                             + `repack_nvfp4_weights` walker
  - `src/models/ops/nvfp4_tp.py`           — column- and row-parallel TP
                                             variants of `NVFP4Linear`
  - `src/models/ops/nvfp4_marlin.py`       — Marlin W4A16 fused-dequant matmul
                                             (uses prebuilt `.so` via ctypes;
                                             no JIT)
  - `src/models/ops/nvfp4_quant_triton.py` — Triton `_quantize_pack` kernel
                                             (byte-exact replacement for
                                             PyTorch argmin path)

## Files modified

  - `src/models/config.py`                  — added `ffn_nvfp4`,
                                              `ffn_nvfp4_marlin`,
                                              `ffn_nvfp4_no_bf16_master` flags
  - `src/models/activation.py`              — `SwiGLU` switches to NVFP4Linear
                                              when `config.ffn_nvfp4`
  - `src/models/tp_model/swiglu.py`         — `TPSwiGLU` switches to NVFP4
                                              column/row-parallel when
                                              `config.ffn_nvfp4`
  - `src/training/param_offload/offload.py` — installs in-backward D2H
                                                callback at
                                                `register_nvfp4_module` time
  - `src/training/param_offload/_state.py` — `_accumulator_target` handles
                                              `adamw_nvfp4` kind
  - `src/training/param_offload/muon.py`    — installs callback after state
                                              create (Muon path)
  - `src/training/param_offload/adamw.py`   — symmetric install (prod unused
                                              but for test path)
  - `src/training/loop/run.py`              — wires `args.ffn_nvfp4*` into
                                              the HippoConfig + calls
                                              `repack_nvfp4_weights` after
                                              each optimizer step
  - `configs/base.yml`                      — sets `ffn_nvfp4: true`,
                                              `ffn_nvfp4_marlin: true`,
                                              `ffn_nvfp4_no_bf16_master: true`

## Tests

  - `test/test_ffn_nvfp4.py` — 6 tests: forward agreement vs BF16
                                  (×2 sizes), backward grads finite
                                  (×2 sizes), training decreases
                                  loss, state_dict roundtrip
  - `test/test_nvfp4_quant_triton.py` — 29 cases: byte-exact diff
                                          vs PyTorch argmin reference
                                          for 9 shapes × {direct
                                          kernel call, public
                                          integration, quant/dequant
                                          round-trip} + non-contig +
                                          boundary cases
  - `test/test_nvfp4_no_bf16_master.py` — exercises the in-backward
                                            D2H callback path with both
                                            async worker on and off
                                            (the worker-status-aware
                                            fallback)

## Known limitations

1. **Backward stays BF16 cuBLAS.** The packed-format math is not
   invertible through the Marlin kernel's register dequant — STE
   keeps the autograd contract correct, but the bwd itself runs
   as plain BF16 GEMM (not FP4). This is **not** on the hot path
   at prod shape (~1 ms / 32 layers, vs ~50 ms for the Marlin
   fwd), so it's not a target. The
   `project_marlin_bwd_blocked.md` memory entry has the full
   `SWAP_MN` analysis showing why a Marlin bwd is BLOCKED.

2. **Checkpoint save error on torch 2.12 / CUDA 13.0.** `torch.save`
   on a payload containing FP8 + uint8 buffers + optimizer
   pinned-memory state can fail with `unexpected pos NNN vs NNN`
   (from `inline_container.cc:664`). The error is **pre-existing**
   (also triggered with `ffn_nvfp4=False`; it's a PyTorch
   serialization bug, not an NVFP4 issue). Training itself is
   unaffected — pass `--checkpoint_interval 0` to skip saving.

3. **Marlin `.so` coverage.** Shipped artifacts cover sm120 only;
   sm80/sm89 templates compile-validate, but only sm120 has
   runtime smoke. New arch needs `scripts/build_marlin.py
   --arch <X>` before claiming it works. See
   [`docs/marlin_build_pipeline.md`](marlin_build_pipeline.md).