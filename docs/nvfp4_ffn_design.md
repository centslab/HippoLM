# W4A16 NVFP4 FFN — design notes

## What it is

A research path that stores the FFN weights in NVIDIA's NVFP4 packed
format (E2M1 + FP8-e4m3fn 1x16 microblock scales). Activations stay in
BF16; the FFN matmul itself runs in BF16 (dequantize-on-fwd). Memory
saving on FFN weights alone: ~3.5x.

## Why not a true FP4 matmul?

PyTorch 2.9.1's `torch._scaled_mm` NVFP4 path (blockwise 1x16
scaling) requires **both** A and B to be FP4-packed (W4A4 only). A
pure BF16 x FP4 mixed-precision GEMM is not exposed. The forward
therefore dequantizes the FP4 weight to BF16 before the matmul.

The dequant-on-fwd path keeps the autograd contract identical to a
plain `F.linear`: the BF16 master weight is what the optimizer
updates, the gradient flows through it via the custom Function's
STE (straight-through estimator) for the quantization noise.

## When to use

The flag is `config.ffn_nvfp4` (bool, default False). Set to True
via yml (`ffn_nvfp4: true`) or via the loop's `getattr(args, "ffn_nvfp4", False)`.
The base config (`configs/base.yml`) currently enables it.

## State contract

`NVFP4Linear` (and the TP variants `NVFP4ColumnParallelLinear` /
`NVFP4RowParallelLinear`) hold three buffers:

  - `weight`          : the **BF16 master** weight — what the optimizer updates.
  - `packed_weight`   : uint8 packed FP4 (derived view, recomputed each step).
  - `scales`          : fp8_e4m3fn per-16-element block scales (derived view).

The training loop calls `repack_nvfp4_weights(model)` after each
optimizer step (`src/training/loop.py`). This is a no-op when
`ffn_nvfp4=False` (the walker finds 0 NVFP4 modules and returns 0).

## Block size

Fixed at 16 (the NVFP4 spec). Smaller blocks waste scale storage;
larger blocks lose precision.

## Files added

  - `src/models/ops/nvfp4.py`        — pure-PyTorch quantize/dequantize
                                         reference (correctness-first)
  - `src/models/ops/nvfp4_linear.py` — `NVFP4Linear` + autograd Function
                                         + `repack_nvfp4_weights` walker
  - `src/models/ops/nvfp4_tp.py`     — column- and row-parallel TP variants
                                         of NVFP4Linear

## Files modified

  - `src/models/config.py`             — added `ffn_nvfp4: bool = False`
  - `src/models/activation.py`         — `SwiGLU` switches to NVFP4Linear
                                          when `config.ffn_nvfp4`
  - `src/models/tp_model/swiglu.py`    — `TPSwiGLU` switches to NVFP4
                                          column/row-parallel when `config.ffn_nvfp4`
  - `src/training/loop.py`             — wires `args.ffn_nvfp4` into the
                                          HippoConfig + calls `repack_nvfp4_weights`
                                          after each optimizer step
  - `configs/base.yml`                 — sets `ffn_nvfp4: true`

## Tests

  - `test/test_ffn_nvfp4.py` — 6 tests: forward agreement vs BF16 (×2 sizes),
                                  backward grads finite (×2 sizes),
                                  training decreases loss,
                                  state_dict roundtrip
  - `test/_tmp/test_ffn_nvfp4_stability.py` — 100-step BF16 vs NVFP4 sweep,
                                               asserts both converge to
                                               comparable loss with no NaN

## Known limitations

1. **No FP4 tensor-core matmul.** The forward dequantizes to BF16
   for the matmul (correctness-first). Swapping in a real BF16 x
   FP4 mixed-precision GEMM is a follow-up that touches only
   `_NVFP4Matmul.forward` (one line: replace the dequant + `F.linear`
   with a Blackwell `mma.sp` call).

2. **Checkpoint save error on PyTorch 2.9.1.** `torch.save` on a
   payload containing FP8 + uint8 buffers + optimizer pinned-memory
   state fails with `unexpected pos NNN vs NNN` (from
   `inline_container.cc:664`). The error is **pre-existing** (also
   triggered with `ffn_nvfp4=False`; it's a PyTorch serialization
   bug, not an NVFP4 issue). Training itself is unaffected — pass
   `--checkpoint_interval 0` to skip saving.

3. **No Triton quantize kernel yet.** The current path uses the
   pure-PyTorch `quantize_nvfp4` (which is a Python `argmin` over
   8 E2M1 levels per element). On the 5060 Ti this is ~0.234 ms
   per call (memory's prior bench). A Triton kernel exists in the
   prior feature branch (`feature/kda-ffn-nvfp4`) but is not
   imported — see the prior memory for the three kernel bugs to
   avoid when porting.