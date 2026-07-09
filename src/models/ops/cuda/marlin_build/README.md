# Marlin FP4 build

This directory holds the vendored sources + per-arch build pipeline
for the W4A16 NVFP4 FFN matmul. The .so files live at
`../lib/marlin_fp4_{kernel_only,repack}_sm{arch}.so` and are loaded
by `src/models/ops/nvfp4_marlin.py` at first forward via ctypes.

The full pipeline + design rationale is documented in
[`docs/marlin_build_pipeline.md`](../../../../../docs/marlin_build_pipeline.md).
This README is the quick-start.

## Quick start

```bash
# Build for the current device (auto-detect SM):
python scripts/build_marlin.py

# Build for the production triple:
python scripts/build_marlin.py --arch 80,89,120

# Build for an H100 (Hopper):
python scripts/build_marlin.py --arch 90
```

The build emits:

```
../lib/marlin_fp4_kernel_only_sm{NN}.so     # ~21 MiB
../lib/marlin_fp4_repack_sm{NN}.so          # ~1.1 MiB
```

where `{NN}` is the SM you asked for (e.g. `120` for 5060 Ti).
`scripts/build_marlin.py` checks that the two header patches (drop
`<iostream>` from `marlin.cuh`, add `#pragma once` to
`marlin_template.h`) are still in place before compiling.

## What's here

```
sources/                     # version-controlled vendored build sources
├── marlin_kernel_only.cu    # matmul wrapper (text-extracted from vllm)
├── marlin_repack_adapter.cu # repack wrapper (extern "C" entry)
├── marlin_helpers.cu        # hand-extracted from vllm marlin.cu
├── marlin_mm_only.cu        # hand-extracted from vllm marlin.cu
├── core/scalar_type.hpp     # vllm
├── libtorch_stable/torch_utils.h  # 2.9.1-compatible shim
└── marlin/                  # patched vllm source tree
    ├── kernel.h
    ├── marlin.cuh           # patch: <iostream> dropped
    ├── marlin_dtypes.cuh
    ├── marlin_mma.h
    ├── marlin_template.h    # patch: #pragma once added
    ├── dequant.h
    ├── kernel_selector.h
    ├── gptq_marlin_repack_kernel_only.cu
    ├── sm80_kernel_bfloat16_fe2m1f_bfloat16.cu
    ├── sm80_kernel_bfloat16_fe4m3fn_bfloat16.cu
    ├── sm80_kernel_float16_fe2m1f_float16.cu
    ├── sm80_kernel_float16_fe4m3fn_float16.cu
    └── sm89_kernel_fe4m3fn_fe2m1f_bfloat16.cu
```

The `.build/sm{NN}/` directory is created on first build (intermediate
`.o` files; not version-controlled).

## Re-extracting from upstream vLLM

When vLLM bumps, see *Maintaining the build tree* in
[`docs/marlin_build_pipeline.md`](../../../../../docs/marlin_build_pipeline.md).
The five hand-extracted files drop vLLM's
`STABLE_TORCH_LIBRARY_IMPL` registration blocks (which need torch's
newer Stable ABI) and replace `torch::stable::Tensor` arguments with
raw `void*` pointers; both wrappers then expose the C entry point as
`extern "C"` for ctypes.
