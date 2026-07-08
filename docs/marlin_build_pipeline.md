# Marlin FP4 build pipeline

How the W4A16 NVFP4 FFN path's fused-dequant matmul is built, why it's
not JIT'd at first forward, and how to add a new GPU arch.

## What this is

`src/models/ops/nvfp4_marlin.py` (the production FFN matmul on the
W4A16 NVFP4 path, ~4.6× faster than dequant+cuBLAS at FFN gate_up
shapes on sm_120) loads a prebuilt `.so` from
`src/models/ops/cuda/lib/` and binds to its C symbols via ctypes.
No JIT compile on the user's machine. The `.so` is **per-arch** and
is selected at first forward by querying
`torch.cuda.get_device_capability()`.

The `.so` is gitignored (`*.so` in repo-root `.gitignore`); it's
re-built locally on each target machine from the vendored source
tree at `src/models/ops/cuda/marlin_build/sources/`.

## Why not JIT (the story so far)

The natural choice is `torch.utils.cpp_extension.load` — same recipe
vLLM uses internally. The first attempt in `/tmp/marlin_jit_test/`
compiled `marlin.cu` + the FP4 instantiations + `gptq_marlin_repack.cu`
together. nvcc's last words were:

```
gptq_marlin_repack.cu(321): error: identifier "get_current_cuda_stream" is undefined
gptq_marlin_repack.cu(324): error: namespace "torch::stable" has no member "empty"
gptq_marlin_repack.cu(326): error: class "torch::stable::Tensor" has no member "device"
gptq_marlin_repack.cu(333): error: class "torch::stable::Tensor" has no member "const_data_ptr"
gptq_marlin_repack.cu(335): error: class "torch::stable::Tensor" has no member "const_data_ptr"
gptq_marlin_repack.cu(336): error: class "torch::stable::Tensor" has no member "mutable_data_ptr"
gptq_marlin_repack.cu(365): error: identifier "TORCH_BOX" is undefined
```

vLLM's `libtorch_stable/torch_utils.h` defines `get_current_cuda_stream`
via `torch_get_current_cuda_blas_handle` from torch's stable ABI.
That symbol, plus `Tensor::const_data_ptr`, `Tensor::mutable_data_ptr`,
`torch::stable::empty`, and `TORCH_BOX`, were **added in a torch
release newer than 2.9.1** (the version on the production box).
The whole `STABLE_TORCH_LIBRARY_IMPL` registration block depends on
the newer stable ABI; it can't be compiled against 2.9.1.

Three options considered:

1. **Upgrade torch.** Out — breaking change for everything else in
   the project. CUDA 12.8 / torch 2.9.1 is the pinned toolchain.
2. **Patch the Stable ABI calls.** Doable but adds maintenance
   surface (any vLLM pull that touches the repack will need re-patching
   against the then-current torch).
3. **Bypass Stable ABI entirely.** Hand-extract just the kernel +
   repack functions (no `STABLE_TORCH_LIBRARY_IMPL`, no
   `torch::stable::Tensor` plumbing) and expose them via plain
   `extern "C"` symbols. Bind from Python via ctypes.

(3) is what we ship. The text-extraction is in
`marlin_kernel_only.cu` (the matmul wrapper, includes `marlin_helpers.cu`
+ `marlin_mm_only.cu` + sm80/sm89 FP4 instantiations) and
`marlin_repack_adapter.cu` (the repack wrapper, includes
`gptq_marlin_repack_kernel_only.cu`). Both are version-controlled in
`src/models/ops/cuda/marlin_build/sources/`.

## Why per-arch

SASS is per-arch. The 5060 Ti (sm_120) SASS we ship is not what an
A100 (sm_80), RTX 3090 (sm_86), RTX 4090 (sm_89), or RTX 5090
(sm_120, but with subtly different features enabled) want.
A single fat-binary .so is possible (compile with both `-gencode`
flags and CUDA picks the right SASS at load time) but it doubles
the file size for every arch and the runtime loader can't tell you
which SASS it picked. Per-arch .so files are simpler to debug and
the loader refuses to load a wrong-arch .so explicitly.

Target SM coverage:

| Arch | Hardware                  | Where it runs                        |
| ---- | ------------------------- | ------------------------------------ |
| 70   | V100                      | V100 (Volta, FP16 MMA only)          |
| 75   | Turing (RTX 20 series)    | RTX 2080 Ti, T4                      |
| 80   | A100                      | A100 40G/80G                         |
| 86   | RTX 30 series             | RTX 3090                             |
| 89   | RTX 40 / L40 / L4         | RTX 4090, L40, L4 (Ada)              |
| 90   | H100                      | H100 (Hopper)                        |
| 120  | RTX 50 / 5060 Ti          | RTX 5090, 5060 Ti                    |

The Marlin FP4 kernel ships SASS for sm_80, sm_89, sm_90, sm_120
(plus sm_75 for Turing). The production env today is 5060 Ti
(sm_120) only. Building for all of the above costs ~30 min × N
arches of nvcc time per machine.

## What's in the build tree

```
src/models/ops/cuda/marlin_build/
├── README.md                       # Quick start
└── sources/                        # Vendored, version-controlled
    ├── marlin_kernel_only.cu       # matmul wrapper (no STABLE block)
    ├── marlin_repack_adapter.cu    # repack wrapper (extern "C")
    ├── marlin_helpers.cu           # hand-extracted from vllm marlin.cu
    ├── marlin_mm_only.cu           # hand-extracted from vllm marlin.cu
    ├── core/
    │   └── scalar_type.hpp         # vllm core/scalar_type.hpp
    ├── libtorch_stable/
    │   └── torch_utils.h           # 2.9.1-compatible shim
    └── marlin/                     # patched vllm source tree
        ├── kernel.h
        ├── marlin.cuh              # patch: drop <iostream>
        ├── marlin_dtypes.cuh
        ├── marlin_mma.h
        ├── marlin_template.h       # patch: add #pragma once
        ├── dequant.h
        ├── kernel_selector.h
        ├── gptq_marlin_repack_kernel_only.cu
        │                            # (hand-extracted from gptq_marlin_repack.cu)
        ├── sm80_kernel_bfloat16_fe2m1f_bfloat16.cu
        ├── sm80_kernel_bfloat16_fe4m3fn_bfloat16.cu
        ├── sm80_kernel_float16_fe2m1f_float16.cu
        ├── sm80_kernel_float16_fe4m3fn_float16.cu
        └── sm89_kernel_fe4m3fn_fe2m1f_bfloat16.cu
```

### Patches applied

The vendored `marlin/marlin.cuh` and `marlin/marlin_template.h` are
vLLM's, with two small edits:

1. **`marlin.cuh`**: drop `#include <iostream>`. vLLM ships
   `<iostream>` at the top of `marlin.cuh`; it pulls in
   `<bits/stringfwd.h>` which expects `std::` at file scope. The
   FP4 matmul path is included from inside `namespace marlin { ... }`
   in `marlin_kernel_only.cu`, and that breaks the standard library
   lookups (`allocator is not a template`). Nothing in the FP4
   matmul uses `std::cout` etc., so it's safe to drop.

2. **`marlin_template.h`**: add `#pragma once`. vLLM relies on
   include order in `marlin.cu` to keep the header idempotent; the
   sm_89 explicit-instantiation file (which we textually include
   in `marlin_kernel_only.cu`) was duplicating `marlin::scale_float`
   etc. The `#pragma once` fix is a one-liner.

`scripts/build_marlin.py` checks that both patches are present at
build time and refuses to build if either has been lost (e.g.
re-vendored from upstream without the patches).

### Patches NOT applied (the "kernel-only" extraction)

These vendored files are text-extracted from vLLM's `marlin.cu` and
`gptq_marlin_repack.cu` to drop the `STABLE_TORCH_LIBRARY_IMPL`
registration blocks. The original files use `torch::stable::Tensor`
APIs that don't exist in torch 2.9.1; the extracted versions
operate on raw `void*` / `uint32_t*` pointers and expose their
entry points via `extern "C"`:

| Vendored file              | What was extracted                                |
| -------------------------- | ------------------------------------------------- |
| `marlin_kernel_only.cu`    | matmul entry (just `marlin::marlin_mm`)           |
| `marlin_repack_adapter.cu` | repack entry (just `marlin_repack`)               |
| `marlin_helpers.cu`        | `thread_config_t`, `MarlinDefault`, etc.          |
| `marlin_mm_only.cu`        | `marlin::marlin_mm` definition                    |
| `gptq_marlin_repack_kernel_only.cu` | `gptq_marlin_repack_kernel<...>` template |

To re-extract after a vLLM bump: see *Maintaining the build tree*
at the bottom.

## Build command

The build script lives at `scripts/build_marlin.py`. Use it directly:

```bash
# Auto-detect (uses torch.cuda.get_device_capability, falls back to nvidia-smi):
python scripts/build_marlin.py

# Build for the production triple (V100 / A100 / 5060 Ti):
python scripts/build_marlin.py --arch 70,80,89,120

# Build a specific arch (e.g. for a CI matrix):
python scripts/build_marlin.py --arch 89

# Build to a non-default location (e.g. /opt/hippolm/marlin/):
python scripts/build_marlin.py --arch 120 --output /opt/hippolm/marlin/
```

Output:

```
src/models/ops/cuda/lib/marlin_fp4_kernel_only_sm120.so
src/models/ops/cuda/lib/marlin_fp4_repack_sm120.so
```

The loader (`src/models/ops/nvfp4_marlin.py`) finds the right pair
at first forward. If neither the per-arch name nor the legacy
`marlin_fp4_kernel_only.so` (sm_120 hard-coded) is present, you
get a `FileNotFoundError` that points at this script.

## Build cost

On a 5060 Ti (sm_120) build of both `kernel_only` and `repack`:

- matmul (`marlin_kernel_only.cu` + 2 FP4 instantiations + helpers):
  **~30 min**, produces a ~21 MiB .so
- repack (`marlin_repack_adapter.cu` + gptq_marlin_repack kernel
  only): **~5 min**, produces a ~1.1 MiB .so

Other arches vary. The cost is dominated by the number of FP4
template instantiations (30 for sm_80 BF16 NVFP4, plus 30 for
sm_80 FP16, plus 30 for sm_89 BF16) — these are what makes the
matmul take so long. The repack has just one template
instantiation per (num_bits, has_perm, is_a_8bit) combo.

Re-running for the same arch reuses the `.o` files in
`src/models/ops/cuda/marlin_build/sources/.build/sm{arch}/` so
a partial rebuild is fast.

## CI / cross-machine

The .so files are gitignored. CI is expected to:

1. `git clone` the repo
2. `python scripts/build_marlin.py --arch 70,80,86,89,120` on a
   representative machine (or per-machine: pass the arch of the
   runner's GPU)
3. Cache the `cuda/lib/*.so` files as a build artifact

For dev machines: the build is a one-time ~30 min cost per
arch. Add a Makefile target if multiple devs want to share
binaries:

```makefile
# Makefile (not in repo — example for sharing between devs)
marlin-sm120:
	python scripts/build_marlin.py --arch 120 --output /opt/hippolm/marlin/
```

## Adding a new arch

For example, to add sm_90 (Hopper H100):

1. Confirm vLLM ships an instantiation for the FP4 NVFP4 path
   at sm_90. Today only sm_80 / sm_89 / sm_120 are exercised
   for the NVFP4 BF16-act path; sm_90 may need
   `marlin_kernel_only.cu` to add `sm90_kernel_*.cu` instantiations
   (see `src/models/ops/cuda/marlin_build/sources/marlin/` for
   the pattern). Copy from vLLM's
   `csrc/libtorch_stable/quantization/marlin/`.
2. Build it: `python scripts/build_marlin.py --arch 90`
3. Verify the test suite still passes:
   `python -m pytest test/test_ffn_nvfp4_marlin.py -v`
4. Verify the speed test still passes at FFN gate_up
   (sm_90 should be even faster than sm_120 — Hopper's WGMMA +
   TMA give Marlin a free 2× on top of the FP4 throughput).

## Maintaining the build tree

When vLLM's `marlin.cu` / `gptq_marlin_repack.cu` change:

1. Diff the vLLM versions against the vendored copies at
   `src/models/ops/cuda/marlin_build/sources/`. vLLM is at
   `/hy-tmp/vllm/csrc/libtorch_stable/quantization/marlin/` on
   the production box; on other machines, clone vLLM.
2. Re-apply the **two header patches** (drop `<iostream>` from
   `marlin.cuh`, add `#pragma once` to `marlin_template.h`) if
   vLLM has re-introduced the original.
3. Re-extract the **five hand-extracted files** (`marlin_kernel_only.cu`,
   `marlin_repack_adapter.cu`, `marlin_helpers.cu`, `marlin_mm_only.cu`,
   `gptq_marlin_repack_kernel_only.cu`) from the new vLLM sources.
   The extraction rules are:
   - Drop every `STABLE_TORCH_LIBRARY_IMPL(...)` block (no-op for
     a ctypes-only build; the loader uses `extern "C"` symbols).
   - Replace `torch::stable::Tensor` / `std::optional<torch::stable::Tensor>`
     with raw `void*` pointers in the function signatures.
   - Replace `TORCH_BOX(...)` with the bare function pointer.
   - Add `extern "C"` linkage to the public entry point.
4. Build + test: `python scripts/build_marlin.py --arch 120 &&
   python -m pytest test/test_ffn_nvfp4_marlin.py -v`
5. Commit the updated sources.

## Files

- `src/models/ops/cuda/marlin_build/sources/` — vendored, version-controlled
- `scripts/build_marlin.py` — build entry point
- `src/models/ops/cuda/lib/marlin_fp4_kernel_only_sm{sm}.so` — built, gitignored
- `src/models/ops/cuda/lib/marlin_fp4_repack_sm{sm}.so` — built, gitignored
- `src/models/ops/nvfp4_marlin.py` — runtime loader (selects per-arch .so)
- `test/test_ffn_nvfp4_marlin.py` — correctness + speed tests
