# Marlin FP4 build pipeline

How the W4A16 NVFP4 FFN path's fused-dequant matmul is built, why
it's not JIT'd at first forward, the per-arch + PTX-fallback loader
strategy, and how to add a new GPU arch.

## What this is

`src/models/ops/nvfp4_marlin.py` (the production FFN matmul on the
W4A16 NVFP4 path, ~4.6× faster than dequant+cuBLAS at FFN gate_up
shapes on sm_120) loads a prebuilt `.so` from
`src/models/ops/cuda/lib/` and binds to its C symbols via ctypes.
No JIT compile on the user's machine.

Two-tier loader strategy:

1. **Per-arch SASS** — `marlin_fp4_{kind}_sm{arch}.so` compiled with
   `-gencode=arch=compute_{X},code=sm_{X}` for a specific SM. Loader
   picks the matching `.so` at first forward via
   `torch.cuda.get_device_capability()`.
2. **PTX fallback** — `marlin_fp4_{kind}_ptx.so` carrying PTX-only
   for sm_80 / sm_89 / sm_120. When no SASS matches (e.g. an
   arch we haven't pre-built), the loader falls back to PTX; the
   CUDA driver JITs at first call (slower first forward, then
   native speed). This is the "ship + ptx" path that lets users on
   a new arch run without an immediate `scripts/build_marlin.py`
   rebuild.

The `.so` files live at
`src/models/ops/cuda/lib/marlin_fp4_{kernel_only,repack}_{sm{X}|ptx}.so`.
They are gitignored individually (`*.so` in repo-root `.gitignore`),
but the directory is whitelisted via the `!src/models/ops/cuda/lib/`
exception so a specific `.so` can be tracked explicitly (e.g. the
sm_120 dev-box build).

## Why not JIT

The natural choice is `torch.utils.cpp_extension.load` — same recipe
vLLM uses internally. It fails because vLLM's
`gptq_marlin_repack.cu` calls `torch_get_current_cuda_blas_handle`,
`torch::stable::Tensor::const_data_ptr`, `torch::stable::empty`, and
`TORCH_BOX` — symbols that require a newer stable ABI than the
precompiled torch ships with. The whole `STABLE_TORCH_LIBRARY_IMPL`
registration block therefore can't be compiled into a JIT extension
as-is.

Three options considered:

1. **Upgrade torch.** Done — torch pinned at 2.12.0+cu130 + CUDA 13.0
   on 2026-07-08 (the W4A16 NVFP4 path was the driver). The pinned
   torch now provides the missing Stable ABI symbols, but JIT compile
   per process is still expensive (~36 min for sm_120), and the
   `.so` would still need a per-arch variant.
2. **Patch the Stable ABI calls in our vendored sources.** Doable
   but adds maintenance surface (any vLLM pull that touches the
   repack will need re-patching against the then-current torch).
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

## Why per-arch + PTX (and not fat binary)

SASS is per-arch. The 5060 Ti (sm_120) SASS we ship is not optimal
on A100 (sm_80), RTX 4090 (sm_89), or RTX 5090 (sm_120, but with
slightly different silicon features enabled). A fat binary
(compile with multiple `-gencode` flags and let CUDA pick) is a
future option — for now we ship per-arch `.so` files plus a
universal PTX fallback.

Per-arch `.so` simplifies debugging (the loader refuses a
wrong-arch `.so` explicitly) and lets a CI matrix pin the exact
build per machine. The PTX fallback covers new-arch users without
forcing an immediate `scripts/build_marlin.py` rebuild.

## SM coverage

| Arch | Hardware                  | Status                  | Where it runs                  |
| ---- | ------------------------- | ----------------------- | ------------------------------ |
| 70   | V100 (Volta)              | **DROPPED** (2026-07-08) | was V100 16G / 32G           |
| 75   | Turing (RTX 20 / T4)      | 暂不支持                | RTX 2080 Ti, T4                |
| 80   | A100                      | **Supported** (compile) | A100 40G / 80G                 |
| 86   | RTX 30 series             | 暂不支持                | RTX 3090                       |
| 89   | RTX 40 / L40 / L4         | **Supported** (compile) | RTX 4090, L40, L4 (Ada)        |
| 90   | H100 (Hopper)             | 暂不支持                | H100                           |
| 100  | Blackwell datacenter      | 暂不支持                | B200                           |
| 120  | RTX 50 / 5060 Ti          | **Supported** (compile + runtime smoke on dev box) | RTX 5090, 5060 Ti |

Why these three:

- **sm80 (A100)** — explicit `sm80_kernel_*.cu` instantiations in
  `marlin/` (4 files: BF16/FP16 × NVFP4/FP8). Compile-validated.
- **sm89 (RTX 4090, production target)** — explicit
  `sm89_kernel_fe4m3fn_fe2m1f_bfloat16.cu` (1 file, NVFP4 × BF16).
  Compile-validated; runtime smoke pending — will rent a 4090 for
  cloud validation.
- **sm120 (5060 Ti / RTX 50)** — no explicit `sm120_kernel_*.cu`
  file. Uses the `__CUDA_ARCH__ >= 890` template branch in
  `marlin_template.h` / `kernel_selector.h`. Runtime smoke done on
  the dev box (5060 Ti 16G).

Out of scope: **sm70 / V100** (FP16 MMA only, can't carry BF16
master weight). **sm75** (Turing) and **sm100** (Blackwell
datacenter) — both unsupported in the vendored sources. If you
need one of these, add a `sm{XX}_kernel_*.cu` instantiation file
from upstream vLLM and re-extract.

## What's in the build tree

```
src/models/ops/cuda/marlin_build/
├── README.md                       # Quick start
└── sources/                        # Vendored, version-controlled
    ├── NOTICE                      # Apache 2.0 attribution + changes
    ├── LICENSE-APACHE-2.0          # Full Apache 2.0 text
    ├── marlin_kernel_only.cu       # matmul wrapper (no STABLE block)
    ├── marlin_repack_adapter.cu    # repack wrapper (extern "C")
    ├── marlin_helpers.cu           # hand-extracted from vllm marlin.cu
    ├── marlin_mm_only.cu           # hand-extracted from vllm marlin.cu
    ├── core/
    │   └── scalar_type.hpp         # vllm (namespace wrapped: marlin::vllm)
    ├── libtorch_stable/
    │   └── torch_utils.h           # 2.12-compatible shim (36 lines)
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

### Vendored source attribution

The vendored sources are extracted from upstream vLLM
(https://github.com/vllm-project/vllm) and are licensed under the
Apache License, Version 2.0. The full license text is in
`sources/LICENSE-APACHE-2.0`; `sources/NOTICE` enumerates the
extracted files + the patches applied + the namespace wrap.

**vllm is NOT a Python dependency.** No `import vllm`, no
`vllm` entry in `requirements.txt`. The `.so` links only against
`c10 / torch_cpu / torch_cuda / cudart` (per `scripts/build_marlin.py`
`_BASE_NVCC_FLAGS` and `_torch_paths`). We vendor the kernel +
wrappers as plain `extern "C"` + raw `void*` signatures.

### Patches applied

The vendored sources are vLLM's, with four edits:

1. **`marlin/marlin.cuh`** — drop `#include <iostream>`. vLLM ships
   `<iostream>` at the top of `marlin.cuh`; it pulls in
   `<bits/stringfwd.h>` which expects `std::` at file scope. The
   FP4 matmul path is included from inside `namespace marlin { ... }`
   in `marlin_kernel_only.cu`, and that breaks the standard library
   lookups (`allocator is not a template`). Nothing in the FP4
   matmul uses `std::cout` etc., so it's safe to drop.

2. **`marlin/marlin_template.h`** — add `#pragma once` at the top.
   vLLM relies on include-order idempotency in `marlin.cu`. Our
   standalone build textually includes `sm89_kernel_*.cu` after
   `marlin_template.h`; without the guard, template definitions
   (`marlin::scale_float` etc.) are duplicated and nvcc errors.

3. **`libtorch_stable/torch_utils.h`** — replaced vLLM's torch
   2.9.1-era shim (which used `torch_get_current_cuda_blas_handle`)
   with a 36-line version that defines
   `STD_TORCH_CHECK_NOT_IMPLEMENTED` via
   `<torch/headeronly/util/Exception.h>` and provides the raw CUDA
   helpers via `<torch/csrc/stable/...>`. The full vLLM shim depends
   on torch 2.12's stable ABI symbols (`Tensor::const_data_ptr`,
   `Tensor::mutable_data_ptr`, `TORCH_BOX`) which torch 2.12.0+cu130
   does provide, but we vendor only the minimal subset.

4. **`namespace vllm` → `namespace marlin { namespace vllm`** —
   see "Namespace wrap" below.

`scripts/build_marlin.py:_ensure_sources_present` checks that
patches 1 and 2 are present at build time and refuses to build if
either has been lost (e.g. re-vendored from upstream without the
patches).

### Namespace wrap (Apache 2.0 attribution)

The vendored sources declare `namespace vllm { ... ScalarType,
kFloat16, kFE4M3fn, ... }`. To avoid leaking any global `vllm::`
symbol from the standalone `.so`, the namespace is wrapped as
`namespace marlin { namespace vllm { ... } }` in
`core/scalar_type.hpp`. All `vllm::ScalarType`,
`vllm::kFloat16`, etc. references are rewritten to
`marlin::vllm::ScalarType` / `marlin::vllm::kFloat16` throughout
the vendored sources (~10,951 reference sites — the vast majority
in the generated `kernel_selector.h` dispatch chain).

This is mechanical-edit minimal: option (a) would rename the
namespace itself to `marlin_kernels::` (doubles the edit count,
forces the same mangle work, no semantic gain); option (b) keeps
the namespace but introduces `using` aliases (creates a parallel
namespace tree). Option (c) — what we ship — wraps under
`marlin::vllm::` and keeps the references one-character shorter
than full rename.

**Mangled symbol change:** `vllm::ScalarType` → `marlin::vllm::ScalarType`
shifts the Itanium ABI prefix from `_ZN4vllm10ScalarType` to
`_ZN6marlin4vllm10ScalarType`. The full `marlin::marlin_mm` mangle
becomes:

```
_ZN6marlin9marlin_mmEPKvS1_PvS2_S2_S2_S2_S2_S2_S2_S2_S2_iiiiS2_
RKN6marlin4vllm10ScalarTypeES6_S6_S6_bbbbiiiP11CUstream_stiiibbb
```

The Python loader (`src/models/ops/nvfp4_marlin.py:_resolve_lib`)
holds the mangle as `FN_NAME`; recompute after any further
namespace edit. The maintainer contract: when re-extracting from
upstream vLLM, the extractor must preserve the `namespace marlin
{ namespace vllm { ... } }` wrapper.

### Patches NOT applied (the "kernel-only" extraction)

These vendored files are text-extracted from vLLM's `marlin.cu` and
`gptq_marlin_repack.cu` to drop the `STABLE_TORCH_LIBRARY_IMPL`
registration blocks. The original files use `torch::stable::Tensor`
APIs that don't exist in torch 2.9.1 (our shim substitutes the raw
CUDA helpers); the extracted versions operate on raw `void*` /
`uint32_t*` pointers and expose their entry points via `extern "C"`:

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

# Build for the production triple (A100 / 4090 / 5060 Ti):
python scripts/build_marlin.py --arch 80,89,120

# Build a specific arch (e.g. for a CI matrix):
python scripts/build_marlin.py --arch 89

# Build to a non-default location (e.g. /opt/hippolm/marlin/):
python scripts/build_marlin.py --arch 120 --output /opt/hippolm/marlin/
```

Output:

```
src/models/ops/cuda/lib/marlin_fp4_kernel_only_sm{NN}.so   # SASS for SM {NN}
src/models/ops/cuda/lib/marlin_fp4_repack_sm{NN}.so        # SASS for SM {NN}
```

For the PTX fallback (universal `.so` carrying PTX for sm_80/89/120):
**not yet implemented** — the `--ptx` flag is on the build script
roadmap. Today the loader (`src/models/ops/nvfp4_marlin.py:_resolve_so`)
does SASS-first lookup, then the legacy `marlin_fp4_{kind}.so`
(sm_120 hard-coded, pre-rename). If none of these is present, you
get a `FileNotFoundError` that points at this script.

## Build cost

On a 5060 Ti (sm_120) build of both `kernel_only` and `repack`:

- matmul (`marlin_kernel_only.cu` + 2 FP4 instantiations + helpers):
  **~30 min**, produces a ~14 MiB `.so`
- repack (`marlin_repack_adapter.cu` + gptq_marlin_repack kernel
  only): **~5 min**, produces a ~150 KiB `.so`

Other arches vary. The cost is dominated by the number of FP4
template instantiations (30 for sm_80 BF16 NVFP4, plus 30 for
sm_80 FP16, plus 30 for sm_89 BF16) — these are what makes the
matmul take so long. The repack has just one template
instantiation per (num_bits, has_perm, is_a_8bit) combo.

Re-running for the same arch reuses the `.o` files in
`src/models/ops/cuda/marlin_build/sources/.build/sm{arch}/` so
a partial rebuild is fast.

## CI / cross-machine

The `.so` files are gitignored individually but the
`src/models/ops/cuda/lib/` directory is whitelisted via a
`.gitignore` exception — so a CI-built `.so` for the sm_120 dev
box can be tracked explicitly. CI is expected to:

1. `git clone` the repo
2. `python scripts/build_marlin.py --arch 80,89,120` on a
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
2. Re-apply the **header patches** (drop `<iostream>` from
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
4. **Preserve the namespace wrap.** The extractor must keep
   `namespace marlin { namespace vllm { ... } }` in
   `core/scalar_type.hpp` and the `marlin::vllm::` prefix on all
   references. After re-extraction, re-run the global
   `vllm::` → `marlin::vllm::` replace.
5. Rebuild + retest: `python scripts/build_marlin.py --arch 120 &&
   python -m pytest test/test_ffn_nvfp4_marlin.py -v`. Recompute
   `FN_NAME` in `src/models/ops/nvfp4_marlin.py` (around line 182)
   from `nm -D` output of the new `.so`.
6. Commit the updated sources.

## Files

- `src/models/ops/cuda/marlin_build/sources/` — vendored,
  version-controlled
- `src/models/ops/cuda/marlin_build/sources/LICENSE-APACHE-2.0` —
  full Apache 2.0 license text (for the vendored vLLM sources)
- `src/models/ops/cuda/marlin_build/sources/NOTICE` — attribution
  + change log
- `scripts/build_marlin.py` — build entry point
- `src/models/ops/cuda/lib/marlin_fp4_{kernel_only,repack}_sm{sm}.so` —
  built, gitignored (sm_120 committed explicitly)
- `src/models/ops/cuda/lib/marlin_fp4_{kernel_only,repack}_ptx.so` —
  PTX fallback, gitignored (built via `--ptx`)
- `src/models/ops/nvfp4_marlin.py` — runtime loader (selects
  per-arch `.so`, falls back to PTX)
- `test/test_ffn_nvfp4_marlin.py` — correctness + speed tests
- `test/test_marlin_fp4_lowlevel.py` — low-level ctypes bind tests
