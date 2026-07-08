# Marlin FP4 — per-arch `.so` manifest

This directory holds **prebuilt Marlin FP4 kernels** for the W4A16
NVFP4 FFN path. The `.so` files are committed in-tree (not gitignored)
so a fresh checkout can run a forward step without paying a ~35 min
`nvcc` compile cost. They are **versioned as code**, with their
provenance recorded below.

## Why per-arch and not JIT

- The kernels are heavily templated (Marlin<a_t, b_t, c_t, s_t, ...>)
  with explicit instantiations keyed on SM arch + dtype tuple. Each
  arch needs its own SASS (sm_120 SASS on a 5060 Ti is not optimal on
  V100 / A100 / RTX 3090 / 4090 / 5090).
- JIT via `torch.utils.cpp_extension.load_inline` was considered; we
  ship prebuilt for the same reason as our other CUDA kernels — first
  step latency is what matters for tight training loops. JIT is fine
  for `torch.utils.cpp_extension`-style ops (see `kda_fwd`); the
  Marlin build is large enough that paying it once per checkout is
  better than paying it per fresh process.

## Files

| File                          | Purpose                          | Loader entry           |
| ----------------------------- | -------------------------------- | ---------------------- |
| `marlin_fp4_kernel_only_sm{NN}.so` | FP4 matmul + sm89 explicit instantiations | `marlin::marlin_mm` |
| `marlin_fp4_repack_sm{NN}.so`      | Weight repack (gptq_marlin_repack)   | `marlin_repack`     |

Naming: `{major}{minor}` of the SM, e.g. `sm120` for compute capability
12.0 (Blackwell consumer, RTX 5060 Ti).

## Build provenance

The build pipeline is in `scripts/build_marlin.py`. Each committed
`.so` is built by running that script on a specific box with:

```
python scripts/build_marlin.py --arch <SM>
```

For a single-arch build, the host machine's driver + CUDA + torch
matter because the `.so` links against torch and c10 shared
libraries via `ctypes.RTLD_GLOBAL`. Cross-version loading is fragile;
see the *Compatibility matrix* below.

### Compatibility matrix

| torch    | CUDA    | driver ≥ | Status     | Notes                          |
| -------- | ------- | -------- | ---------- | ------------------------------ |
| 2.9.1    | 12.8    | 570.x    | Shipped    | Initial ship; deleted in this commit. |
| 2.12.0   | 13.0    | 580.x    | Current    | This build.                    |

If you upgrade torch or CUDA, rebuild every tracked `.so` here:

```
python scripts/build_marlin.py --arch 70,80,86,89,120
git add src/models/ops/cuda/lib/marlin_fp4_*_sm*.so
git commit -m "chore(marlin): rebuild for <torch+CUDA combo>"
```

## Tracking machine config in CI

For a clean rebuild on a new machine, capture provenance with:

```
python -c "import torch; print(f'torch {torch.__version__}, cuda {torch.version.cuda}')"
nvidia-smi --query-gpu=driver_version --format=csv,noheader
nvcc --version | tail -1
```

…and paste the output into the commit message / this file's
*History* section below.

## History

### 2026-07-08 — torch 2.12.0 / CUDA 13.0 / driver 580.76

- box: RTX 5060 Ti (sm_120) on Linux 5.15, Ubuntu 22.04
- torch 2.12.0+cu130, runtime CUDA 13.0, compiled CUDA 13000
- nvcc 13.0.88 from cuda-toolkit-13-0 (apt)
- driver 580.76.05
- Source patches over the vllm-vendored tree:
  - `libtorch_stable/torch_utils.h`: rewrote `STD_TORCH_CHECK_NOT_IMPLEMENTED`
    as a single-line `#define` to silence nvcc 13.0's
    "backslash-newline at end of file" warning, which under torch
    2.12's `HIDDEN_NAMESPACE_BEGIN` macro expansion was corrupting
    namespace resolution.
  - `marlin_kernel_only.cu`: hoisted torch stable headers
    (`<torch/csrc/stable/*.h>`) from inside `namespace marlin { ... }`
    to file scope. Inside the marlin namespace, `HIDDEN_NAMESPACE_BEGIN`
    was expanding to `marlin::torch::stable::detail` and missing
    sibling detail symbols.
  - `marlin_mm_only.cu`: removed the duplicate torch stable header
    includes (now provided by the wrapper at file scope).
- Files rebuilt:
  - `marlin_fp4_kernel_only_sm120.so`
  - `marlin_fp4_repack_sm120.so`

### 2026-07-08 — Apache 2.0 attribution + `vllm::` namespace wrap

The vendored vllm Marlin FP4 sources are derivative of upstream vllm
under Apache 2.0; this commit adds attribution (`LICENSE-APACHE-2.0`
+ `NOTICE` under `marlin_build/sources/`) and wraps the
`namespace vllm { ... }` declaration in `core/scalar_type.hpp` as
`namespace marlin { namespace vllm { ... } }`. All `vllm::X`
references across the vendored sources (~10,951 occurrences,
dominated by the generated `kernel_selector.h` dispatch chain) are
rewritten to `marlin::vllm::X`. This keeps the standalone .so from
leaking any global `vllm::` symbol while staying semantically
identical to vllm's original layout.

Consequence: the C++ ABI mangle for `vllm::ScalarType` shifts
from `_ZN4vllm10ScalarType` to `_ZN6marlin4vllm10ScalarType`.
The full `marlin::marlin_mm` mangle in
`src/models/ops/nvfp4_marlin.py:182` (`FN_NAME`) is recomputed and
updated.

The committed `.so` files are stale until rebuilt — the ctypes bind
in `_resolve_lib` will fail with `AttributeError: undefined symbol`.
Rebuild via:

```
python scripts/build_marlin.py --arch 120
```

then `git add src/models/ops/cuda/lib/marlin_fp4_*_sm120.so` and
commit with a new *History* entry.