# Custom CUDA / Triton kernels

This directory is reserved for in-tree custom kernels that
replace the vendored FLA implementations for the small set of
hot-path ops where the upstream FLA Triton kernel is not
fast enough or has a numerical issue.

## What goes here

- One subdirectory per kernel (e.g. `efkda_fwd/`,
  `efkda_bwd/`, `chunk_packer/`). Each subdir contains
  `kernel.cu` (or `kernel.py` for Triton), a `__init__.py`
  exporting the public function, a numerical-correctness
  test under `test/test_<kernel>.py`, and a benchmark under
  `bench/<kernel>.py`.
- A `CMakeLists.txt` per kernel if the build needs anything
  beyond `torch.utils.cpp_extension.load(...)` (rare; the
  current kernels are pure Triton).

## What does NOT go here

- Anything that's just a wrapper around a vendored FLA op.
  Use `src.models.ops._vendored.fla.ops.<op>` directly.
- Pure-PyTorch reference implementations. Those go in
  `test/fixtures/` and are imported by the test, not the
  production path.
- The vendored FLA library itself. That lives in
  `src/models/ops/_vendored/fla/` and is updated by the
  upstream-merge flow, not by adding files here.

## Conventions for new kernels

1. Numerical correctness test FIRST, then benchmark, then
   the production hook. The test pins the contract;
   failing numerical tests block the PR. See
   `docs/triton_kernel_playbook.md` (TODO) for the
   test-first pattern and the autotune-cache pitfalls.
2. The kernel MUST build with the same `nvcc` /
   `compute_capability` as the rest of the project.
   Don't add new build flags without updating
   `setup.py` and the CI image.
3. Memory budget: per the hardware target (single 5060 Ti
   16G), the kernel's peak scratch must fit in 16 GB
   alongside the rest of the model + optimizer state.
   A kernel that pushes us over the line blocks the PR.
4. The kernel's `__init__.py` exports a function with the
   same signature as the FLA op it replaces, so the
   production call site is a one-line change. The
   `from src.models.ops.cuda.<kernel> import <fn>` import
   in the production code is the only place the swap
   happens — do not scatter the import.

## Current state

Empty. The EFKDA rewrite (replacing GDN2) is the first
candidate; see the project's auto-memory for the
performance targets.
