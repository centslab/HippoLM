#pragma once

// Minimal torch_utils.h shim for torch 2.9.1.
//
// vllm's marlin.cu #includes "libtorch_stable/torch_utils.h" which (in
// newer torch) defines a `get_current_cuda_blas_handle` helper using
// `torch_get_current_cuda_blas_handle` from torch's stable ABI. That
// symbol does NOT exist in torch 2.9.1's shim.h — it was added in a
// later release. So we provide a 2.9.1-compatible shim that defines
// only what the Marlin code actually uses:
//
//   - STD_TORCH_CHECK  (from <torch/headeronly/util/Exception.h>)
//   - STD_TORCH_CHECK_NOT_IMPLEMENTED  (alias around STD_TORCH_CHECK)
//
// Marlin itself does NOT call get_current_cuda_blas_handle, so we just
// omit it. If a future change starts needing it, swap to a newer torch.

#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/util/Exception.h>

#include <cuda_runtime.h>
#include <cublas_v2.h>

// Stable ABI equivalent of TORCH_CHECK_NOT_IMPLEMENTED.
#define STD_TORCH_CHECK_NOT_IMPLEMENTED(cond, ...) \
  STD_TORCH_CHECK(cond, "NotImplementedError: ", __VA_ARGS__)