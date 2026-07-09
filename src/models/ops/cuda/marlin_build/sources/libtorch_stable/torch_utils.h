#pragma once

// Minimal torch_utils.h shim.
//
// vllm's marlin.cu #includes "libtorch_stable/torch_utils.h" which (in
// newer torch) defines a `get_current_cuda_blas_handle` helper using
// `torch_get_current_cuda_blas_handle` from torch's stable ABI. That
// symbol is not stable across torch versions, so we provide a shim
// that defines only what the Marlin code actually uses:
//
//   - STD_TORCH_CHECK  (from <torch/headeronly/util/Exception.h>)
//   - STD_TORCH_CHECK_NOT_IMPLEMENTED  (alias around STD_TORCH_CHECK)
//
// Marlin itself does NOT call get_current_cuda_blas_handle, so we just
// omit it. If a future change starts needing it, add the helper here.
//
// Last touched when torch 2.9.1 -> 2.12.0: the file used to end with
// a trailing backslash on the STD_TORCH_CHECK_NOT_IMPLEMENTED macro.
// nvcc 12.x treated that as harmless whitespace; nvcc 13.0 + torch
// 2.12's stable headers (library.h, etc.) became stricter and the
// dangling backslash-newline caused the next translation-unit token
// to be glued into the macro body, corrupting HIDDEN_NAMESPACE_BEGIN
// expansion. Cleaned: the macro now ends with a real newline.

#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/util/Exception.h>

#include <cuda_runtime.h>
#include <cublas_v2.h>

// Stable ABI equivalent of TORCH_CHECK_NOT_IMPLEMENTED.
// Single-line #define to avoid the nvcc 13.0 "backslash-newline at
// end of file" warning on the trailing line-continuation (which used
// to break HIDDEN_NAMESPACE_BEGIN expansion under torch 2.12).
#define STD_TORCH_CHECK_NOT_IMPLEMENTED(cond, ...) STD_TORCH_CHECK(cond, "NotImplementedError: ", __VA_ARGS__)