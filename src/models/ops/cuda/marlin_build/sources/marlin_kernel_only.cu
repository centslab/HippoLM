// Standalone wrapper for vllm Marlin FP4 kernel against torch 2.9.1.
//
// Goal: compile `marlin::marlin_mm()` and the FP4 (sm89) Marlin kernel
// template instantiations into a single shared library that can be
// loaded from Python via torch.utils.cpp_extension.load.
//
// Key constraint: vllm's headers (kernel.h, marlin_template.h,
// scalar_type.hpp) include <string>, <tuple>, <variant>, etc. These
// pull in <bits/stringfwd.h> which expects `std::` to be at FILE SCOPE.
// So we must include them BEFORE entering `namespace marlin { ... }`.
//
// kernel.h and marlin_template.h themselves open/close
// `namespace marlin` (via MARLIN_NAMESPACE_NAME + #if/#else trick in
// marlin_template.h). The helpers (thread_config_t etc.) and
// marlin_mm() are at FILE SCOPE in vllm main marlin.cu but the orphan
// `} // namespace marlin` at line 543 suggests they should be inside
// the namespace — that's what we replicate here.

#define MARLIN_NAMESPACE_NAME marlin

// Pull in vllm and torch headers at FILE SCOPE so their transitive
// std headers (e.g. <string> from core/scalar_type.hpp) are processed
// at file scope. Otherwise bits/stringfwd.h breaks with errors like
// "allocator is not a template".
#include "kernel.h"           // marlin_dtypes.cuh + core/scalar_type.hpp
#include "marlin_template.h"  // defines Marlin<...> template in namespace marlin
#include "libtorch_stable/torch_utils.h"  // STD_TORCH_CHECK, raw CUDA helpers

// Now enter namespace marlin and define the helpers / marlin_mm.
// helpers.cu forward-declares MarlinDefault / MarlinFuncPtr (defined
// in mm_only.cu) and mm_only.cu's marlin_mm references helpers from
// helpers.cu. We textually include helpers.cu FIRST, then mm_only.cu
// resolves the forward decls with real definitions.
namespace marlin {

#include "marlin_helpers.cu"
#include "marlin_mm_only.cu"

}  // namespace marlin

// The sm89 explicit instantiations self-wrap in `namespace marlin {`
// via MARLIN_NAMESPACE_NAME. They're text-included here, at FILE SCOPE.
#include "marlin/sm89_kernel_fe4m3fn_fe2m1f_bfloat16.cu"

// sm80 instantiations: NVFP4 path (BF16 act + FE2M1f weight + FE4M3fn
// per-block scales). 30 explicit instantiations. Also self-wraps in
// `namespace marlin {` via MARLIN_NAMESPACE_NAME.
#include "marlin/sm80_kernel_bfloat16_fe2m1f_bfloat16.cu"

// Note: vllm's `marlin_gemm` high-level entry (which dispatches
// scalar types + calls `marlin::marlin_mm`) uses `torch::stable::Tensor`
// methods (device(), const_data_ptr(), mutable_data_ptr(),
// torch::stable::empty, TORCH_BOX) that don't exist in torch 2.9.1's
// stable ABI. We call `marlin::marlin_mm` directly via ctypes instead
// (see test_marlin_fp4.py).