// Minimal C++ adapter exposing gptq_marlin_repack_kernel via raw CUDA
// pointers. Bypasses torch::stable::Tensor API (which differs between
// torch 2.9.1 and the version vllm's gptq_marlin_repack.cu was written
// against).
//
// Signature (extern "C" so ctypes can find it):
//   void marlin_repack_4bit(const int32_t* b_q_weight,   // [K/8, N] packed int32
//                           const int32_t* perm,         // [0] or [K] int32
//                           int32_t* out,                // [K/16, N*8] = [K/16, N*8] int32
//                           int size_k, int size_n,
//                           cudaStream_t stream, int device_index);
//
// The Marlin tile layout is 16 K × 64 N FP4 values (= 32 int32). We
// call the same gptq_marlin_repack_kernel from vllm.

#include <cuda_runtime.h>
#include <torch/csrc/inductor/aoti_torch/c/shim.h>
#include <torch/csrc/stable/accelerator.h>

#include "marlin.cuh"
#include "marlin_dtypes.cuh"
#include "libtorch_stable/torch_utils.h"

// Pull in just the repack kernel template (host-side wrapper that
// uses torch::stable::Tensor APIs not in 2.9.1 is omitted). The
// kernel-only file opens and closes `namespace marlin { ... }`
// correctly, so include at file scope.
#include "marlin/gptq_marlin_repack_kernel_only.cu"

extern "C" {

#define CALL_REPACK(NUM_BITS, HAS_PERM, IS_A_8BIT)                    \
  if (num_bits == NUM_BITS && has_perm == HAS_PERM &&                \
      is_a_8bit == IS_A_8BIT) {                                      \
    cudaFuncSetAttribute(                                            \
        (const void*)marlin::gptq_marlin_repack_kernel<               \
            marlin::repack_threads, NUM_BITS, HAS_PERM, IS_A_8BIT>,  \
        cudaFuncAttributeMaxDynamicSharedMemorySize, max_shared_mem);\
    marlin::gptq_marlin_repack_kernel<marlin::repack_threads,        \
                                      NUM_BITS, HAS_PERM, IS_A_8BIT>  \
        <<<blocks, marlin::repack_threads, max_shared_mem, stream>>>(\
            reinterpret_cast<uint32_t const*>(b_q_weight_ptr),       \
            reinterpret_cast<uint32_t const*>(perm_ptr),             \
            reinterpret_cast<uint32_t*>(out_ptr), size_k, size_n);   \
    return;                                                          \
  }

// Dispatch wrapper. We only ever call num_bits=4, has_perm=false,
// is_a_8bit=false in our NVFP4 test, but support the full set for
// completeness.
void marlin_repack(const int32_t* b_q_weight, const int32_t* perm,
                   int32_t* out, int size_k, int size_n,
                   int num_bits, bool has_perm, bool is_a_8bit,
                   cudaStream_t stream, int device_index) {
  int blocks;
  cudaDeviceGetAttribute(&blocks, cudaDevAttrMultiProcessorCount,
                         device_index);
  int max_shared_mem = 0;
  cudaDeviceGetAttribute(&max_shared_mem,
                         cudaDevAttrMaxSharedMemoryPerBlockOptin,
                         device_index);

  uint32_t const* b_q_weight_ptr =
      reinterpret_cast<uint32_t const*>(b_q_weight);
  uint32_t const* perm_ptr = reinterpret_cast<uint32_t const*>(perm);
  uint32_t* out_ptr = reinterpret_cast<uint32_t*>(out);

  CALL_REPACK(4, false, false)
  CALL_REPACK(4, true, false)
  CALL_REPACK(8, false, false)
  CALL_REPACK(8, true, false)
  CALL_REPACK(4, false, true)
  CALL_REPACK(8, false, true)
  // unreachable for valid configs
}

}  // extern "C"