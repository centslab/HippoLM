// Placeholder for shared headers / declarations visible to both
// the kernel (kernel.cpp) and the auto-generated PYBIND11_MODULE
// in main.cpp (created by torch.utils.cpp_extension.load_inline
// from the functions=[...] list in load_inline.py).
//
// E4M3 / E8M0 byte-level conversion helpers and the fused
// dequant+add+requant kernel live in kernel.cpp.

#include <torch/extension.h>
#include <cstdint>

// Fused in-place kernel:
//   for each [block_size] block in mom_buf:
//     dequant(mom_buf) * mom_scale + grad_bf16 → requantize → mom_buf
//
// All pointers are CPU pinned memory. ``rows * cols_p`` is the
// padded storage shape (cols rounded up to a multiple of
// block_size). ``mom_buf`` and ``mom_scale`` are modified in
// place; ``grad`` is read-only.
void fused_mxfp8_dequant_add_requant(
    torch::Tensor mom_buf,
    torch::Tensor mom_scale,
    torch::Tensor grad,
    int64_t rows,
    int64_t cols_p,
    int64_t block_size
);

// Just the requantize step (used in tests + the "no new grad"
// case where we still need to update the storage format):
//   dequant(mom_buf) * mom_scale → requantize → mom_buf
// (computes the cycle's stored momentum from the existing
// dequantized form, used at step() time after NS.)
void requantize_mxfp8(
    torch::Tensor mom_buf,
    torch::Tensor mom_scale,
    int64_t rows,
    int64_t cols_p,
    int64_t block_size
);
