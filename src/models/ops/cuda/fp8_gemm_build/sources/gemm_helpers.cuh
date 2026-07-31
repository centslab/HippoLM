// SPDX-License-Identifier: Apache-2.0
// Common CUDA helpers shared by all FP8 GEMM kernels (R10, hybrid_a1x128,
// w4a8 native, etc.).
//
// Holds TMA descriptor helpers, mbarrier wrappers, smem swizzle offset,
// FP8 E4M3 / NVFP4 E2M1 decode utilities. Keep this file header-only so
// each kernel .cu can include it without link-time dependencies.
#pragma once
#include "common.h"
#include <cuda_fp8.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>
#include <cuda.h>

using fp8e4m3 = __nv_fp8_e4m3;
using bf16 = __nv_bfloat16;
using bf16_2 = __nv_bfloat162;

// Convert two FP32 to packed BF16x2 (used by epilogue stmatrix and
// direct-store paths).
__device__ __forceinline__ uint32_t f32x2_to_bf16x2(float a, float b)
{
    bf16_2 v = __floats2bfloat162_rn(a, b);
    uint32_t r;
    memcpy(&r, &v, 4);
    return r;
}

// OCP MX 1.0 E8M0 decode (1 byte scale).
// Bits: [E7 E6 E5 E4 E3 E2 E1 E0]. Value = 2^(uint_value - 127).
// Special: 0x00 → 0, 0xff → NaN (we use conservative 0.0 for both).
// Sign bit ignored — E8M0 scales are always non-negative in MX.
__device__ __forceinline__ float e8m0_to_float(uint8_t e)
{
    int exp = (int)(e & 0xff);
    if (exp == 0)    return 0.0f;
    if (exp == 0xff) return 0.0f;
    // 2^(exp - 127) as IEEE-754 single: exponent field = exp, mantissa = 0
    return __uint_as_float(((uint32_t)exp) << 23);
}

// NVFP4 E2M1 decode (4-bit signed magnitude).
// Bits: [S | E1 E0 M]  where S is sign, E is exponent, M is mantissa.
// Magnitude lookup table: index (E:M) = 0..7 → [0, 0.5, 1, 1.5, 2, 3, 4, 6].
// Returns the SIGNED magnitude (sign applied). The caller multiplies by
// the per-microblock scale + global scale separately.
__device__ __forceinline__ float e2m1_decode(uint8_t nibble)
{
    uint8_t mag = nibble & 0x7;
    uint8_t sign_bit = (nibble >> 3) & 0x1;
    // Lookup table (PTX constant memory would be ideal but __constant__
    // is per-file; table loads are L1-cached across lanes).
    float val;
    switch (mag) {
        case 0: val = 0.0f; break;
        case 1: val = 0.5f; break;
        case 2: val = 1.0f; break;
        case 3: val = 1.5f; break;
        case 4: val = 2.0f; break;
        case 5: val = 3.0f; break;
        case 6: val = 4.0f; break;
        default: val = 6.0f; break;  // case 7
    }
    return sign_bit ? -val : val;
}

// ── TMA descriptor (host-side, 128 bytes = CUtensorMap) ─────────────
struct TMADescriptor
{
    alignas(64) uint64_t raw[16];

    __device__ __forceinline__ void load_2d(uint32_t tile_coord0, uint32_t tile_coord1,
                                            uint64_t smem_addr, uint64_t mbar_addr) const
    {
        asm volatile(
            "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes"
            " [%0], [%1, {%2, %3}], [%4];"
            :
            : "l"(smem_addr), "l"((const uint64_t *)raw), "r"(tile_coord0), "r"(tile_coord1),
              "r"((uint32_t)mbar_addr)
            : "memory");
    }
};

__device__ __forceinline__ void tma_store_2d(
    const uint64_t *tma_desc,
    uint32_t tile_coord0, uint32_t tile_coord1,
    uint64_t smem_addr)
{
    asm volatile(
        "cp.async.bulk.tensor.2d.global.shared::cta.bulk_group"
        " [%0, {%1, %2}], [%3];" ::
            "l"(tma_desc),
        "r"(tile_coord0), "r"(tile_coord1),
        "l"(smem_addr)
        : "memory");
}

__device__ __forceinline__ void tma_store_arrive()
{
    asm volatile("cp.async.bulk.commit_group;" ::: "memory");
}

__device__ __forceinline__ void tma_store_wait()
{
    asm volatile("cp.async.bulk.wait_group.read 0;" ::: "memory");
}

// Host-side: create a 2D TMA descriptor.
//   globalDim0 = inner-most (contiguous) dim, globalDim1 = outer dim
//   boxDim0/1  = tile sizes along each dim
template <typename T>
static inline TMADescriptor create_tma_desc_2d(
    const T *gmem_ptr,
    uint32_t globalDim0, uint32_t globalDim1,
    uint32_t boxDim0, uint32_t boxDim1,
    CUtensorMapSwizzle swizzle)
{
    TMADescriptor desc{};
    CUtensorMap *map = reinterpret_cast<CUtensorMap *>(&desc.raw);

    uint64_t globalDims[2] = {globalDim0, globalDim1};
    uint64_t globalStrides[1] = {globalDim0 * sizeof(T)};
    uint32_t boxDims[2] = {boxDim0, boxDim1};
    uint32_t elementStrides[2] = {1, 1};

    // Map the element type T to the matching CUtensorMapDataType enum.
    CUtensorMapDataType fmt;
    if constexpr (std::is_same<T, fp8e4m3>::value) {
        fmt = CU_TENSOR_MAP_DATA_TYPE_UINT8;  // FP8 E4M3 = 1 byte, same wire format
    } else if constexpr (std::is_same<T, bf16>::value) {
        fmt = CU_TENSOR_MAP_DATA_TYPE_BFLOAT16;
    } else if constexpr (std::is_same<T, uint8_t>::value) {
        fmt = CU_TENSOR_MAP_DATA_TYPE_UINT8;
    } else {
        static_assert(sizeof(T) == 0, "create_tma_desc_2d: unsupported element type");
    }

    CUresult res = cuTensorMapEncodeTiled(
        map, fmt, 2, (void *)gmem_ptr, globalDims, globalStrides, boxDims, elementStrides,
        CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle,
        CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (res != CUDA_SUCCESS)
    {
        const char *errStr = nullptr;
        cuGetErrorString(res, &errStr);
        fprintf(stderr, "cuTensorMapEncodeTiled failed: %s\n", errStr ? errStr : "unknown");
        exit(EXIT_FAILURE);
    }
    return desc;
}

// ── mbarrier helpers ───────────────────────────────────────────────
__device__ __forceinline__ void mbarrier_init(uint64_t *mbar, uint32_t count)
{
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"l"(mbar), "r"(count) : "memory");
}
__device__ __forceinline__ void mbarrier_inval(uint64_t *mbar)
{
    asm volatile("mbarrier.inval.shared::cta.b64 [%0];" ::"l"(mbar) : "memory");
}
__device__ __forceinline__ void mbarrier_expect_tx(uint64_t smem_mbar_addr, uint32_t bytes)
{
    asm volatile(
        "mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"l"(smem_mbar_addr), "r"(bytes) : "memory");
}
__device__ __forceinline__ void mbarrier_arrive(uint64_t smem_mbar_addr)
{
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"l"(smem_mbar_addr) : "memory");
}
__device__ __forceinline__ void mbarrier_wait(uint64_t smem_mbar_addr, uint32_t phase)
{
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "WAIT_LOOP:\n"
        "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
        "@!p bra WAIT_LOOP;\n"
        "}\n" ::"l"(smem_mbar_addr),
        "r"(phase) : "memory");
}
__device__ __forceinline__ uint32_t smem_u32(const void *smem_ptr)
{
    return static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
}

// Shared-memory swizzle offset. row_elems = BK (row width in fp8 elements).
// BK >= 128 → 128B swizzle (each row is one 128B swizzle segment,
// XOR with (row & 7)).
// BK < 128 → 64B swizzle (each row is one 64B swizzle segment,
// BK=64 case, XOR with (row & 3)).
// The row_elems parameter is always a compile-time constant (BK template
// argument), so the branch is resolved at compile time.
__device__ __forceinline__ int swizzle_smem_offset(int row, int col, int row_elems)
{
    // BK=128: SWIZZLE_128B, 128B segments, 8 banks, XOR with (row & 7).
    // BK=64:  SWIZZLE_NONE (sm_120 ldmatrix .aligned incompatibility
    // with SWIZZLE_64B — ldmatrix uses hardcoded 8-bank XOR even with
    // 64B TMA swizzle, which overflows 64B rows). Use no-swizzle layout.
    if (row_elems < 128) {
        return row * row_elems + col;  // no swizzle for BK=64
    }
    int seg = col / 128;
    int in_seg = col % 128;
    int bank = in_seg >> 4;
    int off = in_seg & 0xF;
    int sw_bank = bank ^ (row & 7);
    int sw_col = seg * 128 + sw_bank * 16 + off;
    return row * row_elems + sw_col;
}

// Same as swizzle_smem_offset but for BF16 rows (16-bit elements).
// For BK=128 BF16, each row is 256 bytes = 2 swizzle segments.
__device__ __forceinline__ int swizzle_smem_offset_bf16(int row, int col, int row_elems)
{
    // row_elems is BF16 elements per row (e.g. 128 for BK=128 BF16 A).
    // Each element is 2 bytes, so byte stride = row_elems * 2.
    // 128-byte swizzle segments → 64 BF16 per segment.
    int byte_col = col * 2;  // BF16 column → byte offset
    int seg = byte_col / 128;
    int in_seg = byte_col % 128;
    int bank = in_seg >> 4;  // 8 banks per 128B segment
    int off = in_seg & 0xF;
    int sw_bank = bank ^ (row & 7);
    int sw_byte = seg * 128 + sw_bank * 16 + off;
    return row * row_elems + sw_byte / 2;
}