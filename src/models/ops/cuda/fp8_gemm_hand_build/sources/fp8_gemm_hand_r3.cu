// SPDX-License-Identifier: Apache-2.0
// Round 3: R2 + warp specialization (CUTLASS-style).
//
// Bottleneck analysis from R2: producer is the limit. 1 lane issuing
// 64 KiB TMA per K-tile × 32 K-tiles × ~28 tiles/SM ≈ 1.96 ms,
// matching observed 1.81 ms. Consumer is 3x faster.
//
// Round 3 fixes the structure: explicit Producer WG (4 warps) only
// issues TMA + waits, Consumer WGs only do MMA. They share the smem
// buffer but don't share warps, so instruction cache / registers /
// scheduler slots don't fight. Match existing per-row kernel's
// `bm128_bn128_bk128_s3_cwg2_wm32_wn64_ds` structure (12 warps).
//
// Round 3 changes vs Round 2:
//   - Warp groups: 1 producer WG (4 warps) + 2 consumer WGs (4 warps each)
//   - Warp grid per consumer WG: 4x2 (WARP_M=32, WARP_N=64) — matches
//     CUTLASS's WARP_M=32, WARP_N=64 split for 128x128 with 8 consumer warps
//   - NUM_STAGES: 2 -> 3 (deeper pipeline so producer can stay ahead of consumer)
//   - DIRECT_STORE: skip Y_out smem buffer (frees BM*BN*2 smem)
//   - Swizzled tile rasterization (SWIZZLE_WIDTH=4) for L2 locality
//
// Same MMA, ldmatrix, scale (per-MmaTile), store pattern as R2.
// Per-MmaTile (1x1) FP32 scale, CUTLASS MN-major layout.
//
// Computes: D[m,n] = sum_k A[m,k] * SFA[m/128, k/128] * B[n,k] * SFB[n/128, k/128]
//   A   : [M, K] float8_e4m3fn  row-major
//   B   : [N, K] float8_e4m3fn  row-major (project convention)
//   SFA : [M/128, K/128] FP32 row-major (one per MmaTile, M-major in memory)
//   SFB : [N/128, K/128] FP32 row-major
//   D   : [M, N] bfloat16 row-major
//
// Scale layout note: CUTLASS MN-major layout is K-major in memory
// (`data[m_tile + k_tile * (M/128)]`). Round 3 accepts the natural
// row-major Python shape and relies on the wrapper to transpose on
// the fly (negligible HBM-to-HBM copy vs the GEMM).

#include "common.h"
#include <cuda_fp8.h>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <cuda.h>

using fp8e4m3 = __nv_fp8_e4m3;
using bf16    = __nv_bfloat16;
using bf16_2  = __nv_bfloat162;

// =====================================================================
// TMA + mbarrier + smem swizzle — copied verbatim from fp8_gemm.cuh
// (sm_120 ABI surface, not a project-specific invention).
// =====================================================================

struct TMADescriptor {
    alignas(64) uint64_t raw[16];

    __device__ __forceinline__ void load_2d(uint32_t tile_coord0, uint32_t tile_coord1,
                                            uint64_t smem_addr, uint64_t mbar_addr) const {
        asm volatile(
            "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes"
            " [%0], [%1, {%2, %3}], [%4];"
            :
            : "l"(smem_addr), "l"((const uint64_t *)raw), "r"(tile_coord0), "r"(tile_coord1),
              "r"((uint32_t)mbar_addr)
            : "memory");
    }
};

__device__ __forceinline__ void mbarrier_init(uint64_t *mbar, uint32_t count) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"l"(mbar), "r"(count) : "memory");
}
__device__ __forceinline__ void mbarrier_inval(uint64_t *mbar) {
    asm volatile("mbarrier.inval.shared::cta.b64 [%0];" ::"l"(mbar) : "memory");
}
__device__ __forceinline__ void mbarrier_expect_tx(uint64_t smem_mbar_addr, uint32_t bytes) {
    asm volatile(
        "mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"l"(smem_mbar_addr), "r"(bytes) : "memory");
}
__device__ __forceinline__ void mbarrier_arrive(uint64_t smem_mbar_addr) {
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"l"(smem_mbar_addr) : "memory");
}
__device__ __forceinline__ void mbarrier_wait(uint64_t smem_mbar_addr, uint32_t phase) {
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "WAIT_LOOP:\n"
        "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
        "@!p bra WAIT_LOOP;\n"
        "}\n" ::"l"(smem_mbar_addr),
        "r"(phase) : "memory");
}
__device__ __forceinline__ uint32_t smem_u32(const void *smem_ptr) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
}
__device__ __forceinline__ uint32_t f32x2_to_bf16x2(float a, float b) {
    bf16_2 v = __floats2bfloat162_rn(a, b);
    uint32_t r;
    memcpy(&r, &v, 4);
    return r;
}
__device__ __forceinline__ int swizzle_smem_offset(int row, int col, int row_elems) {
    constexpr int SWIZZLE_BYTES = 128;
    int col_bytes = col;                       // ELEM_BYTES = 1 for FP8
    int seg = col_bytes / SWIZZLE_BYTES;
    int in_seg = col_bytes % SWIZZLE_BYTES;
    int bank = in_seg >> 4;
    int off = in_seg & 0xF;
    int sw_bank = bank ^ (row & 7);
    int sw_col = seg * SWIZZLE_BYTES + sw_bank * 16 + off;
    return row * row_elems + sw_col;
}

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

    CUresult res = cuTensorMapEncodeTiled(
        map,
        std::is_same<T, fp8e4m3>::value ? CU_TENSOR_MAP_DATA_TYPE_UINT8
                                        : CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        2, (void *)gmem_ptr, globalDims, globalStrides, boxDims, elementStrides,
        CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle,
        CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (res != CUDA_SUCCESS) {
        const char *errStr = nullptr;
        cuGetErrorString(res, &errStr);
        fprintf(stderr, "cuTensorMapEncodeTiled failed: %s\n", errStr ? errStr : "unknown");
        exit(EXIT_FAILURE);
    }
    return desc;
}

// =====================================================================
// Round 3 kernel — warp specialization (1 producer WG + 2 consumer WGs),
// NUM_STAGES=3, direct store, swizzled rasterization.
//
// Structure:
//   - Total 12 warps (1 producer + 2 consumer WGs of 4 warps each)
//   - Total 384 threads/block
//   - Producer WG: warp 0 lane 0 issues TMA for both A and B; others idle
//   - Consumer WGs: 8 warps total in a 4x2 grid (WARP_M=32, WARP_N=64)
//   - NUM_STAGES=3 with direct_store keeps smem under 99 KB cap
//     (3 stages × 32 KiB = 96 KiB)
// =====================================================================

namespace r3 {

constexpr int BM = 128;
constexpr int BN = 128;
constexpr int BK = 128;
constexpr int NUM_STAGES = 3;

constexpr int WARPS_PER_WG = 4;
constexpr int THREADS_PER_WARP = 32;
constexpr int TOTAL_WGS = 3;                  // 1 producer + 2 consumer
constexpr int TOTAL_THREADS = TOTAL_WGS * WARPS_PER_WG * THREADS_PER_WARP;  // 384

constexpr int NUM_CONSUMER_WGS = 2;
constexpr int NUM_CONSUMER_WARPS = NUM_CONSUMER_WGS * WARPS_PER_WG;  // 8

constexpr int WARP_M = 32;        // each consumer warp covers 32 rows
constexpr int WARP_N = 64;        // each consumer warp covers 64 cols
constexpr int WARPS_M = BM / WARP_M;  // 4
constexpr int WARPS_N = BN / WARP_N;  // 2

constexpr int MMA_M = WARP_M / 16;  // 2
constexpr int MMA_N = WARP_N / 8;   // 8 (2x8 sub-tile pattern per warp)
constexpr int MMA_K = BK / 32;      // 4 (m16n8k32 E4M3)

constexpr int SWIZZLE_WIDTH = 4;

constexpr int TX_BYTES = (BM * BK + BK * BN) * sizeof(fp8e4m3);  // 32 KiB per stage

struct SMemStorage {
    fp8e4m3 A[NUM_STAGES][BM * BK];
    fp8e4m3 B[NUM_STAGES][BK * BN];
    uint64_t full_barrier[NUM_STAGES];
    uint64_t empty_barrier[NUM_STAGES];
};

constexpr int SMEM_SIZE = sizeof(SMemStorage);

// Swizzled tile rasterization for better L2 locality (matches CUTLASS).
__device__ __forceinline__ void rasterize_tile(int tile_id, int num_tiles_m, int num_tiles_n,
                                                int &bm, int &bn) {
    int sw = SWIZZLE_WIDTH < num_tiles_n ? SWIZZLE_WIDTH : num_tiles_n;
    int tiles_per_super_col = num_tiles_m * sw;
    int super_col = tile_id / tiles_per_super_col;
    int within = tile_id % tiles_per_super_col;
    int bn_base = super_col * sw;
    int actual_sw = (bn_base + sw <= num_tiles_n) ? sw : (num_tiles_n - bn_base);
    bm = within / actual_sw;
    bn = bn_base + within % actual_sw;
}

__global__ void __launch_bounds__(TOTAL_THREADS, 1, 1)
fp8_gemm_hand_r3_kernel(
    int M, int N, int K,
    int num_tiles_m, int num_tiles_n,
    const float *__restrict__ SFA,  // [num_tiles_m, K/128] (CUTLASS MN-major: data[m_tile + k_tile * num_tiles_m])
    const float *__restrict__ SFB,  // [num_tiles_n, K/128]
    __grid_constant__ const TMADescriptor tma_A,
    __grid_constant__ const TMADescriptor tma_B,
    bf16 *__restrict__ D)
{
    extern __shared__ __align__(128) char smem_raw[];
    auto &smem = *reinterpret_cast<SMemStorage *>(smem_raw);

    const int tid = threadIdx.x;
    const int warp_id = tid / THREADS_PER_WARP;
    const int lane_id = tid % THREADS_PER_WARP;
    const int wg_id = warp_id / WARPS_PER_WG;
    const int warp_in_wg = warp_id % WARPS_PER_WG;

    const int num_k_tiles = K / BK;
    const int num_blocks = gridDim.x;
    const int total_tiles = num_tiles_m * num_tiles_n;

    // ── Init mbarriers ────────────────────────────────────────────
    if (tid == 0) {
        for (int s = 0; s < NUM_STAGES; s++) {
            mbarrier_init(&smem.full_barrier[s], 1);
            mbarrier_init(&smem.empty_barrier[s], NUM_CONSUMER_WARPS);
        }
    }
    __syncthreads();
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    if (tid == 0) { printf("R3: init done\n"); }

    // ── Producer WG (wg_id=0): lane 0 of warp 0 issues TMA ────────
    if (wg_id == 0) {
        if (warp_in_wg == 0 && lane_id == 0) {
            int stage = 0;
            int phase = 0;
            int total_k = 0;
            for (int tile_id = blockIdx.x; tile_id < total_tiles; tile_id += num_blocks) {
                int bm, bn;
                rasterize_tile(tile_id, num_tiles_m, num_tiles_n, bm, bn);
                for (int k = 0; k < num_k_tiles; k++) {
                    if (total_k >= NUM_STAGES) {
                        mbarrier_wait(smem_u32(&smem.empty_barrier[stage]), phase ^ 1);
                    }
                    mbarrier_expect_tx(smem_u32(&smem.full_barrier[stage]), TX_BYTES);
                    tma_A.load_2d(k * BK, bm * BM,
                                  smem_u32(smem.A[stage]),
                                  smem_u32(&smem.full_barrier[stage]));
                    tma_B.load_2d(k * BK, bn * BN,
                                  smem_u32(smem.B[stage]),
                                  smem_u32(&smem.full_barrier[stage]));
                    stage++;
                    if (stage == NUM_STAGES) { stage = 0; phase ^= 1; }
                    total_k++;
                }
            }
            printf("R3: producer done, total_k=%d\n", total_k);
        }
        // Other 3 warps in producer WG are idle (matches CUTLASS's
        // bm128_bn128_bk128_s3_cwg2_wm32_wn64_ds structure).
        return;
    }

    // ── Consumer WGs (wg_id=1, 2): compute (MMA + per-K-tile scale + accumulate) ──
    const int cwg_id = wg_id - 1;
    const int consumer_warp = cwg_id * WARPS_PER_WG + warp_in_wg;  // 0..7
    const int warp_row = consumer_warp / WARPS_N;
    const int warp_col = consumer_warp % WARPS_N;
    const int m_warp_base = warp_row * WARP_M;
    const int n_warp_base = warp_col * WARP_N;

    int stage = 0;
    int phase = 0;

    for (int tile_id = blockIdx.x; tile_id < total_tiles; tile_id += num_blocks) {
        int bm, bn;
        rasterize_tile(tile_id, num_tiles_m, num_tiles_n, bm, bn);

        // accum[m, n] — persistent across K-tiles
        float acc[MMA_M][MMA_N][4]{};

        for (int k = 0; k < num_k_tiles; k++) {
            mbarrier_wait(smem_u32(&smem.full_barrier[stage]), phase);

            const fp8e4m3 *sA = smem.A[stage];
            const fp8e4m3 *sB = smem.B[stage];

            // tmp_accum — per-K-tile MMA output (cleared after scale FMA)
            float tmp_accum[MMA_M][MMA_N][4]{};

            #pragma unroll
            for (int ki = 0; ki < MMA_K; ki++) {
                const int k_base = ki * 32;

                uint32_t b_frag[MMA_N][2];
                #pragma unroll
                for (int ni = 0; ni < MMA_N; ni++) {
                    const int n_base = n_warp_base + ni * 8;
                    int b_row = n_base + (lane_id & 7);
                    int b_col = k_base + ((lane_id >> 3) & 1) * 16;
                    uint32_t b_addr = smem_u32(&sB[swizzle_smem_offset(b_row, b_col, BK)]);
                    asm volatile(
                        "ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                        : "=r"(b_frag[ni][0]), "=r"(b_frag[ni][1])
                        : "r"(b_addr));
                }

                #pragma unroll
                for (int mi = 0; mi < MMA_M; mi++) {
                    const int m_base = m_warp_base + mi * 16;
                    int a_row = m_base + (lane_id & 7) + ((lane_id >> 3) & 1) * 8;
                    int a_col = k_base + ((lane_id >> 4) & 1) * 16;
                    uint32_t a_addr = smem_u32(&sA[swizzle_smem_offset(a_row, a_col, BK)]);
                    uint32_t a[4];
                    asm volatile(
                        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3])
                        : "r"(a_addr));

                    #pragma unroll
                    for (int ni = 0; ni < MMA_N; ni++) {
                        asm volatile(
                            "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
                            "{%0, %1, %2, %3}, "
                            "{%4, %5, %6, %7}, "
                            "{%8, %9}, "
                            "{%10, %11, %12, %13};\n"
                            : "+f"(tmp_accum[mi][ni][0]), "+f"(tmp_accum[mi][ni][1]),
                              "+f"(tmp_accum[mi][ni][2]), "+f"(tmp_accum[mi][ni][3])
                            : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
                              "r"(b_frag[ni][0]), "r"(b_frag[ni][1]),
                              "f"(tmp_accum[mi][ni][0]), "f"(tmp_accum[mi][ni][1]),
                              "f"(tmp_accum[mi][ni][2]), "f"(tmp_accum[mi][ni][3]));
                    }
                }
            }

            // ── Per-K-tile scale: tmp_accum * scale + accum → accum ──
            // CUTLASS MN-major: data[m_tile + k_tile * num_tiles_m]
            const float scale_a = SFA[bm + k * num_tiles_m];
            const float scale_b = SFB[bn + k * num_tiles_n];
            const float scale_ab = scale_a * scale_b;
            #pragma unroll
            for (int mi = 0; mi < MMA_M; mi++) {
                #pragma unroll
                for (int ni = 0; ni < MMA_N; ni++) {
                    acc[mi][ni][0] += tmp_accum[mi][ni][0] * scale_ab;
                    acc[mi][ni][1] += tmp_accum[mi][ni][1] * scale_ab;
                    acc[mi][ni][2] += tmp_accum[mi][ni][2] * scale_ab;
                    acc[mi][ni][3] += tmp_accum[mi][ni][3] * scale_ab;
                }
            }

            // 1 thread per consumer warp signals empty
            if (lane_id == 0) {
                mbarrier_arrive(smem_u32(&smem.empty_barrier[stage]));
            }

            stage++;
            if (stage == NUM_STAGES) { stage = 0; phase ^= 1; }
        }

        // ── Epilogue: BF16 narrow + direct store ─────────────────
        #pragma unroll
        for (int mi = 0; mi < MMA_M; mi++) {
            const int m_base = m_warp_base + mi * 16;
            const int out_row0 = bm * BM + m_base + (lane_id >> 2);
            #pragma unroll
            for (int ni = 0; ni < MMA_N; ni++) {
                const int n_base = n_warp_base + ni * 8;
                const int out_col0 = bn * BN + n_base + (lane_id & 3) * 2;
                uint32_t c01 = f32x2_to_bf16x2(acc[mi][ni][0], acc[mi][ni][1]);
                uint32_t c23 = f32x2_to_bf16x2(acc[mi][ni][2], acc[mi][ni][3]);
                *reinterpret_cast<uint32_t *>(&D[out_row0 * N + out_col0]) = c01;
                *reinterpret_cast<uint32_t *>(&D[(out_row0 + 8) * N + out_col0]) = c23;
            }
        }
    }

    // ── Cleanup mbarriers ────────────────────────────────────────
    __syncthreads();
    if (tid == 0) {
        for (int s = 0; s < NUM_STAGES; s++) {
            mbarrier_inval(&smem.full_barrier[s]);
            mbarrier_inval(&smem.empty_barrier[s]);
        }
    }
}

}  // namespace r3

// =====================================================================
// Public ABI: per-MmaTile (1x1 MmaTile) FP32 scales, CUTLASS MN-major layout
// =====================================================================
//
// Inputs:
//   A   : [M, K] FP8 E4M3 row-major
//   B   : [N, K] FP8 E4M3 row-major (project convention)
//   SFA : [M/128, K/128] FP32 (K-major in memory: data[m_tile + k_tile*(M/128)])
//   SFB : [N/128, K/128] FP32 (same convention)
//   D   : [M, N] BF16 row-major

extern "C" int fp8_hand_r3_gemm_run(
    int M, int N, int K,
    const void *A, const void *B,
    const void *SFA, const void *SFB,
    void *D,
    cudaStream_t stream)
{
    using namespace r3;

    if (M % 128 != 0 || N % 128 != 0 || K % 128 != 0) {
        fprintf(stderr, "fp8_hand_r3: M=%d N=%d K=%d not all divisible by 128\n",
                M, N, K);
        return -1;
    }

    TMADescriptor tma_A = create_tma_desc_2d<fp8e4m3>(
        reinterpret_cast<const fp8e4m3 *>(A), K, M, BK, BM, CU_TENSOR_MAP_SWIZZLE_128B);
    TMADescriptor tma_B = create_tma_desc_2d<fp8e4m3>(
        reinterpret_cast<const fp8e4m3 *>(B), K, N, BK, BN, CU_TENSOR_MAP_SWIZZLE_128B);

    int num_tiles_m = M / BM;
    int num_tiles_n = N / BN;
    int total_tiles = num_tiles_m * num_tiles_n;

    int num_sm = 0;
    CHECK_CUDA(cudaDeviceGetAttribute(&num_sm, cudaDevAttrMultiProcessorCount, 0));

    CHECK_CUDA(cudaFuncSetAttribute(
        fp8_gemm_hand_r3_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_SIZE));

    int occ = 1;
    CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &occ, fp8_gemm_hand_r3_kernel, TOTAL_THREADS, SMEM_SIZE));
    int num_blocks = std::min(num_sm * occ, total_tiles);

    dim3 grid(num_blocks);
    dim3 block(TOTAL_THREADS);

    fp8_gemm_hand_r3_kernel<<<grid, block, SMEM_SIZE, stream>>>(
        M, N, K, num_tiles_m, num_tiles_n,
        reinterpret_cast<const float *>(SFA),
        reinterpret_cast<const float *>(SFB),
        tma_A, tma_B,
        reinterpret_cast<bf16 *>(D));
    CHECK_CUDA(cudaGetLastError());
    return 0;
}