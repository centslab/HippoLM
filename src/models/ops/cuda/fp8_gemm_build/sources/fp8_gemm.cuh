// SPDX-License-Identifier: Apache-2.0
// Custom FP8 E4M3 GEMM for sm_120 (RTX Blackwell consumer).
//
// Adapted from /hy-tmp/sm120_gemm (BF16 path) for FP8 E4M3 inputs.
// Computes Y = (A * a_scale_row) @ (B * b_scale_col).T  →  Y [M,N] BF16
//
//   A : [M, K] float8_e4m3fn  row-major     (activation, per-row scale a_scale [M])
//   B : [N, K] float8_e4m3fn  row-major     (weight,     per-col scale b_scale [N])
//   Y : [M, N] __nv_bfloat16  row-major
//
// Strategy (same as sm120_gemm BF16): TMA loads, producer/consumer warp
// specialization, mbarrier multi-stage pipeline, persistent blocks,
// swizzled tile rasterization, 128B smem swizzle, ldmatrix fragments,
// stmatrix + TMA store epilogue. FP8 deltas vs the BF16 reference:
//
//   * MMA is mma.sync.aligned.m16n8k16.row.col.f32.e4m3.e4m3.f32.
//     A fragment = 2 b32 regs/lane (16 rows × 16 K), B = 1 b32 reg/lane
//     (8 N × 16 K). The BF16 kernel uses x4/x2 ldmatrix for a 16x16
//     2-byte tile; the FP8 tile is 16x16 ONE-byte, so A uses
//     ldmatrix.x2 (not x4) and B uses ldmatrix.x1 (not x2). The data
//     distribution of ldmatrix .b16 (thread i -> row i>>2, bytes
//     (i&3)*4+{0..3}) matches the FP8 MMA fragment layout exactly.
//   * BK = 128 so each smem row is 128 bytes — one full 128B swizzle
//     span, same layout arithmetic as the BF16 kernel's 64-elem rows.
//   * Per-row × per-col scale is applied in the epilogue: lane j's
//     four accumulators sit at (row j>>2, col (j&3)*2+{0,1}) and
//     (row (j>>2)+8, same cols) of the 16x8 tile, so the scale factors
//     are loaded per-lane and multiplied before the BF16 RNE narrow.
#pragma once
#include "common.h"
#include <cuda_fp8.h>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <cuda.h>
using fp8e4m3 = __nv_fp8_e4m3;
using bf16 = __nv_bfloat16;
using bf16_2 = __nv_bfloat162;

__device__ __forceinline__ uint32_t f32x2_to_bf16x2(float a, float b)
{
    bf16_2 v = __floats2bfloat162_rn(a, b);
    uint32_t r;
    memcpy(&r, &v, 4);
    return r;
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

    CUresult res = cuTensorMapEncodeTiled(
        map,
        std::is_same<T, fp8e4m3>::value ? CU_TENSOR_MAP_DATA_TYPE_UINT8
                                        : CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        2, (void *)gmem_ptr, globalDims, globalStrides, boxDims, elementStrides,
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

// 128B shared-memory swizzle offset. Each smem row must be 128 bytes
// (BK=128 fp8) so one row = one swizzle span, matching the BF16
// reference's 64-elem rows.
__device__ __forceinline__ int swizzle_smem_offset(int row, int col, int row_elems)
{
    constexpr int SWIZZLE_BYTES = 128;
    int col_bytes = col;                       // ELEM_BYTES = 1
    int seg = col_bytes / SWIZZLE_BYTES;
    int in_seg = col_bytes % SWIZZLE_BYTES;
    int bank = in_seg >> 4;
    int off = in_seg & 0xF;
    int sw_bank = bank ^ (row & 7);
    int sw_col = seg * SWIZZLE_BYTES + sw_bank * 16 + off;
    return row * row_elems + sw_col;
}

// ════════════════════════════════════════════════════════════════════════
// FP8GemmMMA — TMA + warp specialization + m16n8k16 FP8 MMA
// ════════════════════════════════════════════════════════════════════════
template <int BM, int BN, int BK, int NUM_STAGES, int CWG,
          int WARP_M = 64, int WARP_N = 64, bool DIRECT_STORE = false>
struct FP8GemmMMA
{
    static_assert(BK == 128, "BK must be 128 (128B swizzle span per row)");
    static constexpr int WARPS_PER_WG = 4;
    static constexpr int THREADS_PER_WARP = 32;
    static constexpr int THREADS_PER_WG = WARPS_PER_WG * THREADS_PER_WARP;
    static constexpr int TOTAL_WGS = CWG + 1;
    static constexpr int TOTAL_THREADS = TOTAL_WGS * THREADS_PER_WG;

    static constexpr int TX_BYTES = (BM * BK + BK * BN) * sizeof(fp8e4m3);

    static constexpr int NUM_CONSUMER_WARPS = CWG * WARPS_PER_WG;
    static constexpr int MMA_M = WARP_M / 16;
    static constexpr int MMA_N = WARP_N / 8;
    static constexpr int MMA_K = BK / 32;   // m16n8k32 e4m3: 32 K per MMA

    static constexpr int WARPS_M = BM / WARP_M;
    static constexpr int WARPS_N = BN / WARP_N;
    static_assert(WARPS_M * WARPS_N == NUM_CONSUMER_WARPS,
                  "Warp tiling must cover BM×BN exactly");

    static constexpr int SWIZZLE_WIDTH = 4;

    __device__ static void rasterize_tile(int tile_id, int num_tiles_m, int num_tiles_n,
                                          int &bm, int &bn)
    {
        int sw = SWIZZLE_WIDTH < num_tiles_n ? SWIZZLE_WIDTH : num_tiles_n;
        int tiles_per_super_col = num_tiles_m * sw;
        int super_col = tile_id / tiles_per_super_col;
        int within = tile_id % tiles_per_super_col;
        int bn_base = super_col * sw;
        int actual_sw = (bn_base + sw <= num_tiles_n) ? sw : (num_tiles_n - bn_base);
        bm = within / actual_sw;
        bn = bn_base + within % actual_sw;
    }

    // DIRECT_STORE=true drops the Y_out smem buffer entirely: the
    // epilogue writes scaled BF16 straight to global (each warp's
    // 4-lane groups cover 16 contiguous bytes per row — coalesced).
    // This frees BM*BN*2 bytes, letting BM=128/BN=128/BK=128 run
    // NUM_STAGES=3 under the 99 KB cap.
    struct SMemStorage
    {
        fp8e4m3 A[NUM_STAGES][BM * BK];
        fp8e4m3 B[NUM_STAGES][BK * BN];
        bf16    Y_out[DIRECT_STORE ? 1 : BM * BN];
        uint64_t full_barrier[NUM_STAGES];
        uint64_t empty_barrier[NUM_STAGES];
    };

    static constexpr int SMEM_SIZE = sizeof(SMemStorage);

    static void run(
        int M, int N, int K,
        const fp8e4m3 *__restrict__ A,
        const fp8e4m3 *__restrict__ B,
        const float  *__restrict__ a_scale,   // [M] FP32
        const float  *__restrict__ b_scale,   // [N] FP32
        bf16         *__restrict__ Y,
        cudaStream_t  stream = nullptr);
};

// ── FP8 MMA kernel ───────────────────────────────────────────────────
template <int BM, int BN, int BK, int NUM_STAGES, int CWG, int WARP_M, int WARP_N, bool DIRECT_STORE>
__global__ void __launch_bounds__(FP8GemmMMA<BM, BN, BK, NUM_STAGES, CWG, WARP_M, WARP_N, DIRECT_STORE>::TOTAL_THREADS, 1, 1)
    fp8_gemm_mma_kernel(
        int M, int N, int K,
        int num_tiles_m, int num_tiles_n, int total_tiles,
        __grid_constant__ const TMADescriptor tma_A,
        __grid_constant__ const TMADescriptor tma_B,
        __grid_constant__ const TMADescriptor tma_Y,
        const float *__restrict__ a_scale,
        const float *__restrict__ b_scale,
        bf16 *__restrict__ Y)
{
    using P = FP8GemmMMA<BM, BN, BK, NUM_STAGES, CWG, WARP_M, WARP_N, DIRECT_STORE>;
    using SmemStorage = typename P::SMemStorage;

    extern __shared__ __align__(128) char smem_raw[];
    auto &smem = *reinterpret_cast<SmemStorage *>(smem_raw);

    const int tid = threadIdx.x;
    const int warp_id = tid / P::THREADS_PER_WARP;
    const int lane_id = tid % P::THREADS_PER_WARP;
    const int wg_id = warp_id / P::WARPS_PER_WG;
    const int warp_in_wg = warp_id % P::WARPS_PER_WG;

    const int num_k_tiles = K / BK;
    const int num_blocks = gridDim.x;

    if (tid == 0)
    {
        for (int s = 0; s < NUM_STAGES; s++)
        {
            mbarrier_init(&smem.full_barrier[s], 1);
            mbarrier_init(&smem.empty_barrier[s], CWG * P::WARPS_PER_WG);
        }
    }
    __syncthreads();
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");

    // ── Producer warp group: lane 0 of warp 0 issues TMA loads ────
    if (wg_id == 0)
    {
        if (warp_in_wg == 0 && lane_id == 0)
        {
            int stage = 0;
            int phase = 0;
            int total_k = 0;

            for (int tile_id = blockIdx.x; tile_id < total_tiles; tile_id += num_blocks)
            {
                int bm, bn;
                P::rasterize_tile(tile_id, num_tiles_m, num_tiles_n, bm, bn);

                for (int k = 0; k < num_k_tiles; k++)
                {
                    if (total_k >= NUM_STAGES)
                    {
                        mbarrier_wait(smem_u32(&smem.empty_barrier[stage]), phase ^ 1);
                    }
                    mbarrier_expect_tx(smem_u32(&smem.full_barrier[stage]), P::TX_BYTES);

                    // A: [M, K] row-major, box [BK, BM] (dim0=K contiguous)
                    tma_A.load_2d(
                        k * BK, bm * BM,
                        smem_u32(smem.A[stage]),
                        smem_u32(&smem.full_barrier[stage]));
                    // B: [N, K] row-major, box [BK, BN]
                    tma_B.load_2d(
                        k * BK, bn * BN,
                        smem_u32(smem.B[stage]),
                        smem_u32(&smem.full_barrier[stage]));

                    stage++;
                    if (stage == NUM_STAGES) { stage = 0; phase ^= 1; }
                    total_k++;
                }
            }
        }
    }
    // ── Consumer warp groups ──────────────────────────────────────
    else
    {
        const int cwg_id = wg_id - 1;
        const int consumer_warp = cwg_id * P::WARPS_PER_WG + warp_in_wg;
        const int warp_row = consumer_warp / P::WARPS_N;
        const int warp_col = consumer_warp % P::WARPS_N;

        const int m_warp_base = warp_row * WARP_M;
        const int n_warp_base = warp_col * WARP_N;

        int stage = 0;
        int phase = 0;
        bool has_tma_store_in_flight = false;

        for (int tile_id = blockIdx.x; tile_id < total_tiles; tile_id += num_blocks)
        {
            int bm, bn;
            P::rasterize_tile(tile_id, num_tiles_m, num_tiles_n, bm, bn);

            float acc[P::MMA_M][P::MMA_N][4]{};

            for (int k = 0; k < num_k_tiles; k++)
            {
                mbarrier_wait(smem_u32(&smem.full_barrier[stage]), phase);

                const fp8e4m3 *sA = smem.A[stage];
                const fp8e4m3 *sB = smem.B[stage];

                #pragma unroll
                for (int ki = 0; ki < P::MMA_K; ki++)
                {
                    const int k_base = ki * 32;

                    // B fragments: [BN, BK] smem (row n, col k). For
                    // m16n8k32 e4m3, the B fragment is 8 N × 32 K = 2 b32
                    // regs/lane: reg0 = (k (i&3)*4+{0..3}, n i>>2), reg1 =
                    // (k (i&3)*4+16+{0..3}, n i>>2). ldmatrix.x2 loads two
                    // 8x8 b16 tiles: matrix 0 = 8 N-rows × K bytes
                    // [k_base, k_base+16), matrix 1 = same rows × K bytes
                    // [k_base+16, k_base+32). Addresses from lanes 0..15:
                    // row = n_base + (lane&7), col = k_base +
                    // ((lane>>3)&1)*16.
                    uint32_t b_frag[P::MMA_N][2];
                    #pragma unroll
                    for (int ni = 0; ni < P::MMA_N; ni++)
                    {
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
                    for (int mi = 0; mi < P::MMA_M; mi++)
                    {
                        const int m_base = m_warp_base + mi * 16;

                        // A fragment: [BM, BK] smem (row m, col k). For
                        // m16n8k32 e4m3, the A fragment is 16 M × 32 K = 4
                        // b32 regs/lane: reg0 = (row i>>2, k (i&3)*4+{0..3}),
                        // reg1 = (row (i>>2)+8, same k), reg2 = (row i>>2,
                        // k (i&3)*4+16+{0..3}), reg3 = (row (i>>2)+8, k
                        // +16..19). ldmatrix.x4 loads four 8x8 b16 tiles:
                        // matrix 0 = rows m_base+0..7 × K [k_base, +16),
                        // matrix 1 = rows m_base+8..15 × K [k_base, +16),
                        // matrix 2 = rows m_base+0..7 × K [k_base+16, +32),
                        // matrix 3 = rows m_base+8..15 × K [k_base+16, +32).
                        // Addresses from lanes 0..31: row = m_base +
                        // (lane&7) + ((lane>>3)&1)*8, col = k_base +
                        // ((lane>>4)&1)*16.
                        uint32_t a[4];
                        {
                            int a_row = m_base + (lane_id & 7) + ((lane_id >> 3) & 1) * 8;
                            int a_col = k_base + ((lane_id >> 4) & 1) * 16;
                            uint32_t a_addr = smem_u32(&sA[swizzle_smem_offset(a_row, a_col, BK)]);
                            asm volatile(
                                "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                                : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3])
                                : "r"(a_addr));
                        }
                        #pragma unroll
                        for (int ni = 0; ni < P::MMA_N; ni++)
                        {
                            asm volatile(
                                "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
                                "{%0, %1, %2, %3}, "
                                "{%4, %5, %6, %7}, "
                                "{%8, %9}, "
                                "{%10, %11, %12, %13};\n"
                                : "+f"(acc[mi][ni][0]), "+f"(acc[mi][ni][1]),
                                  "+f"(acc[mi][ni][2]), "+f"(acc[mi][ni][3])
                                : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
                                  "r"(b_frag[ni][0]), "r"(b_frag[ni][1]),
                                  "f"(acc[mi][ni][0]), "f"(acc[mi][ni][1]),
                                  "f"(acc[mi][ni][2]), "f"(acc[mi][ni][3]));
                        }
                    }
                }

                if (lane_id == 0)
                {
                    mbarrier_arrive(smem_u32(&smem.empty_barrier[stage]));
                }
                stage++;
                if (stage == NUM_STAGES) { stage = 0; phase ^= 1; }
            }

            // ── Epilogue: scale + narrow + store ──────────────────
            // Lane j's four accumulators in the 16x8 tile at (mi, ni) are:
            //   acc0 = (row j>>2,     col (j&3)*2)
            //   acc1 = (row j>>2,     col (j&3)*2 + 1)
            //   acc2 = (row j>>2 + 8, col (j&3)*2)
            //   acc3 = (row j>>2 + 8, col (j&3)*2 + 1)
            if constexpr (DIRECT_STORE)
            {
                // Direct global store: per (mi, ni), lanes 0..3 cover
                // row r cols 0..7 (16 contiguous bytes), lanes 4..7
                // row r+1, etc. — fully coalesced 16B segments, no
                // smem round-trip, no bar.sync, no TMA store. Frees
                // BM*BN*2 smem for a deeper pipeline.
                #pragma unroll
                for (int mi = 0; mi < P::MMA_M; mi++)
                {
                    const int m_base = m_warp_base + mi * 16;
                    const int out_row0 = bm * BM + m_base + (lane_id >> 2);
                    #pragma unroll
                    for (int ni = 0; ni < P::MMA_N; ni++)
                    {
                        const int n_base = n_warp_base + ni * 8;
                        const int out_col0 = bn * BN + n_base + (lane_id & 3) * 2;
                        const float a_s0 = a_scale[out_row0];
                        const float a_s1 = a_scale[out_row0 + 8];
                        const float b_s0 = b_scale[out_col0];
                        const float b_s1 = b_scale[out_col0 + 1];

                        uint32_t c01 = f32x2_to_bf16x2(
                            acc[mi][ni][0] * a_s0 * b_s0,
                            acc[mi][ni][1] * a_s0 * b_s1);
                        uint32_t c23 = f32x2_to_bf16x2(
                            acc[mi][ni][2] * a_s1 * b_s0,
                            acc[mi][ni][3] * a_s1 * b_s1);
                        *reinterpret_cast<uint32_t *>(&Y[out_row0 * N + out_col0]) = c01;
                        *reinterpret_cast<uint32_t *>(&Y[(out_row0 + 8) * N + out_col0]) = c23;
                    }
                }
            }
            else
            {
                // stmatrix → smem → TMA store (coalesced bulk store,
                // costs BM*BN*2 smem + 2 bar.syncs per tile).
                // stmatrix.x2 stores matrix0 = tile rows 0..7, matrix1 =
                // rows 8..15, both at cols n_base..n_base+7 — thread i's
                // reg0 goes to matrix0 row (i>>2) cols (i&3)*2+{0,1}
                // (matches acc0/acc1) and reg1 to matrix1 row (i>>2)
                // same cols (matches acc2/acc3). Row addresses from
                // lanes 0..15: st_row = m_base + (lane&7) +
                // ((lane>>3)&1)*8.
                if (has_tma_store_in_flight)
                {
                    if (cwg_id == 0 && warp_in_wg == 0 && lane_id == 0)
                    {
                        tma_store_wait();
                    }
                    asm volatile("bar.sync %0, %1;" :: "r"(P::TOTAL_WGS), "r"(CWG * P::THREADS_PER_WG) : "memory");
                }

                bf16 *sY = smem.Y_out;

                // Per-lane scale factors are loaded per (mi, ni) tile —
                // each tile covers different rows/cols of the output.
                // a_scale/b_scale are tiny and L1-cached across the loop.
                #pragma unroll
                for (int mi = 0; mi < P::MMA_M; mi++)
                {
                    const int m_base = m_warp_base + mi * 16;
                    #pragma unroll
                    for (int ni = 0; ni < P::MMA_N; ni++)
                    {
                        const int n_base = n_warp_base + ni * 8;

                        const int out_row0 = bm * BM + m_base + (lane_id >> 2);
                        const int out_col0 = bn * BN + n_base + (lane_id & 3) * 2;
                        const float a_s0 = a_scale[out_row0];
                        const float a_s1 = a_scale[out_row0 + 8];
                        const float b_s0 = b_scale[out_col0];
                        const float b_s1 = b_scale[out_col0 + 1];

                        uint32_t c01 = f32x2_to_bf16x2(
                            acc[mi][ni][0] * a_s0 * b_s0,
                            acc[mi][ni][1] * a_s0 * b_s1);
                        uint32_t c23 = f32x2_to_bf16x2(
                            acc[mi][ni][2] * a_s1 * b_s0,
                            acc[mi][ni][3] * a_s1 * b_s1);
                        int st_row = m_base + (lane_id & 7) + ((lane_id >> 3) & 1) * 8;
                        uint32_t st_addr = smem_u32(&sY[st_row * BN + n_base]);
                        asm volatile(
                            "stmatrix.sync.aligned.m8n8.x2.shared.b16 [%0], {%1, %2};\n"
                            :: "r"(st_addr), "r"(c01), "r"(c23));
                    }
                }

                asm volatile("bar.sync %0, %1;" :: "r"(P::TOTAL_WGS), "r"(CWG * P::THREADS_PER_WG) : "memory");
                if (cwg_id == 0 && warp_in_wg == 0 && lane_id == 0)
                {
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    tma_store_2d(
                        reinterpret_cast<const uint64_t *>(tma_Y.raw),
                        bn * BN, bm * BM,
                        smem_u32(sY));
                    tma_store_arrive();
                }
                has_tma_store_in_flight = true;
            }
        }
        if (cwg_id == 0 && warp_in_wg == 0 && lane_id == 0)
        {
            if (has_tma_store_in_flight) tma_store_wait();
        }
    }

    __syncthreads();
    if (tid == 0)
    {
        for (int s = 0; s < NUM_STAGES; s++)
        {
            mbarrier_inval(&smem.full_barrier[s]);
            mbarrier_inval(&smem.empty_barrier[s]);
        }
    }
}

// ── Launch ───────────────────────────────────────────────────────────
template <int BM, int BN, int BK, int NUM_STAGES, int CWG, int WARP_M, int WARP_N, bool DIRECT_STORE>
void FP8GemmMMA<BM, BN, BK, NUM_STAGES, CWG, WARP_M, WARP_N, DIRECT_STORE>::run(
    int M, int N, int K,
    const fp8e4m3 *__restrict__ A,
    const fp8e4m3 *__restrict__ B,
    const float  *__restrict__ a_scale,
    const float  *__restrict__ b_scale,
    bf16         *__restrict__ Y,
    cudaStream_t  stream)
{
    if (M % BM != 0 || N % BN != 0 || K % BK != 0)
    {
        throw std::runtime_error("M, N, K must be divisible by BM, BN, BK respectively.");
    }

    TMADescriptor tma_A = create_tma_desc_2d<fp8e4m3>(A, K, M, BK, BM, CU_TENSOR_MAP_SWIZZLE_128B);
    TMADescriptor tma_B = create_tma_desc_2d<fp8e4m3>(B, K, N, BK, BN, CU_TENSOR_MAP_SWIZZLE_128B);
    // Y row is BN bf16 = 256 bytes > 128B swizzle limit → no swizzle.
    // (Unused in DIRECT_STORE mode; created unconditionally to keep
    // the kernel signature fixed.)
    TMADescriptor tma_Y = create_tma_desc_2d<bf16>(Y, N, M, BN, BM, CU_TENSOR_MAP_SWIZZLE_NONE);

    int num_tiles_m = M / BM;
    int num_tiles_n = N / BN;
    int total_tiles = num_tiles_m * num_tiles_n;

    int num_sm = 0;
    CHECK_CUDA(cudaDeviceGetAttribute(&num_sm, cudaDevAttrMultiProcessorCount, 0));

    // The >48KB dynamic-smem attribute must be set BEFORE the
    // occupancy query — otherwise the query assumes the 48 KB
    // default cap and returns occ=0 for the 96 KB configs (which
    // then launches grid(0) = invalid argument).
    CHECK_CUDA(cudaFuncSetAttribute(
        fp8_gemm_mma_kernel<BM, BN, BK, NUM_STAGES, CWG, WARP_M, WARP_N, DIRECT_STORE>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_SIZE));

    // Occupancy-aware persistent grid: 2 blocks/SM when smem+regs
    // allow — one block's epilogue overlaps the other's K loop
    // (cuBLAS/nvjet runs 2 blocks/SM for the same reason).
    int occ = 1;
    CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &occ,
        fp8_gemm_mma_kernel<BM, BN, BK, NUM_STAGES, CWG, WARP_M, WARP_N, DIRECT_STORE>,
        TOTAL_THREADS, SMEM_SIZE));
    int num_blocks = min(num_sm * occ, total_tiles);

    dim3 grid(num_blocks);
    dim3 block(TOTAL_THREADS);

    fp8_gemm_mma_kernel<BM, BN, BK, NUM_STAGES, CWG, WARP_M, WARP_N, DIRECT_STORE>
        <<<grid, block, SMEM_SIZE, stream>>>(
            M, N, K, num_tiles_m, num_tiles_n, total_tiles,
            tma_A, tma_B, tma_Y, a_scale, b_scale, Y);
    CHECK_CUDA(cudaGetLastError());
}
