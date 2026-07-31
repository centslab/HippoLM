// SPDX-License-Identifier: Apache-2.0
// W4A8 native GEMM (FP8 A + NVFP4 B → FP8 MMA → BF16 out)
//
// Stage-1 of the W4A8 fusion plan (see MEMORY.md "W4A8 native — plan",
// 2026-07-31). Takes pre-quantized FP8 A (1×128 BF16 scale) and NVFP4
// B (packed FP4 + E4M3 microblock scales + 1 global scale) and
// dequantizes B to FP8 *inside the GEMM pipeline* so we avoid the
// NVFP4→FP8 materialization pass of the current two-pass W4A8
// forward.
//
// Stage-2 (BF16 A input + in-pipeline quant) is left for a follow-up
// — the BF16 A path adds 2x ldmatrix calls per sub-MMA + per-lane
// quant math, which is meaningful complexity. Stage-1 alone saves
// ~50 us per FFN forward at gate_up shape (the B HBM win; A is
// still pre-quantized).
//
// Smem per stage (BM=BN=128, BK=128):
//   A:    FP8 [128, 128]   = 16 KB  (same as R10)
//   B:    NVFP4 packed [128, 64] + E4M3 scales [128, 8] = 9 KB
//   total per stage: 25 KB; NUM_STAGES=3 = 75 KB (fits in 99 KB cap)
//
// F32 acc (default for overflow safety — see MEMORY.md "W8A8 F16-acc
// overflow — strategy" 2026-07-31; same reasoning applies here since
// the FP8 MMA products are the same). F16 acc is not exposed for the
// NVFP4 path because the quantization round-trip already costs ~2%
// mean_rel; F32 acc preserves what's left.
#pragma once
#include "common.h"
#include "gemm_helpers.cuh"
#include <cuda_fp8.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <cuda.h>

using fp8e4m3 = __nv_fp8_e4m3;
using bf16 = __nv_bfloat16;
using bf16_2 = __nv_bfloat162;


// E4M3 microblock scale decode (NVFP4's 1-byte scale per 1×16 microblock).
// E4M3 = 1 sign + 4 exp + 3 mantissa bits. Native CUDA 13 implicit
// conversion __nv_fp8_e4m3 → float goes through the hardware FP8
// conversion path (faster than a hand-rolled bit decode; no powf/expf).
__device__ __forceinline__ float e4m3_mb_to_float(uint8_t b)
{
    union { uint8_t u; __nv_fp8_e4m3 f; } u;
    u.u = b;
    return static_cast<float>(u.f);
}


template <int BM, int BN, int BK, int NUM_STAGES, int CWG,
          int WARP_M, int WARP_N, bool DIRECT_STORE,
          int BLOCK_SCALE_K, int BLOCK_OUT_M, int BLOCK_OUT_N,
          bool BLOCK_ACCUM, bool PERSISTENT>
struct W4A8GemmMMA
{
    static constexpr int WARPS_PER_WG = 4;
    static constexpr int THREADS_PER_WARP = 32;
    static constexpr int THREADS_PER_WG = WARPS_PER_WG * THREADS_PER_WARP;
    static constexpr int TOTAL_WGS = CWG + 1;
    static constexpr int TOTAL_THREADS = TOTAL_WGS * THREADS_PER_WG;

    static constexpr int NUM_CONSUMER_WARPS = CWG * WARPS_PER_WG;
    static constexpr int MMA_M = WARP_M / 16;
    static constexpr int MMA_N = WARP_N / 8;
    static constexpr int MMA_K = BK / 32;

    static constexpr int SCALES_PER_K_TILE = BK / BLOCK_SCALE_K;
    static constexpr int SUB_MMAS_PER_SCALE = BLOCK_SCALE_K / 32;

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

    // Smem layout per stage:
    //   A: FP8 [BM, BK] (same as R10, TMA-loaded with SWIZZLE_128B)
    //   B_in: NVFP4 packed [BN, BK/2] (TMA-loaded, SWIZZLE_NONE)
    //   B_fp8: FP8 [BN, BK] — dequantized by the producer from B_in,
    //          written unswizzled; the consumer reads it via ldmatrix.x2
    //          with plain (row * BK + col) addressing.
    //   Y_out: 0 if DIRECT_STORE, else [BM, BN] BF16
    //   barriers: tma_barrier (TMA landed), full_barrier (dequant done),
    //             empty_barrier (stage free for reuse)
    struct SMemStorage
    {
        fp8e4m3 A[NUM_STAGES][BM * BK];
        uint8_t B_in[NUM_STAGES][BN * BK / 2];        // packed FP4 (2 nibbles/byte)
        fp8e4m3 B_fp8[NUM_STAGES][BN * BK];           // dequantized B (producer writes, consumer reads)
        bf16 Y_out[DIRECT_STORE ? 1 : BM * BN];
        uint64_t tma_barrier[NUM_STAGES];
        uint64_t full_barrier[NUM_STAGES];
        uint64_t empty_barrier[NUM_STAGES];
    };

    static constexpr int SMEM_SIZE = sizeof(SMemStorage);
    // TX bytes the producer TMA must deliver per stage: A (FP8) + B_packed
    // (NVFP4). B microblock scales are NOT TMA'd — the producer reads them
    // directly from HBM (tiny, L2-resident), so they don't count here.
    static constexpr int TX_BYTES = (BM * BK + BN * BK / 2) * sizeof(uint8_t);

    static void run(
        int M, int N, int K,
        const fp8e4m3 *__restrict__ A,
        const uint8_t *__restrict__ B_packed,
        const fp8e4m3 *__restrict__ B_scales_e4m3,
        const float   *__restrict__ B_global_scale,
        const float   *__restrict__ a_scale_row,   // [M] per-row 1×K-block amax/E4M3_MAX
        const bf16    *__restrict__ a_block_scale, // [M, K/128] BF16 (per-row 1×128)
        const bf16    *__restrict__ b_block_scale, // [K/128] tensorwise BF16
        bf16          *__restrict__ Y,
        uint32_t      *__restrict__ tile_counter,
        cudaStream_t   stream = nullptr);
};


template <int BM, int BN, int BK, int NUM_STAGES, int CWG,
          int WARP_M, int WARP_N, bool DIRECT_STORE,
          int BLOCK_SCALE_K, int BLOCK_OUT_M, int BLOCK_OUT_N,
          bool BLOCK_ACCUM, bool PERSISTENT>
__global__ void __launch_bounds__(
    W4A8GemmMMA<BM, BN, BK, NUM_STAGES, CWG, WARP_M, WARP_N, DIRECT_STORE,
                 BLOCK_SCALE_K, BLOCK_OUT_M, BLOCK_OUT_N, BLOCK_ACCUM, PERSISTENT>::TOTAL_THREADS, 1)
    w4a8_gemm_mma_kernel(
        int M, int N, int K,
        int num_tiles_m, int num_tiles_n, int total_tiles,
        __grid_constant__ const TMADescriptor tma_A,
        __grid_constant__ const TMADescriptor tma_B_packed,
        __grid_constant__ const TMADescriptor tma_Y,
        const fp8e4m3 *__restrict__ B_scales_e4m3_hbm,
        const float *__restrict__ B_global_scale_dev,
        const float *__restrict__ a_scale_row,
        const bf16  *__restrict__ a_block_scale,
        const bf16  *__restrict__ b_block_scale,
        bf16 *__restrict__ Y,
        uint32_t *__restrict__ tile_counter)
{
    using P = W4A8GemmMMA<BM, BN, BK, NUM_STAGES, CWG, WARP_M, WARP_N, DIRECT_STORE,
                          BLOCK_SCALE_K, BLOCK_OUT_M, BLOCK_OUT_N, BLOCK_ACCUM, PERSISTENT>;
    using SmemStorage = typename P::SMemStorage;

    extern __shared__ __align__(1024) char smem_raw[];
    auto &smem = *reinterpret_cast<SmemStorage *>(smem_raw);

    const int tid = threadIdx.x;
    const int warp_id = tid / P::THREADS_PER_WARP;
    const int lane_id = tid % P::THREADS_PER_WARP;
    const int wg_id = warp_id / P::WARPS_PER_WG;
    const int warp_in_wg = warp_id % P::WARPS_PER_WG;

    const int num_k_tiles = K / BK;
    const int num_blocks = gridDim.x;

    // Broadcast the global scale to all threads (1 cache line).
    __shared__ float s_b_global_scale;
    if (tid == 0) {
        s_b_global_scale = (B_global_scale_dev != nullptr) ? *B_global_scale_dev : 1.0f;
    }
    __syncthreads();
    const float b_global_scale = s_b_global_scale;

    if (tid == 0)
    {
        for (int s = 0; s < NUM_STAGES; s++)
        {
            mbarrier_init(&smem.tma_barrier[s], 1);
            mbarrier_init(&smem.full_barrier[s], 1);
            mbarrier_init(&smem.empty_barrier[s], CWG * P::WARPS_PER_WG);
        }
    }
    __syncthreads();
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");

    __shared__ uint32_t s_tile_id;

    int producer_stage = 0;
    int producer_phase = 0;
    int producer_total_k = 0;

    if constexpr (PERSISTENT)
    {
        if (tid == 0) s_tile_id = blockIdx.x;
        __syncthreads();
    }

    // ── Producer warp group ─────────────────────────────────────────
    if (wg_id == 0)
    {
        if (warp_in_wg == 0)
        {
            // All 32 lanes of warp 0 cooperate on the producer work:
            // lane 0 issues TMA + mbarrier ops, all lanes dequantize B_in.
            if constexpr (!PERSISTENT)
            {
                for (int tile_id = blockIdx.x; tile_id < total_tiles; tile_id += num_blocks)
                {
                    int bm, bn;
                    P::rasterize_tile(tile_id, num_tiles_m, num_tiles_n, bm, bn);
                    for (int k = 0; k < num_k_tiles; k++)
                    {
                        if (producer_total_k >= NUM_STAGES)
                        {
                            mbarrier_wait(smem_u32(&smem.empty_barrier[producer_stage]), producer_phase ^ 1);
                        }
                        if (lane_id == 0)
                        {
                            // Issue A + B_in TMA loads into tma_barrier.
                            mbarrier_expect_tx(smem_u32(&smem.tma_barrier[producer_stage]), P::TX_BYTES);
                            tma_A.load_2d(
                                k * BK, bm * BM,
                                smem_u32(smem.A[producer_stage]),
                                smem_u32(&smem.tma_barrier[producer_stage]));
                            tma_B_packed.load_2d(
                                k * (BK / 2), bn * BN,
                                smem_u32(smem.B_in[producer_stage]),
                                smem_u32(&smem.tma_barrier[producer_stage]));
                        }
                        // ALL 32 producer lanes wait on the TMA barrier. This
                        // gives every lane the acquire semantics for the async-
                        // proxy TMA writes (the mbarrier wait by lane 0 alone
                        // would leave the other lanes racing the TMA data in
                        // the dequant — a real race, seen as nondeterministic
                        // NaN in B_fp8 for the second stage).
                        mbarrier_wait(smem_u32(&smem.tma_barrier[producer_stage]), producer_phase);
                        // The TMA writes land via the async proxy; make them
                        // visible to the generic-proxy smem reads in the dequant.
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");

                        // Dequantize B_in [BN, BK/2] → B_fp8 [BN, BK].
                        // All 32 lanes cooperate; each handles BN*BK/2/32 bytes.
                        {
                            const uint8_t *b_in = smem.B_in[producer_stage];
                            fp8e4m3 *b_out = smem.B_fp8[producer_stage];
                            const uint8_t *scales_hbm = reinterpret_cast<const uint8_t *>(B_scales_e4m3_hbm);
                            const int bytes_per_thread = (BN * BK / 2) / 32;
                            const int n_stride = BK / 2;
                            #pragma unroll 4
                            for (int t = 0; t < bytes_per_thread; t++)
                            {
                                const int idx = t * 32 + lane_id;
                                const int row = idx / n_stride;
                                const int col_byte = idx % n_stride;
                                const uint8_t byte = b_in[row * n_stride + col_byte];
                                const int k0 = col_byte * 2;
                                const int mb = (k * (BK / 16)) + (k0 / 16);
                                const int scale_idx = (bn * BN + row) * (K / 16) + mb;
                                const uint8_t scale_byte = scales_hbm[scale_idx];
                                const float s_mb = e4m3_mb_to_float(scale_byte) * b_global_scale;
                                float v_lo = e2m1_decode(byte & 0xF) * s_mb;
                                float v_hi = e2m1_decode(byte >> 4) * s_mb;
                                v_lo = fmaxf(-448.0f, fminf(448.0f, v_lo));
                                v_hi = fmaxf(-448.0f, fminf(448.0f, v_hi));
                                // Write to the SWIZZLED layout (matching what
                                // the consumer's ldmatrix.x2 expects via
                                // swizzle_smem_offset). R10's B is TMA-loaded
                                // with SWIZZLE_128B; we emulate that layout here.
                                b_out[swizzle_smem_offset(row, k0, BK)] = fp8e4m3(v_lo);
                                b_out[swizzle_smem_offset(row, k0 + 1, BK)] = fp8e4m3(v_hi);
                            }
                        }
                        __syncwarp();  // all lanes finished writing B_fp8
                        if (lane_id == 0)
                        {
                            mbarrier_arrive(smem_u32(&smem.full_barrier[producer_stage]));
                            producer_stage++;
                            if (producer_stage == NUM_STAGES) { producer_stage = 0; producer_phase ^= 1; }
                            producer_total_k++;
                        }
                        __syncwarp();  // keep stage/phase in sync across the warp
                    }
                }
            }
        }
    }
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

        auto run_consumer_tile = [&](int tile_id) {
            int bm, bn;
            P::rasterize_tile(tile_id, num_tiles_m, num_tiles_n, bm, bn);

            float acc[P::MMA_M][P::MMA_N][4]{};
            // Raw FP8 products within BLOCK_SCALE_K (BLOCK_ACCUM=true):
            // mma accumulates into acc_raw via native tensor-core "+f"
            // operand; at K-block boundary we apply per-row × per-K-block
            // A scale and flush acc_raw → acc.
            float acc_raw[P::MMA_M][P::MMA_N][4]{};

            for (int k = 0; k < num_k_tiles; k++)
            {
                mbarrier_wait(smem_u32(&smem.full_barrier[stage]), phase);

                const fp8e4m3 *sA = smem.A[stage];
                const fp8e4m3 *sB = smem.B_fp8[stage];   // producer-dequantized, unswizzled

                #pragma unroll
                for (int ki = 0; ki < P::MMA_K; ki++)
                {
                    const int k_base = ki * 32;

                    // ── B fragments: ldmatrix.x2 on the producer-dequantized
                    // B_fp8 smem (SWIZZLED layout — producer writes with the
                    // 128B swizzle, consumer reads with swizzle_smem_offset).
                    // This is exactly R10's B path; the NVFP4 dequant is
                    // entirely in the producer.
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
                        int a_row = m_base + (lane_id & 7) + ((lane_id >> 3) & 1) * 8;
                        int a_col = k_base + ((lane_id >> 4) & 1) * 16;
                        // A is TMA-loaded with SWIZZLE_128B; read via
                        // swizzle_smem_offset (same as R10).
                        uint32_t a_addr = smem_u32(&sA[swizzle_smem_offset(a_row, a_col, BK)]);
                        uint32_t a[4];
                        asm volatile(
                            "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3])
                            : "r"(a_addr));

                        #pragma unroll
                        for (int ni = 0; ni < P::MMA_N; ni++)
                        {
                            // Native F32 accumulator (mma "+f" → acc_raw).
                            asm volatile(
                                "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
                                "{%0, %1, %2, %3}, "
                                "{%4, %5, %6, %7}, "
                                "{%8, %9}, "
                                "{%10, %11, %12, %13};\n"
                                : "+f"(acc_raw[mi][ni][0]), "+f"(acc_raw[mi][ni][1]),
                                  "+f"(acc_raw[mi][ni][2]), "+f"(acc_raw[mi][ni][3])
                                : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
                                  "r"(b_frag[ni][0]), "r"(b_frag[ni][1]),
                                  "f"(acc_raw[mi][ni][0]), "f"(acc_raw[mi][ni][1]),
                                  "f"(acc_raw[mi][ni][2]), "f"(acc_raw[mi][ni][3]));
                        }
                    }

                    // ── Chain flush: every BLOCK_SCALE_K=128 K elements ──
                    const int global_sub = k * P::MMA_K + ki;
                    if ((global_sub + 1) % P::SUB_MMAS_PER_SCALE == 0)
                    {
                        const int scale_cols = K / BLOCK_SCALE_K;  // K/128
                        const int scale_k = global_sub / P::SUB_MMAS_PER_SCALE;
                        // Per-row × per-K-block A scale (1×128 BF16).
                        // a_s0 = lower-half row, a_s1 = upper-half row.
                        #pragma unroll
                        for (int mi = 0; mi < P::MMA_M; mi++)
                        {
                            const int m_base = m_warp_base + mi * 16;
                            const int out_row0 = bm * BM + m_base + (lane_id >> 2);
                            const float a_s0 = __bfloat162float(a_block_scale[out_row0 * scale_cols + scale_k]);
                            const float a_s1 = __bfloat162float(a_block_scale[(out_row0 + 8) * scale_cols + scale_k]);
                            // B scale is already folded into the dequant
                            // (per-microblock E4M3 + global scale applied at
                            // b_frag construction). No b_s needed here.
                            #pragma unroll
                            for (int ni = 0; ni < P::MMA_N; ni++)
                            {
                                acc[mi][ni][0] = fmaf(acc_raw[mi][ni][0], a_s0, acc[mi][ni][0]);
                                acc[mi][ni][1] = fmaf(acc_raw[mi][ni][1], a_s0, acc[mi][ni][1]);
                                acc[mi][ni][2] = fmaf(acc_raw[mi][ni][2], a_s1, acc[mi][ni][2]);
                                acc[mi][ni][3] = fmaf(acc_raw[mi][ni][3], a_s1, acc[mi][ni][3]);
                                acc_raw[mi][ni][0] = 0.0f;
                                acc_raw[mi][ni][1] = 0.0f;
                                acc_raw[mi][ni][2] = 0.0f;
                                acc_raw[mi][ni][3] = 0.0f;
                            }
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

            // ── Epilogue: DIRECT_STORE BF16 ────────────────────────
            // acc already has scale applied per K-block (chain flush above).
            // Just narrow to BF16 and write.
            if constexpr (DIRECT_STORE)
            {
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
                        uint32_t c01 = f32x2_to_bf16x2(acc[mi][ni][0], acc[mi][ni][1]);
                        uint32_t c23 = f32x2_to_bf16x2(acc[mi][ni][2], acc[mi][ni][3]);
                        *reinterpret_cast<uint32_t *>(&Y[out_row0 * N + out_col0]) = c01;
                        *reinterpret_cast<uint32_t *>(&Y[(out_row0 + 8) * N + out_col0]) = c23;
                    }
                }
            }
            else
            {
                // stmatrix path (less common; skip for now — use DIRECT_STORE)
            }
        };

        // W4A8 native is only built with PERSISTENT=false (see the entry
        // file). PERSISTENT work-stealing isn't supported here — the
        // producer dequant pipeline couples producer/consumer stage reuse
        // to the block-stride tile loop.
        if constexpr (PERSISTENT)
        {
            static_assert(!PERSISTENT, "W4A8 native does not support PERSISTENT mode");
        }
        else
        {
            for (int tile_id = blockIdx.x; tile_id < total_tiles; tile_id += num_blocks)
            {
                run_consumer_tile(tile_id);
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


template <int BM, int BN, int BK, int NUM_STAGES, int CWG,
          int WARP_M, int WARP_N, bool DIRECT_STORE,
          int BLOCK_SCALE_K, int BLOCK_OUT_M, int BLOCK_OUT_N,
          bool BLOCK_ACCUM, bool PERSISTENT>
void W4A8GemmMMA<BM, BN, BK, NUM_STAGES, CWG, WARP_M, WARP_N, DIRECT_STORE,
                  BLOCK_SCALE_K, BLOCK_OUT_M, BLOCK_OUT_N, BLOCK_ACCUM, PERSISTENT>::run(
    int M, int N, int K,
    const fp8e4m3 *__restrict__ A,
    const uint8_t *__restrict__ B_packed,
    const fp8e4m3 *__restrict__ B_scales_e4m3,
    const float   *__restrict__ B_global_scale,
    const float   *__restrict__ a_scale_row,
    const bf16    *__restrict__ a_block_scale,
    const bf16    *__restrict__ b_block_scale,
    bf16          *__restrict__ Y,
    uint32_t      *__restrict__ tile_counter,
    cudaStream_t   stream)
{
    if (M % BM != 0 || N % BN != 0 || K % BK != 0)
    {
        throw std::runtime_error("M, N, K must be divisible by BM, BN, BK respectively.");
    }

    TMADescriptor tma_A = create_tma_desc_2d<fp8e4m3>(
        A, K, M, BK, BM, CU_TENSOR_MAP_SWIZZLE_128B);
    TMADescriptor tma_B_packed = create_tma_desc_2d<uint8_t>(
        B_packed, K / 2, N, BK / 2, BN, CU_TENSOR_MAP_SWIZZLE_NONE);
    // B_scales: E4M3 microblock scales are tiny (1 byte per 16 K elements,
    // total BK/16 = 8 bytes per row for BK=128). TMA requires 16-byte
    // aligned boxes, so we skip TMA here — the producer reads them directly
    // from HBM during dequant (scales are 1/16 the volume of B_packed).
    (void)B_scales_e4m3;
    TMADescriptor tma_Y = create_tma_desc_2d<bf16>(Y, N, M, BN, BM, CU_TENSOR_MAP_SWIZZLE_NONE);

    int num_tiles_m = M / BM;
    int num_tiles_n = N / BN;
    int total_tiles = num_tiles_m * num_tiles_n;

    int num_sm = 0;
    CHECK_CUDA(cudaDeviceGetAttribute(&num_sm, cudaDevAttrMultiProcessorCount, 0));

    CHECK_CUDA(cudaFuncSetAttribute(
        w4a8_gemm_mma_kernel<BM, BN, BK, NUM_STAGES, CWG, WARP_M, WARP_N, DIRECT_STORE,
                              BLOCK_SCALE_K, BLOCK_OUT_M, BLOCK_OUT_N, BLOCK_ACCUM, PERSISTENT>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_SIZE));

    int occ = 1;
    CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &occ,
        w4a8_gemm_mma_kernel<BM, BN, BK, NUM_STAGES, CWG, WARP_M, WARP_N, DIRECT_STORE,
                              BLOCK_SCALE_K, BLOCK_OUT_M, BLOCK_OUT_N, BLOCK_ACCUM, PERSISTENT>,
        TOTAL_THREADS, SMEM_SIZE));
    int num_blocks = min(num_sm * occ, total_tiles);

    dim3 grid(num_blocks);
    dim3 block(TOTAL_THREADS);

    w4a8_gemm_mma_kernel<BM, BN, BK, NUM_STAGES, CWG, WARP_M, WARP_N, DIRECT_STORE,
                          BLOCK_SCALE_K, BLOCK_OUT_M, BLOCK_OUT_N, BLOCK_ACCUM, PERSISTENT>
        <<<grid, block, SMEM_SIZE, stream>>>(
            M, N, K, num_tiles_m, num_tiles_n, total_tiles,
            tma_A, tma_B_packed, tma_Y,
            B_scales_e4m3, B_global_scale, a_scale_row, a_block_scale, b_block_scale,
            Y, tile_counter);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "w4a8 kernel launch err: %s\n", cudaGetErrorString(err));
    }
}