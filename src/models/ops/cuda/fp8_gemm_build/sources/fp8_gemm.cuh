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
          int WARP_M = 64, int WARP_N = 64, bool DIRECT_STORE = false,
          bool BLOCKWISE_SCALE = false, bool DUAL_ACCUM = false,
          int BLOCK_SCALE_K = 32, bool BLOCK_TILE_SCALE = false,
          int BLOCK_OUT_M = 64, int BLOCK_OUT_N = 64,
          bool BLOCK_ACCUM = false,
          bool SCALE_FP32 = false,
          bool PERSISTENT = false>
struct FP8GemmMMA
{
    // SCALE_FP32: when true, a_block_scale/b_block_scale are FP32
    // (matched to the QMMA m16n8k32.f32.e4m3.e4m3 accumulator, which
    // is FP32-only on sm_120 — there is no .f16/.bf16 variant). Use
    // FP32 for fine-grained BLOCK_TILE_SCALE (e.g. 32×32 BLOCK_OUT
    // = 32×32×32 = 32K FP8 elements per scale) where BF16's 7-bit
    // mantissa can lose >1% relative precision on outlier magnitudes.
    // For larger blocks (≥64×64 BLOCK_OUT, ≥ 128K elements/scale)
    // BF16 scales are bit-exact convertible to FP32 (the conversion
    // is lossless), so the default stays BF16 to halve the scale HBM
    // bandwidth (2 bytes vs 4 bytes per scale).
    //
    // BLOCK_TILE_SCALE: one scale per (BLOCK_OUT_M rows × BLOCK_OUT_N
    // cols × BLOCK_SCALE_K K). The scale is warp-constant (the compiler
    // hoists the LDG + FMUL out of the (mi, ni) loops) iff each warp's
    // WARP_M×WARP_N output region fits inside a single BLOCK_OUT tile
    // — that is the entire perf win, so require the tile to be a
    // multiple of the warp shape and aligned to the CTA grid.
    static_assert(BLOCK_TILE_SCALE == false || BLOCK_OUT_M >= WARP_M,
                  "BLOCK_TILE_SCALE requires BLOCK_OUT_M >= WARP_M");
    static_assert(BLOCK_TILE_SCALE == false || BLOCK_OUT_N >= WARP_N,
                  "BLOCK_TILE_SCALE requires BLOCK_OUT_N >= WARP_N");
    static_assert(BLOCK_TILE_SCALE == false || BLOCK_OUT_M % WARP_M == 0,
                  "BLOCK_TILE_SCALE requires BLOCK_OUT_M %% WARP_M == 0 (warp fits one tile)");
    static_assert(BLOCK_TILE_SCALE == false || BLOCK_OUT_N % WARP_N == 0,
                  "BLOCK_TILE_SCALE requires BLOCK_OUT_N %% WARP_N == 0 (warp fits one tile)");
    static_assert(BK == 128, "BK must be 128 (128B swizzle span per row)");
    static_assert(!BLOCKWISE_SCALE || (BLOCK_SCALE_K % 32 == 0),
                  "BLOCK_SCALE_K must be a multiple of 32 when BLOCKWISE_SCALE");
    static_assert(!BLOCKWISE_SCALE || BLOCK_ACCUM || BLOCK_SCALE_K <= 128,
                  "BLOCK_SCALE_K > 128 (cross-K-tile) requires BLOCK_ACCUM (native MMA accumulation within the scale-block)");
    static_assert(!BLOCK_ACCUM || BLOCK_TILE_SCALE,
                  "BLOCK_ACCUM currently only implemented with BLOCK_TILE_SCALE (warp-constant scale)");
    static_assert(!SCALE_FP32 || BLOCK_TILE_SCALE,
                  "SCALE_FP32 only meaningful for BLOCK_TILE_SCALE (warp-constant scale path)");
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
        const float  *__restrict__ a_scale,   // [M] FP32 or nullptr
        const float  *__restrict__ b_scale,   // [N] FP32 or nullptr
        // Block scales are passed as `void *` so the same kernel
        // signature can serve both BF16 (default, halve HBM bandwidth)
        // and FP32 (SCALE_FP32=true) layouts. The kernel casts to the
        // appropriate type via `if constexpr (SCALE_FP32)` — see the
        // SCALE_FP32 docstring above for the precision rationale.
        const void   *__restrict__ a_block_scale, // BF16 or FP32, nullptr
        const void   *__restrict__ b_block_scale, // BF16 or FP32, nullptr
        bf16         *__restrict__ Y,
        // PERSISTENT-only: pointer to a uint32_t used as the global
        // atomic tile counter. nullptr when PERSISTENT=false. The host
        // must zero this before launch (entry file uses cudaMemsetAsync).
        uint32_t     *__restrict__ tile_counter,
        cudaStream_t  stream = nullptr);
};

// ── FP8 MMA kernel ───────────────────────────────────────────────────
template <int BM, int BN, int BK, int NUM_STAGES, int CWG, int WARP_M, int WARP_N, bool DIRECT_STORE, bool BLOCKWISE_SCALE, bool DUAL_ACCUM, int BLOCK_SCALE_K, bool BLOCK_TILE_SCALE, int BLOCK_OUT_M, int BLOCK_OUT_N, bool BLOCK_ACCUM, bool SCALE_FP32, bool PERSISTENT>
__global__ void __launch_bounds__(FP8GemmMMA<BM, BN, BK, NUM_STAGES, CWG, WARP_M, WARP_N, DIRECT_STORE, BLOCKWISE_SCALE, DUAL_ACCUM, BLOCK_SCALE_K, BLOCK_TILE_SCALE, BLOCK_OUT_M, BLOCK_OUT_N, BLOCK_ACCUM, SCALE_FP32, PERSISTENT>::TOTAL_THREADS, 1, 1)
    fp8_gemm_mma_kernel(
        int M, int N, int K,
        int num_tiles_m, int num_tiles_n, int total_tiles,
        __grid_constant__ const TMADescriptor tma_A,
        __grid_constant__ const TMADescriptor tma_B,
        __grid_constant__ const TMADescriptor tma_Y,
        const float *__restrict__ a_scale,
        const float *__restrict__ b_scale,
        const void *__restrict__ a_block_scale,
        const void *__restrict__ b_block_scale,
        bf16 *__restrict__ Y,
        // PERSISTENT-only: global atomic counter for work-stealing
        // tile distribution. nullptr when PERSISTENT=false. The host
        // zero-initializes this in run() before launch.
        uint32_t *__restrict__ tile_counter)
{
    using P = FP8GemmMMA<BM, BN, BK, NUM_STAGES, CWG, WARP_M, WARP_N, DIRECT_STORE, BLOCKWISE_SCALE, DUAL_ACCUM, BLOCK_SCALE_K, BLOCK_TILE_SCALE, BLOCK_OUT_M, BLOCK_OUT_N, BLOCK_ACCUM, SCALE_FP32>;
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

    // PERSISTENT: shared atomic counter; producer/consumer share tile_id
    // via shared-memory broadcast (single fetch per CTA per iteration).
    __shared__ uint32_t s_tile_id;

    // Producer pipeline state (NUM_STAGES deep). Defined at the top of
    // the kernel so both the non-persistent producer loop and the
    // unified persistent loop can access them — only the producer
    // thread (wg_id==0, warp_in_wg==0, lane_id==0) actually reads/writes.
    int producer_stage = 0;
    int producer_phase = 0;
    int producer_total_k = 0;

    // Producer TMA lambda: emits NUM_STAGES K-tile TMA loads for one
    // output tile. Captures the pipeline state by reference so calls
    // accumulate across iterations. Defined here (not inside the
    // wg_id==0 branch) so the persistent loop can also call it.
    auto run_producer_tile = [&](int tile_id) {
        int bm, bn;
        P::rasterize_tile(tile_id, num_tiles_m, num_tiles_n, bm, bn);

        for (int k = 0; k < num_k_tiles; k++)
        {
            if (producer_total_k >= NUM_STAGES)
            {
                mbarrier_wait(smem_u32(&smem.empty_barrier[producer_stage]), producer_phase ^ 1);
            }
            mbarrier_expect_tx(smem_u32(&smem.full_barrier[producer_stage]), P::TX_BYTES);

            // A: [M, K] row-major, box [BK, BM] (dim0=K contiguous)
            tma_A.load_2d(
                k * BK, bm * BM,
                smem_u32(smem.A[producer_stage]),
                smem_u32(&smem.full_barrier[producer_stage]));
            // B: [N, K] row-major, box [BK, BN]
            tma_B.load_2d(
                k * BK, bn * BN,
                smem_u32(smem.B[producer_stage]),
                smem_u32(&smem.full_barrier[producer_stage]));

            producer_stage++;
            if (producer_stage == NUM_STAGES) { producer_stage = 0; producer_phase ^= 1; }
            producer_total_k++;
        }
    };

    // Initial fetch for the PERSISTENT path: thread 0 grabs the first
    // tile_id BEFORE either warp group enters its main loop, so the
    // consumer can't race ahead and read uninitialized s_tile_id. The
    // host pre-loads tile_counter with num_blocks (gridDim.x) so each
    // CTA starts at tile_id = blockIdx.x — without that, every CTA
    // would atomicAdd 0,1,2,... and the FIRST iter would have all 36
    // CTAs racing on tile 0.
    if constexpr (PERSISTENT)
    {
        if (tid == 0)
        {
            s_tile_id = blockIdx.x;
        }
        __syncthreads();
    }

    // ── Producer warp group: lane 0 of warp 0 issues TMA loads ────
    if (wg_id == 0)
    {
        if (warp_in_wg == 0 && lane_id == 0)
        {
            if constexpr (PERSISTENT)
            {
                // Persistent path is in the unified loop below — the
                // producer thread participates there.
            }
            else
            {
                for (int tile_id = blockIdx.x; tile_id < total_tiles; tile_id += num_blocks)
                {
                    run_producer_tile(tile_id);
                }
            }
        }
    }
    // ── Consumer warp groups ──────────────────────────────────────
    // Guarded: the producer warp group (wg_id == 0) must NOT enter —
    // its cwg_id would be -1 → negative m_warp_base/n_warp_base →
    // negative smem/global offsets. Benign-ish with the smem epilogue
    // (stmatrix lands inside the CTA's own smem), FATAL with
    // DIRECT_STORE (global store at negative Y offset = illegal
    // access). HEAD production kernel had this as `else`; the
    // blockwise refactor dropped it, which is what crashed R7.
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

        // Helper lambda: run one tile's MMA loop + epilogue for
        // `tile_id`. The producer and persistent consumers both
        // route through this — producer pre-fills smem via TMA, the
        // consumer races against it via mbarriers (per stage).
        auto run_consumer_tile = [&](int tile_id) {
            int bm, bn;
            P::rasterize_tile(tile_id, num_tiles_m, num_tiles_n, bm, bn);

            float acc[P::MMA_M][P::MMA_N][4]{};
            // R6 native-accumulate: raw (unscaled) partials for the
            // current scale-block. The tensor core accumulates into
            // acc_raw across all SUB_MMAS_PER_SCALE sub-MMAs of the block
            // (no intervening FFMA), then one FFMA per (mi, ni) flushes
            // acc_raw into acc at the block boundary and resets it.
            constexpr int RAW_M = BLOCK_ACCUM ? P::MMA_M : 1;
            constexpr int RAW_N = BLOCK_ACCUM ? P::MMA_N : 1;
            float acc_raw[RAW_M][RAW_N][4]{};

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
                            if constexpr (BLOCK_ACCUM)
                            {
                                // R6: chain the sub-MMAs of one scale-block
                                // with the tensor core's NATIVE accumulation
                                // (mma writes acc_raw += A*B via the "+f"
                                // accumulator operand — no FFMA between
                                // consecutive MMAs), then flush once per
                                // block. FFMA count drops from 4 per sub-MMA
                                // to 4 per scale-block = 4·K/BLOCK_SCALE_K
                                // total per (mi, ni).
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
                                const int global_sub = k * P::MMA_K + ki;
                                if ((global_sub + 1) % P::SUB_MMAS_PER_SCALE == 0)
                                {
                                    const int scale_cols = K / BLOCK_SCALE_K;
                                    const int scale_k = global_sub / P::SUB_MMAS_PER_SCALE;
                                    const int m_tile = (bm * BM + m_warp_base) / BLOCK_OUT_M;
                                    const int n_tile = (bn * BN + n_warp_base) / BLOCK_OUT_N;
                                    float a_s, b_s;
                                    if constexpr (SCALE_FP32) {
                                        a_s = reinterpret_cast<const float *>(a_block_scale)[m_tile * scale_cols + scale_k];
                                        b_s = reinterpret_cast<const float *>(b_block_scale)[n_tile * scale_cols + scale_k];
                                    } else {
                                        a_s = __bfloat162float(reinterpret_cast<const bf16 *>(a_block_scale)[m_tile * scale_cols + scale_k]);
                                        b_s = __bfloat162float(reinterpret_cast<const bf16 *>(b_block_scale)[n_tile * scale_cols + scale_k]);
                                    }
                                    const float s = a_s * b_s;
                                    acc[mi][ni][0] = fmaf(acc_raw[mi][ni][0], s, acc[mi][ni][0]);
                                    acc[mi][ni][1] = fmaf(acc_raw[mi][ni][1], s, acc[mi][ni][1]);
                                    acc[mi][ni][2] = fmaf(acc_raw[mi][ni][2], s, acc[mi][ni][2]);
                                    acc[mi][ni][3] = fmaf(acc_raw[mi][ni][3], s, acc[mi][ni][3]);
                                    acc_raw[mi][ni][0] = 0.0f;
                                    acc_raw[mi][ni][1] = 0.0f;
                                    acc_raw[mi][ni][2] = 0.0f;
                                    acc_raw[mi][ni][3] = 0.0f;
                                }
                            }
                            else if constexpr (BLOCKWISE_SCALE && (DUAL_ACCUM || BLOCK_TILE_SCALE))
                            {
                                // mma-into-tmp path used by both R5
                                // (1×N strip, DUAL_ACCUM) and R5b+ (64×64
                                // BLOCK_TILE_SCALE). Both write tmp = A*B
                                // + 0, then 4 fma into the running acc.
                                // The scale indexing differs:
                                //   - DUAL_ACCUM: per-row × per-col
                                //     (lane-dependent out_row0, out_col0)
                                //   - BLOCK_TILE_SCALE: per-(m_tile, n_tile)
                                //     (warp-constant; a_s0==a_s1, b_s0==b_s1
                                //     → 1 unique product per K-block per warp)
                                float tmp[4];
                                const float zero = 0.0f;
                                asm volatile(
                                    "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
                                    "{%0, %1, %2, %3}, "
                                    "{%4, %5, %6, %7}, "
                                    "{%8, %9}, "
                                    "{%10, %11, %12, %13};\n"
                                    : "=f"(tmp[0]), "=f"(tmp[1]),
                                      "=f"(tmp[2]), "=f"(tmp[3])
                                    : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
                                      "r"(b_frag[ni][0]), "r"(b_frag[ni][1]),
                                      "f"(zero), "f"(zero), "f"(zero), "f"(zero));
                                const int scale_cols = K / BLOCK_SCALE_K;
                                const int scale_k = k * P::SCALES_PER_K_TILE + ki / P::SUB_MMAS_PER_SCALE;
                                if constexpr (BLOCK_TILE_SCALE)
                                {
                                    // BLOCK_OUT_M × BLOCK_OUT_N scale: one
                                    // scale per (m_tile, n_tile, k-block).
                                    // BLOCK_OUT_M >= WARP_M and
                                    // BLOCK_OUT_N >= WARP_N (asserted), so the
                                    // whole warp maps to ONE tile → m_tile,
                                    // n_tile are warp-constant across (mi, ni)
                                    // and the compiler hoists the two LDGs +
                                    // FMUL out of the inner loops.
                                    const int m_tile = (bm * BM + m_warp_base) / BLOCK_OUT_M;
                                    const int n_tile = (bn * BN + n_warp_base) / BLOCK_OUT_N;
                                    float a_s, b_s;
                                    if constexpr (SCALE_FP32) {
                                        a_s = reinterpret_cast<const float *>(a_block_scale)[m_tile * scale_cols + scale_k];
                                        b_s = reinterpret_cast<const float *>(b_block_scale)[n_tile * scale_cols + scale_k];
                                    } else {
                                        a_s = __bfloat162float(reinterpret_cast<const bf16 *>(a_block_scale)[m_tile * scale_cols + scale_k]);
                                        b_s = __bfloat162float(reinterpret_cast<const bf16 *>(b_block_scale)[n_tile * scale_cols + scale_k]);
                                    }
                                    const float s = a_s * b_s;
                                    acc[mi][ni][0] = fmaf(tmp[0], s, acc[mi][ni][0]);
                                    acc[mi][ni][1] = fmaf(tmp[1], s, acc[mi][ni][1]);
                                    acc[mi][ni][2] = fmaf(tmp[2], s, acc[mi][ni][2]);
                                    acc[mi][ni][3] = fmaf(tmp[3], s, acc[mi][ni][3]);
                                }
                                else
                                {
                                    // 1×N strip: per-row × per-col.
                                    const int out_row0 = bm * BM + m_base + (lane_id >> 2);
                                    const int n_base = n_warp_base + ni * 8;
                                    const int out_col0 = bn * BN + n_base + (lane_id & 3) * 2;
                                    // 1×N strip path: scales stay BF16
                                    // regardless of SCALE_FP32 (the
                                    // SCALE_FP32 template arg is only
                                    // meaningful for BLOCK_TILE_SCALE,
                                    // asserted above).
                                    const auto a_bs = reinterpret_cast<const bf16 *>(a_block_scale);
                                    const auto b_bs = reinterpret_cast<const bf16 *>(b_block_scale);
                                    const float a_s0 = __bfloat162float(a_bs[out_row0 * scale_cols + scale_k]);
                                    const float a_s1 = __bfloat162float(a_bs[(out_row0 + 8) * scale_cols + scale_k]);
                                    const float b_s0 = __bfloat162float(b_bs[out_col0 * scale_cols + scale_k]);
                                    const float b_s1 = __bfloat162float(b_bs[(out_col0 + 1) * scale_cols + scale_k]);
                                    acc[mi][ni][0] = fmaf(tmp[0], a_s0 * b_s0, acc[mi][ni][0]);
                                    acc[mi][ni][1] = fmaf(tmp[1], a_s0 * b_s1, acc[mi][ni][1]);
                                    acc[mi][ni][2] = fmaf(tmp[2], a_s1 * b_s0, acc[mi][ni][2]);
                                    acc[mi][ni][3] = fmaf(tmp[3], a_s1 * b_s1, acc[mi][ni][3]);
                                }
                            }
                            else
                            {
                                float prev0, prev1, prev2, prev3;
                                if constexpr (BLOCKWISE_SCALE)
                                {
                                    prev0 = acc[mi][ni][0];
                                    prev1 = acc[mi][ni][1];
                                    prev2 = acc[mi][ni][2];
                                    prev3 = acc[mi][ni][3];
                                }
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
                                if constexpr (BLOCKWISE_SCALE)
                                {
                                    const int scale_cols = K / BLOCK_SCALE_K;
                                    const int scale_k = k * P::SCALES_PER_K_TILE + ki / P::SUB_MMAS_PER_SCALE;
                                    const int out_row0 = bm * BM + m_base + (lane_id >> 2);
                                    const int n_base = n_warp_base + ni * 8;
                                    const int out_col0 = bn * BN + n_base + (lane_id & 3) * 2;
                                    const auto a_bs = reinterpret_cast<const bf16 *>(a_block_scale);
                                    const auto b_bs = reinterpret_cast<const bf16 *>(b_block_scale);
                                    const float a_s0 = __bfloat162float(a_bs[out_row0 * scale_cols + scale_k]);
                                    const float a_s1 = __bfloat162float(a_bs[(out_row0 + 8) * scale_cols + scale_k]);
                                    const float b_s0 = __bfloat162float(b_bs[out_col0 * scale_cols + scale_k]);
                                    const float b_s1 = __bfloat162float(b_bs[(out_col0 + 1) * scale_cols + scale_k]);
                                    acc[mi][ni][0] = fmaf(acc[mi][ni][0] - prev0, a_s0 * b_s0, prev0);
                                    acc[mi][ni][1] = fmaf(acc[mi][ni][1] - prev1, a_s0 * b_s1, prev1);
                                    acc[mi][ni][2] = fmaf(acc[mi][ni][2] - prev2, a_s1 * b_s0, prev2);
                                    acc[mi][ni][3] = fmaf(acc[mi][ni][3] - prev3, a_s1 * b_s1, prev3);
                                }
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
                        const float a_s0 = BLOCKWISE_SCALE ? 1.0f : a_scale[out_row0];
                        const float a_s1 = BLOCKWISE_SCALE ? 1.0f : a_scale[out_row0 + 8];
                        const float b_s0 = BLOCKWISE_SCALE ? 1.0f : b_scale[out_col0];
                        const float b_s1 = BLOCKWISE_SCALE ? 1.0f : b_scale[out_col0 + 1];

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
                        const float a_s0 = BLOCKWISE_SCALE ? 1.0f : a_scale[out_row0];
                        const float a_s1 = BLOCKWISE_SCALE ? 1.0f : a_scale[out_row0 + 8];
                        const float b_s0 = BLOCKWISE_SCALE ? 1.0f : b_scale[out_col0];
                        const float b_s1 = BLOCKWISE_SCALE ? 1.0f : b_scale[out_col0 + 1];

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
        };

        if constexpr (PERSISTENT)
        {
            // Unified persistent main loop. Producer (wg_id == 0,
            // warp 0 lane 0) and consumer warps (wg_id >= 1) enter
            // this single while(true) and share its CTA-wide
            // __syncthreads() barriers — that's the only way producer
            // and consumer can synchronize on the same barrier
            // instance (separate code paths reach different barriers).
            //
            // Phase A: producer issues TMA, consumer does MMA + store
            //          (both routed through the same iter boundary so
            //          s_tile_id is read at a consistent point).
            // PHASE A SYNC: producer's TMA + consumer's MMA both done.
            // Producer updates s_tile_id to next tile_id via atomicAdd.
            // PHASE B SYNC: s_tile_id update is visible to all.
            //
            // Exit: when atomicAdd returns >= total_tiles, s_tile_id is
            // set to that value (the sentinel). The consumer reads it
            // at the top of the next iter and breaks; the producer
            // also breaks via the same s_tile_id >= total_tiles check.
            //
            // The non-producer WG-0 threads (other warps in wg 0) have
            // nothing to do — they just spin-wait at both PHASE A and
            // PHASE B barriers, which is required for the CTA-wide
            // __syncthreads to release.
            while (true)
            {
                // PHASE 1: producer does TMA, consumer does MMA.
                int tile_id = (int)s_tile_id;
                if (wg_id == 0)
                {
                    if (warp_in_wg == 0 && lane_id == 0)
                    {
                        if (tile_id < total_tiles)
                        {
                            run_producer_tile(tile_id);
                        }
                    }
                }
                else
                {
                    if (tile_id < total_tiles)
                    {
                        run_consumer_tile(tile_id);
                    }
                }
                __syncthreads();   // PHASE A

                // Producer thread publishes the next tile_id via the
                // global atomic counter. s_tile_id == total_tiles
                // (sentinel) signals exit on the next iter.
                if (wg_id == 0 && warp_in_wg == 0 && lane_id == 0)
                {
                    uint32_t next = atomicAdd(tile_counter, 1);
                    s_tile_id = next;
                }
                __syncthreads();   // PHASE B

                if (s_tile_id >= (uint32_t)total_tiles) break;
            }
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

// ── Launch ───────────────────────────────────────────────────────────
template <int BM, int BN, int BK, int NUM_STAGES, int CWG, int WARP_M, int WARP_N, bool DIRECT_STORE, bool BLOCKWISE_SCALE, bool DUAL_ACCUM, int BLOCK_SCALE_K, bool BLOCK_TILE_SCALE, int BLOCK_OUT_M, int BLOCK_OUT_N, bool BLOCK_ACCUM, bool SCALE_FP32, bool PERSISTENT>
void FP8GemmMMA<BM, BN, BK, NUM_STAGES, CWG, WARP_M, WARP_N, DIRECT_STORE, BLOCKWISE_SCALE, DUAL_ACCUM, BLOCK_SCALE_K, BLOCK_TILE_SCALE, BLOCK_OUT_M, BLOCK_OUT_N, BLOCK_ACCUM, SCALE_FP32, PERSISTENT>::run(
    int M, int N, int K,
    const fp8e4m3 *__restrict__ A,
    const fp8e4m3 *__restrict__ B,
    const float  *__restrict__ a_scale,
    const float  *__restrict__ b_scale,
    const void   *__restrict__ a_block_scale,
    const void   *__restrict__ b_block_scale,
    bf16         *__restrict__ Y,
    // PERSISTENT-only: pointer to a uint32_t used as the global
    // atomic tile counter. The host zeros this before launch. Must
    // be nullptr when PERSISTENT=false.
    uint32_t     *__restrict__ tile_counter,
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
        fp8_gemm_mma_kernel<BM, BN, BK, NUM_STAGES, CWG, WARP_M, WARP_N, DIRECT_STORE, BLOCKWISE_SCALE, DUAL_ACCUM, BLOCK_SCALE_K, BLOCK_TILE_SCALE, BLOCK_OUT_M, BLOCK_OUT_N, BLOCK_ACCUM, SCALE_FP32, PERSISTENT>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_SIZE));

    int num_blocks;
    if constexpr (PERSISTENT)
    {
        // PERSISTENT: launch exactly 1 CTA per SM (no occ > 1
        // overlap), each CTA work-steals tiles via the atomic
        // counter. We pre-seed tile_counter with num_blocks so the
        // FIRST atomicAdd returns num_blocks (not 0) — that way each
        // CTA's first atomicAdd returns a tile > its blockIdx.x, so
        // the post-init s_tile_id (set by the kernel to blockIdx.x)
        // doesn't collide with what other CTAs atomicAdd next.
        num_blocks = min(num_sm, total_tiles);
        CHECK_CUDA(cudaMemsetAsync(tile_counter, 0, sizeof(uint32_t), stream));
        uint32_t seed = (uint32_t)num_blocks;
        CHECK_CUDA(cudaMemcpyAsync(tile_counter, &seed, sizeof(uint32_t),
                                   cudaMemcpyHostToDevice, stream));
    }
    else
    {
        // Occupancy-aware persistent grid: 2 blocks/SM when smem+regs
        // allow — one block's epilogue overlaps the other's K loop
        // (cuBLAS/nvjet runs 2 blocks/SM for the same reason).
        int occ = 1;
        CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &occ,
            fp8_gemm_mma_kernel<BM, BN, BK, NUM_STAGES, CWG, WARP_M, WARP_N, DIRECT_STORE, BLOCKWISE_SCALE, DUAL_ACCUM, BLOCK_SCALE_K, BLOCK_TILE_SCALE, BLOCK_OUT_M, BLOCK_OUT_N, BLOCK_ACCUM, SCALE_FP32, PERSISTENT>,
            TOTAL_THREADS, SMEM_SIZE));
        num_blocks = min(num_sm * occ, total_tiles);
    }

    dim3 grid(num_blocks);
    dim3 block(TOTAL_THREADS);

    fp8_gemm_mma_kernel<BM, BN, BK, NUM_STAGES, CWG, WARP_M, WARP_N, DIRECT_STORE, BLOCKWISE_SCALE, DUAL_ACCUM, BLOCK_SCALE_K, BLOCK_TILE_SCALE, BLOCK_OUT_M, BLOCK_OUT_N, BLOCK_ACCUM, SCALE_FP32, PERSISTENT>
        <<<grid, block, SMEM_SIZE, stream>>>(
            M, N, K, num_tiles_m, num_tiles_n, total_tiles,
            tma_A, tma_B, tma_Y, a_scale, b_scale,
            a_block_scale, b_block_scale, Y, tile_counter);
    CHECK_CUDA(cudaGetLastError());
}
