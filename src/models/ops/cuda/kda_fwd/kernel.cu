// KDA forward CUDA kernels (Round 1 — correctness-first baseline).
//
// Pipeline (the Python orchestrator in __init__.py does the reshape +
// cuBLAS bmm between these; the .cu file holds only the per-element
// custom kernels):
//
//   1. Compute g_cum (chunk-local cumsum of per-token log2-decay) and the
//      per-row / per-col decay scales r = exp2(g - g_last), c = exp2(g_last - g).
//   2. Build A_qk = bmm(q * r, (k * c).T) * scale          [B*HV*NT, BT, BT]
//      Build A_kk = bmm(k * r, (k * c).T) * beta[:,:,None] [B*HV*NT, BT, BT]
//      Mask A_kk to strict lower-tri; A = I + A_kk.
//   3. forward_sub_kernel: in-place A → A^{-1}             [B*HV*NT, BT, BT]
//   4. w = bmm(A_inv, k * beta[:,:,None])                  [B*HV*NT, BT, K]
//      u = bmm(A_inv, v)                                    [B*HV*NT, BT, V]
//   5. delta_h_kernel: per (doc, hv, v_slice) sequential   [num_docs, HV, K, V]
//      v_new = (v - w @ h_prev) * exp2(g_last - g[:])
//      h_next = exp2(g_chunk_total) * h_prev + k^T @ v_new
//      Reset h at doc boundaries.
//   6. chunk_o_kernel: per (hv, chunk, v_slice) output     [B, T, HV, V]
//      o = q^T h + A_qk_for_o v   (A_qk_for_o is the lower-tri causal A_qk
//                                   with no beta, scaled; this is the
//                                   local qk contribution)
//
// Conventions:
//   * Inputs are bf16, row-major (PyTorch standard). Fp32 accumulators.
//   * Data layout: per-token arrays are flat over the (num_docs * NT * BT)
//     tokens with `cu_seqlens` giving doc boundaries; per-chunk arrays
//     (w, A_inv, h) are flat over (num_docs * NT) chunks.
//   * g is per-K throughout (matches FLA USE_GK path in chunk_delta_h
//     and chunk_gla_fwd_o_gk). The chunk_o kernel applies per-K decay via
//     q * exp2(g_cum) on the q^Th contribution; delta_h applies per-K
//     decay on h via exp2(g_last_per_k) and stores v_new UN-DECAYED for
//     chunk_o to consume (FLA's convention when SAVE_NEW_VALUE).
//   * HV = H in the production config (expand_v=1.0).

#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <mma.h>

#include <string>

namespace {

constexpr int BT = 64;
constexpr int BC = 16;

inline int ceil_div(int x, int y) { return (x + y - 1) / y; }

}  // namespace

// ===================================================================== //
// Kernel 1: forward substitution for A^{-1}                            //
//                                                                     //
// A is a [BT, BT] lower-triangular matrix with 1 on the diagonal.    //
// We compute A^{-1} in place using the standard forward substitution.//
// One block per chunk; threads cooperate to fill the BT x BT tile.   //
//                                                                     //
// Grid: (num_chunks,)                                                 //
// Block: 256 threads.                                                 //
// ===================================================================== //
//
// Parallelization note: the "obvious" approach (parallel j within a row)
// has a read-after-write race — thread T(j) reads tile[i*BT + k] for
// k in [j, i-1], but thread T(j') for j' > j can write tile[i*BT + j']
// before T(j) reads it. The fix is to SNAPSHOT row i of A into a
// separate scratch array first, then read from the snapshot. The
// scratch array is read-only during the parallel j phase, so there's
// no race. The output (tile[i*BT + j]) is still per-thread, but only
// thread T(j) writes to (i, j), so no write-write race either.
__global__ void forward_sub_kernel(
    float* __restrict__ A_inv,   // [num_chunks, BT, BT] fp32 (in-place)
    int num_chunks
) {
    const int chunk_id = blockIdx.x;
    if (chunk_id >= num_chunks) return;

    __shared__ float tile[BT * BT];
    __shared__ float row_i[BT];  // snapshot of row i's strict-lower-tri

    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;

    float* A_global = A_inv + chunk_id * BT * BT;
    for (int idx = tid; idx < BT * BT; idx += nthreads) {
        tile[idx] = A_global[idx];
    }
    __syncthreads();

    // Compute A^{-1} row by row. For each row i, snapshot the strict-
    // lower-tri entries of row i into row_i[], then parallelize j.
    for (int i = 0; i < BT; ++i) {
        // Snapshot row i's strict-lower-tri (A[i, 0..i-1]) into row_i[].
        for (int k = tid; k < i; k += nthreads) {
            row_i[k] = tile[i * BT + k];
        }
        __syncthreads();

        // Parallel j loop — read from row_i (read-only), no race.
        for (int j = tid; j < BT; j += nthreads) {
            float s = 0.f;
            if (j == i) {
                s = 1.f;
            } else if (j < i) {
                for (int k = j; k < i; ++k) {
                    s -= row_i[k] * tile[k * BT + j];
                }
            }
            tile[i * BT + j] = s;
        }
        __syncthreads();
    }

    for (int idx = tid; idx < BT * BT; idx += nthreads) {
        A_global[idx] = tile[idx];
    }
}

// ===================================================================== //
// Kernel 2: delta_h — cross-chunk h recurrence                         //
//                                                                     //
// Per (doc, hv, v_slice) we sequentially walk NT chunks:              //
//   v_new[i] = u[i] - sum_k w[i, k] * h_prev[k]   (UN-DECAYED)        //
//   h_next[k, vj] = exp2(g_last[k]) * h_prev[k, vj]                   //
//                + sum_i k[i, k] @ v_new[i, vj]   (un-decayed)        //
//                                                                     //
// Matches FLA's USE_GK path in chunk_delta_h.py (USE_G=False for KDA, //
// so the scalar-g branch is skipped entirely; the per-K h decay is    //
// applied at line 227 of chunk_delta_h.py via exp2(b_gk_last1)).      //
//                                                                     //
// v_new is stored UN-DECAYED to v_new_out; chunk_o consumes it via   //
// the A_qk matrix (which has per-K decay + scale baked in).           //
//                                                                     //
// Inputs:                                                             //
//   u = A_inv @ (v * beta)                  (per-token, per-HV, per-V) //
//   w = A_inv @ (k * beta * exp2(g_cum))    (per-chunk, per-HV, per-K) //
//                                                                     //
// Grid: (num_docs, HV, NV) where NV = V / V_TILE.                     //
// Block: 256 threads.                                                 //
// Templated on K, V, V_TILE.                                          //
// ===================================================================== //
template<int K, int V, int V_TILE>
__global__ void __launch_bounds__(256, 2) delta_h_kernel(
    const __nv_bfloat16* __restrict__ k,           // [T_total, H, K]
    const __nv_bfloat16* __restrict__ u,           // [T_total, HV, V]   = A_inv @ (v * beta)
    const __nv_bfloat16* __restrict__ w,           // [N_chunks, HV, BT, K] = A_inv @ (k * beta * exp2(g_cum))
    const __nv_bfloat16* __restrict__ g,           // [T_total, HV, K]  chunk-local cumsum
    const int* __restrict__ doc_chunk_start,       // [num_docs]
    const int* __restrict__ doc_chunk_count,       // [num_docs]
    const int* __restrict__ chunk_token_base,      // [N_chunks]
    __nv_bfloat16* __restrict__ v_new_out,         // [T_total, HV, V]  per-token v_new (UN-DECAYED)
    float* __restrict__ h_per_chunk,               // [N_chunks, HV, K, V]
    float* __restrict__ h_final,                   // [num_docs, HV, K, V]
    int num_chunks,
    int H,
    int HV
) {
    const int doc_id = blockIdx.x;
    const int i_h = blockIdx.y;
    const int i_v = blockIdx.z;
    const int tid = threadIdx.x;
    constexpr int nthreads = 256;

    const int chunk_start_idx = doc_chunk_start[doc_id];
    const int NT = doc_chunk_count[doc_id];

    __shared__ float h_prev[K * V_TILE];

    for (int idx = tid; idx < K * V_TILE; idx += nthreads) {
        h_prev[idx] = 0.f;
    }
    __syncthreads();

    // v_new buffer for the current chunk (UN-DECAYED).
    __shared__ float v_new_smem[BT * V_TILE];
    // Per-K g_last for the current chunk — used for the per-K h decay.
    __shared__ float g_last_per_k[K];

    const int v_start = i_v * V_TILE;

    for (int i_t = 0; i_t < NT; ++i_t) {
        const int chunk_id = chunk_start_idx + i_t;
        if (chunk_id >= num_chunks) break;
        const int token_base = chunk_token_base[chunk_id];

        // ----- store h at START of this chunk (to global) -----
        // No __syncthreads needed: nothing in this iteration reads h_dst.
        // The output is consumed by a downstream kernel (chunk_o) which
        // sees consistent data via the kernel-launch boundary.
        float* h_dst = h_per_chunk + ((chunk_id * HV + i_h) * K * V) + v_start;
        for (int idx = tid; idx < K * V_TILE; idx += nthreads) {
            const int kk = idx / V_TILE;
            const int vj = idx % V_TILE;
            h_dst[kk * V + vj] = h_prev[idx];
        }

        // ----- pointers (correct strides for [T, H/HV, K/V] layout) -----
        // k:   [T, H,  K] -> stride H*K
        // u/g: [T, HV, V/K] -> stride HV*V / HV*K
        const __nv_bfloat16* k_t = k + (token_base * H + i_h) * K;
        const __nv_bfloat16* u_t = u + (token_base * HV + i_h) * V;
        const __nv_bfloat16* g_t = g + (token_base * HV + i_h) * K;
        const __nv_bfloat16* w_c = w + ((chunk_id * HV + i_h) * BT) * K;
        const int k_token_stride = H * K;
        const int v_token_stride = HV * V;
        const int g_token_stride = HV * K;

        // ----- load g_last per-K (matches FLA USE_GK path line 219) -----
        for (int k = tid; k < K; k += nthreads) {
            g_last_per_k[k] = __bfloat162float(
                g[(token_base + BT - 1) * HV * K + i_h * K + k]);
        }
        __syncthreads();

        // ----- step 1: v_new[i, vj] = u[i, vj] - sum_k w[i, k] * h_prev[k, vj] -----
        // u is the A_inv @ (v * beta) tensor, NOT raw v.
        // v_new is UN-DECAYED here (matches FLA chunk_delta_h.py line 191/195).
        for (int idx = tid; idx < BT * V_TILE; idx += nthreads) {
            const int i = idx / V_TILE;
            const int vj = idx % V_TILE;
            float u_val = __bfloat162float(u_t[i * v_token_stride + v_start + vj]);
            float s = 0.f;
            for (int kk = 0; kk < K; ++kk) {
                float w_val = __bfloat162float(w_c[i * K + kk]);
                s += w_val * h_prev[kk * V_TILE + vj];
            }
            v_new_smem[i * V_TILE + vj] = u_val - s;
        }
        __syncthreads();

        // ----- step 2: write UN-DECAYED v_new to v_new_out (chunk_o reads this) -----
        // No __syncthreads needed: step 5 reads v_new_smem (already
        // synced at the end of step 3), not v_new_out. v_new_out is
        // consumed by the next kernel (chunk_o) which has its own
        // kernel-launch boundary for memory consistency.
        {
            __nv_bfloat16* v_new_dst = v_new_out + (token_base * HV + i_h) * V;
            for (int idx = tid; idx < BT * V_TILE; idx += nthreads) {
                const int i = idx / V_TILE;
                const int vj = idx % V_TILE;
                v_new_dst[i * v_token_stride + v_start + vj] =
                    __float2bfloat16(v_new_smem[i * V_TILE + vj]);
            }
        }

        // ----- step 3: h_prev[k, vj] = exp2(g_last[k]) * h_prev[k, vj] -----
        //                      + sum_i k[i, k] * v_new[i, vj] (UN-DECAYED) -----
        // Per-K h decay matches FLA chunk_delta_h.py line 227.
        for (int idx = tid; idx < K * V_TILE; idx += nthreads) {
            const int kk = idx / V_TILE;
            const int vj = idx % V_TILE;
            float s = 0.f;
            for (int i = 0; i < BT; ++i) {
                float k_val = __bfloat162float(k_t[i * k_token_stride + kk]);
                s += k_val * v_new_smem[i * V_TILE + vj];
            }
            h_prev[idx] = exp2f(g_last_per_k[kk]) * h_prev[idx] + s;
        }
        __syncthreads();
    }

    // ----- write final h to h_final -----
    float* h_final_dst = h_final + ((doc_id * HV + i_h) * K * V) + v_start;
    for (int idx = tid; idx < K * V_TILE; idx += nthreads) {
        const int kk = idx / V_TILE;
        const int vj = idx % V_TILE;
        h_final_dst[kk * V + vj] = h_prev[idx];
    }
}

// ===================================================================== //
// Kernel 2b: wmma_delta_h_kernel — tensor-core delta_h                  //
//                                                                     //
// Round-2 opt #4: replace the scalar GEMMs in delta_h_kernel step 1   //
// (v_new = u - w @ h_prev) and step 3 (h_prev = decay*h_prev + k.T @  //
// v_new) with nvcuda::wmma [16, 16, 16] bf16 inputs + fp32 accumulator.//
// Expected ROI: ~10 ms on prod (delta_h 14.8 ms -> ~4 ms).             //
//                                                                     //
// Layout constraints (matches scalar delta_h_kernel for V_TILE=32):    //
//   * 256 threads = 8 warps per block (matches __launch_bounds).       //
//   * Step 1 GEMM: M=BT=64, N=V_TILE=32, K=K=128. 8 output tiles,    //
//     1 per warp (4 M-tiles * 2 N-tiles).                              //
//   * Step 3 GEMM: M=K=128, N=V_TILE=32, K=BT=64. 16 output tiles,    //
//     2 per warp (8 M-tiles * 2 N-tiles).                             //
//   * h_prev[K * V_TILE] in shmem = 16KB.                              //
//   * v_new_smem[BT * V_TILE] in shmem = 8KB.                          //
//   * g_last_per_k[K] = 512B.                                          //
//   * Per-warp acc temp = 1KB * 8 warps = 8KB.                        //
//   Total ~33KB, fits under 48KB Blackwell consumer shmem cap.        //
// ===================================================================== //
template<int K, int V, int V_TILE>
__global__ void __launch_bounds__(256, 2) wmma_delta_h_kernel(
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ u,
    const __nv_bfloat16* __restrict__ w,
    const __nv_bfloat16* __restrict__ g,
    const int* __restrict__ doc_chunk_start,
    const int* __restrict__ doc_chunk_count,
    const int* __restrict__ chunk_token_base,
    __nv_bfloat16* __restrict__ v_new_out,
    float* __restrict__ h_per_chunk,
    float* __restrict__ h_final,
    int num_chunks,
    int H,
    int HV
) {
    using namespace nvcuda;
    constexpr int nthreads = 256;
    constexpr int n_warps = 8;

    const int doc_id = blockIdx.x;
    const int i_h = blockIdx.y;
    const int i_v = blockIdx.z;
    const int tid = threadIdx.x;
    const int wid = tid / 32;
    const int lane = tid % 32;

    const int chunk_start_idx = doc_chunk_start[doc_id];
    const int NT = doc_chunk_count[doc_id];

    __shared__ float h_prev[K * V_TILE];
    __shared__ float v_new_smem[BT * V_TILE];
    __shared__ float g_last_per_k[K];
    // Per-warp reusable scratch: holds bf16→fp32 staging for w/k during K-tile loop,
    // then holds the WMMA accumulator result for element-wise post-processing.
    __shared__ float acc_smem[n_warps * 16 * 16];

    for (int idx = tid; idx < K * V_TILE; idx += nthreads) {
        h_prev[idx] = 0.f;
    }
    __syncthreads();

    const int v_start = i_v * V_TILE;

    for (int i_t = 0; i_t < NT; ++i_t) {
        const int chunk_id = chunk_start_idx + i_t;
        if (chunk_id >= num_chunks) break;
        const int token_base = chunk_token_base[chunk_id];

        // ----- store h at START of this chunk (to global) -----
        float* h_dst = h_per_chunk + ((chunk_id * HV + i_h) * K * V) + v_start;
        for (int idx = tid; idx < K * V_TILE; idx += nthreads) {
            const int kk = idx / V_TILE;
            const int vj = idx % V_TILE;
            h_dst[kk * V + vj] = h_prev[idx];
        }

        // ----- pointers -----
        const __nv_bfloat16* k_t = k + (token_base * H + i_h) * K;
        const __nv_bfloat16* u_t = u + (token_base * HV + i_h) * V;
        const __nv_bfloat16* g_t = g + (token_base * HV + i_h) * K;
        const __nv_bfloat16* w_c = w + ((chunk_id * HV + i_h) * BT) * K;
        const int v_token_stride = HV * V;

        // ----- load g_last per-K -----
        for (int k = tid; k < K; k += nthreads) {
            g_last_per_k[k] = __bfloat162float(
                g[(token_base + BT - 1) * HV * K + i_h * K + k]);
        }
        __syncthreads();

        // ================================================================ //
        // STEP 1: v_new[BT, V_TILE] = u[BT, V_TILE] - w[BT, K] @ h_prev[K, V_TILE]   //
        // TF32 fragments (K_inner=8, fp32 inputs) — preserves full exponent //
        // range so the bf16 quantization of h_prev doesn't lose precision.   //
        // M=BT=64, N=V_TILE=32, K_inner=K=128. 16 K-tiles per warp.          //
        // ================================================================ //
        {
            const int m_tile = wid / 2;
            const int n_tile = wid % 2;
            const int m_start = m_tile * 16;
            const int n_start = n_tile * 16;

            wmma::fragment<wmma::matrix_a, 16, 16, 8, wmma::precision::tf32, wmma::row_major> a_frag;
            wmma::fragment<wmma::matrix_b, 16, 16, 8, wmma::precision::tf32, wmma::row_major> b_frag;
            wmma::fragment<wmma::accumulator, 16, 16, 8, float> acc_frag;
            wmma::fill_fragment(acc_frag, 0.0f);

            #pragma unroll
            for (int k_tile = 0; k_tile < K / 8; ++k_tile) {
                const int k_start = k_tile * 8;
                // Convert w[m_start..m_start+16, k_start..k_start+8] (bf16 -> fp32)
                // into the warp's reusable staging slot. 32 threads × 4 elements each
                // covers the 16*8 = 128 fp32 cells.
                float* w_stage = &acc_smem[wid * 256];
                for (int e = 0; e < 4; ++e) {
                    const int linear = lane * 4 + e;  // 0..127
                    if (linear < 128) {
                        const int rr = linear / 8;
                        const int cc = linear % 8;
                        w_stage[rr * 8 + cc] = __bfloat162float(
                            w_c[(m_start + rr) * K + k_start + cc]);
                    }
                }
                // A = w[m_start..m_start+16, k_start..k_start+8] (row_major, ldm=8).
                wmma::load_matrix_sync(a_frag, w_stage, 8);
                // B = h_prev[k_start..k_start+8, n_start..n_start+16] (row_major, ldm=V_TILE).
                wmma::load_matrix_sync(b_frag, &h_prev[k_start * V_TILE + n_start], V_TILE);
                wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
            }

            wmma::store_matrix_sync(
                &acc_smem[wid * 16 * 16], acc_frag, 16, wmma::mem_row_major);
        }
        __syncthreads();

        // ---- Apply: v_new = u - acc; store to v_new_smem (fp32) ----
        // Each (i, vj) in [0, BT) x [0, V_TILE) reads from the warp slot
        // that owns its tile (m_tile * 2 + n_tile).
        for (int idx = tid; idx < BT * V_TILE; idx += nthreads) {
            const int i = idx / V_TILE;
            const int vj = idx % V_TILE;
            const int m_tile = i / 16;
            const int n_tile = vj / 16;
            const int warp_for_tile = m_tile * 2 + n_tile;
            const int local_i = i % 16;
            const int local_vj = vj % 16;
            float acc_val = acc_smem[warp_for_tile * 256 + local_i * 16 + local_vj];
            float u_val = __bfloat162float(u_t[i * v_token_stride + v_start + vj]);
            v_new_smem[i * V_TILE + vj] = u_val - acc_val;
        }
        __syncthreads();

        // ---- write UN-DECAYED v_new to v_new_out (chunk_o reads this) ----
        {
            __nv_bfloat16* v_new_dst = v_new_out + (token_base * HV + i_h) * V;
            for (int idx = tid; idx < BT * V_TILE; idx += nthreads) {
                const int i = idx / V_TILE;
                const int vj = idx % V_TILE;
                v_new_dst[i * v_token_stride + v_start + vj] =
                    __float2bfloat16(v_new_smem[i * V_TILE + vj]);
            }
        }
        __syncthreads();

        // ================================================================ //
        // STEP 3: h_prev[K, V_TILE] = exp2(g_last[K]) * h_prev[K, V_TILE]   //
        //                          + k[BT, K].T @ v_new[BT, V_TILE]          //
        // TF32 fragments (K_inner=8, fp32 inputs). M=K=128, N=V_TILE=32.   //
        // 8 M-tiles * 2 N-tiles = 16 tiles, 2 per warp.                      //
        // ================================================================ //
        for (int t_iter = 0; t_iter < 2; ++t_iter) {
            const int m_tile = (wid / 2) + t_iter * 4;
            const int n_tile = wid % 2;
            const int m_start = m_tile * 16;
            const int n_start = n_tile * 16;

            wmma::fragment<wmma::matrix_a, 16, 16, 8, wmma::precision::tf32, wmma::col_major> a_frag;
            wmma::fragment<wmma::matrix_b, 16, 16, 8, wmma::precision::tf32, wmma::row_major> b_frag;
            wmma::fragment<wmma::accumulator, 16, 16, 8, float> acc_frag;
            wmma::fill_fragment(acc_frag, 0.0f);

            #pragma unroll
            for (int k_tile = 0; k_tile < BT / 8; ++k_tile) {
                const int k_start = k_tile * 8;
                // Convert k.T[m_start..m_start+16, k_start..k_start+8] (bf16 -> fp32).
                // For col_major matrix_a, A(i, j) = ptr[j * ldm + i].
                // We want A(m, k_inner) = k[k_inner, m_start + m] (k_inner = k_start + j).
                // ptr[j * ldm + i] = k[(k_start + j) * K + (m_start + i)].
                // So ptr = k_t + m_start, ldm = K, j indexes into K_inner (k_inner = k_start + j).
                // For our 16x8 tile: 128 cells, 32 threads × 4 = 128.
                float* k_stage = &acc_smem[wid * 256];
                for (int e = 0; e < 4; ++e) {
                    const int linear = lane * 4 + e;
                    if (linear < 128) {
                        const int rr = linear / 8;
                        const int cc = linear % 8;
                        k_stage[rr * 8 + cc] = __bfloat162float(
                            k_t[(k_start + cc) * K + m_start + rr]);
                    }
                }
                // A = k.T[m_start..m_start+16, k_start..k_start+8] (col_major, ldm=8).
                wmma::load_matrix_sync(a_frag, k_stage, 8);
                // B = v_new[k_start..k_start+8, n_start..n_start+16] (row_major, ldm=V_TILE).
                wmma::load_matrix_sync(b_frag, &v_new_smem[k_start * V_TILE + n_start], V_TILE);
                wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
            }

            wmma::store_matrix_sync(
                &acc_smem[wid * 16 * 16], acc_frag, 16, wmma::mem_row_major);
            __syncthreads();

            // Apply: h_new[k, vj] = exp2(g_last[k]) * h_prev[k, vj] + acc[k, vj].
            for (int idx = tid; idx < 16 * 16; idx += nthreads) {
                const int i = idx / 16;
                const int vj = idx % 16;
                float acc_val = acc_smem[wid * 256 + i * 16 + vj];
                const int kk = m_start + i;
                const int vjj = n_start + vj;
                float old_h = h_prev[kk * V_TILE + vjj];
                float decay = exp2f(g_last_per_k[kk]);
                h_prev[kk * V_TILE + vjj] = decay * old_h + acc_val;
            }
            __syncthreads();
        }
    }

    // ----- write final h to h_final -----
    float* h_final_dst = h_final + ((doc_id * HV + i_h) * K * V) + v_start;
    for (int idx = tid; idx < K * V_TILE; idx += nthreads) {
        const int kk = idx / V_TILE;
        const int vj = idx % V_TILE;
        h_final_dst[kk * V + vj] = h_prev[idx];
    }
}

// ===================================================================== //
// Kernel 3: chunk_o — per-chunk output                                  //
//                                                                     //
// For each chunk:                                                     //
//   o[i, v] = (q[i] @ (exp2(g_cum[i]) * h)) * scale                   //
//            + sum_{j<=i} A_qk[i,j] * v_new[j, v]                     //
//                                                                     //
// A_qk is PRE-COMPUTED in Python (with per-K decay baked in + scale), //
// matching FLA's chunk_intra output. The q^Th contribution uses per-K //
// g decay (matches FLA chunk_gla_fwd_o_gk: b_qg = q * exp2(g)); scale   //
// is applied at the end to the q^Th term only (b_o *= scale in FLA).   //
// The Aqk @ v_new contribution picks up scale via the pre-scaled Aqk.  //
//                                                                     //
// Round 1: hand-rolled matmuls in shmem (no tensor cores).            //
// One block per (hv, chunk, v_slice).                                  //
//                                                                     //
// Grid: (NV, num_chunks, HV).                                         //
// Block: 256 threads.                                                 //
// Templated on K, V, V_TILE.                                          //
// ===================================================================== //
template<int K, int V, int V_TILE>
__global__ void chunk_o_kernel(
    const __nv_bfloat16* __restrict__ q,         // [T_total, H, K]
    const __nv_bfloat16* __restrict__ v_new,     // [T_total, HV, V]  (delta_h output, UN-DECAYED)
    const __nv_bfloat16* __restrict__ g,         // [T_total, HV, K]  (chunk-local cumsum, per-K)
    const __nv_bfloat16* __restrict__ A_qk,      // [N_chunks, HV, BT, BT]  pre-computed, scale baked in
    const float* __restrict__ h,                 // [N_chunks, HV, K, V] (h at start of each chunk)
    __nv_bfloat16* __restrict__ o,               // [T_total, HV, V]
    const int* __restrict__ chunk_token_base,    // [N_chunks]
    int H,
    int HV,
    float scale
) {
    const int i_v = blockIdx.x;
    const int chunk_id = blockIdx.y;
    const int i_h = blockIdx.z;
    const int tid = threadIdx.x;
    constexpr int nthreads = 256;

    const int token_base = chunk_token_base[chunk_id];

    // Per-token strides in elements:
    //   q:    [T, H,  K] -> stride = H*K
    //   v/g:  [T, HV, V/K] -> stride = HV*V / HV*K
    //   o:    [T, HV, V]   -> stride = HV*V
    const int q_token_stride  = H  * K;
    const int v_token_stride  = HV * V;
    const int g_token_stride  = HV * K;

    // Pointers: input arrays have per-token stride H*K (or HV*K / HV*V).
    const __nv_bfloat16* q_t = q + (token_base * H + i_h) * K;
    const __nv_bfloat16* v_t = v_new + (token_base * HV + i_h) * V;
    const __nv_bfloat16* g_t = g + (token_base * HV + i_h) * K;
    const __nv_bfloat16* a_c = A_qk + ((chunk_id * HV + i_h) * BT) * BT;
    const float* h_c = h + ((chunk_id * HV + i_h) * K) * V;
    __nv_bfloat16* o_t = o + (token_base * HV + i_h) * V;

    // Load per-K g into shmem. Shape [BT, K] = BT*K floats = 32KB for BT=64, K=128.
    // g stride per token is HV*K (NOT K) — every BT-th "row" lives in shmem contiguously.
    __shared__ float g_per_k[BT * K];
    for (int idx = tid; idx < BT * K; idx += nthreads) {
        const int i_g = idx / K;
        const int k_g = idx % K;
        g_per_k[idx] = __bfloat162float(g_t[i_g * g_token_stride + k_g]);
    }
    __syncthreads();

    // Load A_qk tile into shmem. Shape [BT, BT] = 4096 fp32 = 16KB.
    // (bypasses global re-reads in the inner o_v loop)
    __shared__ float a_qk_tile[BT * BT];
    {
        for (int idx = tid; idx < BT * BT; idx += nthreads) {
            a_qk_tile[idx] = __bfloat162float(a_c[idx]);
        }
    }
    __syncthreads();

    const int v_start = i_v * V_TILE;
    for (int idx = tid; idx < BT * V_TILE; idx += nthreads) {
        const int i = idx / V_TILE;
        const int vj = idx % V_TILE;

        // o_h[i, vj] = scale * sum_k q[i, k] * exp2(g[i, k]) * h[k, v_start + vj]
        // Per-K decay (matches FLA chunk_gla_fwd_o_gk). Scale ONCE here.
        // q stride per token is H*K.
        float o_h = 0.f;
        for (int kk = 0; kk < K; ++kk) {
            float q_val = __bfloat162float(q_t[i * q_token_stride + kk]);
            float h_val = h_c[kk * V + v_start + vj];
            float g_val = g_per_k[i * K + kk];
            o_h += q_val * exp2f(g_val) * h_val;
        }
        o_h *= scale;

        // o_v[i, vj] = sum_{j<=i} A_qk[i, j] * v_new[j, v_start + vj]
        // A_qk has per-K decay AND scale baked in. No additional scale here.
        // v_new stride per token is HV*V.
        float o_v = 0.f;
        for (int j = 0; j <= i; ++j) {
            float a_qk = a_qk_tile[i * BT + j];
            float v_val = __bfloat162float(v_t[j * v_token_stride + v_start + vj]);
            o_v += a_qk * v_val;
        }

        // Output: q^Th * scale + Aqk @ v_new
        o_t[i * v_token_stride + v_start + vj] =
            __float2bfloat16(o_h + o_v);
    }
}

// ===================================================================== //
// Host-side dispatch                                                   //
// ===================================================================== //

void forward_sub(torch::Tensor A_inv) {
    TORCH_CHECK(A_inv.is_cuda(), "A_inv must be CUDA");
    TORCH_CHECK(A_inv.scalar_type() == at::kFloat, "A_inv must be fp32");
    TORCH_CHECK(A_inv.dim() == 3, "A_inv must be [N, BT, BT]");
    TORCH_CHECK(A_inv.size(1) == BT && A_inv.size(2) == BT, "A_inv last 2 dims must be [BT, BT]");
    int N = A_inv.size(0);
    auto stream = at::cuda::getCurrentCUDAStream();
    forward_sub_kernel<<<N, 256, 0, stream>>>(
        A_inv.data_ptr<float>(), N
    );
}

void delta_h(
    torch::Tensor k, torch::Tensor u, torch::Tensor w, torch::Tensor g,
    torch::Tensor doc_chunk_start, torch::Tensor doc_chunk_count,
    torch::Tensor chunk_token_base,
    torch::Tensor v_new_out,
    torch::Tensor h_per_chunk, torch::Tensor h_final,
    int64_t num_chunks, int64_t H, int64_t HV
) {
    TORCH_CHECK(k.is_cuda() && u.is_cuda() && w.is_cuda() && g.is_cuda(),
                "k/u/w/g must be CUDA");
    TORCH_CHECK(k.scalar_type() == at::kBFloat16, "k must be bf16");
    TORCH_CHECK(u.scalar_type() == at::kBFloat16, "u must be bf16");
    TORCH_CHECK(w.scalar_type() == at::kBFloat16, "w must be bf16");
    TORCH_CHECK(g.scalar_type() == at::kBFloat16, "g must be bf16");
    TORCH_CHECK(v_new_out.scalar_type() == at::kBFloat16, "v_new_out must be bf16");
    TORCH_CHECK(h_per_chunk.scalar_type() == at::kFloat, "h_per_chunk must be fp32");
    TORCH_CHECK(h_final.scalar_type() == at::kFloat, "h_final must be fp32");

    int num_docs = doc_chunk_start.size(0);
    int HV_v = HV;
    int V = u.size(-1);
    int K = k.size(-1);
    // V_TILE=32 keeps static shmem under the 48KB default cap on
    // Blackwell consumer (sm_120). With V_TILE=64 and K=128 the static
    // h_prev[K*V_TILE] alone is 32KB and total ~49KB > 48KB cap.
    int V_TILE = 32;
    TORCH_CHECK(V % V_TILE == 0, "V must be divisible by V_TILE=32");

    dim3 grid(num_docs, HV_v, V / V_TILE);
    auto stream = at::cuda::getCurrentCUDAStream();

    if (K == 128 && V == 128) {
        delta_h_kernel<128, 128, 32><<<grid, 256, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(k.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(u.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(w.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(g.data_ptr()),
            doc_chunk_start.data_ptr<int>(),
            doc_chunk_count.data_ptr<int>(),
            chunk_token_base.data_ptr<int>(),
            reinterpret_cast<__nv_bfloat16*>(v_new_out.data_ptr()),
            h_per_chunk.data_ptr<float>(),
            h_final.data_ptr<float>(),
            (int)num_chunks, (int)H, (int)HV_v
        );
    } else if (K == 64 && V == 64) {
        delta_h_kernel<64, 64, 32><<<grid, 256, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(k.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(u.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(w.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(g.data_ptr()),
            doc_chunk_start.data_ptr<int>(),
            doc_chunk_count.data_ptr<int>(),
            chunk_token_base.data_ptr<int>(),
            reinterpret_cast<__nv_bfloat16*>(v_new_out.data_ptr()),
            h_per_chunk.data_ptr<float>(),
            h_final.data_ptr<float>(),
            (int)num_chunks, (int)H, (int)HV_v
        );
    } else if (K == 32 && V == 32) {
        delta_h_kernel<32, 32, 32><<<grid, 256, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(k.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(u.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(w.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(g.data_ptr()),
            doc_chunk_start.data_ptr<int>(),
            doc_chunk_count.data_ptr<int>(),
            chunk_token_base.data_ptr<int>(),
            reinterpret_cast<__nv_bfloat16*>(v_new_out.data_ptr()),
            h_per_chunk.data_ptr<float>(),
            h_final.data_ptr<float>(),
            (int)num_chunks, (int)H, (int)HV_v
        );
    } else {
        TORCH_CHECK(false, "Unsupported (K, V) combination for delta_h: ",
                    K, ", ", V, " — Round-1 supports only (32,32), (64,64), (128,128)");
    }
}

void wmma_delta_h(
    torch::Tensor k, torch::Tensor u, torch::Tensor w, torch::Tensor g,
    torch::Tensor doc_chunk_start, torch::Tensor doc_chunk_count,
    torch::Tensor chunk_token_base,
    torch::Tensor v_new_out,
    torch::Tensor h_per_chunk, torch::Tensor h_final,
    int64_t num_chunks, int64_t H, int64_t HV
) {
    TORCH_CHECK(k.is_cuda() && u.is_cuda() && w.is_cuda() && g.is_cuda(),
                "k/u/w/g must be CUDA");
    TORCH_CHECK(k.scalar_type() == at::kBFloat16, "k must be bf16");
    TORCH_CHECK(u.scalar_type() == at::kBFloat16, "u must be bf16");
    TORCH_CHECK(w.scalar_type() == at::kBFloat16, "w must be bf16");
    TORCH_CHECK(g.scalar_type() == at::kBFloat16, "g must be bf16");
    TORCH_CHECK(v_new_out.scalar_type() == at::kBFloat16, "v_new_out must be bf16");
    TORCH_CHECK(h_per_chunk.scalar_type() == at::kFloat, "h_per_chunk must be fp32");
    TORCH_CHECK(h_final.scalar_type() == at::kFloat, "h_final must be fp32");

    int num_docs = doc_chunk_start.size(0);
    int HV_v = HV;
    int V = u.size(-1);
    int K = k.size(-1);
    int V_TILE = 32;
    TORCH_CHECK(V % V_TILE == 0, "V must be divisible by V_TILE=32");

    dim3 grid(num_docs, HV_v, V / V_TILE);
    auto stream = at::cuda::getCurrentCUDAStream();

    if (K == 128 && V == 128) {
        wmma_delta_h_kernel<128, 128, 32><<<grid, 256, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(k.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(u.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(w.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(g.data_ptr()),
            doc_chunk_start.data_ptr<int>(),
            doc_chunk_count.data_ptr<int>(),
            chunk_token_base.data_ptr<int>(),
            reinterpret_cast<__nv_bfloat16*>(v_new_out.data_ptr()),
            h_per_chunk.data_ptr<float>(),
            h_final.data_ptr<float>(),
            (int)num_chunks, (int)H, (int)HV_v
        );
    } else if (K == 64 && V == 64) {
        wmma_delta_h_kernel<64, 64, 32><<<grid, 256, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(k.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(u.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(w.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(g.data_ptr()),
            doc_chunk_start.data_ptr<int>(),
            doc_chunk_count.data_ptr<int>(),
            chunk_token_base.data_ptr<int>(),
            reinterpret_cast<__nv_bfloat16*>(v_new_out.data_ptr()),
            h_per_chunk.data_ptr<float>(),
            h_final.data_ptr<float>(),
            (int)num_chunks, (int)H, (int)HV_v
        );
    } else if (K == 32 && V == 32) {
        wmma_delta_h_kernel<32, 32, 32><<<grid, 256, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(k.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(u.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(w.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(g.data_ptr()),
            doc_chunk_start.data_ptr<int>(),
            doc_chunk_count.data_ptr<int>(),
            chunk_token_base.data_ptr<int>(),
            reinterpret_cast<__nv_bfloat16*>(v_new_out.data_ptr()),
            h_per_chunk.data_ptr<float>(),
            h_final.data_ptr<float>(),
            (int)num_chunks, (int)H, (int)HV_v
        );
    } else {
        TORCH_CHECK(false, "Unsupported (K, V) combination for wmma_delta_h: ",
                    K, ", ", V, " — supports only (32,32), (64,64), (128,128)");
    }
}

void chunk_o(
    torch::Tensor q, torch::Tensor v_new, torch::Tensor g,
    torch::Tensor A_qk, torch::Tensor h_per_chunk, torch::Tensor chunk_token_base,
    torch::Tensor o, double scale_d, int64_t H, int64_t HV
) {
    TORCH_CHECK(q.is_cuda() && v_new.is_cuda() && g.is_cuda() && A_qk.is_cuda(),
                "q/v_new/g/A_qk must be CUDA");
    TORCH_CHECK(A_qk.scalar_type() == at::kBFloat16, "A_qk must be bf16 (scale baked in)");
    int num_chunks = chunk_token_base.size(0);
    int HV_v = HV;
    int V = v_new.size(-1);
    int K = q.size(-1);
    // chunk_o doesn't carry the [K, V_TILE] state — V_TILE only controls
    // per-block output tile width. With per-K g now in shmem (BT*K*4 bytes)
    // and A_qk tile in shmem (BT*BT*4 bytes), pick V_TILE that keeps us under
    // 48KB static shmem cap (Blackwell consumer default).
    //   shmem = BT*K*4 + BT*BT*4 + small
    //   K=128: 32KB + 16KB = 48KB. OK, V_TILE=64.
    //   K=64:  16KB + 16KB = 32KB. OK, V_TILE=64.
    //   K=32:   8KB + 16KB = 24KB. OK, V_TILE=32.
    int V_TILE;
    if (K >= 128) {
        V_TILE = 64;
    } else if (K >= 64) {
        V_TILE = 64;
    } else {
        V_TILE = 32;
    }
    TORCH_CHECK(V % V_TILE == 0, "V must be divisible by V_TILE=", V_TILE);

    dim3 grid(V / V_TILE, num_chunks, HV_v);
    auto stream = at::cuda::getCurrentCUDAStream();

    if (K == 128 && V == 128) {
        chunk_o_kernel<128, 128, 64><<<grid, 256, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(q.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(v_new.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(g.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(A_qk.data_ptr()),
            h_per_chunk.data_ptr<float>(),
            reinterpret_cast<__nv_bfloat16*>(o.data_ptr()),
            chunk_token_base.data_ptr<int>(),
            (int)H, (int)HV_v, (float)scale_d
        );
    } else if (K == 64 && V == 64) {
        chunk_o_kernel<64, 64, 64><<<grid, 256, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(q.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(v_new.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(g.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(A_qk.data_ptr()),
            h_per_chunk.data_ptr<float>(),
            reinterpret_cast<__nv_bfloat16*>(o.data_ptr()),
            chunk_token_base.data_ptr<int>(),
            (int)H, (int)HV_v, (float)scale_d
        );
    } else if (K == 32 && V == 32) {
        chunk_o_kernel<32, 32, 32><<<grid, 256, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(q.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(v_new.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(g.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(A_qk.data_ptr()),
            h_per_chunk.data_ptr<float>(),
            reinterpret_cast<__nv_bfloat16*>(o.data_ptr()),
            chunk_token_base.data_ptr<int>(),
            (int)H, (int)HV_v, (float)scale_d
        );
    } else {
        TORCH_CHECK(false, "Unsupported (K, V) for chunk_o: ", K, ", ", V);
    }
}

torch::Tensor kda_fwd(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor g_cum,
    torch::Tensor beta,
    c10::optional<torch::Tensor> cu_seqlens_opt,
    double scale_d,
    c10::optional<torch::Tensor> initial_state_opt,
    bool output_final_state,
    int64_t BT_arg
) {
    TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda(), "All inputs must be CUDA");
    TORCH_CHECK(q.scalar_type() == at::kBFloat16, "q must be bf16 (Round-1)");
    TORCH_CHECK(k.scalar_type() == at::kBFloat16, "k must be bf16");
    TORCH_CHECK(v.scalar_type() == at::kBFloat16, "v must be bf16");
    TORCH_CHECK(g_cum.scalar_type() == at::kBFloat16, "g_cum must be bf16");
    TORCH_CHECK(beta.scalar_type() == at::kBFloat16, "beta must be bf16");
    TORCH_CHECK(BT_arg == 64, "Round-1 path is hard-coded to BT=64");

    // Round-1 stub: return a clone of v. The real implementation will
    // compute o via the 3-kernel pipeline.
    auto out = v.clone();
    return out;
}

std::string kda_fwd_version() {
    return "kda_fwd-r1-correctness-first-2026-06-27";
}
