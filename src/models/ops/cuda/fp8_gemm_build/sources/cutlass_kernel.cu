// SPDX-License-Identifier: Apache-2.0
// Project-local CUTLASS-based FP8 GEMM with blockwise scaling for sm_120.
//
// Mirrors /hy-tmp/cutlass/examples/87_blackwell_geforce_gemm_blockwise/87a_*.
// The 87a kernel uses per-MmaTile (128x128x128) FP32 scales, not per-row
// or per-1x128-block scales. The scale tensor is laid out as
// (M_blocks, K_blocks) for SFA and (N_blocks, K_blocks) for SFB, with
// M-major / N-major contiguous order in memory.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include "cutlass/tensor_ref.h"
#include "cutlass/epilogue/thread/activation.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/dispatch_policy.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/packed_stride.hpp"

namespace cg = cutlass::gemm::collective;

using namespace cute;

using ElementA = cutlass::float_e4m3_t;
using ElementB = cutlass::float_e4m3_t;
using ElementC = cutlass::bfloat16_t;
using ElementD = cutlass::bfloat16_t;
using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::RowMajor;
using ElementAccumulator = float;
using ElementCompute     = float;

using MmaTileShape_MNK = Shape<_128, _128, _128>;
using ClusterShape_MNK = Shape<_1, _1, _1>;

using ScaleConfig = decltype(
    cutlass::detail::sm120_trivial_blockwise_scale_config(MmaTileShape_MNK{}));
using LayoutSFA = decltype(ScaleConfig::deduce_layoutSFA());
using LayoutSFB = decltype(ScaleConfig::deduce_layoutSFB());

static constexpr int AlignmentA = 16;
static constexpr int AlignmentB = 16;
static constexpr int AlignmentC = 16;

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm120, cutlass::arch::OpClassTensorOp,
    MmaTileShape_MNK, ClusterShape_MNK,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementCompute,
    ElementC, LayoutC, AlignmentC,
    ElementD, LayoutC, AlignmentC,
    cutlass::epilogue::collective::EpilogueScheduleAuto
>::CollectiveOp;

using CollectiveMainloop = typename cg::CollectiveBuilder<
    cutlass::arch::Sm120, cutlass::arch::OpClassTensorOp,
    ElementA, cute::tuple<LayoutA, LayoutSFA>, AlignmentA,
    ElementB, cute::tuple<LayoutB, LayoutSFB>, AlignmentB,
    ElementAccumulator,
    MmaTileShape_MNK, ClusterShape_MNK,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::collective::KernelScheduleAuto
>::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>,
    CollectiveMainloop,
    CollectiveEpilogue,
    void>;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

using StrideA = typename Gemm::GemmKernel::StrideA;
using StrideB = typename Gemm::GemmKernel::StrideB;
using StrideC = typename Gemm::GemmKernel::StrideC;

struct GemmInstance {
    Gemm gemm;
    void* workspace = nullptr;
    size_t workspace_size = 0;
};

static GemmInstance& get_gemm() {
    static GemmInstance inst;
    return inst;
}

// ---- Public ABI: per-MmaTile (1x1 MmaTile) FP32 scales ----
//
// Inputs:
//   A  : [M, K] FP8 E4M3 row-major
//   B  : [K, N] FP8 E4M3 row-major (CUTLASS TN layout; kernel sees
//        B(n, k) = data[k*N + n] in memory)
//   SFA: [M/128, K/128] FP32 row-major (1x1-MmaTile per-row scale)
//   SFB: [N/128, K/128] FP32 row-major (1x1-MmaTile per-col scale)
//   D  : [M, N] BF16 row-major
//
// Computes: D[m, n] = sum_k A[m, k] * SFA[m/128, k/128] * B[k, n] * SFB[n/128, k/128]

extern "C" int cutlass_fp8_blockwise_gemm_run(
    int M, int N, int K,
    const void* A, const void* B,
    const void* SFA, const void* SFB,
    void* D,
    cudaStream_t stream)
{
    if (M % 128 != 0 || N % 128 != 0 || K % 128 != 0) {
        fprintf(stderr, "cutlass_fp8_blockwise: M=%d N=%d K=%d not all divisible by 128\n",
                M, N, K);
        return -1;
    }

    auto& inst = get_gemm();

    StrideA stride_A = cutlass::make_cute_packed_stride(
        StrideA{}, cute::make_shape(M, K, 1));
    StrideB stride_B = cutlass::make_cute_packed_stride(
        StrideB{}, cute::make_shape(N, K, 1));
    StrideC stride_C = cutlass::make_cute_packed_stride(
        StrideC{}, cute::make_shape(M, N, 1));

    LayoutSFA layout_SFA = ScaleConfig::tile_atom_to_shape_SFA(
        make_shape(M, N, K, 1));
    LayoutSFB layout_SFB = ScaleConfig::tile_atom_to_shape_SFB(
        make_shape(M, N, K, 1));

    typename Gemm::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {M, N, K, 1},
        {reinterpret_cast<const ElementA*>(A), stride_A,
         reinterpret_cast<const ElementB*>(B), stride_B,
         reinterpret_cast<const ElementAccumulator*>(SFA), layout_SFA,
         reinterpret_cast<const ElementAccumulator*>(SFB), layout_SFB},
        {{},
         reinterpret_cast<ElementC*>(D), stride_C,
         reinterpret_cast<ElementD*>(D), stride_C}
    };

    cutlass::Status status = inst.gemm.can_implement(args);
    if (status != cutlass::Status::kSuccess) {
        fprintf(stderr, "cutlass_fp8_blockwise: can_implement failed (%d)\n", (int)status);
        return -2;
    }

    size_t needed = Gemm::get_workspace_size(args);
    if (needed > inst.workspace_size) {
        if (inst.workspace) {
            cudaFree(inst.workspace);
            inst.workspace = nullptr;
            inst.workspace_size = 0;
        }
        cudaError_t err = cudaMalloc(&inst.workspace, needed);
        if (err != cudaSuccess) {
            fprintf(stderr, "cutlass_fp8_blockwise: workspace alloc failed: %s\n",
                    cudaGetErrorString(err));
            return -3;
        }
        inst.workspace_size = needed;
    }

    status = inst.gemm.initialize(args, inst.workspace, stream);
    if (status != cutlass::Status::kSuccess) {
        fprintf(stderr, "cutlass_fp8_blockwise: initialize failed (%d)\n", (int)status);
        return -4;
    }

    status = inst.gemm.run(stream);
    if (status != cutlass::Status::kSuccess) {
        fprintf(stderr, "cutlass_fp8_blockwise: run failed (status=%d)\n", (int)status);
        return -5;
    }

    return 0;
}
