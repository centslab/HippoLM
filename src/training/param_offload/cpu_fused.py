"""JIT-compiled fused CPU kernels (DeepSpeed CPUAdam-style).

Owns the OpenMP + AVX kernels that accelerate the per-mb and
per-step CPU work on the ``cpu_add`` offload path:

  * :func:`flush_pending_grads` / :func:`flush_manual_flush_params` —
    per-mb fused BF16 ``tgt += src`` across all per-param
    accumulators in one C++ call. Replaces a Python loop of N
    ``tgt.add_(src)`` calls per microbatch.
  * :meth:`CPUAdamW.step` — fused AdamW step kernel
    (:func:`fused_adam_step_bf16`). In-place ``v`` update
    (BF16→FP32→FMA→BF16 store) plus FP32 ``factor = m /
    (sqrt(v/bc2) + eps)`` in one C++ pass over each param.
    Replaces the Python ``v.mul_(...).addcmul_(...)`` + per-chunk
    FP32 ``promote / sqrt / div`` chain.

  * :func:`zero_cpu_grad_accum` and :meth:`CPUMuon.step` —
    fused ``memset`` across all per-param ``mom_buf`` / ``s.m``
    tensors at end of cycle.

Why one C++ call beats a Python loop:
  PyTorch's per-op internal OMP parallelizes within a single op
  but not across ops. A loop of N ``tgt.add_(slice)`` / ``v.update``
  / ``zero_()`` calls serializes across ops while parallelizing
  within each op — for our shape (17 params × ~80 KiB BF16 at
  4L/256 test scale) the cross-op Python+dispatch overhead alone
  is ~1 ms / layer, the bottleneck for small shapes. The fused
  kernel does all N ops in ONE C++ call with OMP parallel across
  ops.

The same trick DeepSpeed's CPUAdam uses (fused C++ single-pass
with OpenMP ``parallel for`` across cores, plus AVX2/AVX-512
intrinsics inside each op) applies here for the much simpler
zero/add case — both are bandwidth-bound on pinned memory and
benefit identically from multi-core + SIMD parallelism.

JIT-compiled once on first use via
``torch.utils.cpp_extension.load_inline``, cached under
``src/training/param_offload/_fused_ext_build/``. Falls back to
per-op Python loops (the original behavior) if no C++ toolchain
is available at runtime — correctness is preserved either way,
only the speedup is lost.

Set ``HIPPO_FUSED_FORCE_PYLOOP=1`` to force the slow Python-loop
fallback for debugging / A/B benchmarking.

AVX-512 dispatch: the build flags enable both ``-mavx2`` and
``-mavx512f``. At runtime, the C++ layer probes cpuid for AVX-512F
support on the host CPU and dispatches the per-param kernel to
the AVX-512 path when available (16 BF16 per ZMM, 16 FP32 lanes)
or the AVX-2 path otherwise (8 BF16 per YMM, 8 FP32 lanes).
Scalar fallback handles any tail elements and any host without
either ISA. The CPUID probe result is cached.

The zero path follows the same model — same Python-loop
overhead to remove, same multi-core parallelism to unlock.
"""
from __future__ import annotations

import hashlib
import os
import shutil
from typing import Any, List, Optional

import torch


# --------------------------------------------------------------------------- #
# Module state.                                                                #
# --------------------------------------------------------------------------- #
_EXT: Optional[Any] = None
_EXT_FAILED: bool = False


# --------------------------------------------------------------------------- #
# C++ source for the fused kernels.                                             #
# --------------------------------------------------------------------------- #
# Five entry points:
#
#   fused_add_into_many(tgt_ptrs, src_ptrs, sizes)
#     Adds each src into its tgt, element-wise. Sizes are in bytes.
#     AVX-2: 16 BF16 per YMM via ``_mm256_add_epi16`` (2-byte aligned).
#     AVX-512: 32 BF16 per ZMM via ``_mm512_add_epi16`` (2-byte aligned).
#     Scalar fallback for the tail.
#
#   fused_zero_many(ptrs, sizes)
#     Zeros each tensor via ``std::memset``. ``memset(0)`` produces
#     the IEEE +0.0 bit pattern for any floating dtype, so it's
#     dtype-agnostic. glibc's memset already dispatches to AVX/AVX-512
#     stores internally for sizes >= a few hundred bytes, so the
#     per-op BW is competitive with hand-rolled SIMD; the main win
#     here is removing the cross-op Python+dispatch overhead.
#
#   fused_adam_step_bf16(m_ptrs, v_ptrs, factor_ptrs, sizes, steps,
#                        beta2, one_minus_beta2, eps)
#     Per-element AdamW step for one param: v ← β2·v + (1-β2)·m²
#     (BF16 in-place, with FP32 intermediate and round-to-nearest-even
#     FP32→BF16 narrow on the store), and factor[i] ← m[i] /
#     (sqrt(v_new[i]/bc2) + eps) where bc2 = 1 − β2^step.
#     AVX-2: 8 BF16 per YMM per iter → 8 FP32 factor (8 lanes).
#     AVX-512: 16 BF16 per ZMM per iter → 16 FP32 factor (16 lanes).
#     The FP32 → BF16 narrow uses a scalar round-to-nearest-even
#     per lane (8 or 16 per iter) — a 1-ULP-at-BF16-precision error
#     that matches PyTorch's ``.to(torch.bfloat16)``.
#
# All three kernels parallelize across ops via an outer OMP
# ``parallel for`` (the main win for many small ops). The inner
# SIMD loop is per-thread serial (no nested OMP).
_CPP_SRC = r"""
#include <torch/extension.h>
#include <vector>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <cmath>
#include <omp.h>
#ifdef __x86_64__
#include <cpuid.h>
#include <immintrin.h>
#endif

// ===========================================================================
// CPU feature detection (cached).
// ===========================================================================
static inline bool _avx512f_available() {
#ifdef __AVX512F__
    static int cached = -1;
    if (cached < 0) {
        unsigned int a = 0, b = 0, c = 0, d = 0;
        if (__get_cpuid_count(7, 0, &a, &b, &c, &d)) {
            cached = (int)((b >> 16) & 1u);  // EBX bit 16 = AVX-512F
        } else {
            cached = 0;
        }
    }
    return cached != 0;
#else
    return false;
#endif
}

// ===========================================================================
// fused_add_into_many — BF16-only element-wise add across many ops.
//
// BF16-only by construction: the FP32-promote + FP32-add +
// BF16-RNE-narrow inner kernel relies on BF16's bit layout
// being the upper 16 bits of FP32 (sign:1, exp:8, mantissa:7).
// FP16 / FP32 / any other dtype would silently corrupt data
// — both the scalar fallback and the SIMD loop hard-code the
// shift-left-16 + round-to-nearest-even narrow. Microbench
// confirms the discrepancy:
//   bf16: max abs diff = 0.000e+00  (correct)
//   fp16: max abs diff = 1.930e+00  (WRONG)
//   fp32: max abs diff = 2.129e-02  (WRONG)
// The Python wrapper enforces this with a dtype assert —
// important because Muon's configurable ``mom_buf`` storage
// dtype (BF16 / FP16 / FP32) makes a non-BF16 call plausible.
//
// NOTE: as of 2026-07-13 this kernel is NOT wired into the
// per-mb add paths in :mod:`.offload` — microbench on a Zen1
// dev box showed PyTorch's internal ``t.add_(s)`` is already
// 5-50x faster than even the SIMD-narrow version of this
// kernel (the cross-op dispatch overhead we expected to
// amortize is dwarfed by PyTorch's per-op SIMD efficiency).
// The kernel stays exported and bit-correct for future use
// (AVX-512 hosts, fusion opportunities, etc.) but the
// production paths use PyTorch's native ``t.add_(src)``.
// ===========================================================================
// Scalar fallback (also handles the tail in SIMD paths).
// Does proper FP32 promotion + add + RNE narrow — integer add
// on the raw BF16 bit pattern is INCORRECT for different
// exponents (it gives garbage for values like -0.1 + 0.05;
// the result needs FP32 math).
static inline void _add_into_scalar(uint8_t* tgt, const uint8_t* src, int64_t n_bytes) {
    int64_t n_aligned = n_bytes & ~1;
    for (int64_t i = 0; i < n_aligned; i += 2) {
        uint16_t a_bf16, b_bf16;
        std::memcpy(&a_bf16, tgt + i, 2);
        std::memcpy(&b_bf16, src + i, 2);
        // BF16 → FP32 (shift-left-16 of zero-extended uint16).
        uint32_t a_bits = ((uint32_t)a_bf16) << 16;
        uint32_t b_bits = ((uint32_t)b_bf16) << 16;
        float a_fp32, b_fp32;
        std::memcpy(&a_fp32, &a_bits, 4);
        std::memcpy(&b_fp32, &b_bits, 4);
        float sum = a_fp32 + b_fp32;
        // FP32 → BF16 (RNE narrow).
        uint32_t sum_bits;
        std::memcpy(&sum_bits, &sum, 4);
        uint32_t lsb = (sum_bits >> 16) & 1u;
        sum_bits += 0x7FFFu + lsb;
        uint16_t c_bf16 = (uint16_t)(sum_bits >> 16);
        std::memcpy(tgt + i, &c_bf16, 2);
    }
}

#ifdef __AVX2__
// Narrow 8 FP32 lanes → 8 BF16 lanes (RNE), result written to
// the 8 uint16 slots pointed to by ``out``. Used inside the
// SIMD add loop; the scalar RNE per lane is fast enough (8
// stores per inner iter) and avoids the complexity of an
// AVX-2 vector narrow (which would require an interleave
// after the per-lane RNE).
static inline void _narrow_8x_fp32_to_bf16(__m256 fp32, uint16_t* out) {
    alignas(32) float tmp[8];
    _mm256_store_ps(tmp, fp32);
    for (int j = 0; j < 8; j++) {
        uint32_t bits;
        std::memcpy(&bits, &tmp[j], 4);
        uint32_t lsb = (bits >> 16) & 1u;
        bits += 0x7FFFu + lsb;
        out[j] = (uint16_t)(bits >> 16);
    }
}

// AVX-2: 16 BF16 per chunk (32 bytes) — promote to FP32, add,
// narrow back. FP32 promotion is required for correctness
// (``_mm256_add_epi16`` is integer add on the bit pattern and
// only happens to be correct for same-exponent operands).
__attribute__((target("avx2,fma")))
static inline void _add_chunk_avx2(uint8_t* tgt, const uint8_t* src, int64_t n_bytes) {
    int64_t i = 0;
    int64_t n_aligned = n_bytes & ~31;
    for (; i < n_aligned; i += 32) {
        // Load 32 bytes = 16 BF16 per operand.
        __m256i a_i16 = _mm256_loadu_si256((__m256i*)(tgt + i));
        __m256i b_i16 = _mm256_loadu_si256((__m256i*)(src + i));
        // Split into two halves of 8 BF16 each.
        __m128i a_lo = _mm256_castsi256_si128(a_i16);
        __m128i a_hi = _mm256_extracti128_si256(a_i16, 1);
        __m128i b_lo = _mm256_castsi256_si128(b_i16);
        __m128i b_hi = _mm256_extracti128_si256(b_i16, 1);
        // Zero-extend uint16 → uint32, shift-left-16 to make FP32.
        __m256i a_lo_i32 = _mm256_cvtepu16_epi32(a_lo);
        __m256i a_hi_i32 = _mm256_cvtepu16_epi32(a_hi);
        __m256i b_lo_i32 = _mm256_cvtepu16_epi32(b_lo);
        __m256i b_hi_i32 = _mm256_cvtepu16_epi32(b_hi);
        __m256 a_lo_fp = _mm256_castsi256_ps(_mm256_slli_epi32(a_lo_i32, 16));
        __m256 a_hi_fp = _mm256_castsi256_ps(_mm256_slli_epi32(a_hi_i32, 16));
        __m256 b_lo_fp = _mm256_castsi256_ps(_mm256_slli_epi32(b_lo_i32, 16));
        __m256 b_hi_fp = _mm256_castsi256_ps(_mm256_slli_epi32(b_hi_i32, 16));
        // Add in FP32.
        __m256 sum_lo = _mm256_add_ps(a_lo_fp, b_lo_fp);
        __m256 sum_hi = _mm256_add_ps(a_hi_fp, b_hi_fp);
        // Narrow FP32 → BF16 (RNE, per-lane scalar), pack to 32 bytes.
        alignas(32) uint16_t out[16];
        _narrow_8x_fp32_to_bf16(sum_lo, &out[0]);
        _narrow_8x_fp32_to_bf16(sum_hi, &out[8]);
        _mm256_storeu_si256((__m256i*)(tgt + i), _mm256_loadu_si256((__m256i*)out));
    }
    if (i < n_bytes) _add_into_scalar(tgt + i, src + i, n_bytes - i);
}
#endif

#if defined(__AVX512F__) && defined(__AVX512BW__)
// AVX-512F + AVX-512BW: 32 BF16 per chunk (64 bytes) — same
// promote-add-narrow pattern as AVX-2 but with 16 FP32 lanes
// per ZMM. Restricted to AVX-512 via target attribute so the
// host doesn't SIGILL if it lacks AVX-512F; the runtime CPUID
// probe still gates the dispatch.
__attribute__((target("avx512f,avx512bw")))
static inline void _narrow_16x_fp32_to_bf16(__m512 fp32, uint16_t* out) {
    alignas(64) float tmp[16];
    _mm512_store_ps(tmp, fp32);
    for (int j = 0; j < 16; j++) {
        uint32_t bits;
        std::memcpy(&bits, &tmp[j], 4);
        uint32_t lsb = (bits >> 16) & 1u;
        bits += 0x7FFFu + lsb;
        out[j] = (uint16_t)(bits >> 16);
    }
}

__attribute__((target("avx512f,avx512bw")))
static inline void _add_chunk_avx512(uint8_t* tgt, const uint8_t* src, int64_t n_bytes) {
    int64_t i = 0;
    int64_t n_aligned = n_bytes & ~63;
    for (; i < n_aligned; i += 64) {
        // Load 64 bytes = 32 BF16 per operand.
        __m512i a_i16 = _mm512_loadu_si512((__m512i*)(tgt + i));
        __m512i b_i16 = _mm512_loadu_si512((__m512i*)(src + i));
        // Split into two halves of 16 BF16 each.
        __m256i a_lo = _mm512_castsi512_si256(a_i16);
        __m256i a_hi = _mm512_extracti64x4_epi64(a_i16, 1);
        __m256i b_lo = _mm512_castsi512_si256(b_i16);
        __m256i b_hi = _mm512_extracti64x4_epi64(b_i16, 1);
        // Zero-extend uint16 → uint32, shift-left-16 to make FP32.
        __m512i a_lo_i32 = _mm512_cvtepu16_epi32(a_lo);
        __m512i a_hi_i32 = _mm512_cvtepu16_epi32(a_hi);
        __m512i b_lo_i32 = _mm512_cvtepu16_epi32(b_lo);
        __m512i b_hi_i32 = _mm512_cvtepu16_epi32(b_hi);
        __m512 a_lo_fp = _mm512_castsi512_ps(_mm512_slli_epi32(a_lo_i32, 16));
        __m512 a_hi_fp = _mm512_castsi512_ps(_mm512_slli_epi32(a_hi_i32, 16));
        __m512 b_lo_fp = _mm512_castsi512_ps(_mm512_slli_epi32(b_lo_i32, 16));
        __m512 b_hi_fp = _mm512_castsi512_ps(_mm512_slli_epi32(b_hi_i32, 16));
        // Add in FP32.
        __m512 sum_lo = _mm512_add_ps(a_lo_fp, b_lo_fp);
        __m512 sum_hi = _mm512_add_ps(a_hi_fp, b_hi_fp);
        // Narrow FP32 → BF16 (RNE, per-lane scalar), pack to 64 bytes.
        alignas(64) uint16_t out[32];
        _narrow_16x_fp32_to_bf16(sum_lo, &out[0]);
        _narrow_16x_fp32_to_bf16(sum_hi, &out[16]);
        _mm512_storeu_si512((__m512i*)(tgt + i), _mm512_loadu_si512((__m512i*)out));
    }
    #ifdef __AVX2__
    if (i + 32 <= n_bytes) {
        _add_chunk_avx2(tgt + i, src + i, n_bytes - i);
        return;
    }
    #endif
    if (i < n_bytes) _add_into_scalar(tgt + i, src + i, n_bytes - i);
}
#endif

void fused_add_into_many(std::vector<int64_t> tgt_ptrs,
                         std::vector<int64_t> src_ptrs,
                         std::vector<int64_t> sizes) {
    int n = (int)tgt_ptrs.size();
    if (n == 0) return;
    #pragma omp parallel for schedule(dynamic, 1)
    for (int i = 0; i < n; i++) {
        uint8_t* tgt = reinterpret_cast<uint8_t*>(tgt_ptrs[i]);
        const uint8_t* src = reinterpret_cast<const uint8_t*>(src_ptrs[i]);
        int64_t nb = sizes[i];
        bool used_avx512 = false;
    #ifdef __AVX512F__
        if (_avx512f_available() && nb >= 64) {
            _add_chunk_avx512(tgt, src, nb);
            used_avx512 = true;
        }
    #endif
        if (!used_avx512) {
        #ifdef __AVX2__
            if (nb >= 32) _add_chunk_avx2(tgt, src, nb);
            else _add_into_scalar(tgt, src, nb);
        #else
            _add_into_scalar(tgt, src, nb);
        #endif
        }
    }
}

// ===========================================================================
// fused_zero_many — bandwidth-bound memset across many ops.
// ===========================================================================
// One parallel-for across all ops, no inner chunking. Each outer
// thread does one or more whole-op memsets via dynamic scheduling.
// glibc's memset dispatches to AVX/AVX-512 stores internally for
// sizes >= a few hundred bytes, so per-op BW is competitive with
// PyTorch's parallel_for-backed ``zero_()`` while the cross-op
// Python + dispatch overhead (~50 us per op) is eliminated.
void fused_zero_many(std::vector<int64_t> ptrs,
                     std::vector<int64_t> sizes) {
    int n = (int)ptrs.size();
    if (n == 0) return;
    #pragma omp parallel for schedule(dynamic, 1)
    for (int i = 0; i < n; i++) {
        std::memset(reinterpret_cast<uint8_t*>(ptrs[i]), 0, sizes[i]);
    }
}

// ===========================================================================
// fused_adam_step_bf16 — per-element AdamW step (BF16 storage).
// ===========================================================================
// Per element:
//   m_fp32 = bf16_to_fp32(m[i])
//   v_fp32_old = bf16_to_fp32(v[i])
//   v_fp32_new = beta2 * v_fp32_old + (1-beta2) * m_fp32 * m_fp32
//   v[i] = fp32_to_bf16(v_fp32_new)   // round-to-nearest-even
//   bc2 = 1 - pow(beta2, step)         // computed once per param
//   factor[i] = m_fp32 / (sqrt(v_fp32_new / bc2) + eps)
//
// BF16 ↔ FP32 conversion uses the standard "BF16 is the upper 16
// bits of FP32" trick: shift-left-16 to promote (zero-extend the
// uint16 to uint32, then shift into the FP32 position), or
// round-to-nearest-even when narrowing back.

// Scalar fallback (also handles tail).
static inline void _adam_step_scalar(
    const uint16_t* __restrict__ m_bf16,
    uint16_t* __restrict__ v_bf16,
    float* __restrict__ factor_out,
    int64_t n,
    float beta2, float one_minus_beta2,
    float inv_bc2, float eps
) {
    for (int64_t i = 0; i < n; i++) {
        // BF16 → FP32: shift-left-16 of the zero-extended uint16
        // (BF16 has the value in the upper 16 bits of FP32).
        uint32_t m_bits = ((uint32_t)m_bf16[i]) << 16;
        uint32_t v_bits = ((uint32_t)v_bf16[i]) << 16;
        float m_fp32, v_fp32_old;
        std::memcpy(&m_fp32, &m_bits, 4);
        std::memcpy(&v_fp32_old, &v_bits, 4);
        float v_fp32_new = beta2 * v_fp32_old + one_minus_beta2 * m_fp32 * m_fp32;
        // FP32 → BF16: round-to-nearest-even (matches PyTorch).
        uint32_t v_new_bits;
        std::memcpy(&v_new_bits, &v_fp32_new, 4);
        uint32_t lsb = (v_new_bits >> 16) & 1u;
        v_new_bits += 0x7FFFu + lsb;
        v_bf16[i] = (uint16_t)(v_new_bits >> 16);
        float denom = std::sqrt(v_fp32_new * inv_bc2) + eps;
        factor_out[i] = m_fp32 / denom;
    }
}

#ifdef __AVX2__
// AVX-2 AdamW step: 8 BF16 per YMM per iter → 8 FP32 factor.
__attribute__((target("avx2,fma")))
static inline void _adam_step_avx2(
    const uint16_t* __restrict__ m_bf16,
    uint16_t* __restrict__ v_bf16,
    float* __restrict__ factor_out,
    int64_t n,
    float beta2, float one_minus_beta2,
    float inv_bc2, float eps
) {
    const __m256 vec_beta2 = _mm256_set1_ps(beta2);
    const __m256 vec_omb2 = _mm256_set1_ps(one_minus_beta2);
    const __m256 vec_inv_bc2 = _mm256_set1_ps(inv_bc2);
    const __m256 vec_eps = _mm256_set1_ps(eps);
    const int64_t n_aligned = n & ~7;
    int64_t i = 0;
    for (; i < n_aligned; i += 8) {
        __m128i m_i16 = _mm_loadu_si128((__m128i*)(m_bf16 + i));
        __m128i v_i16 = _mm_loadu_si128((__m128i*)(v_bf16 + i));
        __m256i m_i32 = _mm256_cvtepu16_epi32(m_i16);
        __m256i v_i32 = _mm256_cvtepu16_epi32(v_i16);
        __m256 m_fp32 = _mm256_castsi256_ps(_mm256_slli_epi32(m_i32, 16));
        __m256 v_fp32 = _mm256_castsi256_ps(_mm256_slli_epi32(v_i32, 16));
        // v_new = beta2 * v + (1-beta2) * m * m
        __m256 v_new = _mm256_fmadd_ps(
            vec_omb2, _mm256_mul_ps(m_fp32, m_fp32),
            _mm256_mul_ps(vec_beta2, v_fp32)
        );
        // FP32 → BF16 narrow (scalar RNE per lane; 8 lanes / iter).
        alignas(32) float v_new_arr[8];
        _mm256_store_ps(v_new_arr, v_new);
        for (int j = 0; j < 8; j++) {
            uint32_t bits;
            std::memcpy(&bits, &v_new_arr[j], 4);
            uint32_t lsb = (bits >> 16) & 1u;
            bits += 0x7FFFu + lsb;
            v_bf16[i + j] = (uint16_t)(bits >> 16);
        }
        // denom = sqrt(v_new * inv_bc2) + eps; factor = m / denom.
        __m256 denom = _mm256_add_ps(
            _mm256_sqrt_ps(_mm256_mul_ps(v_new, vec_inv_bc2)), vec_eps
        );
        __m256 factor = _mm256_div_ps(m_fp32, denom);
        _mm256_storeu_ps(factor_out + i, factor);
    }
    if (i < n) {
        _adam_step_scalar(
            m_bf16 + i, v_bf16 + i, factor_out + i, n - i,
            beta2, one_minus_beta2, inv_bc2, eps
        );
    }
}
#endif

#if defined(__AVX512F__) && defined(__AVX512BW__)
// AVX-512F AdamW step: 16 BF16 per ZMM per iter → 16 FP32 factor.
__attribute__((target("avx512f,avx512bw")))
static inline void _adam_step_avx512(
    const uint16_t* __restrict__ m_bf16,
    uint16_t* __restrict__ v_bf16,
    float* __restrict__ factor_out,
    int64_t n,
    float beta2, float one_minus_beta2,
    float inv_bc2, float eps
) {
    const __m512 vec_beta2 = _mm512_set1_ps(beta2);
    const __m512 vec_omb2 = _mm512_set1_ps(one_minus_beta2);
    const __m512 vec_inv_bc2 = _mm512_set1_ps(inv_bc2);
    const __m512 vec_eps = _mm512_set1_ps(eps);
    const int64_t n_aligned = n & ~15;
    int64_t i = 0;
    for (; i < n_aligned; i += 16) {
        __m256i m_i16 = _mm256_loadu_si256((__m256i*)(m_bf16 + i));
        __m256i v_i16 = _mm256_loadu_si256((__m256i*)(v_bf16 + i));
        __m512i m_i32 = _mm512_cvtepu16_epi32(m_i16);
        __m512i v_i32 = _mm512_cvtepu16_epi32(v_i16);
        __m512 m_fp32 = _mm512_castsi512_ps(_mm512_slli_epi32(m_i32, 16));
        __m512 v_fp32 = _mm512_castsi512_ps(_mm512_slli_epi32(v_i32, 16));
        __m512 v_new = _mm512_fmadd_ps(
            vec_omb2, _mm512_mul_ps(m_fp32, m_fp32),
            _mm512_mul_ps(vec_beta2, v_fp32)
        );
        // FP32 → BF16 narrow (scalar RNE per lane; 16 lanes / iter).
        alignas(64) float v_new_arr[16];
        _mm512_store_ps(v_new_arr, v_new);
        for (int j = 0; j < 16; j++) {
            uint32_t bits;
            std::memcpy(&bits, &v_new_arr[j], 4);
            uint32_t lsb = (bits >> 16) & 1u;
            bits += 0x7FFFu + lsb;
            v_bf16[i + j] = (uint16_t)(bits >> 16);
        }
        __m512 denom = _mm512_add_ps(
            _mm512_sqrt_ps(_mm512_mul_ps(v_new, vec_inv_bc2)), vec_eps
        );
        __m512 factor = _mm512_div_ps(m_fp32, denom);
        _mm512_storeu_ps(factor_out + i, factor);
    }
    // AVX-2 tail
    #ifdef __AVX2__
    if (i + 8 <= n) {
        _adam_step_avx2(
            m_bf16 + i, v_bf16 + i, factor_out + i, n - i,
            beta2, one_minus_beta2, inv_bc2, eps
        );
        return;
    }
    #endif
    if (i < n) {
        _adam_step_scalar(
            m_bf16 + i, v_bf16 + i, factor_out + i, n - i,
            beta2, one_minus_beta2, inv_bc2, eps
        );
    }
}
#endif

// Public entry — batched over params (one C++ call amortizes
// the cross-op Python overhead across all params).
//
// ``bc2`` is computed in C++ from ``beta2`` and the per-param
// ``step`` count (avoids passing a Python-side list of bc2
// values). ``pow(beta2, step)`` is a single transcendental per
// param per step — negligible vs. the SIMD inner loop.
void fused_adam_step_bf16(
    std::vector<int64_t> m_ptrs,
    std::vector<int64_t> v_ptrs,
    std::vector<int64_t> factor_ptrs,
    std::vector<int64_t> sizes,
    std::vector<int64_t> steps,
    double beta2,
    double one_minus_beta2,
    double eps
) {
    int n = (int)m_ptrs.size();
    if (n == 0) return;
    #pragma omp parallel for schedule(dynamic, 1)
    for (int i = 0; i < n; i++) {
        const uint16_t* m = reinterpret_cast<const uint16_t*>(m_ptrs[i]);
        uint16_t* v = reinterpret_cast<uint16_t*>(v_ptrs[i]);
        float* factor = reinterpret_cast<float*>(factor_ptrs[i]);
        int64_t sz = sizes[i];
        int64_t step = steps[i];
        // bc2 = 1 - beta2^step. Use std::pow (host libm); the
        // single transcendental per param is dwarfed by the
        // SIMD inner loop.
        double bc2_d = 1.0 - std::pow(beta2, (double)step);
        float inv_bc2 = (float)(1.0 / bc2_d);
        bool used_avx512 = false;
    #ifdef __AVX512F__
        if (_avx512f_available()) {
            _adam_step_avx512(
                m, v, factor, sz,
                (float)beta2, (float)one_minus_beta2,
                inv_bc2, (float)eps
            );
            used_avx512 = true;
        }
    #endif
        if (!used_avx512) {
        #ifdef __AVX2__
            _adam_step_avx2(
                m, v, factor, sz,
                (float)beta2, (float)one_minus_beta2,
                inv_bc2, (float)eps
            );
        #else
            _adam_step_scalar(
                m, v, factor, sz,
                (float)beta2, (float)one_minus_beta2,
                inv_bc2, (float)eps
            );
        #endif
        }
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_add_into_many", &fused_add_into_many,
          "Fused multi-target BF16 add with OpenMP + AVX2/AVX-512");
    m.def("fused_zero_many", &fused_zero_many,
          "Fused multi-target zero with OpenMP");
    m.def("fused_adam_step_bf16", &fused_adam_step_bf16,
          "Fused AdamW step (BF16 storage) with OpenMP + AVX2/AVX-512");
}
"""


def _try_load_ext() -> Optional[Any]:
    """JIT-compile the fused kernels once. Cached. Returns None on failure.

    Cache invalidation: hashes the C++ source and stores the hash
    in a sentinel file under the build dir. If the hash doesn't
    match on a subsequent import, the build dir is wiped and the
    extension is recompiled. This is independent of ninja's
    dependency tracking (which load_inline uses internally) — the
    sentinel guarantees correctness even if ninja skips a rebuild
    for whatever reason.
    """
    global _EXT, _EXT_FAILED
    if _EXT is not None:
        return _EXT
    if _EXT_FAILED:
        return None
    # Honor env vars to skip the fused kernel (debug / toolchain
    # issues / A/B benchmarking).
    if os.environ.get("HIPPO_FUSED_FORCE_PYLOOP"):
        _EXT_FAILED = True
        return None
    try:
        from torch.utils.cpp_extension import load_inline
        build_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "_fused_ext_build",
        )
        os.makedirs(build_dir, exist_ok=True)

        # Hash-based cache invalidation: if the C++ source changed
        # since the last build, wipe the build dir before
        # recompiling. Without this, load_inline's internal ninja
        # may load a stale .so that doesn't have the new kernels.
        src_hash = hashlib.sha256(_CPP_SRC.encode()).hexdigest()[:16]
        sentinel = os.path.join(build_dir, "src_hash.txt")
        if os.path.exists(sentinel):
            try:
                with open(sentinel) as f:
                    old_hash = f.read().strip()
            except OSError:
                old_hash = ""
            if old_hash != src_hash:
                for entry in os.listdir(build_dir):
                    p = os.path.join(build_dir, entry)
                    if os.path.isfile(p) or os.path.islink(p):
                        os.unlink(p)
                    else:
                        shutil.rmtree(p, ignore_errors=True)
        try:
            with open(sentinel, "w") as f:
                f.write(src_hash)
        except OSError:
            pass  # Best-effort; recompile still works.

        # -mavx2 + -mfma enable AVX2 + FMA globally (used by the AVX-2
        # paths + the scalar fallback). AVX-512 codegen is
        # restricted to per-function ``__attribute__((target(
        # "avx512f,avx512bw")))`` blocks so that generic code
        # (OMP barrier, memcpy, etc.) doesn't emit AVX-512
        # instructions — that would SIGILL on hosts without
        # AVX-512 support even though the runtime CPUID probe
        # correctly skips the AVX-512 dispatch path.
        _EXT = load_inline(
            name="hippo_fused_cpu",
            cpp_sources=[_CPP_SRC],
            extra_cflags=[
                "-O3", "-mavx2", "-mfma",
                "-fopenmp",
            ],
            extra_ldflags=["-fopenmp"],
            verbose=False,
            build_directory=build_dir,
        )
        return _EXT
    except Exception as e:  # pragma: no cover — toolchain missing
        _EXT_FAILED = True
        return None


# --------------------------------------------------------------------------- #
# Public API.                                                                  #
# --------------------------------------------------------------------------- #
def fused_zero_many(tensors: List[torch.Tensor]) -> bool:
    """Zero a list of CPU tensors using the fused OpenMP kernel.

    Returns True if the fused path was used, False if the
    fallback (per-tensor ``.zero_()``) was used. The fallback
    is only taken if the JIT extension failed to compile (no
    C++ toolchain at runtime) — correctness is preserved either
    way, only the speedup is lost.

    Accepts an empty list (returns True, no-op).
    """
    if not tensors:
        return True
    ext = _try_load_ext()
    if ext is None:
        for t in tensors:
            t.zero_()
        return False
    ptrs = [t.data_ptr() for t in tensors]
    sizes = [t.numel() * t.element_size() for t in tensors]
    ext.fused_zero_many(ptrs, sizes)
    return True


def fused_add_into_many(
    tgt_tensors: List[torch.Tensor],
    src_tensors: List[torch.Tensor],
) -> bool:
    """Add each ``src_tensors[i]`` into ``tgt_tensors[i]`` in place
    using the fused OpenMP kernel.

    BF16-only. The FP32-promote + FP32-add + BF16-RNE-narrow
    inner kernel relies on BF16's bit layout (BF16 = upper 16
    bits of FP32); FP16 / FP32 would silently corrupt data
    (microbench: fp16 max abs diff = 1.93e+00, fp32 = 2.13e-02,
    bf16 = 0.0). The dtype assert below enforces this — important
    because Muon's ``mom_buf`` storage dtype is configurable
    (BF16 / FP16 / FP32) and a non-BF16 call would otherwise pass
    the type system but corrupt training. Use ``t.add_(src)``
    directly for non-BF16 tensors.

    Returns True if the fused path was used, False if the
    fallback was used.
    """
    if not tgt_tensors:
        return True
    assert len(tgt_tensors) == len(src_tensors)
    # BF16-only correctness guard (see docstring). The
    # promote-add-narrow inner kernel hard-codes the BF16
    # bit-layout assumption (sign:1, exp:8, mantissa:7).
    for t in tgt_tensors:
        assert t.dtype == torch.bfloat16, (
            f"fused_add_into_many is BF16-only; got {t.dtype} "
            f"on a tgt tensor. Use t.add_(src) directly for "
            f"non-BF16 tensors."
        )
    for t in src_tensors:
        assert t.dtype == torch.bfloat16, (
            f"fused_add_into_many is BF16-only; got {t.dtype} "
            f"on a src tensor. Use t.add_(src) directly for "
            f"non-BF16 tensors."
        )
    ext = _try_load_ext()
    if ext is None:
        for tgt, src in zip(tgt_tensors, src_tensors):
            tgt.add_(src)
        return False
    tgt_ptrs = [t.data_ptr() for t in tgt_tensors]
    src_ptrs = [t.data_ptr() for t in src_tensors]
    sizes = [t.numel() * t.element_size() for t in tgt_tensors]
    ext.fused_add_into_many(tgt_ptrs, src_ptrs, sizes)
    return True


def fused_adam_step_bf16(
    m_list: List[torch.Tensor],
    v_list: List[torch.Tensor],
    factor_list: List[torch.Tensor],
    step_list: List[int],
    beta2: float,
    eps: float,
) -> bool:
    """Fused AdamW step for a batch of params (BF16 m / v storage).

    Per param:
      v[i] ← β2·v[i] + (1−β2)·m[i]²  (BF16 in-place, FP32 intermediate)
      factor[i] ← m[i] / (√(v_new[i]/bc2) + eps)
    where ``bc2 = 1 − β2^step`` is computed in C++ from the
    per-param ``step_list[i]``.

    ``m_list[i]`` is not modified by the kernel (the cycle-end
    ``m.zero_()`` is batched separately via :func:`fused_zero_many`).

    ``factor_list[i]`` must be a pre-allocated FP32 CPU buffer of
    size ``m_list[i].numel()``. The Python side chunks it for
    H2D + GPU ``add_`` apply.

    Returns True if the fused path was used, False if the
    per-param Python-loop fallback was used (only when the JIT
    extension failed to compile).
    """
    if not m_list:
        return True
    assert len(m_list) == len(v_list) == len(factor_list) == len(step_list)
    ext = _try_load_ext()
    if ext is None:
        one_minus_beta2 = 1.0 - beta2
        for m, v, factor, step in zip(m_list, v_list, factor_list, step_list):
            bc2 = 1.0 - beta2 ** step
            inv_bc2 = 1.0 / bc2
            m_fp32 = m.float()
            v_fp32 = v.float()
            v_new = beta2 * v_fp32 + one_minus_beta2 * m_fp32 * m_fp32
            v.copy_(v_new.to(torch.bfloat16))
            denom = (v_new * inv_bc2).sqrt_().add_(eps)
            factor.copy_((m_fp32 / denom))
        return False
    m_ptrs = [t.data_ptr() for t in m_list]
    v_ptrs = [t.data_ptr() for t in v_list]
    factor_ptrs = [t.data_ptr() for t in factor_list]
    sizes = [t.numel() for t in m_list]
    ext.fused_adam_step_bf16(
        m_ptrs, v_ptrs, factor_ptrs, sizes, step_list,
        float(beta2), float(1.0 - beta2), float(eps),
    )
    return True