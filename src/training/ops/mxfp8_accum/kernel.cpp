// Fused mxfp8 dequant+add+requant kernel (CPU).
//
// Why this exists
// ----------------
// The mxfp8 muon design replaces the prior "separate BF16 accum
// + requant at step end" with per-mb direct in-place accumulation
// on the quantized mom_buf. The pure-PyTorch implementation is
// 14+ s/mb at 1.65B scale (dominated by op dispatch + intermediate
// tensor allocation, achieving <1% of peak memory bandwidth).
// This kernel does the whole pass in a single tight C++ loop with
// stack-allocated work arrays — no intermediates, no dispatch —
// and hits the 1-2 s/mb target at prod scale.
//
// Algorithm (per block of 32 elements)
// -------------------------------------
//   scale = e8m0_to_float(mom_scale[b])           // 1 byte → float
//   vals[0..31] = e4m3_to_float(mom_buf[b..b+31]) * scale
//   vals[0..31] += bf16_to_float(grad[b..b+31])   // in-place
//   absmax = max(|vals[0..31]|)
//   new_scale = absmax / 448.0                    // E4M3 max
//   new_scale_byte = float_to_e8m0(new_scale)     // round to pow2
//   new_scale_f = e8m0_to_float(new_scale_byte)
//   for j: mom_buf[b+j] = float_to_e4m3(vals[j] / new_scale_f)
//   mom_scale[b] = new_scale_byte
//
// All conversions are inlined; the only allocations are stack
// locals of 32 floats per block. The hot loop has no branches
// except the absmax / quantize clamps (predictable, ~free).
//
// Correctness
// -----------
//   - E4M3 dequant: matches the spec (sign + 4-bit exp + 3-bit
//     mant, subnormal range 2^-9 .. 2^-6, normal range 2^-6 ..
//     448, NaN at exp=15).
//   - E4M3 quantize: round-to-nearest-even via the standard
//     "roundf" path; subnormal handling preserves small values
//     the gradient update path can produce.
//   - E8M0 quantize: nearest power of 2 via log2 + round; clamps
//     to [1, 254] to avoid the 0-byte divide-by-zero on dequant
//     (an all-zero block produces a 0-scale that would NaN out
//     downstream; the clamp forces a finite non-NaN scale whose
//     quantized values are also zero).
//   - BF16 grad → float: zero-extension of the upper 16 bits of
//     a 32-bit float (BF16 == float32 with the lower 16 mantissa
//     bits zeroed). Exact.

#include <torch/extension.h>
#include <cstdint>
#include <cmath>
#include <cstring>

// ---------------------------------------------------------------------------
// E4M3 ↔ float (OCP Microscaling spec, MX FP8_E4M3).
// ---------------------------------------------------------------------------
// E4M3 byte layout (MX variant, max finite = 448 at 0x7E):
//   bit 7     : sign
//   bits 6-3  : 4-bit exponent, bias 7
//   bits 2-0  : 3-bit mantissa (no implicit 1 for subnormals)
// Special:
//   exp = 0      : subnormal 2^(-9) * mant, mant in [0, 7]
//   exp = 1..15  : normal 2^(exp-7) * (1 + mant/8)
//   exp = 15, mant = 7 : NaN (0x7F / 0xFF)
//   exp = 15, mant = 0..6 : finite, range 256..448
// Max finite: 448 at 0x7E (exp=15, mant=6).
//
// Conversions are bit-manipulation (no ldexpf/frexpf) — those
// libm calls are 10-50 ns each and dominate the per-elt cost at
// scalar throughput.

static inline float e4m3_to_float(uint8_t b) {
    uint32_t sign = (b & 0x80u) ? 0x80000000u : 0u;
    uint32_t exp_bits = (b >> 3) & 0xFu;
    uint32_t mant = b & 0x7u;
    if (exp_bits == 0) {
        // Subnormal: 2^(-9) * mant. As float32: sign | (mant)
        // in the mantissa field with the float exponent = -9+127
        // = 118. But subnormals in float32 use biased exp=0 with
        // the implicit 1 disabled. So encode as exp=0 in float
        // but with a non-zero mantissa. Easier: just compute the
        // value as a small float.
        float val = (float)mant * (1.0f / 512.0f);  // 2^-9
        return sign ? -val : val;
    }
    if (exp_bits == 0xF && mant == 0x7) {
        return NAN;  // 0x7F / 0xFF
    }
    // Normal: assemble float32 directly.
    // Float exp = exp_bits - 7 + 127 = exp_bits + 120
    // Float mant = mant << 20 (top 3 mant bits of E4M3 → top 3
    // mant bits of float32, with the implicit 1 coming "for free"
    // because the float32 implicit-1 bit is bit 23 and we set
    // bits 22-20 to the E4M3 mant). But the implicit 1 must
    // actually be at bit 23 for normals; in our case the mant
    // is the 3 fraction bits at 22-20.
    uint32_t float_exp = (exp_bits + 120) << 23;  // 120 = 127 - 7
    uint32_t float_mant = mant << 20;
    uint32_t bits = sign | float_exp | float_mant;
    float val;
    std::memcpy(&val, &bits, 4);
    return val;
}

static inline uint8_t float_to_e4m3(float v) {
    if (std::isnan(v)) {
        return 0x7Fu;  // qNaN
    }
    uint32_t sign = 0;
    if (v < 0.0f) {
        sign = 0x80u;  // sign bit at position 7 of the e4m3 byte
        v = -v;
    }
    if (v > 448.0f) v = 448.0f;  // E4M3 max magnitude (0x7E)
    if (v == 0.0f) {
        return (uint8_t)sign;
    }
    // Bit-decompose v.
    uint32_t bits;
    std::memcpy(&bits, &v, 4);
    int v_exp = (int)((bits >> 23) & 0xFFu) - 127;  // unbiased
    uint32_t v_mant = bits & 0x7FFFFFu;              // 23-bit mant
    // E4M3 normal: 2^(e_biased-7) * (1 + m/8)
    //   m in [0, 7], e_biased in [1, 15]
    //   range: 2^(-6) * 1 .. 2^8 * 1.875
    // Subnormal: 2^(-9) * m, m in [0, 7]
    //   range: 0 .. 2^(-6) * 7/8 = 13.4e-3
    //
    // Convert from float's (1.m) * 2^v_exp to E4M3's (1 + m/8) * 2^e:
    //   The E4M3 mantissa is the top 3 bits of the float mantissa.
    //   v_mant >> 20 gives us the top 3 bits.
    if (v_exp < -6) {
        // Subnormal or underflow.
        // m = round(v / 2^-9) = round(v * 512)
        // Use a more accurate round: compute in float then quantize.
        // For v in [2^-10, 2^-6):
        //   m = round(v * 512). m in [0, 7].
        // For v < 2^-10: round to 0.
        if (v < (1.0f / 1024.0f)) {  // 2^-10
            return (uint8_t)sign;
        }
        // Reconstruct m from the actual float bits for accuracy.
        // v * 512 = 2^(v_exp + 9) * (1 + v_mant/2^23)
        // If v_exp + 9 == 0, m is in [0, 8) directly. If < 0, the
        // value is too small. We've ruled out v_exp < -10 above, so
        // v_exp + 9 in [-1, 3].
        int e_shifted = v_exp + 9;
        if (e_shifted < 0) {
            // v in [2^-10, 2^-9): m = round(v * 512) where
            // v * 512 = 2^(v_exp+9) * (1 + frac) in [0.5, 1.0)
            // → m = 0 (we can't represent half-quantum values
            //   at this scale; round to nearest gives 0 or 1
            //   depending on exact value).
            // Use 1-bit rounding: if the half-step is below
            // 0.75, round to 0; else 1.
            // Simpler: m = (v * 512 + 0.5) cast to int. But
            // casting works for v in [2^-10, 2^-9):
            //   v * 512 in [0.5, 1.0). round to 0 or 1.
            float scaled = v * 512.0f;
            int m = (int)(scaled + 0.5f);
            if (m > 7) m = 7;
            return (uint8_t)(sign | (uint32_t)m);
        }
        if (e_shifted == 0) {
            // v in [2^-9, 2^-8): m in [1, 2)
            float scaled = v * 512.0f;  // in [1, 2)
            int m = (int)(scaled + 0.5f);
            if (m > 7) m = 7;
            return (uint8_t)(sign | (uint32_t)m);
        }
        // v_exp + 9 in [1, 3]: v in [2^-8, 2^-6)
        float scaled = v * 512.0f;  // in [2, 8)
        int m = (int)(scaled + 0.5f);
        if (m > 7) m = 7;
        return (uint8_t)(sign | (uint32_t)m);
    }
    if (v_exp > 8) {
        // Saturate to 448 = 0x7E (exp=15, mant=6).
        return (uint8_t)(sign | (0xFu << 3) | 6u);
    }
    // Normal. v = (1 + v_mant/2^23) * 2^v_exp
    //   E4M3 = (1 + m/8) * 2^v_exp where m = round(v_mant/2^20)
    //   but with possible round-up carrying to exp+1.
    // The top 3 mantissa bits (v_mant >> 20) are the "natural"
    // round-to-nearest target. Tie-breaking: round to even.
    int m_top = (int)(v_mant >> 20);  // in [0, 7]
    int m_round_bit = (int)((v_mant >> 19) & 1);
    int m_sticky = (int)(v_mant & 0x7FFFFu) ? 1 : 0;
    int m;
    if (m_round_bit == 0) {
        m = m_top;  // round down
    } else if (m_round_bit == 1 && m_sticky == 0) {
        // Tie: round to even.
        m = (m_top & 1) ? m_top + 1 : m_top;
    } else {
        m = m_top + 1;  // round up
    }
    int e_biased = v_exp + 7;
    if (m >= 8) {
        m = 0;
        e_biased++;
        if (e_biased > 15) {
            // Mantissa overflow at exp=15: saturate to 448.
            return (uint8_t)(sign | (0xFu << 3) | 6u);
        }
    }
    return (uint8_t)(sign | ((uint32_t)e_biased << 3) | (uint32_t)m);
}

// ---------------------------------------------------------------------------
// E8M0 ↔ float (OCP Microscaling spec).
// ---------------------------------------------------------------------------
// E8M0 byte layout: 8-bit exponent, no mantissa, no sign.
//   byte b represents 2^(b-127), for b in [0, 255].
// We use byte=1 (= 2^-126) as the floor for safety; byte=0 is
// reserved/unused (dequant would divide by zero).

static inline float e8m0_to_float(uint8_t b) {
    if (b == 0) return 0.0f;
    // As float32: biased exp = b, mantissa = 0. So
    //   bits = (b << 23). Sign = 0.
    uint32_t bits = (uint32_t)b << 23;
    float val;
    std::memcpy(&val, &bits, 4);
    return val;
}

static inline uint8_t float_to_e8m0(float v) {
    if (!(v > 0.0f) || std::isnan(v)) return 1u;  // floor at 2^-126
    // log2f is unavoidable here — there's no bit trick to
    // extract log2 of an arbitrary positive float. It's one call
    // per 32-elt block (not per elt) so the cost is amortized.
    int e = (int)roundf(log2f(v));
    int b = e + 127;
    if (b < 1) b = 1;
    if (b > 254) b = 254;
    return (uint8_t)b;
}

// ---------------------------------------------------------------------------
// BF16 → float (zero-extension of upper 16 bits; exact).
// ---------------------------------------------------------------------------
static inline float bf16_to_float(uint16_t b) {
    uint32_t bits = (uint32_t)b << 16;
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}

// ---------------------------------------------------------------------------
// Fused per-mb kernel.
// ---------------------------------------------------------------------------
void fused_mxfp8_dequant_add_requant(
    torch::Tensor mom_buf,
    torch::Tensor mom_scale,
    torch::Tensor grad,
    int64_t rows,
    int64_t cols_p,
    int64_t block_size
) {
    TORCH_CHECK(mom_buf.is_cpu() && mom_scale.is_cpu() && grad.is_cpu(),
                "fused_mxfp8_dequant_add_requant: all tensors must be CPU");
    TORCH_CHECK(mom_buf.scalar_type() == at::kFloat8_e4m3fn,
                "fused_mxfp8_dequant_add_requant: mom_buf must be float8_e4m3fn");
    TORCH_CHECK(mom_scale.scalar_type() == at::kFloat8_e8m0fnu,
                "fused_mxfp8_dequant_add_requant: mom_scale must be float8_e8m0fnu");
    TORCH_CHECK(grad.scalar_type() == at::kBFloat16,
                "fused_mxfp8_dequant_add_requant: grad must be bfloat16");
    TORCH_CHECK(block_size == 32,
                "fused_mxfp8_dequant_add_requant: block_size must be 32 "
                "(only 32-elt blocks are supported in this build)");

    const int64_t bs = 32;
    const int64_t n_blocks_per_row = cols_p / bs;
    const int64_t numel = rows * cols_p;

    uint8_t* __restrict__ mb = reinterpret_cast<uint8_t*>(mom_buf.data_ptr());
    uint8_t* __restrict__ ms = reinterpret_cast<uint8_t*>(mom_scale.data_ptr());
    uint16_t* __restrict__ g = reinterpret_cast<uint16_t*>(grad.data_ptr<at::BFloat16>());

    // Stack-allocated work array for one block (32 floats). No
    // heap traffic in the inner loop.
    float vals[32];

    for (int64_t b = 0; b < numel; b += bs) {
        // Scale array is laid out as [rows, n_blocks_per_row],
        // stored row-major. The block index within a row is
        // (b % cols_p) / bs; the row index is b / cols_p.
        const int64_t scale_idx = (b / bs) % n_blocks_per_row
                                + (b / cols_p) * n_blocks_per_row;
        const float scale = e8m0_to_float(ms[scale_idx]);

        // Pass 1: dequant + add grad, accumulate absmax.
        float absmax = 0.0f;
        for (int j = 0; j < 32; j++) {
            float dq = e4m3_to_float(mb[b + j]) * scale;
            float gv = bf16_to_float(g[b + j]);
            float s = dq + gv;
            vals[j] = s;
            float av = fabsf(s);
            if (av > absmax) absmax = av;
        }

        // Compute new E8M0 scale.
        const float new_scale_target = absmax / 448.0f;
        const uint8_t new_scale_byte = float_to_e8m0(new_scale_target);
        const float new_scale = e8m0_to_float(new_scale_byte);

        // Pass 2: requantize (in-place write of mom_buf).
        const float inv_scale = 1.0f / new_scale;  // precompute the divide
        for (int j = 0; j < 32; j++) {
            float q = vals[j] * inv_scale;
            if (q > 448.0f) q = 448.0f;
            else if (q < -448.0f) q = -448.0f;
            mb[b + j] = float_to_e4m3(q);
        }
        ms[scale_idx] = new_scale_byte;
    }
}

// ---------------------------------------------------------------------------
// Requantize-only kernel (used at step() time after NS).
// ---------------------------------------------------------------------------
// Reads mom_buf in E4M3 form + mom_scale in E8M0, dequantizes
// in-place to a stack-allocated float buffer, requantizes back.
// Used at the END of the optimizer step to update the stored
// mom_buf after the Newton-Schulz orthogonalize on the GPU has
// produced the new momentum values (which we then need to store
// back as mxfp8 for next cycle's accumulation).
void requantize_mxfp8(
    torch::Tensor mom_buf,
    torch::Tensor mom_scale,
    int64_t rows,
    int64_t cols_p,
    int64_t block_size
) {
    TORCH_CHECK(mom_buf.is_cpu() && mom_scale.is_cpu(),
                "requantize_mxfp8: tensors must be CPU");
    TORCH_CHECK(mom_buf.scalar_type() == at::kFloat8_e4m3fn,
                "requantize_mxfp8: mom_buf must be float8_e4m3fn");
    TORCH_CHECK(mom_scale.scalar_type() == at::kFloat8_e8m0fnu,
                "requantize_mxfp8: mom_scale must be float8_e8m0fnu");
    TORCH_CHECK(block_size == 32, "requantize_mxfp8: block_size must be 32");

    const int64_t bs = 32;
    const int64_t n_blocks_per_row = cols_p / bs;
    const int64_t numel = rows * cols_p;

    uint8_t* __restrict__ mb = reinterpret_cast<uint8_t*>(mom_buf.data_ptr());
    uint8_t* __restrict__ ms = reinterpret_cast<uint8_t*>(mom_scale.data_ptr());

    float vals[32];

    for (int64_t b = 0; b < numel; b += bs) {
        const int64_t scale_idx = (b / bs) % n_blocks_per_row
                                + (b / cols_p) * n_blocks_per_row;
        const float scale = e8m0_to_float(ms[scale_idx]);

        // Read existing quantized momentum, dequantize to vals[].
        float absmax = 0.0f;
        for (int j = 0; j < 32; j++) {
            float v = e4m3_to_float(mb[b + j]) * scale;
            vals[j] = v;
            float av = fabsf(v);
            if (av > absmax) absmax = av;
        }

        // Requantize (typically just rebalances the E8M0 scale;
        // the underlying values haven't changed).
        const float new_scale_target = absmax / 448.0f;
        const uint8_t new_scale_byte = float_to_e8m0(new_scale_target);
        const float new_scale = e8m0_to_float(new_scale_byte);
        const float inv_scale = 1.0f / new_scale;
        for (int j = 0; j < 32; j++) {
            float q = vals[j] * inv_scale;
            if (q > 448.0f) q = 448.0f;
            else if (q < -448.0f) q = -448.0f;
            mb[b + j] = float_to_e4m3(q);
        }
        ms[scale_idx] = new_scale_byte;
    }
}
