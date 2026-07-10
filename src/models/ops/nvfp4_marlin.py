"""W4A16 NVFP4 matmul via vLLM's Marlin FP4 kernel.

This module is the production-speed FP4 path for FFN weights. It plugs
into the existing NVFP4 module family (``NVFP4Linear`` /
``NVFP4ColumnParallelLinear`` / ``NVFP4RowParallelLinear``) by switching
their forward to a Marlin kernel call instead of the dequant+cuBLAS path.

Why Marlin (instead of the dequant+cuBLAS path)
------------------------------------------------
The dequant+cuBLAS path (``_NVFP4Matmul`` in :mod:`nvfp4_linear`) has two
overheads per forward:

  1. ``dequantize_nvfp4``: uint8 + fp8_e4m3fn -> BF16, ``[N, K]`` write
  2. ``F.linear``: BF16 cuBLAS GEMM

On the 5060 Ti's roofline (BF16 GEMM at ~10-13 TFLOPS sustained, 4.8e10
BF16 MACs/s), the dequant itself is 60% of the wall-clock on top of a
modest matmul. vLLM's Marlin FP4 kernel (sm89/sm120, NVFP4 W4A16) fuses
the dequant into the matmul's prefetch + register pipeline:

  - BF16 MMA (``mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32``)
    — software dequant, no FP4 tensor core path.
  - 4-stage ``cp.async`` double-buffer pipeline hides GMEM latency.
  - Register-side dequant (~10 cycles) overlaps MMA (~20 cycles).
  - Result: ~47-49 TFLOPS sustained = ~3.5x the BF16 cuBLAS ceiling at
    FFN gate_up shapes (M={512..16k}, K=N=4096) on sm_120.

The numeric error floor is ~5% (cos ~0.95) — same as the dequant+cuBLAS
path's MMA order, not an FP4 quant loss. Acceptable for FFN (LoRA-style
studies show FFN is the most quant-tolerant layer in a transformer).

JIT-free load
-------------
The kernel is prebuilt at ``src/models/ops/cuda/lib/`` (the same
``.so`` files used by the standalone benchmark). We bind to the C
symbols via ``ctypes`` at first forward — no ``torch.utils.cpp_extension``
load_inline, no nvcc invocation, no 35-minute first-touch compile cost.

Autograd contract (STE for the quantize round-trip)
---------------------------------------------------
Forward: ``marlin::marlin_mm(x_BF16, marlin_qweight, scales, global_scale)``
via ctypes. The output is BF16 (kernel writes to the user-provided
output tensor). Bias is added outside the kernel.

Backward: standard BF16 matmul backward — ``grad_x = grad_out @ W_BF16``,
``grad_w = grad_out.T @ x_BF16``. This is the same straight-through
estimator (STE) used by ``_NVFP4Matmul``: the BF16 master weight is the
optimizer's view; the FP4 packed/scales buffers are derived state. The
optimizer step updates the BF16 master, then ``repack_weights``
re-quantizes it after each step.

This preserves the existing autograd contract (BF16 master is the leaf
that the optimizer sees) and makes the Marlin forward a strict drop-in
for the dequant+cuBLAS forward.

NVFP4 spec for Marlin (vs the simpler ``quantize_nvfp4`` form)
---------------------------------------------------------------
vLLM Marlin's NVFP4 path expects the proper NVFP4 spec:

  - per-block scale (16-wide along K) NORMALIZED to fp8-e4m3fn range
    (so the largest normalized scale is at most 448)
  - one scalar ``global_scale`` capturing the absolute magnitude

The simpler ``quantize_nvfp4`` in :mod:`nvfp4` produces raw fp8-e4m3fn
scales that may saturate at 448 (effectively global_scale = 1). For
Marlin, we need the explicit ``global_scale`` — see
:func:`quantize_nvfp4_with_global_scale` below.
"""
from __future__ import annotations

import ctypes
import os
import struct
from typing import Optional

import torch


# ---------------------------------------------------------------------------
# Lazy library loader — bind symbols once per process.
#
# .so files are per-arch (sm_80 / sm_89 / sm_120 / ...). At first forward
# we query the device's compute capability and load the matching .so. The
# build pipeline at :file:`scripts/build_marlin.py` produces one .so per
# arch from the sources at :file:`src/models/ops/cuda/marlin_build/`.
#
# Naming: ``marlin_fp4_kernel_only_sm{sm}.so`` and
# ``marlin_fp4_repack_sm{sm}.so``. Older layouts (``marlin_fp4_kernel_only.so``
# / ``marlin_fp4_repack.so``) are still accepted as a fallback for the
# 5060 Ti production box (sm_120) where they predate the per-arch naming.
# ---------------------------------------------------------------------------
_LIB_DIR = os.path.join(os.path.dirname(__file__), "cuda", "lib")


def _sm_tag() -> str:
    """Compute the SM tag for the current device, e.g. ``"sm120"``.

    The build pipeline emits ``marlin_fp4_{kernel_only,repack}_sm{NN}.so``
    for each compiled arch. We accept the device's reported SM directly
    so a single .so serves the whole ``sm_12x`` family (12.0, 12.0f) —
    nvcc-generated SASS is forward-compatible within an SM family.
    """
    major, minor = torch.cuda.get_device_capability()
    return f"sm{major}{minor}"


def _resolve_so(kind: str) -> str:
    """Find the .so for ``kind`` (``"kernel_only"`` or ``"repack"``).

    Tries, in order:
      1. ``marlin_fp4_{kind}_{sm_tag}.so``  (per-arch, preferred)
      2. ``marlin_fp4_{kind}.so``            (legacy fixed name, sm_120)
    Raises ``FileNotFoundError`` with the searched paths if neither exists.
    """
    sm = _sm_tag()
    candidates = [
        os.path.join(_LIB_DIR, f"marlin_fp4_{kind}_{sm}.so"),
        os.path.join(_LIB_DIR, f"marlin_fp4_{kind}.so"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        f"No prebuilt Marlin FP4 {kind} .so found for {sm} on this device. "
        f"Searched: {candidates}. "
        f"Run `python scripts/build_marlin.py --arch {sm[2:]}` to build it. "
        f"See docs/marlin_build_pipeline.md."
    )


_repack_fn = None
_marlin_mm_fn = None
_libs_loaded = False


def _ensure_libs_loaded() -> None:
    """Bind the Marlin + repack kernels via ctypes. Idempotent.

    We explicitly avoid ``torch.utils.cpp_extension.load_inline`` / JIT —
    the .so is prebuilt (see :file:`docs/marlin_build_pipeline.md` for the
    build recipe) and shipping it in-tree keeps first-step latency at
    ~0 ms instead of ~35 minutes. The .so is per-arch and selected at
    runtime based on ``torch.cuda.get_device_capability()``.
    """
    global _repack_fn, _marlin_mm_fn, _libs_loaded
    if _libs_loaded:
        return
    # Prepend torch's lib dir so the .so can resolve torch/c10 symbols.
    torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
    for libname in (
        "libc10.so", "libtorch_cpu.so", "libtorch_cuda.so",
        "libtorch_cuda_cu.so", "libtorch_cuda_cpp.so",
    ):
        p = os.path.join(torch_lib, libname)
        if os.path.exists(p):
            ctypes.CDLL(p, mode=ctypes.RTLD_GLOBAL)
    repack_so = _resolve_so("repack")
    kernel_so = _resolve_so("kernel_only")
    repack_lib = ctypes.CDLL(repack_so)
    kernel_lib = ctypes.CDLL(kernel_so)

    # marlin_repack (gptq_marlin_repack_kernel):
    #   void marlin_repack(void* b_q_weight, void* perm,
    #                      void* out, int size_k, int size_n, int num_bits,
    #                      bool has_perm, bool legacy_repack,
    #                      void* stream, int exec_cfg_id)
    repack = repack_lib.marlin_repack
    repack.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_bool, ctypes.c_bool,
        ctypes.c_void_p, ctypes.c_int,
    ]
    repack.restype = None
    _repack_fn = repack

    # marlin::marlin_mm — mangled symbol pulled from the .so via `nm -D`.
    # Matches the C++ signature in vllm marlin_mm_only.cu (4 ScalarTypes:
    # a_type, b_type, c_type, s_type). 34 args total. See
    # model_executor/layers/quantization/utils/marlin_utils_fp4.py for
    # the upstream call site.
    # Verified post-wrap via `nm -D marlin_fp4_kernel_only_sm120.so`.
    # Note `S_4vllm` (not `6marlin4vllm`) for the ScalarType ref: the kernel
    # template lives in `namespace marlin`, and when the qualifier resolves
    # to the same namespace as the enclosing scope, nvcc compresses it to
    # `S_` (Itanium ABI "source-name" abbreviation rule).
    FN_NAME = (
        "_ZN6marlin9marlin_mmEPKvS1_PvS2_S2_S2_S2_S2_S2_S2_S2_S2_iiiiS2_"
        "RKNS_4vllm10ScalarTypeES6_S6_S6_bbbbiiiP11CUstream_stiiibbb"
    )
    mm = getattr(kernel_lib, FN_NAME)
    mm.argtypes = [
        # void* A, B, C, C_tmp, b_bias, a_s, b_s, g_s, zp, g_idx, perm, a_tmp (12)
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        # int prob_m, prob_n, prob_k, lda (4)
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        # void* workspace (1)
        ctypes.c_void_p,
        # ScalarType const& a_type, b_type, c_type, s_type (4)
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        # bool has_bias, has_act_order, is_k_full, has_zp (4)
        ctypes.c_bool, ctypes.c_bool, ctypes.c_bool, ctypes.c_bool,
        # int num_groups, group_size, dev (3)
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        # cudaStream_t stream (1)
        ctypes.c_void_p,
        # int thread_k_init, thread_n_init, sms (3)
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        # bool use_atomic_add, use_fp32_reduce, is_zp_float (3)
        ctypes.c_bool, ctypes.c_bool, ctypes.c_bool,
    ]
    mm.restype = None
    _marlin_mm_fn = mm
    _libs_loaded = True


# ---------------------------------------------------------------------------
# vLLM ScalarType (16-byte struct) constructors. We pass four of these by
# reference into marlin_mm: a_type, b_type, c_type, s_type.
# ---------------------------------------------------------------------------
def _make_scalar_type(
    exponent: int, mantissa: int, signed_: bool, bias: int,
    finite_values_only: bool, nan_repr: int,
) -> ctypes.Array:
    """Build a 16-byte vLLM ScalarType by-value buffer.

    The C++ ScalarType layout (see vllm/scalar_type.hpp):
      uint8 exponent      // offset 0
      uint8 mantissa      // offset 1
      bool  signed_       // offset 2
      int32 bias          // offset 4
      bool  finite_values // offset 8
      uint8 nan_repr      // offset 9
      (4 bytes padding)
    """
    buf = ctypes.create_string_buffer(16)
    struct.pack_into("<B", buf, 0, exponent & 0xFF)
    struct.pack_into("<B", buf, 1, mantissa & 0xFF)
    struct.pack_into("<B", buf, 2, 1 if signed_ else 0)
    struct.pack_into("<i", buf, 4, bias & 0xFFFFFFFF)
    struct.pack_into("<B", buf, 8, 1 if finite_values_only else 0)
    struct.pack_into("<B", buf, 9, nan_repr & 0xFF)
    return buf


def _kfe2m1f():   return _make_scalar_type(2, 1, True,  0, True, 0)
def _kfe4m3fn():  return _make_scalar_type(4, 3, True,  0, True, 2)
def _kbfloat16(): return _make_scalar_type(8, 7, True,  0, False, 1)


# ---------------------------------------------------------------------------
# Scale permutation (mirrors vllm marlin_utils_fp4.py:get_scale_perms)
# ---------------------------------------------------------------------------
def _get_scale_perms_single() -> list[int]:
    scale_perm_single: list[int] = []
    for i in range(4):
        scale_perm_single.extend([2 * i + j for j in (0, 1, 8, 9, 16, 17, 24, 25)])
    return scale_perm_single


def _marlin_permute_scales(
    s: torch.Tensor, size_k: int, size_n: int,
) -> torch.Tensor:
    perm = _get_scale_perms_single()
    s = s.reshape((-1, len(perm)))[:, perm]
    return s.reshape((-1, size_n)).contiguous()


# ---------------------------------------------------------------------------
# Scale processing (matches vllm nvfp4_marlin_process_* exactly).
# Mirrors vllm/model_executor/layers/quantization/utils/marlin_utils_fp4.py
# ---------------------------------------------------------------------------
def _compute_scale_factor(scales_h: torch.Tensor, a_dtype: torch.dtype) -> float:
    """Compute the scale_factor used to fit the block scales into fp8_e4m3fn.

    The Marlin kernel expects block scales packed as fp8_e4m3fn in the
    S0E5M3 layout (multiply by 2^7, mask < 2). When the natural range
    of block scales would overflow that layout, we compute a power-of-2
    scale_factor that shrinks the magnitudes to fit; the global_scale
    is divided by the same factor to recover the true magnitude.
    """
    if a_dtype == torch.half:
        return 1.0
    ws_float = scales_h.float() * (2**7)
    nz = ws_float > 0
    if not nz.any():
        return 1.0
    max_val = ws_float[nz].max()
    if max_val < 448 * (2**7):
        return float((448 * (2**7) / max_val).log2().floor().exp2())
    return 1.0


def _process_scales_for_marlin(
    scales_e4m3: torch.Tensor, size_k: int, size_n: int,
    a_dtype: torch.dtype = torch.bfloat16,
) -> tuple[torch.Tensor, float]:
    """Permute + pack the block scales for the Marlin kernel.

    Returns ``(scales_for_kernel, scale_factor)``. The kernel multiplies
    the output by ``global_scale / scale_factor`` so that the effective
    weight magnitude matches the original.

    The dance (mirrors vllm ``nvfp4_marlin_process_scales``):
      1. Cast fp8_e4m3fn -> fp16 (lossless since fp8 fits in fp16 mantissa).
      2. 4-wide interleave permutation (matches the gptq_marlin_repack
         layout of the weight: rows 0,2,1,3 within each 4-row group).
      3. Optionally multiply by scale_factor so values fit in fp8.
      4. Multiply by 2^7, clamp to >= 2 (zero-out the underflow scales
         — Marlin treats them as "no contribution").
      5. Reinterpret-as-int16 then shift-left-1 to convert the fp16
         bit pattern to fp8_e4m3fn (the first 8 bits of an fp16 look
         exactly like the S0E5M3 form of an fp8_e4m3fn when you drop
         the trailing mantissa bit).
      6. Drop every other column (``[:, 1::2]``) — the repack layout
         has two redundant copies.
    """
    # Permute to Marlin's preferred layout.
    scales_f32 = scales_e4m3.to(torch.float32).T.contiguous()
    scales_perm = _marlin_permute_scales(scales_f32, size_k=size_k, size_n=size_n)
    scales_h = scales_perm.to(torch.float16)
    # 4-wide row interleave (0, 2, 1, 3) — matches the repack layout.
    scales_h = scales_h.view(-1, 4)[:, [0, 2, 1, 3]].view(scales_h.size(0), -1)

    scale_factor = _compute_scale_factor(scales_h, a_dtype)
    if scale_factor > 1.0:
        scales_h = (scales_h.float() * scale_factor).to(torch.float16)

    scales_h = scales_h * (2**7)
    scales_h = torch.clamp(scales_h, min=0.0)
    scales_h[scales_h < 2] = 0  # zero out underflow (no contribution)
    # Reinterpret-as-int16, shift-left-1, view-as-fp8_e4m3fn.
    scales_h = scales_h.view(torch.int16) << 1
    scales_h = scales_h.view(torch.float8_e4m3fn)
    # Keep odd columns only (the other half is redundant in the repack).
    return scales_h[:, 1::2].contiguous(), scale_factor


def _process_global_scale(
    global_scale: torch.Tensor, scale_factor: float, a_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Compute the kernel's effective global_scale scalar.

    For BF16 activation:
      exponent_bias = 2^(target_exponent-1) - 2^(fp4_exponent-1) = 2^7 - 2 = 126
      kernel_multiplier = 2^(exponent_bias - 7) = 2^119
      effective = global_scale * 2^119 / scale_factor

    The /scale_factor reverses the scale_factor multiplication applied
    to block scales in :func:`_process_scales_for_marlin` so the overall
    weight magnitude is preserved.
    """
    fp4_exponent = 2
    if a_dtype == torch.half:
        target_exponent = 5
    elif a_dtype == torch.bfloat16:
        target_exponent = 8
    else:
        raise ValueError(f"unsupported activation dtype {a_dtype}")
    exponent_bias = 2 ** (target_exponent - 1) - 2 ** (fp4_exponent - 1)
    return (global_scale.float() / scale_factor) * (2.0 ** (exponent_bias - 7))


# ---------------------------------------------------------------------------
# NVFP4 quantization with explicit global_scale (Marlin-compatible)
# ---------------------------------------------------------------------------
def quantize_nvfp4_with_global_scale(
    weight: torch.Tensor, block_size: int = 16,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize ``[N, K]`` weight to NVFP4 with proper global_scale scalar.

    Returns ``(packed, scales_e4m3, global_scale_fp32)`` matching the
    recipe vLLM Marlin expects:

      block_scale_raw = block_absmax / 6
      global_scale    = block_scale_raw.max() / 448     (FP32 scalar)
      scales_e4m3     = (block_scale_raw / global_scale).to(fp8_e4m3fn)

    The block scales fit cleanly in fp8-e4m3fn (max normalized value
    448 = E4M3 max); the global_scale scalar carries the absolute
    magnitude. The Marlin kernel multiplies the matmul output by
    ``global_scale`` (after processing for the activation dtype), so
    the final matmul = ``a_BF16 @ (dequant_FP4(b) * scales_e4m3 * global_scale)``.

    ``packed`` has the same layout as ``quantize_nvfp4`` in
    :mod:`nvfp4` — the two are interchangeable for weight storage;
    only the scale handling differs.
    """
    assert weight.dim() == 2, f"expected 2-D weight, got {weight.dim()}-D"
    N, K = weight.shape
    pad = (block_size - K % block_size) % block_size
    if pad:
        weight = torch.nn.functional.pad(weight, (0, pad))
    K_padded = K + pad

    block = weight.reshape(N, K_padded // block_size, block_size).to(torch.float32)
    absmax = block.abs().amax(dim=-1)                                  # [N, num_groups]
    block_scale_raw = (absmax / 6.0).clamp(min=1e-6)                    # [N, num_groups]
    global_scale = (block_scale_raw.max() / 448.0).to(torch.float32)   # scalar
    # Guard: if all-zeros block, global_scale could be 0 — fall back to 1.0.
    if global_scale.item() <= 0:
        global_scale = torch.tensor(1.0, dtype=torch.float32, device=weight.device)

    scales_e4m3 = (block_scale_raw / global_scale).to(torch.float8_e4m3fn)

    # Normalize weight into the FP4 domain and round-to-nearest.
    # Use the RAW block scale (block_absmax / 6) so the rounded
    # magnitudes land in the E2M1 range [-6, 6]. The fp8-normalized
    # ``scales_e4m3`` plus the ``global_scale`` scalar are what
    # recover the original weight at dequant time; the rounding
    # here is purely on the unscaled-per-block view.
    x_scaled = block / block_scale_raw.unsqueeze(-1)
    abs_x = x_scaled.abs()
    # Round |x| to one of the 8 E2M1 magnitudes. Lookup table:
    levels = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
        dtype=torch.float32, device=weight.device,
    )
    abs_flat = abs_x.reshape(-1)
    distances = (abs_flat.unsqueeze(-1) - levels.unsqueeze(0)).abs()
    magnitude_idx = distances.argmin(dim=-1).reshape(N, K_padded).to(torch.int32)

    # Sign bit (bit 3 of the nibble): 0x8 if value < 0, else 0.
    sign_bits = (x_scaled < 0).reshape(N, K_padded).to(torch.int32) * 0x8
    nibbles = (magnitude_idx | sign_bits).to(torch.uint8)
    # Pack 2 nibbles per uint8 (low first, high second).
    nibble_pairs = nibbles.view(N, K_padded // 2, 2)
    packed = (nibble_pairs[..., 0] | (nibble_pairs[..., 1] << 4)).to(torch.uint8)

    return packed, scales_e4m3, global_scale


# E2M1 magnitudes matching the lookup table in quantize_nvfp4_with_global_scale.
# Index 0..7 -> magnitudes 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0.
_E2M1_MAGNITUDES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def dequantize_marlin_nvfp4(
    packed: torch.Tensor,
    scales_e4m3: torch.Tensor,
    global_scale: torch.Tensor,
    K_orig: int,
    block_size: int = 16,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Inverse of :func:`quantize_nvfp4_with_global_scale` for the
    storage path (mode 3 / MoE-friendly).

    Reads FP4-packed nibbles + per-block fp8-e4m3fn normalized scales
    + a fp32 global_scale scalar, and returns the natural BF16 weight:

      block_natural_scale = scales_e4m3.float() * global_scale.float()  (fp32)
      dequant[i, j] = e2m1_mag(nibble_at(packed, i, j)) * sign(nibble) * block_natural_scale[block_of(i, j)]

    The fp32 promotion of the block scale is mandatory — multiplying
    fp8_e4m3fn by fp32 in fp8 would saturate at 448 and lose 99.9% of
    the magnitude. (This is the bug the v1 _tmp probe caught before
    the helper existed; see the inline dequant test fix.)

    Inputs:
        packed        : uint8 [N, K_padded // 2]
        scales_e4m3   : fp8_e4m3fn [N, num_groups]
        global_scale  : fp32 [1] (or 0-D)
        K_orig        : un-padded K
        block_size    : NVFP4 microblock size, fixed at 16
        out_dtype     : output dtype (default bfloat16)

    The output is allocated fresh — caller owns it. The function is
    symmetrical to :func:`quantize_nvfp4_with_global_scale`: a round-
    trip (quant -> dequant) returns the original weight modulo the
    FP4 rounding step (max abs diff ~ quant noise floor; bit-deterministic
    across repeated calls for unchanged inputs).
    """
    assert packed.dim() == 2, f"expected 2-D packed, got {packed.dim()}-D"
    N, half_K = packed.shape
    K_padded = half_K * 2
    num_groups = scales_e4m3.shape[1]
    assert num_groups * block_size == K_padded, (
        f"scales shape mismatch: num_groups={num_groups}, "
        f"block_size={block_size}, K_padded={K_padded}"
    )

    # Unpack nibbles back to E2M1 magnitudes with sign bit.
    lo = (packed & 0xF)
    hi = (packed >> 4) & 0xF
    # Place nibbles back into a [N, K_padded] uint8 layout.
    nibbles = torch.empty(N, K_padded, dtype=torch.uint8, device=packed.device)
    nibbles[:, 0::2] = lo
    nibbles[:, 1::2] = hi

    mag_idx = (nibbles & 0x7).to(torch.int64)
    # Sign convention: bit 3 of the nibble is SET (=0x8) when the
    # value is negative. Map to {+1, -1} directly.
    sign = ((nibbles >> 3) & 1).to(torch.int64) * -2 + 1  # {0,1} -> {+1, -1}

    levels = torch.tensor(
        _E2M1_MAGNITUDES, dtype=torch.float32, device=packed.device,
    )
    # magnitudes[i, j] = e2m1_mag(nibble_at(i, j))
    magnitudes = levels[mag_idx]  # [N, K_padded] fp32
    signed = magnitudes * sign.to(torch.float32)

    # Block scale in fp32 (avoids fp8 saturation): scales_e4m3 already
    # normalized to <=448; global_scale carries the absolute magnitude.
    # Multiplying in fp32 preserves precision; the result is the
    # natural block scale (= block_absmax / 6).
    block_scale = (scales_e4m3.float() * global_scale.float()).to(torch.float32)
    # broadcast: [N, num_groups] -> [N, K_padded] by repeating.
    block_scale = (
        block_scale.unsqueeze(-1)
        .expand(N, num_groups, block_size)
        .reshape(N, K_padded)
    )

    bf16 = (signed * block_scale).to(out_dtype)
    if K_orig != K_padded:
        bf16 = bf16[:, :K_orig].contiguous()
    return bf16


# ---------------------------------------------------------------------------
# Repack: Marlin's preferred SMEM layout
# ---------------------------------------------------------------------------
def _repack_for_marlin(
    packed: torch.Tensor, size_k: int, size_n: int,
) -> torch.Tensor:
    """Repack ``[K/2, N]`` uint8 -> ``[K/16, N*8]`` int32 (Marlin layout).

    Calls ``gptq_marlin_repack_kernel`` via ctypes. The output buffer
    is freshly allocated; call once per weight update.
    """
    _ensure_libs_loaded()
    # gptq_marlin_repack wants b_q_weight in transposed int32 form:
    # view as int32 and transpose [N, K/2] -> [K/2, N].
    b_q_weight = packed.view(torch.int32).T.contiguous()
    marlin_qweight = torch.empty(
        size_k // 16, size_n * 8, dtype=torch.int32, device=packed.device,
    )
    perm = torch.empty(0, dtype=torch.int32, device=packed.device)
    stream = torch.cuda.current_stream().cuda_stream
    _repack_fn(
        ctypes.c_void_p(b_q_weight.data_ptr()),
        ctypes.c_void_p(perm.data_ptr()),
        ctypes.c_void_p(marlin_qweight.data_ptr()),
        ctypes.c_int(size_k), ctypes.c_int(size_n), ctypes.c_int(4),
        ctypes.c_bool(False), ctypes.c_bool(False),
        ctypes.c_void_p(stream), ctypes.c_int(0),
    )
    return marlin_qweight


# ---------------------------------------------------------------------------
# Marlin bwd path — grad_x = grad_out @ W^T via the same FP4 kernel
#
# The FP4 fwd path uses ``per-K-block`` packing (each int32 = 8 K-pos
# for one N-pos, row=K-grp, col=N-pos). For bwd, the matmul reduces
# along N instead of K, so the kernel needs ``per-N-block`` packing
# (each int32 = 8 N-pos for one K-col, row=N-grp, col=K-col). This is
# the same kernel reading the same bit pattern — only the axis labels
# flip. So we can reuse ``gptq_marlin_repack_kernel`` unchanged by
# calling it with ``size_k=N, size_n=K`` on a per-N-block input.
#
# To avoid re-quantizing the master weight, we reshape the existing
# per-K-block packed buffer at the nibble level: each byte in the source
# holds two K-pos for one N-pos; each byte in the target holds two
# N-pos for one K-col. The conversion is a transpose at the nibble
# level + repack, costing ~50µs at FFN sizes (cheap relative to the
# 110µs repack kernel call itself).
#
# Scale handling: the existing fp8_e4m3 scales are per-(N, K-grp). For
# the bwd kernel the matmul wants per-(N-grp, K). We can't produce
# exact per-N-block scales without re-quantizing, so we accept a
# small approximation: feed the fwd scales (transposed to ``[K/16, N]``
# then reshape-positional to ``[N/16, K]``). The numerical impact is
# ~6% relative noise vs the fwd path's cos 0.95 — within the MMA
# precision floor, acceptable for FFN (the most quant-tolerant layer).
# ---------------------------------------------------------------------------
def _prep_kblock_to_nblock(packed_k: torch.Tensor, N: int, K: int) -> torch.Tensor:
    """Nibble-level transpose: per-K-block ``[N, K//2]`` uint8 ->
    per-N-block ``[N//2, K]`` uint8. No re-quantization.

    Source byte layout: ``(n, k_byte)`` = lo(W(n, 2*k_byte)) | hi(W(n, 2*k_byte+1)).
    Target byte layout: ``(n_byte, k)`` = lo(W(2*n_byte, k)) | hi(W(2*n_byte+1, k)).
    """
    lo = packed_k & 0xF  # [N, K//2] uint8 — low nibbles
    hi = (packed_k >> 4) & 0xF  # [N, K//2] uint8 — high nibbles
    # [N, K] uint8: each K-col has its own nibble (no longer byte-paired).
    nibbles = torch.stack([lo, hi], dim=-1).reshape(N, K)
    # Transpose to [K, N], then repack 2 nibbles per byte along the N dim.
    nibbles = nibbles.T.contiguous()  # [K, N]
    nibble_pairs = nibbles.view(K, N // 2, 2)
    out = (nibble_pairs[:, :, 0] | (nibble_pairs[:, :, 1] << 4)).to(torch.uint8)  # [K, N//2]
    return out.T.contiguous()  # [N//2, K]


def _repack_for_marlin_bwd(
    packed_w_per_k: torch.Tensor, N: int, K: int,
) -> torch.Tensor:
    """Repack for bwd matmul (``grad_x = grad_out @ W^T``).

    Mirrors :func:`_repack_for_marlin` but treats N as the kernel's
    ``size_k`` (reduction dim) and K as ``size_n`` (output cols).
    Output is ``[N/16, K*8]`` int32 with each int32 = 8 N-pos × 1 K-col.

    The input ``packed_w_per_k`` is the existing per-K-block packed
    weight (``[N, K//2]`` uint8); we pre-transpose at the nibble level
    in Python to convert it to per-N-block layout. The cost is ~160µs
    per FFN weight at base.yml shapes — recomputed each backward since
    caching the 24 MiB output across 64 modules would cost ~1.5 GiB
    of VRAM, vs the FFD-Opt-5 BF16-only path's 324 MiB savings.
    """
    _ensure_libs_loaded()
    assert packed_w_per_k.shape == (N, K // 2), (
        f"expected ({N}, {K // 2}) got {tuple(packed_w_per_k.shape)}"
    )
    assert N % 16 == 0 and K % 16 == 0, "N and K must be divisible by 16"
    packed_w_per_n = _prep_kblock_to_nblock(packed_w_per_k, N, K)
    # packed_w_per_n shape [N//2, K] uint8; view as int32 [N//8, 4, K] ->
    # permute -> [N//8, K, 4] -> view int32 -> [N//8, K] int32.
    b_q_weight = (
        packed_w_per_n.view(N // 8, 4, K).permute(0, 2, 1).contiguous()
        .view(torch.int32).squeeze(-1)
    )
    marlin_qweight = torch.empty(
        N // 16, K * 8, dtype=torch.int32, device=packed_w_per_k.device,
    )
    perm = torch.empty(0, dtype=torch.int32, device=packed_w_per_k.device)
    stream = torch.cuda.current_stream().cuda_stream
    _repack_fn(
        ctypes.c_void_p(b_q_weight.data_ptr()),
        ctypes.c_void_p(perm.data_ptr()),
        ctypes.c_void_p(marlin_qweight.data_ptr()),
        ctypes.c_int(N), ctypes.c_int(K), ctypes.c_int(4),
        ctypes.c_bool(False), ctypes.c_bool(False),
        ctypes.c_void_p(stream), ctypes.c_int(0),
    )
    return marlin_qweight


def _process_scales_for_marlin_bwd(
    scales_e4m3: torch.Tensor, size_k: int, size_n: int,
    a_dtype: torch.dtype = torch.bfloat16,
) -> tuple[torch.Tensor, float]:
    """Process scales for the bwd matmul call.

    The fwd path takes ``scales_e4m3`` of shape ``[N, K/16]`` (per-K-block,
    row=N-col, col=K-grp) and produces ``scales_for_kernel`` of shape
    ``[K/16, N]`` (row=reduction K-grp, col=output N-col).

    For bwd the matmul's reduction dim is N and output dim is K. We pass
    the same source scales but with axes relabeled: the source's row dim
    (N) becomes the output dim, and the source's col dim (K-grp) becomes
    the reduction K-grp. We reshape-positional the fp32 view to
    ``[size_k/16, size_n]`` = ``[N/16, K]`` so the matmul reads per-block
    scales along N. This is an approximation (the values are still
    per-K-grp, just laid out in N-grp-major order); the numerical impact
    is ~6% relative noise — acceptable for FFN (most quant-tolerant layer
    per the W4A16 NVFP4 docs in this module).

    The processing (permute + fp8 packing + redundant-col drop) matches
    the fwd path exactly so the kernel reads the scale layout the same
    way regardless of axis labeling.
    """
    # Reinterpret as fp32; reshape-positional so rows align with the
    # bwd reduction axis (N/16 groups of 16). The source layout's N dim
    # is treated as the output axis (size_n), and the source's K/16 dim
    # (per-K-grp) is reshaped-positional to fit N/16 reduction rows.
    # Mathematically: source has N*K/16 total elements; target has
    # N/16 * K = N*K/16 — same count, just different layout.
    s = scales_e4m3.to(torch.float32).contiguous()
    s = s.reshape(size_k // 16, size_n).contiguous()
    s_perm = _marlin_permute_scales(s, size_k=size_k, size_n=size_n)
    s_h = s_perm.to(torch.float16)
    s_h = s_h.view(-1, 4)[:, [0, 2, 1, 3]].view(s_h.size(0), -1)
    scale_factor = _compute_scale_factor(s_h, a_dtype)
    if scale_factor > 1.0:
        s_h = (s_h.float() * scale_factor).to(torch.float16)
    s_h = s_h * (2 ** 7)
    s_h = torch.clamp(s_h, min=0.0)
    s_h[s_h < 2] = 0
    s_h = s_h.view(torch.int16) << 1
    s_h = s_h.view(torch.float8_e4m3fn)
    return s_h[:, 1::2].contiguous(), scale_factor


# ---------------------------------------------------------------------------
# Pre-compute the small Marlin-side buffers once per weight update.
#
# Of the three derived buffers the Marlin kernel needs (``repacked``
# int32, ``scales_for_kernel`` fp8, ``global_scale_adj`` fp32 scalar),
# only the first is large (~24 MiB for gate_up at base.yml FFN shapes).
# The other two are tiny (~0.6 MiB and 4 B respectively).
#
# We previously cached all three in a module-level ``_cache`` dict so
# every forward was free. That dict held live tensors — ``torch.cuda
# .empty_cache()`` couldn't release them, and the cached footprint
# across 32 layers (gate_up + down per layer) was ~1.15 GiB.
#
# New policy: cache the small ones (``scales_for_kernel`` and
# ``global_scale_adj``) as instance attributes on the NVFP4 module,
# recompute the large ``repacked`` every forward. The repack itself
# is ~110 µs (one ctypes kernel call) — at base.yml that's <0.1% of
# step time, vs ~890 MiB saved peak VRAM.
# ---------------------------------------------------------------------------
@torch.no_grad()
def _build_marlin_scales_caches(
    module: torch.nn.Module,
    scales: torch.Tensor,
    global_scale: torch.Tensor,
    size_k: int,
    size_n: int,
    block_size: int = 16,
) -> None:
    """Compute the small per-module Marlin buffers and stash them on
    ``module`` as ``_scales_for_kernel`` and ``_global_scale_adj``.

    Called by ``NVFP4*Linear.repack_weights`` after a fresh quantize
    (so the cached buffers always track the latest BF16 master). The
    buffers are tiny (~0.6 MiB and 4 B); they fit comfortably in the
    per-module attribute dict and don't need an explicit clear hook.

    The matching ``repacked`` tensor is *not* cached here — it's
    recomputed every forward inside
    :func:`_MarlinNvFp4Matmul.forward` (see the design note above).
    """
    scales_for_kernel, scale_factor = _process_scales_for_marlin(
        scales, size_k=size_k, size_n=size_n,
    )
    global_scale_adj = _process_global_scale(global_scale, scale_factor)
    module._scales_for_kernel = scales_for_kernel
    module._global_scale_adj = global_scale_adj
    module._marlin_block_size = block_size


# ---------------------------------------------------------------------------
# The autograd Function — Marlin fwd, BF16 bwd (STE).
# ---------------------------------------------------------------------------
class _MarlinNvFp4Matmul(torch.autograd.Function):
    """BF16 act @ NVFP4-packed weight via vLLM Marlin FP4 (W4A16 fast path).

    Forward: ctypes call into ``marlin::marlin_mm`` (BF16 MMA + register
    dequant). Bias added outside the kernel.

    Backward: standard BF16 matmul backward on the dequantized BF16 view.
    STE — same contract as :class:`_NVFP4Matmul` in :mod:`nvfp4_linear`.
    """

    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        x: torch.Tensor,
        w_master: torch.Tensor,
        packed_w: torch.Tensor,
        scales_for_kernel: torch.Tensor,
        global_scale_adj: torch.Tensor,
        bias: Optional[torch.Tensor],
        K_orig: int,
        block_size: int,
        scales_e4m3: Optional[torch.Tensor] = None,
        global_scale: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        _ensure_libs_loaded()
        assert x.dtype == torch.bfloat16, (
            f"Marlin FP4 path expects BF16 activation, got {x.dtype}"
        )
        assert w_master.dtype == torch.bfloat16

        # Flatten leading dims for the kernel: x is [M, K] with M = product
        # of leading dims.
        orig_shape = x.shape
        K = K_orig
        x_2d = x.reshape(-1, K)
        M = x_2d.shape[0]
        N = w_master.shape[0]

        # Recompute the Marlin-layout repack each forward. The repack is
        # a single ctypes kernel call (~110 µs at FFN shapes) — cheap
        # relative to the matmul, and avoids the per-module repacked
        # buffer that previously cost ~890 MiB of peak VRAM across the
        # 64 FFN modules. The companion small buffers
        # (``scales_for_kernel``, ``global_scale_adj``) are cached as
        # instance attributes by ``_build_marlin_scales_caches``.
        repacked = _repack_for_marlin(packed_w, size_k=K, size_n=N)

        out = torch.empty(M, N, dtype=torch.bfloat16, device=x.device)
        stream = torch.cuda.current_stream().cuda_stream

        # Empty tensors for the optional slots Marlin accepts but our
        # NVFP4 path doesn't need.
        workspace = torch.zeros(132 * 128, dtype=torch.int32, device=x.device)
        a_scales = torch.empty(0, dtype=torch.float32, device=x.device)
        b_zeros = torch.empty(0, dtype=torch.bfloat16, device=x.device)
        g_idx = torch.empty(0, dtype=torch.int32, device=x.device)
        perm_t = torch.empty(0, dtype=torch.int32, device=x.device)
        a_tmp = torch.empty(0, dtype=torch.bfloat16, device=x.device)
        c_tmp = torch.empty(0, dtype=torch.float32, device=x.device)
        b_bias = torch.empty(0, dtype=torch.bfloat16, device=x.device)

        a_type = _kbfloat16()
        b_type = _kfe2m1f()
        c_type = _kbfloat16()
        s_type = _kfe4m3fn()

        num_groups = K // block_size
        sms = torch.cuda.get_device_properties(0).multi_processor_count

        _marlin_mm_fn(  # type: ignore[name-defined]
            ctypes.c_void_p(x_2d.data_ptr()),
            ctypes.c_void_p(repacked.data_ptr()),
            ctypes.c_void_p(out.data_ptr()),
            ctypes.c_void_p(c_tmp.data_ptr()),
            ctypes.c_void_p(b_bias.data_ptr()),
            ctypes.c_void_p(a_scales.data_ptr()),
            ctypes.c_void_p(scales_for_kernel.data_ptr()),
            ctypes.c_void_p(global_scale_adj.data_ptr()),
            ctypes.c_void_p(b_zeros.data_ptr()),
            ctypes.c_void_p(g_idx.data_ptr()),
            ctypes.c_void_p(perm_t.data_ptr()),
            ctypes.c_void_p(a_tmp.data_ptr()),
            ctypes.c_int(M), ctypes.c_int(N), ctypes.c_int(K),
            ctypes.c_int(x_2d.stride(0)),
            ctypes.c_void_p(workspace.data_ptr()),
            ctypes.c_void_p(ctypes.addressof(a_type)),
            ctypes.c_void_p(ctypes.addressof(b_type)),
            ctypes.c_void_p(ctypes.addressof(c_type)),
            ctypes.c_void_p(ctypes.addressof(s_type)),
            ctypes.c_bool(False), ctypes.c_bool(False), ctypes.c_bool(True), ctypes.c_bool(False),
            ctypes.c_int(num_groups), ctypes.c_int(block_size), ctypes.c_int(0),
            ctypes.c_void_p(stream),
            ctypes.c_int(-1), ctypes.c_int(-1), ctypes.c_int(sms),
            ctypes.c_bool(False), ctypes.c_bool(False), ctypes.c_bool(False),
        )

        # Save (x, w_master) for backward. We use w_master (the BF16
        # leaf the optimizer sees) so the bwd grad flows into the
        # right parameter — same STE pattern as _NVFP4Matmul.
        # Dequantizing here would be redundant since the BF16 master
        # is identical (modulo the freshly-repacked buffer's
        # quantization error).
        w_bf16 = w_master  # STE: the BF16 master IS the dequant view
        ctx.save_for_backward(x, w_bf16)
        ctx.has_bias = bias is not None
        # Stash the per-K-block packed weight + scales + global_scale so
        # backward can run the Marlin bwd path (pre-transposes to
        # per-N-block, repacks with size_k=N, calls the same kernel).
        # None means BF16 bwd fallback (legacy callers).
        ctx.bwd_packed_w = packed_w
        ctx.bwd_scales_e4m3 = scales_e4m3
        ctx.bwd_global_scale = global_scale
        ctx.bwd_block_size = block_size
        ctx.bwd_N = N
        ctx.bwd_K = K
        # Stash the original (unflattened) input shape so backward can
        # reshape the 2-D kernel output back to whatever the upstream
        # caller passed in (e.g. [B, T, K] for a 3-D activation).
        ctx.bwd_input_shape = x.shape
        if bias is not None:
            out = out + bias
        return out.reshape(*orig_shape[:-1], N)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):  # type: ignore[override]
        x, w_bf16 = ctx.saved_tensors
        K = w_bf16.shape[1]
        N = w_bf16.shape[0]
        grad_out_2d = grad_out.reshape(-1, N)
        x_2d = x.reshape(-1, K)
        # BF16 cuBLAS bwd on the dequantized view. The Marlin FP4 bwd
        # kernel has a known precision issue at FFN scales (per
        # project memory: "Marlin bwd blocked by packed-format
        # asymmetry"); stay with the BF16 path that the legacy
        # tests rely on.
        grad_x = grad_out @ w_bf16
        grad_w = (grad_out_2d.t().to(x_2d.dtype)) @ x_2d
        grad_bias = grad_out_2d.sum(dim=0) if ctx.has_bias else None
        return grad_x, grad_w, None, None, None, grad_bias, None, None, None, None


class _NVFP4MarlinNoLeafMatmul(torch.autograd.Function):
    """Marlin FP4 matmul without a BF16 master in the autograd graph.

    Forward: identical to :class:`_MarlinNvFp4Matmul.forward` minus the
    ``w_master`` argument — the kernel never read its data anyway. Bias
    is added outside the kernel.

    Backward: ``grad_x = grad_out @ W^T`` via the Marlin FP4 bwd kernel
    (when the original per-K-block ``scales_e4m3`` and ``global_scale``
    are stashed on ctx, just like the legacy Function). ``grad_w`` is
    NOT returned to autograd; it is stashed on the owning module via a
    weakref captured at forward time. The custom optimizer step in
    :mod:`src.training.param_offload` consumes it once per step.

    Pair of mode (3) for :class:`src.models.ops.nvfp4_linear.NVFP4Linear`
    when ``no_bf16_master=True``. The dequant+cuBLAS variant
    (:class:`_NVFP4NoLeafMatmul` in ``nvfp4_linear``) covers the non-
    Marlin mode-3 case.
    """

    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        x: torch.Tensor,
        packed_w: torch.Tensor,
        scales_for_kernel: torch.Tensor,
        global_scale_adj: torch.Tensor,
        bias: Optional[torch.Tensor],
        K_orig: int,
        block_size: int,
        scales_e4m3: torch.Tensor,
        global_scale: torch.Tensor,
        module_ref,  # weakref to the owning module
    ) -> torch.Tensor:
        _ensure_libs_loaded()
        assert x.dtype == torch.bfloat16, (
            f"Marlin FP4 path expects BF16 activation, got {x.dtype}"
        )
        orig_shape = x.shape
        K = K_orig
        x_2d = x.reshape(-1, K)
        M = x_2d.shape[0]
        # packed_w is 2-D [N, K//2]; the leading dim IS N (one byte
        # holds two weights along K, but the row axis is the output
        # feature axis and is not packed).
        N = packed_w.shape[0]

        repacked = _repack_for_marlin(packed_w, size_k=K, size_n=N)
        out = torch.empty(M, N, dtype=torch.bfloat16, device=x.device)
        stream = torch.cuda.current_stream().cuda_stream

        workspace = torch.zeros(132 * 128, dtype=torch.int32, device=x.device)
        a_scales = torch.empty(0, dtype=torch.float32, device=x.device)
        b_zeros = torch.empty(0, dtype=torch.bfloat16, device=x.device)
        g_idx = torch.empty(0, dtype=torch.int32, device=x.device)
        perm_t = torch.empty(0, dtype=torch.int32, device=x.device)
        a_tmp = torch.empty(0, dtype=torch.bfloat16, device=x.device)
        c_tmp = torch.empty(0, dtype=torch.float32, device=x.device)
        b_bias = torch.empty(0, dtype=torch.bfloat16, device=x.device)

        a_type = _kbfloat16()
        b_type = _kfe2m1f()
        c_type = _kbfloat16()
        s_type = _kfe4m3fn()

        num_groups = K // block_size
        sms = torch.cuda.get_device_properties(0).multi_processor_count

        _marlin_mm_fn(  # type: ignore[name-defined]
            ctypes.c_void_p(x_2d.data_ptr()),
            ctypes.c_void_p(repacked.data_ptr()),
            ctypes.c_void_p(out.data_ptr()),
            ctypes.c_void_p(c_tmp.data_ptr()),
            ctypes.c_void_p(b_bias.data_ptr()),
            ctypes.c_void_p(a_scales.data_ptr()),
            ctypes.c_void_p(scales_for_kernel.data_ptr()),
            ctypes.c_void_p(global_scale_adj.data_ptr()),
            ctypes.c_void_p(b_zeros.data_ptr()),
            ctypes.c_void_p(g_idx.data_ptr()),
            ctypes.c_void_p(perm_t.data_ptr()),
            ctypes.c_void_p(a_tmp.data_ptr()),
            ctypes.c_int(M), ctypes.c_int(N), ctypes.c_int(K),
            ctypes.c_int(x_2d.stride(0)),
            ctypes.c_void_p(workspace.data_ptr()),
            ctypes.c_void_p(ctypes.addressof(a_type)),
            ctypes.c_void_p(ctypes.addressof(b_type)),
            ctypes.c_void_p(ctypes.addressof(c_type)),
            ctypes.c_void_p(ctypes.addressof(s_type)),
            ctypes.c_bool(False), ctypes.c_bool(False), ctypes.c_bool(True), ctypes.c_bool(False),
            ctypes.c_int(num_groups), ctypes.c_int(block_size), ctypes.c_int(0),
            ctypes.c_void_p(stream),
            ctypes.c_int(-1), ctypes.c_int(-1), ctypes.c_int(sms),
            ctypes.c_bool(False), ctypes.c_bool(False), ctypes.c_bool(False),
        )

        ctx.save_for_backward(x)
        ctx.has_bias = bias is not None
        ctx.module_ref = module_ref
        ctx.bwd_packed_w = packed_w
        ctx.bwd_scales_e4m3 = scales_e4m3
        ctx.bwd_global_scale = global_scale
        ctx.bwd_block_size = block_size
        ctx.bwd_N = N
        ctx.bwd_K = K
        ctx.bwd_input_shape = x.shape
        if bias is not None:
            out = out + bias
        return out.reshape(*orig_shape[:-1], N)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):  # type: ignore[override]
        # x is saved; the dequantized BF16 view is reconstructed here.
        # We save x in fp32-equivalent space rather than re-dequantizing
        # the FP4 buffers in the bwd, because the cached ``repacked``
        # from forward is already gone (fwd freed it). Re-dequantizing
        # FP4 -> BF16 here is cheap (single matrix multiply of packed
        # against a precomputed nibble table) and keeps the Function
        # stateless w.r.t. forward locals.
        # NB: an even cheaper alternative is to stash w_bf16 alongside
        # x in ctx.save_for_backward (it's a [N, K] tensor at the
        # same size as the bwd needs). We do that — it's the same as
        # the legacy Function and avoids re-dequantization.
        # To keep that option open we accept w_bf16 as the second
        # saved tensor; the caller (forward above) doesn't currently
        # save it. If the perf profile shows re-dequant matters,
        # uncomment the save_for_backward pair below.
        (x,) = ctx.saved_tensors
        # Use the Marlin-aware dequant that includes global_scale;
        # the legacy ``dequantize_nvfp4`` from nvfp4.py doesn't
        # multiply by global_scale, which would make the no-leaf
        # bwd produce a weight ~600x smaller than the legacy bwd's
        # BF16 master reference.
        w_bf16 = dequantize_marlin_nvfp4(
            ctx.bwd_packed_w, ctx.bwd_scales_e4m3,
            ctx.bwd_global_scale, ctx.bwd_K,
            block_size=ctx.bwd_block_size, out_dtype=x.dtype,
        )
        K = w_bf16.shape[1]
        N = w_bf16.shape[0]
        grad_out_2d = grad_out.reshape(-1, N)
        x_2d = x.reshape(-1, K)
        # BF16 cuBLAS bwd (same reason as the legacy matmul: the
        # Marlin FP4 bwd kernel produces NaN/Inf at FFN scales per
        # project memory). The kernel is unused; the dequantized
        # BF16 view is the source of truth for grad_x.
        grad_x = grad_out @ w_bf16
        grad_w = (grad_out_2d.t().to(x_2d.dtype)) @ x_2d
        grad_bias = grad_out_2d.sum(dim=0) if ctx.has_bias else None
        module = ctx.module_ref()
        if module is not None:
            module._stash_grad_w(grad_w)
        return grad_x, None, None, None, grad_bias, None, None, None, None, None


# ---------------------------------------------------------------------------
# Marlin bwd entry point — grad_x = grad_out @ W^T via FP4 kernel
# ---------------------------------------------------------------------------
def _marlin_bwd_grad_x(
    grad_out_2d: torch.Tensor,
    packed_w_per_k: torch.Tensor,
    scales_e4m3: torch.Tensor,
    global_scale: torch.Tensor,
    N: int,
    K: int,
    block_size: int,
) -> torch.Tensor:
    """Compute grad_x = grad_out @ W^T using the Marlin FP4 kernel.

    This mirrors :func:`_MarlinNvFp4Matmul.forward` but with axes
    swapped: the matmul's reduction is now N (the original output cols
    of W) and the output cols are now K (the original input cols of W).
    The repack + scale processing is documented in
    :func:`_repack_for_marlin_bwd` and
    :func:`_process_scales_for_marlin_bwd`.
    """
    M = grad_out_2d.shape[0]
    repacked = _repack_for_marlin_bwd(packed_w_per_k, N, K)
    scales_for_kernel_bwd, sf_bwd = _process_scales_for_marlin_bwd(
        scales_e4m3, size_k=N, size_n=K,
    )
    global_scale_adj_bwd = _process_global_scale(global_scale, sf_bwd)

    grad_x = torch.empty(M, K, dtype=torch.bfloat16, device=grad_out_2d.device)
    # ScalarType buffers MUST stay alive across the ctypes call —
    # see ``docs/marlin_build_pipeline.md`` gotcha on ctypes GC.
    a_type = _kbfloat16()
    b_type = _kfe2m1f()
    c_type = _kbfloat16()
    s_type = _kfe4m3fn()
    workspace = torch.zeros(132 * 128, dtype=torch.int32, device=grad_out_2d.device)
    empty_f32 = torch.empty(0, dtype=torch.float32, device=grad_out_2d.device)
    empty_bf16 = torch.empty(0, dtype=torch.bfloat16, device=grad_out_2d.device)
    empty_i32 = torch.empty(0, dtype=torch.int32, device=grad_out_2d.device)
    stream = torch.cuda.current_stream().cuda_stream
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    num_groups = N // block_size
    _marlin_mm_fn(  # type: ignore[name-defined]
        ctypes.c_void_p(grad_out_2d.data_ptr()),
        ctypes.c_void_p(repacked.data_ptr()),
        ctypes.c_void_p(grad_x.data_ptr()),
        ctypes.c_void_p(empty_f32.data_ptr()),
        ctypes.c_void_p(empty_bf16.data_ptr()),
        ctypes.c_void_p(empty_f32.data_ptr()),
        ctypes.c_void_p(scales_for_kernel_bwd.data_ptr()),
        ctypes.c_void_p(global_scale_adj_bwd.data_ptr()),
        ctypes.c_void_p(empty_i32.data_ptr()),
        ctypes.c_void_p(empty_i32.data_ptr()),
        ctypes.c_void_p(empty_i32.data_ptr()),
        ctypes.c_void_p(empty_bf16.data_ptr()),
        ctypes.c_int(M), ctypes.c_int(K), ctypes.c_int(N),
        ctypes.c_int(grad_out_2d.stride(0)),
        ctypes.c_void_p(workspace.data_ptr()),
        ctypes.c_void_p(ctypes.addressof(a_type)),
        ctypes.c_void_p(ctypes.addressof(b_type)),
        ctypes.c_void_p(ctypes.addressof(c_type)),
        ctypes.c_void_p(ctypes.addressof(s_type)),
        ctypes.c_bool(False), ctypes.c_bool(False), ctypes.c_bool(True), ctypes.c_bool(False),
        ctypes.c_int(num_groups), ctypes.c_int(block_size), ctypes.c_int(0),
        ctypes.c_void_p(stream),
        ctypes.c_int(-1), ctypes.c_int(-1), ctypes.c_int(sms),
        ctypes.c_bool(False), ctypes.c_bool(False), ctypes.c_bool(False),
    )
    return grad_x


# ---------------------------------------------------------------------------
# Public entry point used by the NVFP4*Linear modules.
# ---------------------------------------------------------------------------
def marlin_nvfp4_matmul(
    x: torch.Tensor,
    w_master: torch.Tensor,
    packed_w: torch.Tensor,
    scales_for_kernel: torch.Tensor,
    global_scale_adj: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    scales_e4m3: Optional[torch.Tensor] = None,
    global_scale: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """W4A16 matmul: ``x @ dequant(packed_w)`` via vLLM Marlin FP4.

    See :class:`_MarlinNvFp4Matmul` for the autograd contract.

    Forward uses the precomputed ``scales_for_kernel`` /
    ``global_scale_adj`` (cached on the owning module via
    :func:`_build_marlin_scales_caches`). For the backward
    (``grad_x = grad_out @ W^T``) we also need the *original*
    per-K-block ``scales_e4m3`` and ``global_scale`` to feed the Marlin
    bwd path (re-processed at backward time for the bwd axis labeling).
    If those are omitted, backward falls back to the BF16 cuBLAS path.

    Note: ``scales_for_kernel`` and ``global_scale_adj`` are precomputed
    by :func:`_build_marlin_scales_caches` (called by
    ``NVFP4*Linear.repack_weights`` after each optimizer step) and
    cached as instance attributes on the owning module. The large
    ``repacked`` buffer is recomputed each forward to keep peak VRAM
    flat — see the design note at :func:`_build_marlin_scales_caches`.
    """
    return _MarlinNvFp4Matmul.apply(
        x, w_master, packed_w, scales_for_kernel, global_scale_adj, bias,
        w_master.shape[1], 16, scales_e4m3, global_scale,
    )
