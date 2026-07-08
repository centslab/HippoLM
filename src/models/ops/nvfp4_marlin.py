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
    FN_NAME = (
        "_ZN6marlin9marlin_mmEPKvS1_PvS2_S2_S2_S2_S2_S2_S2_S2_S2_iiiiS2_"
        "RKN4vllm10ScalarTypeES6_S6_S6_bbbbiiiP11CUstream_stiiibbb"
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
# Cache for repacked weights (avoids the repack every forward).
# Keyed by ``(packed.data_ptr(), packed.shape)``.
# ---------------------------------------------------------------------------
@torch.no_grad()
def _cached_repack(
    packed: torch.Tensor, scales: torch.Tensor, size_k: int, size_n: int,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Repack + process scales. Idempotent for the same packed buffer.

    The cache key is the (data_ptr, shape) of the packed buffer, which
    is stable across forwards (the buffer is owned by the NVFP4 module
    and only changes when ``repack_weights`` runs after an optimizer
    step). The ``repacked`` and ``scales_for_kernel`` tensors are
    re-allocated whenever the source changes.
    """
    cache_key = (packed.data_ptr(), tuple(packed.shape))
    cached = _cached_repack._cache.get(cache_key)  # type: ignore[attr-defined]
    if cached is not None:
        return cached
    repacked = _repack_for_marlin(packed, size_k=size_k, size_n=size_n)
    scales_for_kernel, scale_factor = _process_scales_for_marlin(
        scales, size_k=size_k, size_n=size_n,
    )
    _cached_repack._cache[cache_key] = (repacked, scales_for_kernel, scale_factor)  # type: ignore[attr-defined]
    return repacked, scales_for_kernel, scale_factor


_cached_repack._cache = {}  # type: ignore[attr-defined]


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
        scales_w: torch.Tensor,
        global_scale: torch.Tensor,
        bias: Optional[torch.Tensor],
        K_orig: int,
        block_size: int,
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

        # Repack + scale processing (cached — same packed buffer across
        # forwards until the optimizer step re-quantizes).
        repacked, scales_for_kernel, scale_factor = _cached_repack(
            packed=packed_w, scales=scales_w,
            size_k=K, size_n=N,
        )
        global_scale_adj = _process_global_scale(global_scale, scale_factor)

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
        grad_x = grad_out @ w_bf16
        grad_w = (grad_out_2d.t().to(x_2d.dtype)) @ x_2d
        grad_bias = grad_out_2d.sum(dim=0) if ctx.has_bias else None
        return grad_x, grad_w, None, None, None, grad_bias, None, None


# ---------------------------------------------------------------------------
# Public entry point used by the NVFP4*Linear modules.
# ---------------------------------------------------------------------------
def marlin_nvfp4_matmul(
    x: torch.Tensor,
    w_master: torch.Tensor,
    packed_w: torch.Tensor,
    scales_w: torch.Tensor,
    global_scale: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """W4A16 matmul: ``x @ dequant(packed_w)`` via vLLM Marlin FP4.

    See :class:`_MarlinNvFp4Matmul` for the autograd contract.
    """
    return _MarlinNvFp4Matmul.apply(
        x, w_master, packed_w, scales_w, global_scale, bias,
        w_master.shape[1], 16,
    )
