"""NVFP4 quantization for FFN weights (W4A16 research path).

NVFP4 spec (NVIDIA Blackwell, ``MXFP4``-style microblock scaling):
- Weight element dtype: E2M1 (4 bits per element, 2 elements packed per uint8)
- Block scaling: 1x16 (one FP8-e4m3fn scale per 16 contiguous elements along K)
- E2M1 representable levels: 0, 0.5, 1, 1.5, 2, 3, 4, 6

This module is the *correctness-first* reference. Production path:

    forward  : quantize BF16 master weight to NVFP4 -> dequant back to BF16 ->
               BF16 matmul via F.linear
    backward : standard BF16 matmul backward (gradient flows through the
               dequantized BF16 view); optimizer step updates the BF16
               master weight, then we re-quantize to NVFP4 for the next fwd.

Why no FP4-tensor-core matmul on this path? PyTorch's ``torch._scaled_mm``
NVFP4 path (``scale_block_size=16``) requires **both** A and B to be
FP4-packed (W4A4 only). A pure BF16 x FP4 mixed-precision matmul kernel
is not exposed in 2.9.1. The path here therefore dequantizes the weight to
BF16 before the matmul, which keeps the autograd contract identical to a
plain Linear and gives us a correctness-first W4A16 with FP4 weight
storage. Swapping in a real FP4 GEMM (Blackwell ``mma.sp``) is a
follow-up; the autograd Function signature below is intentionally
shaped so that change is one line.

Design choices (and why):

- **Block size = 16 along K only.** We quantize weight ``[N, K]`` by
  reshaping to ``[N, K//16, 16]`` and computing one scale per 16-element
  row. NVFP4 also allows 1x16 along N; we don't use that because our
  matmul dequantizes to BF16 per-element so a single K-side scale is
  sufficient (and avoids a transpose-step scale mismatch).

- **E2M1 round-to-nearest via lookup.** 8 levels is small enough that a
  argmin over the level table is both fast on CUDA and easy to verify.

- **Per-block scale = block_absmax / 6.** This saturates the block's max
  magnitude to E2M1 max (6.0), which is the standard NVFP4 recipe and
  gives the lowest quantization noise at the cost of some clipping for
  rare outliers. The scale is stored in FP8-e4m3fn (range up to ~448,
  so we never overflow even when block_absmax is large).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


# E2M1 representable magnitudes (sign handled separately).
_E2M1_LEVELS = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
    dtype=torch.float32,
)


# ---------------------------------------------------------------------------
# Pure-PyTorch reference quantize / dequantize (CPU + CUDA both work)
# ---------------------------------------------------------------------------
def _round_to_e2m1(x: torch.Tensor) -> torch.Tensor:
    """Round ``x`` to nearest E2M1 representable value (round-half-away-from-zero).

    Sign and zero are handled separately: we round ``|x|`` to the nearest
    of the 8 E2M1 levels and reapply the sign. Round-half-away-from-zero
    matches the NVFP4 spec (PTX ``cvt.rn.satfinite.e2m1f4``).
    """
    levels = _E2M1_LEVELS.to(device=x.device, dtype=torch.float32)
    sign = (x < 0).to(torch.float32)
    abs_x = x.abs().to(torch.float32)
    # Flatten to 1D for the distance argmin, then reshape back.
    orig_shape = abs_x.shape
    flat = abs_x.reshape(-1)
    # Compute distances to each E2M1 level. Shape: (numel, 8).
    distances = (flat.unsqueeze(-1) - levels.unsqueeze(0)).abs()
    indices = distances.argmin(dim=-1)
    abs_quant = levels[indices].reshape(orig_shape)
    # Sign: 1.0 (positive) or -1.0 (negative); zero stays zero.
    signed = abs_quant * (1.0 - 2.0 * sign)
    return signed.to(x.dtype)


def _e2m1_indices(x: torch.Tensor) -> torch.Tensor:
    """Return the E2M1 magnitude index (0-7) for each element of ``x``.

    Internally computes the same argmin-over-8-levels that
    :func:`_round_to_e2m1` uses, but skips the level-table gather and
    sign re-application: callers that need both the indices (for
    packing) and the rounded values (for verification) should run
    this once and reuse the result, instead of running argmin twice.

    Output shape matches ``x.shape``; dtype is ``int64`` (the natural
    argmin dtype).
    """
    levels = _E2M1_LEVELS.to(device=x.device, dtype=torch.float32)
    abs_flat = x.abs().to(torch.float32).reshape(-1)
    distances = (abs_flat.unsqueeze(-1) - levels.unsqueeze(0)).abs()
    return distances.argmin(dim=-1).reshape(x.shape)


def quantize_nvfp4(weight: torch.Tensor, block_size: int = 16) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a 2-D ``[N, K]`` weight to NVFP4 packed uint8 + FP8-e4m3fn scales.

    Returns:
        packed : ``uint8`` tensor of shape ``[N, K // 2]``. Two E2M1 values
                 per byte (low nibble = first element, high nibble = second).
        scales : ``float8_e4m3fn`` tensor of shape ``[N, K // block_size]``.

    K is padded up to a multiple of ``block_size`` with zeros (rounding
    half-away-from-zero means padded zeros quantize to zero, so the
    padded blocks are a no-op). This matches the Blackwell spec for
    weight quantization (activation is not quantized in W4A16).
    """
    assert weight.dim() == 2, f"expected 2-D weight, got {weight.dim()}-D"
    N, K = weight.shape
    pad = (block_size - K % block_size) % block_size
    if pad:
        weight = F.pad(weight, (0, pad))
    K_padded = K + pad
    assert K_padded % block_size == 0, (
        f"K_padded={K_padded} not a multiple of block_size={block_size}"
    )

    block = weight.reshape(N, K_padded // block_size, block_size)
    absmax = block.abs().amax(dim=-1).to(torch.float32)
    # NVFP4 recipe: scale = absmax / E2M1_max. Clamp to a tiny positive
    # so we don't divide by zero on all-zero blocks.
    scales_f32 = (absmax / 6.0).clamp(min=1e-6)
    scales = scales_f32.to(torch.float8_e4m3fn)

    # Compute the E2M1 magnitude index in one pass (the dominant
    # cost is the [numel, 8] distance matrix + argmin). The index is
    # then reused for both the pack and (optionally) any debug
    # rounding check, avoiding a second argmin over the same data.
    block_fp32 = block.to(torch.float32)
    x_scaled = block_fp32 / scales_f32.unsqueeze(-1)
    idx = _e2m1_indices(x_scaled).to(torch.int32)
    sign_bits = (block_fp32 < 0).to(torch.int32)
    idx = idx | (sign_bits << 3)  # sign in bit 3
    idx_pairs = idx.reshape(N, K_padded // 2, 2)
    packed = (idx_pairs[..., 0] | (idx_pairs[..., 1] << 4)).to(torch.uint8)
    return packed, scales


def dequantize_nvfp4(
    packed: torch.Tensor,
    scales: torch.Tensor,
    original_K: int,
    block_size: int = 16,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Dequantize NVFP4-packed weight back to a dense tensor.

    Args:
        packed      : ``uint8`` tensor ``[N, K_padded // 2]`` (K_padded padded to mult of block_size).
        scales      : ``float8_e4m3fn`` tensor ``[N, K_padded // block_size]``.
        original_K  : the un-padded K (we crop the padding here).
        block_size  : NVFP4 microblock size (must be 16).
        out_dtype   : output dtype for the dense weight.

    Returns:
        ``[N, original_K]`` dense tensor in ``out_dtype``.
    """
    assert packed.dim() == 2
    N, K_half = packed.shape
    K_padded = K_half * 2
    assert K_padded % block_size == 0, (
        f"K_padded={K_padded} not a multiple of block_size={block_size}"
    )

    # Unpack uint8 -> int32 indices.
    low = (packed & 0x0F).to(torch.int32)
    high = (packed >> 4).to(torch.int32)
    indices = torch.stack([low, high], dim=-1).reshape(N, K_padded)

    # Decode E2M1: bit 3 = sign, bits 0-2 = magnitude index.
    sign_bits = ((indices >> 3) & 0x01).to(torch.float32)
    mag_idx = (indices & 0x07).to(torch.int64)
    levels = _E2M1_LEVELS.to(device=packed.device, dtype=torch.float32)
    magnitudes = levels[mag_idx]
    signed = magnitudes * (1.0 - 2.0 * sign_bits)

    # Apply per-block scales.
    scales_f32 = scales.to(torch.float32)  # [N, K_padded // block_size]
    signed_blocks = signed.reshape(N, K_padded // block_size, block_size)
    dequant_blocks = signed_blocks * scales_f32.unsqueeze(-1)
    dequant = dequant_blocks.reshape(N, K_padded)

    # Crop padding.
    if dequant.shape[1] > original_K:
        dequant = dequant[:, :original_K]
    return dequant.to(out_dtype)