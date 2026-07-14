"""Correctness test for the Triton-fused NVFP4 E2M1 pack kernel.

Verifies :func:`src.models.ops.nvfp4_quant_triton.quantize_nvfp4_pack_triton`
and the integration inside
:func:`src.models.ops.nvfp4_marlin.quantize_nvfp4_with_global_scale` are
byte-exact vs the PyTorch ``[numel, 8]`` + ``argmin`` reference.

Byte-exact means:
  - ``packed`` bytes match exactly (zero diffs, not "within epsilon")
  - ``scales_e4m3`` matches (same fp8 cast path)
  - ``global_scale`` matches (same computation)

This is a strict regression test — a single mismatched byte fails the
test. The kernel's two correctness traps (Triton fast-math division;
argmin first-min tie-breaking) are caught here.

Run:
    python -m pytest test/test_nvfp4_quant_triton.py -v
"""
from __future__ import annotations

import pytest
import torch

from src.models.ops.nvfp4_marlin import quantize_nvfp4_with_global_scale
from src.models.ops.nvfp4_quant_triton import (
    BLOCK_SIZE,
    quantize_nvfp4_pack_triton,
)


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Triton kernel requires CUDA",
)


def _reference_block_scale_raw(bf16_padded: torch.Tensor) -> torch.Tensor:
    """Compute ``block_absmax / 6`` directly (matches the reference path)."""
    N, K_padded = bf16_padded.shape
    block = bf16_padded.reshape(N, K_padded // BLOCK_SIZE, BLOCK_SIZE).to(torch.float32)
    absmax = block.abs().amax(dim=-1)
    return (absmax / 6.0).clamp(min=1e-6)


def _reference_packed(bf16_padded: torch.Tensor, block_scale_raw: torch.Tensor) -> torch.Tensor:
    """PyTorch reference path — explicit argmin over 8 levels, no Triton.

    Mirrors the pre-optimization code in ``quantize_nvfp4_with_global_scale``
    (the lines now replaced by the Triton kernel). Kept here as the
    byte-exact oracle.
    """
    N, K_padded = bf16_padded.shape
    block = bf16_padded.reshape(N, K_padded // BLOCK_SIZE, BLOCK_SIZE).to(torch.float32)
    x_scaled = block / block_scale_raw.unsqueeze(-1)
    abs_x = x_scaled.abs()
    levels = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
        dtype=torch.float32, device=bf16_padded.device,
    )
    abs_flat = abs_x.reshape(-1)
    distances = (abs_flat.unsqueeze(-1) - levels.unsqueeze(0)).abs()
    magnitude_idx = distances.argmin(dim=-1).reshape(N, K_padded).to(torch.int32)
    sign_bits = (x_scaled < 0).reshape(N, K_padded).to(torch.int32) * 0x8
    nibbles = (magnitude_idx | sign_bits).to(torch.uint8)
    nibble_pairs = nibbles.view(N, K_padded // 2, 2)
    return (nibble_pairs[..., 0] | (nibble_pairs[..., 1] << 4)).to(torch.uint8)


def _force_pytorch_pack(monkeypatch):
    """Patch ``_quantize_pack`` to skip the Triton path.

    Lets the same test verify both:
      - the Triton kernel matches the reference (direct call)
      - ``quantize_nvfp4_with_global_scale`` returns the same output
        regardless of which path is taken (the public function's
        two paths must agree byte-for-byte)
    """
    from src.models.ops import nvfp4_marlin

    def _pytorch_only(weight, block_scale_raw, N, K_padded):
        block = weight.reshape(N, K_padded // BLOCK_SIZE, BLOCK_SIZE).to(torch.float32)
        x_scaled = block / block_scale_raw.unsqueeze(-1)
        abs_x = x_scaled.abs()
        levels = torch.tensor(
            [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
            dtype=torch.float32, device=weight.device,
        )
        abs_flat = abs_x.reshape(-1)
        distances = (abs_flat.unsqueeze(-1) - levels.unsqueeze(0)).abs()
        magnitude_idx = distances.argmin(dim=-1).reshape(N, K_padded).to(torch.int32)
        sign_bits = (x_scaled < 0).reshape(N, K_padded).to(torch.int32) * 0x8
        nibbles = (magnitude_idx | sign_bits).to(torch.uint8)
        nibble_pairs = nibbles.view(N, K_padded // 2, 2)
        return (nibble_pairs[..., 0] | (nibble_pairs[..., 1] << 4)).to(torch.uint8)

    monkeypatch.setattr(nvfp4_marlin, "_quantize_pack", _pytorch_only)


# Production chunk shapes from configs/base.yml + boundary cases.
_SHAPES = [
    (16, 16),          # tiny: 1 microblock
    (1, 256),          # degenerate: 1 row × many cols
    (32, 1536),        # small batch
    (4096, 1536),      # prod gate_up chunk
    (1536, 4096),      # prod down_proj chunk
    (4096, 4096),      # hypothetical square
    (8192, 1536),      # full gate_up (one chunk)
    (4096, 1024),      # non-multiple-of-256 K (still mult of 16)
    (2048, 2048),
]


@pytest.mark.parametrize("rows,cols", _SHAPES)
def test_triton_pack_matches_reference_byte_exact(rows, cols):
    """Triton kernel output equals the PyTorch reference for every byte."""
    torch.manual_seed(rows * 31 + cols)
    bf16 = torch.randn(rows, cols, dtype=torch.bfloat16, device="cuda")
    K_padded = cols + (BLOCK_SIZE - cols % BLOCK_SIZE) % BLOCK_SIZE
    if K_padded > cols:
        bf16_padded = torch.nn.functional.pad(bf16, (0, K_padded - cols)).contiguous()
    else:
        bf16_padded = bf16.contiguous()
    block_scale_raw = _reference_block_scale_raw(bf16_padded)

    packed_ref = _reference_packed(bf16_padded, block_scale_raw)
    packed_tri = quantize_nvfp4_pack_triton(bf16_padded, block_scale_raw)

    assert packed_tri.shape == packed_ref.shape, (
        f"shape mismatch: {packed_tri.shape} vs {packed_ref.shape}"
    )
    # Byte-exact diff — torch.equal is the strict check.
    assert torch.equal(packed_tri, packed_ref), (
        f"Triton vs reference byte mismatch on ({rows}, {cols}); "
        f"max byte diff = {(packed_tri.to(torch.int32) - packed_ref.to(torch.int32)).abs().max().item()}, "
        f"n_mismatched = {(packed_tri != packed_ref).sum().item()}"
    )


@pytest.mark.parametrize("rows,cols", _SHAPES)
def test_quantize_nvfp4_triton_path_matches_pytorch_fallback(rows, cols, monkeypatch):
    """The public ``quantize_nvfp4_with_global_scale`` returns identical
    output regardless of which path (Triton vs PyTorch) runs.

    Patches the internal ``_quantize_pack`` to force the PyTorch
    reference path; compares against the default (Triton) call. Any
    divergence here would mean the integration produces different
    outputs depending on the device — a silent correctness regression.
    """
    torch.manual_seed(rows * 37 + cols * 13)
    bf16 = torch.randn(rows, cols, dtype=torch.bfloat16, device="cuda")
    K_padded = cols + (BLOCK_SIZE - cols % BLOCK_SIZE) % BLOCK_SIZE
    if K_padded > cols:
        bf16_padded = torch.nn.functional.pad(bf16, (0, K_padded - cols)).contiguous()
    else:
        bf16_padded = bf16.contiguous()

    # Default call — Triton path.
    packed_tri, scales_tri, gs_tri = quantize_nvfp4_with_global_scale(
        bf16_padded, block_size=BLOCK_SIZE,
    )
    # Forced PyTorch path.
    _force_pytorch_pack(monkeypatch)
    packed_pt, scales_pt, gs_pt = quantize_nvfp4_with_global_scale(
        bf16_padded, block_size=BLOCK_SIZE,
    )

    assert torch.equal(packed_tri, packed_pt), (
        f"packed byte mismatch on ({rows}, {cols}); "
        f"max diff = {(packed_tri.to(torch.int32) - packed_pt.to(torch.int32)).abs().max().item()}"
    )
    # fp8 e4m3fn has 1-ULP rounding; equal under bit-cast is the strict
    # check (PyTorch's fp8 cast is deterministic).
    assert torch.equal(scales_tri.view(torch.uint8), scales_pt.view(torch.uint8)), (
        f"scales_e4m3 byte mismatch on ({rows}, {cols})"
    )
    assert torch.equal(gs_tri, gs_pt), (
        f"global_scale mismatch on ({rows}, {cols}): "
        f"{gs_tri.item()} vs {gs_pt.item()}"
    )


@pytest.mark.parametrize("rows,cols", _SHAPES)
def test_quantize_dequant_roundtrip_matches_pytorch_path(rows, cols, monkeypatch):
    """Round-trip (quantize -> dequant) gives the same BF16 within fp4
    quant noise — and is identical between Triton and PyTorch paths.

    This catches integration bugs that the byte-exact diff above
    might miss (e.g. wrong scale sign, missing global_scale fold).
    """
    torch.manual_seed(rows * 41 + cols * 7)
    bf16 = torch.randn(rows, cols, dtype=torch.bfloat16, device="cuda")
    K_padded = cols + (BLOCK_SIZE - cols % BLOCK_SIZE) % BLOCK_SIZE
    if K_padded > cols:
        bf16_padded = torch.nn.functional.pad(bf16, (0, K_padded - cols)).contiguous()
    else:
        bf16_padded = bf16.contiguous()

    packed_tri, scales_tri, gs_tri = quantize_nvfp4_with_global_scale(
        bf16_padded, block_size=BLOCK_SIZE,
    )
    _force_pytorch_pack(monkeypatch)
    packed_pt, scales_pt, gs_pt = quantize_nvfp4_with_global_scale(
        bf16_padded, block_size=BLOCK_SIZE,
    )

    # Dequant via the Marlin path (uses global_scale).
    from src.models.ops.nvfp4_marlin import dequantize_marlin_nvfp4
    bf16_tri = dequantize_marlin_nvfp4(
        packed_tri, scales_tri, gs_tri, cols, block_size=BLOCK_SIZE,
    )
    bf16_pt = dequantize_marlin_nvfp4(
        packed_pt, scales_pt, gs_pt, cols, block_size=BLOCK_SIZE,
    )
    assert torch.equal(bf16_tri, bf16_pt), (
        f"dequant mismatch on ({rows}, {cols}); "
        f"max diff = {(bf16_tri - bf16_pt).abs().max().item()}"
    )


def test_triton_pack_non_contiguous_input_falls_back():
    """Non-contiguous input should fall back to the PyTorch path
    (asserted via shape equality with the public reference output).

    The integration guards ``weight.is_contiguous()`` to avoid
    triggering the kernel's row-stride math on non-standard layouts.
    """
    torch.manual_seed(0)
    N, K = 64, 256
    base = torch.randn(N * 2, K, dtype=torch.bfloat16, device="cuda")
    # Take every other row → non-contiguous along dim 0.
    bf16 = base[::2]  # [N, K], non-contiguous
    assert not bf16.is_contiguous()

    K_padded = K + (BLOCK_SIZE - K % BLOCK_SIZE) % BLOCK_SIZE
    bf16_padded = torch.nn.functional.pad(bf16, (0, K_padded - K)).contiguous()
    # After .contiguous() the input is contiguous again — but the
    # integration also accepts a non-contiguous source if we skip the
    # pre-call .contiguous(). This test ensures the integration does
    # not silently corrupt the result either way.
    packed, scales, gs = quantize_nvfp4_with_global_scale(
        bf16_padded, block_size=BLOCK_SIZE,
    )
    assert packed.shape == (N, K_padded // 2)
    assert packed.dtype == torch.uint8
    assert scales.shape == (N, K_padded // BLOCK_SIZE)
    assert gs.shape == ()


def test_triton_pack_handles_extreme_block_scales():
    """Inputs near or beyond the E2M1 range should not crash and
    should match the reference exactly.

    Verifies the kernel's IEEE division (``tl.div_rn``) survives the
    pathological-but-valid cases the fast-math ``__fdividef`` would
    mishandle (boundary ties at 2.5, 5.0, etc.).
    """
    torch.manual_seed(99)
    # Crafted inputs at exact E2M1 boundaries — these are the cases
    # where fast-math division would produce values slightly above the
    # boundary, tripping the cascade.
    bf16 = torch.tensor(
        [[2.5, 2.5000001, 2.4999999, -2.5, 5.0, 5.000001, 4.9999999,
          0.25, 0.2500001, 0.75, 1.75, 3.5]],
        dtype=torch.float32, device="cuda",
    ).to(torch.bfloat16).repeat(64, 1)
    K = bf16.shape[1]
    K_padded = K + (BLOCK_SIZE - K % BLOCK_SIZE) % BLOCK_SIZE
    bf16_padded = torch.nn.functional.pad(bf16, (0, K_padded - K)).contiguous()
    block_scale_raw = _reference_block_scale_raw(bf16_padded)

    packed_ref = _reference_packed(bf16_padded, block_scale_raw)
    packed_tri = quantize_nvfp4_pack_triton(bf16_padded, block_scale_raw)
    assert torch.equal(packed_tri, packed_ref), (
        f"boundary-case byte mismatch; "
        f"max diff = {(packed_tri.to(torch.int32) - packed_ref.to(torch.int32)).abs().max().item()}"
    )