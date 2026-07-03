"""Public Python API for the mxfp8 fused per-mb accumulation kernel.

The kernel itself (C++ source) lives in ``kernel.cpp`` and is
JIT-compiled on first import via :func:`get_module`. The two
exported functions are:

``fused_mxfp8_dequant_add_requant(mom_buf, mom_scale, grad, rows, cols_p, block_size)``
    In-place: dequantize ``mom_buf`` (E4M3) by ``mom_scale`` (E8M0),
    add ``grad`` (BF16) per element, requantize to the same
    E4M3+E8M0 storage. ``rows * cols_p`` is the padded storage
    shape (``cols`` rounded up to a multiple of ``block_size``,
    default 32). This is the per-microb batch hot path in
    :class:`CPUMuon`.

``requantize_mxfp8(mom_buf, mom_scale, rows, cols_p, block_size)``
    Requantize-only path (no add). Used at :meth:`CPUMuon.step`
    time to refresh the stored mom_buf from the dequantized
    form (after NS on the GPU has updated the cycle's momentum
    values in BF16/FP16).

The BF16/E4M3/E8M0 dtypes are required and asserted in C++.
"""
from __future__ import annotations

from typing import Tuple

import torch

from .load_inline import get_module


# Eager compile on first import. The build is cached at
# ``~/.cache/torch_extensions/mxfp8_accum/`` so subsequent
# imports are <100ms (cache hit).
_module = None


def _ensure_module():
    global _module
    if _module is None:
        _module = get_module()
    return _module


# Default block size for the project (set in TensorPrecision's
# __post_init__ when dtype=mxfp8). Hard-coded in the C++ kernel
# for now; widening support is straightforward but unneeded.
_DEFAULT_BLOCK_SIZE = 32


def _validate_inputs(
    mom_buf: torch.Tensor,
    mom_scale: torch.Tensor,
    grad: torch.Tensor,
    rows: int,
    cols_p: int,
    block_size: int,
    grad_required: bool = True,
) -> None:
    """Shape + dtype + device checks. Mirrors the C++ TORCH_CHECKs."""
    assert mom_buf.is_cpu and mom_scale.is_cpu, "tensors must be on CPU"
    if grad_required:
        assert grad.is_cpu, "grad must be on CPU"
    assert mom_buf.dtype == torch.float8_e4m3fn, \
        f"mom_buf must be float8_e4m3fn, got {mom_buf.dtype}"
    assert mom_scale.dtype == torch.float8_e8m0fnu, \
        f"mom_scale must be float8_e8m0fnu, got {mom_scale.dtype}"
    if grad_required:
        assert grad.dtype == torch.bfloat16, \
            f"grad must be bfloat16, got {grad.dtype}"
    assert block_size == _DEFAULT_BLOCK_SIZE, \
        f"block_size must be {_DEFAULT_BLOCK_SIZE} in this build"


def fused_mxfp8_dequant_add_requant(
    mom_buf: torch.Tensor,
    mom_scale: torch.Tensor,
    grad: torch.Tensor,
    rows: int,
    cols: int,
    block_size: int = _DEFAULT_BLOCK_SIZE,
) -> None:
    """In-place fused dequant+add+requant on mxfp8 mom_buf.

    Parameters
    ----------
    mom_buf : [rows * cols_padded] E4M3 CPU tensor (modified in place)
    mom_scale : [rows * cols_padded / block_size] E8M0 CPU tensor
                (modified in place)
    grad : [rows * cols_padded] BF16 CPU tensor (read-only)
    rows, cols : original (unpadded) shape. ``cols_padded = cols``
                 rounded up to a multiple of ``block_size``.
    block_size : E8M0 block size (default 32).

    Notes
    -----
    The trailing padded block (when ``cols % block_size != 0``)
    is filled with zeros and contributes nothing on dequant;
    the kernel writes a zero E4M3 + E8M0=1 for it, so the next
    cycle's accumulation doesn't carry stale garbage.
    """
    mod = _ensure_module()
    cols_p = cols if cols % block_size == 0 else cols + (block_size - cols % block_size)
    _validate_inputs(mom_buf, mom_scale, grad, rows, cols_p, block_size)
    mod.fused_mxfp8_dequant_add_requant(mom_buf, mom_scale, grad,
                                         rows, cols_p, block_size)


def requantize_mxfp8(
    mom_buf: torch.Tensor,
    mom_scale: torch.Tensor,
    rows: int,
    cols: int,
    block_size: int = _DEFAULT_BLOCK_SIZE,
) -> None:
    """Requantize-only (no add) — used at :meth:`CPUMuon.step` end.

    Dequantizes ``mom_buf`` in place, then requantizes (which
    just rebalances the E8M0 scale and saturates the E4M3
    values to ±448 if they overflowed from the NS update).
    """
    mod = _ensure_module()
    cols_p = cols if cols % block_size == 0 else cols + (block_size - cols % block_size)
    _validate_inputs(mom_buf, mom_scale, grad=None,  # type: ignore[arg-type]
                     rows=rows, cols_p=cols_p, block_size=block_size,
                     grad_required=False)
    mod.requantize_mxfp8(mom_buf, mom_scale, rows, cols_p, block_size)
