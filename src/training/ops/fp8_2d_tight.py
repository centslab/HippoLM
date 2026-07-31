"""FP8 2D tight-scale quantization for matrix-valued tensors.

This module provides :func:`quantize_2d` and :func:`dequantize_2d`
for E4M3 storage with a fine-grained per-(row × col-block) +
per-(col × row-block) tight scale (the
``lowbit-Muon/src/quant.py::quantize_2d`` scheme). Used by
:class:`CPUMuon` to store the SGD momentum EMA buffer
(``s.exp_avg``) in FP8 while keeping the Newton-Schulz
NS-output drift under 2% on the prod shape (vs 14-21% for
naive per-tensor E4M3).

Why 2D tight scale vs the alternatives
---------------------------------------
The 14-21% NS output drift from per-tensor E4M3 storage is
almost entirely a U, V (singular vector) rotation effect, not
a Σ (singular value) error. Per-tensor amax is dominated by
outliers in the data, so the E4M3 grid step is ~22-178x
coarser than BF16 for non-outlier elements, which rotates
the singular vectors of the EMA buffer. NS passes U, V error
through unchanged (the orthogonalization produces from the
input's U, V, and there is no projection-back step).

2D tight scale's ``min(scale_dim1, scale_dim2)`` per element
makes the per-element noise direction-aware: within a block,
the scale is bounded by the amax over a ``1 × block`` strip
(row, col-block) AND a ``block × 1`` strip (col, row-block),
not the global max. This breaks the outlier-dominance
assumption and the U, V rotation drops by 5-15x. See
``docs/auto-memory/project_2d_tight_scale.md`` for the
empirical decomposition.

Storage layout per 2D tensor ``x`` of shape ``(rows, cols)``
with block ``B`` (assumes ``rows % B == 0`` and
``cols % B == 0`` — the optimizer only ever sees production
shapes which are multiples of 32):

  * ``q``            : ``(rows * cols,)``  ``torch.float8_e4m3fn``
  * ``scale_dim1``   : ``(rows, cols // B)``  ``torch.float32``
                        (per-row × per-col-block amax)
  * ``scale_dim2``   : ``(cols, rows // B)``  ``torch.float32``
                        (per-col × per-row-block amax)

This is the same convention as
``lowbit-Muon/src/quant.py::quantize_2d``. Block size 32 is
the default (matches the OCP MX-FP8 block size and gives the
best NS-output drift at the prod shape; the prod-shape
empirical NS-output drift at block=32 is 1.45% on regime B
and 1.45% on regime D, vs 14-21% for per-tensor E4M3).

Storage overhead of the two scales at prod shape
(1536×1536, block=32):

  * scale_dim1: 1536 × 48 × 4 bytes = 288 KiB
  * scale_dim2: 1536 × 48 × 4 bytes = 288 KiB
  * FP8 data:   1536 × 1536 × 1 byte ≈ 2.25 MiB

Total scale overhead ≈ 25% of FP8 data size (vs BF16 baseline
4.5 MiB per buffer, the FP8+scales layout is 2.81 MiB,
~37% reduction in per-buffer storage).

Numerical contract
------------------
* :func:`quantize_2d` is a lossy map to E4M3 + per-block FP32
  scales; the residual per-element error is bounded by
  ``E4M3_max / 2^3 * scale_tight = scale_tight * 56`` where
  ``scale_tight = min(amax_row_colblock, amax_col_rowblock)``
  is the per-element scale.
* :func:`dequantize_2d` is exact (a multiply by the
  per-element tight scale recovers the FP32 dequantized
  value).
* Composition is idempotent on E4M3-representable values:
  ``Q(D(Q(D(x)))) == Q(D(x))`` within E4M3's grid step.

Both functions are CPU-only (no GPU kernels); the E4M3 cast
goes through :func:`torch.Tensor.to` which is the same
native dtype path used in the W8A8 KDA path
(:mod:`src.models.ops.fp8_linear`).
"""
from __future__ import annotations

from typing import Tuple

import torch

E4M3_MAX = 448.0
E4M3_MIN_NORMAL = 2.0 ** -9  # smallest positive E4M3 normal = 2^-9 ≈ 1.95e-3
DEFAULT_BLOCK = 32


def quantize_2d(
    x: torch.Tensor,
    block: int = DEFAULT_BLOCK,
    scale_dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """2D tight-scale E4M3 quantization of a 2D FP32 tensor.

    Returns ``(q, scale_dim1, scale_dim2, tight)`` where:

      * ``q`` is a flat E4M3 tensor of shape ``(rows * cols,)``.
      * ``scale_dim1`` is FP32 (or ``scale_dtype``), shape
        ``(rows, cols // block)`` — the per-row × per-col-block
        amax divided by ``E4M3_MAX``.
      * ``scale_dim2`` is FP32 (or ``scale_dtype``), shape
        ``(cols, rows // block)`` — the per-col × per-row-block
        amax divided by ``E4M3_MAX``.
      * ``tight`` is FP32, shape ``(rows, cols)`` — the
        per-element scale ``min(scale_dim1_broadcast,
        scale_dim2_broadcast)``. Returned so
        :func:`dequantize_2d` can apply it without re-broadcasting
        (matters for the per-step EMA hot path).

    Parameters
    ----------
    x : (rows, cols) FP32 (or any floating) tensor. ``rows``
        and ``cols`` MUST be divisible by ``block`` (the
        optimizer only sees production shapes which are
        multiples of 32).
    block : E4M3 block size along each axis (default 32).
    scale_dtype : torch.dtype
        Storage dtype for ``scale_dim1`` and ``scale_dim2``.
        Default ``torch.float32`` (Muon's prod setting — see
        ``docs/auto-memory/project_2d_tight_scale.md``).
        ``torch.bfloat16`` is the cheap-scale alternative —
        BF16 has 7-bit mantissa vs FP32's 23-bit; whether
        that matters for gradient-accumulation precision is
        tested in
        ``test/_tmp/probe_grad_accum_quant_sweep.py``.

    Notes
    -----
    Both ``scale_dim1`` and ``scale_dim2`` are derived from
    per-strip amax divided by ``E4M3_MAX`` (the standard
    "fit into the E4M3 dynamic range" rule), and the
    per-element quantization uses
    ``min(scale_dim1, scale_dim2)`` per element. The min is
    the conservative choice: it ensures neither the
    row-strip nor the col-strip amax overflows in E4M3.

    Implementation: this is the same scheme as
    ``lowbit-Muon/src/quant.py::quantize_2d`` (block_sz_in=128
    there; we use 32 here), but using native
    :class:`torch.float8_e4m3fn` instead of packed INT4. The
    ``(scale_dim1, scale_dim2)`` shapes are the same as
    lowbit-Muon's saved scales.
    """
    assert x.dim() == 2, f"quantize_2d expects 2D, got {x.dim()}D"
    assert block >= 1
    assert scale_dtype in (torch.float32, torch.bfloat16), (
        f"scale_dtype must be float32 or bfloat16, got {scale_dtype}"
    )
    xf = x.to(torch.float32)
    rows, cols = xf.shape
    assert rows % block == 0, (
        f"rows={rows} must be divisible by block={block}; "
        "the optimizer only sees production shapes (multiples of 32)"
    )
    assert cols % block == 0, (
        f"cols={cols} must be divisible by block={block}; "
        "the optimizer only sees production shapes (multiples of 32)"
    )
    n_row_blocks = rows // block
    n_col_blocks = cols // block

    # ---- scale_dim1: per-row × per-col-block amax ----
    # Group consecutive `block` elements along the LAST dim
    # (lowbit-Muon's `tensor.reshape(-1, blk_sz_in)` pattern).
    # This gives one amax per (row, col-block) strip, NOT per
    # (row-block, all cols) strip — the fine granularity is
    # what makes 2D tight isolate outliers to their col-block.
    g_r = xf.reshape(rows * n_col_blocks, block)
    amax_r = g_r.abs().max(dim=-1, keepdim=True)[0]  # (rows * n_col_blocks, 1)
    scale_dim1 = (amax_r / E4M3_MAX).clamp(min=E4M3_MIN_NORMAL / E4M3_MAX)
    scale_dim1 = scale_dim1.reshape(rows, n_col_blocks).contiguous()
    if scale_dtype == torch.bfloat16:
        scale_dim1 = scale_dim1.to(torch.bfloat16)

    # ---- scale_dim2: per-col × per-row-block amax ----
    # Transpose then group consecutive `block` elements along
    # the (transposed) last dim. The transpose + reshape
    # produces a contiguous copy where consecutive `block`
    # elements are `block` rows at the same column.
    g_c = xf.t().reshape(cols * n_row_blocks, block)
    amax_c = g_c.abs().max(dim=-1, keepdim=True)[0]  # (cols * n_row_blocks, 1)
    scale_dim2 = (amax_c / E4M3_MAX).clamp(min=E4M3_MIN_NORMAL / E4M3_MAX)
    scale_dim2 = scale_dim2.reshape(cols, n_row_blocks).contiguous()
    if scale_dtype == torch.bfloat16:
        scale_dim2 = scale_dim2.to(torch.bfloat16)

    # ---- Broadcast both to (rows, cols), take per-element min ----
    # `tight` stays FP32 for the in-kernel arithmetic precision
    # (the per-element scale broadcast is compute-path, not
    # storage-path).
    scale_dim1_full = (
        scale_dim1.to(torch.float32).unsqueeze(-1)
        .repeat(1, 1, block)
        .reshape(rows, cols)
    )
    scale_dim2_full = (
        scale_dim2.to(torch.float32).unsqueeze(-1)
        .repeat(1, 1, block)
        .reshape(cols, rows)
        .t()
    )
    tight = torch.min(scale_dim1_full, scale_dim2_full).contiguous()

    # ---- Quantize via per-element scale = `tight` ----
    q_2d = (xf / tight).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn)
    q_flat = q_2d.reshape(-1).contiguous()
    return q_flat, scale_dim1, scale_dim2, tight


def dequantize_2d(
    q: torch.Tensor,
    scale_dim1: torch.Tensor,
    scale_dim2: torch.Tensor,
    rows: int,
    cols: int,
    block: int = DEFAULT_BLOCK,
) -> torch.Tensor:
    """Inverse of :func:`quantize_2d` (returns the FP32
    dequantized tensor of shape ``(rows, cols)``).

    Re-derives the per-element tight scale from
    ``scale_dim1`` and ``scale_dim2`` (the broadcast +
    per-element min). Both scales must match the shapes
    produced by :func:`quantize_2d`:
    ``scale_dim1.shape == (rows, cols // block)``,
    ``scale_dim2.shape == (cols, rows // block)``.

    This function does the broadcast on every call; the
    CPUMuon step path avoids it by reading the cached
    ``tight`` from :func:`quantize_2d`'s return value (the
    optimizer updates ``tight`` in place via the EMA's
    :func:`torch.Tensor.mul_` and :func:`torch.Tensor.add_`).
    Use this function only for one-off dequant (e.g. the
    initial state load).
    """
    assert q.dim() == 1
    assert scale_dim1.dim() == 2 and scale_dim2.dim() == 2
    n_row_blocks = rows // block
    n_col_blocks = cols // block
    assert scale_dim1.shape == (rows, n_col_blocks), (
        f"scale_dim1 shape {tuple(scale_dim1.shape)} != expected "
        f"({rows}, {n_col_blocks}) for rows={rows}, cols={cols}, block={block}"
    )
    assert scale_dim2.shape == (cols, n_row_blocks), (
        f"scale_dim2 shape {tuple(scale_dim2.shape)} != expected "
        f"({cols}, {n_row_blocks}) for rows={rows}, cols={cols}, block={block}"
    )
    assert q.shape[0] == rows * cols, (
        f"q length {q.shape[0]} != rows*cols = {rows * cols}"
    )
    q_2d = q.view(rows, cols).to(torch.float32)
    scale_dim1_full = (
        scale_dim1.unsqueeze(-1)
        .repeat(1, 1, block)
        .reshape(rows, cols)
    )
    scale_dim2_full = (
        scale_dim2.unsqueeze(-1)
        .repeat(1, 1, block)
        .reshape(cols, rows)
        .t()
    )
    tight = torch.min(scale_dim1_full, scale_dim2_full)
    return q_2d * tight