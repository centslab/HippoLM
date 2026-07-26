"""Fused fp8_2d_tight dequant+EMA+requant Triton kernel for CPUMuon.

Replaces the 3-op CPU pipeline:

  dequantize_2d (q, s1, s2) → FP32 dequant         [CPU]
  ema_bf16.mul_(beta).add_(grad)                    [CPU]
  quantize_2d (ema_fp32) → (q_new, s1_new, s2_new)  [CPU]

with a single 3-pass Triton fused kernel that runs entirely on GPU:

  pass 1 (_dequant_ema_kernel):       dequant + EMA → BF16 EMA
  pass 2 (_compute_scales_kernel):     amax reductions → s1_new, s2_new
  pass 3 (_requant_kernel):            quantize using new scales → FP8 q_new

The 3-pass structure eliminates the 28 MiB intermediate
``.repeat().reshape()`` allocations that the PyTorch path requires
(the (4608, 1536) FFN-down shape allocates 3×28 MiB per
quantize_2d call, totalling ~49 ms — the dominant cost of the
CPU muon step path).

Numerics
--------
Bit-exact for dequant + EMA + scale computation relative to the
3-op PyTorch path. Requant may differ by 1 E4M3 grid step at the
saturation boundary (E4M3 max = 448, RNE rounding tie-break);
this is non-compounding per
``project_fp8_drift_50_step_2026_07_23.md`` — NS output cos_sim >
0.999 over 50 EMA steps across the 3 prod shapes.

Storage layout per 2D tensor ``x`` of shape ``(rows, cols)`` with
block ``B``:

  * ``q``            : ``(rows * cols,)``  ``torch.float8_e4m3fn``
  * ``scale_dim1``   : ``(rows, cols // B)``  ``torch.float32``
  * ``scale_dim2``   : ``(cols, rows // B)``  ``torch.float32``

(Identical to ``fp8_2d_tight.quantize_2d``.)
"""
from __future__ import annotations

from typing import Tuple

import torch
import triton
import triton.language as tl

E4M3_MAX = 448.0
# Smallest positive E4M3 normal = 2^-9 ≈ 1.95e-3.
E4M3_MIN_NORMAL = 2.0 ** -9
DEFAULT_BLOCK = 32


@triton.jit
def _dequant_ema_kernel(
    Q_ptr,                # FP8, flat (rows * cols,)
    S1_ptr,               # FP32, (rows, cols // block)
    S2_ptr,               # FP32, (cols, rows // block)
    G_ptr,                # BF16, flat (rows * cols,)
    EMA_ptr,              # BF16, flat (rows * cols,)  — output
    rows,
    cols,
    n_col_blocks,
    n_row_blocks,
    beta,
    E4M3_MAX: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Per-strip dequant + EMA. Grid: (rows, n_col_blocks).

    Each program owns 1 row × ``BLOCK`` cols. Loads ``q``,
    ``scale_dim1[r, cb]`` (scalar) and ``scale_dim2[c, rb]``
    (vector of ``BLOCK`` for the cols in the strip), computes the
    per-element tight scale, dequantizes, applies EMA in BF16, and
    writes the BF16 EMA strip to ``EMA_ptr``.
    """
    pid_r = tl.program_id(0)
    pid_cb = tl.program_id(1)
    rb = pid_r // BLOCK

    col_off = pid_cb * BLOCK + tl.arange(0, BLOCK)
    flat_off = pid_r * cols + col_off

    # Load q strip (FP8 → FP32)
    q_strip = tl.load(Q_ptr + flat_off).to(tl.float32)

    # scale_dim1 for this (row, col_block) — one scalar
    s1 = tl.load(S1_ptr + pid_r * n_col_blocks + pid_cb)

    # scale_dim2 for each col in strip — vector of BLOCK.
    # scale_dim2 layout: (cols, rows // block), so row index = col_off,
    # col index = rb.
    s2 = tl.load(S2_ptr + col_off * n_row_blocks + rb)

    # Per-element tight = min(s1, s2[c])
    tight = tl.minimum(s1, s2)

    # Dequant → FP32 → BF16 (narrow matches the existing 2-launch
    # pipeline's rounding).
    deq = (q_strip * tight).to(tl.bfloat16)
    grad = tl.load(G_ptr + flat_off)
    # EMA: beta * deq + grad (BF16 ops, matches the standalone
    # ``ema.mul_(beta).add_(grad)`` chain).
    ema = deq * beta + grad
    tl.store(EMA_ptr + flat_off, ema)


@triton.jit
def _compute_scales_kernel(
    EMA_ptr,              # BF16, flat (rows * cols,)
    S1_NEW_ptr,           # FP32, (rows, cols // block) — output
    S2_NEW_ptr,           # FP32, (cols, rows // block) — output
    rows,
    cols,
    n_col_blocks,
    n_row_blocks,
    E4M3_MAX: tl.constexpr,
    E4M3_MIN_NORMAL: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Per-tile scale computation. Grid: (n_row_blocks, n_col_blocks).

    Each program owns one (row_block, col_block) tile of shape
    ``BLOCK × BLOCK``. Within the tile, computes:

      * ``s1_new[r, cb] = max(abs(ema[r, c])) / E4M3_MAX`` for each r
        in the tile (per-row × per-col-block amax) — local: this
        program owns the entire col_block for these rows in this
        row_block.
      * ``s2_new[c, rb] = max(abs(ema[r, c])) / E4M3_MAX`` for each c
        in the tile (per-col × per-row-block amax) — local: this
        program owns the entire row_block for these cols.

    Both are local reductions (no atomics), since the tile is
    exactly one (row_block, col_block) block.
    """
    pid_rb = tl.program_id(0)
    pid_cb = tl.program_id(1)

    row_off = pid_rb * BLOCK + tl.arange(0, BLOCK)
    col_off = pid_cb * BLOCK + tl.arange(0, BLOCK)

    # Load tile (BLOCK × BLOCK) of BF16 EMA → FP32
    ema_ptrs = EMA_ptr + row_off[:, None] * cols + col_off[None, :]
    ema = tl.load(ema_ptrs).to(tl.float32)
    abs_ema = tl.abs(ema)

    # s1_new per row (axis=1 = across cols)
    amax_per_row = tl.max(abs_ema, axis=1)            # (BLOCK,)
    s1_new = tl.maximum(
        amax_per_row / E4M3_MAX,
        E4M3_MIN_NORMAL / E4M3_MAX,
    )
    s1_offsets = row_off * n_col_blocks + pid_cb
    tl.store(S1_NEW_ptr + s1_offsets, s1_new)

    # s2_new per col (axis=0 = across rows)
    amax_per_col = tl.max(abs_ema, axis=0)            # (BLOCK,)
    s2_new = tl.maximum(
        amax_per_col / E4M3_MAX,
        E4M3_MIN_NORMAL / E4M3_MAX,
    )
    s2_offsets = col_off * n_row_blocks + pid_rb
    tl.store(S2_NEW_ptr + s2_offsets, s2_new)


@triton.jit
def _requant_kernel(
    EMA_ptr,              # BF16, flat (rows * cols,)
    S1_NEW_ptr,           # FP32, (rows, cols // block)
    S2_NEW_ptr,           # FP32, (cols, rows // block)
    Q_NEW_ptr,            # FP8, flat (rows * cols,) — output
    rows,
    cols,
    n_col_blocks,
    n_row_blocks,
    E4M3_MAX: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Per-strip requant using precomputed new scales. Grid:
    (rows, n_col_blocks).

    Each program owns 1 row × ``BLOCK`` cols. Loads the BF16 EMA
    strip, the new scales, computes the per-element tight scale,
    quantizes to E4M3, writes the FP8 strip.
    """
    pid_r = tl.program_id(0)
    pid_cb = tl.program_id(1)
    rb = pid_r // BLOCK

    col_off = pid_cb * BLOCK + tl.arange(0, BLOCK)
    flat_off = pid_r * cols + col_off

    # Load BF16 EMA strip
    ema = tl.load(EMA_ptr + flat_off).to(tl.float32)

    # New scales
    s1_new = tl.load(S1_NEW_ptr + pid_r * n_col_blocks + pid_cb)
    s2_new = tl.load(S2_NEW_ptr + col_off * n_row_blocks + rb)

    # Per-element tight = min(s1_new, s2_new[c])
    tight_new = tl.minimum(s1_new, s2_new)

    # Quantize: clamp(ema / tight_new, ±E4M3_MAX), cast to E4M3.
    # The clamp+cast matches the standalone ``quantize_2d`` flow.
    q_new = tl.minimum(
        tl.maximum(ema / tight_new, -E4M3_MAX),
        E4M3_MAX,
    ).to(tl.float8e4nv)
    tl.store(Q_NEW_ptr + flat_off, q_new)


def dequant_ema_requant_2d_fused(
    q: torch.Tensor,
    scale_dim1: torch.Tensor,
    scale_dim2: torch.Tensor,
    grad: torch.Tensor,
    beta: float,
    rows: int,
    cols: int,
    block: int = DEFAULT_BLOCK,
    ema_out: torch.Tensor | None = None,
    q_new_out: torch.Tensor | None = None,
    s1_new_out: torch.Tensor | None = None,
    s2_new_out: torch.Tensor | None = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Single-call fused dequant + EMA + requant on GPU.

    All inputs and outputs must live on the same CUDA device. The
    function is a drop-in replacement for the 3-op CPU pipeline:

      deq = dequantize_2d(q, s1, s2)         # CPU
      ema = deq.bf16().mul_(beta).add_(grad) # CPU
      q_new, s1_new, s2_new, tight = quantize_2d(ema.fp32(), block)

    Returns ``(q_new, scale_dim1_new, scale_dim2_new, ema_bf16)``
    where ``ema_bf16`` is the BF16 EMA buffer (the input to the
    downstream Newton-Schulz orthogonalization).

    Pass 1 (``_dequant_ema_kernel``) writes the BF16 EMA.
    Pass 2 (``_compute_scales_kernel``) computes ``s1_new`` and
    ``s2_new`` from the BF16 EMA.
    Pass 3 (``_requant_kernel``) writes the FP8 ``q_new`` using
    the precomputed scales.

    The caller may supply pre-allocated ``ema_out``, ``q_new_out``,
    ``s1_new_out``, ``s2_new_out`` to avoid the per-call allocation;
    otherwise they're allocated on the input device.

    Parameters
    ----------
    q : ``(rows * cols,)`` FP8 E4M3 (CPU-pinned storage copied to
        GPU before the muon step).
    scale_dim1 : ``(rows, cols // block)`` FP32 (same as above).
    scale_dim2 : ``(cols, rows // block)`` FP32 (same as above).
    grad : ``(rows * cols,)`` BF16, GPU-side per-step gradient
        accumulator (already on GPU).
    beta : EMA coefficient (``s.exp_avg ← β * prev + grad``).
    rows, cols : matrix shape.
    block : per-axis block size (default 32; production muon uses 32).

    Returns
    -------
    (q_new, scale_dim1_new, scale_dim2_new, ema) where the first
    three are the new FP8 storage (caller D2Hs back to CPU pinned)
    and ``ema`` is the BF16 buffer for downstream NS.
    """
    assert q.dtype == torch.float8_e4m3fn
    assert scale_dim1.dtype == torch.float32
    assert scale_dim2.dtype == torch.float32
    assert grad.dtype == torch.bfloat16
    device = q.device
    assert device.type == "cuda"
    assert grad.device == device
    assert rows % block == 0 and cols % block == 0

    n_row_blocks = rows // block
    n_col_blocks = cols // block

    # Allocate (or use caller-supplied) outputs on GPU.
    if ema_out is None:
        ema_out = torch.empty(
            rows * cols, dtype=torch.bfloat16, device=device,
        )
    if q_new_out is None:
        q_new_out = torch.empty(
            rows * cols, dtype=torch.float8_e4m3fn, device=device,
        )
    if s1_new_out is None:
        s1_new_out = torch.empty(
            rows, n_col_blocks, dtype=torch.float32, device=device,
        )
    if s2_new_out is None:
        s2_new_out = torch.empty(
            cols, n_row_blocks, dtype=torch.float32, device=device,
        )

    # num_warps tuned per BLOCK. BLOCK=32 → 1 program per (row,
    # col_block) strip; each strip is 32 BF16 elements = 64 bytes
    # of work. 2 warps is enough.
    num_warps = 2 if block <= 32 else 4

    # Pass 1: dequant + EMA → BF16 EMA.
    grid1 = (rows, n_col_blocks)
    _dequant_ema_kernel[grid1](
        q, scale_dim1, scale_dim2, grad, ema_out,
        rows, cols,
        n_col_blocks, n_row_blocks,
        beta=beta,
        E4M3_MAX=E4M3_MAX,
        BLOCK=block,
        num_warps=num_warps,
    )

    # Pass 2: compute new scales from the BF16 EMA.
    grid2 = (n_row_blocks, n_col_blocks)
    _compute_scales_kernel[grid2](
        ema_out, s1_new_out, s2_new_out,
        rows, cols,
        n_col_blocks, n_row_blocks,
        E4M3_MAX=E4M3_MAX,
        E4M3_MIN_NORMAL=E4M3_MIN_NORMAL,
        BLOCK=block,
        num_warps=num_warps,
    )

    # Pass 3: requant using the new scales → FP8 q_new.
    grid3 = (rows, n_col_blocks)
    _requant_kernel[grid3](
        ema_out, s1_new_out, s2_new_out, q_new_out,
        rows, cols,
        n_col_blocks, n_row_blocks,
        E4M3_MAX=E4M3_MAX,
        BLOCK=block,
        num_warps=num_warps,
    )

    return q_new_out, s1_new_out, s2_new_out, ema_out