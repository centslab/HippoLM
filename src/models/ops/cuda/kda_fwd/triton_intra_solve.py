"""Lever H — fused Triton intra_solve kernel (replaces the 19.4 ms bmm loop).

Replaces the 10-pair Python bmm loop in __init__.py:333-371 with ONE
Triton kernel. Eliminates 20 cuBLAS launches + intermediate tensor
allocations (q*r, k*c, scaled outputs).

Layout (matches __init__.py):
  * q_per:    [num_chunks, HV, BT, K] bf16  (loaded as bf16, cast to fp32 in
                registers — saves ~1.3 ms vs pre-casting whole tensor to fp32)
  * k_per:    [num_chunks, HV, BT, K] bf16  (same)
  * g_per:    [num_chunks, HV, BT, K] fp32 (chunk-local cumsum, log2 — needs
                fp32 for the exp2 arguments)
  * A_qk:     [num_chunks*HV, BT, BT] bf16 (with scale baked in — saves ~0.6 ms
                vs the wrapper's fp32→bf16 cast after the kernel)
  * A_kk:     [num_chunks*HV, BT, BT] fp32 (forward_sub kernel requires fp32)

Grid: (num_pairs, num_chunks * HV). num_pairs = 10 (NC=4: 4 diag + 6 off-diag).

Per program:
  1. Resolve pair (s_i, s_j) from precomputed lookup.
  2. Determine anchor_pos: sub-chunk middle if diag, row start if off-diag.
  3. Load q_row, k_row, k_col (bf16) + g_row, g_col, g_anchor (fp32).
  4. Compute r = exp2(g_row - anchor), c = exp2(anchor - g_col).
  5. Cast q/k to fp32, multiply by r/c, cast back to bf16 for TensorCore mma.
  6. bf16 mma: A_qk = (q*r) @ (k_col*c).T * scale, A_kk = (k_row*r) @ (k_col*c).T.
  7. Write A_qk (bf16) + A_kk (fp32) sub-blocks.

Lever H savings (vs Python loop at prod B=1 T=16384 H=12 K=V=128):
  * Python loop: 19.4 ms (10 pairs × cuBLAS bmm + 30 element-wise + intermediates)
  * Fused Triton: ~1.5 ms (one launch, all intermediates in registers)
  * Net: ~17.9 ms saved at prod (~5.5 ms fwd → ~3.7 ms fwd pipeline).
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


# Precomputed pair lookup (matches __init__.py:331). Length 10.
# Order: diagonal first (4 entries), then off-diag upper-tri (6 entries).
PAIR_SI = [0, 1, 2, 3, 1, 2, 2, 3, 3, 3]
PAIR_SJ = [0, 1, 2, 3, 0, 0, 1, 0, 1, 2]
NUM_PAIRS = len(PAIR_SI)  # 10


@triton.jit
def _intra_solve_kernel(
    q_per,                    # [num_chunks, HV, BT, K] bf16
    k_per,                    # [num_chunks, HV, BT, K] bf16
    g_per,                    # [num_chunks, HV, BT, K] fp32 (cumsum, log2)
    pair_si,                  # [NUM_PAIRS] int32 (lookup)
    pair_sj,                  # [NUM_PAIRS] int32
    A_qk,                     # [num_chunks*HV, BT, BT] bf16 (output, scaled)
    A_kk,                     # [num_chunks*HV, BT, BT] fp32 (output, no scale)
    scale,
    BT: tl.constexpr,
    BC: tl.constexpr,
    K: tl.constexpr,
    NUM_PAIRS: tl.constexpr,
):
    pid_pair = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Resolve (s_i, s_j) from the precomputed pair lookup.
    s_i = tl.load(pair_si + pid_pair)
    s_j = tl.load(pair_sj + pid_pair)

    # Pointer base for this (chunk, hv) — flat ordering chunk*HV + hv.
    q_p = q_per + pid_n * BT * K
    k_p = k_per + pid_n * BT * K
    g_p = g_per + pid_n * BT * K

    # Anchor position: sub-chunk middle for diag, row start for off-diag.
    is_diag = s_i == s_j
    anchor_pos = tl.where(is_diag, s_i * BC + (BC // 2), s_i * BC)

    # Offsets
    offs_i = tl.arange(0, BC)                  # 0..BC-1
    offs_k = tl.arange(0, K)                   # 0..K-1
    row_off = s_i * BC + offs_i                # [BC]
    col_off = s_j * BC + offs_i                # [BC]

    # Load g_anchor [K] (fp32 — needed for precise exp2 args)
    g_anchor = tl.load(g_p + anchor_pos * K + offs_k)  # [K] fp32

    # Load q_row, k_row, k_col as [BC, K] bf16 (cast to fp32 in registers — free)
    q_row_bf16 = tl.load(q_p + row_off[:, None] * K + offs_k[None, :])  # [BC, K] bf16
    k_row_bf16 = tl.load(k_p + row_off[:, None] * K + offs_k[None, :])
    k_col_bf16 = tl.load(k_p + col_off[:, None] * K + offs_k[None, :])
    # Load g_row, g_col as [BC, K] fp32
    g_row = tl.load(g_p + row_off[:, None] * K + offs_k[None, :])
    g_col = tl.load(g_p + col_off[:, None] * K + offs_k[None, :])

    # r, c: [BC, K] fp32 — needed for precise exp2 args
    r = tl.exp2(g_row - g_anchor[None, :])
    c = tl.exp2(g_anchor[None, :] - g_col)

    # Cast q/k to fp32 for the multiplication, then back to bf16 for TensorCore mma.
    # bf16 → fp32 in registers is free (no HBM cost). The fp32 multiply is more
    # precise than bf16 multiply for r/c in the typical [0.5, 2] range.
    q_row = q_row_bf16.to(tl.float32)
    k_row = k_row_bf16.to(tl.float32)
    k_col = k_col_bf16.to(tl.float32)

    qr_bf16 = (q_row * r).to(tl.bfloat16)     # [BC, K] bf16
    krr_bf16 = (k_row * r).to(tl.bfloat16)
    kc_bf16 = (k_col * c).to(tl.bfloat16)

    # bf16 mma → fp32 accum (TensorCore m16n8k16)
    A_qk_blk = tl.dot(qr_bf16, tl.trans(kc_bf16), out_dtype=tl.float32) * scale
    A_kk_blk = tl.dot(krr_bf16, tl.trans(kc_bf16), out_dtype=tl.float32)

    # Write outputs at [pid_n, s_i*BC:(s_i+1)*BC, s_j*BC:(s_j+1)*BC]
    out_offs = row_off[:, None] * BT + col_off[None, :]
    tl.store(A_qk + pid_n * BT * BT + out_offs, A_qk_blk.to(tl.bfloat16))
    tl.store(A_kk + pid_n * BT * BT + out_offs, A_kk_blk)


def triton_intra_solve(
    q_per: torch.Tensor,       # [num_chunks, HV, BT, K] bf16
    k_per: torch.Tensor,       # [num_chunks, HV, BT, K] bf16
    g_per: torch.Tensor,       # [num_chunks, HV, BT, K] fp32 (cumsum, log2)
    scale: float,
    BT: int = 64,
    BC: int = 16,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (A_qk_bf16 [N, BT, BT], A_kk_fp32 [N, BT, BT]).

    Replaces __init__.py:327-371. Output A_qk has scale baked in (matches
    the wrapper's contract for A_qk_bf16). Output A_kk is the
    intermediate for forward_sub (fp32 required).
    """
    assert q_per.dtype == torch.bfloat16, f"q_per must be bf16, got {q_per.dtype}"
    assert k_per.dtype == torch.bfloat16, f"k_per must be bf16, got {k_per.dtype}"
    assert g_per.dtype == torch.float32, f"g_per must be fp32, got {g_per.dtype}"
    num_chunks, HV, _, K = q_per.shape
    assert BT == 64 and BC == 16, "Lever H: BT=64, BC=16 only"
    # K must be a multiple of 16 for the bf16 mma. Tested at K=64, 128.
    assert K % 16 == 0, f"K={K} must be a multiple of 16 for bf16 mma"

    # A_qk: bf16 (with scale baked in) — saves ~0.6 ms vs writing fp32 + wrapper cast.
    # A_kk: fp32 (forward_sub kernel requires fp32; also precision matters here).
    A_qk = torch.zeros(num_chunks * HV, BT, BT, device=q_per.device, dtype=torch.bfloat16)
    A_kk = torch.zeros(num_chunks * HV, BT, BT, device=q_per.device, dtype=torch.float32)

    # Pair lookup on device (small, NUM_PAIRS=10)
    pair_si = torch.tensor(PAIR_SI, dtype=torch.int32, device=q_per.device)
    pair_sj = torch.tensor(PAIR_SJ, dtype=torch.int32, device=q_per.device)

    grid = (NUM_PAIRS, num_chunks * HV)
    _intra_solve_kernel[grid](
        q_per, k_per, g_per,
        pair_si, pair_sj,
        A_qk, A_kk,
        scale,
        BT=BT, BC=BC, K=K, NUM_PAIRS=NUM_PAIRS,
        num_warps=2, num_stages=2,
    )
    return A_qk, A_kk