"""Lever H + M + P — fused Triton intra_solve kernel (replaces the 19.4 ms bmm loop).

Replaces the 10-pair Python bmm loop in __init__.py with ONE Triton kernel.
Eliminates 20 cuBLAS launches + intermediate tensor allocations.

Lever P (June 2026-30): the kernel now ALSO multiplies A_kk row-wise by
beta[i] and adds 1.0 on the diagonal, so the output is ready for
forward_sub directly — no separate beta multiply or eye add in the wrapper.
Saves ~0.72 ms at prod (0.40 ms beta mul + 0.32 ms I add).

Strided-read layout (Lever M, June 2026-30):
  * q_tok, k_tok:    [T, H, K] bf16  (natural [B, T, H, K] flattened, no
                                       transpose+contiguous needed)
  * g_cum_tok:       [T, H, K] bf16  (chunk-local cumsum, log2 — produced by
                                       triton_g_cumsum.g_cumsum_fused)
                                       Cast to fp32 in registers (free).
  * beta_tok:        [T, HV]   bf16  (strided; loaded per-row in registers)
  * A_qk:            [num_chunks*HV, BT, BT] bf16 (scale baked in)
  * A_kk:            [num_chunks*HV, BT, BT] fp32 (forward_sub kernel consumes)
                     Already multiplied by beta[i] row-wise and has 1.0 on
                     the diagonal — direct output of `A = I + A_kk * beta`
                     for the strictly-lower-tri entries.

Grid: (NUM_PAIRS=10, num_chunks * HV).

Per program:
  1. Resolve (chunk_idx, hv_idx) from pid_n.
  2. Resolve (s_i, s_j) from precomputed lookup.
  3. Compute base pointer in [T, H, K] layout: chunk_idx*BT*H*K + hv_idx*K.
  4. Anchor: sub-chunk middle if diag, row start if off-diag.
  5. Load q_row, k_row, k_col (bf16) + g_row, g_col, g_anchor (bf16→fp32).
  6. Load beta_row [BC] (bf16 → fp32 in registers; strided read).
  7. r = exp2(g_row - anchor), c = exp2(anchor - g_col).
  8. Cast q/k to fp32, multiply by r/c, cast back to bf16 for TensorCore mma.
  9. bf16 mma: A_qk = (q*r) @ (k*c).T * scale, A_kk = (k*r) @ (k*c).T.
 10. Apply beta: A_kk = A_kk * beta_row[:, None]. Add 1.0 on the diagonal
     entries (only present in s_i==s_j blocks).
 11. Write A_qk (bf16) + A_kk (fp32) sub-blocks. For s_i==s_j pairs, only
     the strict-lower-tri + diagonal are written (Lever K — upper-tri is
     left at the torch.zeros init value, which is 0.0).

Lever H savings (vs Python loop at prod B=1 T=16384 H=12 K=V=128):
  * Python loop: 19.4 ms → Fused Triton: ~1.5 ms (~17.9 ms saved).
Lever M savings (strided reads, vs contiguous [NC, HV, BT, K]):
  * Eliminates 4 .contiguous() calls in __init__.py:310-316 (~0.5 ms).
Lever P savings (fused beta + I into kernel):
  * Eliminates 0.40 ms (A_kk * beta) + 0.32 ms (A_kk + I) at prod
    = ~0.72 ms saved.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


# Precomputed pair lookup (matches FLA's intra sub-chunk loop). Length 10.
# Order: diagonal first (4 entries), then off-diag upper-tri (6 entries).
PAIR_SI = [0, 1, 2, 3, 1, 2, 2, 3, 3, 3]
PAIR_SJ = [0, 1, 2, 3, 0, 0, 1, 0, 1, 2]
NUM_PAIRS = len(PAIR_SI)  # 10


@triton.jit
def _intra_solve_kernel(
    q_tok,                   # [T, H, K] bf16 (strided)
    k_tok,                   # [T, H, K] bf16 (strided)
    g_cum_tok,               # [T, H, K] bf16 (strided; cumsum, log2)
    beta_tok,                # [T, HV]   bf16 (strided; per-token beta scalars)
    pair_si,                 # [NUM_PAIRS] int32
    pair_sj,                 # [NUM_PAIRS] int32
    A_qk,                    # [num_chunks*HV, BT, BT] bf16 (output, scaled)
    A_kk,                    # [num_chunks*HV, BT, BT] fp32 (output: A = I + A_kk*beta)
    scale,
    H: tl.constexpr,         # number of heads (== HV in this wrapper)
    HV: tl.constexpr,        # == H for this wrapper (kept separate for clarity)
    T: tl.constexpr,         # total tokens = num_chunks * BT (const for fixed-len)
    BT: tl.constexpr,
    BC: tl.constexpr,
    K: tl.constexpr,
    NUM_PAIRS: tl.constexpr,
):
    pid_pair = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Resolve (chunk_idx, hv_idx) from pid_n. H is constexpr so the
    # division compiles to a fast shift+mask for power-of-2 H, or to a
    # general idiv otherwise. H == 12 in production.
    chunk_idx = pid_n // H
    hv_idx = pid_n - chunk_idx * H
    chunk_start = chunk_idx * BT  # absolute token start of this chunk

    # Strided pointer base in [T, H, K] layout: per-token stride = H*K,
    # per-head stride = K. So for token t and head h, the offset is
    # t * H * K + h * K. For our (chunk, hv) tile, base offset is
    # chunk_start * H * K + hv_idx * K.
    base_off = chunk_start * (H * K) + hv_idx * K
    q_p = q_tok + base_off
    k_p = k_tok + base_off
    g_p = g_cum_tok + base_off

    # beta_tok has shape [T, HV]. For our (chunk, hv) tile, base is
    # chunk_start * HV + hv_idx; per-token stride is HV.
    beta_p = beta_tok + chunk_start * HV + hv_idx
    beta_stride_within = HV

    # Resolve (s_i, s_j) from the precomputed pair lookup.
    s_i = tl.load(pair_si + pid_pair)
    s_j = tl.load(pair_sj + pid_pair)

    # Anchor position: sub-chunk middle for diag, row start for off-diag.
    is_diag = s_i == s_j
    anchor_pos = tl.where(is_diag, s_i * BC + (BC // 2), s_i * BC)

    # Offsets within [BT]. Within a (chunk, hv) tile, the per-row stride
    # is H*K (one token's worth of H heads × K dim).
    stride_within = H * K
    offs_i = tl.arange(0, BC)
    offs_k = tl.arange(0, K)
    row_off = s_i * BC + offs_i
    col_off = s_j * BC + offs_i

    # Load g_anchor [K] (bf16 → fp32 in registers).
    g_anchor = tl.load(g_p + anchor_pos * stride_within + offs_k).to(tl.float32)

    # Load q_row, k_row, k_col as [BC, K] bf16 (cast to fp32 in regs — free).
    q_row_bf16 = tl.load(q_p + row_off[:, None] * stride_within + offs_k[None, :])
    k_row_bf16 = tl.load(k_p + row_off[:, None] * stride_within + offs_k[None, :])
    k_col_bf16 = tl.load(k_p + col_off[:, None] * stride_within + offs_k[None, :])
    # Load g_row, g_col as [BC, K] fp32 (bf16 → fp32 in registers).
    g_row = tl.load(g_p + row_off[:, None] * stride_within + offs_k[None, :]).to(tl.float32)
    g_col = tl.load(g_p + col_off[:, None] * stride_within + offs_k[None, :]).to(tl.float32)

    # r, c: [BC, K] fp32 — needed for precise exp2 arguments.
    r = tl.exp2(g_row - g_anchor[None, :])
    c = tl.exp2(g_anchor[None, :] - g_col)

    q_row = q_row_bf16.to(tl.float32)
    k_row = k_row_bf16.to(tl.float32)
    k_col = k_col_bf16.to(tl.float32)

    qr_bf16 = (q_row * r).to(tl.bfloat16)     # [BC, K] bf16
    krr_bf16 = (k_row * r).to(tl.bfloat16)
    kc_bf16 = (k_col * c).to(tl.bfloat16)

    # bf16 mma → fp32 accum (TensorCore m16n8k16).
    A_qk_blk = tl.dot(qr_bf16, tl.trans(kc_bf16), out_dtype=tl.float32) * scale
    A_kk_blk = tl.dot(krr_bf16, tl.trans(kc_bf16), out_dtype=tl.float32)

    # Lever P (June 2026-30): fuse beta multiply + I add into the kernel.
    # Load beta_row [BC] (one scalar per row of the BC subblock, strided read).
    beta_row = tl.load(beta_p + row_off * beta_stride_within).to(tl.float32)
    # Apply beta row-wise: A_kk[i, j] *= beta[i] for all (i, j) in this BC subblock.
    A_kk_blk = A_kk_blk * beta_row[:, None]

    # Write outputs at [pid_n, s_i*BC:(s_i+1)*BC, s_j*BC:(s_j+1)*BC].
    out_offs = row_off[:, None] * BT + col_off[None, :]
    tl.store(A_qk + pid_n * BT * BT + out_offs, A_qk_blk.to(tl.bfloat16))
    # Lever K + P: for s_i == s_j pairs, only the strict-lower-tri + diagonal
    # is written. We add 1.0 to the diagonal entries (I), then mask off the
    # upper-tri. Off-diag blocks write all entries (no mask needed since
    # they're all below the global diagonal).
    if is_diag:
        # Add I (1.0 on the diagonal of the BC subblock).
        diag_mask = (offs_i[:, None] == offs_i[None, :]).to(tl.float32)
        A_kk_with_I = A_kk_blk + diag_mask
        # Mask off the upper-tri (i < j within the subblock).
        store_mask = offs_i[:, None] >= offs_i[None, :]
        tl.store(A_kk + pid_n * BT * BT + out_offs, A_kk_with_I, mask=store_mask)
    else:
        tl.store(A_kk + pid_n * BT * BT + out_offs, A_kk_blk)


def triton_intra_solve(
    q_tok: torch.Tensor,         # [T, H, K] bf16
    k_tok: torch.Tensor,         # [T, H, K] bf16
    g_cum_tok: torch.Tensor,     # [T, H, K] bf16 (cumsum, log2)
    beta_tok: torch.Tensor,      # [T, HV] bf16 (per-token beta scalars)
    scale: float,
    BT: int = 64,
    BC: int = 16,
    H: int = 12,                 # number of heads (= HV in this wrapper)
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (A_qk_bf16 [N, BT, BT], A_kk_fp32 [N, BT, BT]).

    Strided reads (Lever M): no transpose+contiguous needed on q/k/g/beta.
    g_cum_tok is the bf16 chunk-local cumsum produced by g_cumsum_fused
    (cast to fp32 in-kernel).

    Lever P (June 2026-30): A_kk_fp32 is multiplied by beta row-wise AND
    has 1.0 on the diagonal — direct output of `A = I + A_kk * beta` for
    the lower-tri entries. Forward_sub can consume this directly.

    Constraints: BT=64, BC=16, K % 16 == 0 (bf16 mma).
    """
    assert q_tok.dtype == torch.bfloat16, f"q_tok must be bf16, got {q_tok.dtype}"
    assert k_tok.dtype == torch.bfloat16, f"k_tok must be bf16, got {k_tok.dtype}"
    assert g_cum_tok.dtype == torch.bfloat16, f"g_cum_tok must be bf16, got {g_cum_tok.dtype}"
    assert beta_tok.dtype == torch.bfloat16, f"beta_tok must be bf16, got {beta_tok.dtype}"
    T, H_dim, K = q_tok.shape
    assert H_dim == H, f"H mismatch: q_tok has {H_dim}, expected {H}"
    assert BT == 64 and BC == 16, "Lever H: BT=64, BC=16 only"
    assert K % 16 == 0, f"K={K} must be a multiple of 16 for bf16 mma"
    num_chunks = T // BT
    HV = H  # wrapper requires H == HV
    N = num_chunks * HV

    # A_qk: bf16 (with scale baked in).
    # A_kk: fp32 (forward_sub kernel requires fp32). Allocated as zeros —
    # the upper-tri entries that the kernel skips stay at 0, which is
    # what we want (forward_sub only reads the lower-tri).
    A_qk = torch.zeros(N, BT, BT, device=q_tok.device, dtype=torch.bfloat16)
    A_kk = torch.zeros(N, BT, BT, device=q_tok.device, dtype=torch.float32)

    # Pair lookup on device (small, NUM_PAIRS=10).
    pair_si = torch.tensor(PAIR_SI, dtype=torch.int32, device=q_tok.device)
    pair_sj = torch.tensor(PAIR_SJ, dtype=torch.int32, device=q_tok.device)

    grid = (NUM_PAIRS, N)
    _intra_solve_kernel[grid](
        q_tok, k_tok, g_cum_tok, beta_tok,
        pair_si, pair_sj,
        A_qk, A_kk,
        scale,
        H=H, HV=HV, T=T, BT=BT, BC=BC, K=K, NUM_PAIRS=NUM_PAIRS,
        num_warps=2, num_stages=2,
    )
    return A_qk, A_kk