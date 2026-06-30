"""Lever L — fused g cumsum Triton kernel.

Replaces 4 ops in __init__.py:281-286:
  g_log2_chunks_fp32 = (g_tok.float() * RCP_LN2).view(num_chunks, BT, HV, K)
  g_cum_fp32 = g_log2_chunks_fp32.cumsum(dim=1)
  g_cum_tok = g_cum_fp32.to(torch.bfloat16).view(T_total, HV, K).contiguous()
  g_last = g_cum_fp32[:, BT - 1, :, :].contiguous()

with ONE Triton kernel that produces both:
  - g_per    (fp32 [NC, H, BT, K]) — used by intra_solve
  - g_cum_tok (bf16 [T, H, K])     — used by delta_h, chunk_o

Grid: (H, NC). One program per (chunk, head) handles a [BT, K] tile.
Each program loads BT*K bf16 values, casts to fp32 + scales, does cumsum
along BT axis (Triton tl.cumsum = parallel scan), writes both output layouts.

The read is strided (stride = H*K between BT rows), but [BT, K]=64*128=8K
bf16 elements per program is small enough that one program per (chunk,head)
gives good coalescing within each row.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _g_cumsum_kernel(
    g_in,                 # [NC*BT, H, K] bf16 — per-token g, in NATURAL log space
    g_per_out,            # [NC, H, BT, K] fp32 — cumsum (log2) for intra_solve
    g_cum_tok_out,        # [NC*BT, H, K] bf16 — cumsum (log2) for delta_h, chunk_o
    H,                    # runtime arg: number of heads
    BT: tl.constexpr,
    K: tl.constexpr,
    RCP_LN2: tl.constexpr,
):
    pid_h = tl.program_id(0)
    pid_chunk = tl.program_id(1)

    offs_bt = tl.arange(0, BT)
    offs_k = tl.arange(0, K)

    # Load [BT, K] tile. g_in is [NC*BT, H, K]; row stride is H*K, K stride is 1.
    # Strided load: BT rows × K cols. Each row is H*K apart.
    g_tile = tl.load(
        g_in + pid_chunk * BT * H * K + pid_h * K
        + offs_bt[:, None] * (H * K) + offs_k[None, :]
    ).to(tl.float32) * RCP_LN2  # [BT, K] fp32

    # Cumsum over BT (axis=0) using Triton's parallel scan.
    g_cum = tl.cumsum(g_tile, axis=0)  # [BT, K] fp32

    # Write g_per_out: [NC, H, BT, K] fp32 at [pid_chunk, pid_h, :, :]
    # Contiguous in [NC, H, BT, K]: row stride is K.
    tl.store(
        g_per_out + pid_chunk * H * BT * K + pid_h * BT * K
        + offs_bt[:, None] * K + offs_k[None, :],
        g_cum,
    )

    # Write g_cum_tok_out: [NC*BT, H, K] bf16 at [pid_chunk*BT + b, pid_h, :]
    # Strided store: same pattern as the load.
    tl.store(
        g_cum_tok_out + pid_chunk * BT * H * K + pid_h * K
        + offs_bt[:, None] * (H * K) + offs_k[None, :],
        g_cum.to(tl.bfloat16),
    )


def g_cumsum_fused(
    g_tok: torch.Tensor,        # [NC*BT, H, K] bf16 (per-token g, natural log)
    num_chunks: int,
    H: int,
    K: int,
    BT: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (g_per [NC, H, BT, K] fp32, g_cum_tok [NC*BT, H, K] bf16).

    Replaces __init__.py:281-286 + the g_per contiguous call from:310.
    """
    assert g_tok.dtype == torch.bfloat16
    assert g_tok.is_contiguous(), f"g_tok must be contiguous [NC*BT, H, K]; got strides {g_tok.stride()}"
    assert g_tok.shape == (num_chunks * BT, H, K)
    assert K % 16 == 0, f"K={K} must be a multiple of 16"
    assert BT == 64, "BT=64 only (mask_2d is [BT, BT])"

    device = g_tok.device
    g_per = torch.empty(num_chunks, H, BT, K, device=device, dtype=torch.float32)
    g_cum_tok = torch.empty(num_chunks * BT, H, K, device=device, dtype=torch.bfloat16)
    RCP_LN2 = 1.4426950408889634

    grid = (H, num_chunks)
    _g_cumsum_kernel[grid](
        g_tok, g_per, g_cum_tok, H,
        BT=BT, K=K, RCP_LN2=RCP_LN2,
        num_warps=2, num_stages=2,
    )
    return g_per, g_cum_tok