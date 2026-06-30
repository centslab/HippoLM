"""Lever I — fused wy_transform Triton kernel (replaces the 4.7 ms 2-bmm loop).

Strided-read layout (Lever M, June 2026-30):
  * A_kk_fp32: [N, BT, BT] fp32  (post-forward_sub = A_inv, contiguous)
  * v_tok:     [T, HV, V] bf16   (strided; natural [B, T, HV, V] flattened)
  * k_tok:     [T, H, K]  bf16   (strided)
  * g_cum_tok: [T, H, K]  bf16   (strided; cumsum, log2)
  * beta_tok:  [T, HV]    bf16   (strided)
  * u:         [N, BT, BV] bf16
  * w:         [N, BT, BK] bf16

Grid: (1, N). One program per (chunk, hv).

Computation (matches FLA wy_fast):
  beta_v = v * beta                       [BT, BV] (registers)
  u = A_inv @ beta_v                      [BT, BV] bf16
  k_with_beta_g = k * beta * exp2(g)      [BT, BK] (registers)
  w = A_inv @ k_with_beta_g               [BT, BK] bf16

Lever I savings at prod (B=1 T=16384 H=12 K=V=128):
  Reference: 4.69 ms (2 cuBLAS bmms + 4 .to() + 4 element-wise + 2 .contiguous())
  Fused:     0.89 ms (1 Triton launch, all intermediates in registers)
  Net:       ~3.8 ms saved at prod.
Lever M savings (strided reads): ~0.4 ms (eliminates v/k/beta per-tile
  .contiguous() calls in the wrapper).
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _wy_fused_kernel(
    A_kk,                     # [N, BT, BT] fp32 (post-forward_sub = A_inv)
    v_tok,                    # [T, HV, V] bf16 (strided)
    k_tok,                    # [T, H, K]  bf16 (strided)
    g_cum_tok,                # [T, H, K]  bf16 (strided; cumsum, log2)
    beta_tok,                 # [T, HV]    bf16 (strided)
    u_out,                    # [N, BT, BV] bf16 (output, contiguous)
    w_out,                    # [N, BT, BK] bf16 (output, contiguous)
    H: tl.constexpr,          # == HV in this wrapper
    HV: tl.constexpr,
    T: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
    BK: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
):
    pid_n = tl.program_id(1)
    chunk_idx = pid_n // HV
    hv_idx = pid_n - chunk_idx * HV
    chunk_start = chunk_idx * BT

    # ---- Load A_kk [BT, BT] fp32 → bf16 in registers ----
    offs_i = tl.arange(0, BT)
    offs_j = tl.arange(0, BT)
    A_fp32 = tl.load(A_kk + pid_n * BT * BT + offs_i[:, None] * BT + offs_j[None, :])
    A_bf16 = A_fp32.to(tl.bfloat16)

    # ---- Load v_tok [BT, BV] bf16 and beta [BT] bf16 ----
    # v_tok per-token stride = HV*V; per-head stride = V.
    v_stride_within = HV * V
    v_base = chunk_start * v_stride_within + hv_idx * V
    offs_b = tl.arange(0, BV)
    v_blk = tl.load(v_tok + v_base + offs_i[:, None] * v_stride_within + offs_b[None, :])

    # beta_tok per-token stride = HV; per-head stride = 1.
    beta_stride_within = HV
    beta_base = chunk_start * beta_stride_within + hv_idx
    beta_blk = tl.load(beta_tok + beta_base + offs_i * beta_stride_within)
    # beta_v in registers [BT, BV] bf16 (the multiply casts to fp32 in regs).
    beta_v = (v_blk * beta_blk[:, None]).to(tl.bfloat16)

    # ---- u = A @ beta_v : bf16 mma → fp32 accum → bf16 ----
    u_blk = tl.dot(A_bf16, beta_v, out_dtype=tl.float32).to(tl.bfloat16)

    # ---- Load k_tok [BT, BK] bf16 and g_cum_tok [BT, K] fp32 ----
    # k_tok per-token stride = H*K; per-head stride = K.
    k_stride_within = H * K
    k_base = chunk_start * k_stride_within + hv_idx * K
    offs_k = tl.arange(0, K)
    k_blk = tl.load(k_tok + k_base + offs_i[:, None] * k_stride_within + offs_b[None, :])
    g_blk = tl.load(g_cum_tok + k_base + offs_i[:, None] * k_stride_within + offs_k[None, :])
    # k_with_beta_g = k * beta * exp2(g) [BT, BK] bf16.
    k_fp32 = k_blk.to(tl.float32)
    g_exp2 = tl.exp2(g_blk.to(tl.float32))
    beta_fp32 = beta_blk.to(tl.float32)
    kbg_fp32 = k_fp32 * beta_fp32[:, None] * g_exp2
    kbg_bf16 = kbg_fp32.to(tl.bfloat16)

    # ---- w = A @ k_with_beta_g ----
    w_blk = tl.dot(A_bf16, kbg_bf16, out_dtype=tl.float32).to(tl.bfloat16)

    # ---- Write u and w (output buffers are contiguous [N, BT, BV/BK]) ----
    tl.store(u_out + pid_n * BT * BV + offs_i[:, None] * BV + offs_b[None, :], u_blk)
    tl.store(w_out + pid_n * BT * BK + offs_i[:, None] * BK + offs_b[None, :], w_blk)


def wy_fused_transform(
    A_kk_fp32: torch.Tensor,       # [N, BT, BT] fp32 (post-forward_sub = A_inv)
    v_tok: torch.Tensor,           # [T, HV, V] bf16 (strided)
    k_tok: torch.Tensor,           # [T, H, K]  bf16 (strided)
    g_cum_tok: torch.Tensor,       # [T, H, K]  bf16 (strided; cumsum, log2)
    beta_tok: torch.Tensor,        # [T, HV]    bf16 (strided)
    BT: int = 64,
    BV: int = 128,
    BK: int = 128,
    K: int = 128,
    V: int = 128,
    H: int = 12,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (u_bf16 [N, BT, V], w_bf16 [N, BT, K]).

    Strided reads (Lever M): no per-(chunk, hv) .contiguous() needed.
    Replaces the original 2-bmm + 4 .to() + 2 .contiguous() + 4
    element-wise ops with one Triton launch.

    Constraints: BT=64, BV=V, BK=K, V=K. K must be a power of 2 >= 16.
    """
    assert A_kk_fp32.dtype == torch.float32, f"A_kk_fp32 must be fp32, got {A_kk_fp32.dtype}"
    assert v_tok.dtype == torch.bfloat16
    assert k_tok.dtype == torch.bfloat16
    assert g_cum_tok.dtype == torch.bfloat16
    assert beta_tok.dtype == torch.bfloat16
    assert BT == 64, "BT=64 only"
    assert BV == BK == K == V, f"Requires V=K=BV=BK (got V={V}, K={K}, BV={BV}, BK={BK})"
    assert K % 16 == 0

    N = A_kk_fp32.shape[0]
    T = v_tok.shape[0]
    HV = v_tok.shape[1]
    u = torch.empty(N, BT, BV, device=A_kk_fp32.device, dtype=torch.bfloat16)
    w = torch.empty(N, BT, BK, device=A_kk_fp32.device, dtype=torch.bfloat16)

    grid = (1, N)
    _wy_fused_kernel[grid](
        A_kk_fp32, v_tok, k_tok, g_cum_tok, beta_tok,
        u, w,
        H=H, HV=HV, T=T, BT=BT, BV=BV, BK=BK, K=K, V=V,
        num_warps=4, num_stages=2,
    )
    return u, w