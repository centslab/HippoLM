"""Lever I — fused wy_transform Triton kernel (replaces the 4.7 ms 2-bmm loop).

Replaces __init__.py:350-360 with ONE Triton kernel. Two cuBLAS bmms +
4 .to() casts + 2 .contiguous() + 4 element-wise ops are replaced by a
single kernel that loads A_kk_fp32 (post-forward_sub = A_inv) once into
shmem and produces both u and w.

Computation (matches __init__.py:355-360):
  beta_v = v * beta                              [N, BT, V] bf16
  u = A_inv @ beta_v                             [N, BT, V] bf16
  k_with_beta_g = k * beta * exp2(g_cum)         [N, BT, K] bf16
  w = A_inv @ k_with_beta_g                      [N, BT, K] bf16

Grid: (1, N). One program per (chunk, hv) batch — does BOTH u and w.
Loads A_kk [BT, BT] fp32 + v [BT, BV] bf16 + k [BT, BK] bf16 +
g_cum [BT, K] fp32 + beta [BT] bf16 → produces u and w.

Why ONE kernel (vs 2):
- A_kk is loaded ONCE into shmem (16 KB at BT=64), shared by both bmms.
- The RHS element-wise (v*beta, k*beta*exp2(g)) happens in registers.
- 1 launch instead of 2 — saves ~5 ms of CPU overhead.
- Eliminates 4 .to() casts + 2 intermediate (beta_v, k_with_beta_g) tensors.

Lever I savings at prod (B=1 T=16384 H=12 K=V=128):
  Reference: 4.69 ms (2 cuBLAS bmms + 4 .to() + 4 element-wise + 2 .contiguous())
  Fused:     0.89 ms (1 Triton launch, all intermediates in registers)
  Net:       ~3.8 ms saved at prod.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _wy_fused_kernel(
    A_kk,                     # [N, BT, BT] fp32 (post-forward_sub = A_inv)
    v_per_st,                 # [N, BT, BV] bf16  (full V dim = BV)
    k_per_st,                 # [N, BT, BK] bf16  (full K dim = BK)
    g_per_st,                 # [N, BT, K] fp32 (cumsum, log2)
    beta_st,                  # [N, BT] bf16
    u_out,                    # [N, BT, BV] bf16 (output)
    w_out,                    # [N, BT, BK] bf16 (output)
    BT: tl.constexpr,
    BV: tl.constexpr,
    BK: tl.constexpr,
    K: tl.constexpr,
):
    pid_n = tl.program_id(1)
    # Single program per (chunk, hv). Does BOTH u and w in one launch.
    # Future: tile over BV/BK if needed for larger V/K.

    # ---- Load A_kk [BT, BT] fp32 → bf16 in registers ----
    offs_i = tl.arange(0, BT)
    offs_j = tl.arange(0, BT)
    A_fp32 = tl.load(A_kk + pid_n * BT * BT + offs_i[:, None] * BT + offs_j[None, :])
    A_bf16 = A_fp32.to(tl.bfloat16)

    # ---- Load v_per_st [BT, BV] bf16 and beta [BT] bf16 ----
    offs_b = tl.arange(0, BV)
    v_blk = tl.load(v_per_st + pid_n * BT * BV + offs_i[:, None] * BV + offs_b[None, :])
    beta_blk = tl.load(beta_st + pid_n * BT + offs_i)
    # beta_v in registers [BT, BV] bf16 (the multiply casts to fp32 in regs)
    beta_v = (v_blk * beta_blk[:, None]).to(tl.bfloat16)

    # ---- u = A @ beta_v : bf16 mma → fp32 accum → bf16 ----
    u_blk = tl.dot(A_bf16, beta_v, out_dtype=tl.float32).to(tl.bfloat16)

    # ---- Load k_per_st [BT, BK] bf16 and g_per_st [BT, K] fp32 ----
    offs_k = tl.arange(0, K)
    k_blk = tl.load(k_per_st + pid_n * BT * BK + offs_i[:, None] * BK + offs_b[None, :])
    g_blk = tl.load(g_per_st + pid_n * BT * K + offs_i[:, None] * K + offs_k[None, :])
    # k_with_beta_g = k * beta * exp2(g) [BT, BK] bf16
    k_fp32 = k_blk.to(tl.float32)
    g_exp2 = tl.exp2(g_blk)
    beta_fp32 = beta_blk.to(tl.float32)
    kbg_fp32 = k_fp32 * beta_fp32[:, None] * g_exp2
    kbg_bf16 = kbg_fp32.to(tl.bfloat16)

    # ---- w = A @ k_with_beta_g ----
    w_blk = tl.dot(A_bf16, kbg_bf16, out_dtype=tl.float32).to(tl.bfloat16)

    # ---- Write u and w ----
    tl.store(u_out + pid_n * BT * BV + offs_i[:, None] * BV + offs_b[None, :], u_blk)
    tl.store(w_out + pid_n * BT * BK + offs_i[:, None] * BK + offs_b[None, :], w_blk)


def wy_fused_transform(
    A_kk_fp32: torch.Tensor,       # [N, BT, BT] fp32 (post-forward_sub = A_inv)
    v_per_stacked: torch.Tensor,   # [N, BT, V] bf16
    k_per_stacked: torch.Tensor,   # [N, BT, K] bf16
    g_per_stacked: torch.Tensor,   # [N, BT, K] fp32 (cumsum, log2)
    beta_stacked: torch.Tensor,    # [N, BT] bf16
    BT: int = 64,
    BV: int = 128,
    BK: int = 128,
    K: int = 128,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (u_bf16 [N, BT, V], w_bf16 [N, BT, K]).

    Replaces __init__.py:350-360 (2 cuBLAS bmms + 4 .to() casts + 2 .contiguous()
    + 4 element-wise ops). Single Triton launch.

    Constraints:
      * BT=64, BV=V, BK=K, K dim of k_with_beta_g matches V of u output.
      * V=K (production shape has V=K=128).
      * K must be a power of 2 >= 16 for the bf16 mma.
    """
    assert A_kk_fp32.dtype == torch.float32, f"A_kk_fp32 must be fp32, got {A_kk_fp32.dtype}"
    assert v_per_stacked.dtype == torch.bfloat16
    assert k_per_stacked.dtype == torch.bfloat16
    assert g_per_stacked.dtype == torch.float32
    assert beta_stacked.dtype == torch.bfloat16
    assert BT == 64, "BT=64 only"
    assert BV == BK == K, f"Lever I requires BV=BK=K (got BV={BV}, BK={BK}, K={K})"
    assert K % 16 == 0

    N = A_kk_fp32.shape[0]
    u = torch.empty(N, BT, BV, device=A_kk_fp32.device, dtype=torch.bfloat16)
    w = torch.empty(N, BT, BK, device=A_kk_fp32.device, dtype=torch.bfloat16)

    grid = (1, N)
    _wy_fused_kernel[grid](
        A_kk_fp32, v_per_stacked, k_per_stacked, g_per_stacked, beta_stacked,
        u, w,
        BT=BT, BV=BV, BK=BK, K=K,
        num_warps=4, num_stages=2,
    )
    return u, w