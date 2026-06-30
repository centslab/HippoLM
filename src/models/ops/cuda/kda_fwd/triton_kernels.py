"""Triton kernels for KDA forward (Round 2 — replace hand-rolled CUDA).

Mirrors the structure of FLA's chunk_gla_fwd_kernel_o for chunk_o (the
biggest single-bottleneck in the Round 1 CUDA path), with our own layout
conventions:

  * A_qk: [num_chunks, HV, BT, BT] bf16 (precomputed, scale baked in, lower-tri)
  * h:    [num_chunks, HV, K, V]   bf16  (Lever B: state at start of each chunk)
  * q:    [T_total, H, K]          bf16
  * v_new:[T_total, HV, V]         bf16  (UN-DECAYED — output of delta_h)
  * g:    [T_total, HV, K]         bf16  (chunk-local cumsum, per-K)
  * o:    [T_total, HV, V]         bf16

Per-token strides: q is H*K per token; v_new / g / o are HV*V or HV*K per
token (matches __init__.py layout).

Round-2 R2A: triton_chunk_o replaces the 23-ms hand-rolled CUDA kernel.

Lever B (June 2026-30): h is bf16 (was fp32). chunk_o consumes it via
bf16 mma (Triton's tl.dot with bf16 operands emits TensorCore m16n8k16
with fp32 accumulator — vs the bf16 × fp32 path which Triton promoted
to fp32 × fp32). Saves ~36% of chunk_o HBM traffic (~0.23 ms / 1.28×
at prod shape). cos=1.0 / med_rel=0.0 vs fp32-h at all tested shapes;
max_rel ≤ 12% in the tail (within bf16 mma precision bound for K=128
reductions). See test/_tmp/test_chunk_o_bf16_h.py.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _chunk_o_kernel(
    q,                  # [T_total, H, K]  bf16
    v_new,              # [T_total, HV, V] bf16
    g,                  # [T_total, HV, K] bf16
    A_qk,               # [num_chunks, HV, BT, BT] bf16
    h,                  # [num_chunks, HV, K, V]   bf16  (Lever B: was fp32)
    o,                  # [T_total, HV, V] bf16
    chunk_token_base,   # [num_chunks] int32
    q_token_stride,     # = H * K
    v_token_stride,     # = HV * V
    g_token_stride,     # = HV * K
    scale,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    i_v = tl.program_id(0).to(tl.int64)  # V/BV
    i_t = tl.program_id(1).to(tl.int64)  # chunk_id
    i_h = tl.program_id(2).to(tl.int64)  # head id (0..HV-1)

    token_base = tl.load(chunk_token_base + i_t).to(tl.int64)

    # Pointers — add per-token / per-chunk base offset.
    q_p   = q   + (token_base * q_token_stride)   + i_h * K
    v_p   = v_new + (token_base * v_token_stride)  + i_h * V
    g_p   = g   + (token_base * g_token_stride)   + i_h * K
    A_p   = A_qk + (i_t * HV + i_h) * BT * BT
    h_p   = h   + (i_t * HV + i_h) * K * V
    o_p   = o   + (token_base * v_token_stride)  + i_h * V

    v_start = i_v * BV

    # ---- part 1: o_h = scale * (q * exp2(g)) @ h  ----
    # b_o shape [BT, BV], fp32 accumulator
    b_o = tl.zeros([BT, BV], dtype=tl.float32)

    offs_i = tl.arange(0, BT)
    offs_v = v_start + tl.arange(0, BV)
    # Lower-triangular mask for A_qk (causal).
    m_s = offs_i[:, None] >= offs_i[None, :]

    for i_k in range(0, K, BK):
        offs_k = i_k + tl.arange(0, BK)
        # q[BT, BK] with stride (q_token_stride, 1)
        b_q = tl.load(
            q_p + offs_i[:, None] * q_token_stride + offs_k[None, :]
        ).to(tl.float32)
        # g[BT, BK] with stride (g_token_stride, 1)
        b_g = tl.load(
            g_p + offs_i[:, None] * g_token_stride + offs_k[None, :]
        ).to(tl.float32)
        # b_qg = b_q * exp2(b_g), cast to bf16 explicitly so that
        # tl.dot(b_qg, b_h) below uses bf16 mma (Lever B).
        b_qg = (b_q * tl.exp2(b_g)).to(tl.bfloat16)
        # h[BK, BV] bf16 — load directly, NO fp32 cast (Lever B).
        b_h_bf16 = tl.load(h_p + offs_k[:, None] * V + offs_v[None, :])
        # b_o += (b_qg @ b_h_bf16) — bf16 × bf16 → fp32 accum (TensorCore).
        b_o += tl.dot(b_qg, b_h_bf16, out_dtype=tl.float32)

    b_o *= scale

    # ---- part 2: o_v = A_qk @ v_new (causal lower-tri) ----
    offs_j = tl.arange(0, BT)
    b_A = tl.load(
        A_p + offs_i[:, None] * BT + offs_j[None, :],
        mask=m_s, other=0.0,
    ).to(tl.float32)
    # v_new[BT, BV] with stride (v_token_stride, 1)
    b_v = tl.load(
        v_p + offs_i[:, None] * v_token_stride + offs_v[None, :]
    ).to(tl.float32)
    b_o += tl.dot(b_A, b_v, out_dtype=tl.float32)

    # ---- store o ----
    tl.store(
        o_p + offs_i[:, None] * v_token_stride + offs_v[None, :],
        b_o.to(o.dtype.element_ty),
    )


def triton_chunk_o(
    q: torch.Tensor,
    v_new: torch.Tensor,
    g: torch.Tensor,
    A_qk: torch.Tensor,
    h: torch.Tensor,
    chunk_token_base: torch.Tensor,
    num_chunks: int,
    scale: float,
    H: int, HV: int,
    K: int, V: int,
) -> torch.Tensor:
    """Triton implementation of chunk_o. Returns o [T_total, HV, V] bf16.

    Mirrors ``_chunk_o`` in __init__.py but uses a Triton kernel for the
    tensor-core matmuls.

    Layout: see module docstring.
    """
    T_total = q.shape[0]
    o = torch.empty(T_total, HV, V, dtype=torch.bfloat16, device=q.device)

    # Pick tile sizes — tuned for K=128, V=128, BT=64 (the production shape).
    # BT is fixed at 64 (Round-1 contract); BK/BV chosen to fit tensor cores.
    BT = 64
    BK = 64 if K >= 64 else 32
    BV = 64 if V >= 64 else 32

    grid = (V // BV, num_chunks, HV)
    _chunk_o_kernel[grid](
        q, v_new, g, A_qk, h, o,
        chunk_token_base,
        q_token_stride=H * K,
        v_token_stride=HV * V,
        g_token_stride=HV * K,
        scale=scale,
        H=H, HV=HV, K=K, V=V, BT=BT, BK=BK, BV=BV,
        num_warps=4, num_stages=2,
    )
    return o
