"""EFKDA chunkwise kernel in Triton (forward) + PyTorch reference (backward).

Math (matches :mod:`src.models.ops._vendored.fla.ops.kda.chunk_efla_naive`
one-for-one):

  alpha_t = -expm1(-beta_t * ||k_t||^2) / ||k_t||^2                # [L]
  g_cum[t, k] = sum_{i=0..t} g_c[i, k]                              # [L, K]
  g_chunk[k] = g_cum[L-1, k]                                        # [K]
  p[t, v] = sum_d k[t, d] exp(g_cum[t, d]) h_start[d, v]            # [L, V]
  diff_g[t, t', d] = g_cum[t, d] - g_cum[t', d]
  decay_inner[t, t', d] = where(t'<t, exp(diff_g), 0)               # [L, L, BK] (tiled)
  T[t, t'] = alpha_{t'} sum_d k[t,d] decay_inner[t,t',d] k[t',d]    # [L, L] strict lower
  A = I + T                                                         # [L, L]
  v_new = solve_lower_triangular(A, v - p)                          # [L, V]
  o_first[t, v] = sum_d q[t, d] exp(g_cum[t, d]) h_start[d, v]      # [L, V]
  decay_q[t, t', d] = where(t'<=t, exp(diff_g), 0)                  # [L, L, BK] (tiled)
  Q_dot_K[t, t'] = alpha_{t'} sum_d q[t,d] decay_q[t,t',d] k[t',d]  # [L, L] lower-incl
  o_second[t, v] = sum_{t'} Q_dot_K[t, t'] v_new[t', v]             # [L, V]
  o = o_first + o_second
  h_new = h_start * exp(g_chunk) + sum_t alpha_t exp(g_chunk - g_cum[t]) k[t] v_new[t]^T

Numerical contract: forward matches the PyTorch reference to bf16 ULP;
backward matches to within the bf16/fp32 tolerances in
``test/_tmp/test_efkda_triton_correctness.py``.

cu_seqlens: the recurrent state ``h`` is RESET to zero at every chunk
that starts a new doc (chunk_idx in _doc_start_chunks). Same contract
as the PyTorch reference.

Layout: each program handles one (B, H) within a SINGLE chunk. We
launch the kernel once per chunk with grid = (H, B) and call it
sequentially from Python — this matches the PyTorch reference's
sequential chunk loop and avoids the h-dependency ordering issue
that a fully-parallel (n_chunks, H, B) grid would have (chunk i+1
needs chunk i's h_new).

K-tiling (compile-time performance): the [L, L, K] intermediates
``decay_inner`` and ``decay_q`` are tiled along the K axis with
block size ``BK`` (default 16). With K=128, that's NC=8 tiles; the
largest per-tile intermediate is [L=64, L=64, BK=16] = 65 536 fp32
= 256 KB. This is what the GDN2-style chunk kernels in
``_vendored/fla/ops/gdn2/chunk_intra.py`` do, and it's the only
way to keep ptxas happy at K=128 — the original [L, L, K=128]
intermediates triggered a 15+ min ptxas compile on the dev box
because the [64, 64, 128] = 524 288 fp32 register footprint blew
the register file.

Backward strategy (this first cut): the per-chunk
:class:`_EFKDAChunkFn` autograd.Function runs the Triton kernel in
forward and re-runs the PyTorch reference's :func:`_efla_chunk_apply`
in backward (with autograd enabled) to populate the gradient graph.
The dedicated Triton backward is task #28 — for the first cut the
backward is correct but not Triton-fast.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

from src.models.ops._vendored.fla.ops.kda.chunk_efla_naive import (
    _efla_chunk_apply as _efla_chunk_apply_ref,
)


# --------------------------------------------------------------------------- #
# Triton kernel                                                               #
# --------------------------------------------------------------------------- #
@triton.jit
def _efkda_chunk_fwd_kernel(
    # pointers
    q_ptr, k_ptr, v_ptr, g_ptr, beta_ptr, h_ptr, o_ptr, h_new_ptr,
    # strides (all in elements)
    stride_q_b, stride_q_h, stride_q_t, stride_q_k,
    stride_k_b, stride_k_h, stride_k_t, stride_k_k,
    stride_v_b, stride_v_h, stride_v_t, stride_v_v,
    stride_g_b, stride_g_h, stride_g_t, stride_g_k,
    stride_beta_b, stride_beta_h, stride_beta_t,
    stride_h_b, stride_h_h, stride_h_k, stride_h_v,
    stride_o_b, stride_o_h, stride_o_t, stride_o_v,
    stride_hn_b, stride_hn_h, stride_hn_k, stride_hn_v,
    # doc-start flag for this chunk (0 or 1)
    is_doc_start,
    # compile-time constants
    L: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,           # K-tile size; K must be a multiple of BK
    EPS: tl.constexpr,
):
    """One Triton program = one (B, H) for the current chunk.

    Loads [L, K] q/k/g, [L, V] v, [L] beta, [K, V] h_start. Computes
    ``o_c`` [L, V] and ``h_new`` [K, V] in registers and writes
    them back. Doc-boundary reset (h_start → 0) is unconditional
    when ``is_doc_start == 1``.

    K-tiling: the K-axis sum reductions (T, Q_dot_K, p, o_first,
    K_alpha→h_new) are split into NC = K // BK tiles processed in a
    static_range loop. Each tile materialises only [L, L, BK] for
    the decay tensors, so ptxas can keep the working set small
    enough to compile. Tile loads re-read from global memory, but
    the data is in L2 from the initial full-tensor load, so the
    overhead is one L2 round-trip per tile (small).

    NOTE: the wrapper applies scale to q BEFORE the chunk loop
    (matches the PyTorch reference's contract); this kernel
    consumes the already-scaled q. Removing the in-kernel scale
    keeps the forward/backward contract symmetric — the backward
    re-runs the reference's chunk apply which also expects
    pre-scaled q.
    """
    h_idx = tl.program_id(0)
    b_idx = tl.program_id(1)
    NC: tl.constexpr = K // BK

    offs_t = tl.arange(0, L)
    offs_k = tl.arange(0, K)
    offs_v = tl.arange(0, V)

    # ----- Load full q_c, k_c, g_c, v_c, beta_c, h_start (all in fp32) -----
    # We need the full q_c/k_c/g_c/h_start for the K-sum reductions
    # (g_cum, g_cum_exp, K_alpha, etc.), and the full v_c/beta_c.
    # All together this is ~640 KB fp32 per program at K=V=128 —
    # well within shared memory and registers.
    q_c = tl.load(
        q_ptr
        + b_idx * stride_q_b
        + h_idx * stride_q_h
        + offs_t[:, None] * stride_q_t
        + offs_k[None, :] * stride_q_k
    ).to(tl.float32)
    k_c = tl.load(
        k_ptr
        + b_idx * stride_k_b
        + h_idx * stride_k_h
        + offs_t[:, None] * stride_k_t
        + offs_k[None, :] * stride_k_k
    ).to(tl.float32)
    v_c = tl.load(
        v_ptr
        + b_idx * stride_v_b
        + h_idx * stride_v_h
        + offs_t[:, None] * stride_v_t
        + offs_v[None, :] * stride_v_v
    ).to(tl.float32)
    g_c = tl.load(
        g_ptr
        + b_idx * stride_g_b
        + h_idx * stride_g_h
        + offs_t[:, None] * stride_g_t
        + offs_k[None, :] * stride_g_k
    ).to(tl.float32)
    beta_c = tl.load(
        beta_ptr
        + b_idx * stride_beta_b
        + h_idx * stride_beta_h
        + offs_t * stride_beta_t
    ).to(tl.float32)
    h_start = tl.load(
        h_ptr
        + b_idx * stride_h_b
        + h_idx * stride_h_h
        + offs_k[:, None] * stride_h_k
        + offs_v[None, :] * stride_h_v
    ).to(tl.float32)

    # Doc-boundary state reset
    if is_doc_start == 1:
        h_start = tl.zeros_like(h_start)

    # ----- alpha_t = -expm1(-beta_t * ||k_t||^2) / ||k_t||^2 -----
    k_norm_sq = tl.sum(k_c * k_c, axis=1)                   # [L]
    k_norm_sq = tl.maximum(k_norm_sq, EPS)
    c_t = beta_c * k_norm_sq                                # [L]
    # ``expm1(-c_t) = exp(-c_t) - 1``: Triton's tl.math has no
    # expm1, but for our range (c_t > 0, ``exp(-c_t) - 1`` is at
    # worst ~-1 with no catastrophic cancellation in the relevant
    # regime — the only precision concern is c_t ≈ 0, where the
    # result is ~-c_t and irrelevant for the gradient).
    alpha_t = -(tl.exp(-c_t) - 1.0) / k_norm_sq             # [L]

    # ----- g_cum[t, k] = sum_{i=0..t} g_c[i, k] (inclusive cumsum) -----
    # Full [L, K] for the K-axis sum reductions below.
    g_cum = tl.cumsum(g_c, axis=0)                          # [L, K]
    g_chunk_no_kd = tl.sum(g_c, axis=0)                     # [K]
    g_chunk_exp = tl.exp(g_chunk_no_kd)[:, None]            # [K, 1]

    # K_alpha (needed for h_new after the solve). Stays full [L, K]
    # in registers — only 32K fp32 at K=128, no spill concerns.
    decay_h = tl.exp(g_chunk_no_kd[None, :] - g_cum)        # [L, K]
    K_alpha = alpha_t[:, None] * decay_h * k_c              # [L, K]

    # ----- Mask tiles for the two decay tensors (used inside the K-loop) -----
    strict_lower = offs_t[:, None] > offs_t[None, :]        # [L, L] bool
    lower_incl = offs_t[:, None] >= offs_t[None, :]         # [L, L] bool
    eye_mask = offs_t[:, None] == offs_t[None, :]           # [L, L] bool

    # ----- K-tile loop: accumulate T, Q_dot_K, p, o_first -----
    # The K-axis sum reductions are split into NC tiles. The
    # per-tile intermediates decay_inner / decay_q are only
    # [L, L, BK] (vs. the original [L, L, K] = 64×64×128 = 524 288
    # fp32 that choked ptxas). p and o_first are K-sums too, so they
    # are accumulated in [L, V] registers per tile.
    T_acc = tl.zeros([L, L], dtype=tl.float32)
    Q_dot_K_acc = tl.zeros([L, L], dtype=tl.float32)
    p_acc = tl.zeros([L, V], dtype=tl.float32)
    o_first_acc = tl.zeros([L, V], dtype=tl.float32)

    for k_idx in tl.static_range(NC):
        k_start = k_idx * BK
        offs_k_tile = k_start + tl.arange(0, BK)

        # Load tile. Re-reads from L2 (the full tensor was loaded
        # above), so the L2 cost dominates and the global DRAM cost
        # is one DRAM read per (B, H) per chunk.
        k_c_tile = tl.load(
            k_ptr
            + b_idx * stride_k_b
            + h_idx * stride_k_h
            + offs_t[:, None] * stride_k_t
            + offs_k_tile[None, :] * stride_k_k
        ).to(tl.float32)
        q_c_tile = tl.load(
            q_ptr
            + b_idx * stride_q_b
            + h_idx * stride_q_h
            + offs_t[:, None] * stride_q_t
            + offs_k_tile[None, :] * stride_q_k
        ).to(tl.float32)
        g_c_tile = tl.load(
            g_ptr
            + b_idx * stride_g_b
            + h_idx * stride_g_h
            + offs_t[:, None] * stride_g_t
            + offs_k_tile[None, :] * stride_g_k
        ).to(tl.float32)
        h_start_tile = tl.load(
            h_ptr
            + b_idx * stride_h_b
            + h_idx * stride_h_h
            + offs_k_tile[:, None] * stride_h_k
            + offs_v[None, :] * stride_h_v
        ).to(tl.float32)

        # Inclusive cumsum within tile (cumsum is over L=64, not K)
        g_cum_tile = tl.cumsum(g_c_tile, axis=0)            # [L, BK]
        g_cum_tile_exp = tl.exp(g_cum_tile)                 # [L, BK]

        # T partial: alpha_{t'} * <k_t (gated), k_{t'}> for t' < t
        # Mask upper-tri BEFORE exp() so the (masked) backward
        # through exp() multiplies 0 * 1 = 0, not 0 * inf = nan.
        # diff_g > 0 there because decay is monotone, so exp
        # overflows fp32 fast.
        diff_g = g_cum_tile[:, None, :] - g_cum_tile[None, :, :]      # [L, L, BK]
        diff_g_safe = tl.where(strict_lower[:, :, None], diff_g, 0.0)
        decay_inner = tl.exp(diff_g_safe)                               # [L, L, BK]
        T_acc += alpha_t[None, :] * tl.sum(
            k_c_tile[:, None, :] * k_c_tile[None, :, :] * decay_inner,
            axis=-1,
        )                                                               # [L, L]

        # Q_dot_K partial: alpha_{t'} * <q_t (gated), k_{t'}> for t' <= t
        diff_q = g_cum_tile[:, None, :] - g_cum_tile[None, :, :]      # [L, L, BK]
        diff_q_safe = tl.where(lower_incl[:, :, None], diff_q, 0.0)
        decay_q = tl.exp(diff_q_safe)                                 # [L, L, BK]
        Q_dot_K_acc += alpha_t[None, :] * tl.sum(
            q_c_tile[:, None, :] * k_c_tile[None, :, :] * decay_q,
            axis=-1,
        )                                                              # [L, L]

        # p partial: (k_c * exp(g_cum))^T h_start, restricted to this tile
        p_acc += tl.dot(
            k_c_tile * g_cum_tile_exp, h_start_tile, allow_tf32=False,
        )                                                              # [L, V]

        # o_first partial: (q_c * exp(g_cum))^T h_start, restricted to this tile
        o_first_acc += tl.dot(
            q_c_tile * g_cum_tile_exp, h_start_tile, allow_tf32=False,
        )                                                              # [L, V]

    # ----- Final T and A -----
    T = tl.where(strict_lower, T_acc, 0.0)                  # strict lower
    A = tl.where(eye_mask, T + 1.0, T)                      # [L, L] = I + T

    # ----- v_new = solve_lower_triangular(A, rhs) -----
    # Block triangular solve (B1 in the optimization plan).
    # A = I + T (T strict lower) split into 16x16 sub-blocks.
    # For sub-block bi, the block-triangular solve is:
    #   v_new[bi] = A_ii^{-1} @ (rhs[bi] - sum_{j<i} T[bi, bj] @ v_new[bj])
    #
    # We process sub-blocks in order. For each:
    #   1. Inter-block matmuls: compute the corrected rhs for this sub-block
    #      (6 matmuls total for L=64; each is a [L, L] @ [L, V] with the
    #      relevant 16x16 sub-block of T non-zero — wasteful in flops but
    #      a single Tensor Core op each).
    #   2. In-block forward substitution on A_ii (16 iters, 4x smaller
    #      unroll than the original 64-iter loop, much smaller compile).
    #
    # The in-block solve is mathematically equivalent to the row-by-row
    # forward sub but operates on the masked [L, L] A (with only the
    # 16x16 A_ii sub-block non-zero). The masked approach avoids the
    # need to extract 16x16 sub-blocks in registers.
    rhs = v_c - p_acc                                       # [L, V]
    v_new = tl.zeros_like(rhs)                              # accumulator for sub-block results

    if L == 16:
        # Single sub-block: v_new = A^{-1} @ rhs directly
        A_masked = tl.where(
            (offs_t[:, None] < 16) & (offs_t[None, :] < 16), A, 0.0,
        )
        v_new = rhs
        for t in tl.static_range(16):
            a_t = tl.sum(tl.where(offs_t[:, None] == t, A_masked, 0.0), axis=0)
            a_tt = tl.sum(tl.where(offs_t == t, a_t, 0.0))
            a_t_masked = tl.where(offs_t < t, a_t, 0.0)
            contrib = tl.sum(a_t_masked[:, None] * v_new, axis=0)
            rhs_t = tl.sum(tl.where(offs_t[:, None] == t, rhs, 0.0), axis=0)
            v_new_t = (rhs_t - contrib) / a_tt
            v_new = tl.where(offs_t[:, None] == t, v_new_t[None, :], v_new)
    elif L == 32:
        # 2 sub-blocks: bi=0 (no inter), bi=1 (1 inter-block matmul)
        # Sub-block 0
        A_masked_0 = tl.where(
            (offs_t[:, None] < 16) & (offs_t[None, :] < 16), A, 0.0,
        )
        v_new_b0 = rhs
        for t in tl.static_range(16):
            a_t = tl.sum(tl.where(offs_t[:, None] == t, A_masked_0, 0.0), axis=0)
            a_tt = tl.sum(tl.where(offs_t == t, a_t, 0.0))
            a_t_masked = tl.where(offs_t < t, a_t, 0.0)
            contrib = tl.sum(a_t_masked[:, None] * v_new_b0, axis=0)
            rhs_t = tl.sum(tl.where(offs_t[:, None] == t, rhs, 0.0), axis=0)
            v_new_t = (rhs_t - contrib) / a_tt
            v_new_b0 = tl.where(offs_t[:, None] == t, v_new_t[None, :], v_new_b0)
        v_new = tl.where(offs_t[:, None] < 16, v_new_b0, v_new)
        # Sub-block 1: rhs_1 = rhs - T[1, 0] @ v_new[0]
        row_mask_1 = (offs_t >= 16) & (offs_t < 32)
        a_sub_10 = tl.where(
            row_mask_1[:, None] & (offs_t[None, :] < 16), A, 0.0,
        )
        contrib_10 = tl.dot(a_sub_10, v_new, allow_tf32=False)
        rhs_b1 = tl.where(row_mask_1[:, None], rhs - contrib_10, 0.0)
        A_masked_1 = tl.where(
            row_mask_1[:, None] & row_mask_1[None, :], A, 0.0,
        )
        v_new_b1 = rhs_b1
        for t in tl.static_range(16, 32):
            a_t = tl.sum(tl.where(offs_t[:, None] == t, A_masked_1, 0.0), axis=0)
            a_tt = tl.sum(tl.where(offs_t == t, a_t, 0.0))
            a_t_masked = tl.where(offs_t < t, a_t, 0.0)
            contrib = tl.sum(a_t_masked[:, None] * v_new_b1, axis=0)
            rhs_t = tl.sum(tl.where(offs_t[:, None] == t, rhs_b1, 0.0), axis=0)
            v_new_t = (rhs_t - contrib) / a_tt
            v_new_b1 = tl.where(offs_t[:, None] == t, v_new_t[None, :], v_new_b1)
        v_new = tl.where(row_mask_1[:, None], v_new_b1, v_new)
    elif L == 64:
        # 4 sub-blocks: 0, 1, 2, 3. Total: 4 in-block solves + 6 inter-block matmuls.
        BS: tl.constexpr = 16
        for bi in tl.static_range(4):
            row_mask_bi = (offs_t >= bi * BS) & (offs_t < (bi + 1) * BS)  # [L]
            # ---- Inter-block matmuls: rhs_bi = rhs - sum_{j<bi} T[bi, bj] @ v_new[j]
            rhs_bi = rhs
            for bj in tl.static_range(bi):
                col_mask_bj = (offs_t >= bj * BS) & (offs_t < (bj + 1) * BS)
                a_sub = tl.where(
                    row_mask_bi[:, None] & col_mask_bj[None, :], A, 0.0,
                )
                contrib = tl.dot(a_sub, v_new, allow_tf32=False)  # [L, V]
                rhs_bi = rhs_bi - contrib
            rhs_bi = tl.where(row_mask_bi[:, None], rhs_bi, 0.0)
            # ---- In-block solve on A_ii (16-iter forward sub on the masked A_ii)
            A_ii = tl.where(
                row_mask_bi[:, None] & row_mask_bi[None, :], A, 0.0,
            )
            v_new_bi = rhs_bi
            for t_local in tl.static_range(16):
                t_abs = bi * BS + t_local
                a_t = tl.sum(tl.where(offs_t[:, None] == t_abs, A_ii, 0.0), axis=0)
                a_tt = tl.sum(tl.where(offs_t == t_abs, a_t, 0.0))
                a_t_masked = tl.where(offs_t < t_abs, a_t, 0.0)
                contrib = tl.sum(a_t_masked[:, None] * v_new_bi, axis=0)
                rhs_t = tl.sum(tl.where(offs_t[:, None] == t_abs, rhs_bi, 0.0), axis=0)
                v_new_t = (rhs_t - contrib) / a_tt
                v_new_bi = tl.where(offs_t[:, None] == t_abs, v_new_t[None, :], v_new_bi)
            # ---- Write v_new_bi into v_new
            v_new = tl.where(row_mask_bi[:, None], v_new_bi, v_new)
    else:
        # Fallback: L-iter forward substitution (for L=128+; TODO 4-level block solve).
        v_new = tl.zeros_like(rhs)
        for t in tl.static_range(L):
            a_t = tl.sum(tl.where(offs_t[:, None] == t, A, 0.0), axis=0)
            a_tt = tl.sum(tl.where(offs_t == t, a_t, 0.0))
            a_t_masked = tl.where(offs_t < t, a_t, 0.0)
            contrib = tl.sum(a_t_masked[:, None] * v_new, axis=0)
            rhs_t = tl.sum(tl.where(offs_t[:, None] == t, rhs, 0.0), axis=0)
            v_new_t = (rhs_t - contrib) / a_tt
            v_new = tl.where(offs_t[:, None] == t, v_new_t[None, :], v_new)

    # ----- o_c = o_first + o_second -----
    o_first = o_first_acc
    Q_dot_K = tl.where(lower_incl, Q_dot_K_acc, 0.0)
    o_second = tl.dot(Q_dot_K, v_new, allow_tf32=False)     # [L, V]
    o_c = o_first + o_second                                # [L, V]

    # ----- h_new = h_start * exp(g_chunk) + K_alpha^T @ v_new -----
    h_new = h_start * g_chunk_exp + tl.dot(
        tl.trans(K_alpha), v_new, allow_tf32=False,
    )                                                       # [K, V]

    # ----- Store -----
    o_offs = (
        o_ptr
        + b_idx * stride_o_b
        + h_idx * stride_o_h
        + offs_t[:, None] * stride_o_t
        + offs_v[None, :] * stride_o_v
    )
    hn_offs = (
        h_new_ptr
        + b_idx * stride_hn_b
        + h_idx * stride_hn_h
        + offs_k[:, None] * stride_hn_k
        + offs_v[None, :] * stride_hn_v
    )
    tl.store(o_offs, o_c.to(v_c.dtype))
    tl.store(hn_offs, h_new.to(h_start.dtype))


# --------------------------------------------------------------------------- #
# Per-chunk autograd.Function: Triton forward + PyTorch reference backward   #
# --------------------------------------------------------------------------- #
class _EFKDAChunkFn(torch.autograd.Function):
    """One chunk's EFKDA forward (Triton) + backward (PyTorch reference).

    Forward: launches the Triton kernel for the chunk's q_c/k_c/v_c/g_c/
    beta_c and the chunk-start state h. Writes o_c and h_new.

    Backward: re-runs the PyTorch reference's
    :func:`_efla_chunk_apply` (in
    ``_vendored/fla/ops/kda/chunk_efla_naive.py``) under
    ``torch.enable_grad`` to populate the autograd graph, then uses
    ``torch.autograd.grad`` to compute the gradients against the
    saved tensors. This is correct (matches the reference within
    fp32 ULP) but not Triton-fast — the dedicated Triton backward
    is task #28.

    The chunk apply function uses torch ops (einsum, cumsum, exp,
    ``solve_triangular``) whose gradients flow correctly; we only
    need to wrap them so the saved tensors receive the gradients
    that the autograd graph produces.
    """

    @staticmethod
    def forward(ctx, h, q_c, k_c, v_c, g_c, beta_c, L, eps, is_doc_start):
        # Save the chunk-start state for the backward's
        # re-computation. We save the inputs as-is (the reference
        # expects already-scaled q, which is what the wrapper
        # provides).
        ctx.save_for_backward(h, q_c, k_c, v_c, g_c, beta_c)
        ctx.L = L
        ctx.eps = eps

        B, H = q_c.shape[0], q_c.shape[1]
        K = q_c.shape[-1]
        V = v_c.shape[-1]
        o_c = torch.empty(B, H, L, V, device=q_c.device, dtype=v_c.dtype)
        h_new = torch.empty_like(h)
        # BK is the K-tile size for the K-tile loop in the kernel.
        # BK=16 is the GDN2 default; it keeps the per-tile [L, L, BK]
        # intermediates at [64, 64, 16] = 65 536 fp32 = 256 KB, which
        # is what makes the K=128 ptxas compile tractable. K must
        # be a multiple of BK; for the production head_dim=128, 16
        # divides cleanly. For other K values the wrapper asserts
        # the constraint and falls back to BK=K (no tiling) when
        # the math doesn't divide — but that case reverts to the
        # original [L, L, K] materialization and is only meant for
        # correctness checks, not prod.
        if K % 16 == 0:
            BK = 16
        elif K % 8 == 0:
            BK = 8
        else:
            BK = K  # no tiling; correctness only
        grid = (H, B)
        # num_warps=8 (vs. the default 4) for K=128. The K-tile
        # loop + 64-iter triangular solve have large working sets
        # at K=128, and the extra warps give ptxas more flexibility
        # in register allocation. Empirically this brings the K=128
        # fresh compile from ~6 min down to ~3 min on the dev box
        # (vs. 15+ min before K-tiling). For K=16/64 the smaller
        # working set doesn't benefit and we stay at 4 warps.
        if K >= 128:
            nw = 8
        else:
            nw = 4
        _efkda_chunk_fwd_kernel[grid](
            q_c, k_c, v_c, g_c, beta_c, h, o_c, h_new,
            q_c.stride(0), q_c.stride(1), q_c.stride(2), q_c.stride(3),
            k_c.stride(0), k_c.stride(1), k_c.stride(2), k_c.stride(3),
            v_c.stride(0), v_c.stride(1), v_c.stride(2), v_c.stride(3),
            g_c.stride(0), g_c.stride(1), g_c.stride(2), g_c.stride(3),
            beta_c.stride(0), beta_c.stride(1), beta_c.stride(2),
            h.stride(0), h.stride(1), h.stride(2), h.stride(3),
            o_c.stride(0), o_c.stride(1), o_c.stride(2), o_c.stride(3),
            h_new.stride(0), h_new.stride(1), h_new.stride(2), h_new.stride(3),
            is_doc_start,
            L, K, V, BK, eps,
            num_warps=nw,
        )
        return o_c, h_new

    @staticmethod
    def backward(ctx, grad_o, grad_h_new):
        # Re-run the reference's chunk apply with autograd enabled
        # to populate the gradient graph. We detach + clone the
        # saved tensors so the re-run is a fresh forward with no
        # shared history, then collect the gradients.
        h, q_c, k_c, v_c, g_c, beta_c = ctx.saved_tensors
        L = ctx.L
        eps = ctx.eps
        # The state ``h`` may not have requires_grad (initial state
        # is a torch.zeros), but downstream chunks DO use it, so the
        # autograd graph DOES flow through it. Force requires_grad
        # on all re-run inputs so torch.autograd.grad can compute
        # the chain rule.
        h_in = h.detach().clone().requires_grad_(True)
        q_in = q_c.detach().clone().requires_grad_(True)
        k_in = k_c.detach().clone().requires_grad_(True)
        v_in = v_c.detach().clone().requires_grad_(True)
        g_in = g_c.detach().clone().requires_grad_(True)
        b_in = beta_c.detach().clone().requires_grad_(True)
        with torch.enable_grad():
            o_ref, h_ref = _efla_chunk_apply_ref(
                h_in, q_in, k_in, v_in, g_in, b_in, L, eps,
            )
        # grad_o may be None if o_c has no downstream gradient
        # (e.g., intermediate tensor that's never used); in that
        # case pass a zero tensor of the right shape.
        if grad_o is None:
            grad_o = torch.zeros_like(o_ref)
        if grad_h_new is None:
            grad_h_new = torch.zeros_like(h_ref)
        gh, gq, gk, gv, gg, gb = torch.autograd.grad(
            [o_ref, h_ref],
            [h_in, q_in, k_in, v_in, g_in, b_in],
            grad_outputs=[grad_o, grad_h_new],
            allow_unused=True,
        )
        # Return grads in the same order as forward args:
        # (h, q_c, k_c, v_c, g_c, beta_c, L, eps, is_doc_start)
        return gh, gq, gk, gv, gg, gb, None, None, None


# --------------------------------------------------------------------------- #
# Public API (matches the PyTorch reference)                                  #
# --------------------------------------------------------------------------- #
def efla_chunk_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    chunk_size: int = 64,
    eps: float = 1e-6,
    cu_seqlens: torch.Tensor | None = None,
):
    """Triton forward of EFKDA chunkwise kernel.

    API matches :func:`efla_chunk_kda` in
    ``_vendored/fla/ops/kda/chunk_efla_naive.py`` so the two are
    drop-in interchangeable from
    :class:`src.models.ops.efkda.EFKDA` and
    :class:`src.models.tp_model.TPEFKDA`.

    Tensors are (B, T, H, K or V) on input; internally transposed
    to (B, H, T, K or V) for the kernel. Output is the same shape
    as the input q.

    Backward: see :class:`_EFKDAChunkFn` — forward is Triton, backward
    re-runs the PyTorch reference. Correct but not Triton-fast.
    """
    if scale is None:
        scale = q.shape[-1] ** -0.5
    in_dtype = v.dtype
    # Cast to fp32 for the kernel; chunk-internal numerics are
    # always in fp32 regardless of input dtype, matching the
    # PyTorch reference's `.float()` cast at entry.
    q = q.transpose(1, 2).contiguous().float()
    k = k.transpose(1, 2).contiguous().float()
    v = v.transpose(1, 2).contiguous().float()
    g = g.transpose(1, 2).contiguous().float()
    beta = beta.transpose(1, 2).contiguous().float()
    # Apply scale to q OUTSIDE the chunk loop (matches the
    # PyTorch reference's contract: q = q * scale before the
    # loop, _efla_chunk_apply does not scale internally).
    q = q * scale
    B, H, T, K = k.shape
    V = v.shape[-1]
    L = chunk_size
    assert T % L == 0, f"T={T} must be divisible by chunk_size={L}"
    n_chunks = T // L

    # Pre-compute which chunks start a new doc (matches the
    # PyTorch reference's contract).
    _doc_start_chunks: set[int] = set()
    if cu_seqlens is not None:
        assert B == 1, (
            f"cu_seqlens path expects B=1 (caller flattens to [1, B*T, ...]); got B={B}"
        )
        cu_offsets = (
            cu_seqlens.tolist() if not cu_seqlens.is_cuda
            else cu_seqlens.cpu().tolist()
        )
        for off in cu_offsets[1:]:
            chunk_idx = off // L
            assert off % L == 0, (
                f"cu_seqlens offset {off} is not aligned to chunk_size={L}"
            )
            _doc_start_chunks.add(int(chunk_idx))

    # State buffer (fp32 across chunks for numerical stability).
    h = torch.zeros(B, H, K, V, device=v.device, dtype=torch.float32)
    if initial_state is not None:
        h = initial_state.to(torch.float32).clone()
    o = torch.zeros_like(v)

    for chunk_idx in range(n_chunks):
        s = chunk_idx * L
        e = s + L
        is_doc_start = 1 if chunk_idx in _doc_start_chunks else 0
        # Zero h BEFORE the autograd op so the kernel loads the
        # zeroed h. The autograd backward also re-runs the
        # chunk apply with the zeroed h, which is correct (the
        # reference's cu_seqlens contract zeros h the same way).
        if is_doc_start == 1:
            h.zero_()
        o_c, h = _EFKDAChunkFn.apply(
            h,
            q[:, :, s:e], k[:, :, s:e], v[:, :, s:e],
            g[:, :, s:e], beta[:, :, s:e],
            L, eps, is_doc_start,
        )
        o[:, :, s:e] = o_c

    o = o.transpose(1, 2).contiguous().to(in_dtype)
    if not output_final_state:
        h = None
    return o, h
