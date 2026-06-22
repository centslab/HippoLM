"""EFKDA chunkwise backward kernel in Triton.

The bwd kernel is the "fast path" complement to the fwd kernel in
:mod:`src.models.ops.efkda_triton`. The fwd already runs at 163.5ms
(prod dims B=4 T=4096 H=8 K=V=128 bf16); this bwd kernel is the
math-correct Triton port that, when it fits in 99KB shmem, brings
the bwd into the millisecond regime.

Math summary (per chunk; A = I + T, T strict lower; L=64, K=V=128):

  Forward (matches the fwd kernel):
    alpha_t = -expm1(-beta_t * ||k_t||^2) / ||k_t||^2
    g_cum = cumsum(g_c, axis=0)            # [L, K]
    g_chunk[k] = g_cum[L-1, k]              # [K]
    p[t, v] = sum_d k[t,d] * exp(g_cum[t,d]) * h_start[d, v]   # [L, V]
    decay_inner[t, t', d] = where(t'<t, exp(g_cum[t,d]-g_cum[t',d]), 0)
    T[t, t'] = alpha_{t'} * sum_d k[t,d] * decay_inner * k[t',d]   # [L, L]
    A = I + T                            # [L, L]
    v_new = solve_lower_triangular(A, v - p)        # [L, V]
    o_first[t, v] = sum_d q[t,d] * exp(g_cum[t,d]) * h_start[d, v]
    decay_q[t, t', d] = where(t'<=t, exp(...), 0)
    Q_dot_K[t, t'] = alpha_{t'} * sum_d q[t,d] * decay_q * k[t',d]
    o_second = Q_dot_K @ v_new
    o = o_first + o_second
    K_alpha[t, k] = alpha_t * exp(g_chunk[k] - g_cum[t, k]) * k[t, k]
    h_new = h_start * exp(g_chunk) + K_alpha^T @ v_new

  Backward:
    grad_v_new = Q_dot_K^T @ grad_o + K_alpha @ grad_h_new
    grad_u = solve_lower_triangular(A^T, grad_v_new)   # upper-tri solve
    grad_v = grad_u
    grad_p = -grad_u
    grad_h = grad_h_new * exp(g_chunk) + (q * exp(g_cum))^T @ grad_o
    grad_h += (k * exp(g_cum))^T @ grad_p
    grad_A = -grad_u @ v_new^T
    grad_T = grad_A (masked strict lower)
    grad_Q_dot_K = grad_o @ v_new^T (masked lower-incl)
    grad_K_alpha = v_new @ grad_h_new^T            # [L, K]

[BS, BS, BK] sub-tile approach (2026-06-19): The bwd per-tile loop
backprops through T and Q_dot_K, which require [L, L, BK] intermediates
(N_tile, M1, M2, Nq, Mq1, Mq2). At L=64, BK=16, K=128, each is 256KB —
and 6 of them simultaneously exceed the 99KB shmem budget on the 5060 Ti.
The fix: process L=64 in 4x4 sub-blocks of [BS=16, BS=16, BK=16] =
4KB per sub-tile. The K-axis sum (over BK) is still tiled via NC=K/BK
= 8 outer iterations; only the L axis is sub-tiled. Per-sub-tile
working set is ~4-8KB (the [BS, BS, BK] sub-tile + a [L, BS, BK]
scatter mask). Total per-(ti, tj) iteration: <20KB.

The K_alpha, p, o_first, and alpha backward paths do NOT use L×L
intermediates and are kept as [L, BK] operations (with sub-tile
loops over L for the ones that depend on v_new, since v_new [L, V]
is too large to materialize as a single broadcast).

Hardware budget: target 99KB shmem on 5060 Ti. The main [L, NC, BK]
accumulators (3 × 32KB = 96KB at K=128) are kept in registers by the
Triton compiler when possible; if not, they spill and we may need
to reduce BK to 8 (NC=16, accumulators become 3 × 16KB = 48KB).

Layout: one program per (B, H) per chunk, grid (H, B), called from
Python sequentially across chunks.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


# --------------------------------------------------------------------------- #
# Triton backward kernel                                                     #
# --------------------------------------------------------------------------- #
@triton.jit
def _efkda_chunk_bwd_kernel(
    # pointers
    q_ptr, k_ptr, v_ptr, g_ptr, beta_ptr, h_ptr,
    grad_o_ptr, grad_h_new_ptr,
    grad_q_ptr, grad_k_ptr, grad_v_ptr, grad_g_ptr, grad_beta_ptr, grad_h_ptr,
    # strides (all in elements)
    stride_q_b, stride_q_h, stride_q_t, stride_q_k,
    stride_k_b, stride_k_h, stride_k_t, stride_k_k,
    stride_v_b, stride_v_h, stride_v_t, stride_v_v,
    stride_g_b, stride_g_h, stride_g_t, stride_g_k,
    stride_beta_b, stride_beta_h, stride_beta_t,
    stride_h_b, stride_h_h, stride_h_k, stride_h_v,
    stride_go_b, stride_go_h, stride_go_t, stride_go_v,
    stride_ghn_b, stride_ghn_h, stride_ghn_k, stride_ghn_v,
    stride_gq_b, stride_gq_h, stride_gq_t, stride_gq_k,
    stride_gk_b, stride_gk_h, stride_gk_t, stride_gk_k,
    stride_gv_b, stride_gv_h, stride_gv_t, stride_gv_v,
    stride_gg_b, stride_gg_h, stride_gg_t, stride_gg_k,
    stride_gb_b, stride_gb_h, stride_gb_t,
    stride_gh_b, stride_gh_h, stride_gh_k, stride_gh_v,
    is_doc_start,
    L: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BS: tl.constexpr,
    EPS: tl.constexpr,
):
    h_idx = tl.program_id(0)
    b_idx = tl.program_id(1)
    NC: tl.constexpr = K // BK
    NS: tl.constexpr = L // BS

    offs_t = tl.arange(0, L)
    offs_k = tl.arange(0, K)
    offs_v = tl.arange(0, V)
    bs_arange = tl.arange(0, BS)

    # ----- Load full q, k, v, g, beta, h, grad_o, grad_h_new -----
    q_c = tl.load(
        q_ptr + b_idx * stride_q_b + h_idx * stride_q_h
        + offs_t[:, None] * stride_q_t + offs_k[None, :] * stride_q_k,
    ).to(tl.float32)
    k_c = tl.load(
        k_ptr + b_idx * stride_k_b + h_idx * stride_k_h
        + offs_t[:, None] * stride_k_t + offs_k[None, :] * stride_k_k,
    ).to(tl.float32)
    v_c = tl.load(
        v_ptr + b_idx * stride_v_b + h_idx * stride_v_h
        + offs_t[:, None] * stride_v_t + offs_v[None, :] * stride_v_v,
    ).to(tl.float32)
    g_c = tl.load(
        g_ptr + b_idx * stride_g_b + h_idx * stride_g_h
        + offs_t[:, None] * stride_g_t + offs_k[None, :] * stride_g_k,
    ).to(tl.float32)
    beta_c = tl.load(
        beta_ptr + b_idx * stride_beta_b + h_idx * stride_beta_h
        + offs_t * stride_beta_t,
    ).to(tl.float32)
    h_start = tl.load(
        h_ptr + b_idx * stride_h_b + h_idx * stride_h_h
        + offs_k[:, None] * stride_h_k + offs_v[None, :] * stride_h_v,
    ).to(tl.float32)
    grad_o = tl.load(
        grad_o_ptr + b_idx * stride_go_b + h_idx * stride_go_h
        + offs_t[:, None] * stride_go_t + offs_v[None, :] * stride_go_v,
    ).to(tl.float32)
    grad_h_new = tl.load(
        grad_h_new_ptr + b_idx * stride_ghn_b + h_idx * stride_ghn_h
        + offs_k[:, None] * stride_ghn_k + offs_v[None, :] * stride_ghn_v,
    ).to(tl.float32)

    if is_doc_start == 1:
        h_start = tl.zeros_like(h_start)

    # ===== Recompute forward intermediates =====
    k_norm_sq = tl.sum(k_c * k_c, axis=1)
    k_norm_sq = tl.maximum(k_norm_sq, EPS)
    c_t = beta_c * k_norm_sq
    alpha_t = -(tl.exp(-c_t) - 1.0) / k_norm_sq

    g_cum = tl.cumsum(g_c, axis=0)
    g_chunk_no_kd = tl.sum(g_c, axis=0)
    g_cum_exp = tl.exp(g_cum)
    g_chunk_exp = tl.exp(g_chunk_no_kd)[:, None]

    # K_alpha stays in [L, K] (small enough at K=128)
    decay_h = tl.exp(g_chunk_no_kd[None, :] - g_cum)
    K_alpha = alpha_t[:, None] * decay_h * k_c

    # Masks
    strict_lower = offs_t[:, None] > offs_t[None, :]
    lower_incl = offs_t[:, None] >= offs_t[None, :]
    eye_mask = offs_t[:, None] == offs_t[None, :]

    # K-tile loop: T, Q_dot_K, p, o_first (recompute)
    T_acc = tl.zeros([L, L], dtype=tl.float32)
    Q_dot_K_acc = tl.zeros([L, L], dtype=tl.float32)
    p_acc = tl.zeros([L, V], dtype=tl.float32)
    o_first_acc = tl.zeros([L, V], dtype=tl.float32)

    for k_idx in tl.static_range(NC):
        k_start = k_idx * BK
        offs_k_tile = k_start + tl.arange(0, BK)
        k_tile = tl.load(
            k_ptr + b_idx * stride_k_b + h_idx * stride_k_h
            + offs_t[:, None] * stride_k_t + offs_k_tile[None, :] * stride_k_k,
        ).to(tl.float32)
        q_tile = tl.load(
            q_ptr + b_idx * stride_q_b + h_idx * stride_q_h
            + offs_t[:, None] * stride_q_t + offs_k_tile[None, :] * stride_q_k,
        ).to(tl.float32)
        g_tile = tl.load(
            g_ptr + b_idx * stride_g_b + h_idx * stride_g_h
            + offs_t[:, None] * stride_g_t + offs_k_tile[None, :] * stride_g_k,
        ).to(tl.float32)
        h_start_tile = tl.load(
            h_ptr + b_idx * stride_h_b + h_idx * stride_h_h
            + offs_k_tile[:, None] * stride_h_k + offs_v[None, :] * stride_h_v,
        ).to(tl.float32)

        g_cum_tile = tl.cumsum(g_tile, axis=0)
        g_cum_tile_exp = tl.exp(g_cum_tile)

        diff_g = g_cum_tile[:, None, :] - g_cum_tile[None, :, :]
        diff_g_safe = tl.where(strict_lower[:, :, None], diff_g, 0.0)
        decay_inner = tl.exp(diff_g_safe)
        T_acc += alpha_t[None, :] * tl.sum(
            k_tile[:, None, :] * k_tile[None, :, :] * decay_inner, axis=-1,
        )

        diff_q = g_cum_tile[:, None, :] - g_cum_tile[None, :, :]
        diff_q_safe = tl.where(lower_incl[:, :, None], diff_q, 0.0)
        decay_q = tl.exp(diff_q_safe)
        Q_dot_K_acc += alpha_t[None, :] * tl.sum(
            q_tile[:, None, :] * k_tile[None, :, :] * decay_q, axis=-1,
        )

        if BK >= 16:
            p_acc += tl.dot(k_tile * g_cum_tile_exp, h_start_tile, allow_tf32=False)
            o_first_acc += tl.dot(q_tile * g_cum_tile_exp, h_start_tile, allow_tf32=False)
        else:
            p_acc += tl.sum(
                (k_tile * g_cum_tile_exp)[:, :, None] * h_start_tile[None, :, :],
                axis=1,
            )
            o_first_acc += tl.sum(
                (q_tile * g_cum_tile_exp)[:, :, None] * h_start_tile[None, :, :],
                axis=1,
            )

    T = tl.where(strict_lower, T_acc, 0.0)
    A = tl.where(eye_mask, T + 1.0, T)
    Q_dot_K = tl.where(lower_incl, Q_dot_K_acc, 0.0)

    # ===== Solve for v_new (block triangular, same as forward kernel) =====
    rhs = v_c - p_acc
    v_new = tl.zeros_like(rhs)
    if L == 64:
        BS_BL: tl.constexpr = 16
        for bi in tl.static_range(4):
            row_mask_bi = (offs_t >= bi * BS_BL) & (offs_t < (bi + 1) * BS_BL)
            rhs_bi = rhs
            for bj in tl.static_range(bi):
                col_mask_bj = (offs_t >= bj * BS_BL) & (offs_t < (bj + 1) * BS_BL)
                a_sub = tl.where(
                    row_mask_bi[:, None] & col_mask_bj[None, :], A, 0.0,
                )
                contrib = tl.dot(a_sub, v_new, allow_tf32=False)
                rhs_bi = rhs_bi - contrib
            rhs_bi = tl.where(row_mask_bi[:, None], rhs_bi, 0.0)
            A_ii = tl.where(
                row_mask_bi[:, None] & row_mask_bi[None, :], A, 0.0,
            )
            v_new_bi = rhs_bi
            for t_local in tl.static_range(16):
                t_abs = bi * BS_BL + t_local
                a_t = tl.sum(tl.where(offs_t[:, None] == t_abs, A_ii, 0.0), axis=0)
                a_tt = tl.sum(tl.where(offs_t == t_abs, a_t, 0.0))
                a_t_masked = tl.where(offs_t < t_abs, a_t, 0.0)
                contrib = tl.sum(a_t_masked[:, None] * v_new_bi, axis=0)
                rhs_t = tl.sum(tl.where(offs_t[:, None] == t_abs, rhs_bi, 0.0), axis=0)
                v_new_t = (rhs_t - contrib) / a_tt
                v_new_bi = tl.where(offs_t[:, None] == t_abs, v_new_t[None, :], v_new_bi)
            v_new = tl.where(row_mask_bi[:, None], v_new_bi, v_new)
    else:
        # Generic fallback (L=16/32): simple forward sub
        v_new = rhs
        for t in tl.static_range(L):
            a_tt = tl.sum(tl.where(offs_t == t, tl.where(eye_mask, A, 0.0), 0.0))
            a_t = tl.sum(tl.where(offs_t[:, None] == t, A, 0.0), axis=0)
            a_t_masked = tl.where(offs_t < t, a_t, 0.0)
            contrib = tl.sum(a_t_masked[:, None] * v_new, axis=0)
            v_new_t = (rhs[t, :] - contrib) / a_tt
            v_new = tl.where(offs_t[:, None] == t, v_new_t[None, :], v_new)

    # ===== Backward =====
    grad_v_new = tl.dot(tl.trans(Q_dot_K), grad_o, allow_tf32=False)
    grad_v_new += tl.dot(K_alpha, grad_h_new, allow_tf32=False)

    grad_u = tl.zeros_like(grad_v_new)
    if L == 64:
        A_masked = A  # already lower-triangular
        grad_u = grad_v_new
        for t in tl.static_range(63, -1, -1):
            a_col = tl.sum(tl.where(offs_t[None, :] == t, A_masked, 0.0), axis=1)
            a_tt = tl.sum(tl.where(offs_t == t, a_col, 0.0))
            a_col_masked = tl.where(offs_t > t, a_col, 0.0)
            contrib = tl.sum(a_col_masked[:, None] * grad_u, axis=0)
            b_t = tl.sum(tl.where(offs_t[:, None] == t, grad_v_new, 0.0), axis=0)
            grad_u_t = (b_t - contrib) / a_tt
            grad_u = tl.where(offs_t[:, None] == t, grad_u_t[None, :], grad_u)
    else:
        A_masked = A
        for t in tl.static_range(L - 1, -1, -1):
            a_col = tl.sum(tl.where(offs_t[None, :] == t, A_masked, 0.0), axis=1)
            a_tt = tl.sum(tl.where(offs_t == t, a_col, 0.0))
            a_col_masked = tl.where(offs_t > t, a_col, 0.0)
            contrib = tl.sum(a_col_masked[:, None] * grad_u, axis=0)
            b_t = tl.sum(tl.where(offs_t[:, None] == t, grad_v_new, 0.0), axis=0)
            grad_u_t = (b_t - contrib) / a_tt
            grad_u = tl.where(offs_t[:, None] == t, grad_u_t[None, :], grad_u)

    # Direct gradients.
    grad_T = tl.where(
        strict_lower,
        -tl.dot(grad_u, tl.trans(v_new), allow_tf32=False),
        0.0,
    )
    grad_Q_dot_K = tl.where(
        lower_incl,
        tl.dot(grad_o, tl.trans(v_new), allow_tf32=False),
        0.0,
    )
    grad_K_alpha = tl.dot(v_new, tl.trans(grad_h_new), allow_tf32=False)  # [L, K]

    grad_v = grad_u
    grad_p = -grad_u
    grad_h = grad_h_new * g_chunk_exp
    grad_h += tl.dot(tl.trans(q_c * g_cum_exp), grad_o, allow_tf32=False)
    grad_h += tl.dot(tl.trans(k_c * g_cum_exp), grad_p, allow_tf32=False)

    # ===== Accumulators =====
    # Task #60 [BS, BS, BK] sub-tile rewrite — step 1: replace the
    # [L, NC, BK] cross-tile accumulators (3 × 32KB = 96KB at K=128,
    # OOM on the 5060 Ti's 99KB shmem) with per-K-tile stores to
    # global memory. Each per-K-tile store writes only the K-local
    # contributions; the chunk-final terms (grad_g_chunk[d],
    # grad_k_from_alpha[t,d]) are added in a post-loop fixup pass
    # below since they require the FULL accumulators across all
    # K-tiles.
    #
    #   grad_alpha_acc  : [L]              — small, in registers
    #   grad_g_chunk_acc: [NC, BK]         — Kalpha contribution,
    #                                        accumulated across K-tiles
    #
    #   grad_q, grad_k, grad_g: written per-K-tile at offset k_idx*BK.
    #     grad_g is the cumsum part only (grad_g_chunk added post-loop).
    #     grad_k is K-local only (grad_k_from_alpha added post-loop).
    grad_alpha_acc = tl.zeros([L], dtype=tl.float32)
    grad_g_chunk_acc = tl.zeros([NC, BK], dtype=tl.float32)

    nc_arange = tl.arange(0, NC)

    # ===== Per-K-tile loop with L sub-tiling =====
    # For each K-tile, process NS×NS sub-tiles of [BS, BS, BK] for the
    # T and Q_dot_K backward. K_alpha, p, o_first backward is also
    # sub-tiled over L to keep intermediate shapes ≤ [BS, BK].
    for k_idx in tl.static_range(NC):
        k_start = k_idx * BK
        offs_k_tile = k_start + tl.arange(0, BK)

        # Load K-tile
        k_tile = tl.load(
            k_ptr + b_idx * stride_k_b + h_idx * stride_k_h
            + offs_t[:, None] * stride_k_t + offs_k_tile[None, :] * stride_k_k,
        ).to(tl.float32)
        q_tile = tl.load(
            q_ptr + b_idx * stride_q_b + h_idx * stride_q_h
            + offs_t[:, None] * stride_q_t + offs_k_tile[None, :] * stride_q_k,
        ).to(tl.float32)
        g_tile = tl.load(
            g_ptr + b_idx * stride_g_b + h_idx * stride_g_h
            + offs_t[:, None] * stride_g_t + offs_k_tile[None, :] * stride_g_k,
        ).to(tl.float32)
        h_start_tile = tl.load(
            h_ptr + b_idx * stride_h_b + h_idx * stride_h_h
            + offs_k_tile[:, None] * stride_h_k + offs_v[None, :] * stride_h_v,
        ).to(tl.float32)
        grad_h_new_tile = tl.load(
            grad_h_new_ptr + b_idx * stride_ghn_b + h_idx * stride_ghn_h
            + offs_k_tile[:, None] * stride_ghn_k + offs_v[None, :] * stride_ghn_v,
        ).to(tl.float32)

        g_cum_tile = tl.cumsum(g_tile, axis=0)
        g_cum_tile_exp = tl.exp(g_cum_tile)
        g_chunk_tile = tl.sum(g_tile, axis=0)  # [BK]
        exp_factor_tile = tl.exp(g_chunk_tile[None, :] - g_cum_tile)  # [L, BK]

        # Per-K-tile [L, BK] accumulators (4KB each at L=64, BK=16)
        grad_q_kt = tl.zeros([L, BK], dtype=tl.float32)
        grad_k_kt = tl.zeros([L, BK], dtype=tl.float32)
        grad_g_cum_kt = tl.zeros([L, BK], dtype=tl.float32)
        grad_alpha_kt = tl.zeros([L], dtype=tl.float32)

        # grad_o_first_inner and grad_p_inner are [L, BK] (h_start_tile is [BK, V],
        # tl.trans is [V, BK], grad_o is [L, V], dot is [L, BK]).
        grad_o_first_inner = tl.dot(grad_o, tl.trans(h_start_tile), allow_tf32=False)  # [L, BK]
        grad_p_inner = tl.dot(grad_p, tl.trans(h_start_tile), allow_tf32=False)  # [L, BK]

        # ===== Sub-tile loop: T and Q_dot_K backward =====
        for ti in tl.static_range(NS):
            ti_offs = ti * BS + bs_arange  # [BS]
            # Mask to scatter [BS, BK] sub-tile results to [L, BK] at positions ti_offs.
            # ti_mask[i, j] = 1 iff i == ti*BS + j. Used as both gather
            # and scatter mask via matmul.
            ti_mask = (offs_t[:, None] == ti_offs[None, :])  # [L, BS], bool
            ti_mask_T = tl.trans(ti_mask.to(tl.float32))  # [BS, L]
            # Sub-tile slices (load directly with computed indices)
            k_sub_t = tl.load(
                k_ptr + b_idx * stride_k_b + h_idx * stride_k_h
                + ti_offs[:, None] * stride_k_t + offs_k_tile[None, :] * stride_k_k,
            ).to(tl.float32)  # [BS, BK]
            q_sub_t = tl.load(
                q_ptr + b_idx * stride_q_b + h_idx * stride_q_h
                + ti_offs[:, None] * stride_q_t + offs_k_tile[None, :] * stride_q_k,
            ).to(tl.float32)  # [BS, BK]
            # g_cum_sub_t must be the GLOBAL g_cum at positions ti_offs,
            # not the local cumsum of g_sub_t (which would be offset by
            # g_cum[ti*BS-1] for ti > 0, breaking the cross-sub-tile decay).
            # Gather via matmul: g_cum_sub_t[bi, d] = g_cum_tile[ti*BS+bi, d].
            g_cum_sub_t = tl.dot(ti_mask_T, g_cum_tile, allow_tf32=False)  # [BS, BK]
            g_cum_sub_t_exp = tl.exp(g_cum_sub_t)  # [BS, BK]
            # Gather alpha_t[ti_offs] via [L, BS] bool mask + sum
            alpha_t_sub_t = tl.sum(ti_mask.to(tl.float32) * alpha_t[:, None], axis=0)  # [BS]
            exp_factor_sub_t = tl.exp(g_chunk_tile[None, :] - g_cum_sub_t)  # [BS, BK]
            # Gather grad_o_first_inner[ti_offs, :] via [BS, L] @ [L, BK] = [BS, BK]
            grad_o_first_inner_sub = tl.dot(
                ti_mask_T, grad_o_first_inner, allow_tf32=False,
            )  # [BS, BK]
            grad_p_inner_sub = tl.dot(
                ti_mask_T, grad_p_inner, allow_tf32=False,
            )  # [BS, BK]

            for tj in tl.static_range(NS):
                tj_offs = tj * BS + bs_arange  # [BS]
                tj_mask = (offs_t[:, None] == tj_offs[None, :])  # [L, BS], bool
                tj_mask_T = tl.trans(tj_mask.to(tl.float32))  # [BS, L]
                k_sub_tj = tl.load(
                    k_ptr + b_idx * stride_k_b + h_idx * stride_k_h
                    + tj_offs[:, None] * stride_k_t + offs_k_tile[None, :] * stride_k_k,
                ).to(tl.float32)  # [BS, BK]
                q_sub_tj = tl.load(
                    q_ptr + b_idx * stride_q_b + h_idx * stride_q_h
                    + tj_offs[:, None] * stride_q_t + offs_k_tile[None, :] * stride_q_k,
                ).to(tl.float32)  # [BS, BK]
                # g_cum_sub_tj must also be the global g_cum (matmul gather),
                # not the local cumsum of g_sub_tj.
                g_cum_sub_tj = tl.dot(tj_mask_T, g_cum_tile, allow_tf32=False)  # [BS, BK]
                alpha_t_sub_tj = tl.sum(tj_mask.to(tl.float32) * alpha_t[:, None], axis=0)  # [BS]

                # Sub-tile strict_lower mask
                strict_lower_sub = ti_offs[:, None] > tj_offs[None, :]  # [BS, BS]
                lower_incl_sub = ti_offs[:, None] >= tj_offs[None, :]  # [BS, BS]
                # Sub-tile decay_inner [BS, BS, BK]
                diff_g_sub = g_cum_sub_t[:, None, :] - g_cum_sub_tj[None, :, :]
                decay_inner_sub = tl.exp(
                    tl.where(strict_lower_sub[:, :, None], diff_g_sub, 0.0),
                )  # [BS, BS, BK]
                # Sub-tile decay_q
                decay_q_sub = tl.exp(
                    tl.where(lower_incl_sub[:, :, None], diff_g_sub, 0.0),
                )  # [BS, BS, BK]
                # Sub-tile grad_T, grad_Q_dot_K (gather via two matmuls).
                # [BS, L] @ [L, L] @ [L, BS] = [BS, BS].
                grad_T_tmp = tl.dot(ti_mask_T, grad_T, allow_tf32=False)  # [BS, L]
                grad_T_sub = tl.dot(grad_T_tmp, tl.trans(tj_mask_T), allow_tf32=False)  # [BS, BS]
                grad_Q_dot_K_tmp = tl.dot(ti_mask_T, grad_Q_dot_K, allow_tf32=False)  # [BS, L]
                grad_Q_dot_K_sub = tl.dot(grad_Q_dot_K_tmp, tl.trans(tj_mask_T), allow_tf32=False)  # [BS, BS]

                # ===== T backward =====
                # N_sub has BOTH k factors (correct for grad_g_cum and grad_alpha,
                # where both k[t,d] and k[t',d] appear in dT/dg_cum and dT/dalpha).
                # For grad_k we need separate sums with only ONE k factor:
                #   dT[t,t']/dk[t,d] = α_{t'} * decay * k[t',d]  (k_tj only)
                #   dT[t,t']/dk[t',d] = α_{t'} * k[t,d] * decay (k_t only)
                N_sub = (grad_T_sub[:, :, None]
                         * alpha_t_sub_tj[None, :, None]
                         * decay_inner_sub
                         * k_sub_t[:, None, :]
                         * k_sub_tj[None, :, :])  # [BS, BS, BK]

                # grad_alpha: sum over (t, d), result is per t' (tj)
                grad_alpha_sub = tl.sum(tl.sum(
                    grad_T_sub[:, :, None] * decay_inner_sub * k_sub_t[:, None, :] * k_sub_tj[None, :, :],
                    axis=0,  # sum over t (in ti_offs)
                ), axis=-1)  # sum over d → [BS]
                # Scatter to grad_alpha_kt at positions tj_offs
                grad_alpha_kt += tl.sum(
                    tj_mask.to(tl.float32) * grad_alpha_sub[None, :], axis=1,
                )  # [L]

                # grad_k_first: sum over t' (tj) → [BS, BK]. dT/dk[t,d] = α·decay·k[t',d]
                # so we use k_tj only (NOT k_t — that would multiply by an extra k[t,d]).
                grad_k_first_sub = tl.sum(
                    grad_T_sub[:, :, None] * alpha_t_sub_tj[None, :, None] * decay_inner_sub * k_sub_tj[None, :, :],
                    axis=1,
                )  # [BS, BK]
                grad_k_kt += tl.dot(ti_mask.to(tl.float32), grad_k_first_sub, allow_tf32=False)

                # grad_k_second: sum over t (ti) → [BS, BK]. dT/dk[t',d] = α·k[t,d]·decay
                # so we use k_t only.
                grad_k_second_sub = tl.sum(
                    grad_T_sub[:, :, None] * alpha_t_sub_tj[None, :, None] * decay_inner_sub * k_sub_t[:, None, :],
                    axis=0,
                )  # [BS, BK]
                grad_k_kt += tl.dot(tj_mask.to(tl.float32), grad_k_second_sub, allow_tf32=False)

                # grad_g_cum_first (sum over t', tj): [BS, BK]. dT/dg_cum[t,d] = α·k[t,d]·decay·k[t',d]
                # so the BOTH-k-factors N_sub is correct.
                grad_g_cum_first_sub = tl.sum(N_sub, axis=1)
                grad_g_cum_kt += tl.dot(ti_mask.to(tl.float32), grad_g_cum_first_sub, allow_tf32=False)
                # grad_g_cum_second (sum over t, ti): [BS, BK]. dT/dg_cum[t',d] = -α·k[t,d]·decay·k[t',d]
                # so the BOTH-k-factors N_sub is correct (with a sign flip).
                grad_g_cum_second_sub = tl.sum(N_sub, axis=0)
                grad_g_cum_kt -= tl.dot(tj_mask.to(tl.float32), grad_g_cum_second_sub, allow_tf32=False)

                # ===== Q_dot_K backward =====
                # Same pattern as T: Nq_sub has BOTH q and k factors (correct for
                # grad_g_cum and grad_alpha, where both appear in dQ/dg_cum and dQ/dalpha).
                # For grad_q and grad_k_from_QdotK we need separate sums with only ONE factor:
                #   dQ/dq[t,d] = α·decay_q·k[t',d]  (k_tj only)
                #   dQ/dk[t',d] = α·q[t,d]·decay_q  (q_t only)
                Nq_sub = (grad_Q_dot_K_sub[:, :, None]
                          * alpha_t_sub_tj[None, :, None]
                          * decay_q_sub
                          * q_sub_t[:, None, :]
                          * k_sub_tj[None, :, :])  # [BS, BS, BK]

                # grad_alpha: sum over (t, d), per t'
                grad_alpha_q_sub = tl.sum(tl.sum(
                    grad_Q_dot_K_sub[:, :, None] * decay_q_sub * q_sub_t[:, None, :] * k_sub_tj[None, :, :],
                    axis=0,
                ), axis=-1)  # [BS]
                grad_alpha_kt += tl.sum(
                    tj_mask.to(tl.float32) * grad_alpha_q_sub[None, :], axis=1,
                )  # [L]

                # grad_q_from_QdotK: dQ/dq[t,d] = α·decay_q·k[t',d] — k_tj only
                grad_q_from_QdotK_sub = tl.sum(
                    grad_Q_dot_K_sub[:, :, None] * alpha_t_sub_tj[None, :, None] * decay_q_sub * k_sub_tj[None, :, :],
                    axis=1,
                )  # [BS, BK]
                grad_q_kt += tl.dot(ti_mask.to(tl.float32), grad_q_from_QdotK_sub, allow_tf32=False)

                # grad_k_from_QdotK: dQ/dk[t',d] = α·q[t,d]·decay_q — q_t only
                grad_k_from_QdotK_sub = tl.sum(
                    grad_Q_dot_K_sub[:, :, None] * alpha_t_sub_tj[None, :, None] * decay_q_sub * q_sub_t[:, None, :],
                    axis=0,
                )  # [BS, BK]
                grad_k_kt += tl.dot(tj_mask.to(tl.float32), grad_k_from_QdotK_sub, allow_tf32=False)

                # grad_g_cum_from_QdotK: dQ/dg_cum[t,d] and dQ/dg_cum[t',d] both have
                # both factors (with a sign flip for the t' derivative), so Nq_sub is correct.
                grad_g_cum_q_first_sub = tl.sum(Nq_sub, axis=1)
                grad_g_cum_kt += tl.dot(ti_mask.to(tl.float32), grad_g_cum_q_first_sub, allow_tf32=False)
                grad_g_cum_q_second_sub = tl.sum(Nq_sub, axis=0)
                grad_g_cum_kt -= tl.dot(tj_mask.to(tl.float32), grad_g_cum_q_second_sub, allow_tf32=False)

            # ===== K_alpha backward (only depends on ti sub-tile) =====
            # grad_K_alpha[t, d] = sum_v v_new[t, v] * grad_h_new[d, v]
            # Sub-tile: gather v_new[ti_offs, :] via [BS, L] @ [L, V] = [BS, V]
            v_new_sub = tl.dot(ti_mask_T, v_new, allow_tf32=False)  # [BS, V]
            grad_K_alpha_sub = tl.dot(
                v_new_sub, tl.trans(grad_h_new_tile), allow_tf32=False,
            )  # [BS, BK]
            # K_alpha_sub = alpha_t[ti] * exp(g_chunk - g_cum_sub_t) * k_sub_t
            # dK_alpha[t,d]/d(alpha_t[t]) = exp(g_chunk - g_cum) * k   (no alpha_t factor;
            # alpha_t is the linear coefficient, not a self-multiplier).
            grad_alpha_kalpha_sub = tl.sum(
                grad_K_alpha_sub * exp_factor_sub_t * k_sub_t,
                axis=1,
            )  # [BS]
            grad_alpha_kt += tl.sum(
                ti_mask.to(tl.float32) * grad_alpha_kalpha_sub[None, :], axis=1,
            )  # [L]
            # grad_g_cum_from_Kalpha = -grad_K_alpha * alpha_t * exp_factor * k
            grad_g_cum_kalpha_sub = -grad_K_alpha_sub * alpha_t_sub_t[:, None] * exp_factor_sub_t * k_sub_t  # [BS, BK]
            grad_g_cum_kt += tl.dot(ti_mask.to(tl.float32), grad_g_cum_kalpha_sub, allow_tf32=False)
            # grad_k_from_Kalpha
            grad_k_kalpha_sub = grad_K_alpha_sub * alpha_t_sub_t[:, None] * exp_factor_sub_t  # [BS, BK]
            grad_k_kt += tl.dot(ti_mask.to(tl.float32), grad_k_kalpha_sub, allow_tf32=False)
            # grad_g_chunk_from_Kalpha (sum over t in ti_offs, then to chunk)
            grad_g_chunk_sub = tl.sum(
                grad_K_alpha_sub * alpha_t_sub_t[:, None] * exp_factor_sub_t * k_sub_t,
                axis=0,
            )  # [BK]
            grad_g_chunk_acc += grad_g_chunk_sub[None, :] * (nc_arange == k_idx)[:, None].to(tl.float32)

            # ===== o_first backward (per ti sub-tile) =====
            grad_q_ofirst_sub = g_cum_sub_t_exp * grad_o_first_inner_sub  # [BS, BK]
            grad_q_kt += tl.dot(ti_mask.to(tl.float32), grad_q_ofirst_sub, allow_tf32=False)
            grad_g_cum_ofirst_sub = q_sub_t * g_cum_sub_t_exp * grad_o_first_inner_sub  # [BS, BK]
            grad_g_cum_kt += tl.dot(ti_mask.to(tl.float32), grad_g_cum_ofirst_sub, allow_tf32=False)

            # ===== p backward (per ti sub-tile) =====
            grad_k_p_sub = g_cum_sub_t_exp * grad_p_inner_sub  # [BS, BK]
            grad_k_kt += tl.dot(ti_mask.to(tl.float32), grad_k_p_sub, allow_tf32=False)
            grad_g_cum_p_sub = k_sub_t * g_cum_sub_t_exp * grad_p_inner_sub  # [BS, BK]
            grad_g_cum_kt += tl.dot(ti_mask.to(tl.float32), grad_g_cum_p_sub, allow_tf32=False)

        # ===== Per-K-tile stores to global memory =====
        # Replace the [L, NC, BK] accumulators with direct stores at
        # offset k_idx*BK. Each store writes only K-local contributions;
        # grad_g_chunk[d] and grad_k_from_alpha[t,d] are added in the
        # post-loop fixup pass below.
        tl.store(
            grad_q_ptr + b_idx * stride_gq_b + h_idx * stride_gq_h
            + offs_t[:, None] * stride_gq_t + offs_k_tile[None, :] * stride_gq_k,
            grad_q_kt.to(grad_q_ptr.dtype.element_ty),
        )
        tl.store(
            grad_k_ptr + b_idx * stride_gk_b + h_idx * stride_gk_h
            + offs_t[:, None] * stride_gk_t + offs_k_tile[None, :] * stride_gk_k,
            grad_k_kt.to(grad_k_ptr.dtype.element_ty),
        )
        # grad_g (cumsum part only — grad_g_chunk added post-loop)
        grad_g_cum_reversed = tl.flip(grad_g_cum_kt, 0)
        grad_g_reversed_cumsum = tl.cumsum(grad_g_cum_reversed, axis=0)
        grad_g_kt = tl.flip(grad_g_reversed_cumsum, 0)
        tl.store(
            grad_g_ptr + b_idx * stride_gg_b + h_idx * stride_gg_h
            + offs_t[:, None] * stride_gg_t + offs_k_tile[None, :] * stride_gg_k,
            grad_g_kt.to(grad_g_ptr.dtype.element_ty),
        )

        grad_alpha_acc += grad_alpha_kt

    # ===== grad_beta from grad_alpha =====
    exp_neg_c = tl.exp(-c_t)  # [L]
    grad_beta = grad_alpha_acc * exp_neg_c

    grad_k_norm_sq = grad_alpha_acc * (
        (c_t + 1.0) * exp_neg_c - 1.0
    ) / (k_norm_sq * k_norm_sq)  # [L]

    # ===== Write per-chunk outputs (grad_v, grad_h, grad_beta) =====
    tl.store(
        grad_v_ptr + b_idx * stride_gv_b + h_idx * stride_gv_h
        + offs_t[:, None] * stride_gv_t + offs_v[None, :] * stride_gv_v,
        grad_v.to(grad_v_ptr.dtype.element_ty),
    )
    tl.store(
        grad_h_ptr + b_idx * stride_gh_b + h_idx * stride_gh_h
        + offs_k[:, None] * stride_gh_k + offs_v[None, :] * stride_gh_v,
        grad_h.to(grad_h_ptr.dtype.element_ty),
    )
    tl.store(
        grad_beta_ptr + b_idx * stride_gb_b + h_idx * stride_gb_h
        + offs_t * stride_gb_t,
        grad_beta.to(grad_beta_ptr.dtype.element_ty),
    )

    # ===== Post-loop fixup pass =====
    # Add grad_g_chunk[d] to grad_g[t, d] and grad_k_from_alpha[t, d]
    # to grad_k[t, d]. Both depend on the FULL chunk accumulators
    # (grad_alpha_acc, grad_g_chunk_acc), so they run AFTER the main
    # K-tile loop completes.
    #
    # grad_g_chunk[d] = (sum over K-tiles of grad_K_alpha @ h_new_first contribution)
    #                 + grad_h_new[d] @ h_new_first_term[d]
    #   where h_new_first_term[d, v] = h_start[d, v] * exp(g_chunk[d]).
    #
    # grad_k_from_alpha[t, d] = 2 * k[t, d] * grad_k_norm_sq[t]
    #   where grad_k_norm_sq depends on FULL grad_alpha_acc.
    #
    # ``range`` (not tl.static_range) keeps ptxas's unroll analysis
    # bounded by one body (not NC copies).
    for k_idx_post in range(NC):
        k_start_post = k_idx_post * BK
        offs_k_tile_post = k_start_post + tl.arange(0, BK)

        # Reload K-tile slices (small, fits in shmem).
        g_tile = tl.load(
            g_ptr + b_idx * stride_g_b + h_idx * stride_g_h
            + offs_t[:, None] * stride_g_t + offs_k_tile_post[None, :] * stride_g_k,
        ).to(tl.float32)
        h_start_tile = tl.load(
            h_ptr + b_idx * stride_h_b + h_idx * stride_h_h
            + offs_k_tile_post[:, None] * stride_h_k + offs_v[None, :] * stride_h_v,
        ).to(tl.float32)
        grad_h_new_tile = tl.load(
            grad_h_new_ptr + b_idx * stride_ghn_b + h_idx * stride_ghn_h
            + offs_k_tile_post[:, None] * stride_ghn_k + offs_v[None, :] * stride_ghn_v,
        ).to(tl.float32)
        k_tile_post = tl.load(
            k_ptr + b_idx * stride_k_b + h_idx * stride_k_h
            + offs_t[:, None] * stride_k_t + offs_k_tile_post[None, :] * stride_k_k,
        ).to(tl.float32)

        # grad_g_chunk_from_h_first: per-d contribution from h_start * exp(g_chunk).
        g_chunk_tile = tl.sum(g_tile, axis=0)  # [BK]
        g_chunk_exp_tile = tl.exp(g_chunk_tile)[:, None]  # [BK, 1]
        h_new_first_term_kt = h_start_tile * g_chunk_exp_tile  # [BK, V]
        grad_g_chunk_from_h_first_kt = tl.sum(
            grad_h_new_tile * h_new_first_term_kt, axis=1,
        )  # [BK]
        # grad_g_chunk_acc[k_idx_post, :] is the Kalpha contribution
        # for this K-tile (full sum, since grad_g_chunk_acc is the
        # FULL cross-tile accumulator).
        grad_g_chunk_kt = (
            tl.sum(grad_g_chunk_acc * (nc_arange == k_idx_post)[:, None].to(tl.float32), axis=0)
            + grad_g_chunk_from_h_first_kt
        )  # [BK]

        # Load existing grad_g from memory (cumsum part), add grad_g_chunk_kt, store back.
        grad_g_existing = tl.load(
            grad_g_ptr + b_idx * stride_gg_b + h_idx * stride_gg_h
            + offs_t[:, None] * stride_gg_t + offs_k_tile_post[None, :] * stride_gg_k,
        ).to(tl.float32)
        grad_g_new = grad_g_existing + grad_g_chunk_kt[None, :]
        tl.store(
            grad_g_ptr + b_idx * stride_gg_b + h_idx * stride_gg_h
            + offs_t[:, None] * stride_gg_t + offs_k_tile_post[None, :] * stride_gg_k,
            grad_g_new.to(grad_g_ptr.dtype.element_ty),
        )

        # grad_k_from_alpha (per-K-tile): 2 * k_tile * grad_k_norm_sq
        grad_k_from_alpha_kt = 2.0 * k_tile_post * grad_k_norm_sq[:, None]  # [L, BK]
        grad_k_existing = tl.load(
            grad_k_ptr + b_idx * stride_gk_b + h_idx * stride_gk_h
            + offs_t[:, None] * stride_gk_t + offs_k_tile_post[None, :] * stride_gk_k,
        ).to(tl.float32)
        grad_k_new = grad_k_existing + grad_k_from_alpha_kt
        tl.store(
            grad_k_ptr + b_idx * stride_gk_b + h_idx * stride_gk_h
            + offs_t[:, None] * stride_gk_t + offs_k_tile_post[None, :] * stride_gk_k,
            grad_k_new.to(grad_k_ptr.dtype.element_ty),
        )


# --------------------------------------------------------------------------- #
# Per-chunk bwd wrapper (gated by env var EFKDA_BWD_KERNEL=triton)             #
# --------------------------------------------------------------------------- #
def _chunk_bwd_launch(
    h, q_c, k_c, v_c, g_c, beta_c,
    grad_o, grad_h_new,
    L, eps, is_doc_start,
):
    """Launch the Triton bwd kernel for one chunk."""
    B, H = q_c.shape[0], q_c.shape[1]
    K = q_c.shape[-1]
    V = v_c.shape[-1]

    # Sub-tile size. BS=16, NS=L//BS=4 at L=64.
    BS = 16
    if L % BS != 0:
        # Fallback to non-sub-tiled (only K=16/K=32 have been tested).
        BS = L

    # BK tuning: smaller BK = smaller per-tile intermediates but more
    # NC iterations. For K=128, BK=16 gives [BS, BS, BK] = 4KB per
    # sub-tile. For smaller K, BK=16 still works (NC=1, 2, 4).
    if K % 16 == 0:
        BK = 16
    elif K % 8 == 0:
        BK = 8
    else:
        BK = K

    if K >= 128:
        num_warps = 8
    else:
        num_warps = 4

    grad_q = torch.empty(B, H, L, K, device=q_c.device, dtype=q_c.dtype)
    grad_k = torch.empty(B, H, L, K, device=q_c.device, dtype=q_c.dtype)
    grad_v = torch.empty(B, H, L, V, device=q_c.device, dtype=q_c.dtype)
    grad_g = torch.empty(B, H, L, K, device=q_c.device, dtype=q_c.dtype)
    grad_beta = torch.empty(B, H, L, device=q_c.device, dtype=q_c.dtype)
    grad_h = torch.empty(B, H, K, V, device=h.device, dtype=h.dtype)

    grid = (H, B)
    _efkda_chunk_bwd_kernel[grid](
        q_c, k_c, v_c, g_c, beta_c, h,
        grad_o, grad_h_new,
        grad_q, grad_k, grad_v, grad_g, grad_beta, grad_h,
        q_c.stride(0), q_c.stride(1), q_c.stride(2), q_c.stride(3),
        k_c.stride(0), k_c.stride(1), k_c.stride(2), k_c.stride(3),
        v_c.stride(0), v_c.stride(1), v_c.stride(2), v_c.stride(3),
        g_c.stride(0), g_c.stride(1), g_c.stride(2), g_c.stride(3),
        beta_c.stride(0), beta_c.stride(1), beta_c.stride(2),
        h.stride(0), h.stride(1), h.stride(2), h.stride(3),
        grad_o.stride(0), grad_o.stride(1), grad_o.stride(2), grad_o.stride(3),
        grad_h_new.stride(0), grad_h_new.stride(1), grad_h_new.stride(2), grad_h_new.stride(3),
        grad_q.stride(0), grad_q.stride(1), grad_q.stride(2), grad_q.stride(3),
        grad_k.stride(0), grad_k.stride(1), grad_k.stride(2), grad_k.stride(3),
        grad_v.stride(0), grad_v.stride(1), grad_v.stride(2), grad_v.stride(3),
        grad_g.stride(0), grad_g.stride(1), grad_g.stride(2), grad_g.stride(3),
        grad_beta.stride(0), grad_beta.stride(1), grad_beta.stride(2),
        grad_h.stride(0), grad_h.stride(1), grad_h.stride(2), grad_h.stride(3),
        is_doc_start,
        L, K, V, BK, BS, eps,
        num_warps=num_warps,
        num_stages=1,
    )
    return grad_h, grad_q, grad_k, grad_v, grad_g, grad_beta
