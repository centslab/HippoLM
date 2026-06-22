"""Chunkwise EFLA KDA (per-token closed form composed within chunks).

For each chunk of L tokens, the per-token EFLA KDA gives

    S_t = (I - alpha_t k_t k_t^T) D_t S_{t-1} + alpha_t k_t v_t^T

where ``alpha_t = -expm1(-beta_t * ||k_t||^2) / ||k_t||^2`` is a SCALAR
per step (KDA's beta is per-head scalar, so the closed-form
coefficient is also scalar — unlike GDN2 where b is per-channel and
the closed form needs a per-channel correction).

Viewing ``S_decayed_t = D_t S_{t-1}`` as known input at step t, the
delta-rule form is

    v_new_t = v_t - k_t^T S_decayed_t
    S_t     = S_decayed_t + alpha_t k_t v_new_t^T

Unrolling the v_new recursion over the chunk (t' < t in the chunk):

    v_new_t = v_t - k_t^T D_t S_{t-2}
              - sum_{t'<t} alpha_{t'} (k_t^T D_t k_{t'}) v_new_{t'}^T

Putting S_{t-2} = diag(exp(g_prefix[t-1])) S_start (relative to chunk
start), the S_start contribution is

    p_t = (k_t * exp(g_cum[t]))^T S_start

where ``g_cum[t] = sum_{i=0}^{t} g_i`` is the inclusive cumsum of g
within the chunk. The remaining lower-triangular interaction can be
written as a single matrix equation:

    (I + T) v_new = v - p
    v_new         = (I + T)^{-1} (v - p)

with the strict-lower-triangular matrix

    T[t, t'] = alpha_{t'} * <k_t (gated), k_{t'}>    for t' < t
             = alpha_{t'} * sum_d k_t[d] exp(g_cum[t, d] - g_cum[t', d]) k_{t'}[d]

The chunk-end state is then

    S_chunk_end = D_chunk S_start
                + sum_t alpha_t exp(g_chunk - g_cum[t]) k_t v_new_t^T

with ``g_chunk = g_cum[L-1]``. Outputs at step t are

    o[t] = (q_t * exp(g_cum[t]))^T S_start
         + sum_{t'=0}^{t} alpha_{t'} (q_t * exp(g_cum[t] - g_cum[t'])
                                       * k_{t'})^T v_new[t']

This is mathematically equivalent to running ``efla_kda_per_token`` for
all tokens in the chunk — the rank-1 closed form composes through the
chunk via the (I + T)^{-1} linear solve (the same algebraic structure
used in the per-step WY representation, but with COLUMN alpha instead
of ROW beta; the fla chunk KDA's row-beta structure gives beta^2 on
writes, which doesn't match the per-token EFLA's alpha on writes).
"""
from __future__ import annotations

import torch


def _efla_chunk_apply(
    h: torch.Tensor,
    q_c: torch.Tensor,
    k_c: torch.Tensor,
    v_c: torch.Tensor,
    g_c: torch.Tensor,
    beta_c: torch.Tensor,
    L: int,
    eps: float,
):
    """One chunk's EFKDA forward. Returns (o_c, h_new).

    Designed to be called inside :func:`torch.utils.checkpoint.checkpoint`
    so that per-chunk intermediates (notably the [B, H, L, L, K] rank-5
    tensors ``diff_g`` / ``decay_inner`` / ``diff_q`` / ``decay_q``) are
    freed after the chunk's forward and recomputed on backward. Without
    this, all 64 chunks' saved-for-backward tensors are alive in the
    autograd graph at once, blowing the 16 GB budget at base.yml prod
    dims (B=4, T=4096, L=32) — 4.5 GB per layer × 4 layers per block
    recompute ≈ 18 GB.

    The recurrent state ``h`` is a tensor input/output, so the autograd
    graph connects ``h_new`` of chunk i to ``h_start`` of chunk i+1 and
    the gradient flows correctly through the chunk boundary.
    """
    B, H, _, K = k_c.shape
    V = v_c.shape[-1]

    # 1. Alpha per step (scalar per (B, H, L)). beta is scalar per
    #    step and k is not normalized, so the rank-1 update is
    #    β k kᵀ. The eigenvalue of the inner matrix N = k kᵀ is
    #    λ = ||k||² (NOT β||k||²), and the matrix exp of -β N is
    #    I - α k kᵀ with α = (1 - exp(-β λ)) / λ = -expm1(-c) / ||k||².
    #    (Compare GDN-2: the recurrence is (I - N) with N = k(b⊘k)ᵀ
    #    and eigenvalue c = k·(b⊘k), so alpha = -expm1(-c)/c.)
    k_norm_sq = (k_c * k_c).sum(-1).clamp(min=eps)        # [B, H, L] = ||k||²
    c_t = beta_c * k_norm_sq                              # [B, H, L] = β ||k||²
    alpha_t = -torch.expm1(-c_t) / k_norm_sq              # [B, H, L]

    # 2. Cumulative decay within chunk (inclusive cumsum over TIME, not K).
    #    g_cum[t, k] = sum_{i=0}^{t} g_c[i, k]   (log-decay accumulated
    #    up to and including step t, for each channel k).
    g_cum = g_c.cumsum(2)                                 # [B, H, L, K]
    g_chunk = g_cum[:, :, -1:]                            # [B, H, 1, K] = g_cum[L-1]

    # 3. T matrix: strict lower-triangular (L x L) per (B, H).
    #    T[t, t'] = alpha_{t'} * <k_t (gated), k_{t'}>   for t' < t
    #    COLUMN alpha: matches per-token EFLA which uses alpha_t in the
    #    write term. Sign is +alpha (not -alpha): the v_new recursion
    #    gives (I + T) v_new = v - p, not (I - T).
    #
    #    STABILITY: mask upper-triangle positions to 0 BEFORE exp().
    #    For t' > t (upper tri), g_cum[t] > g_cum[t'] (decay is monotone),
    #    so diff_g > 0 and exp(diff_g) overflows fp32 for diff_g > 89.
    #    The values are masked out by ``tril`` later, but the BACKWARD
    #    through exp() still multiplies the (masked-out) 0 gradient by
    #    inf, giving nan. Masking here to 0 forces the unmasked gradient
    #    to be 0 * 1 = 0, not 0 * inf = nan.
    diff_g = g_cum.unsqueeze(-2) - g_cum.unsqueeze(-3)    # [B, H, L, L, K]
    strict_lower_mask = torch.tril(
        torch.ones(L, L, device=diff_g.device, dtype=diff_g.dtype),
        diagonal=-1,
    ).bool().view(1, 1, L, L, 1)
    diff_g_safe = torch.where(strict_lower_mask, diff_g, torch.zeros_like(diff_g))
    decay_inner = diff_g_safe.exp()
    T = alpha_t.unsqueeze(-2) * (
        k_c.unsqueeze(-2) * k_c.unsqueeze(-3) * decay_inner
    ).sum(-1)                                             # [B, H, L, L]
    T = torch.tril(T, diagonal=-1)                        # strict lower tri

    # 4. (I + T) v_new = (v - p), solved by forward substitution.
    #
    #    STABILITY: do NOT compute the explicit inverse T_inv = (I+T)^{-1}
    #    and multiply. The entries of a lower-triangular inverse of a
    #    dense lower-triangular matrix grow exponentially with the
    #    matrix dimension: at L=64 with α=0.5, the (L-1, 0) entry of
    #    (I+T)^{-1} can hit 1e+18. The autograd VJP of ``einsum`` then
    #    propagates this amplification back to k, g, v, alpha.
    #
    #    ``solve_triangular(., upper=False)`` is forward-substitution;
    #    its VJP is backward-substitution on the transpose (also
    #    bounded). Use it for the RHS solve and skip materializing T_inv.
    eye = torch.eye(L, device=T.device, dtype=T.dtype).expand(B, H, L, L)

    # 5. p = (k * exp(g_cum))^T * h  -- S_start contribution to v_new.
    #    No per-channel b factor (KDA's beta is scalar, not per-channel).
    p = torch.einsum("bhlk,bhkv->bhlv", k_c * g_cum.exp(), h)

    # 6. v_new = (I + T)^{-1} (v - p) via forward sub (no T_inv).
    #    No w scaling (KDA has no separate write gate; v_new uses raw v
    #    minus the state projection).
    #    Flatten (B, H) into a single batch dim so solve_triangular's
    #    [M, M] x [M, K] signature lines up cleanly.
    v_aug = v_c - p                                          # [B, H, L, V]
    A_flat = (eye + T).reshape(B * H, L, L)                 # [B*H, L, L]
    v_new = torch.linalg.solve_triangular(
        A_flat, v_aug.reshape(B * H, L, V), upper=False,
    ).reshape(B, H, L, V)                                   # [B, H, L, V]

    # 7. Outputs.
    #    o[t] = q_t · h_{t+1}
    #         = q_t · diag(exp(g_cum[t])) h_start          (first term)
    #         + sum_{t'=0}^{t} alpha_{t'} (q_t ⊙ exp(g_cum[t]-g_cum[t']) ⊙ k_{t'}).sum(-1) v_new[t']
    # Note: g_cum[t] is INCLUSIVE; Q_dot_K includes the diagonal
    # (step t's write does contribute to o[t]).

    # o_first: contribution from the chunk-start state.
    o_first = torch.einsum("bhlk,bhkv->bhlv", q_c * g_cum.exp(), h)

    # o_second: contribution from the rank-1 writes within the chunk.
    # Same stability fix as for T: mask the upper triangle (incl diagonal
    # exclusion for diff_q >= 0) BEFORE exp() to avoid fp32 overflow.
    diff_q = g_cum.unsqueeze(-2) - g_cum.unsqueeze(-3)    # [B, H, L, L, K]
    lower_incl_mask = torch.tril(
        torch.ones(L, L, device=diff_q.device, dtype=diff_q.dtype),
        diagonal=0,
    ).bool().view(1, 1, L, L, 1)
    diff_q_safe = torch.where(lower_incl_mask, diff_q, torch.zeros_like(diff_q))
    decay_q = diff_q_safe.exp()
    Q_dot_K = alpha_t.unsqueeze(-2) * (
        q_c.unsqueeze(-2) * k_c.unsqueeze(-3) * decay_q
    ).sum(-1)                                             # [B, H, L, L]
    Q_dot_K = torch.tril(Q_dot_K, diagonal=0)             # lower-tri INCL diagonal

    o_second = torch.einsum("bhij,bhjv->bhiv", Q_dot_K, v_new)

    o_c = o_first + o_second

    # 8. Update h: chunk-end state = D_chunk · h_start + K_alpha · v_new^T.
    # g_chunk has shape [B, H, 1, K] (cumulative decay over chunk); reshape to
    # [B, H, K, 1] so it broadcasts against h[B, H, K, V] without injecting
    # an unwanted dim.
    h_new = h * g_chunk.squeeze(2).exp().unsqueeze(-1)
    K_alpha = alpha_t.unsqueeze(-1) * (g_chunk - g_cum).exp() * k_c   # [B, H, L, K]
    h_new = h_new + torch.einsum("bhlk,bhlv->bhkv", K_alpha, v_new)

    return o_c, h_new


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
    """Chunkwise EFLA KDA (closed form per token, matrix ops per chunk).

    Tensors are (B, T, H, K or V) on input; internally transposed to
    (B, H, T, K or V) for the recurrence.

    Per-chunk checkpoint: each chunk's forward is wrapped in
    :func:`torch.utils.checkpoint.checkpoint` so its intermediates
    (the [B, H, L, L, K] rank-5 tensors) are freed after the chunk
    and recomputed on backward. See :func:`_efla_chunk_apply` for the
    per-chunk body and the VRAM rationale.

    cu_seqlens (varlen) contract
    ----------------------------
    When supplied, ``cu_seqlens`` is ``[total_docs + 1]`` with
    ``cu_seqlens[0] == 0`` and strictly-increasing offsets into the
    flattened sequence (the caller flattens ``[B, T, hidden]`` →
    ``[1, B*T, hidden]`` so ``B == 1`` here). The kernel processes
    the chunks sequentially and the recurrent state ``h`` MUST be
    reset to zero at every chunk that starts a new doc — otherwise
    ``p = (k * exp(g_cum))^T h_start`` for that chunk reads stale
    state from the previous doc, the (I+T)^{-1} solve produces an
    incorrect ``v_new``, and the chain rule back through the stale
    ``p`` amplifies across chunks until gradients are inf. The
    failure mode at base.yml prod dims (B=4 packs, FFD packer →
    ~6 packs at T=4096 = 384 chunks) is exactly this: short_conv
    resets q/k/v state at each doc boundary, but the kernel sees
    one flat sequence and was carrying ``h`` across doc starts. T
    ≤ 2048 (≤ 192 chunks) masked it; 384 chunks let the error
    amplify past fp32.

    ``cu_seqlens`` MUST be aligned to chunk boundaries — every
    ``cu_seqlens[i]`` must be a multiple of ``chunk_size`` — which
    is exactly what ``pack_chunk_aligned`` (chunk-aligned FFD
    packing) produces. With chunk-aligned boundaries, "doc starts
    at chunk k" is a clean integer check and we never need to
    split a chunk across a boundary.

    The pre-computed ``_doc_start_chunks`` set is built once on the
    Python side (one CPU sync at kernel entry, then O(1) Python set
    lookup per chunk). Avoiding per-chunk GPU sync matters at 384
    chunks × 32 layers.
    """
    if scale is None:
        scale = q.shape[-1] ** -0.5
    in_dtype = v.dtype
    q, k, v, g, beta = (
        x.transpose(1, 2).contiguous().float() for x in (q, k, v, g, beta)
    )
    B, H, T, K = k.shape
    V = v.shape[-1]
    L = chunk_size
    assert T % L == 0, f"T={T} must be divisible by chunk_size={L}"
    n_chunks = T // L

    # Pre-compute which chunks start a new doc. ``pack_chunk_aligned``
    # guarantees cu_seqlens[i] is a multiple of chunk_size, so a doc
    # boundary always coincides with a chunk start and the chunkwise
    # decomposition is clean. cu_seqlens[0] == 0 is the sequence
    # start, not a doc-internal boundary, so we skip it.
    _doc_start_chunks: set[int] = set()
    if cu_seqlens is not None:
        assert B == 1, (
            f"cu_seqlens path expects B=1 (caller flattens to [1, B*T, ...]); got B={B}"
        )
        cu_offsets = cu_seqlens.tolist() if not cu_seqlens.is_cuda else cu_seqlens.cpu().tolist()
        for off in cu_offsets[1:]:
            chunk_idx = off // L
            assert off % L == 0, (
                f"cu_seqlens offset {off} is not aligned to chunk_size={L}; "
                "pack_chunk_aligned must be used."
            )
            _doc_start_chunks.add(int(chunk_idx))

    h = torch.zeros(B, H, K, V, device=v.device, dtype=v.dtype)
    if initial_state is not None:
        h = initial_state.to(v.dtype).clone()
    q = q * scale
    o = torch.zeros_like(v)

    for chunk_idx in range(n_chunks):
        # State reset at doc boundaries: without this the kernel
        # carries the previous doc's accumulated h_start into a
        # chunk whose q/k/v are FRESH (short_conv state-reset at the
        # cu_seqlens boundary). p reads the stale h_start, the
        # (I+T)^{-1} solve mismatches v against an unrelated state,
        # and the error propagates into inf gradients over many
        # chunks. With packer cs at T=4096 (384 chunks, ~30 doc
        # starts) this fails reliably; with T ≤ 2048 (≤ 192 chunks)
        # the error is bounded and the training is stable.
        if chunk_idx in _doc_start_chunks:
            h = torch.zeros(B, H, K, V, device=v.device, dtype=v.dtype)
        s = chunk_idx * L
        e = s + L
        o_c, h = torch.utils.checkpoint.checkpoint(
            _efla_chunk_apply,
            h,
            q[:, :, s:e], k[:, :, s:e], v[:, :, s:e],
            g[:, :, s:e], beta[:, :, s:e],
            L, eps,
            use_reentrant=False,
        )
        o[:, :, s:e] = o_c

    o = o.transpose(1, 2).contiguous().to(in_dtype)
    if not output_final_state:
        h = None
    return o, h
