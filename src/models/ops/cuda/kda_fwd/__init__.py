"""KDA forward CUDA kernels (Round 1 — correctness-first baseline).

Public API
----------
``chunk_kda_fwd(q, k, v, g, beta, cu_seqlens=None, scale=None,
                initial_state=None, output_final_state=False, BT=64)``
    Replacement for the vendored FLA Triton path. bf16 only.
    ``g`` is in **natural log** space (matches FLA's user contract); the
    wrapper multiplies by ``RCP_LN2`` to convert to log2 before cumsum,
    matching ``chunk_local_cumsum(g, scale=RCP_LN2)``.

Pipeline (4 stages)
-------------------
1. **intra_solve** (Python orchestration + cuBLAS bmm):
   - Chunk-local cumsum of ``g`` over each BT-token block (in log2).
   - Compute decay scales r = exp2(g - g_last), c = exp2(g_last - g).
   - A_qk = bmm(q * r, (k * c).T) * scale   [B*HV*NT, BT, BT]
   - A_kk = bmm(k * r, (k * c).T) * beta[:,:,None], strict-lower-tri mask
   - A = I + A_kk (fp32)
2. **forward_sub** (custom CUDA): in-place A → A^{-1}.
3. **wy_transform** (Python orchestration + cuBLAS bmm):
   - w = bmm(A_inv, k * beta[:,:,None])    [B*HV*NT, BT, K]
   - u = bmm(A_inv, v)                      [B*HV*NT, BT, V]
4. **delta_h** (custom CUDA, sequential over NT):
   - per (doc, hv, v_slice), iterates NT chunks
   - h_next = exp2(g_last) * h_prev + k^T @ (v - w @ h_prev) * exp2(g_last - g[:])
5. **chunk_o** (custom CUDA):
   - per (hv, chunk, v_slice)
   - o = q^T h + A_qk_for_o * v

For non-bf16 inputs the wrapper falls back to the vendored FLA path.
"""
from __future__ import annotations

import math
from typing import Optional

import torch

from .load_inline import get_module, version_string


_module = None


def _ensure_compiled(verbose: bool = False):
    """Compile (or load cached) the CUDA module on first call."""
    global _module
    if _module is None:
        _module = get_module(verbose=verbose)
    return _module


def compile(verbose: bool = False):
    """Force-compile the CUDA extension. Subsequent calls are no-ops."""
    return _ensure_compiled(verbose=verbose)


def _is_bf16(t: torch.Tensor) -> bool:
    return t.dtype == torch.bfloat16


# ===================================================================== //
# Per-stage CUDA launches (called by the orchestrator)                  //
# ===================================================================== //

def _forward_sub(A_kk_fp32: torch.Tensor, BT: int = 64) -> None:
    """In-place: A → A^{-1}. A_kk_fp32 shape: [N, BT, BT] fp32, lower-tri.

    The C++ wrapper reads ``N`` from ``A_kk_fp32.size(0)`` internally,
    so we just pass the tensor.
    """
    mod = _ensure_compiled()
    assert A_kk_fp32.dim() == 3
    assert A_kk_fp32.size(1) == A_kk_fp32.size(2) == BT, (
        f"A_kk_fp32 last 2 dims must be [BT, BT]={BT}; got "
        f"{A_kk_fp32.shape}"
    )
    mod.forward_sub(A_kk_fp32)


def _delta_h(
    k: torch.Tensor,           # [T_total, H, K] bf16
    u: torch.Tensor,           # [T_total, HV, V] bf16  (= A_inv @ (v * beta))
    w: torch.Tensor,           # [num_chunks, HV, BT, K] bf16  (= A_inv @ (k * beta * exp2(g_cum)))
    g_cum: torch.Tensor,       # [T_total, HV, K] bf16  (chunk-local cumsum)
    chunk_token_base: torch.Tensor,  # [num_chunks] int32 absolute token start
    doc_chunk_start: torch.Tensor,  # [num_docs] int32
    doc_chunk_count: torch.Tensor,  # [num_docs] int32
    v_new_out: torch.Tensor,   # [T_total, HV, V] bf16, pre-allocated output for v_new_per_token
    num_chunks: int,
    num_docs: int,
    H: int, HV: int,
    K: int, V: int, V_TILE: int,
):
    """Run delta_h_kernel.

    Computes per-token v_new = (u - w @ h_prev) * exp2(g_last - g[i])
    (matches the FLA chunk_delta_h.py recurrence) and writes it to
    ``v_new_out`` for chunk_o to consume.

    Returns (h_per_chunk, h_final):
      * h_per_chunk [num_chunks, HV, K, V] bf16 — h at the start of each chunk
        (chunk_o consumes this; Lever B June 2026-30: was fp32, now bf16
        to halve HBM traffic on the qg @ h matmul).
      * h_final     [num_docs, HV, K, V]    fp32 — h at the end of each doc
        (returned to the caller if output_final_state=True; not used in
        Round-1 since output_final_state is hardcoded False).

    Backend selection:
      * HIPPOLM_KDA_DELTA_H_BACKEND = "wmma"   (default; tensor cores via nvcuda::wmma)
      * HIPPOLM_KDA_DELTA_H_BACKEND = "scalar" (Round-1 scalar GEMM fallback)
    """
    import os
    mod = _ensure_compiled()
    # Lever B (June 2026-30): h_per_chunk is bf16 (was fp32). The delta_h
    # kernels accumulate in fp32 (h_prev in shmem), but only the per-chunk
    # SNAPSHOT is bf16 — chunk_o consumes it via bf16 mma with cos=1.0 vs
    # fp32 at prod shape (see test/_tmp/test_chunk_o_bf16_h.py).
    h_per_chunk = torch.empty(
        num_chunks, HV, K, V, dtype=torch.bfloat16, device=u.device,
    )
    h_final = torch.empty(
        num_docs, HV, K, V, dtype=torch.float32, device=u.device,
    )
    backend = os.environ.get("HIPPOLM_KDA_DELTA_H_BACKEND", "wmma").lower()
    fn = mod.wmma_delta_h if backend == "wmma" else mod.delta_h
    fn(
        k, u, w, g_cum,
        doc_chunk_start, doc_chunk_count, chunk_token_base,
        v_new_out,
        h_per_chunk, h_final,
        num_chunks, H, HV,
    )
    return h_per_chunk, h_final


def _chunk_o(
    q: torch.Tensor,           # [T_total, H, K] bf16
    v_new: torch.Tensor,       # [T_total, HV, V] bf16  (delta_h output: per-token v_new)
    g_cum: torch.Tensor,       # [T_total, HV, K] bf16
    A_qk: torch.Tensor,        # [num_chunks, HV, BT, BT] bf16  (precomputed, scale baked in)
    h: torch.Tensor,           # [num_chunks, HV, K, V] fp32  (state at start of each chunk)
    chunk_token_base: torch.Tensor,  # [num_chunks] int32
    num_chunks: int,
    scale: float,
    H: int, HV: int, K: int, V: int, V_TILE: int,
) -> torch.Tensor:
    """Run chunk_o kernel. Returns o [T_total, HV, V] bf16.

    Output contract (matches FLA chunk_gla_fwd_o_gk):
        o[i, v] = (q[i] @ (exp2(g_cum[i]) * h)) * scale + Aqk @ v_new
    where ``A_qk`` is the local causal matrix with ``scale`` already baked in
    (computed by the intra phase). The q^Th contribution picks up scale
    ONCE; the Aqk @ v_new contribution picks it up via the pre-scaled Aqk.

    Round-2: defaults to the Triton kernel (10× faster than the hand-rolled
    CUDA). CUDA is kept as a fallback via the ``HIPPOLM_KDA_CHUNK_O_BACKEND``
    env var — useful for debugging or environments without Triton.
    """
    import os
    backend = os.environ.get("HIPPOLM_KDA_CHUNK_O_BACKEND", "triton").lower()
    if backend == "triton":
        from .triton_kernels import triton_chunk_o
        return triton_chunk_o(
            q, v_new, g_cum, A_qk, h, chunk_token_base,
            num_chunks, scale, H, HV, K, V,
        )
    # Fallback: hand-rolled CUDA
    mod = _ensure_compiled()
    T_total = q.shape[0]
    o = torch.empty(T_total, HV, V, dtype=torch.bfloat16, device=q.device)
    mod.chunk_o(
        q, v_new, g_cum, A_qk, h,
        chunk_token_base,
        o, scale, H, HV,
    )
    return o


# ===================================================================== //
# Public entry point                                                    //
# ===================================================================== //

def chunk_kda_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,           # raw per-token decay in NATURAL log space
    beta: torch.Tensor,
    cu_seqlens: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    BT: int = 64,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """KDA forward via custom CUDA kernels (bf16 only).

    Args:
        q: ``[B, T, H, K]`` bf16 — post-L2-norm queries.
        k: ``[B, T, H, K]`` bf16 — post-L2-norm keys.
        v: ``[B, T, HV, V]`` bf16 — values.
        g: ``[B, T, HV, K]`` bf16 — per-token decay in NATURAL log space
           (matches the FLA user-facing contract; internally we multiply by
           ``RCP_LN2`` to convert to log2 domain before cumsum).
        beta: ``[B, T, HV]`` bf16 — per-token beta scalars.
        cu_seqlens: optional ``[num_docs+1]`` int64 for varlen. When set,
            ``q, k, v, g`` are reshaped to ``[1, T, ...]`` first (folded
            batch dim into the sequence dim).
        scale: defaults to ``1/sqrt(K)``.
        initial_state: not implemented in Round 1 (must be None).
        output_final_state: not implemented in Round 1 (must be False).
        BT: chunk size (default 64).

    Returns:
        ``(o, final_state)`` matching the FLA chunk_kda contract:
        ``o`` has shape ``[B, T, HV, V]`` bf16; ``final_state`` is None
        in Round 1.
    """
    # ----- input validation -----
    assert _is_bf16(q) and _is_bf16(k) and _is_bf16(v), "q/k/v must be bf16"
    assert _is_bf16(g) and _is_bf16(beta), "g/beta must be bf16"
    assert q.is_cuda, "q must be on CUDA"
    assert BT == 64, "Round-1 path is hard-coded to BT=64"
    assert initial_state is None, "Round-1 does not support initial_state"
    assert not output_final_state, "Round-1 does not support output_final_state"

    B, T, H, K = q.shape
    HV = v.shape[2]
    V = v.shape[3]
    assert H == HV, f"Round-1 requires HV == H (no GVA); got H={H}, HV={HV}"
    assert K == V, f"Round-1 requires K == V; got K={K}, V={V}"
    if scale is None:
        scale = 1.0 / math.sqrt(K)

    # ----- varlen handling: fold B into T if cu_seqlens is set -----
    if cu_seqlens is not None:
        # Pack: B*T tokens total, B "logical" rows, but the kernel sees B=1.
        # The Python wrapper lays out [B*T, H, K] flat; cu_seqlens is the
        # token offsets for each "doc" (where doc = original batch row).
        q_flat = q.reshape(1, B * T, H, K)
        k_flat = k.reshape(1, B * T, H, K)
        v_flat = v.reshape(1, B * T, HV, V)
        g_flat = g.reshape(1, B * T, HV, K)
        beta_flat = beta.reshape(1, B * T, HV)
        # Per-doc NT: each doc has ceil(len / BT) chunks.
        NT_per_doc = ((cu_seqlens[1:] - cu_seqlens[:-1] + BT - 1) // BT).tolist()
        num_docs = len(NT_per_doc)
        num_chunks = sum(NT_per_doc)
    else:
        q_flat = q
        k_flat = k
        v_flat = v
        g_flat = g
        beta_flat = beta
        NT_per_doc = [T // BT]
        num_docs = 1
        num_chunks = T // BT

    T_total = q_flat.shape[1]
    assert T_total == num_chunks * BT, (
        f"T_total ({T_total}) must equal num_chunks * BT ({num_chunks * BT}); "
        f"varlen padding/alignment is the caller's responsibility in Round 1"
    )

    # ----- flatten to per-token layout: [T_total, H_or_HV, K_or_V] -----
    q_tok = q_flat.view(T_total, H, K).contiguous()
    k_tok = k_flat.view(T_total, H, K).contiguous()
    v_tok = v_flat.view(T_total, HV, V).contiguous()
    g_tok = g_flat.view(T_total, HV, K).contiguous()
    beta_tok = beta_flat.view(T_total, HV).contiguous()

    # ----- chunk-local cumsum of g (per chunk, per K-dim) -----
    # FLA user contract: g is in NATURAL log. Kernels use exp2 internally,
    # so we convert to log2 via RCP_LN2 = 1/ln(2) BEFORE the cumsum —
    # matches FLA's `chunk_local_cumsum(g, scale=RCP_LN2)` exactly.
    # g shape: [T_total, HV, K]. Reshape to [num_chunks, BT, HV, K].
    # cumsum over BT (the chunk dim). We do the cumsum in **fp32** for
    # numerical stability (matching FLA's chunk_local_cumsum output_dtype
    # default of fp32) and cast back to bf16 only at the kernel boundary.
    RCP_LN2 = 1.4426950408889634
    g_log2_chunks_fp32 = (g_tok.float() * RCP_LN2).view(num_chunks, BT, HV, K)
    g_cum_fp32 = g_log2_chunks_fp32.cumsum(dim=1)  # [num_chunks, BT, HV, K] fp32
    g_cum_tok = g_cum_fp32.to(torch.bfloat16).view(T_total, HV, K).contiguous()
    # g_last per chunk: [num_chunks, HV, K] = g_cum[:, BT-1, :, :] (in fp32)
    g_last = g_cum_fp32[:, BT - 1, :, :].contiguous()  # [num_chunks, HV, K] fp32

    # (The middle-aligned r/c scales for the intra loop are computed
    # inline below in the (s_i, s_j) pair loop — no need to precompute
    # them here.)
    BC = 16  # sub-chunk width — matches FLA chunk_intra's token_parallel path

    # ----- batched matmuls for A_qk and A_kk (block-by-block, FLA-exact) -----
    # FLA's intra splits the [BT, BT] causal matrix into BC x BC sub-chunks.
    # For BT=64, BC=16 there are 4 sub-chunks per chunk and 10 (s_i, s_j) pairs
    # where s_i >= s_j. The decay anchor differs by pair type:
    #
    #   * Diagonal block (s_i == s_j): anchor = g_cum[s * BC + BC//2] (sub-chunk middle).
    #     r = exp2(g_cum[i] - anchor), c = exp2(anchor - g_cum[j]).
    #   * Off-diagonal block (s_i > s_j): anchor = g_cum[s_i * BC] (ROW sub-chunk start).
    #     r = exp2(g_cum[i] - anchor), c = exp2(anchor - g_cum[j]).
    #
    # This per-block anchor keeps exp2 arguments bounded by ~BC * max|g| ≈ 70 in log2,
    # well within bf16's ±3.4e38 range. Using a single chunk-level anchor (g_last)
    # would overflow for the early positions of any chunk with a large cumulative g.
    NC = BT // BC  # 4 sub-chunks per chunk
    q_per = q_tok.view(num_chunks, BT, H, K).transpose(1, 2).contiguous()  # [NC, H, BT, K]
    k_per = k_tok.view(num_chunks, BT, H, K).transpose(1, 2).contiguous()
    v_per = v_tok.view(num_chunks, BT, HV, V).transpose(1, 2).contiguous()
    g_per = g_cum_fp32.view(num_chunks, BT, HV, K).transpose(1, 2).contiguous()  # fp32
    beta_per = beta_tok.view(num_chunks, BT, HV).transpose(1, 2).contiguous()  # [NC, HV, BT]

    # ----- Cast q, k to fp32 ONCE (was the per-pair bottleneck) -----
    # Lever H (June 2026-30): the fused Triton intra_solve kernel reads q/k
    # as bf16 and casts to fp32 IN REGISTERS (free — no HBM cost). This
    # eliminates the two big fp32 pre-casts (~2.6 ms at prod) and replaces
    # the 10-pair Python bmm loop (~17 ms at prod) with a single Triton
    # launch. A_qk comes back bf16 with scale baked in; A_kk_fp32 stays
    # fp32 for forward_sub (which requires fp32). See triton_intra_solve.py
    # for the kernel and the (cos=0.999995, med_rel<0.1%) verification.
    from .triton_intra_solve import triton_intra_solve
    A_qk_bf16, A_kk_unscaled = triton_intra_solve(q_per, k_per, g_per, scale, BT, BC)

    # Apply beta to A_kk (row-side scaling).
    # beta is per-token: shape [num_chunks, HV, BT]
    # In stacked [num_chunks*HV, BT, BT] layout, beta is [num_chunks*HV, BT]
    beta_stacked = beta_per.reshape(num_chunks * HV, BT)
    # A_kk[i, m, n] *= beta[i, m] (row-side beta for the lower-tri mask)
    A_kk_unscaled = A_kk_unscaled * beta_stacked.unsqueeze(-1)
    # Apply causal mask (strict lower-tri for A_kk)
    mask = torch.tril(torch.ones(BT, BT, device=q.device, dtype=torch.float32), diagonal=-1)
    A_kk_unscaled = A_kk_unscaled * mask.unsqueeze(0)
    # Add I on diagonal.
    eye = torch.eye(BT, device=q.device, dtype=torch.float32).unsqueeze(0)
    A_kk_fp32 = A_kk_unscaled + eye  # A = I + A_kk
    # A_qk_bf16 already bf16 with scale baked in (from the fused kernel).

    # ----- forward_sub in place -----
    _forward_sub(A_kk_fp32, BT)

    # ----- Wy transform via bmm -----
    # FLA's recompute_w_u_kda_kernel computes (matching wy_fast.py + KDA intra):
    #   u = A_inv @ (v * beta)
    #   w = A_inv @ (k * beta * exp2(g_cum))
    # Then delta_h does v_new = (u - w @ h_prev) * exp2(g_last - g[i]),
    # and chunk_o consumes v_new (NOT raw v) plus the q^Th contribution scaled
    # by exp2(g_cum[i]) and the local-qk contribution via Aqk (which already
    # has scale baked in).
    # Reuse v_per from above (line ~321).
    v_per_stacked = v_per.view(num_chunks * HV, BT, V)
    k_per_stacked = k_per.view(num_chunks * HV, BT, K)
    beta_stacked = beta_per.reshape(num_chunks * HV, BT)
    g_per_stacked = g_per.view(num_chunks * HV, BT, K)

    beta_v = (v_per_stacked * beta_stacked.unsqueeze(-1)).contiguous()
    u = torch.bmm(A_kk_fp32, beta_v.to(torch.float32)).to(torch.bfloat16)  # [N, BT, V]
    # Per-token exp2(g_cum[i, k]) per K-dim, broadcast across V.
    g_cum_exp2 = g_per_stacked.exp2()  # [N, BT, K] fp32
    k_with_beta_g = (k_per_stacked * beta_stacked.unsqueeze(-1) * g_cum_exp2.to(torch.bfloat16)).contiguous()
    w = torch.bmm(A_kk_fp32, k_with_beta_g.to(torch.float32)).to(torch.bfloat16)  # [N, BT, K]

    # ----- layout tensors for the per-chunk kernels -----
    # chunk_token_base: [num_chunks] int32 = the absolute token start of each chunk.
    # For fixed-len, chunk_token_base[i] = i * BT. For varlen, it's cumulative.
    if cu_seqlens is not None:
        # Per-doc chunk token bases.
        bases = []
        for d in range(num_docs):
            doc_start = int(cu_seqlens[d].item())
            for i_t in range(NT_per_doc[d]):
                bases.append(doc_start + i_t * BT)
        chunk_token_base = torch.tensor(bases, dtype=torch.int32, device=q.device)
        doc_chunk_start_list = [0]
        for n in NT_per_doc[:-1]:
            doc_chunk_start_list.append(doc_chunk_start_list[-1] + n)
        doc_chunk_start = torch.tensor(doc_chunk_start_list, dtype=torch.int32, device=q.device)
    else:
        chunk_token_base = torch.arange(num_chunks, dtype=torch.int32, device=q.device) * BT
        doc_chunk_start = torch.tensor([0], dtype=torch.int32, device=q.device)
    doc_chunk_count = torch.tensor(NT_per_doc, dtype=torch.int32, device=q.device)

    # ----- delta_h: per (doc, hv, v_slice) -----
    # V_TILE=32 keeps delta_h static shmem under 48KB cap (see kernel.cu).
    # chunk_o keeps V_TILE=64 (it doesn't carry the [K, V_TILE] state).
    V_TILE_DELTA = 32
    V_TILE_O = 64
    NV = V // V_TILE_DELTA
    # Reshape w and u from [N=NC*HV, BT, ...] to [NC, HV, BT, ...] for delta_h.
    # w: [num_chunks, HV, BT, K]. u: [num_chunks, HV, BT, V].
    w_per_hv = w.view(num_chunks, HV, BT, K).contiguous()
    # u is per-chunk, per-hv, per-token. Reshape to per-token [T_total, HV, V].
    # u_per_chunk[c, hv, i, v] = u_tok[c*BT + i, hv, v]
    u_per_chunk = u.view(num_chunks, HV, BT, V)
    u_tok = u_per_chunk.transpose(1, 2).contiguous().view(T_total, HV, V)
    v_new_tok = torch.empty(T_total, HV, V, dtype=torch.bfloat16, device=q.device)
    h_per_chunk, h_final = _delta_h(
        k_tok, u_tok, w_per_hv, g_cum_tok,
        chunk_token_base, doc_chunk_start, doc_chunk_count,
        v_new_tok,
        num_chunks, num_docs, H, HV, K, V, V_TILE_DELTA,
    )

    # ----- chunk_o: per (hv, chunk, v_slice) -----
    # A_qk is [N, BT, BT] bf16 (where N = num_chunks*HV) with per-K decay
    # AND scale baked in (matches FLA chunk_intra storage). Reshape to
    # [num_chunks, HV, BT, BT] for the kernel.
    A_qk_per_hv = A_qk_bf16.view(num_chunks, HV, BT, BT).contiguous()
    o_flat = _chunk_o(
        q_tok, v_new_tok, g_cum_tok, A_qk_per_hv,
        h_per_chunk, chunk_token_base,
        num_chunks, scale, H, HV, K, V, V_TILE_O,
    )
    # Reshape back to [B, T, HV, V]. For varlen we folded B into T (B'=1).
    o_out = o_flat.view(q_flat.shape[0], q_flat.shape[1], HV, V)

    if cu_seqlens is not None:
        # Restore original [B, T, HV, V] view from the [B*T, ...] flat.
        o_out = o_out.view(B, T, HV, V)

    final_state = h_final if output_final_state else None
    return o_out, final_state


__all__ = ["chunk_kda_fwd", "compile", "version_string"]
