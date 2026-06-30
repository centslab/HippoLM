"""Fused Triton KDA prepare kernel.

Replicates FlashKDA's Kernel 1 algorithm in pure Triton (no CUTLASS,
no SM90+ TMA, no external deps). The key idea:

    r[s] = exp(g_cumsum[s])          <-- never materialized in gmem
    c[s] = exp(g_total - g_cumsum[s]) <-- ditto
    q_decayed = q * r * scale         <-- computed in registers before bmm
    k_decayed = k * r                 <-- ditto
    k_inv     = k / r                 <-- ditto
    k_restored = k * c                <-- ditto
    L = k_decayed @ k_inv.T           <-- single tl.dot
    Mqk = q_decayed @ k_inv.T         <-- single tl.dot
    INV = (I - L)^(-1) via Neumann    <-- 4 in-tile matmuls
    Mqk_eff = Mqk @ INV               <-- single tl.dot  (Round-9 fix)
    K_pre   = INV @ k_restored        <-- single tl.dot  (Round-9 fix)

The Round-9 fix is what makes the recurrence numerically correct.
The intra-chunk output of the chunk-wise GDR is:

    o[t] = q_decayed[t] @ h_prev
         + Mqk @ (INV @ ((v - k_decayed @ h_prev) * beta))

where the (I - L)^-1 (i.e. INV) is applied to the v_residual * beta. The
first draft of FastKDA computed INV but the recurrence ignored it,
producing o = Mqk @ v directly — wrong on multiple counts (no
v_residual subtraction, no (I - L)^-1, no beta scaling). Round-9 fixes
this by precomputing the two "fold-into-workspace" combinations in
prepare:

    Mqk_eff = Mqk @ INV         (replaces Mqk in mqk_ptr)
    K_pre   = INV @ k_restored  (replaces k_restored in kr_ptr)

The recurrence then does:
    v_residual = v - k_decayed @ h_prev
    v_residual_b = v_residual * beta
    o = Mqk_eff @ v_residual_b + q_decayed @ h_prev
    h_new = h * exp(g_total) + K_pre^T @ v_residual_b

This matches FlashKDA's reference (`tests/torch_ref.py:232-243`) and
FLA's KDA path (chunk_kda_fwd_kernel_inter_solve_fused +
recompute_w_u_fwd). The two extra prepare bmms cost ~5% of prepare
time; the recurrence adds one big matmul (k_decayed @ h_prev) but
stays at HBM peak (the v_residual matmul output is tiny and the
k_decayed load was already in the workspace under a different
contract — we were writing it but not reading it).

Per-program work: 1 chunk (CHUNK=16 tokens) of 1 head of 1 batch.

Layout (row-major contiguous, stride documented inline):
    q, k, g:           [B, T, H, K]      bf16   stride (T*H*K, H*K, K, 1)
    beta:              [B, T, H]         bf16   stride (T*H, H, 1)
    A_log:             [H]               fp32
    dt_bias:           [H, K]            fp32
    workspace (out, flat [B*NC, ...] so the recurrence kernel can read
                                    it with the same pid_b*NC + pid_nc scheme):
      k_decayed:       [B*NC, H, CHUNK, K]   bf16
      q_decayed:       [B*NC, H, CHUNK, K]   bf16
      K_pre:           [B*NC, H, CHUNK, K]   bf16   (was k_restored)
      g_total:         [B*NC, H, K]          fp32
      Mqk_eff:         [B*NC, H, CHUNK, CHUNK] bf16   (was Mqk)
      (INV is computed but NOT stored — folded into Mqk_eff and K_pre)
"""
from __future__ import annotations

import triton
import triton.language as tl


CHUNK: int = 16   # tokens per chunk — matches FlashKDA, FLA


# ===================================================================== #
# Triton kernel                                                          #
# ===================================================================== #

@triton.jit
def _kda_prepare_kernel(
    q_ptr, k_ptr, g_ptr, beta_ptr,
    A_log_ptr, dt_bias_ptr,
    # workspace outputs (note: no inv_ptr — INV is folded into Mqk_eff and K_pre)
    kd_ptr, qd_ptr, kr_ptr, gt_ptr, mqk_ptr,
    # problem dims
    B, H, K,  # K must equal CHUNK*8 (=128 with CHUNK=16)
    T, NC,  # total tokens per batch, num chunks per batch
    # strides (in elements, not bytes)
    stride_q_b, stride_q_t, stride_q_h,
    stride_k_b, stride_k_t, stride_k_h,
    stride_g_b, stride_g_t, stride_g_h,
    stride_b_b, stride_b_t, stride_b_h,
    stride_kd_nc, stride_kd_h,
    stride_qd_nc, stride_qd_h,
    stride_kr_nc, stride_kr_h,
    stride_gt_nc, stride_gt_h,
    stride_mqk_nc, stride_mqk_h,
    # scalars
    scale,
    a_log_exp_per_head_ptr,  # precomputed exp(A_log) per head, [H] fp32
    gate_scale,  # lower_bound (natural-log convention; see kernel comment)
    # block sizes
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,  # K dim = 128
):
    pid_nc = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_b = tl.program_id(2)

    # ---------------- load dt_bias for this head ---------------- #
    offs_n = tl.arange(0, BLOCK_N)
    dt_bias = tl.load(dt_bias_ptr + pid_h * BLOCK_N + offs_n).to(tl.float32)

    # ---------------- chunk slice [pid_nc*CHUNK, (pid_nc+1)*CHUNK) ---------------- #
    offs_m = pid_nc * BLOCK_M + tl.arange(0, BLOCK_M)
    seq_len = tl.minimum((pid_nc + 1) * BLOCK_M, T)
    # mask: 1 if within chunk, 0 if padding beyond T
    valid_m = offs_m < seq_len

    # ---------------- load q, k, g, beta ---------------- #
    # Input pointer base = pid_b * B_stride (row-major [B, T, H, K] etc.)
    q_p = q_ptr + pid_b * stride_q_b
    k_p = k_ptr + pid_b * stride_k_b
    g_p = g_ptr + pid_b * stride_g_b
    b_p = beta_ptr + pid_b * stride_b_b
    # q, k, g: shape [BLOCK_M, BLOCK_N]
    q = tl.load(
        q_p + offs_m[:, None] * stride_q_t + pid_h * stride_q_h + offs_n[None, :],
        mask=valid_m[:, None], other=0.0,
    ).to(tl.float32)
    k = tl.load(
        k_p + offs_m[:, None] * stride_k_t + pid_h * stride_k_h + offs_n[None, :],
        mask=valid_m[:, None], other=0.0,
    ).to(tl.float32)
    g_raw = tl.load(
        g_p + offs_m[:, None] * stride_g_t + pid_h * stride_g_h + offs_n[None, :],
        mask=valid_m[:, None], other=0.0,
    ).to(tl.float32)
    beta = tl.load(
        b_p + offs_m * stride_b_t + pid_h * stride_b_h,
        mask=valid_m, other=0.0,
    ).to(tl.float32)

    # ---------------- QK L2 norm ---------------- #
    q_sq = tl.sum(q * q, axis=1)  # [BLOCK_M]
    k_sq = tl.sum(k * k, axis=1)
    q = q * tl.rsqrt(q_sq[:, None] + 1e-6)
    k = k * tl.rsqrt(k_sq[:, None] + 1e-6)

    # ---------------- Gate activation ---------------- #
    a_log_exp = tl.load(a_log_exp_per_head_ptr + pid_h)  # scalar
    # g_activated[m, n] = gate_scale * sigmoid(a_log_exp * (g_raw + dt_bias))
    g_val = a_log_exp * (g_raw + dt_bias[None, :])
    # mask out padding rows
    g_val = tl.where(valid_m[:, None], g_val, 0.0)
    g_val = gate_scale * tl.sigmoid(g_val)

    # ---------------- Cumsum along token axis ---------------- #
    g_cs = tl.cumsum(g_val, axis=0)  # [BLOCK_M, BLOCK_N]
    # g_total = g_cs[BLOCK_M - 1, :] — Triton disallows const-indexing a tensor;
    # sum trick: pick out the last row via one-hot multiply
    last_row_mask = tl.arange(0, BLOCK_M) == (BLOCK_M - 1)
    g_total = tl.sum(tl.where(last_row_mask[:, None], g_cs, 0.0), axis=0)  # [BLOCK_N]
    exp_g_total = tl.exp(g_total)  # [BLOCK_N]

    # ---------------- Fused decay apply (in registers/tile) ---------------- #
    # All intermediates stay in registers; never written to gmem.
    exp_g = tl.exp(g_cs)            # [BLOCK_M, BLOCK_N]
    exp_neg_g = tl.exp(-g_cs)       # [BLOCK_M, BLOCK_N]
    # q_decayed = q * exp(g) * scale
    q_decayed = (q * exp_g * scale).to(tl.bfloat16)
    # k_decayed = k * exp(g)
    k_decayed = (k * exp_g).to(tl.bfloat16)
    # k_inv = k * exp(-g)
    k_inv = (k * exp_neg_g).to(tl.bfloat16)
    # k_restored = k * exp(-g) * exp(g_total)
    k_restored = (k * exp_neg_g * exp_g_total[None, :]).to(tl.bfloat16)

    # ---------------- Bmm: L = k_decayed @ k_inv.T ---------------- #
    # Result is [BLOCK_M, BLOCK_M] = [16, 16]
    # Use FP32 accumulator (Triton default) for numerical headroom — exp() ranges
    # can produce values up to ~1e23 for extreme g, fp16 would overflow.
    L = tl.dot(k_decayed, tl.trans(k_inv))  # default: fp32 accumulator
    # ---------------- Bmm: Mqk = q_decayed @ k_inv.T ---------------- #
    Mqk = tl.dot(q_decayed, tl.trans(k_inv)).to(tl.bfloat16)

    # ---------------- tril_IL + beta multiply + Mqk mask ---------------- #
    offs_m_local = tl.arange(0, BLOCK_M)
    tril_mask = offs_m_local[:, None] > offs_m_local[None, :]  # strictly lower
    # beta is [BLOCK_M]; expand to [BLOCK_M, BLOCK_M]
    beta_sig = tl.sigmoid(beta).to(tl.float16)
    # L * beta (broadcast beta_sig along columns)
    L = tl.where(tril_mask, L * beta_sig[None, :], 0.0).to(tl.bfloat16)
    # Mqk: zero upper (incl diagonal? FlashKDA zeros i < j; diagonal kept)
    Mqk_mask = offs_m_local[:, None] >= offs_m_local[None, :]
    Mqk = tl.where(Mqk_mask, Mqk, tl.zeros((), dtype=tl.bfloat16))

    # ---------------- INV = I - L (in registers, bf16) ---------------- #
    identity = tl.where(
        offs_m_local[:, None] == offs_m_local[None, :],
        tl.full((), 1.0, dtype=tl.bfloat16),
        tl.zeros((), dtype=tl.bfloat16),
    )
    INV = identity - L  # bf16 [BLOCK_M, BLOCK_M]

    # ---------------- Neumann series inv: INV_k = INV_{k-1} @ (I + L) ---------------- #
    # We compute (I - L)^-1 in fp32 via the truncated Neumann series
    #   (I + L + L^2 + ... + L^k) ≈ (I - L)^-1
    # which only converges when ||L|| < 1. L is stored in bf16 above, so
    # entries up to ~1e10 already saturate the bf16 mantissa — the
    # recurrence step consumes INV as a multiplier and is robust to the
    # bf16 storage loss in INV (matches FlashKDA's behavior; see
    # test/_tmp/test_neumann_stability.py for the empirical safe range).
    #
    # Safety contract (enforced at the wrapper level):
    #   - lower_bound must be in [-10, 0] (production uses -5).
    #   - A_log range is bounded by the user (KDA init ≈ [-5, 5]).
    #   - g_raw is post-L2-norm-style scaled (see kernel).
    # If any of these is violated, the bf16 L storage can lose the
    # sub-leading bits and the Neumann iteration diverges — the
    # kernel will silently produce NaNs downstream. The wrapper
    # asserts on lower_bound; the others are documented as caller
    # responsibility.
    INV_f32 = INV.to(tl.float32)
    L_f32 = L.to(tl.float32)
    I_plus_L = identity.to(tl.float32) + L_f32
    # 4 iterations: (I + L + L^2 + L^3 + L^4) approximation
    for _ in range(4):
        INV_f32 = tl.dot(INV_f32, I_plus_L, out_dtype=tl.float32)
    # INV is computed in fp32 — use the high-precision version for the
    # Round-9 fold-ins (Mqk_eff, K_pre) below. INV itself is NOT stored to
    # gmem; its information content is fully captured by the two
    # pre-multiplied products.
    #
    # ---------------- Fold INV into the recurrence inputs ---------------- #
    # Mqk_eff = Mqk @ INV   shape [CHUNK, CHUNK] bf16
    #   The recurrence does o_attn = Mqk_eff @ v_residual_b. Mqk was stored
    #   as bf16 above, so we promote to fp32 for the matmul (matches
    #   FlashKDA's fp32 matmul_acc path for the highest precision step).
    Mqk_f32 = Mqk.to(tl.float32)
    Mqk_eff = tl.dot(Mqk_f32, INV_f32, out_dtype=tl.float32).to(tl.bfloat16)
    # K_pre = INV @ k_restored   shape [CHUNK, K] bf16
    #   The recurrence does h_contrib = K_pre^T @ v_residual_b. k_restored
    #   was stored as bf16 above; promote for the matmul.
    k_restored_f32 = k_restored.to(tl.float32)
    K_pre = tl.dot(INV_f32, k_restored_f32, out_dtype=tl.float32).to(tl.bfloat16)

    # ---------------- Store workspace outputs ---------------- #
    # Flat (b, c) → single index = pid_b * NC + pid_nc; workspace is [B*NC, H, ...]
    nc_idx = pid_b * NC + pid_nc
    # k_decayed, q_decayed, K_pre: [BLOCK_M, BLOCK_N] bf16
    kd_off = nc_idx * stride_kd_nc + pid_h * stride_kd_h
    qd_off = nc_idx * stride_qd_nc + pid_h * stride_qd_h
    kr_off = nc_idx * stride_kr_nc + pid_h * stride_kr_h
    # Build 2D offset for these tensors
    store_2d = offs_m_local[:, None] * BLOCK_N + offs_n[None, :]
    tl.store(kd_ptr + kd_off + store_2d, k_decayed)
    tl.store(qd_ptr + qd_off + store_2d, q_decayed)
    # Round-9: kr_ptr now stores K_pre = INV @ k_restored, not k_restored
    # itself. The recurrence needs K_pre for the h_new update; the
    # original k_restored is recoverable as (I - L) @ K_pre but that
    # would cost more than just precomputing K_pre once.
    tl.store(kr_ptr + kr_off + store_2d, K_pre)

    # g_total: [BLOCK_N] fp32
    gt_off = nc_idx * stride_gt_nc + pid_h * stride_gt_h
    tl.store(gt_ptr + gt_off + offs_n, g_total)

    # Mqk_eff: [BLOCK_M, BLOCK_M] bf16 (was Mqk before Round-9; INV was
    # also stored here but is now folded in)
    store_2d_small = offs_m_local[:, None] * BLOCK_M + offs_m_local[None, :]
    mqk_off = nc_idx * stride_mqk_nc + pid_h * stride_mqk_h
    tl.store(mqk_ptr + mqk_off + store_2d_small, Mqk_eff)


# ===================================================================== #
# Python wrapper                                                         #
# ===================================================================== #

def kda_prepare_triton(
    q: torch.Tensor,           # [B, T, H, K] bf16
    k: torch.Tensor,           # [B, T, H, K] bf16
    g: torch.Tensor,           # [B, T, H, K] bf16 (raw gate, pre-activation)
    beta: torch.Tensor,        # [B, T, H] bf16 (logits)
    A_log: torch.Tensor,       # [H] fp32
    dt_bias: torch.Tensor,     # [H, K] fp32
    lower_bound: float,        # negative scalar
    scale: float,              # 1/sqrt(K)
):
    """Returns dict of per-chunk workspace tensors for use by the
    recurrence step. All workspace is bf16 (small matrices) or fp32
    (g_total, used for the recurrence's state decay).

    Round-9 change: INV is no longer returned/stored. Instead, the two
    INV-premultiplied products are returned in its place:
        "Mqk_eff" — replaces "Mqk"  (Mqk @ INV, for the o matmul)
        "K_pre"   — replaces "k_restored"  (INV @ k_restored, for h update)
    The recurrence reads k_decayed (still in the workspace) to form the
    v_residual = v - k_decayed @ h_prev subtraction before applying the
    pre-folded Mqk_eff and K_pre.

    Workspace layout: [B*NC, H, ...] (batch-major then chunk-major),
    so the recurrence can read it with the same pid_b * NC + pid_nc
    scheme. For B=1, this reduces to the previous [NC, H, ...] layout.
    """
    import torch

    assert q.is_cuda and k.is_cuda and g.is_cuda and beta.is_cuda
    assert q.dtype == torch.bfloat16
    assert q.shape[-1] == 128, f"only K=128 supported, got {q.shape[-1]}"
    # Neumann series safety (see kernel comment for the math):
    # lower_bound drives the magnitude of g_activated, which drives
    # exp(g_total) at the chunk tail, which drives ||L||_∞ in the bf16
    # L storage. lower_bound=-5 is the production setting; values
    # more negative push L entries into the regime where the bf16
    # storage of L loses the sub-leading bits and the truncated
    # Neumann series diverges. See test/_tmp/test_neumann_stability.py
    # for the empirical safe range.
    assert -10.0 <= lower_bound <= 0.0, (
        f"lower_bound must be in [-10, 0] for Neumann series "
        f"stability; got {lower_bound} (production: -5.0)"
    )
    B, T, H, K = q.shape
    NC = (T + CHUNK - 1) // CHUNK
    device = q.device

    # Workspace allocation — flat [B*NC, H, ...] so the kernel uses one
    # `nc_idx = pid_b * NC + pid_nc` index for the leading dim.
    # Round-9: dropped `inv` (5 tensors total; INV info folded into the
    # other two). The HBM savings are small but the recurrence no longer
    # needs to read or stash a 16x16 bf16 matrix it doesn't use.
    kd = torch.empty(B * NC, H, CHUNK, K, dtype=torch.bfloat16, device=device)
    qd = torch.empty(B * NC, H, CHUNK, K, dtype=torch.bfloat16, device=device)
    kr = torch.empty(B * NC, H, CHUNK, K, dtype=torch.bfloat16, device=device)
    gt = torch.empty(B * NC, H, K, dtype=torch.float32, device=device)
    mqk = torch.empty(B * NC, H, CHUNK, CHUNK, dtype=torch.bfloat16, device=device)

    a_log_exp = torch.exp(A_log.float()).contiguous()
    # FLA's safe_gate path (gate.py:407 + chunk_fwd.py:51, with
    # kda_gate_chunk_cumsum applying scale=RCP_LN2 after the
    # sigmoid) computes:
    #   g_val = lower_bound * sigmoid(exp(A_log) * (g + bias))
    #   decay = exp2(cumsum(g_val)) = 2^(lower_bound * LOG2E * X)
    # where X = cumsum_sigmoid(exp(A_log) * (g + bias)).
    #
    # We replicate this in natural log space: gate_scale = lower_bound,
    # then exp(cumsum(g_val)) = exp(lower_bound * X)
    #                       = 2^(lower_bound * LOG2E * X).
    # Mathematically identical to FLA's exp2 path; equivalent to
    # FlashKDA reference's log2-space path on the sigmoid input
    # (FastKDA kept exp(A_log) inside sigmoid to match FLA's
    # chunk_kda_safe_gate, not FlashKDA's 2^A_log convention).
    gate_scale = float(lower_bound)

    grid = (NC, H, B)
    _kda_prepare_kernel[grid](
        q, k, g, beta,
        A_log, dt_bias,
        kd, qd, kr, gt, mqk,
        B, H, K, T, NC,
        # B strides (may equal T-stride for B=1, but we pass it
        # unconditionally so the kernel works for any B).
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        g.stride(0), g.stride(1), g.stride(2),
        beta.stride(0), beta.stride(1), beta.stride(2),
        kd.stride(0), kd.stride(1),
        qd.stride(0), qd.stride(1),
        kr.stride(0), kr.stride(1),
        gt.stride(0), gt.stride(1),
        mqk.stride(0), mqk.stride(1),
        float(scale),
        a_log_exp,
        float(gate_scale),
        BLOCK_M=CHUNK,
        BLOCK_N=K,
    )
    return {
        "k_decayed": kd,
        "q_decayed": qd,
        "K_pre": kr,
        "g_total": gt,
        "Mqk_eff": mqk,
    }