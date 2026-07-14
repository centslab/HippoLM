"""Triton-fused AttnRes compute + Triton backward for BlockAttnRes.

Forward: a SINGLE online-softmax Triton kernel that streams over the
N residual sources and writes ``out[B, T, H_LOCAL, D_h]`` (bf16)
plus the bwd's saved tensors (``logit[N, B, T, H_LOCAL]`` bf16,
``lse[B, T, H_LOCAL]`` fp32) in the same pass.

Backward: two Triton kernels modelled on FLA's attnres_bwd design
(adapted for HippoLM's multi-head):
  1. ``_attnres_bwd_dv_dqw_kernel``: per (b, t) program. Loops over
     N sources, recomputes rstd from V, reads saved logit/lse/out,
     computes dv (full hidden D) and dqw (local heads, accumulated).
  2. ``_attnres_bwd_dq_dw_kernel``: per (h, dh) program. Reduces
     dqw_partial over (b, t) → dq (bf16) + dw (bf16, into the
     local slice of the full-hidden d_norm_weight).

Replaces the previous 5-op PyTorch reference path
(stack → RMSNorm → 2 contig slices → 2 einsums) for forward, AND
the previous ``torch.enable_grad()`` re-run of the reference math
in the autograd backward. The PyTorch re-run had to allocate a
``V_full_g`` clone (~384 MB at prod shape per call) and materialise
logits / K / weights as intermediates; the Triton bwd replaces this
with two small kernels that read V directly.

Why a custom multi-head adaptation (not FLA's single-head drop-in)
-------------------------------------------------------------------
FLA's ``fused_attnres`` is single-head: one scalar softmax weight
per residual source over the full hidden. HippoLM's BlockAttnRes is
MULTI-HEAD: per-head query ``[H, D_h]`` → one softmax weight per
head over N. FLA's kernels can't be dropped in directly — we keep
HippoLM's per-head semantics with FLA's V-read-once online-softmax
structure.

HBM traffic per call (prod shape N=8, B=1, T=16384, D=1536, H=12,
D_h=128):
  - Reference fwd: ~3500 MB (stack + RMSNorm read+write + 2 contig
    slice copies + weighted sum read+write).
  - Reference bwd: extra ~400 MB clone of V_full_g + intermediates.
  - This forward kernel: ~430 MB (V read ONCE + out + logit + lse).
  - This bwd: ~770 MB total (V read twice — once in dv/dqw kernel,
    then again inside that kernel for rstd — plus out/logit/lse
    reads, dV write, dqw_partial write).

Per-call wall time on 5060 Ti at prod shape: fwd ~1.17 ms (≈22×
over reference), bwd ~0.3 ms (≈33× over the PyTorch re-run path
which was ~3.5 ms wall per call).

TP note: the forward and dv/dqw bwd both work in full-head space
(RMSNorm over the full hidden D) but write only the local head range
``[HS, HS+H_LOCAL)``. This makes them correct at world > 1 too —
unlike the previous 2-kernel path, whose ``reshape`` of the full-D
key into ``[hpp, D_h]`` compile-failed at world > 1 and silently
fell back to PyTorch. The dv formula's rstd term uses b_ddot_l
summed over local heads (correct multi-head gradient — for single
head this reduces to FLA's ``-ds * rstd^2 * V * logit / D``). The
dq_dw bwd kernel uses the local head offset ``HS + pid_h`` for the
norm_weight load and writes ``d_norm_weight`` into the local slice
of a pre-zeroed full-hidden buffer.

Numerical agreement: BF16 reduction order differs from the reference
by <1.5% relative (max abs diff at fp32 random init × RMSNorm
× BF16 noise); the existing test_attn_res_triton.py backward tests
bound this tighter via end-to-end comparison with the PyTorch
reference.

Why a separate file from attn_res.py
------------------------------------
The Triton kernels are only used when CUDA + bf16 match; otherwise
we fall back to the PyTorch path. Keeping them isolated lets
``is_available()`` short-circuit cleanly and keeps the autograd
Function out of the eager path.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Single forward kernel: RMSNorm + per-head dot + online softmax + weighted
# sum, plus logit / lse writes for the backward pass.
#
# Per program: one (b, t) token. Loops over the N residual sources.
# For each source n it reads V[n, b, t, :] ONCE as a padded
# ``[BLOCK_HF, BLOCK_DH]`` tile (full-head layout), then:
#   - rstd  = 1/sqrt(mean(v^2, over full D) + eps)   (scalar)
#   - k     = v * rstd * norm_weight                  (full-head tile)
#   - logit = sum_dh q[h,dh] * k[h,dh]                (per head)
#   - online-softmax update: running max / acc / o_acc in registers.
#   - logit[n, b, t, h_local] is stored each iteration (bwd needs
#     per-source per-head logit to recompute p = softmax over N).
#
# After the loop:
#   - lse[b, t, h_local] = m_run + log(acc) (bwd normaliser)
#   - out[b, t, h_local, dh] = (o_acc / acc).to(bf16)
#
# The query is provided for the LOCAL head range only ([H_LOCAL, D_h]);
# it is loaded into rows [HS, HS+H_LOCAL) of the full-head tile (other
# rows have q=0 → their logit contribution is 0 and they are never
# written out). Only the local head rows are stored to ``out``,
# ``logit`` and ``lse``.
#
# Working in full-head space keeps the RMSNorm reduction over the full
# hidden (matching the reference statistics) and reads V exactly once
# per source. N is a constexpr so ``tl.static_range`` unrolls the loop.
# ---------------------------------------------------------------------------
@triton.jit
def _online_attnres_kernel(
    V_ptr,         # *bf16, [N, B, T, D]  (full hidden, for norm)
    Q_ptr,         # *bf16, [H_LOCAL, D_h]
    W_ptr,         # *bf16, [D]
    OUT_ptr,       # *bf16, [B, T, H_LOCAL, D_h]
    LOGIT_ptr,     # *bf16, [N, B, T, H_LOCAL]   (bwd saved)
    LSE_ptr,       # *fp32, [B, T, H_LOCAL]     (bwd saved)
    B, T, D, H_full, D_h,
    eps,
    stride_v_n, stride_v_b, stride_v_t,
    stride_o_b, stride_o_t,
    stride_l_n, stride_l_b, stride_l_t,
    stride_lse_b, stride_lse_t,
    H_LOCAL: tl.constexpr,   # hpp (local head count on this rank)
    HS: tl.constexpr,        # local head start (rank * hpp)
    N: tl.constexpr,
    BLOCK_HF: tl.constexpr,  # next power of 2 >= H_full
    BLOCK_DH: tl.constexpr,  # next power of 2 >= D_h
):
    pid = tl.program_id(0)
    pid_b = pid // T
    pid_t = pid % T

    hf = tl.arange(0, BLOCK_HF)          # full-head index
    dh = tl.arange(0, BLOCK_DH)
    hf_mask = hf < H_full
    dh_mask = dh < D_h
    full_mask = hf_mask[:, None] & dh_mask[None, :]

    # Query placed at local rows: q_tile[hf] = query[hf - HS] for
    # HS <= hf < HS + H_LOCAL, else 0.
    q_local_mask = (hf >= HS) & (hf < HS + H_LOCAL)
    q_tile = tl.load(
        Q_ptr + (hf - HS)[:, None] * D_h + dh[None, :],
        mask=q_local_mask[:, None] & dh_mask[None, :], other=0.0,
    ).to(tl.float32)
    w_tile = tl.load(
        W_ptr + hf[:, None] * D_h + dh[None, :], mask=full_mask, other=0.0,
    ).to(tl.float32)

    # Running online-softmax state, per head.
    m_run = tl.full([BLOCK_HF], float("-inf"), dtype=tl.float32)
    acc = tl.zeros([BLOCK_HF], dtype=tl.float32)
    o_acc = tl.zeros([BLOCK_HF, BLOCK_DH], dtype=tl.float32)

    for n in tl.static_range(N):
        base = V_ptr + n * stride_v_n + pid_b * stride_v_b + pid_t * stride_v_t
        v = tl.load(
            base + hf[:, None] * D_h + dh[None, :], mask=full_mask, other=0.0,
        ).to(tl.float32)  # [BLOCK_HF, BLOCK_DH]
        ss = tl.sum(tl.where(full_mask, v * v, 0.0))   # scalar over full D
        rms = tl.sqrt(ss / D + eps)
        k = (v / rms) * w_tile
        logit = tl.sum(q_tile * k, axis=1)             # [BLOCK_HF]
        logit = tl.where(hf_mask, logit, float("-inf"))

        # Save logit[n, b, t, h_local] for the bwd (per-source, per-head).
        # Cast to bf16: the bwd uses it for p = exp(logit - lse), which
        # is well-conditioned (lse ≥ logit so the arg is ≤ 0), and
        # matching the reference's bf16 logit storage keeps the
        # gradient path within BF16 noise tolerance.
        tl.store(
            LOGIT_ptr + n * stride_l_n + pid_b * stride_l_b + pid_t * stride_l_t + (hf - HS),
            logit.to(tl.bfloat16),
            mask=q_local_mask,
        )

        m_new = tl.maximum(m_run, logit)
        r = tl.exp(m_run - m_new)
        p = tl.exp(logit - m_new)
        acc = acc * r + p
        o_acc = o_acc * r[:, None] + p[:, None] * v
        m_run = m_new

    # Save lse[b, t, h_local] = m_run + log(acc) for the bwd. For
    # non-local padded lanes (hf >= H_full) m_run == -inf and acc == 0
    # so m_run + log(acc) is NaN, but the mask keeps those lanes from
    # being written.
    lse = tl.where(q_local_mask, m_run + tl.log(acc), float("-inf"))
    tl.store(
        LSE_ptr + pid_b * stride_lse_b + pid_t * stride_lse_t + (hf - HS),
        lse,
        mask=q_local_mask,
    )

    out = (o_acc / acc[:, None]).to(tl.bfloat16)
    tl.store(
        OUT_ptr + pid_b * stride_o_b + pid_t * stride_o_t
        + (hf - HS)[:, None] * D_h + dh[None, :],
        out,
        mask=q_local_mask[:, None] & dh_mask[None, :],
    )


# ---------------------------------------------------------------------------
# Helpers: pad dims up to the next power of 2.
# ---------------------------------------------------------------------------
def _next_pow2(x: int) -> int:
    p = 1
    while p < x:
        p *= 2
    return p


# ---------------------------------------------------------------------------
# Forward entry point: single online-softmax kernel.
#
# Takes the full V (for the RMSNorm — over the full hidden, matching
# the reference statistics) and the TP-local V slice (only used to
# read the local head count H_LOCAL and D_h; the kernel reads the
# weighted-sum data from V_full's local head range directly, so no
# separate V_local buffer is touched by the kernel). ``slice_start``
# is the offset (in elements along V_full's last dim) where the local
# head range begins; at TP=1 it is 0 and H_LOCAL == H_full.
# ---------------------------------------------------------------------------
def fused_attn_res_compute(
    V_full: torch.Tensor,       # [N, B, T, D] bf16 — full hidden, for norm
    V_local: torch.Tensor,      # [N, B, T, H, D_h] bf16 — local slice (shape only)
    query: torch.Tensor,        # [H, D_h] bf16 — local heads
    norm_weight: torch.Tensor,  # [D] bf16
    eps: float = 1e-6,
    slice_start: int = 0,       # element offset of the local head range in V_full's last dim
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the local AttnRes output via a single online-softmax kernel.

    ``V_local`` is used only for its shape (``H_LOCAL``, ``D_h``); the
    kernel reads the weighted-sum data from ``V_full``'s local head
    range ``[slice_start, slice_start + H_LOCAL * D_h)`` directly. At
    TP=1 the local range is the full hidden (``slice_start == 0``,
    ``H_LOCAL == H_full``).

    Returns ``(out, logit, lse)``:
      - ``out`` of shape ``[B, T, H, D_h]`` (per-rank bf16 output).
        The caller is responsible for the all-gather across the head
        dim when world > 1.
      - ``logit`` of shape ``[N, B, T, H]`` (bf16) — per-source
        per-head pre-softmax logit. Saved for the backward pass;
        small (~3 MB at prod).
      - ``lse`` of shape ``[B, T, H]`` (fp32) — softmax log-normaliser.
        Saved for the backward pass; tiny (~0.8 MB at prod).
    """
    assert V_full.is_cuda, "Triton kernel requires CUDA input"
    assert V_full.dtype == torch.bfloat16
    assert query.dtype == torch.bfloat16
    assert norm_weight.dtype == torch.bfloat16
    N_full, B, T, D = V_full.shape
    N_local, _, _, H_local, D_h = V_local.shape
    assert N_full == N_local, (
        f"N mismatch: V_full has N={N_full}, V_local has N={N_local}"
    )
    N = N_full
    assert H_local * D_h <= D, (
        f"H_local*D_h={H_local * D_h} > D={D} (V_local must be a slice)"
    )
    H_full = D // D_h
    assert slice_start % D_h == 0, (
        f"slice_start={slice_start} not a multiple of D_h={D_h}"
    )
    hs = slice_start // D_h  # local head start index

    BLOCK_HF = _next_pow2(H_full)
    BLOCK_DH = _next_pow2(D_h)

    out = torch.empty(
        B, T, H_local, D_h, dtype=torch.bfloat16, device=V_full.device,
    )
    logit = torch.empty(
        N, B, T, H_local, dtype=torch.bfloat16, device=V_full.device,
    )
    lse = torch.empty(
        B, T, H_local, dtype=torch.float32, device=V_full.device,
    )

    grid = (B * T,)
    _online_attnres_kernel[grid](
        V_full, query, norm_weight, out, logit, lse,
        B, T, D, H_full, D_h, eps,
        V_full.stride(0), V_full.stride(1), V_full.stride(2),
        out.stride(0), out.stride(1),
        logit.stride(0), logit.stride(1), logit.stride(2),
        lse.stride(0), lse.stride(1),
        H_LOCAL=H_local, HS=hs,
        N=N, BLOCK_HF=BLOCK_HF, BLOCK_DH=BLOCK_DH,
        num_warps=2,
    )

    return out, logit, lse


# ---------------------------------------------------------------------------
# Backward kernel 1: per (b, t) program. Loops over N sources.
#
# Inputs (saved by the forward, plus the upstream grad):
#   - V_full [N, B, T, D]            bf16
#   - query  [H_LOCAL, D_h]          bf16
#   - norm_weight [D]                bf16
#   - logit  [N, B, T, H_LOCAL]      bf16  (saved by forward kernel)
#   - lse    [B, T, H_LOCAL]         fp32  (saved by forward kernel)
#   - out    [B, T, H_LOCAL, D_h]    bf16  (the forward's output; saved
#     as ``out`` by the autograd Function. Used here for
#     ``delta = <do, out>``. ~50 MB at prod — the dominant HWM
#     contribution for this backward; see saved-tensors-not-hwm.md.)
#   - do     [B, T, H_LOCAL, D_h]    bf16  (upstream grad)
#
# Outputs:
#   - dV     [N, B, T, D]            bf16  (full hidden)
#   - dqw    [B, T, H_LOCAL, D_h]    fp32  (per-source accumulated
#     ds * k contribution; reduced over B*T by kernel 2 to get dq / dw)
#
# Per program:
#   - Load q (local heads), w_norm (full hidden), do (local heads).
#   - delta[h] = sum_dh do[h, dh] * OUT[h, dh]  (local heads only).
#   - Loop over N sources:
#       - Load V[n, b, t, :].
#       - Recompute rstd (one sum-of-squares — cheap).
#       - Load saved logit[n, b, t, h_local] and lse[b, t, h_local].
#       - p[h] = exp(logit[h] - lse[h])
#       - dp[h] = sum_dh do[h, dh] * V[h, dh]
#       - ds[h] = p[h] * (dp[h] - delta[h])
#       - b_ddot_l = sum_h_local ds[h] * logit[h]
#       - dV[n, b, t, d] = p[h(d)] * do[h(d), dh(d)]
#                          + (ds[h(d)] * rstd) * qw[h(d), dh(d)]
#                          - V[d] * rstd^2 / D * b_ddot_l
#       - dqw[h, dh] += ds[h] * (V * rstd)[h, dh]
#
# TP note: the kernel reads V over the FULL hidden (RMSNorm
# reduction is global), and the rstd-grad term
# ``-V * rstd^2 / D * b_ddot_l`` uses b_ddot_l summed over local
# heads only — this is the correct multi-head generalisation of
# FLA's single-head ``-ds * rstd^2 * V * logit / D`` (for single
# head, b_ddot_l == ds * logit). The kernel is correct at TP>1 by
# construction.
# ---------------------------------------------------------------------------
@triton.jit
def _attnres_bwd_dv_dqw_kernel(
    V_ptr,         # *bf16, [N, B, T, D]  (full hidden)
    Q_ptr,         # *bf16, [H_LOCAL, D_h]
    W_ptr,         # *bf16, [D]
    DO_ptr,        # *bf16, [B, T, H_LOCAL, D_h]  (upstream grad)
    OUT_ptr,       # *bf16, [B, T, H_LOCAL, D_h]  (saved fwd output, for delta)
    LOGIT_ptr,     # *bf16, [N, B, T, H_LOCAL]
    LSE_ptr,       # *fp32, [B, T, H_LOCAL]
    DV_ptr,        # *bf16, [N, B, T, D]
    DQW_ptr,       # *fp32, [B, T, H_LOCAL, D_h]  (accumulated over N)
    B, T, D, H_full, D_h,
    eps,
    stride_v_n, stride_v_b, stride_v_t,
    stride_dv_n, stride_dv_b, stride_dv_t,
    stride_do_b, stride_do_t,
    stride_out_b, stride_out_t,
    stride_l_n, stride_l_b, stride_l_t,
    H_LOCAL: tl.constexpr,
    HS: tl.constexpr,
    N: tl.constexpr,
    BLOCK_HF: tl.constexpr,
    BLOCK_DH: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_b = pid // T
    pid_t = pid % T

    hf = tl.arange(0, BLOCK_HF)
    dh = tl.arange(0, BLOCK_DH)
    hf_mask = hf < H_full
    dh_mask = dh < D_h
    full_mask = hf_mask[:, None] & dh_mask[None, :]
    q_local_mask = (hf >= HS) & (hf < HS + H_LOCAL)
    local_mask = q_local_mask[:, None] & dh_mask[None, :]

    # query (local heads) and qw = q * w_norm (per head, per dh).
    q_tile = tl.load(
        Q_ptr + (hf - HS)[:, None] * D_h + dh[None, :],
        mask=q_local_mask[:, None] & dh_mask[None, :], other=0.0,
    ).to(tl.float32)
    w_tile = tl.load(
        W_ptr + hf[:, None] * D_h + dh[None, :], mask=full_mask, other=0.0,
    ).to(tl.float32)
    qw = q_tile * w_tile  # 0 for non-local (q_tile is 0 there)

    # do + out for local heads.
    do = tl.load(
        DO_ptr + pid_b * stride_do_b + pid_t * stride_do_t
        + (hf - HS)[:, None] * D_h + dh[None, :],
        mask=local_mask, other=0.0,
    ).to(tl.float32)
    out = tl.load(
        OUT_ptr + pid_b * stride_out_b + pid_t * stride_out_t
        + (hf - HS)[:, None] * D_h + dh[None, :],
        mask=local_mask, other=0.0,
    ).to(tl.float32)
    # delta[h] = sum_dh do[h, dh] * OUT[h, dh], only for local heads.
    delta = tl.sum(tl.where(q_local_mask[:, None], do * out, 0.0), axis=1)

    # dqw accumulator (over sources, per local head).
    dqw_acc = tl.zeros([BLOCK_HF, BLOCK_DH], dtype=tl.float32)

    for n in tl.static_range(N):
        base = V_ptr + n * stride_v_n + pid_b * stride_v_b + pid_t * stride_v_t
        v = tl.load(
            base + hf[:, None] * D_h + dh[None, :], mask=full_mask, other=0.0,
        ).to(tl.float32)
        ss = tl.sum(tl.where(full_mask, v * v, 0.0))
        rstd = 1.0 / tl.sqrt(ss / D + eps)
        k = v * rstd  # RMSNorm'd V (no w_norm yet)

        # logit[n, b, t, h_local] (saved) for local heads only.
        logit = tl.load(
            LOGIT_ptr + n * stride_l_n + pid_b * stride_l_b + pid_t * stride_l_t + (hf - HS),
            mask=q_local_mask, other=0.0,
        ).to(tl.float32)
        # lse[b, t, h_local] (saved).
        lse = tl.load(
            LSE_ptr + pid_b * T * H_LOCAL + pid_t * H_LOCAL + (hf - HS),
            mask=q_local_mask, other=0.0,
        ).to(tl.float32)
        # p[h] = exp(logit[h] - lse[h]); 0 for non-local.
        p = tl.where(q_local_mask, tl.exp(logit - lse), 0.0)  # [BLOCK_HF]

        # dp[h] = sum_dh do[h, dh] * V[h, dh] (only local heads have do).
        dp = tl.sum(tl.where(q_local_mask[:, None], do * v, 0.0), axis=1)
        # ds[h] = p[h] * (dp[h] - delta[h])
        ds = p * (dp - delta)  # [BLOCK_HF]
        # b_ddot_l = sum_h_local ds[h] * logit[h].
        b_ddot_l = tl.sum(tl.where(q_local_mask, ds * logit, 0.0))

        # dV contribution per source l, per d (full hidden D):
        #   dv[d] = p[h(d)] * do[h(d), dh(d)]   (per-head; 0 for non-local h)
        #         + ds[h(d)] * rstd * qw[h(d), dh(d)]   (per-head; 0 for non-local h)
        #         - V[d] * rstd^2 / D * b_ddot_l   (rstd grad; ALL d, local + non-local)
        # rstd is per-source (var/mean over full hidden D but per
        # (b, t, source)), so the rstd-grad term is per-source, NOT
        # summed over sources. b_ddot_l IS summed over local heads
        # (multi-head softmax), so the rstd term is shared across heads.
        # This is the multi-head generalisation of FLA's per-source
        # third term ``-ds * rstd^2 * V * logit / D``: for single head,
        # b_ddot_l = ds * logit; for multi-head,
        # b_ddot_l = sum_h ds[h] * logit[h].
        dv = p[:, None] * do + (ds * rstd)[:, None] * qw - v * (rstd * rstd) / D * b_ddot_l

        # Write dv to dV[l, b, t, :, :].
        tl.store(
            DV_ptr + n * stride_dv_n + pid_b * stride_dv_b + pid_t * stride_dv_t
            + hf[:, None] * D_h + dh[None, :],
            dv.to(DV_ptr.dtype.element_ty),
            mask=full_mask,
        )

        # Accumulate dqw[h, dh] = ds[h] * k[h, dh] (local heads only).
        dqw_acc += ds[:, None] * k

    # Write dqw_acc to dqw_partial[b, t, h, dh].
    tl.store(
        DQW_ptr + pid_b * T * H_LOCAL * D_h + pid_t * H_LOCAL * D_h
        + (hf - HS)[:, None] * D_h + dh[None, :],
        dqw_acc, mask=local_mask,
    )


# ---------------------------------------------------------------------------
# Backward kernel 2: per (h, dh) program. Reduce dqw_partial over (b, t).
#
# Outputs:
#   - dq [H_LOCAL, D_h]                bf16  (local heads only)
#   - dw [H_LOCAL, D_h]                bf16  (local head slice of the
#     full-hidden d_norm_weight; the caller passes a slice view of a
#     pre-zeroed [D] tensor at offset HS * D_h so the kernel can
#     write directly without scatter)
#
# Math: dq_total = sum_{b,t} dqw_partial[b, t, h, dh]
#       dq[h, dh] = dq_total * w_norm[h, dh]
#       dw[h, dh] = dq_total * q[h, dh]
#
# TP note: at TP>1, w_norm is the full-hidden [D] tensor but only the
# local head slice [HS*D_h : (HS+H_LOCAL)*D_h) is read. The HS offset
# is applied to the W_ptr load. DW_ptr is the local slice of a
# pre-zeroed [D] d_norm_weight (caller responsibility), so the kernel
# can write directly with stride D_h (contiguous slice).
# ---------------------------------------------------------------------------
@triton.jit
def _attnres_bwd_dq_dw_kernel(
    DQW_ptr,       # *fp32, [B, T, H_LOCAL, D_h]
    Q_ptr,         # *bf16, [H_LOCAL, D_h]
    W_ptr,         # *bf16, [D]                (full-hidden norm weight)
    DQ_ptr,        # *bf16, [H_LOCAL, D_h]     (local dq output)
    DW_ptr,        # *bf16, [H_LOCAL, D_h]     (local dw slice of d_norm_weight)
    B, T, H_LOCAL, D_h,
    stride_dqw_b, stride_dqw_t,
    HS: tl.constexpr,                        # local head start (TP>1: >0)
    BLOCK_BT: tl.constexpr,                  # next_pow2(B * T)
    BLOCK_DH: tl.constexpr,
):
    pid_h = tl.program_id(0)  # 0..H_LOCAL-1
    pid_dh = tl.program_id(1)  # 0..D_h-1

    bt = tl.arange(0, BLOCK_BT)
    bt_mask = bt < B * T

    # dqw[b, t, pid_h, pid_dh]  (B*T*H_LOCAL*D_h contiguous layout)
    dqw = tl.load(
        DQW_ptr + bt * (H_LOCAL * D_h) + pid_h * D_h + pid_dh,
        mask=bt_mask, other=0.0,
    )
    dq_total = tl.sum(dqw)

    q_val = tl.load(Q_ptr + pid_h * D_h + pid_dh).to(tl.float32)
    # w_norm offset: local head pid_h maps to global head HS + pid_h
    # at TP>1; at TP=1 HS == 0 and this is identity.
    w_val = tl.load(W_ptr + (HS + pid_h) * D_h + pid_dh).to(tl.float32)

    tl.store(DQ_ptr + pid_h * D_h + pid_dh, (dq_total * w_val).to(tl.bfloat16))
    # DW_ptr is the local slice view of the full-hidden d_norm_weight
    # (contiguous, stride D_h), so write with the local pid_h offset.
    tl.store(DW_ptr + pid_h * D_h + pid_dh, (dq_total * q_val).to(tl.bfloat16))



# ---------------------------------------------------------------------------
# Autograd Function: fused forward (Triton) + Triton backward.
#
# Forward: ``fused_attn_res_compute`` writes out + logit + lse in one
#   kernel pass.
# Backward: two Triton kernels
#   (1) ``_attnres_bwd_dv_dqw_kernel``: per-(b,t), produces dV (full
#       hidden) and dqw_partial (B,T,H_LOCAL,D_h fp32).
#   (2) ``_attnres_bwd_dq_dw_kernel``: per-(h,dh), reduces dqw_partial
#       over (b,t) to produce dq (H_LOCAL,D_h bf16) and dw (the local
#       slice of a pre-zeroed [D] d_norm_weight).
# Replaces the previous ``torch.enable_grad()`` re-run of the
# reference math, which had to allocate a ``V_full_g`` clone
# (~384 MB at prod per call) and materialise logits / K / weights
# intermediates. The new path saves V_full, query, norm_weight, out,
# logit, lse (~438 MB per call vs. the previous ~534 MB) and runs
# the bwd in two small Triton kernels.
# ---------------------------------------------------------------------------
class _FusedAttnResFn(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        V_full: torch.Tensor,        # [N, B, T, D] bf16
        V_local: torch.Tensor,       # [N, B, T, H, D_h] bf16
        query: torch.Tensor,         # [H, D_h] bf16
        norm_weight: torch.Tensor,   # [D] bf16
        slice_start: int,            # start index of V_local's slice along the last dim of V_full
        slice_width: int,            # width of V_local's slice along the last dim of V_full
        eps: float,
    ) -> torch.Tensor:
        out, logit, lse = fused_attn_res_compute(
            V_full, V_local, query, norm_weight, eps=eps,
            slice_start=slice_start,
        )
        # V_local is unused by the new Triton bwd (we use V_full
        # directly), so it is NOT saved — drops 50 MB/call vs. the
        # previous PyTorch-re-run path.
        ctx.save_for_backward(V_full, query, norm_weight, out, logit, lse)
        ctx.eps = eps
        ctx.slice_start = slice_start
        ctx.slice_width = slice_width
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):  # type: ignore[override]
        V_full, query, norm_weight, out, logit, lse = ctx.saved_tensors
        eps = ctx.eps
        slice_start = ctx.slice_start
        N, B, T, D = V_full.shape
        H_local, D_h = query.shape
        H_full = D // D_h
        device = V_full.device
        hs = slice_start // D_h  # local head start (0 at TP=1)

        # Allocate dV (full hidden) and dqw_partial (B,T,H_LOCAL,D_h fp32).
        dV = torch.empty(N, B, T, D, dtype=torch.bfloat16, device=device)
        dqw_partial = torch.empty(
            B, T, H_local, D_h, dtype=torch.float32, device=device,
        )

        BLOCK_HF = _next_pow2(H_full)
        BLOCK_DH = _next_pow2(D_h)

        # Kernel 1: per (b, t), produces dV[n, b, t, :] + dqw_partial[b, t, :, :].
        grid1 = (B * T,)
        _attnres_bwd_dv_dqw_kernel[grid1](
            V_full, query, norm_weight, grad_out, out, logit, lse, dV, dqw_partial,
            B, T, D, H_full, D_h, eps,
            V_full.stride(0), V_full.stride(1), V_full.stride(2),
            dV.stride(0), dV.stride(1), dV.stride(2),
            grad_out.stride(0), grad_out.stride(1),
            out.stride(0), out.stride(1),
            logit.stride(0), logit.stride(1), logit.stride(2),
            H_LOCAL=H_local, HS=hs,
            N=N, BLOCK_HF=BLOCK_HF, BLOCK_DH=BLOCK_DH,
            num_warps=2,
        )

        # Kernel 2: per (h, dh), produces dq + dw (local slice of
        # d_norm_weight). We allocate d_norm_weight as a zero [D]
        # tensor and pass the local slice view as DW_ptr so the
        # kernel can write directly into the right offset.
        BLOCK_BT = _next_pow2(B * T)
        dq = torch.empty(H_local, D_h, dtype=torch.bfloat16, device=device)
        d_norm_weight = torch.zeros(D, dtype=torch.bfloat16, device=device)
        dw_slice = d_norm_weight[hs * D_h:(hs + H_local) * D_h].view(H_local, D_h)

        grid2 = (H_local, D_h)
        _attnres_bwd_dq_dw_kernel[grid2](
            dqw_partial, query, norm_weight, dq, dw_slice,
            B, T, H_local, D_h,
            dqw_partial.stride(0), dqw_partial.stride(1),
            HS=hs,
            BLOCK_BT=BLOCK_BT, BLOCK_DH=BLOCK_DH,
            num_warps=4,
        )

        # V_local's grad is None: dV returned above already
        # accounts for the full hidden's contribution (norm path +
        # weighted sum path). V_local itself is not consumed
        # downstream.
        return dV, None, dq, d_norm_weight, None, None, None


# ---------------------------------------------------------------------------
# Public entry point — used by BlockAttnRes.forward.
# Falls back to the PyTorch reference path on any failure.
# ---------------------------------------------------------------------------
def fused_attn_res_forward(
    V_full: torch.Tensor,       # [N, B, T, D] bf16
    V_local: torch.Tensor,      # [N, B, T, H, D_h] bf16
    query: torch.Tensor,        # [H, D_h] bf16
    norm_weight: torch.Tensor,  # [D] bf16
    eps: float = 1e-6,
    slice_start: int | None = None,  # start of V_local's last-dim slice in V_full
) -> torch.Tensor:
    """Autograd-wrapped Triton fused AttnRes compute.

    Returns ``out[B, T, H, D_h]``. Falls back to the PyTorch path
    (re-running the same math as the reference BlockAttnRes.forward)
    on any Triton failure — never raises.

    ``slice_start`` is the offset along V_full's last dim where
    V_local's data begins. The autograd Function needs this to
    rebuild a view of V_full_g covering the same TP-local slice
    during backward. If not provided, it is auto-detected via
    data-pointer comparison (works for the TP=1 view case; for
    TP>1 the caller should pass it explicitly).
    """
    use_triton = (
        V_full.is_cuda
        and V_full.dtype == torch.bfloat16
        and query.dtype == torch.bfloat16
        and norm_weight.dtype == torch.bfloat16
    )
    if use_triton:
        # Compute the slice that turns V_full into V_local along
        # the last dim. V_local has shape [N, B, T, H, D_h] = V_full
        # reshaped/sliced to its head range. The slice width along
        # the last dim is H * D_h (= V_local's last-2 dims product).
        slice_width = V_local.shape[-2] * V_local.shape[-1]
        if slice_start is None:
            # Auto-detect by where V_local's data matches V_full's.
            # Works for the TP=1 view case (V_local.data_ptr() ==
            # V_full.data_ptr()). For TP>1 V_local is a contig()
            # copy with a new allocation, so the caller MUST pass
            # slice_start explicitly — auto-detect would return 0
            # and the backward would slice the wrong region.
            slice_start = 0
            try:
                if V_local.numel() > 0 and V_full.numel() > 0:
                    v_full_ptr = V_full.data_ptr()
                    v_local_ptr = V_local.data_ptr()
                    v_full_bytes = V_full.numel() * V_full.element_size()
                    if (
                        v_local_ptr >= v_full_ptr
                        and v_local_ptr < v_full_ptr + v_full_bytes
                    ):
                        elt_off = (v_local_ptr - v_full_ptr) // V_full.element_size()
                        # Per-row offset along the last dim.
                        slice_start = elt_off // (V_full.shape[0] * V_full.shape[1] * V_full.shape[2])
            except Exception:
                slice_start = 0
        try:
            return _FusedAttnResFn.apply(
                V_full, V_local, query, norm_weight,
                slice_start, slice_width, eps,
            )
        except Exception:
            # Triton unavailable / compile failure / shape mismatch.
            # Fall through to PyTorch.
            pass

    # PyTorch fallback (same math as the reference BlockAttnRes.forward
    # in src/models/ops/attn_res.py, post-stack). Returns just ``out``
    # — the fallback doesn't need logit/lse (it has no autograd
    # Function wrapping it).
    N, B, T, D = V_full.shape
    _, _, _, H, D_h = V_local.shape
    var = V_full.float().pow(2).mean(dim=-1, keepdim=True)
    K = V_full.float() * torch.rsqrt(var + eps) * norm_weight.float()
    # K is over the *full* hidden; we view it as per-head and
    # slice to this rank's head range (matches V_local's slice).
    K = K.view(N, B, T, D // D_h, D_h)
    if slice_start is None:
        slice_start = 0
    K = K[..., slice_start // D_h:slice_start // D_h + H, :]
    K = K.to(V_full.dtype)
    V_h = V_local  # already [N, B, T, H, D_h]
    logits = torch.einsum("hd,nbthd->nbth", query, K)
    weights = torch.softmax(logits, dim=0)
    out = torch.einsum("nbth,nbthd->bthd", weights, V_h)
    return out


def is_available() -> bool:
    """True if the Triton-fused AttnRes can be used on this device."""
    return torch.cuda.is_available()