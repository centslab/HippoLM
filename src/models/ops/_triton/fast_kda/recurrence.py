"""Fused Triton KDA recurrence kernel.

Implements FlashKDA's Kernel 2 algorithm in pure Triton (no CUTLASS,
no external deps).

Critical design point: each program processes ALL chunks for ONE head
and ONE V-slice sequentially via a Python for-loop inside the kernel.
This guarantees that h[c-1] is written before h[c] reads it (no
cross-program race).

Per-program algorithm (Round-9 — applies (I - L)^-1 correctly):
  for c in 0..NC-1:
      v_residual = V_chunk - k_decayed[c] @ h_prev           # NEW
      v_residual_b = v_residual * beta[c]                    # NEW
      o_chunk     = Mqk_eff[c] @ v_residual_b + q_decayed[c] @ h_prev
      h_new       = h_prev * exp(g_total[c]) + K_pre[c]^T @ v_residual_b
      store o[c] and h[c]

The Round-9 fix is what makes the output numerically correct. The
first draft did:
  o_chunk = Mqk[c] @ V_chunk + q_decayed[c] @ h_prev
  h_new   = h_prev * exp(g_total[c]) + k_restored[c]^T @ V_chunk
which is the "naive" form that ignores the (I - L)^-1 contribution
to the intra-chunk output. The correct chunk-wise GDR (matching
FlashKDA's reference at /hy-tmp/FlashKDA/tests/torch_ref.py:232-243
and FLA's KDA path) is:
  o[t] = q_decayed[t] @ h_prev
       + Mqk @ (INV @ ((V - k_decayed @ h_prev) * beta))
which we rewrite as Mqk_eff = Mqk @ INV and K_pre = INV @ k_restored
precomputed in prepare, plus the per-chunk v_residual = V - k_decayed
@ h_prev subtraction. See prepare.py for the fold-in math.

Workspace inputs (produced by prepare.py, flat [B*NC, H, ...]):
  k_decayed    [B*NC, H, CHUNK, K]   bf16   (NEW read in Round-9)
  q_decayed    [B*NC, H, CHUNK, K]   bf16
  K_pre        [B*NC, H, CHUNK, K]   bf16   (was k_restored)
  g_total      [B*NC, H, K]          fp32
  Mqk_eff      [B*NC, H, CHUNK, CHUNK] bf16   (was Mqk)
  beta         [B, T, H]             bf16   (NEW input; per-token)
  v            [B, T, H, V]          bf16   (main input; not workspace)

State I/O:
  h_intermediate [B*NC, H, K, V]     bf16  (only last chunk per batch populated)
  initial_state  [B, H, K, V]        fp32  (state at start of first chunk)
  final_state    [B, H, K, V]        fp32  (state at end of last chunk)

Outputs:
  O            [B, T, H, V]         bf16

Design notes
------------
V-split: Each program processes a V-block of BLOCK_V columns. Grid is
(H, V // BLOCK_V). With BLOCK_V=16 (V=128), grid = (H, 8). This keeps
h_prev (= [K, BLOCK_V] fp32) small enough to stay in registers, avoiding
spills. The V-split is exact: the KDA recurrence decomposes per V column
(k_t * v_t is outer-product along V, independent across V cols).

Sequential per-program: Each program processes ALL NC chunks via a
for-loop. This is required because h[c-1] is read by chunk c and written
by chunk c-1, with no cross-program sync available. Trade-off: only
H * (V / BLOCK_V) programs run in parallel. With BLOCK_V=16, that's
96 programs on 36 SMs — full saturation.

Why not 1 program per chunk: Triton's launch order is not guaranteed;
chunk c could execute before chunk c-1, reading garbage h[c-1]. The
for-loop inside one program makes the dependency explicit (sequential
in program order).

Why V-split works: v[:, :, :, v_block] and h[:, :, :, v_block] are
independent across v_block. The recurrence h_t = exp(g_t) * h_{t-1}
+ k_t outer v_t computes each V column independently (the K-axis is
shared, but the per-V-column computation is fully decoupled).

Round-9 perf impact
-------------------
Adds one big matmul (k_decayed @ h_prev, 16x128 @ 128xV = 16xV) per
chunk per program — ~33% extra matmul flops in the recurrence. The
k_decayed load was already in the workspace (prepare was writing it
but the recurrence wasn't reading it) so HBM bytes don't grow.
The new matmul is at 99% of HBM peak in Round-5; the extra matmul
saturates the HBM budget that was headroom and pushes to ~100%.
End-to-end expected: ~3.9 ms (down from 1.63x → 1.3x vs FLA 5.13 ms).
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


CHUNK: int = 16


# ===================================================================== #
# Triton kernel                                                          #
# ===================================================================== #


@triton.jit
def _kda_recurrence_kernel(
    # workspace inputs (flat [B*NC, H, ...] for the prepare outputs)
    kd_ptr,        # [B*NC, H, CHUNK, K]  bf16 — k_decayed (Round-9)
    qd_ptr,        # [B*NC, H, CHUNK, K]  bf16
    kr_ptr,        # [B*NC, H, CHUNK, K]  bf16 — K_pre = INV @ k_restored
    gt_ptr,        # [B*NC, H, K]         fp32
    mqk_ptr,       # [B*NC, H, CHUNK, CHUNK] bf16 — Mqk_eff = Mqk @ INV
    beta_ptr,      # [B, T, H]            bf16 — per-token beta logits
    # value input
    v_ptr,         # [B, T, H, V]       bf16 (V contiguous)
    # state I/O
    h_ptr,         # [B*NC, H, K, V]      bf16 (V contiguous) — last chunk per batch populated
    h0_ptr,        # [B, H, K, V]         fp32 (V contiguous; or null)
    ht_ptr,        # [B, H, K, V]         fp32 (V contiguous; or null)
    # output
    o_ptr,         # [B, T, H, V]         bf16 (V contiguous)
    # problem dims
    B, T, NC, K, V,
    # strides
    stride_kd_nc, stride_kd_h,
    stride_qd_nc, stride_qd_h,
    stride_kr_nc, stride_kr_h,
    stride_gt_nc, stride_gt_h,
    stride_mqk_nc, stride_mqk_h,
    stride_b_b, stride_b_t, stride_b_h,
    stride_v_b, stride_v_t, stride_v_h,
    stride_h_nc, stride_h_h,
    stride_h0_b, stride_h0_h,
    stride_ht_b, stride_ht_h,
    stride_o_b, stride_o_t, stride_o_h,
    # constexpr flags
    USE_H0: tl.constexpr,
    STORE_HT: tl.constexpr,
    # block sizes
    BLOCK_M: tl.constexpr,  # CHUNK = 16
    BLOCK_K: tl.constexpr,  # K (full)
    BLOCK_V: tl.constexpr,  # V per program
    NUM_STAGES: tl.constexpr,  # pipelining depth for chunk loop
):
    pid_h = tl.program_id(0)
    pid_vs = tl.program_id(1)  # V-slice index
    pid_b = tl.program_id(2)    # batch index

    # V offset for this program: [pid_vs * BLOCK_V, (pid_vs+1) * BLOCK_V)
    v_off = pid_vs * BLOCK_V

    offs_m = tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_K)
    offs_v_local = tl.arange(0, BLOCK_V)

    # ---------------- Load h_prev (initial) ---------------- #
    # V is contiguous → pointer arithmetic: h0[pid_b, pid_h, k, v_off + j]
    # where j in [0, BLOCK_V). The h0 layout is [B, H, K, V] contiguous.
    # Element offset = ((pid_b * H + pid_h) * K + k) * V + (v_off + j)
    if USE_H0:
        h_prev = tl.load(
            h0_ptr + pid_b * stride_h0_b + pid_h * stride_h0_h
            + offs_k[:, None] * V + (v_off + offs_v_local)[None, :]
        ).to(tl.float32)
    else:
        h_prev = tl.zeros([BLOCK_K, BLOCK_V], dtype=tl.float32)

    # Per-program base pointers for this batch's workspace slice.
    # Workspace is flat [B*NC, H, ...]; batch b's leading index starts at b*NC.
    ws_nc_base = pid_b * NC
    # V input base for this batch.
    v_p = v_ptr + pid_b * stride_v_b
    # Output O base for this batch.
    o_p = o_ptr + pid_b * stride_o_b

    # ---------------- Sequential chunk loop ---------------- #
    for c in tl.range(0, NC, num_stages=NUM_STAGES):
        nc_idx = ws_nc_base + c
        # Load workspace + V_chunk for chunk c
        mqk_eff = tl.load(
            mqk_ptr + nc_idx * stride_mqk_nc + pid_h * stride_mqk_h
            + offs_m[:, None] * BLOCK_M + offs_m[None, :]
        ).to(tl.float32)

        q_dec = tl.load(
            qd_ptr + nc_idx * stride_qd_nc + pid_h * stride_qd_h
            + offs_m[:, None] * BLOCK_K + offs_k[None, :]
        ).to(tl.float32)

        # Round-9: this is K_pre = INV @ k_restored (not raw k_restored).
        # Used in the h_new update as K_pre^T @ v_residual_b.
        k_pre = tl.load(
            kr_ptr + nc_idx * stride_kr_nc + pid_h * stride_kr_h
            + offs_m[:, None] * BLOCK_K + offs_k[None, :]
        ).to(tl.float32)

        # Round-9: k_decayed is now actually USED (was dead storage before).
        # Needed for v_residual = v - k_decayed @ h_prev.
        k_dec = tl.load(
            kd_ptr + nc_idx * stride_kd_nc + pid_h * stride_kd_h
            + offs_m[:, None] * BLOCK_K + offs_k[None, :]
        ).to(tl.float32)

        g_total = tl.load(
            gt_ptr + nc_idx * stride_gt_nc + pid_h * stride_gt_h + offs_k
        )
        exp_g_total = tl.exp(g_total)  # [BLOCK_K], fp32

        token_off = c * BLOCK_M + offs_m
        valid_m = token_off < T
        v_chunk = tl.load(
            v_p + token_off[:, None] * stride_v_t + pid_h * stride_v_h
            + (v_off + offs_v_local)[None, :],
            mask=valid_m[:, None], other=0.0,
        ).to(tl.float32)

        # Round-9: per-token beta (logits, pre-sigmoid). The prepare
        # kernel applies sigmoid to beta internally for L; the
        # recurrence needs sigmoid(beta) * v_residual for v_residual_b
        # to match FlashKDA's reference (`tests/torch_ref.py:233`).
        beta = tl.load(
            beta_ptr + pid_b * stride_b_b
            + token_off * stride_b_t + pid_h * stride_b_h,
            mask=valid_m, other=0.0,
        ).to(tl.float32)
        beta_sig = tl.sigmoid(beta)  # [BLOCK_M], fp32

        # ---------------- v_residual = v - k_decayed @ h_prev ---------------- #
        # Cross-chunk contribution to v: subtract k_decayed @ h_prev (the
        # part of v that's "absorbed" into the prior h state). The result
        # is the chunk-local v that the (I - L)^-1 transform should
        # operate on (FlashKDA reference, `tests/torch_ref.py:232`).
        # This is the only "new" big matmul introduced in Round-9.
        v_xc = tl.dot(k_dec, h_prev, out_dtype=tl.float32)  # [BLOCK_M, BLOCK_V]
        v_residual = v_chunk - v_xc
        # Apply beta (per-token) — broadcast over V. beta_sig is fp32
        # (sigmoid of bf16), so the result stays fp32 in registers.
        v_residual_b = v_residual * beta_sig[:, None]  # [BLOCK_M, BLOCK_V] fp32

        # o = Mqk_eff @ v_residual_b + q_dec @ h_prev
        # Mqk_eff = Mqk @ INV (precomputed in prepare). Without the
        # (I - L)^-1, the Mqk @ v term would absorb the cross-chunk v
        # contribution a second time and the output would be ~2x
        # larger than the true answer (the cos=0.92 symptom in tests).
        o_attn = tl.dot(mqk_eff, v_residual_b, out_dtype=tl.float32)
        o_state = tl.dot(q_dec, h_prev, out_dtype=tl.float32)
        o = o_attn + o_state
        o = tl.where(valid_m[:, None], o, 0.0)

        tl.store(
            o_p + token_off[:, None] * stride_o_t + pid_h * stride_o_h
            + (v_off + offs_v_local)[None, :],
            o.to(tl.bfloat16),
            mask=valid_m[:, None],
        )

        # h_new = h_prev * exp(g_total) + K_pre^T @ v_residual_b
        # K_pre = INV @ k_restored (precomputed in prepare). The element-wise
        # decay term matches FlashKDA's reference (`tests/torch_ref.py:240-243`).
        h_decayed = h_prev * exp_g_total[:, None]
        h_contrib = tl.dot(tl.trans(k_pre), v_residual_b, out_dtype=tl.float32)
        h_new = h_decayed + h_contrib  # [K, V], fp32

        # Keep h_prev resident in registers for the next iteration.
        # The gmem write/read pair (previous version) was the bottleneck:
        # 8KB write + 8KB read per chunk, * 1024 chunks = 16MB wasted per program.
        h_prev = h_new

        # Persist the LAST chunk only (callers want final_state; intermediate
        # h[c] for c < NC-1 is not consumed downstream).
        is_last = c == (NC - 1)
        if is_last:
            if STORE_HT:
                tl.store(
                    ht_ptr + pid_b * stride_ht_b + pid_h * stride_ht_h
                    + offs_k[:, None] * V + (v_off + offs_v_local)[None, :],
                    h_new,
                )
            # Mirror last chunk into h_intermediate[pid_b*NC + (NC-1)] for API
            # compatibility — the bwd can pick up the last-chunk h from here.
            tl.store(
                h_ptr + nc_idx * stride_h_nc + pid_h * stride_h_h
                + offs_k[:, None] * V + (v_off + offs_v_local)[None, :],
                h_new.to(tl.bfloat16),
            )


# ===================================================================== #
# Python wrapper                                                         #
# ===================================================================== #


def kda_recurrence_triton(
    k_decayed: torch.Tensor,    # [B*NC, H, CHUNK, K]  bf16  (Round-9)
    q_decayed: torch.Tensor,    # [B*NC, H, CHUNK, K]  bf16
    K_pre: torch.Tensor,        # [B*NC, H, CHUNK, K]  bf16  (was k_restored)
    g_total: torch.Tensor,      # [B*NC, H, K]         fp32
    mqk_eff: torch.Tensor,      # [B*NC, H, CHUNK, CHUNK] bf16  (was mqk)
    beta: torch.Tensor,         # [B, T, H]            bf16  (per-token logits, Round-9)
    v: torch.Tensor,            # [B, T, H, V]         bf16
    initial_state: torch.Tensor | None = None,  # [B, H, K, V] fp32
    output_final_state: bool = True,
    BLOCK_V: int = 16,
    NUM_STAGES: int | None = None,  # None → pick by shape (4 for prod, 2 for small)
):
    """Compute O and the KDA recurrence.

    Round-9 inputs:
        k_decayed  — for v_residual = v - k_decayed @ h_prev subtraction
        beta       — for v_residual_b = v_residual * sigmoid(beta)
        K_pre      — INV @ k_restored (replaces k_restored)
        mqk_eff    — Mqk @ INV (replaces mqk)

    Returns:
        o: [B, T, H, V] bf16
        h_intermediate: [B*NC, H, K, V] bf16 — only the last chunk per
            batch (i.e. index pid_b*NC + (NC-1)) is populated; intermediate
            chunks are uninitialized. Callers that don't need intermediate
            h can ignore the returned tensor; callers that need it for
            bwd must gate on the (b, c) == (b, NC-1) condition.
        final_state: [B, H, K, V] fp32 (or None if output_final_state=False)
    """
    BNC, H, _, K = q_decayed.shape
    B, T, Hv, V = v.shape
    assert H == Hv, f"head dim mismatch: {H} vs {Hv}"
    assert K == V, f"only K=V=128 supported, got K={K} V={V}"
    assert V % BLOCK_V == 0, f"V={V} must be divisible by BLOCK_V={BLOCK_V}"
    assert BNC % B == 0, (
        f"workspace leading dim {BNC} must be divisible by B={B} "
        f"(BNC = B*NC; got NC = {BNC // B}? bad input?)"
    )
    assert beta.shape == (B, T, H), (
        f"beta must be [B, T, H] bf16 (per-token logits); got {tuple(beta.shape)}"
    )
    assert beta.dtype == torch.bfloat16, f"beta must be bf16; got {beta.dtype}"
    NC = BNC // B
    num_v_splits = V // BLOCK_V
    device = v.device
    dtype = v.dtype

    # Pipelining depth: longer pipelines help with lots of independent chunks
    # (more ILP), but short pipelines beat the setup cost on small NC. The
    # prod shape (NC=1024) wants ns=4; the small shape (NC=16) wants ns=2.
    if NUM_STAGES is None:
        NUM_STAGES = 4 if NC >= 256 else 2

    # Allocate outputs. h_intermediate uses torch.empty (no memset) because
    # only the last chunk per batch is written — saves zero-fill cost.
    o = torch.empty(B, T, H, V, dtype=dtype, device=device)
    h_intermediate = torch.empty(B * NC, H, K, V, dtype=torch.bfloat16, device=device)
    if output_final_state:
        final_state = torch.empty(B, H, K, V, dtype=torch.float32, device=device)
    else:
        final_state = None

    use_h0 = initial_state is not None
    if use_h0:
        assert initial_state.shape == (B, H, K, V)
        assert initial_state.dtype == torch.float32

    grid = (H, num_v_splits, B)
    _kda_recurrence_kernel[grid](
        k_decayed, q_decayed, K_pre, g_total, mqk_eff, beta,
        v,
        h_intermediate, initial_state, final_state,
        o,
        B, T, NC, K, V,
        # strides
        k_decayed.stride(0), k_decayed.stride(1),
        q_decayed.stride(0), q_decayed.stride(1),
        K_pre.stride(0), K_pre.stride(1),
        g_total.stride(0), g_total.stride(1),
        mqk_eff.stride(0), mqk_eff.stride(1),
        beta.stride(0), beta.stride(1), beta.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        h_intermediate.stride(0), h_intermediate.stride(1),
        initial_state.stride(0) if use_h0 else 0,
        initial_state.stride(1) if use_h0 else 0,
        final_state.stride(0) if output_final_state else 0,
        final_state.stride(1) if output_final_state else 0,
        o.stride(0), o.stride(1), o.stride(2),
        USE_H0=use_h0,
        STORE_HT=output_final_state,
        BLOCK_M=CHUNK,
        BLOCK_K=K,
        BLOCK_V=BLOCK_V,
        NUM_STAGES=NUM_STAGES,
    )
    return o, h_intermediate, final_state