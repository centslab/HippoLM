"""Fused Triton KDA recurrence kernel.

Implements FlashKDA's Kernel 2 algorithm in pure Triton (no CUTLASS,
no external deps).

Critical design point: each program processes ALL chunks for ONE head
and ONE V-slice sequentially via a Python for-loop inside the kernel.
This guarantees that h[c-1] is written before h[c] reads it (no
cross-program race).

Per-program algorithm:
  for c in 0..NC-1:
      h_prev = h[c-1] from gmem (or initial_state for c=0)
      o_chunk = Mqk[c] @ V_chunk + q_decayed[c] @ h_prev
      h_new   = h_prev * exp(g_total[c]) + k_restored[c] @ V_chunk
      store o[c] and h[c]

Workspace inputs (produced by prepare.py):
  q_decayed    [NC, H, CHUNK, K]   bf16
  k_restored   [NC, H, CHUNK, K]   bf16
  g_total      [NC, H, K]          fp32
  Mqk          [NC, H, CHUNK, CHUNK] bf16

State I/O:
  h_intermediate [NC, H, K, V]     bf16  (state at END of each chunk)
  initial_state  [B, H, K, V]      fp32  (state at start of first chunk)
  final_state    [B, H, K, V]      fp32  (state at end of last chunk)

Outputs:
  O            [B, T, H, V]       bf16

Design notes
------------
V-split: Each program processes a V-block of BLOCK_V columns. Grid is
(H, V // BLOCK_V). With BLOCK_V=64 (V=128), grid = (H, 2). This keeps
h_prev (= [K, BLOCK_V] fp32) small enough to stay in registers, avoiding
spills. The V-split is exact: the KDA recurrence decomposes per V column
(k_t * v_t is outer-product along V, independent across V cols).

Sequential per-program: Each program processes ALL NC chunks via a
for-loop. This is required because h[c-1] is read by chunk c and written
by chunk c-1, with no cross-program sync available. Trade-off: only
H * (V / BLOCK_V) programs run in parallel. With BLOCK_V=64, that's
24 programs on 36 SMs — ~67% SM occupancy at H=12.

Why not 1 program per chunk: Triton's launch order is not guaranteed;
chunk c could execute before chunk c-1, reading garbage h[c-1]. The
for-loop inside one program makes the dependency explicit (sequential
in program order).

Why V-split works: v[:, :, :, v_block] and h[:, :, :, v_block] are
independent across v_block. The recurrence h_t = exp(g_t) * h_{t-1}
+ k_t outer v_t computes each V column independently (the K-axis is
shared, but the per-V-column computation is fully decoupled).
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
    # workspace inputs
    qd_ptr,        # [NC, H, CHUNK, K]  bf16
    kr_ptr,        # [NC, H, CHUNK, K]  bf16
    gt_ptr,        # [NC, H, K]         fp32
    mqk_ptr,       # [NC, H, CHUNK, CHUNK] bf16
    # value input
    v_ptr,         # [B, T, H, V]       bf16 (V contiguous)
    # state I/O
    h_ptr,         # [NC, H, K, V]      bf16 (V contiguous)
    h0_ptr,        # [B, H, K, V]       fp32 (V contiguous; or null)
    ht_ptr,        # [B, H, K, V]       fp32 (V contiguous; or null)
    # output
    o_ptr,         # [B, T, H, V]       bf16 (V contiguous)
    # problem dims
    B, T, NC, K, V,
    # strides
    stride_qd_nc, stride_qd_h,
    stride_kr_nc, stride_kr_h,
    stride_gt_nc, stride_gt_h,
    stride_mqk_nc, stride_mqk_h,
    stride_v_t, stride_v_h,
    stride_h_nc, stride_h_h,
    stride_h0_b, stride_h0_h,
    stride_ht_b, stride_ht_h,
    stride_o_t, stride_o_h,
    # constexpr flags
    USE_H0: tl.constexpr,
    STORE_HT: tl.constexpr,
    # block sizes
    BLOCK_M: tl.constexpr,  # CHUNK = 16
    BLOCK_K: tl.constexpr,  # K (full)
    BLOCK_V: tl.constexpr,  # V per program
):
    pid_h = tl.program_id(0)
    pid_vs = tl.program_id(1)  # V-slice index
    pid_b = 0  # B=1 in prod

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

    # ---------------- Sequential chunk loop ---------------- #
    for c in tl.range(0, NC, num_stages=3):
        # Load workspace + V_chunk for chunk c
        mqk = tl.load(
            mqk_ptr + c * stride_mqk_nc + pid_h * stride_mqk_h
            + offs_m[:, None] * BLOCK_M + offs_m[None, :]
        ).to(tl.float32)

        q_dec = tl.load(
            qd_ptr + c * stride_qd_nc + pid_h * stride_qd_h
            + offs_m[:, None] * BLOCK_K + offs_k[None, :]
        ).to(tl.float32)

        k_res = tl.load(
            kr_ptr + c * stride_kr_nc + pid_h * stride_kr_h
            + offs_m[:, None] * BLOCK_K + offs_k[None, :]
        ).to(tl.float32)

        g_total = tl.load(
            gt_ptr + c * stride_gt_nc + pid_h * stride_gt_h + offs_k
        )
        exp_g_total = tl.exp(g_total)  # [BLOCK_K], fp32

        token_off = c * BLOCK_M + offs_m
        valid_m = token_off < T
        v_chunk = tl.load(
            v_ptr + token_off[:, None] * stride_v_t + pid_h * stride_v_h
            + (v_off + offs_v_local)[None, :],
            mask=valid_m[:, None], other=0.0,
        ).to(tl.float32)

        # o = Mqk @ V + q_dec @ h_prev   (scale baked into both via prepare)
        o_attn = tl.dot(mqk, v_chunk)
        o_state = tl.dot(q_dec, h_prev)
        o = o_attn + o_state
        o = tl.where(valid_m[:, None], o, 0.0)

        tl.store(
            o_ptr + token_off[:, None] * stride_o_t + pid_h * stride_o_h
            + (v_off + offs_v_local)[None, :],
            o.to(tl.bfloat16),
            mask=valid_m[:, None],
        )

        # h_new = h_prev * exp(g_total) + k_res^T @ V
        h_decayed = h_prev * exp_g_total[:, None]
        h_contrib = tl.dot(tl.trans(k_res), v_chunk)
        h_new = h_decayed + h_contrib  # [K, V], fp32

        # Store h[c] (intermediate for next chunk, or final state)
        is_last = c == (NC - 1)
        if is_last and STORE_HT:
            tl.store(
                ht_ptr + pid_b * stride_ht_b + pid_h * stride_ht_h
                + offs_k[:, None] * V + (v_off + offs_v_local)[None, :],
                h_new,
            )
        else:
            tl.store(
                h_ptr + c * stride_h_nc + pid_h * stride_h_h
                + offs_k[:, None] * V + (v_off + offs_v_local)[None, :],
                h_new.to(tl.bfloat16),
            )

        # Update h_prev for next iteration
        if not is_last:
            h_prev = tl.load(
                h_ptr + c * stride_h_nc + pid_h * stride_h_h
                + offs_k[:, None] * V + (v_off + offs_v_local)[None, :]
            ).to(tl.float32)


# ===================================================================== #
# Python wrapper                                                         #
# ===================================================================== #


def kda_recurrence_triton(
    q_decayed: torch.Tensor,    # [NC, H, CHUNK, K]  bf16
    k_restored: torch.Tensor,   # [NC, H, CHUNK, K]  bf16
    g_total: torch.Tensor,      # [NC, H, K]         fp32
    mqk: torch.Tensor,          # [NC, H, CHUNK, CHUNK] bf16
    v: torch.Tensor,            # [B, T, H, V]       bf16
    initial_state: torch.Tensor | None = None,  # [B, H, K, V] fp32
    output_final_state: bool = True,
    BLOCK_V: int = 16,
):
    """Compute O and per-chunk state h via the KDA recurrence.

    Returns:
        o: [B, T, H, V] bf16
        h_intermediate: [NC, H, K, V] bf16 (state at end of each chunk)
        final_state: [B, H, K, V] fp32 (or None if output_final_state=False)
    """
    NC, H, _, K = q_decayed.shape
    B, T, Hv, V = v.shape
    assert H == Hv, f"head dim mismatch: {H} vs {Hv}"
    assert K == V, f"only K=V=128 supported, got K={K} V={V}"
    assert V % BLOCK_V == 0, f"V={V} must be divisible by BLOCK_V={BLOCK_V}"
    num_v_splits = V // BLOCK_V
    device = v.device
    dtype = v.dtype

    # Allocate outputs
    o = torch.empty(B, T, H, V, dtype=dtype, device=device)
    h_intermediate = torch.zeros(NC, H, K, V, dtype=torch.bfloat16, device=device)
    if output_final_state:
        final_state = torch.empty(B, H, K, V, dtype=torch.float32, device=device)
    else:
        final_state = None

    use_h0 = initial_state is not None
    if use_h0:
        assert initial_state.shape == (B, H, K, V)
        assert initial_state.dtype == torch.float32

    grid = (H, num_v_splits)
    _kda_recurrence_kernel[grid](
        q_decayed, k_restored, g_total, mqk,
        v,
        h_intermediate, initial_state, final_state,
        o,
        B, T, NC, K, V,
        # strides
        q_decayed.stride(0), q_decayed.stride(1),
        k_restored.stride(0), k_restored.stride(1),
        g_total.stride(0), g_total.stride(1),
        mqk.stride(0), mqk.stride(1),
        v.stride(1), v.stride(2),
        h_intermediate.stride(0), h_intermediate.stride(1),
        initial_state.stride(0) if use_h0 else 0,
        initial_state.stride(1) if use_h0 else 0,
        final_state.stride(0) if output_final_state else 0,
        final_state.stride(1) if output_final_state else 0,
        o.stride(1), o.stride(2),
        USE_H0=use_h0,
        STORE_HT=output_final_state,
        BLOCK_M=CHUNK,
        BLOCK_K=K,
        BLOCK_V=BLOCK_V,
    )
    return o, h_intermediate, final_state