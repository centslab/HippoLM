"""fp32 k_inv experiment: does keeping k_inv in fp32 enable CHUNK=32?

The bf16 k_inv in prepare.py:135 overflows at CHUNK=32 (exp(-103) = 2.6e44
> bf16 max 3.4e38). Test if keeping k_inv in fp32 (in registers, no gmem
hit) allows CHUNK=32 to produce finite output.

Also tests CHUNK=16 (regression check) and times both.

Run: python3 bench/fast_kda_fp32_kinv.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
import triton
import triton.language as tl

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))


LN2 = 0.6931471805599453


# =====================================================================
# Modified prepare kernel: k_inv in fp32, k_decayed/q_decayed stay bf16
# =====================================================================
@triton.jit
def _kda_prepare_kernel_fp32_kinv(
    q_ptr, k_ptr, g_ptr, beta_ptr,
    A_log_ptr, dt_bias_ptr,
    kd_ptr, qd_ptr, kr_ptr, gt_ptr, inv_ptr, mqk_ptr,
    H, K, T, NC,
    stride_q_t, stride_q_h,
    stride_k_t, stride_k_h,
    stride_g_t, stride_g_h,
    stride_b_t, stride_b_h,
    stride_kd_nc, stride_kd_h,
    stride_qd_nc, stride_qd_h,
    stride_kr_nc, stride_kr_h,
    stride_gt_nc, stride_gt_h,
    stride_inv_nc, stride_inv_h,
    stride_mqk_nc, stride_mqk_h,
    scale,
    a_log_exp_per_head_ptr,
    gate_scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_nc = tl.program_id(0)
    pid_h = tl.program_id(1)

    offs_n = tl.arange(0, BLOCK_N)
    dt_bias = tl.load(dt_bias_ptr + pid_h * BLOCK_N + offs_n).to(tl.float32)

    offs_m = pid_nc * BLOCK_M + tl.arange(0, BLOCK_M)
    seq_len = tl.minimum((pid_nc + 1) * BLOCK_M, T)
    valid_m = offs_m < seq_len

    q = tl.load(
        q_ptr + offs_m[:, None] * stride_q_t + pid_h * stride_q_h + offs_n[None, :],
        mask=valid_m[:, None], other=0.0,
    ).to(tl.float32)
    k = tl.load(
        k_ptr + offs_m[:, None] * stride_k_t + pid_h * stride_k_h + offs_n[None, :],
        mask=valid_m[:, None], other=0.0,
    ).to(tl.float32)
    g_raw = tl.load(
        g_ptr + offs_m[:, None] * stride_g_t + pid_h * stride_g_h + offs_n[None, :],
        mask=valid_m[:, None], other=0.0,
    ).to(tl.float32)
    beta = tl.load(
        beta_ptr + offs_m * stride_b_t + pid_h * stride_b_h,
        mask=valid_m, other=0.0,
    ).to(tl.float32)

    q_sq = tl.sum(q * q, axis=1)
    k_sq = tl.sum(k * k, axis=1)
    q = q * tl.rsqrt(q_sq[:, None] + 1e-6)
    k = k * tl.rsqrt(k_sq[:, None] + 1e-6)

    a_log_exp = tl.load(a_log_exp_per_head_ptr + pid_h)
    g_val = a_log_exp * (g_raw + dt_bias[None, :])
    g_val = tl.where(valid_m[:, None], g_val, 0.0)
    g_val = gate_scale * tl.sigmoid(g_val)

    g_cs = tl.cumsum(g_val, axis=0)
    last_row_mask = tl.arange(0, BLOCK_M) == (BLOCK_M - 1)
    g_total = tl.sum(tl.where(last_row_mask[:, None], g_cs, 0.0), axis=0)
    exp_g_total = tl.exp(g_total)

    exp_g = tl.exp(g_cs)            # fp32, [BLOCK_M, BLOCK_N]
    exp_neg_g = tl.exp(-g_cs)       # fp32, [BLOCK_M, BLOCK_N]
    # *** KEY CHANGE: everything in fp32 for the bmm (Triton requires matching dtypes) ***
    q_decayed_fp32 = q * exp_g * scale
    k_decayed_fp32 = k * exp_g
    k_inv_fp32 = k * exp_neg_g
    k_restored_bf16 = (k * exp_neg_g * exp_g_total[None, :]).to(tl.bfloat16)

    # Bmm: full fp32 (Triton uses TF32 on tensor cores)
    L = tl.dot(k_decayed_fp32, tl.trans(k_inv_fp32), out_dtype=tl.float32)
    Mqk = tl.dot(q_decayed_fp32, tl.trans(k_inv_fp32), out_dtype=tl.float32)

    offs_m_local = tl.arange(0, BLOCK_M)
    tril_mask = offs_m_local[:, None] > offs_m_local[None, :]
    beta_sig = tl.sigmoid(beta).to(tl.float16)
    L = tl.where(tril_mask, (L * beta_sig[None, :]).to(tl.bfloat16),
                 tl.zeros((), dtype=tl.bfloat16))
    Mqk_mask = offs_m_local[:, None] >= offs_m_local[None, :]
    Mqk = tl.where(Mqk_mask, Mqk.to(tl.bfloat16), tl.zeros((), dtype=tl.bfloat16))

    identity = tl.where(
        offs_m_local[:, None] == offs_m_local[None, :],
        tl.full((), 1.0, dtype=tl.bfloat16),
        tl.zeros((), dtype=tl.bfloat16),
    )
    INV = identity - L

    INV_f32 = INV.to(tl.float32)
    L_f32 = L.to(tl.float32)
    I_plus_L = identity.to(tl.float32) + L_f32
    for _ in range(4):
        INV_f32 = tl.dot(INV_f32, I_plus_L, out_dtype=tl.float32)
    INV = INV_f32.to(tl.bfloat16)

    kd_off = pid_nc * stride_kd_nc + pid_h * stride_kd_h
    qd_off = pid_nc * stride_qd_nc + pid_h * stride_qd_h
    kr_off = pid_nc * stride_kr_nc + pid_h * stride_kr_h
    store_2d = offs_m_local[:, None] * BLOCK_N + offs_n[None, :]
    tl.store(kd_ptr + kd_off + store_2d, k_decayed_fp32.to(tl.bfloat16))
    tl.store(qd_ptr + qd_off + store_2d, q_decayed_fp32.to(tl.bfloat16))
    tl.store(kr_ptr + kr_off + store_2d, k_restored_bf16)

    gt_off = pid_nc * stride_gt_nc + pid_h * stride_gt_h
    tl.store(gt_ptr + gt_off + offs_n, g_total)

    store_2d_small = offs_m_local[:, None] * BLOCK_M + offs_m_local[None, :]
    inv_off = pid_nc * stride_inv_nc + pid_h * stride_inv_h
    mqk_off = pid_nc * stride_mqk_nc + pid_h * stride_mqk_h
    tl.store(inv_ptr + inv_off + store_2d_small, INV)
    tl.store(mqk_ptr + mqk_off + store_2d_small, Mqk)


def prepare_fp32_kinv(q, k, g, beta, A_log, dt_bias, lower_bound, scale, CHUNK):
    B, T, H, K = q.shape
    NC = (T + CHUNK - 1) // CHUNK
    device = q.device
    kd = torch.empty(NC, H, CHUNK, K, dtype=torch.bfloat16, device=device)
    qd = torch.empty(NC, H, CHUNK, K, dtype=torch.bfloat16, device=device)
    kr = torch.empty(NC, H, CHUNK, K, dtype=torch.bfloat16, device=device)
    gt = torch.empty(NC, H, K, dtype=torch.float32, device=device)
    inv = torch.empty(NC, H, CHUNK, CHUNK, dtype=torch.bfloat16, device=device)
    mqk = torch.empty(NC, H, CHUNK, CHUNK, dtype=torch.bfloat16, device=device)
    a_log_exp = torch.exp(A_log.float()).contiguous()
    grid = (NC, H)
    _kda_prepare_kernel_fp32_kinv[grid](
        q, k, g, beta, A_log, dt_bias,
        kd, qd, kr, gt, inv, mqk,
        H, K, T, NC,
        q.stride(1), q.stride(2),
        k.stride(1), k.stride(2),
        g.stride(1), g.stride(2),
        beta.stride(1), beta.stride(2),
        kd.stride(0), kd.stride(1),
        qd.stride(0), qd.stride(1),
        kr.stride(0), kr.stride(1),
        gt.stride(0), gt.stride(1),
        inv.stride(0), inv.stride(1),
        mqk.stride(0), mqk.stride(1),
        float(scale), a_log_exp, float(lower_bound * LN2),
        BLOCK_M=CHUNK, BLOCK_N=K,
    )
    return {"q_decayed": qd, "k_restored": kr, "g_total": gt, "Mqk": mqk}


# =====================================================================
# Original (bf16 k_inv) for comparison
# =====================================================================
from src.models.ops._triton.fast_kda.prepare import (
    _kda_prepare_kernel, kda_prepare_triton,
)
from src.models.ops._triton.fast_kda.recurrence import (
    _kda_recurrence_kernel, kda_recurrence_triton,
)


def prepare_bf16_kinv(q, k, g, beta, A_log, dt_bias, lower_bound, scale, CHUNK):
    """Original kernel with bf16 k_inv (for direct comparison)."""
    B, T, H, K = q.shape
    NC = (T + CHUNK - 1) // CHUNK
    device = q.device
    kd = torch.empty(NC, H, CHUNK, K, dtype=torch.bfloat16, device=device)
    qd = torch.empty(NC, H, CHUNK, K, dtype=torch.bfloat16, device=device)
    kr = torch.empty(NC, H, CHUNK, K, dtype=torch.bfloat16, device=device)
    gt = torch.empty(NC, H, K, dtype=torch.float32, device=device)
    inv = torch.empty(NC, H, CHUNK, CHUNK, dtype=torch.bfloat16, device=device)
    mqk = torch.empty(NC, H, CHUNK, CHUNK, dtype=torch.bfloat16, device=device)
    a_log_exp = torch.exp(A_log.float()).contiguous()
    grid = (NC, H)
    _kda_prepare_kernel[grid](
        q, k, g, beta, A_log, dt_bias,
        kd, qd, kr, gt, inv, mqk,
        H, K, T, NC,
        q.stride(1), q.stride(2),
        k.stride(1), k.stride(2),
        g.stride(1), g.stride(2),
        beta.stride(1), beta.stride(2),
        kd.stride(0), kd.stride(1),
        qd.stride(0), qd.stride(1),
        kr.stride(0), kr.stride(1),
        gt.stride(0), gt.stride(1),
        inv.stride(0), inv.stride(1),
        mqk.stride(0), mqk.stride(1),
        float(scale), a_log_exp, float(lower_bound * LN2),
        BLOCK_M=CHUNK, BLOCK_N=K,
    )
    return {"q_decayed": qd, "k_restored": kr, "g_total": gt, "Mqk": mqk}


def recurrence(ws, v, BLOCK_V, NUM_STAGES, CHUNK):
    qd, kr, gt, mqk = ws["q_decayed"], ws["k_restored"], ws["g_total"], ws["Mqk"]
    NC, H, _, K = qd.shape
    B, T, Hv, V = v.shape
    num_v_splits = V // BLOCK_V
    o = torch.empty(B, T, H, V, dtype=v.dtype, device=v.device)
    h_int = torch.empty(NC, H, K, V, dtype=torch.bfloat16, device=v.device)
    final = torch.empty(B, H, K, V, dtype=torch.float32, device=v.device)
    grid = (H, num_v_splits)
    _kda_recurrence_kernel[grid](
        qd, kr, gt, mqk, v,
        h_int, None, final, o,
        B, T, NC, K, V,
        qd.stride(0), qd.stride(1),
        kr.stride(0), kr.stride(1),
        gt.stride(0), gt.stride(1),
        mqk.stride(0), mqk.stride(1),
        v.stride(1), v.stride(2),
        h_int.stride(0), h_int.stride(1),
        0, 0, final.stride(0), final.stride(1),
        o.stride(1), o.stride(2),
        USE_H0=False, STORE_HT=True,
        BLOCK_M=CHUNK, BLOCK_K=K, BLOCK_V=BLOCK_V, NUM_STAGES=NUM_STAGES,
    )
    return o, final


class CudaTimer:
    def __init__(self):
        self._start = None
        self._end = None
    def __enter__(self):
        torch.cuda.synchronize()
        self._start = torch.cuda.Event(enable_timing=True)
        self._end = torch.cuda.Event(enable_timing=True)
        self._start.record()
        return self
    def __exit__(self, *exc):
        self._end.record()
        torch.cuda.synchronize()
        self.ms = self._start.elapsed_time(self._end)


def _bench(fn, iters=20):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        with CudaTimer() as t:
            fn()
        times.append(t.ms)
    times.sort()
    return times[len(times) // 2]


def _make_inputs(B, T, H, K, V, device, dtype, seed):
    torch.manual_seed(seed)
    q = torch.randn(B, T, H, K, device=device, dtype=torch.float32)
    k = torch.randn(B, T, H, K, device=device, dtype=torch.float32)
    q = (q / q.norm(dim=-1, keepdim=True)).to(dtype)
    k = (k / k.norm(dim=-1, keepdim=True)).to(dtype)
    g_raw = (torch.randn(B, T, H, K, device=device, dtype=torch.float32) - 0.5)
    g_raw = g_raw.to(dtype)
    beta_logits = torch.randn(B, T, H, device=device, dtype=torch.float32) * 2.0
    beta_logits = beta_logits.to(dtype)
    A_log = torch.rand(H, device=device, dtype=torch.float32) * 0.5 + 0.3
    dt_bias = torch.randn(H, K, device=device, dtype=torch.float32) * 0.5
    v = torch.randn(B, T, H, V, device=device, dtype=dtype)
    return q, k, g_raw, beta_logits, A_log, dt_bias, v


def main():
    device = torch.device("cuda:0")
    dtype = torch.bfloat16
    B, T, H, K, V = 1, 16384, 12, 128, 128
    scale = 1.0 / math.sqrt(K)

    q, k, g_raw, beta, A_log, dt_bias, v = _make_inputs(B, T, H, K, V, device, dtype, seed=0)

    print(f"=== fp32 k_inv experiment (B={B} T={T} H={H} K=V={V}) ===\n")

    print("[1] Correctness: does CHUNK=32 produce finite output?")
    for CHUNK in [16, 32]:
        for label, prep_fn in [("bf16 k_inv", prepare_bf16_kinv), ("fp32 k_inv", prepare_fp32_kinv)]:
            try:
                ws = prep_fn(q, k, g_raw, beta, A_log, dt_bias, -5.0, scale, CHUNK)
                mqk_finite = torch.isfinite(ws["Mqk"]).all().item()
                qd_finite = torch.isfinite(ws["q_decayed"]).all().item()
                gt_max = ws["g_total"].abs().max().item()
                print(f"  CHUNK={CHUNK:>3}  {label:<12}  Mqk finite={mqk_finite!s:<5}  "
                      f"q_decayed finite={qd_finite!s:<5}  g_total max|.|={gt_max:.1f}")
            except Exception as e:
                print(f"  CHUNK={CHUNK:>3}  {label:<12}  FAIL: {str(e)[:60]}")

    print("\n[2] Prepare timing")
    for CHUNK in [16, 32]:
        for label, prep_fn in [("bf16", prepare_bf16_kinv), ("fp32", prepare_fp32_kinv)]:
            try:
                def _run(p=prep_fn, c=CHUNK):
                    p(q, k, g_raw, beta, A_log, dt_bias, -5.0, scale, c)
                ms = _bench(_run, 30)
                print(f"  CHUNK={CHUNK:>3}  k_inv={label:<5}  prepare = {ms*1000:>7.1f} us")
            except Exception as e:
                print(f"  CHUNK={CHUNK:>3}  k_inv={label:<5}  FAIL: {str(e)[:60]}")

    print("\n[3] End-to-end fwd (prepare + recurrence)")
    for CHUNK, BLOCK_V, NS, label in [
        (16, 16, 4, "C=16 V=16 ns=4 (current, bf16)"),
        (32, 16, 2, "C=32 V=16 ns=2 (NEW, fp32 k_inv)"),
    ]:
        if CHUNK == 16:
            prep_fn = prepare_bf16_kinv
        else:
            prep_fn = prepare_fp32_kinv
        try:
            def _run(p=prep_fn, c=CHUNK, bv=BLOCK_V, ns=NS):
                ws = p(q, k, g_raw, beta, A_log, dt_bias, -5.0, scale, c)
                recurrence(ws, v, bv, ns, c)
            ms = _bench(_run, 30)
            print(f"  {label:<40}  fwd = {ms*1000:>7.1f} us  ({ms:.3f} ms)")
        except Exception as e:
            print(f"  {label:<40}  FAIL: {str(e)[:60]}")


if __name__ == "__main__":
    main()
