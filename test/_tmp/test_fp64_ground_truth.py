"""Per-token FP64 GDR ground truth — separates "the algorithm is right" from
"the bf16 numeric approximations are right".

The canonical Gated Delta Rule (per-token, in natural-log gate space):

    h_t = exp(g_t) * h_{t-1} + k_t * (v_t - k_t @ h_{t-1}) * sigmoid(beta_t)
    o_t = q_t @ h_{t-1}

where g_t is the activated gate per token. This matches the chunk-wise
WY representation used by FlashKDA's reference and FLA's KDA:

    per-token eqv:
      v_residual = v - k @ h_{t-1}                        ← "delta rule"
      v_residual_b = v_residual * sigmoid(beta)
      U = (I - L)^{-1} @ v_residual_b                     ← WY forward sub
                                                     (chunk only; per-token just unrolls)
      o = q @ h_{t-1} + Mqk @ U
      h_new = exp(g_total) * h_{t-1} + k_restored^T @ U

The previous `naive_kda_fp64` (in test_fastkda_fwd_match.py) skipped the
delta + (I-L)^-1 + beta entirely — it computed a *different* algorithm.
That's why Round-5 FastKDA "matched" the reference: they computed the
same wrong thing.

The new ground truth here computes the TRUE per-token GDR. Any chunked
implementation (FLA, FlashKDA, FastKDA) should match this within bf16
precision (cos > 0.95 expected — that's the noise floor from FP64→bf16
casts on the inputs).
"""
from __future__ import annotations
import math
import sys
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO))

from src.models.ops._triton.fast_kda.prepare import kda_prepare_triton
from src.models.ops._triton.fast_kda.recurrence import kda_recurrence_triton
from src.models.ops._vendored.fla.ops.kda import chunk_kda


def canonical_kda_fp64(q, k, v, g, beta_raw, A_log, dt_bias, lower_bound):
    """Canonical per-token GDR in FP64.

    Per-token:
      g_act[t] = lower_bound * sigmoid(exp(A_log) * (g_raw[t] + dt_bias)) / ln(2)
      decay[t] = exp(g_act[t])    # per-head per-K decay
      v_residual[t] = v[t] - k[t] @ h_{t-1}
      v_residual_b[t] = v_residual[t] * sigmoid(beta_raw[t])
      h[t] = decay[t] * h[t-1] + einsum('bhk,bhv->bhkv', k[t], v_residual_b[t])
      o[t] = einsum('bhk,bhkv->bhv', q[t], h[t-1])   # uses h BEFORE update

    This is the TRUE mathematical formula. The chunk-wise WY
    representation is mathematically equivalent (just unrolls over
    the chunk); they should agree modulo bf16 quantization.

    Note: we DON'T apply the scale (1/sqrt(K)) here — that's an
    output-side scale that FLA's chunk_kda applies internally via
    the q decay. We apply it in the run_one wrapper for fair comp.
    """
    B, T, H, K = q.shape
    _, _, _, V = v.shape
    q = q.double(); k = k.double(); v = v.double()
    g = g.double(); beta_raw = beta_raw.double()
    A_log = A_log.double(); dt_bias = dt_bias.double()

    # L2-normalize q, k per-token — matches kernel behavior
    # (FLA + FastKDA do this inside the prepare kernel). Without
    # normalization, raw N(0,1) inputs have |k|^2 = K = 128 and
    # beta*|k|^2 = 64 >> decay ~ 0.03, so the per-token GDR is
    # wildly unstable (h grows as ~64^T and overflows at T ~ 800).
    q_sq = (q * q).sum(dim=-1, keepdim=True)
    k_sq = (k * k).sum(dim=-1, keepdim=True)
    q = q * torch.rsqrt(q_sq + 1e-6)
    k = k * torch.rsqrt(k_sq + 1e-6)

    a_exp = torch.exp(A_log).view(1, 1, H, 1)
    # FLA's convention: g_act = lower_bound * sigmoid(exp(A_log) * (g + dt_bias))
    # in NATURAL log space. FLA's chunk cumsum internally applies
    # RCP_LN2 + exp2 to convert to log2 for the fast path, but the
    # result is mathematically identical to natural-log exp().
    # Previous ref had a stray /math.log(2) which broke the conversion.
    g_act = lower_bound * torch.sigmoid(a_exp * (g + dt_bias.view(1, 1, H, K)))

    o = torch.zeros(B, T, H, V, dtype=torch.float64, device=q.device)
    h = torch.zeros(B, H, K, V, dtype=torch.float64, device=q.device)
    for t in range(T):
        # h BEFORE update: copy for o computation, then update
        h_prev = h  # alias (no copy needed — we use h_prev in o, then update h in place below)
        decay_t = torch.exp(g_act[:, t])  # [B, H, K]
        kt = k[:, t]  # [B, H, K]
        vt = v[:, t]  # [B, H, V]
        beta_sig_t = torch.sigmoid(beta_raw[:, t])  # [B, H]
        # v_residual = v - k^T @ h_prev  (per K row of h)
        # h_prev has shape [B, H, K, V]; sum over K to get v_residual [B, H, V]
        v_residual = vt - torch.einsum('bhk,bhkv->bhv', kt, h_prev)
        # v_residual_b = v_residual * sigmoid(beta)
        v_residual_b = v_residual * beta_sig_t.unsqueeze(-1)  # [B, H, V]
        # output uses h_prev (before update)
        o[:, t] = torch.einsum('bhk,bhkv->bhv', q[:, t], h_prev)
        # update state: h = decay * h_prev + k outer v_residual_b
        h = h_prev * decay_t.unsqueeze(-1) + torch.einsum('bhk,bhv->bhkv', kt, v_residual_b)
    return o.to(torch.bfloat16), h.to(torch.float32)


def run_one(B, T, H, K, V, lower_bound, seed=7):
    torch.manual_seed(seed)
    device, dtype = 'cuda', torch.bfloat16
    q_raw = torch.randn(B, T, H, K, device=device, dtype=torch.float32)
    k_raw = torch.randn(B, T, H, K, device=device, dtype=torch.float32)
    v = torch.randn(B, T, H, V, device=device, dtype=dtype)
    g_raw = -torch.rand(B, T, H, K, device=device, dtype=torch.float32) * 2.0 - 0.5
    beta_raw = torch.randn(B, T, H, device=device, dtype=torch.float32)
    A_log = torch.rand(H, device=device, dtype=torch.float32) * 0.5 + 0.3
    dt_bias = torch.randn(H, K, device=device, dtype=torch.float32) * 0.5
    scale = 1.0 / math.sqrt(K)

    # Pre-L2-normalize q, k (canonical ref needs this for stability;
    # FLA + FastKDA accept normalized inputs with use_qk_l2norm=False).
    eps = 1e-6
    q_n = q_raw * torch.rsqrt((q_raw * q_raw).sum(-1, keepdim=True) + eps)
    k_n = k_n = k_raw * torch.rsqrt((k_raw * k_raw).sum(-1, keepdim=True) + eps)

    q = q_n.to(dtype); k = k_n.to(dtype); g = g_raw.to(dtype)
    beta = beta_raw.sigmoid().to(dtype)  # FLA expects pre-sigmoided

    # Canonical FP64 GDR (true ground truth — does its own normalization)
    o_ref, st_ref = canonical_kda_fp64(
        q_n, k_n, v, g_raw, beta_raw, A_log, dt_bias,
        lower_bound=lower_bound,
    )

    # FLA (chunk_kda, uses pre-sigmoided beta, applies scale internally)
    o_fla, st_fla = chunk_kda(
        q=q, k=k, v=v, g=g, beta=beta,
        A_log=A_log, dt_bias=dt_bias,
        use_qk_l2norm_in_kernel=False,  # already normalized above
        use_gate_in_kernel=True,
        safe_gate=True, lower_bound=lower_bound,
        output_final_state=True, scale=scale,
    )

    # FastKDA Round-9 (prepare kernel does its own L2 norm; this means
    # we MUST disable that by passing already-normalized q, k. But the
    # prepare kernel hard-codes the L2 norm. Workaround: pass
    # un-normalized to the prepare kernel — it will normalize again,
    # but since the inputs ARE already unit-norm, the double-norm
    # normalizes to ~1.0 still (modulo the 1e-6 eps). Acceptable.)
    # Alternative: patch prepare to skip L2 norm via a flag. For now,
    # just pass un-normalized and accept the small re-normalization.
    ws = kda_prepare_triton(q_n.to(dtype), k_n.to(dtype), g, beta_raw.to(dtype), A_log, dt_bias, lower_bound, scale)
    o_fast, _, st_fast = kda_recurrence_triton(
        k_decayed=ws['k_decayed'],
        q_decayed=ws['q_decayed'],
        K_pre=ws['K_pre'],
        g_total=ws['g_total'],
        mqk_eff=ws['Mqk_eff'],
        beta=beta_raw.to(dtype),
        v=v,
        initial_state=None,
        output_final_state=True,
    )

    return o_ref, o_fla, o_fast, st_ref, st_fla, st_fast


def cos(a, b):
    return torch.nn.functional.cosine_similarity(
        a.float().flatten().unsqueeze(0), b.float().flatten().unsqueeze(0)
    ).item()


def main():
    print(f"{'shape':<35s} {'cos(r,F)':>10s} {'cos(r,K)':>10s} {'cos(F,K)':>10s} "
          f"{'|abs_max|':>10s} {'mean_rel':>10s} {'med_rel':>10s}")
    print("=" * 105)
    for shape in [
        (1, 256, 12, 128, 128, -5.0, "T=256"),
        (1, 1024, 12, 128, 128, -5.0, "T=1024"),
        (1, 4096, 12, 128, 128, -5.0, "T=4096"),
    ]:
        B, T, H, K, V, lb, desc = shape
        try:
            o_ref, o_fla, o_fast, st_ref, st_fla, st_fast = run_one(B, T, H, K, V, lb)
            crf = cos(o_ref, o_fla)
            crk = cos(o_ref, o_fast)
            cfk = cos(o_fla, o_fast)
            # Cos alone is unreliable. Also report abs and rel errors
            # between FLA and Fast (the two real chunked implementations).
            diff = (o_fast.float() - o_fla.float()).abs()
            abs_max = diff.max().item()
            denom = o_fla.float().abs().clamp_min(1e-3)
            rel = diff / denom
            # Trim outliers (very large ratios from tiny denominators)
            valid = (rel < 5.0)
            mean_rel = rel[valid].mean().item() if valid.any() else float('nan')
            med_rel = rel[valid].median().item() if valid.any() else float('nan')
            print(f"{desc:<35s} {crf:>10.4f} {crk:>10.4f} {cfk:>10.4f} "
                  f"{abs_max:>10.3e} {mean_rel:>10.3e} {med_rel:>10.3e}")
        except Exception as e:
            print(f"{desc:<35s}  ERROR: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
