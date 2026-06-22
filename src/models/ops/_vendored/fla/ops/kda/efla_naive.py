"""EFLA-style exact closed-form KDA per-token (forward only).

KDA's per-token recurrence (no chunkwise) is

    S_t = (I - beta_t k_t k_t^T) Diag(exp(g_t)) S_{t-1} + beta_t k_t v_t^T

where ``beta_t in R`` is a per-head scalar gate, ``g_t in R^K`` is a
per-channel log-decay, and the matrix state ``S in R^{K x V}``.

EFLA (Lei et al., 2025; arXiv 2512.12602) absorbs the diagonal decay
into a per-row normalisation and solves the resulting rank-1 ODE
exactly. With the local gamma ``gamma_t = exp(cumsum g)`` and the
closed-form coefficient

    alpha_t = -expm1(-beta_t * ||k_t||^2) / ||k_t||^2,

the per-token update becomes

    S_new = gamma_new * [ (I - alpha_t k_t k_t^T) S_old / gamma_old
                          + alpha_t k_t v_t^T ],

which differs from fla-KDA's Euler update (which uses ``beta_t`` in
place of ``alpha_t``) precisely in the higher-order terms. The gap is
small when ``beta ||k||^2 << 1`` and large otherwise — that's the
EFLA paper's central claim.

This is the per-token "exact" path. It exists for KDA only because
``beta`` is a scalar; for GDN-2 with channel-wise ``b`` and ``w`` the
closed form is not separable per-channel, and the chunkwise EFLA form
is mathematically equivalent to fla-chunk (see test_efla_gdn2_*).
"""
from __future__ import annotations

import torch


def efla_kda_per_token(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    eps: float = 1e-6,
):
    """Per-token EFLA KDA (no chunkwise, fp32)."""
    if scale is None:
        scale = q.shape[-1] ** -0.5
    B, T, H, K, V = *q.shape, v.shape[-1]
    G = beta.shape[2] // H  # fla broadcasts q,k across value-head groups
    q, k, v, g, beta = (x.transpose(1, 2).contiguous().float() for x in (q, k, v, g, beta))
    if G != 1:
        q = q.repeat_interleave(G, dim=1)
        k = k.repeat_interleave(G, dim=1)
    q = q * scale

    S = torch.zeros(B, beta.shape[1], K, V, device=v.device, dtype=torch.float32)
    if initial_state is not None:
        S = initial_state.to(torch.float32).clone()
    o = torch.zeros_like(v)
    # Per-channel cumulative gamma (running decay), starts at 1.
    gamma = torch.ones(B, beta.shape[1], K, device=v.device, dtype=torch.float32)
    for t in range(T):
        q_t = q[:, :, t]
        k_t = k[:, :, t]
        v_t = v[:, :, t]
        g_t = g[:, :, t]
        b_t = beta[:, :, t]                   # [B, HV]
        gamma_new = gamma * g_t.exp()         # running decay

        # EFLA closed-form coefficient. beta is per-head scalar here, so
        # ||k_t||^2 is a per-head scalar; broadcast over K.
        k_norm_sq = (k_t * k_t).sum(-1, keepdim=True).clamp(min=eps)   # [B,HV,1]
        # alpha broadcast: [B,HV,1] for the matrix, applied row-wise on K
        # because the (I - alpha k k^T) is the same scalar alpha per row
        # in KDA. For GDN-2 the per-channel case needs a different form.
        alpha = -torch.expm1(-b_t.unsqueeze(-1) * k_norm_sq) / k_norm_sq
        # alpha: [B,HV,1] — same scalar for all K rows of S (KDA's beta
        # is a scalar, so the closed form is also a scalar per (B,HV,T)).

        # Per-step update (matches fla KDA's structure to keep numerical
        # behaviour identical except for the alpha-vs-beta substitution):
        #   1. Apply the per-channel decay exp(g_t) to S.
        #   2. Apply the EFLA closed-form delta rule on the decayed S.
        # The EFLA form replaces the fla-Euler coefficient ``beta`` with
        # ``alpha = -expm1(-beta * ||k||^2) / ||k||^2``, which is the
        # exact continuous-time solution for the rank-1 update.
        # The two formulas coincide to first order in ``beta ||k||^2`` and
        # diverge as that product grows past ~0.1.
        S = S * g_t.exp().unsqueeze(-1)
        kS = torch.einsum("bhk,bhkv->bhv", k_t, S)         # k^T S: [B,HV,V]
        S = S - alpha.unsqueeze(-1) * k_t.unsqueeze(-1) * kS.unsqueeze(-2)
        S = S + alpha.unsqueeze(-1) * k_t.unsqueeze(-1) * v_t.unsqueeze(-2)

        o[:, :, t] = torch.einsum("bhk,bhkv->bhv", q_t, S)
        gamma = gamma_new  # noqa: F841 — kept for diagnostics only

    o = o.transpose(1, 2).contiguous().to(v.dtype)
    if not output_final_state:
        S = None
    return o, S
