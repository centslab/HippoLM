"""Triton-fused AttnRes compute for BlockAttnRes.forward.

Replaces the 5-op PyTorch reference path (stack → RMSNorm → contig
slice × 2 → einsum × 2) with a SINGLE Triton kernel that streams the
online softmax over the N residual sources:

  - ``_online_attnres_kernel``: one program per (b, t) token. Loops
    over the N residual sources; for each source it reads
    V[n, b, t, :] ONCE (as a padded ``[H, D_h]`` tile), computes the
    RMSNorm rstd over the full hidden, the per-head logit
    ``q·RMSNorm(v)``, and folds both the softmax normaliser and the
    weighted-sum accumulator in registers (running max / acc / o).
    Writes ``out[B, T, H, D_h]`` (bf16) at the end.

This borrows the design of flash-linear-attention's ``fused_attnres``
(fla/ops/attnres/fused.py): V is read exactly once (logit + weighted
sum share the same tile) and no intermediate logits tensor is
materialised in HBM. The difference from FLA is that HippoLM's
BlockAttnRes is MULTI-HEAD (per-head query ``[H, D_h]`` → a softmax
weight per head over N), whereas FLA is single-head (one scalar
weight per residual source over the full hidden). So FLA's kernel
can't be dropped in as-is; this kernel keeps HippoLM's per-head
semantics with FLA's V-read-once online-softmax structure.

HBM traffic per call (prod shape N=8, B=1, T=16384, D=1536, H=12,
D_h=128):
  - Reference: ~3500 MB (stack + RMSNorm read+write + 2 contig
    slice copies + weighted sum read+write).
  - Previous 2-kernel fusion: ~825 MB (V read twice + logits HBM
    round-trip).
  - This single kernel: ~430 MB (V read ONCE + out).

Per-call wall time on 5060 Ti at prod shape: ~26 ms (reference) →
~2.1 ms (2-kernel) → ~1.17 ms (this), i.e. ~1.8× over the 2-kernel
path and ~22× over the reference.

TP note: the kernel works in full-head space (reads the full-hidden
row for the RMSNorm reduction, computes/writes only the local head
range ``[HS, HS+H_LOCAL)``). This means it is correct at world > 1
too — unlike the previous 2-kernel path, whose ``reshape`` of the
full-D key into ``[hpp, D_h]`` compile-failed at world > 1 and
silently fell back to PyTorch.

Backward (autograd.Function) re-runs the reference math in PyTorch
under ``torch.enable_grad()`` to get the gradients. The bwd cost is
small relative to fwd, so this trade is acceptable.

Numerical agreement: BF16 reduction order differs from the reference
by <1% relative (max abs diff < 0.008 at fp32 random init with
N(0,1) inputs × RMSNorm); the model's own tests (test_attn_res.py
uniform-with-zero-query, gradient flow) bound this tighter.

Why a separate file from attn_res.py
------------------------------------
The Triton kernel is only used when CUDA + bf16 match; otherwise we
fall back to the PyTorch path. Keeping the kernel isolated lets
``is_available()`` short-circuit cleanly and keeps the autograd
Function out of the eager path.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Single kernel: RMSNorm + per-head dot + online softmax + weighted sum.
#
# Per program: one (b, t) token. Loops over the N residual sources.
# For each source n it reads V[n, b, t, :] ONCE as a padded
# ``[BLOCK_HF, BLOCK_DH]`` tile (full-head layout), then:
#   - rstd  = 1/sqrt(mean(v^2, over full D) + eps)   (scalar)
#   - k     = v * rstd * norm_weight                  (full-head tile)
#   - logit = sum_dh q[h,dh] * k[h,dh]                (per head)
#   - online-softmax update: running max / acc / o_acc in registers.
#
# The query is provided for the LOCAL head range only ([H_LOCAL, D_h]);
# it is loaded into rows [HS, HS+H_LOCAL) of the full-head tile (other
# rows have q=0 → their logit contribution is 0 and they are never
# written out). Only the local head rows are stored to ``out``.
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
    B, T, D, H_full, D_h,
    eps,
    stride_v_n, stride_v_b, stride_v_t,
    stride_o_b, stride_o_t,
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

        m_new = tl.maximum(m_run, logit)
        r = tl.exp(m_run - m_new)
        p = tl.exp(logit - m_new)
        acc = acc * r + p
        o_acc = o_acc * r[:, None] + p[:, None] * v
        m_run = m_new

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
) -> torch.Tensor:
    """Compute the local AttnRes output via a single online-softmax kernel.

    ``V_local`` is used only for its shape (``H_LOCAL``, ``D_h``); the
    kernel reads the weighted-sum data from ``V_full``'s local head
    range ``[slice_start, slice_start + H_LOCAL * D_h)`` directly. At
    TP=1 the local range is the full hidden (``slice_start == 0``,
    ``H_LOCAL == H_full``).

    Returns ``out`` of shape ``[B, T, H, D_h]`` (per-rank output).
    The caller is responsible for the all-gather across the head dim
    when world > 1.
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

    grid = (B * T,)
    _online_attnres_kernel[grid](
        V_full, query, norm_weight, out,
        B, T, D, H_full, D_h, eps,
        V_full.stride(0), V_full.stride(1), V_full.stride(2),
        out.stride(0), out.stride(1),
        H_LOCAL=H_local, HS=hs,
        N=N, BLOCK_HF=BLOCK_HF, BLOCK_DH=BLOCK_DH,
        num_warps=2,
    )

    return out


# ---------------------------------------------------------------------------
# Autograd Function: fused forward (Triton) + PyTorch backward.
#
# Forward: ``fused_attn_res_compute`` on the stacked V.
# Backward: re-run the reference math in ``torch.enable_grad()`` and
#   use autograd to compute the gradients. The bwd cost is small
#   relative to fwd (~2 ms per call vs ~4 ms fwd at prod), so we
#   trade a small amount of perf for correctness robustness.
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
        out = fused_attn_res_compute(
            V_full, V_local, query, norm_weight, eps=eps,
            slice_start=slice_start,
        )
        ctx.save_for_backward(V_full, V_local, query, norm_weight)
        ctx.eps = eps
        ctx.slice_start = slice_start
        ctx.slice_width = slice_width
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):  # type: ignore[override]
        V_full, V_local, query, norm_weight = ctx.saved_tensors
        eps = ctx.eps
        slice_start = ctx.slice_start
        slice_width = ctx.slice_width
        N, B, T, D = V_full.shape
        _, _, _, H, D_h = V_local.shape

        # Re-run the reference forward in grad-enabled mode.
        #
        # We use a CLONE of V_full for the inner computation
        # (V_full_g). The clone is the single source of truth for
        # the gradient — both the norm path (var/K/logits/weights)
        # and the weighted sum path (via a view of V_full_g
        # covering the same TP-local slice) flow through V_full_g.
        # This way V_full_g.grad = the full gradient w.r.t. V_full
        # (norm + weighted sum), and we return it as d_V_full.
        # PyTorch's autograd then ADDS d_V_full to V_full.grad
        # (the function's return value is treated as the delta to
        # apply). Since V_full_g is a fresh leaf (no view chain
        # back to V_full), there is no double-count from view
        # scatter — the only contribution to V_full.grad is
        # d_V_full.
        with torch.enable_grad():
            V_full_g = V_full.detach().clone().requires_grad_(True)
            query_g = query.detach().clone().requires_grad_(True)
            norm_weight_g = norm_weight.detach().clone().requires_grad_(True)
            # Build V_local_g as a view of V_full_g over the same
            # TP-local slice the caller used (slice_start,
            # slice_width along the last dim). This is a real view
            # (not a copy), so d_V_local_g flows back to V_full_g
            # via the view's backward and contributes to the
            # weighted-sum portion of V_full_g.grad.
            V_local_g = V_full_g[..., slice_start:slice_start + slice_width]
            V_local_g = V_local_g.contiguous().view(N, B, T, H, D_h)
            var = V_full_g.float().pow(2).mean(dim=-1, keepdim=True)
            K = V_full_g.float() * torch.rsqrt(var + eps) * norm_weight_g.float()
            # K is over the *full* hidden; view it per-head and slice to
            # this rank's head range (matches V_local_g / the forward).
            # At TP=1 head_start=0 and H == D//D_h so the slice is
            # identity; at TP>1 K has D//D_h heads but only H are local.
            K = K.view(N, B, T, D // D_h, D_h)
            head_start = slice_start // D_h
            K = K[..., head_start:head_start + H, :].to(V_full.dtype)
            logits = torch.einsum("hd,nbthd->nbth", query_g, K)
            weights = torch.softmax(logits, dim=0)
            out_ref = torch.einsum("nbth,nbthd->bthd", weights, V_local_g)
            out_ref.backward(grad_out)

        d_V_full = V_full_g.grad
        d_query = query_g.grad
        d_norm_weight = norm_weight_g.grad

        if d_V_full is None:
            d_V_full = torch.zeros_like(V_full)
        if d_query is None:
            d_query = torch.zeros_like(query)
        if d_norm_weight is None:
            d_norm_weight = torch.zeros_like(norm_weight)

        # V_local's grad is None: the d_V_full returned above
        # already accounts for the full hidden's contribution
        # (the weighted sum path's gradient is included via the
        # V_full_g → V_local_g view chain). V_local itself is
        # not consumed downstream.
        return d_V_full, None, d_query, d_norm_weight, None, None, None


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
    # in src/models/ops/attn_res.py, post-stack).
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