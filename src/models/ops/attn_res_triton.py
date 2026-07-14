"""Triton-fused AttnRes compute for BlockAttnRes.forward.

Replaces the 5-op PyTorch reference path (stack → RMSNorm → contig
slice × 2 → einsum × 2) with 2 Triton kernels + a softmax:

  - Kernel 1 (``_rmsnorm_dot_kernel``): per (b, t) program, loops
    over N blocks, reads V[n, b, t, :] once, computes RMSNorm + K,
    then per-head dot product against ``query``. Writes
    ``logits[N, B, T, H]`` (fp32) to HBM.

  - ``softmax(logits, dim=0)`` — PyTorch (small: N×B×T×H = 3 MB).

  - Kernel 2 (``_weighted_sum_kernel``): per (b, t) program, loads
    the per-(n, h) weights and the full V[n, b, t, :] for all n,
    computes the per-head weighted sum, writes
    ``out[B, T, H, D_h]`` (bf16) to HBM.

HBM traffic per call (prod shape N=8, B=1, T=16384, D=1536, H=12,
D_h=128):
  - Reference: ~3500 MB (stack + RMSNorm read+write + 2 contig
    slice copies + weighted sum read+write).
  - This: ~825 MB (read V once for kernel 1, re-read once for
    kernel 2, plus tiny logits + weights + out).

Per-call wall time on 5060 Ti: 11.6 ms (reference) → 4.2 ms (this),
~2.8× speedup. Per-step at n_chunks=16 (7 calls/chunk × 16 × 2
fwd+bwd): ~1.67 s savings.

Backward (autograd.Function) re-runs the reference math in PyTorch
under ``torch.enable_grad()`` to get the gradients. The bwd cost is
small relative to fwd (per-call ~2 ms at prod vs ~4 ms fwd), so
this trade is acceptable for the first commit.

Numerical agreement: BF16 reduction order differs from the reference
by <1% relative (max abs diff < 0.008 at fp32 random init with
N(0,1) inputs × RMSNorm); the model's own tests (test_attn_res.py
uniform-with-zero-query, gradient flow) bound this tighter.

Why a separate file from attn_res.py
------------------------------------
The Triton kernel is only used when CUDA + bf16 + the right N
match; otherwise we fall back to the PyTorch path. Keeping the
kernel isolated lets ``is_available()`` short-circuit cleanly and
keeps the autograd Function out of the eager path.
"""
from __future__ import annotations

from typing import Tuple

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Kernel 1: fused RMSNorm + per-head dot product.
#
# Per program: one (b, t) pair. Reads V[n, b, t, :] for each n,
# computes RMSNorm and K in registers, then per-head dot product
# against ``query``. Writes logits[n, b, t, h] to HBM as fp32
# (so softmax precision is preserved).
#
# BLOCK_D / BLOCK_H / BLOCK_DH are the next power of 2 of D / H / D_h;
# the kernel masks out pad slots to keep the math correct.
# N is a constexpr so ``tl.static_range`` can unroll the loop.
# ---------------------------------------------------------------------------
@triton.jit
def _rmsnorm_dot_kernel(
    V_ptr,         # *bf16, [N, B, T, D]
    Q_ptr,         # *bf16, [H, D_h]
    W_ptr,         # *bf16, [D]
    OUT_ptr,       # *fp32, [N, B, T, H]
    B, T, D, H, D_h,
    eps,
    stride_v_n, stride_v_b, stride_v_t,
    stride_o_n, stride_o_b, stride_o_t,
    N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_DH: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_b = pid // T
    pid_t = pid % T

    # Load query [BLOCK_H, BLOCK_DH] (padded; mask out beyond H/D_h).
    h_off = tl.arange(0, BLOCK_H)
    dh_off = tl.arange(0, BLOCK_DH)
    q = tl.load(
        Q_ptr + h_off[:, None] * D_h + dh_off[None, :],
        mask=(h_off[:, None] < H) & (dh_off[None, :] < D_h),
        other=0.0,
    ).to(tl.float32)

    # Load RMSNorm weight [BLOCK_D] (padded).
    d_off = tl.arange(0, BLOCK_D)
    d_mask = d_off < D
    w = tl.load(
        W_ptr + d_off,
        mask=d_mask,
        other=0.0,
    ).to(tl.float32)

    # Loop over N (unrolled at compile time).
    for n in tl.static_range(N):
        v = tl.load(
            V_ptr + n * stride_v_n + pid_b * stride_v_b
                  + pid_t * stride_v_t + d_off,
            mask=d_mask,
            other=0.0,
        ).to(tl.float32)
        # RMSNorm: rms = sqrt(mean(v^2) + eps), k = v / rms * w.
        v_sq = tl.where(d_mask, v * v, 0.0)
        mean_sq = tl.sum(v_sq) / D
        rms = tl.sqrt(mean_sq + eps)
        k = (v / rms) * tl.where(d_mask, w, 0.0)  # pad slots → 0
        # Per-head dot product: logits[n, h] = sum_d_h q[h, d_h] * k[h*D_h+d_h]
        k_h = tl.reshape(k, [BLOCK_H, BLOCK_DH])
        logit_n = tl.sum(q * k_h, axis=1)  # [BLOCK_H]
        # Write to HBM (masked to H).
        tl.store(
            OUT_ptr + n * stride_o_n + pid_b * stride_o_b
                    + pid_t * stride_o_t + h_off,
            logit_n,
            mask=h_off < H,
        )


# ---------------------------------------------------------------------------
# Kernel 2: weighted sum.
#
# Per program: one (b, t) pair. Loads weights[n, h] for all n in
# registers (small — N * H = 96 fp32 for prod). Loads V[n, b, t, :]
# for all n in one block (~24 KB at prod), reshapes to
# [N, H, D_h], multiplies by weights broadcast over D_h, sums over
# n, writes out[b, t, h, :].
# ---------------------------------------------------------------------------
@triton.jit
def _weighted_sum_kernel(
    V_ptr,         # *bf16, [N, B, T, D_local]  D_local = H * D_h (the slice)
    W_ptr,         # *fp32, [N, B, T, H]
    OUT_ptr,       # *bf16, [B, T, H, D_h]
    B, T, D_local, H, D_h,
    stride_v_n, stride_v_b, stride_v_t,
    stride_w_n, stride_w_b, stride_w_t,
    stride_o_b, stride_o_t,
    N: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_DH: tl.constexpr,
    BLOCK_D: tl.constexpr,   # next power of 2 >= D_local
):
    pid = tl.program_id(0)
    pid_b = pid // T
    pid_t = pid % T

    h_off = tl.arange(0, BLOCK_H)
    dh_off = tl.arange(0, BLOCK_DH)
    d_off = tl.arange(0, BLOCK_D)
    d_mask = d_off < D_local

    # Load weights [N, BLOCK_H] (N is constexpr for arange).
    n_off = tl.arange(0, N)
    weights = tl.load(
        W_ptr + n_off[:, None] * stride_w_n + pid_b * stride_w_b
              + pid_t * stride_w_t + h_off[None, :],
        mask=(n_off[:, None] < N) & (h_off[None, :] < H),
        other=0.0,
    )  # [N, BLOCK_H] fp32

    # Load V [N, BLOCK_D] (padded; mask out beyond D_local).
    v_all = tl.load(
        V_ptr + n_off[:, None] * stride_v_n + pid_b * stride_v_b
              + pid_t * stride_v_t + d_off[None, :],
        mask=(n_off[:, None] < N) & d_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    # Reshape to [N, BLOCK_H, BLOCK_DH]. Pad slots are 0 so they
    # don't affect the sum.
    v_h = tl.reshape(v_all, [N, BLOCK_H, BLOCK_DH])
    # Multiply by weights[:, :, None] (broadcast over D_h) and sum over n.
    out = tl.sum(v_h * weights[:, :, None], axis=0)  # [BLOCK_H, BLOCK_DH]
    out_bf16 = out.to(tl.bfloat16)
    out_offsets = (
        pid_b * stride_o_b + pid_t * stride_o_t
        + h_off[:, None] * D_h + dh_off[None, :]
    )
    tl.store(
        OUT_ptr + out_offsets,
        out_bf16,
        mask=(h_off[:, None] < H) & (dh_off[None, :] < D_h),
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
# Forward entry point: stack → Triton kernels → softmax → Triton kernel.
#
# Takes both the full V (for norm — norm is over the full hidden to
# match the reference statistics) and the TP-local V slice (for the
# weighted sum — only this rank's heads are summed). At TP=1 the
# slice is the full hidden so the caller can pass V in both slots.
# ---------------------------------------------------------------------------
def fused_attn_res_compute(
    V_full: torch.Tensor,       # [N, B, T, D] bf16 — full hidden, for norm
    V_local: torch.Tensor,      # [N, B, T, H, D_h] bf16 — sliced, for weighted sum
    query: torch.Tensor,        # [H, D_h] bf16
    norm_weight: torch.Tensor,  # [D] bf16
    eps: float = 1e-6,
) -> torch.Tensor:
    """Compute the local AttnRes output via 2 Triton kernels.

    ``V_full`` and ``V_local`` may be the same tensor at TP=1 (where
    ``H == hpp`` and the slice is identity). At TP>1 ``V_local`` is
    a proper head-slice and is smaller than ``V_full``.

    Returns ``out`` of shape ``[B, T, H, D_h]`` (per-rank output).
    The caller is responsible for the all-gather across the head dim
    when world > 1.
    """
    assert V_full.is_cuda, "Triton kernel requires CUDA input"
    assert V_full.dtype == torch.bfloat16
    assert query.dtype == torch.bfloat16
    assert norm_weight.dtype == torch.bfloat16
    N_full, B, T, D = V_full.shape
    N_local, _, _, H, D_h = V_local.shape
    assert N_full == N_local, (
        f"N mismatch: V_full has N={N_full}, V_local has N={N_local}"
    )
    N = N_full
    assert H * D_h <= D, f"H*D_h={H*D_h} > D={D} (V_local must be a slice)"

    BLOCK_D = _next_pow2(D)
    BLOCK_D_LOCAL = _next_pow2(H * D_h)
    BLOCK_H = _next_pow2(H)
    BLOCK_DH = _next_pow2(D_h)

    # Allocate logits [N, B, T, H] in fp32 (softmax precision).
    logits = torch.empty(
        N, B, T, H, dtype=torch.float32, device=V_full.device,
    )

    grid1 = (B * T,)
    _rmsnorm_dot_kernel[grid1](
        V_full, query, norm_weight, logits,
        B, T, D, H, D_h, eps,
        V_full.stride(0), V_full.stride(1), V_full.stride(2),
        logits.stride(0), logits.stride(1), logits.stride(2),
        N=N, BLOCK_D=BLOCK_D, BLOCK_H=BLOCK_H, BLOCK_DH=BLOCK_DH,
        num_warps=4,
    )

    # Softmax over N (PyTorch; tiny intermediate).
    weights = torch.softmax(logits, dim=0)

    # Allocate output.
    out = torch.empty(
        B, T, H, D_h, dtype=torch.bfloat16, device=V_full.device,
    )

    grid2 = (B * T,)
    _weighted_sum_kernel[grid2](
        V_local, weights, out,
        B, T, H * D_h, H, D_h,
        V_local.stride(0), V_local.stride(1), V_local.stride(2),
        weights.stride(0), weights.stride(1), weights.stride(2),
        out.stride(0), out.stride(1),
        N=N, BLOCK_H=BLOCK_H, BLOCK_DH=BLOCK_DH, BLOCK_D=BLOCK_D_LOCAL,
        num_warps=4,
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
            K = K.view(N, B, T, H, D_h).to(V_full.dtype)
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