"""EFKDA: EFLA-form Kimi Delta Attention (KDA closed form, production).

Replaces the KDA Euler-step recurrence

    S_t = (I - β_t k_t kᵀ) Diag(exp(g_t)) S_{t-1} + β_t k_t v_tᵀ

with the EFLA paper's exact closed form (Lei et al., 2025; arXiv 2512.12602),
which absorbs the diagonal decay into a per-row normalisation and solves the
resulting rank-1 ODE exactly. With α_t = -expm1(-β_t ||k_t||²) / ||k_t||²
(the SCALAR per-step closed-form coefficient, derived from the matrix
exponential of -β N with N = k kᵀ and eigenvalue λ = ||k||²):

    S_t = (I - α_t k_t k_tᵀ) D S_{t-1} + α_t k_t v_tᵀ

Both forms have the same structural rank-1 update. The gap between them
is β_t ||k_t||² at small β||k||² (Euler ≈ closed form) and grows
quadratically past that — that's the EFLA paper's central claim.

The wrapper mirrors :class:`src.models.ops.kda.KDA` (projections,
short-conv, gated RMSNorm, o_proj) one-for-one and replaces the fla
:func:`chunk_kda` kernel with our PyTorch reference chunkwise EFLA
solver (:func:`efla_chunk_kda`). The kernel is a correct-but-slow
reference; a Triton kernel is the production path. The recurrence
semantics are identical: the wrapper's per-head layout, l2norm-on-qk,
and gate path match the KDA layer so the model weights transfer
without re-initialisation.

Reference: ``flash-linear-attention`` (vendored under
``src.models.ops._vendored.fla``); the EFLA reference in
``/hy-tmp/EFLA/mnist.py``; KDA paper (Kimi Linear, arXiv 2510.26692).

Contract: the kernel contract that this wrapper relies on (cu_seqlens
state-reset, per-chunk checkpoint, fp32 NaN guard, forward-substitution
for the (I+T) solve) is documented in ``docs/efkda_kernel_contract.md``.
The Triton kernel must hold the same contract; otherwise it will
re-introduce the inf-grad failure mode at higher throughput.
"""
import torch
import torch.nn as nn
from einops import rearrange
from torch.nn import functional as F

from src.models.ops._vendored.fla.layers.utils import (
    get_layer_cache,
    index_first_axis,
    pad_input,
    update_layer_cache,
)
from src.models.ops._vendored.fla.modules import FusedRMSNormGated, ShortConvolution
from src.models.ops._vendored.fla.ops.kda.chunk_efla_naive import (
    efla_chunk_kda as _efla_chunk_kda_ref,
)


class EFKDA(nn.Module):
    """EFLA-form KDA wrapper.

    Thin wrapper around :func:`efla_chunk_kda` for global (NoPE, non-causal)
    attention. Mirrors :class:`src.models.ops.kda.KDA`'s structure exactly
    so the HippoConfig surface is stable across the KDA → EFKDA swap (the
    only difference is the underlying recurrence).

    The ``allow_neg_eigval`` / ``safe_gate`` / ``lower_bound`` KDA flags are
    accepted but silently ignored: the EFLA path uses the closed form, so
    there's no "negative eigenvalue" extension to opt into and no M=16
    TensorCore fast path (the latter is a fla kernel-only optimization).
    """

    def __init__(self, config, layer_idx: int = 0):
        super().__init__()
        # ---- Cached kernel reference (avoids the import-time lookup on
        #      every forward). ----
        # Kernel backend: ``triton`` is the production path (Triton
        # fwd + PyTorch ref bwd via autograd.Function). Falls back to
        # the pure-PyTorch reference if the triton module fails to
        # import. ``ref`` selects the pure-PyTorch reference
        # explicitly (used for debugging numerical regressions).
        efkda_kernel = getattr(config, "efkda_kernel", "triton")
        self._efkda_kernel_backend = efkda_kernel
        self._efla_chunk_kda = _efla_chunk_kda_ref
        if efkda_kernel == "triton":
            try:
                from src.models.ops.efkda_triton import (
                    efla_chunk_kda as _efla_chunk_kda_tri,
                )
                self._efla_chunk_kda = _efla_chunk_kda_tri
            except ImportError as _e:
                import warnings
                warnings.warn(
                    f"[EFKDA] efkda_kernel='triton' requested but "
                    f"src.models.ops.efkda_triton failed to import "
                    f"({_e!r}); falling back to the PyTorch reference "
                    f"kernel (will be slow).",
                    RuntimeWarning,
                    stacklevel=2,
                )
                self._efkda_kernel_backend = "ref"

        # ---- Architectural mirrors of KimiDeltaAttention ----
        self.mode = config.efkda_mode
        self.allow_neg_eigval = config.allow_neg_eigval
        self.hidden_size = config.hidden_size
        self.expand_v = config.expand_v

        self.use_short_conv = config.use_short_conv
        self.conv_size = config.conv_size
        self.conv_bias = config.conv_bias

        self.head_dim = config.head_dim
        self.num_heads = config.num_heads
        # EFKDA's recurrence is KDA's scalar-β one, so the value-head
        # count matches the qk-head count (no GVA at the kernel level).
        self.num_v_heads = config.num_heads

        self.head_k_dim = self.head_dim
        self.head_v_dim = int(self.head_dim * self.expand_v)
        self.key_dim = int(self.num_heads * self.head_k_dim)
        self.value_dim = int(self.num_v_heads * self.head_v_dim)
        self.layer_idx = layer_idx

        # ---- Q/K/V projections ----
        self.q_proj = nn.Linear(self.hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.value_dim, bias=False)

        # ---- Short convolutions (depthwise, per-channel) ----
        if self.use_short_conv:
            self.q_conv1d = ShortConvolution(
                hidden_size=self.key_dim, kernel_size=self.conv_size,
                bias=self.conv_bias, activation="silu",
            )
            self.k_conv1d = ShortConvolution(
                hidden_size=self.key_dim, kernel_size=self.conv_size,
                bias=self.conv_bias, activation="silu",
            )
            self.v_conv1d = ShortConvolution(
                hidden_size=self.value_dim, kernel_size=self.conv_size,
                bias=self.conv_bias, activation="silu",
            )

        # ---- Per-channel log-decay: low-rank f_proj → key_dim ----
        self.f_proj = nn.Sequential(
            nn.Linear(self.hidden_size, self.head_v_dim, bias=False),
            nn.Linear(self.head_v_dim, self.key_dim, bias=False),
        )
        # ---- Scalar beta: b_proj → num_v_heads (KDA's signature) ----
        self.b_proj = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)

        # ---- A_log: per-value-head decay rate, log-uniform init ----
        # The KDA layer uses this even when safe_gate=False. We do
        # the same to keep the parameter set byte-identical to the
        # KDA layer — weight transfer is then a straight .copy_().
        self.A_log = nn.Parameter(
            torch.log(torch.empty(self.num_v_heads, dtype=torch.float32).uniform_(1, 16))
        )
        self.A_log._no_weight_decay = True
        # ---- dt_bias: per-key-channel softplus bias (FP32) ----
        import math as _math
        dt = torch.exp(
            torch.rand(self.key_dim, dtype=torch.float32)
            * (_math.log(0.1) - _math.log(0.001))
            + _math.log(0.001)
        ).clamp(min=1e-4)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)
        self.dt_bias._no_weight_decay = True

        # ---- Output path: sigmoid-gated RMSNorm + projection ----
        self.g_proj = nn.Sequential(
            nn.Linear(self.hidden_size, self.head_v_dim, bias=False),
            nn.Linear(self.head_v_dim, self.value_dim, bias=True),
        )
        self.o_norm = FusedRMSNormGated(self.head_v_dim, activation="sigmoid",
                                        eps=config.rms_norm_eps)
        self.o_proj = nn.Linear(self.value_dim, self.hidden_size, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass. Mirrors :class:`KDA.forward` and :class:`GDN2.forward`.

        Args:
            x: Input tensor ``[batch_size, seq_len, hidden_size]``.
            cu_seqlens: optional varlen offsets for chunk-aligned FFD-packed
                inputs. When set, the batch dim is folded into the seq dim
                so the kernel sees ``batch_size=1`` and resets the recurrent
                state at each ``cu_seqlens`` boundary.

        Returns:
            Output tensor ``[batch_size, seq_len, hidden_size]``.
        """
        B, T, _ = x.shape
        # Packed path: fold batch into seq so the kernel sees B=1.
        x_in = x.reshape(1, B * T, -1) if cu_seqlens is not None else x

        # ---- Short conv (or SiLU on the projection) ----
        if self.use_short_conv:
            q, _ = self.q_conv1d(self.q_proj(x_in), cu_seqlens=cu_seqlens)
            k, _ = self.k_conv1d(self.k_proj(x_in), cu_seqlens=cu_seqlens)
            v, _ = self.v_conv1d(self.v_proj(x_in), cu_seqlens=cu_seqlens)
        else:
            q = F.silu(self.q_proj(x_in))
            k = F.silu(self.k_proj(x_in))
            v = F.silu(self.v_proj(x_in))

        # ---- Gate: g = softplus(f_proj + dt_bias), then -exp(A_log) per head ----
        g = F.softplus(self.f_proj(x_in).float() + self.dt_bias)
        # ---- Scalar beta: b_proj sigmoid ----
        beta = self.b_proj(x_in).sigmoid()

        # ---- Per-head reshape ----
        q, k, g = (rearrange(x, "... (h d) -> ... h d", d=self.head_k_dim)
                   for x in (q, k, g))
        v = rearrange(v, "... (h d) -> ... h d", d=self.head_v_dim)
        # Apply per-head A_log decay rate (FP32 for the cumsum's stability).
        g = -self.A_log.float().exp().unsqueeze(-1) * g

        if self.allow_neg_eigval:
            beta = beta * 2.0

        # ---- L2-normalize q and k (matches the fla KDA layer's
        #      ``use_qk_l2norm_in_kernel=True`` semantics; the EFLA
        #      closed form handles the l2norm'd k through its standard
        #      α = -expm1(-β·||k||²)/||k||² formula, which simplifies
        #      to -expm1(-β) when ||k||=1). ----
        q = F.normalize(q, p=2, dim=-1, eps=1e-6)
        k = F.normalize(k, p=2, dim=-1, eps=1e-6)

        # ---- The EFLA chunkwise recurrence ----
        # The kernel handles the closed-form rank-1 update; the
        # per-chunk (I+T)^{-1} solve composes the per-step α on
        # writes (column-α, matching the per-token EFLA derivation
        # in arXiv 2512.12602). cu_seqlens is threaded through to
        # the kernel so the recurrent state ``h`` is reset to zero
        # at every chunk that starts a new doc — required when
        # short_conv resets q/k/v state at cu_seqlens boundaries
        # but the kernel sees one flat sequence. The kernel asserts
        # chunk-alignment, which is exactly what pack_chunk_aligned
        # produces.
        o, _ = self._efla_chunk_kda(
            q=q, k=k, v=v, g=g, beta=beta,
            initial_state=None, output_final_state=False,
            chunk_size=64,  # matches the fla chunk_kda's hardcoded 64
            cu_seqlens=cu_seqlens,
        )

        # ---- Output path: gated RMSNorm + projection ----
        if cu_seqlens is not None:
            o = o.reshape(B, T, self.num_v_heads, self.head_v_dim)
        g_for_norm = rearrange(self.g_proj(x_in), "... (h d) -> ... h d",
                               d=self.head_v_dim)
        o = self.o_norm(o, g_for_norm)
        o = rearrange(o, "b t h d -> b t (h d)")
        o = self.o_proj(o)
        return o
