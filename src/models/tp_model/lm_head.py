"""TP-sharded lm_head: training (FusedLinearCE) and inference (shim).

The training path is :class:`TPFusedLceLoss` — fused
``hidden @ embed.T`` matmul + cross-entropy in a single
fla kernel that chunks over the sequence dim and uses online
softmax. The full ``[B, T, V // world]`` sharded logits tensor
is never materialised (~3.9 GB savings at B=4, T=1024, V=248k,
TP=1 vs the old sharded_logits + contiguous-copy path).

The inference / generation path is :class:`TPLmHead` — a thin
shim that does the narrow-view matmul and returns the full
sharded logits. Used by ``model.generate`` and any caller that
needs the raw logits tensor.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.tp_layers import get_tp_rank, get_tp_world_size, get_tp_group
from src.models.ops._vendored.fla.modules.fused_linear_cross_entropy import (
    fused_linear_cross_entropy_backward,
    fused_linear_cross_entropy_forward,
    fused_linear_cross_entropy_tp_forward,
)


class _TiedFusedLCEFunction(torch.autograd.Function):
    """Custom autograd that wraps fla's FusedLinearCrossEntropy for
    tied-embedding setups.

    The fla FusedLinearCE is designed to take a weight tensor and
    compute both the linear matmul ``hidden @ weight.T`` and the
    cross-entropy loss, returning ``(loss, dx, dw, db)``. The
    ``dw`` it returns is a **fresh** tensor (not a view of
    ``weight``), so for tied embeddings we cannot rely on
    PyTorch's autograd to scatter it back to
    ``embed_tokens.weight.grad``.

    This wrapper:
      - forward: calls the raw ``fused_linear_cross_entropy_forward``
        with the narrow view of the embed as the weight; saves
        ``(dx, dw, embed_weight, start, vp)`` for backward.
      - backward: scales ``dx`` and ``dw`` by ``do`` via the fla
        backward kernel, then ``add_``'s ``dw`` into
        ``embed_weight.grad[start:start+vp]`` and returns ``dx``
        as the gradient w.r.t. the hidden state.

    The full ``[B, T, V // world]`` sharded logits tensor is
    never materialised, and the per-chunk logits are written
    in-place by the fla kernel (the dlogits overwrite the logits
    buffer), so the per-chunk peak is just ``[C, V // world]``
    where ``C`` is the chunk size (small, set by ``num_chunks``).
    """

    @staticmethod
    def forward(
        ctx,
        hidden: torch.Tensor,           # [..., H]
        target: torch.Tensor,           # [...]
        local_w: torch.Tensor,          # [V/world, H], view of embed
        embed_weight: torch.Tensor,     # [V, H], parent embedding weight
        start: int,                     # vocab slice start for this rank
        vp: int,                        # vocab_per_partition = V / world
        ignore_index: int,
        num_chunks: int,
    ) -> torch.Tensor:
        # Flatten to [N, H] / [N] for the fla kernel.
        N = hidden.shape[:-1].numel()
        H = hidden.shape[-1]
        flat_hidden = hidden.reshape(N, H)
        flat_target = target.reshape(N)

        if get_tp_world_size() > 1:
            # TP-sharded vocab path: the kernel reads
            # ``logits[b_y]`` where b_y is the GLOBAL target id and
            # logits is the rank-local ``[N, V/world]`` slice. The
            # stock fla FusedLinearCE would OOB-read for off-rank
            # targets; ``fused_linear_cross_entropy_tp_forward``
            # wraps the kernel with TP-aware lse/loss all-reduce
            # and a per-rank target remap. See
            # ``fused_linear_cross_entropy.py::cross_entropy_kernel_tp``
            # for the per-row kernel details.
            loss, dx, dw, _ = fused_linear_cross_entropy_tp_forward(
                flat_hidden, flat_target, local_w, None,
                ignore_index=ignore_index, num_chunks=num_chunks,
                tp_group=get_tp_group(),
                tp_rank=get_tp_rank(),
                tp_world=get_tp_world_size(),
            )
        else:
            loss, dx, dw, _ = fused_linear_cross_entropy_forward(
                flat_hidden, flat_target, local_w, None,
                ignore_index=ignore_index, num_chunks=num_chunks,
                reduction="mean",
            )

        # ``dw`` comes out as FP32 from the kernel (accumulated in
        # FP32 inside the chunked loop for precision); cast to the
        # embed weight's dtype (FP16 in our setup) so the add into
        # ``embed_weight.grad`` doesn't need a type-promoted copy.
        if dw is not None and dw.dtype != embed_weight.dtype:
            dw = dw.to(embed_weight.dtype)

        # ``dx`` and ``dw`` stay 2D ``[N, H]`` / ``[V/world, H]``
        # in the saved tensors because the fla
        # ``fused_linear_cross_entropy_backward`` kernel does
        # ``N, H = dx.shape`` and ``V, H = dw.shape`` — passing a
        # 3D dx (e.g. ``[B, T, H]``) blows up with "too many
        # values to unpack (expected 2)". We reshape dx back to
        # ``hidden``'s original shape in :meth:`backward` after
        # the do-scaling kernel is done with it.
        ctx.save_for_backward(dx, dw, embed_weight)
        ctx.start = start
        ctx.vp = vp
        ctx.hidden_shape = hidden.shape
        return loss

    @staticmethod
    def backward(ctx, do):
        dx, dw, embed_weight = ctx.saved_tensors

        # Scale dx and dw by do via the fla backward kernel. Both
        # must be 2D here (see comment in forward). The kernel
        # skips the multiplication if do is exactly 1.0 (the
        # common case for a scalar loss seed).
        dx, dw, _ = fused_linear_cross_entropy_backward(do, dx, dw, None)

        # Reshape dx back to ``hidden``'s original shape so the
        # caller (autograd) sees the right gradient for the
        # hidden state.
        dx = dx.view(ctx.hidden_shape)

        # Scatter the (now do-scaled) dw back into the parent
        # embedding's gradient. ``embed_weight.grad`` may not
        # exist yet (PyTorch lazily allocates it on first .grad
        # access); standard pattern is to alloc and assign.
        if dw is not None:
            if embed_weight.grad is None:
                embed_weight.grad = torch.zeros_like(embed_weight)
            embed_weight.grad[ctx.start:ctx.start + ctx.vp].add_(dw)

        return dx, None, None, None, None, None, None, None


class TPFusedLceLoss(nn.Module):
    """Fused lm_head + cross-entropy with TP sharding and online
    chunked softmax.

    Replaces the v0.0.0 ``TPLmHead`` + ``FusedCrossEntropyLoss`` pair.

    Memory characteristics vs. the old path
    ----------------------------------------

    Old path peak (per rank, B=4, T=1024, V=248320, TP=1):

      * ``sharded_logits`` = [B, T, V] FP16 = 1.99 GB
      * ``shift_logits = sharded_logits[..., :-1, :].contiguous()``
        = [B, T-1, V] FP16 = 1.94 GB (the .contiguous() is a fresh
        copy because the time-slice is non-contiguous)
      * lse / losses / z_losses intermediates of FusedCrossEntropyLoss
        = [n_splits, N] FP32 ≈ 64 KB (small)

    New path peak (per rank, same config, ``num_chunks=64``):

      * Per-chunk ``[C, V // world]`` logits in registers inside
        the fla kernel, where C = next_pow2(ceil(N / NC)) ≈ 64 →
        chunk peak ≈ 32 MB. The dlogits overwrite the logits
        buffer in-place inside the kernel, so backward does not
        require the full logits to be saved.
      * The full [B, T, V // world] tensor is never materialised.
      * A small ``[N]`` lse buffer and a ``[N, H]`` dx accumulator
        (= 8 MB at B=4, T=1024, H=1024).
      * A ``[V // world, H]`` dw accumulator in the kernel
        (FP32, cast to FP16 at the end), scattered into
        ``embed_weight.grad[start:start+vp]`` in backward.

    Net savings at TP=1: ~3.9 GB (the two sharded_logits
    tensors disappear, replaced by ~40 MB of chunked workspace
    + the dw scatter).

    Online softmax
    --------------

    The fla FusedLinearCE does a 2-pass online softmax over the
    vocab dim of each chunk: pass 1 (``logsumexp_fwd_kernel``)
    computes a per-row ``(max, sum_exp)`` in one streaming
    pass, pass 2 (``cross_entropy_kernel``) uses the precomputed
    lse to compute the loss + dx in a second streaming pass.
    No ``[n_splits, n_rows]`` intermediate lse / losses tensor
    is ever allocated — the running stats are kept in registers
    per thread block.

    Configuration
    -------------

    ``num_chunks``: how many pieces to split the N (sequence)
    dim into. The fla kernel chooses ``C = next_pow2(ceil(N/NC))``
    tokens per chunk, so the per-chunk peak logits are
    ``[C, V // world]``. The default is 8 (matches fla's
    default), but we override to 64 to bound per-chunk peak to
    ~32 MB at our config (vs. ~254 MB at the default).
    """

    def __init__(
        self,
        embed_tokens: nn.Embedding,
        hidden_size: int,
        vocab_size: int,
        bias: bool = False,
        ignore_index: int = -100,
        num_chunks: int = 8,
        device=None,
        dtype=None,
        # When the embed is already sharded (see :class:`TPShardedEmbed`),
        # ``embed_tokens.weight`` is the local [vp, H] shard and
        # ``start_offset`` is 0 (no logical-vocab offset within the
        # local weight). When the embed is replicated (legacy v0.0.0
        # path), ``start_offset = rank * vocab_per_partition`` so
        # the FusedLinearCE matmul sees the right slice of the full
        # [V, H] weight. The default below is the legacy behavior;
        # the sharded path overrides both.
        start_offset: int | None = None,
        local_vocab_size: int | None = None,
    ) -> None:
        super().__init__()
        # Bypass nn.Module.__setattr__ so the embed is NOT registered
        # as a submodule. Otherwise TPFusedLceLoss.parameters()
        # recurses into the embed and the trainable-parameter counter
        # double-counts it (962M reported vs 708M actual — see the
        # long note in :meth:`TPHippoModel.trainable_parameters`).
        # The attribute is still accessible via ``self.embed_tokens``
        # from Python; we just don't want it in ``self._modules``.
        object.__setattr__(self, "embed_tokens", embed_tokens)
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.world = get_tp_world_size()
        assert vocab_size % self.world == 0, (
            f"vocab_size={vocab_size} not divisible by tp_world={self.world}"
        )
        self.vocab_per_partition = vocab_size // self.world
        self.rank = get_tp_rank()
        # ``_start`` is the offset into the LOGICAL vocab space for
        # this rank's slice. It is used in two places:
        #   1. The narrow of ``embed_tokens.weight`` for the matmul
        #      (in the replicated case, this is a real slice; in
        #      the sharded case, the narrow is identity and the
        #      offset is 0).
        #   2. The ``ctx.start`` passed to the custom autograd, so
        #      the FusedLinearCE dw is added to the correct slice
        #      of ``embed_tokens.weight.grad``.
        if start_offset is None:
            # Legacy replicated-embed path: this rank's slice
            # starts at ``rank * vp`` in the full [V, H] weight.
            self._start = self.rank * self.vocab_per_partition
        else:
            self._start = start_offset
        # ``_local_vocab_size`` is the size of this rank's slice
        # of the weight. In the sharded case, the local weight
        # already has shape [vp, H] so this is just ``vp``. In
        # the replicated case, it's also ``vp`` (the slice length).
        if local_vocab_size is None:
            self._local_vocab_size = self.vocab_per_partition
        else:
            self._local_vocab_size = local_vocab_size
        self.ignore_index = ignore_index
        self.num_chunks = num_chunks

        if bias:
            self.bias = nn.Parameter(
                torch.empty(self._local_vocab_size, device=device, dtype=dtype)
            )
            nn.init.zeros_(self.bias)
        else:
            self.register_parameter("bias", None)

    def forward(self, hidden: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Narrow view of the embed for this rank's vocab shard.
        # The slice is a view, but FusedLinearCE doesn't write
        # back into the weight (it allocates its own dw), so the
        # view only serves as a read-only matmul operand.
        #
        # In the sharded case (``embed_tokens`` is a
        # :class:`TPShardedEmbed`), the weight is already the
        # local [vp, H] shard and the narrow is identity.
        local_w = self.embed_tokens.weight.narrow(0, self._start, self._local_vocab_size)
        return _TiedFusedLCEFunction.apply(
            hidden, target, local_w, self.embed_tokens.weight,
            self._start, self._local_vocab_size,
            self.ignore_index, self.num_chunks,
        )

    # Kept for backward-compat with any caller that introspects
    # the module name. The old ``TPLmHead`` was registered under
    # ``lm_head_per_device``; the new module plays the same role.
    @property
    def weight(self) -> torch.Tensor:
        return self.embed_tokens.weight

    def extra_repr(self) -> str:
        return (
            f"hidden_size={self.hidden_size}, vocab_size={self.vocab_size}, "
            f"vocab_per_partition={self.vocab_per_partition}, "
            f"tp_world={self.world}, bias={self.bias is not None}, "
            f"tied_to=embed_tokens, num_chunks={self.num_chunks}"
        )


# Backward-compat alias. The old ``TPLmHead`` is still referenced
# by inference paths (e.g. model.generate); we keep a thin shim
# that returns the sharded logits without going through CE.
class TPLmHead(nn.Module):
    """Thin shim kept for inference / generation paths that need
    the full sharded logits tensor (e.g. ``model.generate``).

    The training path uses :class:`TPFusedLceLoss` instead, which
    fuses the matmul with the cross-entropy and never
    materialises the sharded logits.

    This shim is a no-op for parameter allocation: it holds no
    parameters of its own and just exposes the same narrow-view
    matmul the old TPLmHead did.
    """

    def __init__(
        self,
        embed_tokens: nn.Embedding,
        hidden_size: int,
        vocab_size: int,
        bias: bool = False,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        # Bypass nn.Module.__setattr__ so the embed is NOT registered
        # as a submodule — see the matching note in
        # :class:`TPFusedLceLoss.__init__`. TPLmHead is the inference
        # / generation shim; the embed is borrowed by id() so any
        # double-counting of parameters in introspection would
        # corrupt trainable-parameter logs.
        object.__setattr__(self, "embed_tokens", embed_tokens)
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.world = get_tp_world_size()
        assert vocab_size % self.world == 0
        self.vocab_per_partition = vocab_size // self.world
        self.rank = get_tp_rank()
        self._start = self.rank * self.vocab_per_partition

        if bias:
            self.bias = nn.Parameter(
                torch.empty(self.vocab_per_partition, device=device, dtype=dtype)
            )
            nn.init.zeros_(self.bias)
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        local_w = self.embed_tokens.weight.narrow(0, self._start, self.vocab_per_partition)
        return F.linear(x, local_w, self.bias)

    def extra_repr(self) -> str:
        return (
            f"hidden_size={self.hidden_size}, vocab_size={self.vocab_size}, "
            f"vocab_per_partition={self.vocab_per_partition}, "
            f"tp_world={self.world}, bias={self.bias is not None}, "
            f"tied_to=embed_tokens (inference-only shim)"
        )
