"""Tensor-parallel HippoModel for Megatron-style TP.

TP design (assumes a single process spanning N GPUs):

  Replicated on every rank
  ------------------------
  - RMSNorm (full hidden)
  - BlockAttnRes (full hidden pseudo-query and norm)

  Vocab-sharded (one rank holds [vocab // world, hidden])
  --------------------------------------------------------
  - embed_tokens: column-parallel on the vocab dim. Forward
    does a masked lookup + all-reduce to produce the full
    [B, T, H] hidden on every rank (see :class:`TPShardedEmbed`).
    The lm_head matmul is naturally the column-parallel
    ``hidden @ local_embed.T`` → [N, vp] sharded logits, which
    is exactly what the FusedLinearCE consumes. Memory savings
    at TP=8: ~7 * 508 MB of duplicate embed params and ~7 *
    2032 MB of duplicate AdamW state = ~17.7 GB across the
    node (vs the v0.0.0 replicated path).

  Column-parallel (output sharded, no all-reduce)
  -----------------------------------------------
  - SwiGLU gate_proj, up_proj (output = intermediate, sharded)
  - Fused lm_head: matmul ``hidden @ embed.T`` is done inside the
    fused linear+CE kernel along the sequence dim, so the full
    ``[B, T, vocab // world]`` sharded logits tensor is never
    materialised (replaces the old TPLmHead path which did the
    matmul and then ``.contiguous()``'d a copy for the CE kernel).
  - GDN2 q/k/v projections (output = key_dim / value_dim, sharded
    along the head dim)

  Row-parallel (input sharded, all-reduce output)
  -----------------------------------------------
  - SwiGLU down_proj (input = intermediate, sharded; output
    all-reduced to full hidden)
  - GDN2 o_proj (input = value_dim sharded, output all-reduced)

The forward pass is: sharded embed (masked lookup + all-reduce
on the partial embeddings) -> 32 layers (attn_norm -> GDN2 ->
mlp_norm -> SwiGLU -> all-reduce inside down_proj and inside
o_proj) -> final norm -> fused lm_head+CE (sharded matmul +
online chunked softmax) -> scalar loss.

Inter-rank comm per forward:
  - Embed lookup: one all-reduce of [B, T, H] = 8 MB.
  - Per layer: one all-reduce in down_proj and one in o_proj.
  - lm_head: no inter-rank comm (each rank consumes its own
    vocab shard; FusedLinearCE accumulates the LSE locally
    and the loss is reduced inside the module wrapper).

For tied embeddings, the FusedLinearCE's ``weight`` arg is the
local [vp, H] embed shard, and the resulting ``dw`` gradient is
a fresh tensor that we add back to ``embed_tokens.weight.grad``
in our custom autograd :class:`_TiedFusedLCEFunction`. The
sharded embed lookup's backward computes the dw for the embed
via ``index_add_`` (no inter-rank comm); PyTorch's autograd
sums the two dw contributions into ``embed_tokens.weight.grad``.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .norms import RMSNorm
from .ops.attn_res import BlockAttnRes
from .tp_layers import (
    ColumnParallelLinear,
    RowParallelLinear,
    get_tp_group,
    get_tp_rank,
    get_tp_world_size,
    tp_all_reduce_sum,
)
from src.models.ops._vendored.fla.modules.fused_cross_entropy import FusedCrossEntropyLoss
from src.models.ops._vendored.fla.modules.fused_linear_cross_entropy import (
    fused_linear_cross_entropy_forward,
    fused_linear_cross_entropy_backward,
)


# --------------------------------------------------------------------------- #
# TP sharded embedding (vocab-parallel nn.Embedding)                           #
# --------------------------------------------------------------------------- #
class _ShardedEmbedLookup(torch.autograd.Function):
    """Custom autograd for vocab-sharded embedding lookup.

    Forward (per rank):
      - weight:           [vp, H]  (the local vocab shard)
      - input_ids:        [B, T]   (replicated, full vocab indices)
      - Returns hidden:   [B, T, H] (replicated via all-reduce)

    The forward does a masked lookup ``weight[clamp(input_ids -
    start, 0, vp - 1)]`` and multiplies by a ``[B, T]`` mask that is
    True only for tokens belonging to this rank's vocab shard, then
    all-reduces the result across the TP group. The all-reduce
    sums the per-rank partial embeddings into the full ``[B, T, H]``
    hidden on every rank.

    Backward:
      - dhidden:  [B, T, H] (replicated, flowing in from the layers)
      - Returns dw: [vp, H] (gradient for the local weight shard)

    The backward does NOT need any inter-rank comm. Each rank
    receives the full dhidden; it scatters dhidden into the local
    weight shard using ``index_add_`` (the per-row mask is already
    zero for off-rank rows, so the scatter is a no-op for them).
    """

    @staticmethod
    def forward(ctx, weight, input_ids, start, vp):
        # ``input_ids`` lives on this rank's device; ``weight`` is
        # the local [vp, H] shard. Both are on the same device (we
        # constructed the embed with .to(device=d)).
        local_ids = input_ids - start                       # [B, T]
        mask = (local_ids >= 0) & (local_ids < vp)          # [B, T] bool
        local_ids_safe = local_ids.clamp(0, vp - 1)
        local_emb = weight[local_ids_safe]                  # [B, T, H]
        # Zero out off-rank rows so the all-reduce sums to the
        # correct value (only the owning rank contributes
        # non-zero to each row of the sum).
        local_emb = local_emb * mask.unsqueeze(-1).to(local_emb.dtype)
        hidden = tp_all_reduce_sum(local_emb)               # [B, T, H]
        ctx.save_for_backward(input_ids, local_ids_safe, mask)
        ctx.start = start
        ctx.vp = vp
        return hidden

    @staticmethod
    def backward(ctx, dhidden):
        input_ids, local_ids_safe, mask = ctx.saved_tensors
        # dhidden: [B, T, H] on every rank (it flows backward from
        # the layers, which see the replicated full hidden).
        B, T, H = dhidden.shape
        flat_dh = dhidden.reshape(B * T, H)                 # [B*T, H]
        flat_mask = mask.reshape(B * T)                     # [B*T]
        flat_ids = local_ids_safe.reshape(B * T)            # [B*T]
        # Zero out dhidden for off-rank rows. The index_add_ would
        # otherwise write those rows to the clamped index (0 or
        # vp-1), corrupting the local weight gradient.
        masked = flat_dh * flat_mask.unsqueeze(-1).to(flat_dh.dtype)
        dw = masked.new_zeros(ctx.vp, H)                    # [vp, H]
        # index_add_ accumulates: dw[flat_ids[i], :] += masked[i, :]
        # for each i. Multiple (b, t) pairs with the same token
        # naturally sum (correct gradient for a token that appears
        # multiple times in the sequence).
        dw.index_add_(0, flat_ids, masked)
        return dw, None, None, None


class TPShardedEmbed(nn.Embedding):
    """``nn.Embedding`` with the vocab dim sharded across the TP group.

    Memory characteristics (vs. the replicated v0.0.0 embed)
    --------------------------------------------------------

    Replicated (V=248320, H=1024, FP16):
      * embed param:  V * H * 2 B = 508 MB per device, duplicated
        across world devices (waste = (world - 1) * 508 MB).
      * AdamW state:  V * H * 2 * 2 B = 2032 MB per device
        (BF16 m + BF16 v on the embed's optimizer group).

    Sharded (world=8, vp = V / 8 = 31040):
      * embed param:  vp * H * 2 B = 64 MB per device (no
        duplication).
      * AdamW state:  vp * H * 2 * 2 B = 256 MB per device.
      * Per-forward comm: one all-reduce of [B, T, H] = 8 MB
        (the masked partial embeddings are summed to the full
        hidden).

    Net saving at TP=8: 7 * 508 + 7 * 256 = 5352 MB across the
    node, or 669 MB per device.

    Forward (per rank)
    ------------------
    1. Masked lookup: ``weight[clamp(input_ids - start, 0, vp-1)]``
       multiplied by a [B, T] bool mask (True only for tokens in
       this rank's vocab shard).
    2. ``tp_all_reduce_sum`` of the masked [B, T, H] partial
       embeddings. The result is the full [B, T, H] hidden, on
       every rank.

    Backward
    --------
    No inter-rank comm. Each rank scatters dhidden into the local
    weight shard via ``index_add_``; off-rank rows have mask=False
    so they contribute zero.

    Interaction with tied lm_head
    -----------------------------
    The matmul ``hidden @ weight.T`` is naturally a column-parallel
    matmul on the local [vp, H] shard: the result is [N, vp]
    sharded logits, which is exactly what :class:`TPFusedLceLoss`
    expects. The FusedLinearCE's ``dw`` is a [vp, H] gradient that
    goes into ``weight.grad`` (the local shard's grad) via the
    custom autograd in :class:`_TiedFusedLCEFunction`. PyTorch's
    autograd sums the FusedLinearCE dw and the lookup backward dw
    (both are gradients for the same parameter from different
    paths) into ``weight.grad`` at the end of backward.
    """

    def __init__(self, vocab_size: int, hidden_size: int, *args, **kwargs) -> None:
        # Save the LOGICAL vocab size for ``forward`` to compute
        # the per-token rank routing. The parent's
        # ``num_embeddings`` is the local shard size.
        self.logical_vocab_size = vocab_size
        self.world = get_tp_world_size()
        assert vocab_size % self.world == 0, (
            f"vocab_size={vocab_size} not divisible by tp_world={self.world}"
        )
        self.vocab_per_partition = vocab_size // self.world
        self.rank = get_tp_rank()
        self._start = self.rank * self.vocab_per_partition
        # Initialize the parent ``nn.Embedding`` with the LOCAL
        # shard size. The weight has shape [vp, H] and lives on
        # this rank's device.
        super().__init__(self.vocab_per_partition, hidden_size, *args, **kwargs)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return _ShardedEmbedLookup.apply(
            self.weight, input_ids, self._start, self.vocab_per_partition
        )

    def extra_repr(self) -> str:
        return (
            f"logical_vocab_size={self.logical_vocab_size}, "
            f"vocab_per_partition={self.vocab_per_partition}, "
            f"hidden_size={self.embedding_dim}, "
            f"tp_world={self.world}, tp_rank={self.rank}, "
            f"shard_start={self._start}"
        )


# --------------------------------------------------------------------------- #
# TP SwiGLU                                                                   #
# --------------------------------------------------------------------------- #
class TPSwiGLU(nn.Module):
    """Column-row parallel SwiGLU.

    gate_proj:  hidden  -> intermediate (column parallel, sharded)
    up_proj:    hidden  -> intermediate (column parallel, sharded)
    down_proj:  intermediate (sharded) -> hidden (row parallel, all-reduce)
    """

    def __init__(self, config, device=None, dtype=None) -> None:
        super().__init__()
        # Fused gate+up projection: one ColumnParallelLinear with
        # output = 2 * intermediate_size. The first ``intermediate``
        # output channels are the gate, the next ``intermediate``
        # are the up. This halves the matmul kernel-launch count
        # vs. two separate ColumnParallelLinear's (3 -> 2 per
        # SwiGLU). Bias: when ``use_bias`` is True the fused bias
        # is one tensor of size ``2 * intermediate_per_partition``
        # on this rank (gate half then up half); we expose it as
        # ``self.gate_up_bias`` so the forward can split it.
        self.gate_up_proj = ColumnParallelLinear(
            config.hidden_size, 2 * config.intermediate_size,
            bias=config.use_bias, device=device, dtype=dtype,
        )
        self.down_proj = RowParallelLinear(
            config.intermediate_size, config.hidden_size,
            bias=config.use_bias, device=device, dtype=dtype,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is replicated (full hidden). gate_up_proj produces
        # sharded intermediate slices on this rank: the first
        # ``inter_per_partition`` channels are the gate, the next
        # ``inter_per_partition`` are the up. down is row-parallel
        # and all-reduces back to full hidden.
        gu = self.gate_up_proj(x)
        inter_per_partition = self.gate_up_proj.out_features_per_partition // 2
        gate, up = gu.split(inter_per_partition, dim=-1)
        return self.down_proj(F.silu(gate) * up)


# --------------------------------------------------------------------------- #
# TP fused linear + cross-entropy (replaces TPLmHead + FusedCrossEntropyLoss)  #
# --------------------------------------------------------------------------- #
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
        self.embed_tokens = embed_tokens
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
        self.embed_tokens = embed_tokens
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


# --------------------------------------------------------------------------- #
# TP Gated DeltaNet 2 (GDN2)                                                  #
# --------------------------------------------------------------------------- #
class TPGDN2(nn.Module):
    """TP-sharded Gated DeltaNet 2.

    Direct successor to the (now removed) TPKDA class. The only
    structural change versus the KDA version is that the scalar
    beta gate is split into two channel-wise gates (``b`` on the
    key axis, ``w`` on the value axis) and that the kernel call
    passes ``b`` and ``w`` separately to :func:`chunk_gdn2`.

    Per-rank layout (world = TP group size):

    - ``qkv_proj``: column-parallel on ``2*key_dim + value_dim``.
      Each rank: ``[2*key_dim + value_dim // world, hidden_size]``
      (see fused QKV below).
    - ``q_conv1d`` / ``k_conv1d`` / ``v_conv1d``: depthwise conv
      with per-channel filter; each rank has the channels in its
      head slice. Shape: ``[c_per_rank, 1, conv_size]``.
    - ``f_proj1``: column-parallel ``head_v_dim → key_dim``.
      The first f_proj layer (``fg_first``) is replicated because
      its output (``head_v_dim``) is a per-head bottleneck that
      is the same on every rank.
    - ``g_proj1``: column-parallel ``head_v_dim → value_dim``.
      Same replicated-first-layer story as f_proj.
    - ``b_proj``: column-parallel on ``key_dim`` (NEW vs KDA:
      erase gate is per-key-channel, not per-v-head-scalar).
    - ``w_proj``: column-parallel on ``value_dim`` (NEW for GDN2:
      channel-wise write gate).
    - ``A_log``: ``[num_heads // world]`` FP32.
    - ``dt_bias``: ``[key_dim // world]`` FP32 (matches KDA's
      gate_dim since ``num_v_heads == num_heads`` in our setup).
    - ``o_norm``: per-head RMSNorm on ``head_v_dim``; weight
      ``[head_v_dim]`` (replicated within each rank's head slice).
    - ``o_proj``: row-parallel on ``value_dim``; all-reduces the
      output to ``[hidden_size]``.

    The chunk-GDN2 kernel itself is rank-local: each rank runs the
    full chunk algorithm on its ``H // world`` heads with a
    rank-local recurrent state. No inter-rank communication is
    required inside the GDN2 op — only the o_proj all-reduce.
    """

    def __init__(self, config, layer_idx: int, device=None, dtype=None) -> None:
        super().__init__()
        from src.models.ops._vendored.fla.modules import FusedRMSNormGated, ShortConvolution
        from src.models.ops._vendored.fla.ops.gdn2 import chunk_gdn2

        self._chunk_gdn2 = chunk_gdn2  # cache for forward

        self.world = get_tp_world_size()
        self.rank = get_tp_rank()

        h = config.hidden_size
        nh = config.num_heads
        nvh = config.num_heads  # no GVA
        d_h = config.head_dim
        d_v = int(d_h * config.expand_v)
        key_dim = nh * d_h
        value_dim = nvh * d_v

        assert nh % self.world == 0, f"num_heads={nh} not divisible by tp_world={self.world}"
        assert nvh % self.world == 0, f"num_v_heads={nvh} not divisible by tp_world={self.world}"
        assert value_dim % self.world == 0, f"value_dim={value_dim} not divisible by tp_world={self.world}"
        assert key_dim % self.world == 0, f"key_dim={key_dim} not divisible by tp_world={self.world}"

        self.hpp = nh // self.world           # heads per partition (qk and v share this)
        self.key_per_partition = key_dim // self.world
        self.value_per_partition = value_dim // self.world
        self.head_k_dim = d_h
        self.head_v_dim = d_v
        self.num_heads = nh
        self.num_v_heads = nvh
        self.hidden_size = h
        self.expand_v = config.expand_v
        self.mode = config.gdn2_mode
        self.use_short_conv = config.use_short_conv
        self.conv_size = config.conv_size
        self.conv_bias = config.conv_bias
        self.allow_neg_eigval = config.allow_neg_eigval
        # GDN2 has no ``safe_gate`` / ``lower_bound`` knobs; the
        # gate path is the standard softplus-on-pre-activation
        # scaled by ``-exp(A_log)``. The KDA-side config fields
        # were removed from HippoConfig in the KDA → GDN2 swap.
        self.layer_idx = layer_idx
        self.norm_eps = config.rms_norm_eps

        # ---- Fused QKV projection (column-parallel) ---- #
        # One ColumnParallelLinear with output = 2*key_dim + value_dim.
        # Output channels: [0:key_per_partition] = Q,
        # [key_per_partition:2*key_per_partition] = K,
        # [2*key_per_partition:2*key_per_partition+value_per_partition] = V.
        # Saves 2 kernel launches per GDN2 layer (3 -> 1).
        self.qkv_proj = ColumnParallelLinear(
            h, 2 * key_dim + value_dim, bias=False,
            device=device, dtype=dtype,
        )

        # ---- Depthwise conv1d (per-channel, sharded) ---- #
        if config.use_short_conv:
            self.q_conv1d = ShortConvolution(
                hidden_size=self.key_per_partition, kernel_size=config.conv_size,
                bias=config.conv_bias, activation="silu",
            ).to(device=device, dtype=dtype)
            self.k_conv1d = ShortConvolution(
                hidden_size=self.key_per_partition, kernel_size=config.conv_size,
                bias=config.conv_bias, activation="silu",
            ).to(device=device, dtype=dtype)
            self.v_conv1d = ShortConvolution(
                hidden_size=self.value_per_partition, kernel_size=config.conv_size,
                bias=config.conv_bias, activation="silu",
            ).to(device=device, dtype=dtype)

        # ---- Fused FG first layer + split FG second layer ---- #
        # The first layer of each is ``hidden → head_v_dim`` where
        # ``head_v_dim`` is a per-head bottleneck (the same on every
        # rank), so it's a vanilla ``nn.Linear`` (replicated). We
        # fuse the two first layers into one ``nn.Linear`` with
        # output = 2*d_v (saves 1 kernel launch per GDN2 layer).
        #
        # The second layers are NOT fused: packing f and g into one
        # big ``ColumnParallelLinear(2*d_v, key_dim + value_dim)``
        # requires a weight matrix with 50% zeros (the off-diagonal
        # blocks). The fused matmul would execute those zero*input
        # FMAs, doubling the FLOPs of this layer with no compute
        # benefit. So we keep f_proj[1] and g_proj[1] as separate
        # ColumnParallelLinear's. (Same reasoning as the KDA
        # implementation — see test_fused_proj_bench.py.)
        self.fg_first = nn.Linear(
            h, 2 * d_v, bias=False, device=device, dtype=dtype,
        )
        self.f_proj1 = ColumnParallelLinear(
            d_v, key_dim, bias=False, device=device, dtype=dtype,
        )
        self.g_proj1 = ColumnParallelLinear(
            d_v, value_dim, bias=True, device=device, dtype=dtype,
        )

        # ---- b_proj (NEW vs KDA shape): channel-wise erase gate on K axis ---- #
        self.b_proj = ColumnParallelLinear(h, key_dim, bias=False, device=device, dtype=dtype)
        # ---- w_proj: channel-wise write gate on V axis (NEW for GDN2) ---- #
        self.w_proj = ColumnParallelLinear(h, value_dim, bias=False, device=device, dtype=dtype)

        # ---- A_log, dt_bias: per-rank shards ---- #
        # A_log: per-QK-head decay rate (FP32, replicated as log-uniform init).
        import math as _math
        self.A_log = nn.Parameter(
            torch.log(
                torch.empty(self.hpp, dtype=torch.float32, device=device).uniform_(1, 16)
            )
        )
        self.A_log._no_weight_decay = True
        # dt_bias: per-key-channel softplus bias (FP32). Size matches
        # the per-rank ``key_per_partition``.
        dt = torch.exp(
            torch.rand(self.key_per_partition, dtype=torch.float32, device=device) *
            (_math.log(0.1) - _math.log(0.001)) + _math.log(0.001)
        ).clamp(min=1e-4)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)
        self.dt_bias._no_weight_decay = True

        # ---- o_norm: per-head RMSNorm on head_v_dim ---- #
        # weight has shape [head_v_dim] = [d_v]; same value applied
        # to each head on this rank. Replicated within rank.
        self.o_norm = FusedRMSNormGated(
            d_v, activation="sigmoid", eps=self.norm_eps,
            device=device, dtype=dtype,
        )

        # ---- o_proj: row-parallel on value_dim ---- #
        self.o_proj = RowParallelLinear(
            value_dim, h, bias=False, device=device, dtype=dtype,
        )

    def forward(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass. ``x`` is the full hidden (replicated
        across the TP group). Returns full hidden (after the
        ``o_proj`` all-reduce).

        When ``cu_seqlens`` is supplied the batch dim is folded
        into the sequence dim so the GDN2 chunkwise kernel sees
        ``batch_size=1`` (its hard requirement when varlen state
        reset is in use). All projections run on the flattened
        ``[1, B*T, hidden]`` tensor and the output is reshaped
        back to ``[B, T, hidden]`` before ``o_proj``.
        """
        from einops import rearrange

        # Packed path: fold batch dim into seq dim. The QKV /
        # FG / gate projections are just matmuls and don't care
        # about the leading-dim interpretation; only the chunkwise
        # recurrence cares, and it requires ``B=1`` when
        # ``cu_seqlens`` is set.
        B, T, hidden = x.shape
        if cu_seqlens is not None:
            x_in = x.reshape(1, B * T, hidden)
        else:
            x_in = x

        # Fused QKV: one matmul produces [B, T, 2*key+v] which
        # we split into Q / K / V per rank. Saves 2 kernel
        # launches vs. 3 separate projections.
        qkv = self.qkv_proj(x_in)
        q, k, v = qkv.split(
            [self.key_per_partition, self.key_per_partition, self.value_per_partition],
            dim=-1,
        )

        if self.use_short_conv:
            # Depthwise conv1d on each rank's channel slice.
            # The kernel expects [B, T, C] -> [B, T, C] with
            # causal padding, applied per-channel.
            # ``ShortConvolution.forward`` returns ``(y, cache)``;
            # we discard the cache because training uses the
            # chunk path (no incremental decoding). When
            # ``cu_seqlens`` is set the vendored
            # :class:`ShortConvolution` resets the causal state at
            # each doc boundary so a depthwise conv kernel does
            # not blur tokens across an unrelated document.
            q, _ = self.q_conv1d(q, cu_seqlens=cu_seqlens)
            k, _ = self.k_conv1d(k, cu_seqlens=cu_seqlens)
            v, _ = self.v_conv1d(v, cu_seqlens=cu_seqlens)
        else:
            q = F.silu(q)
            k = F.silu(k)
            v = F.silu(v)

        # FG path: one nn.Linear (``fg_first``) produces the
        # concatenated f/g intermediates; two separate
        # ColumnParallelLinear's (``f_proj1``, ``g_proj1``) project
        # each to its output. We keep the second layers split
        # because fusing them would require a 50%-zero weight
        # matrix and double the FLOPs of this layer (see comment
        # in __init__).
        fg_first = self.fg_first(x_in)
        f_inter, g_inter = fg_first.chunk(2, dim=-1)
        g = self.f_proj1(f_inter)
        g_for_norm = self.g_proj1(g_inter)

        # Channel-wise gates (NEW vs KDA). Both squashed to [0, 1]:
        #   b: erase gate on the K axis (replaces KDA's scalar beta)
        #   w: write gate on the V axis (NEW for GDN2)
        b = self.b_proj(x_in).sigmoid()
        w = self.w_proj(x_in).sigmoid()

        # Reshape to per-head: each rank has ``hpp`` heads.
        # q, k: [B, T, hpp, head_k_dim]
        # g:    [B, T, hpp, head_k_dim]  (key_per_partition = hpp*head_k_dim)
        # v:    [B, T, hpp, head_v_dim]
        # b:    [B, T, hpp, head_k_dim]
        # w:    [B, T, hpp, head_v_dim]
        q = rearrange(q, "... (h d) -> ... h d", d=self.head_k_dim)
        k = rearrange(k, "... (h d) -> ... h d", d=self.head_k_dim)
        g = rearrange(g, "... (h d) -> ... h d", d=self.head_k_dim)
        v = rearrange(v, "... (h d) -> ... h d", d=self.head_v_dim)
        b = rearrange(b, "... (h d) -> ... h d", d=self.head_k_dim)
        w = rearrange(w, "... (h d) -> ... h d", d=self.head_v_dim)

        if self.allow_neg_eigval:
            b = b * 2.0

        # chunk_gdn2: rank-local. No TP comm inside the kernel.
        # ``use_gate_in_kernel=True`` makes the kernel compute the
        # softplus(g + dt_bias) * -exp(A_log) activation internally
        # (saves a separate activation kernel launch).
        # ``cu_seqlens`` flows straight through to the chunkwise
        # recurrence when set.
        o, _ = self._chunk_gdn2(
            q=q, k=k, v=v, g=g, b=b, w=w,
            A_log=self.A_log, dt_bias=self.dt_bias,
            initial_state=None, output_final_state=False,
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True,
            cu_seqlens=cu_seqlens,
        )

        # o has shape ``x_in.shape[:-1] + (hpp, head_v_dim)``.
        # Reshape back to ``[B, T, hpp, head_v_dim]`` when the
        # packed path flattened the leading dims.
        if cu_seqlens is not None:
            o = o.reshape(B, T, self.hpp, self.head_v_dim)
            g_for_norm = g_for_norm.reshape(B, T, self.hpp, self.head_v_dim)

        # o_norm: per-head RMSNorm on the last dim.
        g_for_norm = rearrange(g_for_norm, "... (h d) -> ... h d", d=self.head_v_dim)
        o = self.o_norm(o, g_for_norm)
        o = rearrange(o, "b t h d -> b t (h d)")
        # o_proj: row-parallel, all-reduces to [B, T, hidden_size].
        o = self.o_proj(o)
        return o


# --------------------------------------------------------------------------- #
# TP HippoModel                                                               #
# --------------------------------------------------------------------------- #
class TPHippoLayer(nn.Module):
    """TP version of HippoLayer. GDN2 is sharded along the head dim
    via :class:`TPGDN2`; FFN is sharded via :class:`TPSwiGLU`.

    AttnRes no longer lives inside each layer. The block-boundary
    AttnRes is a single per-device replicated module
    (``TPHippoModel.replicated_per_device[d]["attn_res"]``);
    it is invoked once per non-first block to compute the next
    block's input. Inside a block the residual is standard
    ``x = x + SubLayer(x)``.
    """

    def __init__(self, layer_idx: int, config, device=None, dtype=None) -> None:
        super().__init__()
        self.layer_idx = layer_idx

        # NB: ``device`` may be 0 (an int) which is falsy in Python
        # — guard with ``is not None`` to avoid the conditional
        # silently falling through to the no-op branch.
        def _to(m: nn.Module) -> nn.Module:
            if device is None:
                return m
            if dtype is not None:
                return m.to(device=device, dtype=dtype)
            return m.to(device=device)

        self.attn_norm = _to(RMSNorm(config.hidden_size, eps=config.rms_norm_eps))
        self.mlp_norm = _to(RMSNorm(config.hidden_size, eps=config.rms_norm_eps))
        self.gdn2 = _to(TPGDN2(config, layer_idx=layer_idx))
        self.ffn = TPSwiGLU(config, device=device, dtype=dtype)

    def forward(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Standard residual transformer layer on the local device.

        Args:
            x: ``[B, T, hidden_size]`` replicated hidden state.
            cu_seqlens: optional ``[total_docs + 1]`` long tensor
                with global offsets across the flattened
                ``[batch_size * seq_len]`` sequence. Forwarded
                to the GDN2 sub-layer only; FFN / RMSNorm do not
                depend on the doc layout.

        Returns:
            ``[B, T, hidden_size]`` after GDN2 and FFN with
            standard residual connections.
        """
        x = x + self.gdn2(self.attn_norm(x), cu_seqlens=cu_seqlens)
        x = x + self.ffn(self.mlp_norm(x))
        return x


class TPHippoModel(nn.Module):
    """Megatron-style TP version of HippoModel.

    The model is fully replicated on each rank except for:
    - SwiGLU FFN (column-row parallel)
    - lm_head (column parallel)

    Construction places the replicated params on ``devices[0]`` only
    (the master device); the sharded FFN/lm_head params are placed
    on each rank's local device. The forward runs each rank on its
    own device using its own parameters.
    """

    def __init__(self, config, devices: list[int], dtype: torch.dtype | None = None) -> None:
        super().__init__()
        self.config = config
        self.devices = list(devices)
        self.world = len(self.devices)
        assert self.world == get_tp_world_size(), (
            f"TP world size mismatch: devices={self.world} vs group={get_tp_world_size()}"
        )
        if dtype is None:
            # Default to FP16. The model is intended to run with
            # FP16 forward/backward (V100 tensor cores); the loss
            # reduction is FP32 inside forward().
            dtype = torch.float16

        master = self.devices[0]
        # Replicated modules live on the master device. The forward
        # will move them to the local device for each rank via
        # ``.to(local_device)`` no — the simpler trick is to
        # construct one copy per device, but then we'd have N copies
        # in state_dict. Instead, we use ``register_parameter`` with
        # the same shared storage across ranks.
        #
        # PyTorch doesn't natively support cross-device parameter
        # aliasing inside a single nn.Module, so we instead use a
        # simpler design: each rank constructs its own copy of
        # the replicated modules, and we keep them in sync via
        # broadcast at the start of each forward (handled in the
        # training script, not here).
        #
        # For the v0.0.0 validation the simplest correct approach
        # is: build a full copy of the replicated portion on each
        # device at construction time. The replicated portion's
        # memory cost is dominated by embed_tokens (~508 MB FP16
        # per device) and KDA weights (~14 MB/layer * 32 = 448 MB
        # per device). Total replicated per device: ~1 GB, which
        # fits comfortably in 16 GB.
        self.replicated_per_device: dict[int, dict[str, nn.Module]] = {}
        for d in self.devices:
            # Cast the embedding to the training dtype at construction
            # time. Otherwise ``.to(device=...)`` only moves and
            # leaves the weight at FP32, defeating the FP16 path.
            #
            # The embed is **vocab-sharded** across the TP group:
            # each rank holds ``[vocab_size // world, hidden_size]``
            # rows of the logical embed. The forward does a masked
            # lookup + all-reduce to produce the full [B, T, H]
            # hidden on every rank (see :class:`TPShardedEmbed`).
            # Memory savings at TP=8, V=248320, H=1024, FP16:
            # 7 * 508 MB of duplicate embed params + 7 * 2032 MB
            # of duplicate AdamW state = ~17.7 GB saved across the
            # node.
            device_mods = {
                "embed_tokens": TPShardedEmbed(
                    config.vocab_size, config.hidden_size,
                ).to(device=d, dtype=dtype),
                "norm": RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
                .to(device=d, dtype=dtype),
                # Block-boundary AttnRes: per-device, replicated.
                # All-gathers the head dim across the TP group on
                # every forward. Pseudo-query is sharded along
                # ``num_heads`` (each rank holds ``num_heads //
                # world`` heads).
                "attn_res": BlockAttnRes(config).to(device=d, dtype=dtype),
            }
            device_mods["embed_tokens"] = self._init_embed(device_mods["embed_tokens"])
            self.replicated_per_device[d] = device_mods

        # Sharded modules: one copy per device, with sharded weights.
        self.layers_per_device: dict[int, nn.ModuleList] = {}
        for d in self.devices:
            layers = nn.ModuleList([
                TPHippoLayer(i, config, device=d, dtype=dtype) for i in range(config.num_layers)
            ])
            self.layers_per_device[d] = layers
            # Replace embedding-init for embedding table on this device
            nn.init.normal_(self.replicated_per_device[d]["embed_tokens"].weight, mean=0.0, std=0.02)

        # lm_head: column-parallel on each device, tied to the
        # local ``embed_tokens`` so the underlying weight storage
        # is shared with the embedding (no separate ``[vocab/world,
        # hidden]`` allocation per rank — saves 254 MB per rank at
        # TP=2 vs the previous untied design). The optimizer sees
        # only the embedding's weight (deduplicated by ``id()``)
        # and updates the parent; the gradient on the head's view
        # is summed into ``embed_tokens.weight.grad`` via PyTorch's
        # standard view graph, which is exactly the
        # ``tie_word_embeddings=True`` semantics.
        #
        # The matmul ``hidden @ embed.T`` is fused inside the
        # :class:`TPFusedLceLoss` below (chunks over the sequence
        # dim, online softmax, no [B,T,V] tensor materialised),
        # so the lm_head is no longer called separately on the
        # training path. The :class:`TPLmHead` shim is still
        # constructed here for inference / generation paths that
        # need the full sharded logits (see ``model.generate``).
        self.lm_head_per_device: dict[int, TPLmHead] = {}
        for d in self.devices:
            head = TPLmHead(
                self.replicated_per_device[d]["embed_tokens"],
                config.hidden_size, config.vocab_size,
                bias=config.use_bias, device=d, dtype=dtype,
            )
            self.lm_head_per_device[d] = head

        # Fused lm_head + cross-entropy, TP-aware, online chunked
        # softmax. Replaces the previous TPLmHead +
        # FusedCrossEntropyLoss pair. The full [B, T, V // world]
        # sharded logits tensor is never materialised: the matmul
        # is done inside the fla kernel over chunks of the
        # sequence dim, with per-chunk size set so peak logits
        # ≈ 32 MB at our config (vs ~254 MB with the fla default
        # of num_chunks=8). See :class:`TPFusedLceLoss` for the
        # full memory analysis.
        #
        # With the vocab-sharded embed (see :class:`TPShardedEmbed`),
        # the FusedLinearCE matmul is naturally column-parallel:
        # ``hidden @ local_embed.T`` produces [N, vp] sharded
        # logits, and the FusedLinearCE's dw is the gradient for
        # the local [vp, H] weight shard. We pass ``start_offset=0``
        # and ``local_vocab_size=vp`` so the custom autograd adds
        # the dw to the full local grad (not a slice of a
        # replicated full [V, H] grad, which would be the legacy
        # behavior).
        self.fused_lm_ce_per_device: dict[int, TPFusedLceLoss] = {}
        for d in self.devices:
            local_embed = self.replicated_per_device[d]["embed_tokens"]
            local_vp = local_embed.weight.size(0)
            flce = TPFusedLceLoss(
                local_embed,
                config.hidden_size, config.vocab_size,
                bias=config.use_bias,
                ignore_index=-100,
                # 64 chunks → C = next_pow2(ceil(4092/64)) = 64
                # tokens per chunk → per-chunk peak logits
                # 64 × 248320 × 2 B = ~32 MB at V=248k, H=1k.
                num_chunks=64,
                device=d, dtype=dtype,
                # Vocab-sharded embed: the local weight is the
                # full [vp, H] shard, no narrow needed for the
                # matmul, and the FusedLinearCE dw goes into the
                # full local grad.
                start_offset=0,
                local_vocab_size=local_vp,
            )
            self.fused_lm_ce_per_device[d] = flce

        # Per-device trainable parameter list. Built lazily by
        # ``trainable_parameters(device)``.
        self._trainable_cache: dict[int, list[nn.Parameter]] = {}

    def _init_embed(self, emb: nn.Embedding) -> nn.Embedding:
        nn.init.normal_(emb.weight, mean=0.0, std=0.02)
        return emb

    def trainable_parameters(self, device: int) -> list[nn.Parameter]:
        """Return all trainable nn.Parameters that live on ``device``.

        Used by the per-device optimizer factory to build AdamW/Muon
        groups.
        """
        if device in self._trainable_cache:
            return self._trainable_cache[device]
        params: list[nn.Parameter] = []
        # Replicated modules on this device
        for m in self.replicated_per_device[device].values():
            params.extend(p for p in m.parameters() if p.requires_grad)
        # Sharded layers
        for layer in self.layers_per_device[device]:
            params.extend(p for p in layer.parameters() if p.requires_grad)
        # lm_head
        params.extend(p for p in self.lm_head_per_device[device].parameters() if p.requires_grad)
        self._trainable_cache[device] = params
        return params

    def named_parameters_per_device(self, device: int):
        """Yield (name, parameter) for everything on ``device``."""
        seen: set[int] = set()
        # Replicated
        for mname, mod in self.replicated_per_device[device].items():
            for pname, p in mod.named_parameters(recurse=True):
                full = f"replicated.{device}.{mname}.{pname}"
                if id(p) in seen:
                    continue
                seen.add(id(p))
                yield full, p
        # Sharded layers
        for li, layer in enumerate(self.layers_per_device[device]):
            for pname, p in layer.named_parameters(recurse=True):
                full = f"layer.{li}.{pname}"
                if id(p) in seen:
                    continue
                seen.add(id(p))
                yield full, p
        # lm_head
        for pname, p in self.lm_head_per_device[device].named_parameters(recurse=True):
            full = f"lm_head.{pname}"
            if id(p) in seen:
                continue
            seen.add(id(p))
            yield full, p

    def sync_replicated_from(self, src_device: int) -> None:
        """Broadcast replicated-module params from ``src_device`` to
        every other device. Run once after construction so the
        KDA / embed / norm params on every device start from the
        same random init (otherwise each device's independent
        ``xavier_uniform_`` would diverge, which doesn't matter
        for OOM testing but does matter for correctness).

        The vocab-sharded :class:`TPShardedEmbed` is explicitly
        skipped: each rank holds a different vocab shard, and
        broadcasting ``src_device``'s shard to every other rank
        would corrupt the per-rank shards (every rank would end
        up with the same vocab rows). The sharded embed is
        intentionally initialised independently on each device
        (each rank gets a different slice of the random init).
        """
        # 1) Top-level replicated modules (norm, attn_res, and
        # embed_tokens if it happens to be a plain nn.Embedding
        # rather than a TPShardedEmbed). AttnRes is shared across
        # all boundaries so the single module is replicated
        # exactly once per device.
        for mname, src_mod in self.replicated_per_device[src_device].items():
            # Skip the sharded embed: each rank holds its own
            # [vp, H] shard and a broadcast from src_device would
            # collapse the shards together (every rank would
            # receive the same vocab rows).
            if isinstance(src_mod, TPShardedEmbed):
                continue
            src_params = dict(src_mod.named_parameters(recurse=True))
            for dst_device, dst_mods in self.replicated_per_device.items():
                if dst_device == src_device:
                    continue
                dst_params = dict(dst_mods[mname].named_parameters(recurse=True))
                for pname, p in src_params.items():
                    dst_params[pname].data.copy_(p.data)

        # 2) Per-layer replicated sub-modules (GDN2, attn_norm,
        # mlp_norm). AttnRes is no longer per-layer; the block
        # boundary version lives at the model level and is
        # already synced above.
        for li, src_layer in enumerate(self.layers_per_device[src_device]):
            for sub_name in ("gdn2", "attn_norm", "mlp_norm"):
                src_sub = getattr(src_layer, sub_name)
                src_params = dict(src_sub.named_parameters(recurse=True))
                for dst_device, dst_layers in self.layers_per_device.items():
                    if dst_device == src_device:
                        continue
                    dst_sub = getattr(dst_layers[li], sub_name)
                    dst_params = dict(dst_sub.named_parameters(recurse=True))
                    for pname, p in src_params.items():
                        dst_params[pname].data.copy_(p.data)

    def _block_forward(
        self,
        block_layers,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run a contiguous group of ``len(block_layers)`` layers
        (one block) on the local device and return the block's
        output. Used as the body of
        :func:`torch.utils.checkpoint.checkpoint` for completed
        blocks so we only keep the per-block output b_n for
        backward, not the per-layer activations.

        The residual inside the block is the standard
        ``x = x + SubLayer(x)``; there is no per-layer AttnRes
        call. AttnRes is invoked at the block boundary by the
        outer :meth:`forward` loop, not inside this body.

        ``cu_seqlens`` is forwarded to every layer's GDN2 sub-
        layer (FFN / RMSNorm ignore it). When this body runs
        inside ``checkpoint.checkpoint`` the cu_seqlens tensor
        is saved with the recomputation inputs automatically.
        """
        for layer in block_layers:
            x = layer(x, cu_seqlens=cu_seqlens)
        return x

    def forward(
        self,
        input_ids: torch.Tensor,  # [B, T], lives on this rank's device
        labels: torch.Tensor | None = None,  # [B, T], same device
        cu_seqlens: torch.Tensor | None = None,  # [total_docs+1]
    ) -> dict[str, torch.Tensor]:
        """Run the full TP model forward.

        ``cu_seqlens`` is the global offset tensor produced by
        :func:`pack_chunk_aligned` — see that function's docstring
        for the contract. When set the GDN2 sub-layer inside every
        block resets the recurrent state at each ``cu_seqlens``
        boundary; the rest of the model (FFN, RMSNorm, embed,
        BlockAttnRes, lm_head) ignores it. The hidden state shape
        is unchanged (``[B, T, hidden]`` on every rank) — only
        the GDN2 sub-layer flattens / unflattens internally.
        """
        device = input_ids.device.index if input_ids.device.type == "cuda" else input_ids.device
        embeds = self.replicated_per_device[device]["embed_tokens"](input_ids)
        attn_res = self.replicated_per_device[device]["attn_res"]
        blocks: list[torch.Tensor] = [embeds]  # b_0 = embedding
        x = embeds  # Block 0's input is the embedding.

        # Block-level checkpointing + KDA cache in the last block
        # --------------------------------------------------------
        # The model has ``num_blocks`` blocks of ``block_size``
        # layers each. For every block except the last, we wrap
        # the entire 4-layer forward in a single
        # ``torch.utils.checkpoint.checkpoint`` call so that
        # backward recomputes the per-layer activations from the
        # block output b_n. Only the per-block output is held in
        # memory, which is roughly ``block_size``× cheaper than
        # the per-layer checkpointing we used to do. Inside the
        # checkpoint wrapper the KDA Triton kernels' intermediate
        # state (g_cumsum, Aqk, Akk, w, u, qg, kg, v_new, h) is
        # *not* retained — backward re-runs the chunked KDA
        # forward inside the checkpoint.
        #
        # The *last* block is intentionally NOT wrapped in
        # ``torch.utils.checkpoint.checkpoint``. KDA's autograd
        # graph benefits from not being checkpointed: the Triton
        # kernels cache their forward intermediates and the KDA
        # backward is strictly more expensive than re-doing the
        # chunked forward would be. So the last block pays the
        # activation memory (per-layer inputs to the residual)
        # in exchange for a fast backward — the KDA cache is
        # retained.
        #
        # AttnRes is invoked at every non-first block boundary,
        # *outside* the checkpoint wrapper, so the per-block
        # output ``b_n`` captures the boundary-attention input
        # ``x``. ``blocks`` (the list of completed block reps) is
        # held live across blocks but is small (one ``[B,T,D]``
        # per block — same order as before).
        layers = self.layers_per_device[device]
        block_size = self.config.block_size
        num_blocks = self.config.num_blocks
        for block_idx in range(num_blocks - 1):
            if block_idx > 0:
                x = attn_res(blocks)
            start = block_idx * block_size
            end = start + block_size
            block_layers = layers[start:end]
            x = torch.utils.checkpoint.checkpoint(
                self._block_forward, block_layers, x, cu_seqlens,
                use_reentrant=False, preserve_rng_state=False,
            )
            blocks.append(x)
        # Last block: per-layer checkpointing for every layer
        # except the very last one. The first n-1 blocks are
        # already block-level-checkpointed above (one
        # ``_block_forward`` re-run per block, ~7 recomputations
        # across the full 8-block model). For the last block we
        # want a finer granularity: 3 of the 4 layers get
        # wrapped in ``torch.utils.checkpoint.checkpoint`` with
        # ``use_reentrant=True`` so their recomputed GDN2
        # internals are freed as soon as that layer's backward
        # finishes (peak memory: 1 layer worth, not 4). The
        # very last layer is left un-checkpointed because its
        # GDN2 internals feed the lm_head + fused CE loss —
        # recomputing them would force a 1-forward-step
        # recompute on every backward, and the GDN2 backward
        # kernel is comparable in cost to a forward so caching
        # is the right trade. Net: 3 extra recomputations per
        # step (one per checkpointed layer in the last block),
        # saving ~3/4 of the last-block activation memory.
        last_block = num_blocks - 1
        if last_block > 0:
            x = attn_res(blocks)
        last_start = last_block * block_size
        last_block_layers = layers[last_start:]
        for layer in last_block_layers[:-1]:
            x = torch.utils.checkpoint.checkpoint(
                layer, x, cu_seqlens,
                use_reentrant=True, preserve_rng_state=False,
            )
        # Final layer of the last block: NO checkpoint wrapper.
        # Its GDN2 internals (q, k, v, b, w, Aqk, Akk, w_wy,
        # u_wy, qg, kg, v_new, h) are held for backward.
        x = last_block_layers[-1](x, cu_seqlens=cu_seqlens)

        hidden_states = self.replicated_per_device[device]["norm"](x)
        out: dict[str, torch.Tensor] = {}
        if labels is not None:
            # Fused lm_head + CE. The kernel:
            #   1. Slices hidden to drop the last time position
            #      (no next-token target for it) and labels to
            #      drop the first (the embedding at t=0 has no
            #      preceding context to predict from — the standard
            #      next-token-prediction shift).
            #   2. Does hidden[..., :-1, :] @ local_embed.T inside
            #      a Triton kernel that chunks over the sequence
            #      dim, so the per-chunk peak logits are
            #      ``[C, V // world]`` (≈32 MB at our config with
            #      num_chunks=64) instead of the full
            #      ``[B, T, V // world]`` (1.99 GB at TP=1).
            #   3. Computes an online chunked softmax over the
            #      vocab dim, accumulates the per-token loss, and
            #      returns a scalar ``mean`` loss.
            #   4. Pre-computes dx (grad w.r.t. hidden) and dw
            #      (grad w.r.t. the embed slice) in forward, so
            #      backward does not need to re-materialise the
            #      per-chunk logits.
            # The full [B, T, V // world] sharded logits tensor
            # is therefore never materialised on the device.
            shift_hidden = hidden_states[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = self.fused_lm_ce_per_device[device](shift_hidden, shift_labels)
            out["loss"] = loss
        else:
            # Inference / generation path: caller expects the
            # sharded logits. The TPLmHead shim above does the
            # narrow-view matmul; the caller (e.g. model.generate)
            # is responsible for any all-gather / argmax.
            sharded_logits = self.lm_head_per_device[device](hidden_states)
            out["logits"] = sharded_logits
        return out
