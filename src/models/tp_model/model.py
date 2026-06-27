"""TPHippoModel: Megatron-style TP version of HippoModel.

The model is fully replicated on each rank except for:
- SwiGLU FFN (column-row parallel)
- lm_head (column parallel)

Construction places the replicated params on ``devices[0]`` only
(the master device); the sharded FFN/lm_head params are placed
on each rank's local device. The forward runs each rank on its
own device using its own parameters.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from src.models.norms import RMSNorm
from src.models.ops.attn_res import BlockAttnRes
from src.models.tp_layers import get_tp_world_size

from .embed import TPShardedEmbed
from .layer import TPHippoLayer
from .lm_head import TPFusedLceLoss, TPLmHead


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
        # Gradient-checkpoint sub-block size for the non-last block
        # forward (see docs/gradient_checkpointing.md). 2 is the
        # sweet spot at the base config; sweep harness overrides
        # this per-build.
        self.sub_block_size = 2
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
        # NOTE: the four per-device containers below are nn.ModuleDict
        # (NOT plain dicts) so that PyTorch's ``state_dict()`` walk sees
        # every submodule and its parameters. A plain ``dict[int, ...]``
        # is invisible to ``_modules`` traversal, which silently drops
        # every model weight at save time — the bug fixed in June 2026.
        # ``nn.ModuleDict`` requires string keys, so all access uses
        # ``str(d)`` for the device index.
        self.replicated_per_device: nn.ModuleDict = nn.ModuleDict()
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
            device_mods = nn.ModuleDict({
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
            })
            device_mods["embed_tokens"] = self._init_embed(device_mods["embed_tokens"])
            self.replicated_per_device[str(d)] = device_mods

        # Sharded modules: one copy per device, with sharded weights.
        self.layers_per_device: nn.ModuleDict = nn.ModuleDict()
        for d in self.devices:
            layers = nn.ModuleList([
                TPHippoLayer(i, config, device=d, dtype=dtype) for i in range(config.num_layers)
            ])
            self.layers_per_device[str(d)] = layers
            # Replace embedding-init for embedding table on this device
            nn.init.normal_(self.replicated_per_device[str(d)]["embed_tokens"].weight, mean=0.0, std=0.02)

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
        self.lm_head_per_device: nn.ModuleDict = nn.ModuleDict()
        for d in self.devices:
            head = TPLmHead(
                self.replicated_per_device[str(d)]["embed_tokens"],
                config.hidden_size, config.vocab_size,
                bias=config.use_bias, device=d, dtype=dtype,
            )
            self.lm_head_per_device[str(d)] = head

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
        self.fused_lm_ce_per_device: nn.ModuleDict = nn.ModuleDict()
        for d in self.devices:
            local_embed = self.replicated_per_device[str(d)]["embed_tokens"]
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
            self.fused_lm_ce_per_device[str(d)] = flce

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
        for m in self.replicated_per_device[str(device)].values():
            params.extend(p for p in m.parameters() if p.requires_grad)
        # Sharded layers
        for layer in self.layers_per_device[str(device)]:
            params.extend(p for p in layer.parameters() if p.requires_grad)
        # lm_head
        params.extend(p for p in self.lm_head_per_device[str(device)].parameters() if p.requires_grad)
        self._trainable_cache[device] = params
        return params

    def named_parameters_per_device(self, device: int):
        """Yield (name, parameter) for everything on ``device``."""
        seen: set[int] = set()
        # Replicated
        for mname, mod in self.replicated_per_device[str(device)].items():
            for pname, p in mod.named_parameters(recurse=True):
                full = f"replicated.{device}.{mname}.{pname}"
                if id(p) in seen:
                    continue
                seen.add(id(p))
                yield full, p
        # Sharded layers
        for li, layer in enumerate(self.layers_per_device[str(device)]):
            for pname, p in layer.named_parameters(recurse=True):
                full = f"layer.{li}.{pname}"
                if id(p) in seen:
                    continue
                seen.add(id(p))
                yield full, p
        # lm_head
        for pname, p in self.lm_head_per_device[str(device)].named_parameters(recurse=True):
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
        for mname, src_mod in self.replicated_per_device[str(src_device)].items():
            # Skip the sharded embed: each rank holds its own
            # [vp, H] shard and a broadcast from src_device would
            # collapse the shards together (every rank would
            # receive the same vocab rows).
            if isinstance(src_mod, TPShardedEmbed):
                continue
            src_params = dict(src_mod.named_parameters(recurse=True))
            for dst_device, dst_mods in self.replicated_per_device.items():
                if dst_device == str(src_device):
                    continue
                dst_params = dict(dst_mods[mname].named_parameters(recurse=True))
                for pname, p in src_params.items():
                    dst_params[pname].data.copy_(p.data)

        # 2) Per-layer replicated sub-modules (KDA, attn_norm,
        # mlp_norm). AttnRes is no longer per-layer; the block
        # boundary version lives at the model level and is
        # already synced above.
        for li, src_layer in enumerate(self.layers_per_device[str(src_device)]):
            for sub_name in ("kda", "attn_norm", "mlp_norm"):
                src_sub = getattr(src_layer, sub_name)
                src_params = dict(src_sub.named_parameters(recurse=True))
                for dst_device, dst_layers in self.layers_per_device.items():
                    if dst_device == str(src_device):
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

        ``cu_seqlens`` is forwarded to every layer's KDA sub-
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
        for the contract. When set the KDA sub-layer inside every
        block resets the recurrent state at each ``cu_seqlens``
        boundary; the rest of the model (FFN, RMSNorm, embed,
        BlockAttnRes, lm_head) ignores it. The hidden state shape
        is unchanged (``[B, T, hidden]`` on every rank) — only
        the KDA sub-layer flattens / unflattens internally.
        """
        device = input_ids.device.index if input_ids.device.type == "cuda" else input_ids.device
        embeds = self.replicated_per_device[str(device)]["embed_tokens"](input_ids)
        attn_res = self.replicated_per_device[str(device)]["attn_res"]
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
        # Optimization (new): per-2-layer checkpoint with
        # use_reentrant=True. Splits each 4-layer non-last block
        # into 2 sub-blocks of 2 layers each, checkpointing each
        # sub-block. Trades block-level checkpoint (1 saved input
        # per 4 layers, all 4 layers' KDA state alive during
        # block bwd) for sub-block checkpoint (1 saved input per
        # 2 layers, 2 layers' KDA state alive during sub-block
        # bwd). Net: ~448 MB FWD increase (extra sub-block
        # inputs), ~1.0 GB BWD decrease (fewer KDA intermediates
        # in flight). Trade: 1 extra fwd per sub-block per bwd
        # (~50ms × 2 layers × 7 blocks × 2 sub-blocks = 1.4s/step).
        layers = self.layers_per_device[str(device)]
        block_size = self.config.block_size
        num_blocks = self.config.num_blocks
        # 2 layers per sub-block (sweet spot for L=32 / block_size=4
        # at the base config; sweep shows it dominates per-block and
        # per-layer at this L — see docs/gradient_checkpointing.md).
        # Exposed as an attribute so the sweep harness can vary it
        # without editing the file.
        sub_block_size = getattr(self, "sub_block_size", 2)
        for block_idx in range(num_blocks - 1):
            if block_idx > 0:
                x = attn_res(blocks)
            start = block_idx * block_size
            end = start + block_size
            for sub_start in range(start, end, sub_block_size):
                sub_end = min(sub_start + sub_block_size, end)
                sub_layers = layers[sub_start:sub_end]
                x = torch.utils.checkpoint.checkpoint(
                    self._block_forward, sub_layers, x, cu_seqlens,
                    use_reentrant=True, preserve_rng_state=False,
                )
            blocks.append(x)
        # Last block: per-layer checkpointing for every layer
        # except the very last one. The first n-1 blocks are
        # already block-level-checkpointed above (one
        # ``_block_forward`` re-run per block, ~7 recomputations
        # across the full 8-block model). For the last block we
        # want a finer granularity: 3 of the 4 layers get
        # wrapped in ``torch.utils.checkpoint.checkpoint`` with
        # ``use_reentrant=True`` so their recomputed KDA
        # internals are freed as soon as that layer's backward
        # finishes (peak memory: 1 layer worth, not 4). The
        # very last layer is left un-checkpointed because its
        # KDA internals feed the lm_head + fused CE loss —
        # recomputing them would force a 1-forward-step
        # recompute on every backward, and the KDA backward
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
        # Its KDA internals (q, k, v, beta, g, Aqk, Akk, w_wy,
        # u_wy, qg, kg, v_new, h) are held for backward.
        x = last_block_layers[-1](x, cu_seqlens=cu_seqlens)

        hidden_states = self.replicated_per_device[str(device)]["norm"](x)
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
            loss = self.fused_lm_ce_per_device[str(device)](shift_hidden, shift_labels)
            out["loss"] = loss
        else:
            # Inference / generation path: caller expects the
            # sharded logits. The TPLmHead shim above does the
            # narrow-view matmul; the caller (e.g. model.generate)
            # is responsible for any all-gather / argmax.
            sharded_logits = self.lm_head_per_device[str(device)](hidden_states)
            out["logits"] = sharded_logits
        return out
