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
from ._primitives import get_tp_world_size

from .embed import TPShardedEmbed
from .layer import TPHippoLayer
from .lm_head import TPFusedLceLoss, TPLmHead


class _BlockManualCkptFunction(torch.autograd.Function):
    """Block-level gradient checkpoint with **manual per-layer re-fwd
    in the backward phase**.

    Why this exists (rev 4 vs rev 2 / rev 3)
    ----------------------------------------
    ``torch.utils.checkpoint.checkpoint(use_reentrant=True)`` keeps
    ALL inner Function ``save_for_backward`` tensors alive across
    the entire re-fwd region:

      - **rev 2** (1 ckpt per block, 4 layers): re-fwd pops all 4
        layers' saves simultaneously → peak 4 × 1184 MiB = 4736 MiB
        at T=16384.
      - **rev 3** (2 sub-blocks of 2 each): each re-fwd pops 2
        layers' saves → peak 2 × 1184 = 2368 MiB (polling avg ~1.3
        layers).
      - **rev 1** (per-layer, 4 ckpts per block): peak 1 × 1184
        MiB but each ckpt input is held at fwd_out (+672 MiB × 4
        ckpts × 7 non-last blocks = ~+18.8 GiB total → wrong).

    This custom Function (``rev 4``) achieves the **per-layer peak
    (1 layer alive)** while keeping the **block-level fwd_out cost
    (1 ckpt boundary per block)**:

      - Forward: save only the per-layer INPUT tensors
        (5 × [B,T,D] = 120 MiB for a 4-layer block — negligible)
        and the detached KDA final states. Run the forward under
        ``torch.no_grad()`` so NO inner Function saves are
        retained.
      - Backward: walk the layers in REVERSE. For each layer K
        from N-1 down to 0: re-fwd layer K with grad enabled,
        backward through layer K's outputs, release its inner
        Function saves, then move to layer K-1. Peak inner
        Function saves alive at any time = 1 layer.

    See ``docs/gradient_checkpointing.md`` §"Block-level ckpt with
    manual per-layer re-fwd (rev 4)" for the per-bucket VRAM
    breakdown.
    """

    @staticmethod
    def forward(ctx, block_layers, x, cu_seqlens,
                layer_kda_states, return_states):
        # Persist non-tensor / non-grad args. ``block_layers`` is
        # held as a Python list of ``nn.Module`` references; the
        # Python objects are stable across the forward/backward
        # boundary, so we can safely index into them in bwd.
        ctx.block_layers = list(block_layers)
        ctx.cu_seqlens = cu_seqlens
        ctx.layer_kda_states = (
            list(layer_kda_states)
            if layer_kda_states is not None else None
        )
        ctx.return_states = return_states

        seeds = (
            list(layer_kda_states)
            if layer_kda_states is not None
            else [None] * len(block_layers)
        )

        # Run the block under no_grad. Cache per-layer INPUTS in
        # ``layer_inputs`` so the bwd walk can re-seed each layer's
        # re-fwd without rebuilding earlier layers.
        #   layer_inputs[i]   = input to layer i
        #                      = block input if i == 0
        #                      = output of layer i-1 if i > 0
        #   layer_inputs[n]   = block output (= layer N-1's output)
        # These are all [B,T,D] (D=1536) = 24 MiB at T=16384; 5
        # entries for a 4-layer block = 120 MiB. The savings come
        # from NOT caching the inner Function saves (those are
        # 1184 MiB/layer — we want them released between layers).
        with torch.no_grad():
            layer_inputs = [x]
            collected_states = []
            cur = x
            for layer, init in zip(block_layers, seeds):
                cur, s = layer(
                    cur, cu_seqlens=cu_seqlens, initial_state=init,
                )
                layer_inputs.append(cur)
                collected_states.append(s)

        ctx.save_for_backward(*layer_inputs)
        # Collected states are detached — they are side outputs
        # used as the next chunk's initial states, not in the
        # autograd graph. Storing as a tuple because
        # ``autograd.Function`` requires Tensor-or-tuple-of-Tensor
        # outputs.
        ctx.collected_states = tuple(
            s.detach() for s in collected_states
        )

        block_out = layer_inputs[-1]
        if return_states:
            return block_out, ctx.collected_states
        return block_out

    @staticmethod
    def backward(ctx, *grads):
        grad_block_out = grads[0]
        # When ``return_states=True``, ``grads[1:]`` are grads for
        # the collected states. Those are detached Tensors with no
        # upstream gradient (the next-chunk caller treats them as
        # constants), so we ignore them by returning ``None`` for
        # the ``layer_kda_states`` slot below.

        block_layers = ctx.block_layers
        cu_seqlens = ctx.cu_seqlens
        seeds = ctx.layer_kda_states
        if seeds is None:
            seeds = [None] * len(block_layers)
        saved = list(ctx.saved_tensors)  # length n+1
        n = len(block_layers)

        # Manual bwd: walk layers in reverse, re-fwd one at a time
        # so only the current layer's inner Function saves are
        # alive. After each layer's bwd, drop the local references
        # so the next layer's re-fwd starts from a clean slate.
        grad_out = grad_block_out
        for K in range(n - 1, -1, -1):
            x_in = saved[K]  # input to layer K (no-grad, small)
            x_in_g = x_in.detach().requires_grad_(True)
            with torch.enable_grad():
                layer = block_layers[K]
                init = seeds[K]
                cur_out, _s = layer(
                    x_in_g, cu_seqlens=cu_seqlens,
                    initial_state=init,
                )

            # Backward through this layer alone. Autograd will use
            # the inner Function saves (populated by the re-fwd
            # above), propagate ``grad_out`` through the entire
            # chain from ``cur_out`` back to ``x_in_g``, and the
            # inner saves are released as soon as their backward
            # hooks fire.
            if grad_out is not None:
                cur_out.backward(grad_out)

            # ``x_in_g.grad`` is the gradient w.r.t. the input of
            # layer K = the gradient that flows back from layer K
            # to layer K-1 (= ``grad_out`` for the next iteration,
            # or the final block-input gradient when K == 0).
            grad_out = x_in_g.grad

            # Drop local references so the inner Function saves can
            # be GC'd BEFORE the next layer's re-fwd pops its own
            # set of saves. This is the critical step: without the
            # ``del``, the re-fwd'd layer K+1's saves (still bound
            # via ``cur_out`` and ``x_in_g``) would keep peak
            # memory constant at 2 layers' worth.
            del x_in_g, layer, init, cur_out, _s

        # Backward return shape must match forward's args (minus
        # ctx). Only ``x`` is a Tensor we backprop into; the rest
        # are non-Tensor (``block_layers``, ``layer_kda_states``,
        # ``return_states``) or Tensor-with-no-grad
        # (``cu_seqlens``).
        return None, grad_out, None, None, None


def _block_manual_ckpt(block_layers, x, cu_seqlens,
                       layer_kda_states, return_states):
    """Public wrapper around ``_BlockManualCkptFunction.apply``.

    Mirrors the call signature of
    ``torch.utils.checkpoint.checkpoint(_block_forward, ...)`` so
    the model code can swap implementations cleanly.

    Returns
    -------
    If ``return_states=True``:
        ``(block_out, list_of_per_layer_states)`` — ``list`` to
        match the legacy ``_block_forward`` helper output type so
        callers can ``final_kda_states.extend(sub_final)``.
    Otherwise:
        ``block_out``
    """
    out = _BlockManualCkptFunction.apply(
        block_layers, x, cu_seqlens, layer_kda_states, return_states,
    )
    if return_states:
        return out[0], list(out[1])
    return out


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
                # num_chunks=32 → C = next_pow2(ceil(16384/32)) = 512
                # tokens per chunk → per-chunk peak logits
                # 512 × 248320 × 2 B ≈ 242 MB at V=248320, H=1536
                # (base.yml). The per-chunk dw GEMM is [V, C] @ [C, H]
                # = [248320, 512] @ [512, 1536]; K=512 lifts cuBLAS BF16
                # utilisation from 82% (K=256, NC=64) to 93% (K=512).
                # In-model FLCE forward+backward at N=16384: NC=64 takes
                # 1296 ms, NC=32 takes 1058 ms (−238 ms/mb). Projected to
                # n_chunks=16 production: −3.8 s/step (−11.5% of 33 s).
                # See auto-memory project_logit_gather_fold.md.
                num_chunks=32,
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
        layer_kda_states: list[torch.Tensor | None] | None = None,
        return_states: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
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

        ``layer_kda_states`` (length ``len(block_layers)``)
        holds the KDA recurrent state to seed each layer; pass
        ``None`` for "start from zeros" for every layer in the
        block. When ``return_states=True`` the block returns
        ``(output, collected_states)`` where ``collected_states``
        is the per-layer final state list; otherwise just the
        output (the legacy path).
        """
        collected: list[torch.Tensor] = []
        for i, layer in enumerate(block_layers):
            init = (
                layer_kda_states[i] if layer_kda_states is not None
                else None
            )
            x, s = layer(
                x, cu_seqlens=cu_seqlens, initial_state=init,
            )
            if return_states:
                collected.append(s)
        if return_states:
            return x, collected
        return x

    def forward(
        self,
        input_ids: torch.Tensor,  # [B, T], lives on this rank's device
        labels: torch.Tensor | None = None,  # [B, T], same device
        cu_seqlens: torch.Tensor | None = None,  # [total_docs+1]
        kda_states: list[torch.Tensor | None] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Run the full TP model forward.

        ``cu_seqlens`` is the chunk-local offset tensor produced
        by :func:`src.training.loop._slice_cu_seqlens` (or the
        full-sequence tensor for non-chunked callers). When set
        it is forwarded only to ``ShortConvolution`` so the
        depthwise conv resets its causal state at each doc
        boundary. The KDA sub-layer does NOT receive cu_seqlens
        — its recurrence sees the whole chunk as one stream, so
        the carried ``initial_state`` (= the previous chunk's
        ``final_state``) flows without per-doc resets.

        ``kda_states`` (optional) is a list of length
        ``num_layers`` carrying the KDA recurrent state at the
        start of this chunk, one entry per layer. Each entry is
        either ``None`` (use zeros — the legacy / first-chunk-
        of-step behavior) or a ``[1, hpp, head_k_dim,
        head_v_dim]`` float32 tensor. Pass ``None`` (or a list
        of ``None``) for the first chunk of each step.

        Returns ``{"loss" or "logits", "kda_states": [Tensor]×L}``.
        ``out["kda_states"][li]`` is the KDA state at the last
        token of this chunk for layer ``li``, ready to seed the
        same layer's call in the next chunk. When ``labels`` is
        ``None`` (inference), only ``out["kda_states"]`` and
        ``out["logits"]`` are populated.
        """
        device = input_ids.device.index if input_ids.device.type == "cuda" else input_ids.device
        embeds = self.replicated_per_device[str(device)]["embed_tokens"](input_ids)
        attn_res = self.replicated_per_device[str(device)]["attn_res"]
        blocks: list[torch.Tensor] = [embeds]  # b_0 = embedding
        x = embeds  # Block 0's input is the embedding.

        # Per-layer KDA final-state collection for the chunked
        # training invariant: each layer's ``final_state`` is
        # captured at the end of the forward so the caller can
        # pass it as the next chunk's ``initial_state``.
        final_kda_states: list[torch.Tensor] = []
        # Track which absolute layer indices we're collecting for
        # (the per-sub-block loop below returns local indices).
        layer_offset = 0

        # Two-tier ckpt structure (block + sub-block, rev 3, current prod)
        # --------------------------------------------------------
        # Non-last blocks (block 0..num_blocks-2): each 4-layer block
        # is split into **2 sub-blocks of 2 layers each**, each sub-
        # block wrapped in its own
        # ``torch.utils.checkpoint.checkpoint(use_reentrant=True)``
        # call. **2 rounds of recomputation per block during bwd**:
        #   - Round 1: re-fwd sub-block 0 (layers 0..1) → 2 layers'
        #     saved_tensors alive simultaneously (peak ~2368 MiB
        #     KDA-style dynamic state at T=16384 = 2 × 1184 MiB) →
        #     bwd L0, L1 → release sub-block 0 saves.
        #   - Round 2: re-fwd sub-block 1 (layers 2..3) → 2 layers'
        #     saved_tensors alive → bwd L2, L3 → release.
        #
        # Why this and not rev 4 (block-level + manual per-layer
        # re-fwd): rev 4 was implemented and ABANDONED on 2026-07-11.
        # Caching 5 per-layer inputs per block to enable the
        # per-layer bwd walk costs +1008 MiB at fwd_out (5 × 48 MiB
        # × 7 blocks), which exceeds the -621 MiB saved at peak
        # delta. Net rev 4 peak = 10371 MiB (vs 9984 rev 3 baseline,
        # +387 MiB WORSE). The rev 4 implementation is preserved at
        # ``_BlockManualCkptFunction`` (model.py:26-210) as a
        # reference but is NOT invoked by this loop. See
        # ``docs/gradient_checkpointing.md`` §"rev 4 abandoned" and
        # ``project_rev4_ckpt.md`` for full analysis.
        #
        # Why not block-level ckpt (rev 2, 1 ckpt per 4-layer block):
        #   use_reentrant=True keeps ALL inner Functions' saves
        #   alive during the re-fwd → 4 layers' saves alive at the
        #   re-fwd peak (4736 MiB).
        # Why not per-layer ckpt (rev 1, 1 ckpt per layer):
        #   31 ckpts × +672 MiB each ≈ +18.8 GiB fwd_out penalty.
        # 2 sub-blocks of 2 layers is the empirical sweet spot.
        #
        # Last block (block num_blocks-1): **per-layer ckpt** for
        # layers 28..30 (3 ckpts); layer 31 un-ckpt'd (KDA internals
        # feed lm_head + fused CE loss directly).
        #
        # Total ckpt() invocations per fwd: 7 × 2 (block split into
        # 2 sub-blocks) + 3 (per-layer last block) = **17 ckpt calls**.
        #
        # AttnRes is invoked at every non-first block boundary,
        # *outside* the ckpt wrapper, so the per-block output ``b_n``
        # captures the boundary-attention input ``x``. ``blocks`` (the
        # list of completed block reps) is held live across blocks
        # but is small (one ``[B,T,D]`` per block).
        # Number of layers per sub-block (hardcoded; see comment for
        # the trade-off rationale). Must divide ``block_size`` evenly.
        _sub_block_layers = 2
        layers = self.layers_per_device[str(device)]
        block_size = self.config.block_size
        num_blocks = self.config.num_blocks
        assert block_size % _sub_block_layers == 0, (
            f"block_size ({block_size}) must be divisible by "
            f"_sub_block_layers ({_sub_block_layers})"
        )
        for block_idx in range(num_blocks - 1):
            if block_idx > 0:
                x = attn_res(blocks)
            start = block_idx * block_size
            end = start + block_size
            # 2 sub-blocks of 2 layers each; each sub-block is wrapped
            # in its own ckpt so the bwd has 2 rounds of recomputation
            # (peak 2 layers alive instead of 4).
            for sub_start in range(start, end, _sub_block_layers):
                sub_end = min(sub_start + _sub_block_layers, end)
                sub_layers = layers[sub_start:sub_end]
                # Slice the per-layer kda_states for this sub-block.
                # ``None`` means "all zeros" — the sub-block helper
                # handles ``None`` uniformly by passing ``None`` to
                # every layer.
                if kda_states is None:
                    sub_states = None
                else:
                    sub_states = kda_states[sub_start:sub_end]
                x, sub_final = torch.utils.checkpoint.checkpoint(
                    self._block_forward, sub_layers, x, cu_seqlens,
                    sub_states, True,  # return_states=True
                    use_reentrant=True, preserve_rng_state=False,
                )
                # ``sub_final`` is local to this sub-block (length
                # ``len(sub_layers)``). Remap to absolute layer indices.
                final_kda_states.extend(sub_final)
                layer_offset += len(sub_layers)
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
        for li, layer in enumerate(last_block_layers[:-1]):
            init = (
                kda_states[last_start + li] if kda_states is not None
                else None
            )
            x, final_state = torch.utils.checkpoint.checkpoint(
                layer, x, cu_seqlens, init,
                use_reentrant=True, preserve_rng_state=False,
            )
            final_kda_states.append(final_state)
            layer_offset += 1
        # Final layer of the last block: NO checkpoint wrapper.
        # Its KDA internals (q, k, v, beta, g, Aqk, Akk, w_wy,
        # u_wy, qg, kg, v_new, h) are held for backward.
        final_init = (
            kda_states[last_start + len(last_block_layers) - 1]
            if kda_states is not None else None
        )
        x, final_state = last_block_layers[-1](
            x, cu_seqlens=cu_seqlens, initial_state=final_init,
        )
        final_kda_states.append(final_state)
        layer_offset += 1

        hidden_states = self.replicated_per_device[str(device)]["norm"](x)
        out: dict[str, torch.Tensor] = {"kda_states": final_kda_states}
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
