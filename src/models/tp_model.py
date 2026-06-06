"""Tensor-parallel HippoModel for Megatron-style TP.

TP design (assumes a single process spanning N GPUs):

  Replicated on every rank
  ------------------------
  - embed_tokens (full vocab x hidden)
  - RMSNorm (full hidden)
  - BlockAttnRes (full hidden pseudo-query and norm)
  - KDA / KimiDeltaAttention (full weights; each rank runs the
    full attention with its own copy). The KDA replication
    overhead is small relative to FFN/lm_head.

  Column-parallel (output sharded, no all-reduce)
  -----------------------------------------------
  - SwiGLU gate_proj, up_proj (output = intermediate, sharded)
  - lm_head (output = vocab, sharded; loss computed on the
    sharded vocab with a final reduce)

  Row-parallel (input sharded, all-reduce output)
  -----------------------------------------------
  - SwiGLU down_proj (input = intermediate, sharded; output
    all-reduced to full hidden)

The forward pass is: embed (replicated) -> 32 layers (attn_norm
-> KDA -> mlp_norm -> SwiGLU -> all-reduce inside down_proj) ->
final norm -> lm_head (sharded) -> parallel loss.

All-reduce happens once per layer (inside the FFN's down_proj) and
once at the lm_head -> loss boundary (inside the parallel cross
entropy). No communication is needed for the KDA since it is
replicated.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .norms import RMSNorm
from .kda import KDA
from .block_attn_res import BlockAttnRes
from .tp_layers import (
    ColumnParallelLinear,
    RowParallelLinear,
    get_tp_rank,
    get_tp_world_size,
    tp_all_reduce,
    tp_all_reduce_max,
    tp_all_reduce_sum,
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
        self.gate_proj = ColumnParallelLinear(
            config.hidden_size, config.intermediate_size,
            bias=config.use_bias, device=device, dtype=dtype,
        )
        self.up_proj = ColumnParallelLinear(
            config.hidden_size, config.intermediate_size,
            bias=config.use_bias, device=device, dtype=dtype,
        )
        self.down_proj = RowParallelLinear(
            config.intermediate_size, config.hidden_size,
            bias=config.use_bias, device=device, dtype=dtype,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is replicated (full hidden). gate/up produce
        # sharded intermediate slices on this rank. down is
        # row-parallel and all-reduces back to full hidden.
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)


# --------------------------------------------------------------------------- #
# TP lm_head (column-parallel on vocab dim)                                   #
# --------------------------------------------------------------------------- #
class TPLmHead(nn.Module):
    """Column-parallel LM head.

    Weight shape on each rank: ``[vocab // world, hidden]``. Output
    is ``[B, T, vocab // world]`` (sharded vocab). The full vocab
    logits are only materialized during loss computation, via a
    gather-or-reduce step inside :class:`ParallelCrossEntropy`.
    """

    def __init__(self, hidden_size: int, vocab_size: int, bias: bool = False,
                 device=None, dtype=None) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.world = get_tp_world_size()
        assert vocab_size % self.world == 0, (
            f"vocab_size={vocab_size} not divisible by tp_world={self.world}"
        )
        self.vocab_per_partition = vocab_size // self.world
        self.weight = nn.Parameter(
            torch.empty(self.vocab_per_partition, hidden_size, device=device, dtype=dtype)
        )
        if bias:
            self.bias = nn.Parameter(
                torch.empty(self.vocab_per_partition, device=device, dtype=dtype)
            )
        else:
            self.register_parameter("bias", None)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)

    def extra_repr(self) -> str:
        return (
            f"hidden_size={self.hidden_size}, vocab_size={self.vocab_size}, "
            f"vocab_per_partition={self.vocab_per_partition}, "
            f"tp_world={self.world}, bias={self.bias is not None}"
        )


# --------------------------------------------------------------------------- #
# Parallel cross entropy                                                      #
# --------------------------------------------------------------------------- #
class ParallelCrossEntropy(nn.Module):
    """Cross-entropy on sharded logits.

    On each rank the logits cover a disjoint slice of the vocab
    dim. We compute log-softmax locally, then all-reduce the
    elementwise log-prob and gather the per-token target logit
    (handled inside ``forward``). Numerically equivalent to the
    standard CE on the full logits because log-softmax is
    shift-invariant.
    """

    def forward(
        self,
        sharded_logits: torch.Tensor,  # [..., vocab // world]
        targets: torch.Tensor,         # [...] int64, replicated
    ) -> torch.Tensor:
        # Local log-softmax over the sharded vocab slice.
        # log_softmax is shift-invariant, so we can use the local
        # max for numerical stability without any communication.
        local_max = sharded_logits.detach().max(dim=-1, keepdim=True).values
        local_max = local_max.clamp_min(-1e9)  # avoid -inf
        local_centered = sharded_logits - local_max
        local_logsumexp = local_centered.exp().sum(dim=-1, keepdim=True)
        local_log_probs = local_centered - local_logsumexp.log()  # [..., vocab/world]
        # All-reduce the log-probs: each rank's contribution is its
        # slice of log p(target | vocab slice). To get the full log
        # probability we need to combine across ranks.
        #
        # Equivalently, log p(target) = logsumexp over vocab of
        # (logit_target - logsumexp(logits)). We split this into
        # rank-local parts and sum:
        #   log p = logsumexp_rank( local_log_probs ).sum_across_ranks()
        # = (local_logsumexp_rank + local_max_rank).sum_across_ranks() -
        #   log(sum exp(rank_lse + rank_max))
        #
        # Implementation: gather the per-token target logit on each
        # rank by picking the local slice that contains the target.
        # If target vocab idx is in this rank's range, the local
        # log-prob at target is correct. If not, we need to fetch
        # the logit from the owning rank. For simplicity and
        # correctness, we do a small all-gather of the *gathered
        # logit* — but vocab is 248k which is fine for an all-gather
        # of 248k * batch * seq * 2 bytes = ~ 250MB at batch=2, seq=512.
        #
        # To avoid that, we use a different identity: compute the
        # full logsumexp via:
        #   global_max = max(local_max)        (all-reduce max)
        #   global_lse = sum( exp(local_lse + local_max - global_max) )
        #   log p = logits[target] - global_lse - global_max
        # We still need logits[target] on each rank. Get it by
        # gathering target's per-rank slice via an all-gather of
        # the gathered logit *only for the target position*, but
        # that's per-token. The cheap option: do an all-gather of
        # local_max only and use logsumexp math (which is exact if
        # we ignore the logit-at-target term, which we cannot).
        # So we DO need to gather target logits.
        #
        # Pragmatic choice: do a vocab-dim all-gather of the local
        # log-probs. The result is replicated log p(v) for v in
        # full vocab; we then index the target and NLL-sum.
        # Memory: B*T*V*4 bytes (FP32 log-probs) per rank.
        # At B=2, T=512, V=248320, FP32: ~ 1 GB per rank. Heavy.
        #
        # Better: gather only the target logit. For each token, do
        # an all-gather of ONE float. Use an all-gather of shape
        # [B*T, world] by gathering local_log_probs at the
        # (target - rank_offset) index on each rank. Then each
        # rank sees the per-token log-prob at target from every
        # rank and picks the one whose rank owns the target.
        rank = get_tp_rank()
        world = get_tp_world_size()
        vpp = sharded_logits.size(-1)
        rank_offset = rank * vpp
        # Local index of target within this rank's vocab slice,
        # or -1 if target belongs to a different rank.
        local_target_idx = targets - rank_offset  # [...], int64
        target_in_rank = (local_target_idx >= 0) & (local_target_idx < vpp)
        safe_idx = local_target_idx.clamp(min=0, max=vpp - 1)
        local_target_logit = torch.gather(
            sharded_logits, dim=-1, index=safe_idx.unsqueeze(-1)
        ).squeeze(-1)  # [...]

        # All-reduce the target logit SUM (each rank contributes
        # local_target_logit or 0 via the mask). The result equals
        # the true target logit (which lives on exactly one rank).
        local_target_logit = local_target_logit * target_in_rank.to(local_target_logit.dtype)
        target_logit = tp_all_reduce(local_target_logit)  # [...]

        # Global logsumexp.
        # NOTE: ``local_logsumexp`` above is the *sum* of
        # exp(local_centered) (the ``.log()`` is only applied when
        # computing local_log_probs on the next line). To get the
        # log-domain value used in the LSE identity below we must
        # take its log here. Without this, ``local_lse`` is ~1.5e5
        # and exp(1.5e5) overflows FP32 → global_lse=inf → nll=inf.
        local_lse = local_logsumexp.log().squeeze(-1)  # [...]
        local_max_sq = local_max.squeeze(-1)  # [...]
        # global_max = max over ranks of local_max
        global_max = tp_all_reduce_max(local_max_sq)
        # global_lse = log sum_r exp(local_lse + local_max - global_max)
        scaled = (local_lse + (local_max_sq - global_max)).exp()
        global_lse = tp_all_reduce_sum(scaled).log() + global_max  # [...]

        # NLL = - (logit[target] - logsumexp(logits))
        nll = -(target_logit - global_lse)
        return nll


# --------------------------------------------------------------------------- #
# TP HippoModel                                                               #
# --------------------------------------------------------------------------- #
class TPHippoLayer(nn.Module):
    """TP version of HippoLayer. KDA and AttnRes are replicated;
    FFN is sharded via TPSwiGLU."""

    def __init__(self, layer_idx: int, config, device=None, dtype=None) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.block_size = config.block_size

        # NB: ``device`` may be 0 (an int) which is falsy in Python
        # — guard with ``is not None`` to avoid the conditional
        # silently falling through to the no-op branch.
        def _to(m: nn.Module) -> nn.Module:
            if device is None:
                return m
            if dtype is not None:
                return m.to(device=device, dtype=dtype)
            return m.to(device=device)

        self.attn_res = _to(BlockAttnRes(config))
        self.mlp_res = _to(BlockAttnRes(config))
        self.attn_norm = _to(RMSNorm(config.hidden_size, eps=config.rms_norm_eps))
        self.mlp_norm = _to(RMSNorm(config.hidden_size, eps=config.rms_norm_eps))
        self.kda = _to(KDA(config, layer_idx=layer_idx))
        self.ffn = TPSwiGLU(config, device=device, dtype=dtype)

    def forward(
        self,
        blocks: list[torch.Tensor],
        partial_block: torch.Tensor | None,
    ) -> tuple[list[torch.Tensor], torch.Tensor | None]:
        layer_number = self.layer_idx + 1

        if layer_number % self.block_size == 0:
            if partial_block is not None:
                blocks = blocks + [partial_block]
            partial_block = None

        # KDA path: replicated, no comm.
        h_attn, attn_out = torch.utils.checkpoint.checkpoint(
            self._attn_forward, blocks, partial_block,
            use_reentrant=False, preserve_rng_state=False,
        )
        if partial_block is None:
            partial_block = attn_out
        else:
            partial_block = partial_block + attn_out

        # FFN path: sharded; all-reduce inside down_proj.
        ffn_out = torch.utils.checkpoint.checkpoint(
            self._ffn_forward, blocks, partial_block,
            use_reentrant=False, preserve_rng_state=False,
        )
        partial_block = partial_block + ffn_out
        return blocks, partial_block

    def _attn_forward(self, blocks, partial_block):
        h_attn = self.attn_res(blocks, partial_block)
        attn_out = self.kda(self.attn_norm(h_attn))
        return h_attn, attn_out

    def _ffn_forward(self, blocks, partial_block):
        h_mlp = self.mlp_res(blocks, partial_block)
        return self.ffn(self.mlp_norm(h_mlp))


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
            device_mods = {
                "embed_tokens": nn.Embedding(
                    config.vocab_size, config.hidden_size,
                ).to(device=d, dtype=dtype),
                "norm": RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
                .to(device=d, dtype=dtype),
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

        # lm_head: column-parallel on each device, but with tied
        # weights to the local embed_tokens. Since lm_head's weight
        # is sharded on vocab dim, we cannot tie to the full
        # embed_tokens (which is replicated full vocab). For
        # v0.0.0 validation we untie the weights: each rank has its
        # own [vocab/world, hidden] lm_head weight, NOT tied to
        # the embedding. This is a one-time extra cost: vocab *
        # hidden * 2 bytes = 508 MB total, divided by world (so
        # 254 MB per rank for world=2).
        self.lm_head_per_device: dict[int, TPLmHead] = {}
        for d in self.devices:
            head = TPLmHead(
                config.hidden_size, config.vocab_size,
                bias=config.use_bias, device=d, dtype=dtype,
            )
            self.lm_head_per_device[d] = head

        # Parallel cross entropy, one per device (state-less, just
        # convenience for forward).
        self.loss_fn_per_device: dict[int, ParallelCrossEntropy] = {
            d: ParallelCrossEntropy() for d in self.devices
        }

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
        """
        # 1) Top-level replicated modules (embed_tokens, norm).
        for mname, src_mod in self.replicated_per_device[src_device].items():
            src_params = dict(src_mod.named_parameters(recurse=True))
            for dst_device, dst_mods in self.replicated_per_device.items():
                if dst_device == src_device:
                    continue
                dst_params = dict(dst_mods[mname].named_parameters(recurse=True))
                for pname, p in src_params.items():
                    dst_params[pname].data.copy_(p.data)

        # 2) Per-layer replicated sub-modules (KDA, attn_res, mlp_res,
        # attn_norm, mlp_norm). Each layer has exactly one of each
        # so we can copy by explicit attribute name.
        for li, src_layer in enumerate(self.layers_per_device[src_device]):
            for sub_name in ("kda", "attn_res", "mlp_res", "attn_norm", "mlp_norm"):
                src_sub = getattr(src_layer, sub_name)
                src_params = dict(src_sub.named_parameters(recurse=True))
                for dst_device, dst_layers in self.layers_per_device.items():
                    if dst_device == src_device:
                        continue
                    dst_sub = getattr(dst_layers[li], sub_name)
                    dst_params = dict(dst_sub.named_parameters(recurse=True))
                    for pname, p in src_params.items():
                        dst_params[pname].data.copy_(p.data)

    def forward(
        self,
        input_ids: torch.Tensor,  # [B, T], lives on this rank's device
        labels: torch.Tensor | None = None,  # [B, T], same device
    ) -> dict[str, torch.Tensor]:
        device = input_ids.device.index if input_ids.device.type == "cuda" else input_ids.device
        embeds = self.replicated_per_device[device]["embed_tokens"](input_ids)
        blocks: list[torch.Tensor] = [embeds]
        partial_block: torch.Tensor | None = embeds
        for layer in self.layers_per_device[device]:
            blocks, partial_block = layer(blocks, partial_block)
        hidden_states = self.replicated_per_device[device]["norm"](partial_block)
        sharded_logits = self.lm_head_per_device[device](hidden_states)
        out: dict[str, torch.Tensor] = {"logits": sharded_logits}
        if labels is not None:
            shift_logits = sharded_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Loss reduction in FP32. With FP16 params, the sharded
            # logits are FP16 and a direct cross-entropy would lose
            # precision in logsumexp over the vocab dim. The cast
            # is cheap (the gather step in ParallelCrossEntropy is
            # already an all-reduce over the full vocab).
            nll = self.loss_fn_per_device[device](
                shift_logits.float(), shift_labels,
            )
            # ignore_index handling: NLL for ignored positions is 0
            # (so they don't contribute to mean). We mask post-hoc.
            mask = (shift_labels != -100).to(nll.dtype)
            loss = (nll * mask).sum() / mask.sum().clamp_min(1.0)
            out["loss"] = loss
        return out
