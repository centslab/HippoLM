"""Param-group builder for the training loop.

The loop calls :func:`build_param_groups` once per rank to
materialise the (Muon optimizer, AdamW optimizer) pair from
the model's per-device parameter list. The routing rules
(2-D → Muon, 1-D / embed / lm_head / KDA conv1d → AdamW) live
in this module — see :func:`build_param_groups` docstring.

Splitting this off from the rest of
:mod:`src.training.param_offload` keeps the routing logic in
one place without mixing optimizer algorithm or offload
plumbing; the optimizers themselves are imported as types and
returned to the caller; they hold the per-param state after
construction.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import torch.nn as nn

from ..precision_config import PrecisionConfig
from .adamw import CPUAdamW
from .muon import CPUMuon


def build_param_groups(
    model: nn.Module,
    device: int,
    lr_muon: float = 1e-3,
    lr_adamw: float = 1e-4,
    weight_decay: float = 0.01,
    adamw_beta1: float = 0.9,
    adamw_beta2: float = 0.95,
    adamw_eps: float = 1e-8,
    muon_momentum: float = 0.95,
    muon_weight_decay: float = 0.0,
    precision: Optional[PrecisionConfig] = None,
) -> Tuple[List, List]:
    """Build a (muon_optimizer, adamw_optimizer) pair for a single
    rank's model fragment.

    Standard Muon routing:
        - 1D params (RMSNorm.weight, BlockAttnRes.query,
          KDA A_log/dt_bias with ``_no_weight_decay``): AdamW
        - 2D Linear weights: Muon
        - Embedding + lm_head: AdamW (per user spec)

    ``weight_decay`` is the AdamW-side decay (1D / embed / lm_head /
    the routed 3D short-conv weights). ``muon_weight_decay`` is the
    Muon-side decay (2D Linear weights) and defaults to ``0.0`` to
    match the pre-refactor hardcoded behavior. Both come from the
    ``optimizer:`` block in the yml (via ``scripts.cli._flatten_
    optimizer_overrides``) or the corresponding ``--weight_decay`` /
    ``--muon_weight_decay`` CLI flags.

    ``adamw_beta1`` / ``adamw_beta2`` (defaults 0.9 / 0.95) are the
    first / second moment decay for :class:`CPUAdamW`. They come from
    ``optimizer.adamw.beta1`` / ``optimizer.adamw.beta2`` in the yml
    or the ``--adamw_beta1`` / ``--adamw_beta2`` CLI flags. With the
    merged-accumulator design, ``beta1`` is stored / serialized /
    logged but not consumed by ``step()`` — the algorithm only uses
    ``beta2`` for ``v = β2*v + (1-β2)*m² + bc2`` (see
    :meth:`CPUAdamW.step`). The defaults match the pre-refactor
    hardcoded values so existing checkpoints are numerically identical.

    ``adamw_eps`` (default 1e-8) is the AdamW epsilon inside
    ``sqrt(v/bc2) + eps`` (see :meth:`CPUAdamW.step`). Comes from
    ``optimizer.adamw.eps`` in the yml or ``--adamw_eps`` on the CLI.

    ``precision`` (optional :class:`PrecisionConfig`) is forwarded
    to both optimizers; the model weights themselves are
    constructed at the dtype from ``precision.model_weights`` by
    the caller (``scripts.train`` / ``src.training.loop``).
    Defaults to the canonical yml precision when ``None``.
    """
    muon_params: list[nn.Parameter] = []
    adamw_params: list[nn.Parameter] = []
    seen: set[int] = set()
    for name, p in model.named_parameters_per_device(device):
        if not p.requires_grad or id(p) in seen:
            continue
        seen.add(id(p))
        # Per user spec: lm_head and embed_tokens -> AdamW.
        # The AttnRes pseudo-query lives under
        # ``replicated.{device}.attn_res.query`` so it is also
        # caught by the ``replicated.`` prefix here.
        if name.startswith("lm_head.") or name.startswith("replicated."):
            adamw_params.append(p)
        # KDA short-conv weights are 3D (nn.Conv1d: [D, 1, W]).
        # CPUMuon._newton_schulz does ``g.t()`` which only works on
        # 2-D matrices, so we must route them to AdamW. These params
        # are tiny (393K total) so the precision/regularization
        # difference vs Muon is negligible. Match by name suffix
        # (``...q_conv1d.weight``) for explicitness; the 3D guard
        # is a backstop in case fla adds more depthwise convs.
        elif (p.ndim == 3 or any(
                name.endswith(suffix) for suffix in
                (".q_conv1d.weight", ".k_conv1d.weight", ".v_conv1d.weight"))):
            adamw_params.append(p)
        elif p.ndim < 2:
            # 1D: norms, A_log, dt_bias, o_norm.weight, lm_head.bias.
            adamw_params.append(p)
        else:
            muon_params.append(p)

    muon_opt = CPUMuon(
        muon_params,
        lr=lr_muon,
        momentum=muon_momentum,
        nesterov=True,
        ns_steps=5,
        weight_decay=muon_weight_decay,   # resolved per-config (default 0.0; was hardcoded)
        precision=precision,
    )
    adamw_opt = CPUAdamW(
        adamw_params,
        lr=lr_adamw,
        betas=(adamw_beta1, adamw_beta2),
        eps=adamw_eps,
        weight_decay=weight_decay,
        precision=precision,
    )

    # ------------------------------------------------------------------
    # NVFP4 mode-3 wiring: register every no_bf16_master NVFP4 module
    # with the matching optimizer so its FP4 packed buffers actually
    # receive updates and its side-channel grad_w stash gets consumed
    # by accumulate_grads_to_cpu. Without this step the modules
    # would be invisible to the per-param ``named_parameters`` loop
    # above (mode-3 has no leaf ``Parameter`` for autograd to attach
    # to), and the bwd's stashed grad would leak ~24 MiB / gate_up +
    # ~12 MiB / down × num_layers = ~1152 MiB at base.yml as a pure
    # unreferenced tensor.
    #
    # Routing: every NVFP4 module represents a 2-D FFN weight (the
    # swiglu gate_up / down projections and their TP counterparts),
    # which matches the BF16 2-D → Muon rule above. The module-side
    # material / commit API in NVFP4Linear handles the per-chunk
    # update routing for both storages.
    # ------------------------------------------------------------------
    for _mod in model.modules():
        if not getattr(_mod, "no_bf16_master", False):
            continue
        if _mod.__class__.__name__ not in (
            "NVFP4Linear",
            "NVFP4ColumnParallelLinear",
            "NVFP4RowParallelLinear",
        ):
            continue
        muon_opt.register_nvfp4_module(_mod)

    return muon_opt, adamw_opt