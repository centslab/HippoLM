"""CLI argument parsing for the training entry point.

Builds the argparse parser, applies the YAML config overlay
(yaml values become parser defaults — explicit CLI flags win),
and returns the :class:`argparse.Namespace`. The script
itself (``scripts/train.py``) imports :func:`parse_args` and
hands the result to :func:`scripts.train.train`.

Why a separate module: ``train.py`` was bloated to ~1700 lines
before the PR-3..PR-6 refactor. The CLI surface (~50 flags) is
mechanical and changes on a different cadence from the training
loop; keeping it in its own module makes both easier to read.

Overlay precedence (PR-9 follow-up):

  - YAML is canonical for **defaults** (the yml in
    :file:`configs/base.yml` defines the project's default
    precision / batch / lr / etc.).
  - **CLI flags win** when explicitly passed: any value the
    user supplies on the command line overrides the yml.
  - Nested dicts (e.g. ``precision: { model_weights: {...} }``)
    flow through as a single attribute (``args.precision`` is
    the whole dict); the training loop constructs the typed
    :class:`PrecisionConfig` from it via
    :meth:`PrecisionConfig.from_dict`.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Optional, Sequence


def build_parser() -> argparse.ArgumentParser:
    """Build the argparse parser with all training flags.

    Defaults match :file:`configs/base.yml` so the script can be
    invoked with no arguments and pick up the canonical run
    defaults. The ``--help`` text is intentionally verbose —
    the per-flag help is the primary user-facing documentation
    for the CLI surface, and the cost of a few extra lines in
    this function is much lower than the cost of "how do I set
    X?" questions in chat.
    """
    p = argparse.ArgumentParser(description="Train HippoLM")

    # ---- Config file ----
    p.add_argument("--config", type=str, default="configs/base.yml",
                   help="YAML config file. Top-level keys set the parser "
                        "defaults (explicit CLI flags still win).")

    # ---- Model ----
    p.add_argument("--vocab_size", type=int, default=248320)
    p.add_argument("--hidden_size", type=int, default=1024)
    p.add_argument("--tie_word_embeddings", type=bool, default=True)
    p.add_argument("--use_bias", type=bool, default=False)
    p.add_argument("--num_heads", type=int, default=16)
    p.add_argument("--head_dim", type=int, default=64)
    p.add_argument("--expand_v", type=float, default=1.0)
    p.add_argument("--kda_mode", type=str, default="chunk",
                   choices=["chunk", "fused_recurrent"])
    p.add_argument("--use_short_conv", type=bool, default=False)
    p.add_argument("--allow_neg_eigval", type=bool, default=False)
    p.add_argument("--safe_gate", type=bool, default=False,
                   help="Whether the KDA kernel can assume the gate "
                        "values (in log space) are in [lower_bound, 0) "
                        "and use the M=16 TensorCore acceleration. "
                        "Requires --lower_bound to be set.")
    p.add_argument("--lower_bound", type=float, default=None,
                   help="Lower bound for the KDA forget gate in log "
                        "space. Clamps the gate output to "
                        "[lower_bound, 0). Required when --safe_gate "
                        "is set. -5 gives exp(g) ~= 0.0067 at minimum.")
    p.add_argument("--conv_size", type=int, default=4)
    p.add_argument("--conv_bias", type=bool, default=False)
    p.add_argument("--num_layers", type=int, default=32)
    p.add_argument("--num_blocks", type=int, default=8)
    p.add_argument("--intermediate_size", type=int, default=2736)
    p.add_argument("--rms_norm_eps", type=float, default=1e-6)

    # ---- Training ----
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--seq_len", type=int, default=512)
    # ---- Chunked-training: tokens per micro-batch (chunk) ----
    # ``seq_len`` is now the TOTAL tokens per step (one super-long
    # FFD-packed sequence). The forward is split into
    # ``seq_len // micro_batch_size`` chunks of this many tokens
    # each, with the KDA recurrent state carried between chunks.
    # Default = previous seq_len default (16384) — same total
    # tokens per step as before when combined with
    # gradient_accumulation_steps=16 → seq_len=262144.
    p.add_argument(
        "--micro_batch_size", type=int, default=16384,
        help="Tokens per micro-batch chunk. The super-long sequence"
             " produced per step (length ``seq_len``) is split into"
             " ``seq_len // micro_batch_size`` chunks, and the"
             " KDA recurrent state is carried between chunks (full"
             " BPTT through the carried state). Default 16384."
             " Set to ``seq_len`` to disable chunking (legacy"
             " single-chunk behavior).",
    )
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    # AdamW beta1 / beta2 (first / second moment decay). Default
    # values match the legacy hardcoded (0.9, 0.95) on the CPUAdamW
    # merged-accumulator path so existing checkpoints stay numerically
    # identical. Override via ``optimizer.adamw.beta1`` /
    # ``optimizer.adamw.beta2`` in the yml, or these flat CLI flags.
    p.add_argument("--adamw_beta1", type=float, default=0.9,
                   help="AdamW first-moment decay. Default 0.9 "
                        "(matches the legacy hardcoded value).")
    p.add_argument("--adamw_beta2", type=float, default=0.95,
                   help="AdamW second-moment decay. Default 0.95 "
                        "(matches the legacy hardcoded value).")
    p.add_argument("--adamw_eps", type=float, default=1e-8,
                   help="AdamW epsilon inside ``sqrt(v/bc2) + eps``."
                        " Default 1e-8 (matches the legacy hardcoded value).")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--max_steps", type=int, default=1000)
    p.add_argument("--gradient_accumulation_steps", type=int, default=4)
    # ---- WSD LR scheduler (warmup-stable-decay) ----
    # Both peaks (``--learning_rate`` and ``--muon_lr``) are scaled
    # by the same multiplier, so Muon and AdamW track each other
    # through warmup / stable / decay. Defaults of 0/0 disable
    # scheduling (constant peak LR); set both in ``configs/base.yml``
    # (or on the CLI) to opt in. See ``src/training/loop.py:wsd_lr``.
    p.add_argument(
        "--lr_warmup_steps", type=int, default=0,
        help="Number of warmup steps at the start of training."
             " Linear ramp 0 -> peak. 0 disables the warmup phase.",
    )
    p.add_argument(
        "--lr_decay_steps", type=int, default=0,
        help="Number of decay steps at the end of training."
             " Linear ramp peak -> 0 over the last N steps."
             " 0 disables the decay phase (constant LR).",
    )
    p.add_argument("--max_grad_norm", type=float, default=1.0,
                   help="Max global (TP-reduced) L2 grad norm. "
                        "Set <= 0 to disable clipping. Default 1.0.")
    p.add_argument("--log_interval", type=int, default=10)
    p.add_argument("--checkpoint_interval", type=int, default=100,
                   help="Save a checkpoint every N steps. Set <= 0 "
                        "to disable periodic save (e.g. for short smoke runs).")
    p.add_argument("--checkpoint_keep_last_n", type=int, default=None,
                   help="After each checkpoint save, prune older "
                        "checkpoint_step_*.pt files so at most N "
                        "remain (newest N kept). Default: keep all. "
                        "Use 0 to keep only the just-saved file.")

    # ---- Output ----
    p.add_argument("--output_dir", type=str, default="output")

    # ---- Data ----
    p.add_argument("--stage", type=str, default="pretrain",
                   choices=["pretrain", "sft"],
                   help="Training stage: pretrain or sft")
    p.add_argument("--use_modelscope", type=bool, default=True,
                   help="Prefer ModelScope (Aliyun CDN) over HF. Falls "
                        "back to HF only on network/connection errors.")
    p.add_argument("--pretrain_dataset_hf", type=str,
                   default="openbmb/Ultra-FineWeb-L3",
                   help="HF dataset id for pretraining (fallback source).")
    p.add_argument("--pretrain_dataset_ms", type=str,
                   default="OpenBMB/Ultra-FineWeb-L3",
                   help="ModelScope dataset id for pretraining (primary).")
    p.add_argument("--pretrain_config", type=str,
                   default="Ultra-FineWeb-L3-en-QA-Synthetic")
    # ---- Multi-source pretrain (Ultra-FineWeb-L3) ----
    # When ``--pretrain_multi_source`` is true, the loader streams
    # 4 subsets in a 2-phase schedule: phase 0 is multi-style
    # (English *2 + Chinese *1), phase 1 is QA (English *2 + Chinese
    # *1). When all 4 are exhausted, the schedule restarts from
    # phase 0. The single-source ``--pretrain_config`` is ignored in
    # this mode. Set to false to fall back to the single-source path.
    p.add_argument("--pretrain_multi_source", type=bool, default=False,
                   help="Stream 4 Ultra-FineWeb-L3 subsets (en/zh x "
                        "multi-style/QA) with a 2-phase, 2:1 en:zh "
                        "schedule. Overrides --pretrain_config.")
    p.add_argument("--pretrain_en_multi_config", type=str,
                   default="Ultra-FineWeb-L3-en-Multi-Style-Synthetic",
                   help="Multi-source mode: English multi-style subset.")
    p.add_argument("--pretrain_zh_multi_config", type=str,
                   default="Ultra-FineWeb-L3-zh-Multi-Style-Synthetic",
                   help="Multi-source mode: Chinese multi-style subset.")
    p.add_argument("--pretrain_en_qa_config", type=str,
                   default="Ultra-FineWeb-L3-en-QA-Synthetic",
                   help="Multi-source mode: English QA subset.")
    p.add_argument("--pretrain_zh_qa_config", type=str,
                   default="Ultra-FineWeb-L3-zh-QA-Synthetic",
                   help="Multi-source mode: Chinese QA subset.")
    p.add_argument("--sft_dataset_hf", type=str,
                   default="openbmb/UltraData-SFT-2605",
                   help="HF dataset id for SFT (fallback source).")
    p.add_argument("--sft_dataset_ms", type=str,
                   default="OpenBMB/UltraData-SFT-2605",
                   help="ModelScope dataset id for SFT (primary).")
    p.add_argument("--sft_config", type=str, default=None)
    p.add_argument("--text_field", type=str, default="content")
    p.add_argument("--tokenizer_path", type=str,
                   default="src/tokenizer")
    p.add_argument(
        "--cache_dir", type=str, default=None,
        help="Local cache directory for streamed parquet shards. "
             "Wired through to the HIPPOLM_CACHE_DIR env var (the "
             "resolution path that get_cache_dir() honors). "
             "Default (null) = repo-relative .cache/hippolm/datasets. "
             "Point this at a larger disk when the default location "
             "is short on space.",
    )
    p.add_argument("--use_dummy_data", action="store_true", default=False,
                   help="Skip the streaming/tokenize path and use a "
                        "random dummy DataLoader. Fastest smoke test.")
    p.add_argument("--mb_timing", type=int, default=0,
                   help="If > 0, log per-microbatch timing breakdown "
                        "(data_ms / h2d_ms / fwd_ms / bwd_ms / sync_ms) for "
                        "the first N microbatches of every step. Use to "
                        "localize which phase of a microbatch is slow.")
    p.add_argument("--empty_cache_between_mb", type=bool, default=True,
                   help="Call torch.cuda.empty_cache() after each microbatch "
                        "(after flush_manual_flush_params) to release the "
                        "caching-allocator slack pool back to the driver. "
                        "Frees ~4 GB of pool on 16 GB production config at "
                        "the cost of ~24 ms/mb (+0.7%% wall-clock). Disable "
                        "only if benchmarking the raw allocator behavior.")
    p.add_argument(
        "--offload_strategy", type=str, default="cpu_add",
        choices=["cpu_add", "per_layer_gpu"],
        help="Grad-offload strategy. ``cpu_add`` (default, prod-proven): "
             "per-mb async D2H of every param's .grad + CPU bf16 add into "
             "the per-param accumulator. ``per_layer_gpu`` (opt-in, A/B "
             "wins 33-44%% step time at the test scale): accumulate grads "
             "per-layer on a shared GPU buffer, then async D2H + worker-"
             "thread CPU add — amortizes the D2H into the per-mb bwd tail. "
             "Persistent VRAM cost is ~80 MiB (one shared gpu_buf + "
             "small mf_buf); peak during bwd is +500 MiB at 245M scale "
             "(scales linearly with model size — budget +700 MiB at "
             "1.37B). See :mod:`src.training.param_offload.per_layer_gpu_accum` "
             "for the full design.",
    )
    p.add_argument(
        "--shuffle", type=bool, default=True,
        help="Shuffle the streaming dataset. Disable (--shuffle false) for "
             "faster first-batch on slow mirrors — sequential order is fine "
             "for short validation runs.",
    )

    # ---- Chunk-aware FFD packing ----
    p.add_argument(
        "--pack_chunk_size", type=int, default=0,
        help="Alignment granularity for doc boundaries inside a pack. "
             "Each doc rounds up to a multiple of this size so the KDA "
             "chunkwise kernel's state reset lands exactly at a doc "
             "boundary. 0 means 'use head_dim' (HippoConfig resolves). "
             "Rounded up to a multiple of 64 internally.",
    )
    p.add_argument(
        "--pack_buffer_size", type=int, default=8,
        help="How many input docs the owner prefetcher accumulates per "
             "packing window. Higher = denser FFD packs at the cost of "
             "one window's latency. Each window produces batch_size "
             "packed rows.",
    )
    p.add_argument(
        "--kda_skip_aqk_akk_saved", type=bool, default=False,
        help="Skip saving the KDA per-chunk attention statistics Aqk + "
             "Akk in the forward pass; recompute them in the backward "
             "pass via chunk_kda_fwd_intra. Trades a small bwd time "
             "increase for ~32 MiB of saved-tensor memory per KDA layer "
             "(~1 GB at 30 layers). Default False.",
    )

    # ---- GPU ----
    p.add_argument("--min_gpu_memory_mb", type=int, default=10240)
    p.add_argument("--muon_lr", type=float, default=0.02,
                   help="Learning rate for Muon (2D weight matrices).")
    p.add_argument("--muon_weight_decay", type=float, default=0.0,
                   help="Weight decay for Muon (2D weight matrices)."
                        " Default 0.0 (matches the legacy hardcoded"
                        " behavior). Set via the nested optimizer"
                        " block in the yml, or override per-run on the"
                        " CLI.")
    p.add_argument("--muon_momentum", type=float, default=0.95,
                   help="SGD momentum for the Muon path.")
    p.add_argument("--seed", type=int, default=42,
                   help="Init seed for replicated params on each device.")

    # ---- TP simulation ----
    # When --tp_sim is set we run N child processes all bound to
    # the same physical GPU and use the gloo backend to transport
    # CUDA tensors (host-staged; much slower than NVLink NCCL but
    # the sharding math is byte-for-byte the same). --tp_size is
    # only consulted when --tp_sim is set.
    p.add_argument("--tp_sim", action="store_true", default=False,
                   help="Simulate TP on a single GPU with gloo.")
    p.add_argument("--tp_size", type=int, default=2,
                   help="Simulated TP world size when --tp_sim is set."
                        " Ignored otherwise. Default 2. Practical max"
                        " is ~2-4 on a 16 GB GPU because replicated"
                        " weights are duplicated per process.")

    # ---- Precision (yml-only for now; no --precision CLI flag) ----
    # The yml schema is::
    #
    #     precision:
    #       model_weights: { dtype: fp16 }
    #       gradients:     { dtype: bf16 }
    #       activations:   { dtype: fp16 }    # fp32/fp16/bf16 only
    #       muon_momentum: { dtype: int8, scale: per-channel }
    #       adamw_m:       { dtype: bf16 }
    #       adamw_v:       { dtype: bf16 }
    #
    # ``activations`` controls ``torch.amp.autocast``: fp16 / bf16
    # enable autocast with that dtype (tensor-core matmul); fp32
    # disables autocast and runs a pure FP32 forward. Integer
    # dtypes are rejected by :class:`PrecisionConfig`.
    #
    # argparse has no native nested-dict type, so the precision
    # config flows through as a raw ``dict`` (set via
    # ``set_defaults`` from the yml payload). The training loop
    # converts it to :class:`PrecisionConfig` via
    # :meth:`PrecisionConfig.from_dict`.
    p.add_argument(
        "--precision", type=str, default=None,
        help=argparse.SUPPRESS,  # yml-only; suppress from --help
    )

    return p


def _load_yaml(path: str) -> Dict[str, Any]:
    """Tiny YAML loader used only by the overlay step. Inlined
    to avoid a circular import: ``scripts.train`` already depends
    on ``scripts.cli``, so ``scripts.cli`` cannot import from
    ``scripts.train`` (the reverse direction is also taken). The
    function is three lines so duplicating it is cheaper than
    introducing a third module.
    """
    import yaml
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _flatten_optimizer_overrides(yml_dict: Dict[str, Any]) -> None:
    """Walk a yml dict's nested ``optimizer:`` block and populate the
    flat argparse-default keys used by the training loop.

    The canonical yml shape (see ``configs/base.yml``) is::

        optimizer:
          adamw: { lr: 0.01, weight_decay: 0.01 }
          muon:  { lr: 0.02, weight_decay: 0.0, momentum: 0.95 }

    argparse has no native nested-dict type, so the CLI surface
    stays flat (``--learning_rate``, ``--weight_decay``,
    ``--muon_lr``, ``--muon_weight_decay``, ``--muon_momentum``).
    This helper unpacks the yml nested block into the same flat
    keys so the loop.py / param_offload.py reading code is
    untouched.

    Precedence: when a yml mixes the nested block with legacy
    flat keys (``learning_rate: 0.007`` AND ``optimizer.adamw.lr:
    0.01``), the FLAT key wins — explicit beats implicit. Users
    who list the old flat key meant it. The helper only writes a
    destination key when it is not already present.

    The ``optimizer:`` key itself is consumed in-place (``pop``)
    so it never reaches ``set_defaults`` as an unknown argparse
    target.
    """
    opt = yml_dict.pop("optimizer", None)
    if not isinstance(opt, dict):
        return
    # (group, nested-key, flat-argparse-name) triples.
    mappings = (
        ("adamw", "lr",            "learning_rate"),
        ("adamw", "weight_decay",  "weight_decay"),
        ("adamw", "beta1",         "adamw_beta1"),
        ("adamw", "beta2",         "adamw_beta2"),
        ("adamw", "eps",           "adamw_eps"),
        ("muon",  "lr",            "muon_lr"),
        ("muon",  "weight_decay",  "muon_weight_decay"),
        ("muon",  "momentum",      "muon_momentum"),
    )
    for group, key, dest in mappings:
        if dest in yml_dict:
            # Legacy flat key already set in the yml — keep it.
            continue
        sub = opt.get(group)
        if isinstance(sub, dict) and key in sub:
            yml_dict[dest] = sub[key]


def _load_yaml_with_extends(
    path: str,
    _visited: Optional[set] = None,
) -> Dict[str, Any]:
    """Load a YAML file with optional ``extends: <relative-path>``
    inheritance.

    Parent is loaded first, then the child's top-level keys
    shallow-override the parent's (child wins for any key it
    specifies). The ``extends`` key is consumed and never reaches
    ``set_defaults``.

    This lets the configs/test/*.yml files list only the fields
    they need to change from ``configs/base.yml`` rather than
    duplicating the full base config. The merge is shallow by
    design — predictable, easy to reason about, and matches what
    most users expect from a yml overlay. If a child needs to
    override one sub-key of a nested dict (e.g. just
    ``precision.adamw_m.dtype``), it must repeat the whole
    nested block.

    The visited set guards against circular chains
    (``a -> b -> a``) by absolute path.
    """
    import yaml
    if _visited is None:
        _visited = set()
    abs_path = str(Path(path).resolve())
    if abs_path in _visited:
        raise ValueError(
            f"Circular extends chain detected: {abs_path!r} already loaded. "
            f"Chain: {sorted(_visited)}"
        )
    _visited.add(abs_path)
    with open(path, "r") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(
            f"YAML root of {path!r} must be a mapping, got {type(data).__name__}"
        )
    extends = data.pop("extends", None)
    if extends is None:
        return data
    # Resolve relative path against the *child's* directory, not
    # the cwd — yml authors expect ``extends: ../base.yml`` to
    # mean "sibling of the parent dir", not "sibling of cwd".
    parent_path = str(Path(path).parent / extends)
    parent = _load_yaml_with_extends(parent_path, _visited)
    # Shallow merge: child wins for any key it specifies.
    merged = {**parent, **data}
    return merged


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse argv with yml-driven defaults; CLI flags win.

    Two-pass approach:

      1. ``parse_known_args`` with the hardcoded defaults to
         discover ``--config`` (or the default ``configs/base.yml``).
      2. Load the yml (with optional ``extends:`` resolution —
         see :func:`_load_yaml_with_extends`), then
         ``parser.set_defaults(**yml_dict)`` so the yml values
         become the parser defaults, then re-parse. Any explicit
         CLI flag overrides the yml-set default.

    Nested dicts (``precision: { ... }``) flow through as
    whole-dict attributes — ``set_defaults`` accepts any value,
    including a nested dict. The training loop picks up
    ``args.precision`` and converts to :class:`PrecisionConfig`.

    The yml is read but no key validation is done against the
    parser schema. Unknown keys are simply added as
    ``Namespace`` attributes, which is intentional: the precision
    block and any future yml-only field don't need a matching
    ``add_argument`` call.
    """
    parser = build_parser()
    known, _ = parser.parse_known_args(argv)
    yml_path = Path(known.config)
    if yml_path.exists():
        yml_dict = _load_yaml_with_extends(str(yml_path))
        if yml_dict:
            # Unpack the nested ``optimizer:`` block (if any) into
            # the flat argparse-default keys the training loop reads
            # (learning_rate / weight_decay / muon_lr / muon_weight_
            # decay / muon_momentum). Flat keys in the yml that
            # conflict with the nested block win (explicit beats
            # implicit). Must run before set_defaults below.
            _flatten_optimizer_overrides(yml_dict)
            parser.set_defaults(**yml_dict)
    return parser.parse_args(argv)
