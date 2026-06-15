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
    p.add_argument("--gdn2_mode", type=str, default="chunk",
                   choices=["chunk", "fused_recurrent"])
    p.add_argument("--use_short_conv", type=bool, default=False)
    p.add_argument("--allow_neg_eigval", type=bool, default=False)
    p.add_argument("--conv_size", type=int, default=4)
    p.add_argument("--conv_bias", type=bool, default=False)
    p.add_argument("--num_layers", type=int, default=32)
    p.add_argument("--num_blocks", type=int, default=8)
    p.add_argument("--intermediate_size", type=int, default=2736)
    p.add_argument("--rms_norm_eps", type=float, default=1e-6)

    # ---- Training ----
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--seq_len", type=int, default=512)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--max_steps", type=int, default=1000)
    p.add_argument("--gradient_accumulation_steps", type=int, default=4)
    p.add_argument("--max_grad_norm", type=float, default=1.0,
                   help="Max global (TP-reduced) L2 grad norm. "
                        "Set <= 0 to disable clipping. Default 1.0.")
    p.add_argument("--log_interval", type=int, default=10)
    p.add_argument("--checkpoint_interval", type=int, default=100,
                   help="Save a checkpoint every N steps. Set <= 0 "
                        "to disable periodic save (e.g. for short smoke runs).")

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
    p.add_argument("--use_dummy_data", action="store_true", default=False,
                   help="Skip the streaming/tokenize path and use a "
                        "random dummy DataLoader. Fastest smoke test.")
    p.add_argument("--mb_timing", type=int, default=0,
                   help="If > 0, log per-microbatch timing breakdown "
                        "(data_ms / h2d_ms / fwd_ms / bwd_ms / sync_ms) for "
                        "the first N microbatches of every step. Use to "
                        "localize which phase of a microbatch is slow.")
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
             "Each doc rounds up to a multiple of this size so the GDN2 "
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

    # ---- GPU ----
    p.add_argument("--min_gpu_memory_mb", type=int, default=10240)
    p.add_argument("--muon_lr", type=float, default=0.02,
                   help="Learning rate for Muon (2D weight matrices).")
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


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse argv with yml-driven defaults; CLI flags win.

    Two-pass approach:

      1. ``parse_known_args`` with the hardcoded defaults to
         discover ``--config`` (or the default ``configs/base.yml``).
      2. Load the yml, ``parser.set_defaults(**yml_dict)`` so the
         yml values become the parser defaults, then re-parse.
         Any explicit CLI flag overrides the yml-set default.

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
        yml_dict = _load_yaml(str(yml_path))
        if yml_dict:
            parser.set_defaults(**yml_dict)
    return parser.parse_args(argv)
