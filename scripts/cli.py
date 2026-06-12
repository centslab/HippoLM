"""CLI argument parsing for the training entry point.

Builds the argparse parser and applies the YAML config overlay
before returning the :class:`argparse.Namespace`. The script
itself (``scripts/train.py``) imports :func:`parse_args` and
hands the result to :func:`scripts.train.train`.

Why a separate module: ``train.py`` was bloated to ~1700 lines
before the PR-3..PR-6 refactor. The CLI surface (~50 flags) is
mechanical and changes on a different cadence from the training
loop; keeping it in its own module makes both easier to read.

YAML overlay behavior: every top-level key in the YAML that
matches an existing argparse attribute overrides the CLI
default. Nested dicts (e.g. ``precision.model_weights.dtype``)
are silently dropped because ``hasattr`` does not recognize
dotted keys. This is a known limitation tracked for the
precision-wiring phase (it needs nested-dict support in the
overlay matcher).
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence


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
                   help="YAML config file. Top-level keys override the "
                        "argparse defaults below.")

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
    p.add_argument("--safe_gate", type=bool, default=False)
    p.add_argument("--lower_bound", type=float, default=None)
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
    p.add_argument(
        "--shuffle", type=bool, default=True,
        help="Shuffle the streaming dataset. Disable (--shuffle false) for "
             "faster first-batch on slow mirrors — sequential order is fine "
             "for short validation runs.",
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

    return p


def _load_yaml(path: str) -> dict:
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
    """Parse argv, apply YAML config overlay, return Namespace.

    The overlay step: for every top-level key in the YAML that
    matches an existing argparse attribute, the YAML value wins.
    ``hasattr`` is the match — nested dicts (e.g. ``precision``)
    are silently dropped because the flat matcher doesn't
    recognize them. See the module docstring for the tracked
    follow-up.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    if Path(args.config).exists():
        config_dict = _load_yaml(args.config)
        if config_dict:
            for key, value in config_dict.items():
                if hasattr(args, key):
                    setattr(args, key, value)
    return args
