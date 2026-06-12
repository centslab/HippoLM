"""Local Qwen3.5 tokenizer loader.

The tokenizer is bundled in ``src/tokenizer/`` and was copied
from ``/home/wlx/Qwen3.5-9B`` (see CLAUDE.md). :func:`load_tokenizer`
is the single entry point the training loop uses.
"""
from __future__ import annotations

from typing import Any


def load_tokenizer(tokenizer_path: str) -> Any:
    """Load a HuggingFace tokenizer from a local path.

    Falls back to setting ``pad_token = eos_token`` when the
    tokenizer has no pad token defined (Qwen3.5 family uses
    ``<|endoftext|>`` as both pad and eos). ``trust_remote_code``
    is enabled because the Qwen tokenizer ships a custom
    ``tokenization_qwen.py`` in its config dir.
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer
