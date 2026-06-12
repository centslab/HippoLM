"""Streaming text/SFT dataset that wraps a source-agnostic iterable."""
from __future__ import annotations

import logging
from typing import Optional

from torch.utils.data import IterableDataset

from .sources import load_streaming_with_fallback


class StreamingDataset(IterableDataset):
    """Stream text samples from a ModelScope or HuggingFace dataset.

    Primary source is ModelScope (Aliyun CDN, faster). On network or
    connection errors only, falls back to HuggingFace via the
    configured ``HF_ENDPOINT`` (hf-mirror.com by default). Non-network
    errors (HTTP 4xx, schema mismatch, missing fields) propagate so
    the user sees the real failure rather than a silent source switch.

    Supports both pretraining (text/content field) and SFT (messages
    field) formats.

    Single-threaded by design: with num_workers=0 (used together with
    :class:`PrefetchBatcher`) the dataset iterates sequentially, so
    there is no per-worker duplication of HTTP fetches and no
    modulo-skip waste.
    """

    def __init__(
        self,
        dataset_name: str,
        tokenizer,
        split: str = "train",
        max_seq_len: int = 2048,
        text_field: str = "content",
        is_sft: bool = False,
        config_name: Optional[str] = None,
        seed: int = 42,
        shuffle: bool = False,
        ms_dataset_name: Optional[str] = None,
        use_modelscope: bool = True,
    ):
        super().__init__()
        self.dataset_name = dataset_name
        self.ms_dataset_name = ms_dataset_name
        self.use_modelscope = use_modelscope
        self.tokenizer = tokenizer
        self.split = split
        self.max_seq_len = max_seq_len
        self.text_field = text_field
        self.is_sft = is_sft
        self.config_name = config_name
        self.seed = seed
        self.shuffle = shuffle

        # Lazy init dataset
        self._ds = None

    def _ensure_dataset(self) -> None:
        if self._ds is None:
            self._ds = load_streaming_with_fallback(
                hf_name=self.dataset_name,
                ms_name=self.ms_dataset_name,
                config_name=self.config_name,
                split=self.split,
                use_ms=self.use_modelscope,
                log=logging.getLogger(__name__),
            )
            if self.shuffle:
                self._ds = self._ds.shuffle(seed=self.seed, buffer_size=1000)

    def _format_sft(self, example) -> Optional[str]:
        """Format SFT example as a single text string via chat template."""
        messages = example.get("messages")
        if not messages:
            return None
        try:
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            return text
        except Exception:
            return None

    def _example_to_text(self, example) -> Optional[str]:
        if self.is_sft:
            return self._format_sft(example)
        return example.get(self.text_field)

    def __iter__(self):
        self._ensure_dataset()
        for example in self._ds:
            text = self._example_to_text(example)
            if text:
                tokens = self.tokenizer(
                    text,
                    max_length=self.max_seq_len,
                    truncation=True,
                    return_tensors="pt",
                )
                input_ids = tokens["input_ids"].squeeze(0)
                if len(input_ids) >= 2:
                    yield {"input_ids": input_ids, "labels": input_ids.clone()}

    def __len__(self) -> int:
        # Streaming: arbitrary large number for any caller that needs it.
        return 1_000_000
