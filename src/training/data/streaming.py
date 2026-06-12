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


class MultiSourceStreamingDataset(IterableDataset):
    """Stream from multiple sub-datasets with a phase schedule.

    Each phase is a list of ``(sub_index, weight)`` pairs. The draw
    order within a phase repeats each sub index by its weight, e.g.
    ``[(0, 2), (1, 1)]`` produces ``[0, 0, 1, 0, 0, 1, ...]`` until
    sub 0 and sub 1 are both exhausted. Then the next phase starts.
    When all phases are exhausted, the schedule restarts from phase 0
    (infinite stream).

    Sub iterators are pulled lazily — each ``StreamingDataset`` only
    resolves its underlying source on first ``__iter__``, so a phase
    that doesn't include a given sub pays zero network cost.

    Used by the "ultra fineweb L3 multi-source" loader: phase 0 is
    multi-style (en *2 + zh *1), phase 1 is QA (en *2 + zh *1).
    """

    def __init__(self, sub_datasets, phase_schedule):
        """
        Args:
            sub_datasets: list of underlying iterable datasets
                (typically :class:`StreamingDataset` instances). Each
                must be a tokenized stream yielding ``{input_ids, labels}``
                dicts so the upstream :class:`PrefetchBatcher` sees a
                uniform sample shape.
            phase_schedule: list of phases; each phase is a list of
                ``(sub_index, weight)`` pairs. ``sub_index`` must be
                in ``[0, len(sub_datasets))`` and ``weight >= 1``.
        """
        super().__init__()
        if not sub_datasets:
            raise ValueError(
                "MultiSourceStreamingDataset: sub_datasets must be non-empty"
            )
        if not phase_schedule:
            raise ValueError(
                "MultiSourceStreamingDataset: phase_schedule must be non-empty"
            )
        n = len(sub_datasets)
        for pi, phase in enumerate(phase_schedule):
            for idx, w in phase:
                if not (0 <= idx < n):
                    raise ValueError(
                        f"phase {pi} references sub index {idx}"
                        f" but only {n} sub_datasets provided"
                    )
                if w < 1:
                    raise ValueError(
                        f"phase {pi} weight must be >= 1, got {w} for sub {idx}"
                    )
        self._subs = list(sub_datasets)
        self._phases = phase_schedule

    def __iter__(self):
        # Pre-compute the per-phase draw list, e.g. [0, 0, 1] for
        # phase [(0, 2), (1, 1)]. Sampling from this list cyclically
        # is O(1) per draw and is what gives the 2:1 en:zh weighting.
        phase_draw_lists = []
        for phase in self._phases:
            draw = []
            for idx, w in phase:
                draw.extend([idx] * w)
            phase_draw_lists.append(draw)

        # Infinite stream: each outer-loop iteration walks every
        # phase. A phase completes when every sub in it has been
        # drained; a single sub dying mid-phase is fine — we just
        # skip its draw slots for the rest of that phase. This
        # matches the "multi-style first, then QA" contract at the
        # block level rather than the sample level.
        while True:
            for draw_list in phase_draw_lists:
                # Lazily create per-sub iterators on phase entry so
                # phases that don't reference a given sub still pay
                # zero cost. ``dict.get`` returns None for skipped subs.
                iters = {}
                for idx in set(draw_list):
                    iters[idx] = iter(self._subs[idx])

                any_alive = True
                while any_alive:
                    any_alive = False
                    for idx in draw_list:
                        sub_iter = iters.get(idx)
                        if sub_iter is None:
                            continue
                        try:
                            yield next(sub_iter)
                            any_alive = True
                        except StopIteration:
                            iters[idx] = None  # mark exhausted

    def __len__(self) -> int:
        # Streaming: arbitrary large number for any caller that needs it.
        return 1_000_000
