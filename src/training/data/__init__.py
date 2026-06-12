"""Streaming data pipeline: ModelScope/HF streaming, prefetch, collate, batching.

Public API re-exports the leaf modules' symbols so callers can do::

    from src.training.data import StreamingDataset, PrefetchBatcher, ...
"""
from .cache import get_cache_dir, cache_path_for
from .collate import collate_batch
from .dummy import dummy_dataloader
from .parquet import LocalParquetIterable
from .prefetch import ms_first_parquet_url, prefetch_first_parquet
from .prefetch_batcher import QueueIterator, PrefetchBatcher
from .sources import (
    network_exceptions,
    load_modelscope_streaming,
    load_hf_streaming,
    load_streaming_with_fallback,
)
from .streaming import MultiSourceStreamingDataset, StreamingDataset

__all__ = [
    "get_cache_dir", "cache_path_for",
    "collate_batch",
    "dummy_dataloader",
    "LocalParquetIterable",
    "ms_first_parquet_url", "prefetch_first_parquet",
    "QueueIterator", "PrefetchBatcher",
    "network_exceptions",
    "load_modelscope_streaming", "load_hf_streaming", "load_streaming_with_fallback",
    "StreamingDataset",
    "MultiSourceStreamingDataset",
]
