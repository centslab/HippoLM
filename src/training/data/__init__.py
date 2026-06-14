"""Streaming data pipeline: ModelScope/HF streaming, prefetch, batching.

Public API re-exports the leaf modules' symbols so callers can do::

    from src.training.data import StreamingDataset, PrefetchBatcher, ...
"""
from .cache import (
    cache_path_for,
    cache_path_for_part,
    get_cache_dir,
    list_cached_parts,
)
from .collate import collate_batch, ffd_pack_samples
from .dummy import dummy_dataloader
from .parquet import LocalParquetIterable
from .prefetch import ms_first_parquet_url, prefetch_first_parquet
from .prefetch_batcher import QueueIterator, PrefetchBatcher
from .rotating import (
    RotatingParquetIterable,
    config_to_subdir,
    download_part_to_cache,
    list_parquet_parts_via_api,
    resolve_parts_for_config,
)
from .sources import (
    network_exceptions,
    load_hf_streaming,
    load_modelscope_streaming,
    load_streaming_with_fallback,
)
from .streaming import MultiSourceStreamingDataset, StreamingDataset

__all__ = [
    "get_cache_dir", "cache_path_for", "cache_path_for_part",
    "list_cached_parts",
    "collate_batch", "ffd_pack_samples",
    "dummy_dataloader",
    "LocalParquetIterable",
    "ms_first_parquet_url", "prefetch_first_parquet",
    "QueueIterator", "PrefetchBatcher",
    "RotatingParquetIterable",
    "config_to_subdir", "download_part_to_cache",
    "list_parquet_parts_via_api", "resolve_parts_for_config",
    "network_exceptions",
    "load_modelscope_streaming", "load_hf_streaming", "load_streaming_with_fallback",
    "StreamingDataset",
    "MultiSourceStreamingDataset",
]
