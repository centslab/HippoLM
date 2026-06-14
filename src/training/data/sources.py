"""Streaming dataset loaders: ModelScope (Aliyun CDN) first, HF fallback."""
from __future__ import annotations

import logging
import os
from typing import Optional


def network_exceptions() -> tuple:
    """Tuple of exception types considered 'connection error' for fallback.

    Network/connection errors trigger the silent fallback from
    ModelScope to the HF mirror. Other exception types (HTTPError on
    4xx, ValueError on schema mismatch, etc.) are intentionally NOT
    caught here — they signal a real problem that should not be
    masked by silently switching data sources.
    """
    import socket
    import requests
    import urllib3.exceptions
    return (
        requests.exceptions.ConnectionError,
        requests.exceptions.Timeout,
        urllib3.exceptions.NewConnectionError,
        urllib3.exceptions.MaxRetryError,
        urllib3.exceptions.ConnectTimeoutError,
        urllib3.exceptions.ReadTimeoutError,
        socket.gaierror,        # DNS resolution failure
        socket.timeout,         # raw socket timeout
        TimeoutError,           # builtin; superset of socket.timeout on py3
        ConnectionError,        # builtin; OSError subclass incl. refused/reset
    )


def load_modelscope_streaming(
    ms_name: str,
    config_name: Optional[str],
    split: str,
):
    """Load a streaming dataset from ModelScope Hub (Aliyun CDN).

    Returns a NativeIterableDataset that yields dicts in the same
    shape as ``datasets.load_dataset(..., streaming=True)``, so the
    rest of the pipeline (tokenize, collate) is source-agnostic.
    """
    from modelscope.msdatasets import MsDataset

    kwargs: dict = dict(split=split, use_streaming=True)
    if config_name:
        kwargs["subset_name"] = config_name
    return MsDataset.load(ms_name, **kwargs)


def load_hf_streaming(
    hf_name: str,
    config_name: Optional[str],
    split: str,
):
    """Load a streaming dataset from HuggingFace Hub (via HF_ENDPOINT)."""
    from datasets import load_dataset

    kwargs: dict = dict(split=split, streaming=True)
    if config_name:
        kwargs["name"] = config_name
    return load_dataset(hf_name, **kwargs)


def load_streaming_with_fallback(
    hf_name: str,
    ms_name: Optional[str],
    config_name: Optional[str],
    split: str,
    use_ms: bool,
    log: logging.Logger,
):
    """Try local cached parquet first, then MS streaming, then HF streaming.

    Behavior matrix:
      - ``use_ms=False`` or ``ms_name is None``  -> HF streaming directly
      - :class:`RotatingParquetIterable` discovery + part-0 download
        succeeds                                    -> return rotating iterator
        (this is the v0.0.2 path: one-time MS API
         listing of the dataset's part-*.parquet
         files, then walk through them with 50%
         pre-download of the next part and a
         bounded local cache).
      - MS part discovery fails (HTTP error /
        unknown config)                -> MS streaming (legacy path)
      - MS streaming raises a network exception     -> warn, fall back to HF
      - MS streaming raises ``ImportError``         -> warn, fall back to HF
      - MS raises anything else (HTTPError 4xx,
        ValueError on schema, KeyError on field)    -> re-raise unchanged

    The local cache is preferred over streaming because the MS CDN
    has been observed to silently stall 100s+ minutes on a single
    row-group read when its mirror is having a bad day. Reading a
    1GB parquet from local NVMe takes ~3-5s end-to-end; reading it
    row-group by row-group from a slow CDN can take >10 min and is
    non-deterministic.
    """
    from .rotating import (
        RotatingParquetIterable,
        resolve_parts_for_config,
    )

    if use_ms and ms_name and not os.environ.get("HIPPOLM_NO_PREFETCH"):
        try:
            parts = resolve_parts_for_config(ms_name, config_name, log)
            if parts:
                log.info(
                    f"Discovered {len(parts)} part files for {ms_name}"
                    f" (subset={config_name}); using RotatingParquetIterable"
                    f" (50% pre-download, max 2 cached files)"
                )
                rot = RotatingParquetIterable(
                    ms_name, config_name, log=log,
                )
                rot.open()  # synchronous part-0 download
                return rot
        except Exception as e:
            log.warning(
                f"RotatingParquetIterable discovery failed for {ms_name}"
                f" (config={config_name}): {type(e).__name__}: {e};"
                f" falling back to MS streaming."
            )
        try:
            ds = load_modelscope_streaming(ms_name, config_name, split)
            log.info(
                f"Streaming dataset from ModelScope: {ms_name}"
                f"{f' (subset={config_name})' if config_name else ''}"
                f" (split={split})"
            )
            return ds
        except network_exceptions() as e:
            log.warning(
                f"ModelScope unreachable for {ms_name} "
                f"({type(e).__name__}: {e}); "
                f"falling back to HF endpoint "
                f"{os.environ.get('HF_ENDPOINT', 'huggingface.co')}: {hf_name}"
            )
        except ImportError as e:
            log.warning(
                f"modelscope is not installed ({e}); "
                f"falling back to HF: {hf_name}"
            )
        # Other exceptions (HTTPError on 4xx, schema errors, etc.) are
        # intentionally not caught - they indicate a real problem that
        # the HF mirror will not solve.

    ds = load_hf_streaming(hf_name, config_name, split)
    log.info(
        f"Streaming dataset from HF: {hf_name}"
        f"{f' (config={config_name})' if config_name else ''}"
        f" (split={split}, endpoint={os.environ.get('HF_ENDPOINT')},"
        f" HF_HOME={os.environ.get('HF_HOME', '~/.cache/huggingface')})"
    )
    return ds
