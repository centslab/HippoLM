"""Pre-download the first ModelScope parquet shard for fast local iteration.

Streaming from the ModelScope CDN has been observed to silently
stall for 100s+ minutes on a single row-group read when the CDN's
mirror is slow but still answering TCP opens. The 60s socket-level
timeout (configured in :mod:`src.training.env`) converts a stall
into an exception that the retry wrapper recovers from, but the
effective throughput is then bounded by the CDN's bad day.

Workaround: download the first parquet shard (~1 GB) to a stable
local cache once, then iterate the on-disk file with pyarrow.
pyarrow reads row groups one at a time (peak RAM ~1 row group,
~250MB compressed), so loading IS streaming — the full file is
never in memory. Subsequent runs use the cache and start iterating
within ~1 second.

Cache policy: at most ONE shard is ever kept (the
``HIPPOLM_CACHE_DIR`` contains a single file). We don't
pre-download more shards because v0.0.0 validation (max_steps=1000)
does not need more data than the first shard, and 1000 steps × 4
microbatches × 2 samples = 8K samples fits easily inside the
first 64K-row group.

Set ``HIPPOLM_NO_PREFETCH=1`` to skip the pre-download and stream
from the CDN directly. Set ``HIPPOLM_CACHE_DIR=/path`` to relocate
the cache (e.g. to a larger disk if the default is constrained).

If the pre-download fails (network down, disk full, MS outage), we
fall back to streaming so the user still has a path forward.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

from .cache import cache_path_for
from .sources import network_exceptions


def ms_first_parquet_url(
    ms_name: str,
    config_name: Optional[str],
    timeout: float = 15,
) -> str:
    """Resolve the URL of the first parquet shard for a MS streaming dataset.

    The MS dataset is sharded by subset (e.g. ``qa``) and each shard
    is a snappy-compressed parquet file on the Aliyun CDN. We hit
    the canonical ``resolve/master`` URL (which 302-redirects to the
    CDN with a fresh auth_key) and return the resolved URL.

    ``timeout`` defaults to 15s; the caller may pass a longer value
    when it intends to retry on transient CDN timeouts.
    """
    import requests
    # Hard-coded path is OK here: the dataset is
    # OpenBMB/Ultra-FineWeb-L3 and the qa subset is the one we train
    # on. If the path ever changes the fallback to streaming still
    # works.
    guess_paths = [
        "data/ultrafineweb_en_l3/qa/"
        "part-00000-37dc9f21-f87f-4f43-8dd2-134424f1537a-c000.snappy.parquet",
    ]
    base = f"https://www.modelscope.cn/datasets/{ms_name}/resolve/master"
    last_err: Optional[BaseException] = None
    for path in guess_paths:
        url = f"{base}/{path}"
        try:
            r = requests.head(url, allow_redirects=True, timeout=timeout)
            if r.status_code == 200 and "Content-Length" in r.headers:
                return r.url
        except network_exceptions() as e:
            last_err = e
            continue
    raise RuntimeError(
        f"Could not resolve first parquet URL for {ms_name}"
        f" (last error: {last_err})"
    )


def prefetch_first_parquet(
    ms_name: str,
    config_name: Optional[str],
    log: logging.Logger,
) -> Optional[Path]:
    """Download the first parquet shard to the local cache.

    Returns the local path on success, or ``None`` on any failure
    (the caller should then fall back to streaming). Logs progress
    every ~5 seconds so a slow download is observable, not mysterious.
    """
    import requests
    cache_path = cache_path_for(ms_name, config_name)
    expected_size: Optional[int] = None
    try:
        # The MS CDN has been observed to spuriously ReadTimeout on
        # the first HEAD from one of the two worker processes while
        # the other worker gets through fine. A single 15s timeout
        # then pushes the unlucky worker into the MS-streaming path,
        # which can take tens of minutes to warm up; meanwhile the
        # other worker enters the TP forward loop and the model
        # all-reduce hangs for the full NCCL timeout (default 30 min)
        # waiting for the slow rank. Retry a few times with a longer
        # timeout before giving up so both ranks converge on the
        # local cache when it's available.
        auth_url: Optional[str] = None
        last_err: Optional[BaseException] = None
        for attempt in range(3):
            try:
                auth_url = ms_first_parquet_url(
                    ms_name, config_name, timeout=30,
                )
                last_err = None
                break
            except network_exceptions() as e:
                last_err = e
                if attempt < 2:
                    time.sleep(2.0 * (attempt + 1))
                    continue
        if auth_url is None:
            raise RuntimeError(
                f"MS URL HEAD kept failing after 3 attempts"
                f" (last error: {last_err})"
            )
        head = requests.head(auth_url, allow_redirects=True, timeout=15)
        expected_size = int(head.headers.get("Content-Length", 0)) or None
    except Exception as e:
        log.warning(
            f"Could not resolve MS first-shard URL ({type(e).__name__}: {e});"
            f" skipping pre-download and falling back to streaming."
        )
        return None

    if cache_path.exists() and expected_size and cache_path.stat().st_size == expected_size:
        log.info(
            f"Reusing cached parquet: {cache_path}"
            f" ({cache_path.stat().st_size / 1024**2:.1f} MB)"
        )
        return cache_path

    if cache_path.exists():
        # Partial or stale; redownload. Resumable download would be
        # nicer but the CDN does not always honor Range reliably,
        # and a fresh start is simpler and fast enough at ~1-2 MB/s.
        cache_path.unlink()

    log.info(
        f"Pre-downloading first parquet shard to {cache_path}"
        f" ({expected_size / 1024**2:.1f} MB) ..."
    )
    t0 = time.monotonic()
    got = 0
    last_log = t0
    try:
        with requests.get(auth_url, stream=True, timeout=30) as r:
            r.raise_for_status()
            tmp = cache_path.with_suffix(cache_path.suffix + ".part")
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(chunk_size=4 * 1024 * 1024):
                    if not chunk:
                        continue
                    f.write(chunk)
                    got += len(chunk)
                    now = time.monotonic()
                    if now - last_log > 5:
                        rate = got / (now - t0) / 1024
                        pct = (
                            f" ({got / expected_size * 100:.1f}%)"
                            if expected_size else ""
                        )
                        log.info(
                            f"  pre-download: {got / 1024**2:.1f} MB"
                            f" in {now - t0:.1f}s, {rate:.0f} KB/s{pct}"
                        )
                        last_log = now
            tmp.rename(cache_path)
        now = time.monotonic()
        rate = got / (now - t0) / 1024
        pct = (
            f" ({got / expected_size * 100:.1f}%)" if expected_size else ""
        )
        log.info(
            f"  pre-download: {got / 1024**2:.1f} MB in {now - t0:.1f}s,"
            f" {rate:.0f} KB/s{pct}"
        )
    except network_exceptions() as e:
        log.warning(
            f"Pre-download failed ({type(e).__name__}: {e});"
            f" falling back to streaming."
        )
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass
        return None
    except Exception as e:
        log.warning(
            f"Pre-download error ({type(e).__name__}: {e});"
            f" falling back to streaming."
        )
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass
        return None

    return cache_path
