"""Training script for HippoLM v0.0.1.

Features (v0.0.1):
- Contiguous 2/4/8 GPU selection (largest viable run, never odd)
- Megatron-style TP (column-row parallel FFN, column-parallel
  lm_head, replicated KDA / AttnRes / RMSNorm / embed)
- Muon optimizer (Newton-Schulz on GPU) for 2D Linear weights
- AdamW for 1D params (norms, queries) and lm_head / embed
- CPU-offloaded optimizer state (m, v, momentum) on pinned memory
- No gradients stored on GPU: each micro-batch streams
  ``.grad`` to the CPU accumulators, frees the GPU copy, and
  only triggers a real optimizer step after
  ``gradient_accumulation_steps`` micro-batches
- Background-thread streaming data prefetch
- Persistent HF cache; ModelScope preferred, HF mirror fallback
- Qwen3.5 tokenizer (local)
- max_step training limit

Replaces v0.0.0's DataParallel-with-GPU-AdamW path.

TP simulation (development on a single-GPU box, e.g. one 5060 Ti 16G)
-----------------------------------------------------------------------
Pass ``--tp_sim`` to exercise the TP code paths without owning N
physical GPUs. Internally we:
  1. Keep ``get_available_gpus`` exactly as-is (still scans for
     contiguous 2/4/8 runs of GPUs with >= 10 GB free).
  2. If the scan returns [] (the single-GPU case), fall back to
     ``cuda:0``.
  3. Expand the GPU list to ``[gpus[0]] * tp_size`` so the
     downstream mp.spawn launches N child processes all bound to
     the same physical device.
  4. Swap NCCL for the gloo backend. gloo can transport CUDA
     tensors (it D2H-stages each collective, so it's much slower
     than NVLink NCCL — we are testing correctness, not speed).
  5. Sharding math (out_features // world, Column/RowParallelLinear,
     replicated-broadcast, lm_head narrow view, fused-CE TP-aware
     all-reduce) is byte-for-byte the same code as the real path;
     only the transport differs.

Memory note: replicated modules (KDA / AttnRes / embed / RMSNorm)
are constructed independently in every process, so N processes
each carry a full copy. On a 16 GB GPU the practical ceiling is
``--tp_size=2`` (use with caution) or ``--tp_size=4`` if you shrink
the config; ``--tp_size=8`` will OOM. Use ``--use_dummy_data`` for
the cheapest smoke run.
"""

# CRITICAL: HF_ENDPOINT and HF_TOKEN must be set BEFORE huggingface_hub is
# imported, because HfApi reads the env vars at construction time. Set them
# at the very top, before any other imports, so that any chain import of
# huggingface_hub (e.g. via `datasets` or `transformers`) picks them up.
import math
import os as _os
from pathlib import Path as _Path

# Minimal .env loader — no python-dotenv dependency.
_env_path = _Path(__file__).resolve().parent.parent / ".env"
if _env_path.is_file():
    for _line in _env_path.read_text().splitlines():
        _line = _line.strip()
        if not _line or _line.startswith("#") or "=" not in _line:
            continue
        _k, _v = _line.split("=", 1)
        _os.environ.setdefault(_k.strip(), _v.strip())

if "HF_ENDPOINT" not in _os.environ:
    _os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# HF_HOME / cache default to ~/.cache/huggingface (persistent). We only
# override them if the user has not already set them in the environment.
# This way repeated runs reuse downloaded parquet shards instead of
# paying the 3-hop redirect chain on every startup.

# --- NCCL transport pinning -------------------------------------------
# On this 8xV100-SXM2 box, GPUs 0-3 belong to a vLLM TP deployment
# and GPU 4 to ComfyUI; we must not touch them. ``nvidia-smi topo -m``
# shows the training pair (e.g. 5,6) connected via NVLink (NV1) so
# inter-GPU traffic should stay on the NVLink fabric, not fall through
# to PCIe/SYS (which would be ~5x slower and would also let NCCL probe
# the vLLM GPUs).
#
# Set these BEFORE the first import of torch.distributed, so they are
# in effect when NCCL initializes. ``setdefault`` leaves anything the
# user pinned in the environment (e.g. for an NCCL_DEBUG run) alone.
_os.environ.setdefault("NCCL_P2P_LEVEL", "NVL")
_os.environ.setdefault("NCCL_IB_DISABLE", "1")
# Force NCCL to use only the loopback / IB-disallowed path; on a
# single-node job the cross-NIC traffic is never useful and can
# otherwise steal a few hundred ms at startup.
_os.environ.setdefault("NCCL_SOCKET_IFNAME", "^lo,docker,veth,br-")
# Disable the NCCL heartbeat watchdog. The first step compiles
# many Triton kernels (chunk_kda fwd/bwd × 32 layers, fused CE,
# fused RMSNormGated) which can hold the GIL for >480s, making
# the watchdog falsely report a hang and abort the job.
_os.environ.setdefault("TORCH_NCCL_ENABLE_MONITORING", "0")
# Belt and braces: also widen the heartbeat timeout in case the
# monitoring flag is interpreted differently in this PyTorch build.
_os.environ.setdefault("TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC", "7200")

import os
import sys
import argparse
import logging
import time
import queue
import threading
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader, IterableDataset

sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

# ---------------------------------------------------------------------------
# datasets streaming read retries
# ---------------------------------------------------------------------------
# ModelScope streaming for parquet datasets fetches a streaming reader
# wrapped by ``datasets.utils.file_utils._add_retries_to_file_obj_read_method``,
# which on a mid-stream disconnect sleeps one of these intervals and
# retries up to ``*_MAX_RETRIES`` times:
#
#   CONNECTION_ERRORS_TO_RETRY     -> STREAMING_READ_RETRY_INTERVAL         (5s)
#   HTTP 503 SERVER_UNAVAILABLE     -> STREAMING_READ_SERVER_UNAVAILABLE_…  (20s)
#   HTTP 429 RATE_LIMIT             -> STREAMING_READ_RATE_LIMIT_RETRY_…    (60s)
#   open(...) failures              -> STREAMING_OPEN_*                    (5s)
#
# Combined with the default ``MAX_RETRIES=20`` the worst case is
# ``20 × 60s = 1200s`` of silent sleeping, which is what the init
# "hang" actually is.  The log shows cdn-lfs-cn-1.modelscope.cn
# returning 503 partway through a 1GB parquet read; we observe
# 100s+ of empty time between log lines, exactly matching
# ``20 × 5s = 100s`` of unrecoverable ``time.sleep``.
#
# Tighten the *retries* budget (50, 5s total) AND the *interval* on the
# slow paths (503 → 1s, 429 → 2s) so transient CDN disconnects are
# recovered transparently inside the streaming read rather than
# surfacing as a 100s+ wall-clock stall.  The HF mirror is also routed
# through the same wrapper, so this tuning helps its disconnects too.
import datasets as _datasets_for_cfg
_datasets_for_cfg.config.STREAMING_READ_MAX_RETRIES = 50
_datasets_for_cfg.config.STREAMING_READ_RETRY_INTERVAL = 0.1
_datasets_for_cfg.config.STREAMING_OPEN_MAX_RETRIES = 10
_datasets_for_cfg.config.STREAMING_OPEN_RETRY_INTERVAL = 0.5
_datasets_for_cfg.config.STREAMING_READ_SERVER_UNAVAILABLE_RETRY_INTERVAL = 1
_datasets_for_cfg.config.STREAMING_READ_RATE_LIMIT_RETRY_INTERVAL = 2
del _datasets_for_cfg

# Default socket timeout for streaming HTTP reads. Without this, a
# CDN that accepts the connection then stops sending bytes (silent
# stall, no 503, no TCP RST) leaves ``datasets`` / ``modelscope``
# blocked in a recv() until the OS TCP keepalive fires (default
# ~7200s on Linux). Setting 60s means a stalled connection raises
# a timeout exception after 1 minute, which the retry config above
# then turns into a transparent reconnect rather than a wall-clock
# hang. Healthy chunks of this dataset arrive in 1-3s, so 60s is
# generous headroom for slow CDN hops.
import socket as _socket
_socket.setdefaulttimeout(60.0)
del _socket

from configs.base_config import HippoConfig


def setup_logging(log_file: Path | None = None):
    """Setup logging with optional file output."""
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
    )
    return logging.getLogger(__name__)


def get_available_gpus(min_memory_mb: int = 10240) -> list[int]:
    """Select a contiguous run of 2/4/8 GPUs, each with at least
    ``min_memory_mb`` free VRAM.

    Strategy: collect the indices of all eligible GPUs (in
    increasing physical order), then look for the largest
    contiguous run of length 2, 4, or 8 and return that. 1, 3, 5,
    6, 7 are intentionally not supported — odd counts would force
    uneven sharding and 6/7 are not power-of-two friendly. The
    largest viable prefix is preferred (8 > 4 > 2) so we don't
    under-utilize the box.
    """
    if not torch.cuda.is_available():
        return []

    eligible: list[int] = []

    try:
        import pynvml

        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            free_mb = info.free / (1024 ** 2)
            if free_mb >= min_memory_mb:
                eligible.append(i)
        pynvml.nvmlShutdown()
    except ImportError:
        import subprocess

        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=index,memory.free",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, check=True,
            )
            for line in result.stdout.strip().split("\n"):
                parts = line.split(",")
                if len(parts) >= 2:
                    idx = int(parts[0].strip())
                    free = float(parts[1].strip())
                    if free >= min_memory_mb:
                        eligible.append(idx)
        except FileNotFoundError:
            for i in range(torch.cuda.device_count()):
                total = torch.cuda.get_device_properties(i).total_memory / (1024 ** 2)
                if total >= min_memory_mb:
                    eligible.append(i)

    # Respect CUDA_VISIBLE_DEVICES if set.
    visible = torch.cuda.device_count()
    eligible = [g for g in eligible if g < visible]
    if not eligible:
        eligible = list(range(visible))

    # Find the largest contiguous run of size 2, 4, or 8.
    for target in (8, 4, 2):
        for start in range(len(eligible) - target + 1):
            window = eligible[start:start + target]
            if window[-1] - window[0] == target - 1:
                # All contiguous and in order.
                return window

    return []


def load_tokenizer(tokenizer_path: str):
    """Load Qwen3.5 tokenizer from local path."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


# Network/connection-related exception types. If loading from ModelScope
# fails with one of these, we transparently retry on the HF mirror. Other
# exception types (HTTPError on 4xx, ValueError on schema mismatch, etc.)
# are intentionally NOT caught here — they signal a real problem that
# should not be masked by silently switching data sources.
def _network_exceptions() -> tuple:
    """Tuple of exception types considered 'connection error' for fallback."""
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


# ---------------------------------------------------------------------------
# Local parquet cache: pre-download the first shard, iterate from disk.
# ---------------------------------------------------------------------------
# Streaming from the ModelScope CDN has been observed to silently stall for
# 100s+ minutes on a single row-group read when the CDN's mirror is slow
# but still answering TCP opens. The 60s socket-level timeout converts a
# stall into an exception that the retry wrapper recovers from, but the
# effective throughput is then bounded by the CDN's bad day.
#
# Workaround: download the first parquet shard (~1 GB) to a stable local
# cache once, then iterate the on-disk file with pyarrow. pyarrow reads
# row groups one at a time (peak RAM ~1 row group, ~250MB compressed),
# so loading IS streaming — the full file is never in memory. Subsequent
# runs use the cache and start iterating within ~1 second.
#
# Cache policy: at most ONE shard is ever kept (``_HIPPOLM_CACHE_DIR``
# contains a single file). We don't pre-download more shards because
# v0.0.0 validation (max_steps=1000) does not need more data than the
# first shard, and 1000 steps × 4 microbatches × 2 samples = 8K samples
# fits easily inside the first 64K-row group.
#
# Set ``HIPPOLM_NO_PREFETCH=1`` to skip the pre-download and stream
# from the CDN directly. Set ``HIPPOLM_CACHE_DIR=/path`` to relocate
# the cache (e.g. to a larger disk if the default is constrained).
#
# If the pre-download fails (network down, disk full, MS outage), we
# fall back to streaming so the user still has a path forward.

_HIPPOLM_CACHE_DIR = _Path(_os.environ.get("HIPPOLM_CACHE_DIR") or str(_Path.home() / ".cache" / "hippolm" / "datasets"))
_HIPPOLM_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Per-microbatch loss logging threshold: log every step for the
# first N accumulated steps so we can see when nan/inf first
# appears. Set via HIPPOLM_DIAG_STEPS env var (default 5).
_DIAG_STEPS = int(_os.environ.get("HIPPOLM_DIAG_STEPS", "5"))


def _ms_first_parquet_url(ms_name: str, config_name: Optional[str], timeout: float = 15) -> str:
    """Resolve the URL of the first parquet shard for a MS streaming dataset.

    The MS dataset is sharded by subset (e.g. ``qa``) and each shard is a
    snappy-compressed parquet file on the Aliyun CDN. We hit the
    canonical ``resolve/master`` URL (which 302-redirects to the CDN
    with a fresh auth_key) and return the resolved URL.

    ``timeout`` defaults to 15s; the caller may pass a longer value when
    it intends to retry on transient CDN timeouts.
    """
    import requests
    # Hard-coded path is OK here: the dataset is OpenBMB/Ultra-FineWeb-L3
    # and the qa subset is the one we train on. If the path ever changes
    # the fallback to streaming still works.
    guess_paths = [
        f"data/ultrafineweb_en_l3/qa/part-00000-37dc9f21-f87f-4f43-8dd2-134424f1537a-c000.snappy.parquet",
    ]
    base = f"https://www.modelscope.cn/datasets/{ms_name}/resolve/master"
    last_err: Optional[BaseException] = None
    for path in guess_paths:
        url = f"{base}/{path}"
        try:
            r = requests.head(url, allow_redirects=True, timeout=timeout)
            if r.status_code == 200 and "Content-Length" in r.headers:
                return r.url
        except _network_exceptions() as e:
            last_err = e
            continue
    raise RuntimeError(
        f"Could not resolve first parquet URL for {ms_name}"
        f" (last error: {last_err})"
    )


def _prefetch_first_parquet(
    ms_name: str,
    config_name: Optional[str],
    log: logging.Logger,
) -> Optional[Path]:
    """Download the first parquet shard to ``~/.cache/hippolm/datasets``.

    Returns the local path on success, or ``None`` on any failure (the
    caller should then fall back to streaming). Logs progress every
    ~5 seconds so a slow download is observable, not mysterious.
    """
    import requests
    cache_path = _HIPPOLM_CACHE_DIR / f"{ms_name.replace('/', '__')}__{config_name or 'default'}__part0.snappy.parquet"
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
                auth_url = _ms_first_parquet_url(
                    ms_name, config_name, timeout=30,
                )
                last_err = None
                break
            except _network_exceptions() as e:
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
        # Partial or stale; redownload. Resumable download would be nicer
        # but the CDN does not always honor Range reliably, and a fresh
        # start is simpler and fast enough at ~1-2 MB/s.
        cache_path.unlink()

    log.info(
        f"Pre-downloading first parquet shard to {cache_path}"
        f" ({expected_size / 1024**2:.1f} MB) ..."
    )
    t0 = time.monotonic()
    got = 0
    last_log = t0
    rate = 0.0
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
    except _network_exceptions() as e:
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


class _LocalParquetIterable:
    """Yield examples from a local parquet shard via pyarrow row groups.

    The MS streaming reader returns dicts with the same keys as the
    parquet schema (``content`` for text, ``messages`` for SFT). This
    wrapper does the same, so the existing ``HFStreamingDataset`` text
    / SFT path works unchanged.

    Iteration is row-group-by-row-group, not row-by-row, so the
    inner loop is fast even on multi-GB files.
    """

    def __init__(self, path: Path):
        self._path = Path(path)

    def __iter__(self):
        import pyarrow.parquet as pq
        pf = pq.ParquetFile(str(self._path))
        # ``iter_batches`` reads one row group at a time and yields an
        # Arrow RecordBatch. Converting to pandas then to dicts is
        # cheap for our schema (1-2 text columns) and avoids the
        # per-row Python overhead of ``to_pylist``.
        for batch in pf.iter_batches(batch_size=1024):
            df = batch.to_pandas()
            for _, row in df.iterrows():
                yield row.to_dict()


def _load_modelscope_streaming(
    ms_name: str,
    config_name: Optional[str],
    split: str,
):
    """Load a streaming dataset from ModelScope Hub (Aliyun CDN).

    Returns a NativeIterableDataset that yields dicts in the same shape
    as ``datasets.load_dataset(..., streaming=True)``, so the rest of
    the pipeline (tokenize, collate) is source-agnostic.
    """
    from modelscope.msdatasets import MsDataset

    kwargs: dict = dict(split=split, use_streaming=True)
    if config_name:
        kwargs["subset_name"] = config_name
    return MsDataset.load(ms_name, **kwargs)


def _load_hf_streaming(
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


def _load_streaming_with_fallback(
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
      - Local cached parquet available              -> use ``_LocalParquetIterable``
      - MS streaming succeeds                       -> return MS iterator
      - MS raises ``_network_exceptions``           -> warn, fall back to HF
      - MS raises ``ImportError`` (not installed)   -> warn, fall back to HF
      - MS raises anything else (HTTPError 4xx,
        ValueError on schema, KeyError on field)    -> re-raise unchanged

    The local cache is preferred over streaming because the MS CDN has
    been observed to silently stall 100s+ minutes on a single row-group
    read when its mirror is having a bad day. Reading a 1GB parquet
    from local NVMe takes ~3-5s end-to-end; reading it row-group by
    row-group from a slow CDN can take >10 min and is non-deterministic.
    """
    if use_ms and ms_name and not _os.environ.get("HIPPOLM_NO_PREFETCH"):
        cached = _prefetch_first_parquet(ms_name, config_name, log)
        if cached is not None:
            log.info(
                f"Using local cached parquet for {ms_name}"
                f" (subset={config_name}): {cached}"
            )
            return _LocalParquetIterable(cached)
        try:
            ds = _load_modelscope_streaming(ms_name, config_name, split)
            log.info(
                f"Streaming dataset from ModelScope: {ms_name}"
                f"{f' (subset={config_name})' if config_name else ''}"
                f" (split={split})"
            )
            return ds
        except _network_exceptions() as e:
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

    ds = _load_hf_streaming(hf_name, config_name, split)
    log.info(
        f"Streaming dataset from HF: {hf_name}"
        f"{f' (config={config_name})' if config_name else ''}"
        f" (split={split}, endpoint={os.environ.get('HF_ENDPOINT')},"
        f" HF_HOME={os.environ.get('HF_HOME', '~/.cache/huggingface')})"
    )
    return ds


def load_config(config_path: str) -> dict:
    """Load training config from YAML file."""
    import yaml

    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def create_output_dir(output_dir: str) -> Path:
    """Create timestamped output directory."""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = Path(output_dir) / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / "checkpoints").mkdir(exist_ok=True)
    (run_dir / "logs").mkdir(exist_ok=True)
    (run_dir / "eval").mkdir(exist_ok=True)

    return run_dir


class HFStreamingDataset(IterableDataset):
    """Stream text samples from a ModelScope or HuggingFace dataset.

    Primary source is ModelScope (Aliyun CDN, faster). On network or
    connection errors only, falls back to HuggingFace via the configured
    ``HF_ENDPOINT`` (hf-mirror.com by default). Non-network errors
    (HTTP 4xx, schema mismatch, missing fields) propagate so the user
    sees the real failure rather than a silent source switch.

    Supports both pretraining (text/content field) and SFT (messages field) formats.

    Single-threaded by design: with num_workers=0 (used together with
    ``_PrefetchBatcher`` below) the dataset iterates sequentially, so there is
    no per-worker duplication of HTTP fetches and no modulo-skip waste.
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

        # Note: HF_ENDPOINT and HF_TOKEN are set at module import time
        # (top of this file) so that huggingface_hub picks them up before
        # HfApi() caches its default endpoint / auth token.

        # Lazy init dataset
        self._ds = None

    def _ensure_dataset(self):
        if self._ds is None:
            self._ds = _load_streaming_with_fallback(
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

    def __len__(self):
        # Streaming: arbitrary large number for any caller that needs it.
        return 1_000_000


def _collate_batch(samples):
    """Stack a list of per-sample dicts into a batched dict.

    Each sample is ``{"input_ids": [T], "labels": [T]}``. Lengths may differ,
    so we right-pad to the longest sequence in the batch with the tokenizer's
    pad token id (or 0 if none is set).
    """
    pad_id = 0
    if "input_ids" in samples[0]:
        first = samples[0]["input_ids"]
        if hasattr(first, "new_full"):
            pad_id = 0
    max_len = max(s["input_ids"].size(0) for s in samples)
    out_ids = []
    out_labels = []
    for s in samples:
        ids = s["input_ids"]
        labs = s["labels"]
        pad = max_len - ids.size(0)
        if pad:
            ids = torch.cat([ids, ids.new_full((pad,), pad_id)])
            labs = torch.cat([labs, labs.new_full((pad,), -100)])
        out_ids.append(ids)
        out_labels.append(labs)
    return {
        "input_ids": torch.stack(out_ids, dim=0),
        "labels": torch.stack(out_labels, dim=0),
    }


class _QueueIterator:
    """Iterate a queue (``multiprocessing.Queue`` or ``queue.Queue``)
    until a ``None`` sentinel, re-raising any exception object the
    producer put in our place.

    This is the **single read path** for every TP rank — the owner's
    prefetch worker pushes each batch into every rank's queue, and
    every rank (including the owner) pulls from its own queue with
    one of these iterators. No more owner / non-owner branching at
    the consumer side.
    """

    # ``None`` is a singleton in CPython and survives pickle round-
    # trips, so it works as the end-of-stream marker for both
    # in-process ``queue.Queue`` and cross-process ``mp.Queue``.
    _SENTINEL = None

    def __init__(self, q):
        self._q = q

    def __iter__(self):
        return self

    def __next__(self):
        item = self._q.get()
        if item is self._SENTINEL:
            raise StopIteration
        if isinstance(item, BaseException):
            raise item
        return item


class _PrefetchBatcher:
    """One-deep async prefetch fanning out into per-rank queues.

    Constructed **only on the TP owner rank**. Spawns one daemon
    worker that loops::

        fetch batch  ->  for q in queues: q.put(batch)  ->  repeat

    The queue ``maxsize=2`` caps in-flight batches at 2 per rank.
    When any consumer is slow, the worker blocks on that consumer's
    ``queue.put`` — correct backpressure for TP. On end-of-stream a
    ``None`` sentinel is broadcast to every queue; on exception the
    exception object itself is broadcast (each consumer's
    :class:`_QueueIterator` re-raises it).

    There is no owner-side fast path: the owner reads from its own
    queue via :class:`_QueueIterator` exactly like every non-owner
    rank, so the consumer code is uniform across ranks.
    """

    _SENTINEL = None

    def __init__(self, dataset, batch_size: int, queues):
        """
        Args:
            dataset: an iterable yielding raw samples.
            batch_size: number of samples to collate per output batch.
            queues: list of queue-likes (``mp.Queue`` cross-process,
                ``queue.Queue`` in-process), one per consumer rank
                **including the owner's own queue**. The worker
                broadcasts each batch into every queue here.
        """
        if not queues:
            raise ValueError("_PrefetchBatcher: queues must be non-empty")
        self._batch_size = batch_size
        self._queues = list(queues)
        self._iter = iter(dataset)
        self._stop = threading.Event()
        self._log = logging.getLogger(__name__)
        self._t0 = time.monotonic()
        self._first_sample_logged = False
        self._worker = threading.Thread(
            target=self._worker_loop, name="hf-prefetch", daemon=True,
        )
        self._worker.start()

    def _fetch_one(self):
        """Pull ``batch_size`` samples from the dataset and collate.

        Returns the collated batch dict, or :attr:`_SENTINEL` if the
        dataset iterator is exhausted before any sample was read.
        """
        batch = []
        for _ in range(self._batch_size):
            try:
                batch.append(next(self._iter))
            except StopIteration:
                break
        if not batch:
            return self._SENTINEL
        if not self._first_sample_logged:
            self._log.info(
                f"prefetch: first sample after"
                f" {time.monotonic() - self._t0:.2f}s"
            )
            self._first_sample_logged = True
        return _collate_batch(batch)

    def _broadcast(self, item) -> None:
        """Put ``item`` into every consumer queue. Blocking — slow
        consumers gate the worker, which is the backpressure we want."""
        for q in self._queues:
            q.put(item)

    def _worker_loop(self) -> None:
        while not self._stop.is_set():
            try:
                result = self._fetch_one()
            except BaseException as e:  # noqa: BLE001 — propagate
                self._broadcast(e)
                return
            self._broadcast(result)
            if result is self._SENTINEL:
                return  # end of stream; consumers will raise StopIteration

    def close(self, timeout: float = 2.0) -> None:
        """Signal the worker to stop and wait briefly for it to exit.

        Worker is a daemon thread, so a stuck ``queue.put`` (e.g.
        consumer abandoned mid-stream) will be killed at interpreter
        exit even if the join times out.
        """
        self._stop.set()
        if self._worker is not None:
            self._worker.join(timeout=timeout)


def create_dummy_dataloader(batch_size: int, seq_len: int, vocab_size: int):
    """Create a dummy dataloader for testing."""
    class DummyDataset(torch.utils.data.Dataset):
        def __init__(self, size=1000):
            self.size = size

        def __len__(self):
            return self.size

        def __getitem__(self, idx):
            input_ids = torch.randint(0, vocab_size, (seq_len,))
            return {"input_ids": input_ids, "labels": input_ids.clone()}

    return DataLoader(DummyDataset(), batch_size=batch_size, shuffle=True)


def _compute_and_clip_grad_norm(
    optimizers: list,
    max_norm: float,
) -> float:
    """Compute the TP-reduced L2 norm of the accumulated grads and
    clip them in place if the norm exceeds ``max_norm``.

    v0.0.2: the grads live on the GPU in FP32 (cast in
    :func:`accumulate_grads_to_cpu`). The squared sum is computed
    in FP32 on the GPU and all-reduced across the TP group, so a
    parameter that is sharded across N ranks contributes ``1/N`` of
    its squared L2 to each rank's local sum, and the reduction
    makes the global norm exact.

    Returns the **pre-clip** total norm. If ``max_norm <= 0`` the
    function is a no-op and returns 0.0 (caller can use this to
    detect "clipping disabled" without an extra flag).
    """
    if max_norm <= 0:
        return 0.0
    import torch.distributed as dist
    from src.models.tp_layers import get_tp_group, get_tp_world_size

    # Grab the device from the first param we see. The optimizer
    # state is keyed on the param itself, and the param was created
    # on ``gpus[rank]`` during model construction, so its
    # ``.device`` is guaranteed to be the device the NCCL group is
    # bound to. We deliberately do NOT use ``torch.cuda.current_device()``
    # here: it's a per-thread default that can drift away from the
    # NCCL backend's device constraint.
    reduce_device: Optional[torch.device] = None
    n_tensors = 0
    # Per-tensor squared sums on the GPU (FP32). We accumulate
    # them in a Python list and reduce with one all-reduce of the
    # total at the end.
    gpu_sq_sums: list[torch.Tensor] = []
    for opt in optimizers:
        for s in opt.state.values():
            g = s.param.grad
            if g is None:
                continue
            if reduce_device is None:
                reduce_device = g.device
            # g is FP32 (cast by accumulate_grads_to_cpu). pow(2)
            # and sum stay on the GPU; the result is a 1-element
            # tensor that we keep on the device for the all-reduce.
            gpu_sq_sums.append(g.detach().float().pow(2).sum())
            n_tensors += 1

    if n_tensors == 0 or reduce_device is None:
        return 0.0

    total_sq = torch.stack(gpu_sq_sums).sum()

    # TP-reduce the squared norm. Each rank owns a disjoint shard
    # of the sharded params, so the per-rank ``total_sq`` is
    # ``1/world_size`` of the global squared norm; ``SUM`` recovers
    # the global value. When world_size == 1 the reduction is a
    # no-op (we still call it through the default group, which is
    # a no-op for world_size==1 because we wrap in an if).
    if get_tp_world_size() > 1:
        dist.all_reduce(total_sq, op=dist.ReduceOp.SUM, group=get_tp_group())
    total_norm = total_sq.sqrt().item()

    if total_norm > max_norm:
        scale = max_norm / (total_norm + 1e-6)
        for opt in optimizers:
            for s in opt.state.values():
                g = s.param.grad
                if g is not None:
                    g.mul_(scale)

    return total_norm


def save_checkpoint(model, optimizer, scaler, step, loss, checkpoint_dir: Path):
    """Save training checkpoint."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = checkpoint_dir / f"checkpoint_step_{step}.pt"
    torch.save({
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "loss": loss,
    }, path)
    return path


def _train_worker(
    rank: int,
    args,
    gpus: list[int],
    port: int,
    run_dir_path: str,
    shared_batch_queues=None,
):
    """Single-worker training loop (one process per GPU).

    Each worker:
      - Initializes the NCCL process group with rank=rank, world=len(gpus)
      - Builds its own TP model fragment (TPHippoModel constructs all
        device fragments; each worker only uses its own slot)
      - Builds per-device optimizers (Muon for 2D, AdamW for 1D + lm_head)
      - Runs the data-parallel-style training loop: forward, backward,
        stream grad to CPU, accumulate, optimizer step
      - Logs to a per-worker log file under ``run_dir/logs``

    The replicated modules (KDA, embed, RMSNorm, BlockAttnRes) are
    broadcast from worker 0 to all others at construction time, so
    the params on every device are identical. The sharded modules
    (FFN, lm_head) are per-device from the start.
    """
    import torch.distributed as dist

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(len(gpus))
    os.environ["LOCAL_RANK"] = str(rank)

    # Bind this process to its training GPU BEFORE NCCL initializes.
    # Without this, PyTorch's default CUDA device is cuda:0 (the first
    # visible device), and NCCL's init_process_group infers the comm
    # device from current_device() — so it allocates its comm buffers
    # on cuda:0. On this 8xV100 box cuda:0 is occupied by vLLM (only
    # ~239 MiB free), and the first barrier() / all-reduce from inside
    # the model construction blows up with cudaErrorMemoryAllocation
    # at the next synchronous NCCL call.
    #
    # Fix: set the default device first, then pass device_id= to
    # init_process_group as belt-and-braces. The device_id= argument
    # is the forward-compatible way to pin the NCCL comm to a specific
    # GPU; set_device also silences NCCL's "Guessing device ID" warning.
    #
    # TP simulation (``--tp_sim``) uses the gloo backend instead of
    # NCCL: gloo transports CUDA tensors via host staging, so it has
    # no device_id concept and the "pin comm to a specific GPU" trick
    # doesn't apply. The sharding math is identical; only the comm
    # transport changes.
    torch.cuda.set_device(gpus[rank])
    backend = "gloo" if args.tp_sim else "nccl"
    pg_kwargs: dict = dict(
        backend=backend,
        init_method="env://",
        rank=rank,
        world_size=len(gpus),
    )
    if backend == "nccl":
        pg_kwargs["device_id"] = torch.device(f"cuda:{gpus[rank]}")
    dist.init_process_group(**pg_kwargs)

    # Per-worker logging.
    run_dir = Path(run_dir_path)
    log_file = run_dir / "logs" / f"train_rank{rank}.log"
    global logger
    logger = setup_logging(log_file)
    logger.info(
        f"Worker {rank}/{len(gpus)} starting on cuda:{gpus[rank]}"
    )

    # ---- Model config ----
    config = HippoConfig(
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        tie_word_embeddings=args.tie_word_embeddings,
        use_bias=args.use_bias,
        num_heads=args.num_heads,
        head_dim=args.head_dim,
        expand_v=args.expand_v,
        kda_mode=args.kda_mode,
        use_short_conv=args.use_short_conv,
        allow_neg_eigval=args.allow_neg_eigval,
        safe_gate=args.safe_gate,
        lower_bound=args.lower_bound,
        conv_size=args.conv_size,
        conv_bias=args.conv_bias,
        num_layers=args.num_layers,
        num_blocks=args.num_blocks,
        intermediate_size=args.intermediate_size,
        rms_norm_eps=args.rms_norm_eps,
    )
    if rank == 0:
        logger.info(f"Model config: {config}")

    # ---- Data (centralised: rank 0 owns the iterator, all ranks consume
    # the same shared queue). TP is "shard weights, replicate input", so
    # every rank MUST see the same batch on every step. Building an
    # independent producer per rank is the bug we're fixing. ----
    if args.use_dummy_data:
        # Dummy data has no IO; each rank builds its own random
        # DataLoader. The batches are random per rank, which is fine
        # for a smoke test of the sharded forward/backward path even
        # though it doesn't strictly satisfy the TP input-replication
        # contract. We deliberately don't route this through
        # _PrefetchBatcher because the DataLoader already yields
        # collated batches and re-collating would crash.
        dataloader = create_dummy_dataloader(
            args.batch_size, args.seq_len, config.vocab_size
        )
        prefetcher = None
    else:
        # Per-rank read path: every rank pulls from its own queue via
        # a tiny :class:`_QueueIterator`. The owner (rank 0)
        # additionally spawns a :class:`_PrefetchBatcher` that
        # broadcasts each batch into **every** rank's queue, so the
        # consumer code is uniform — there is no owner / non-owner
        # branching at the consumer side. For single-GPU runs the
        # caller passes ``shared_batch_queues=None`` and we use a
        # local in-process ``queue.Queue`` as the single rank's queue.
        if shared_batch_queues is None:
            my_queue = queue.Queue(maxsize=2)
            all_queues = [my_queue]
        else:
            my_queue = shared_batch_queues[rank]
            all_queues = list(shared_batch_queues)
        if rank == 0:
            # Owner: build the dataset and start the producer.
            # All the IO + tokenize cost is paid here.
            tokenizer = load_tokenizer(args.tokenizer_path)
            if args.stage == "sft":
                hf_name = args.sft_dataset_hf
                ms_name = args.sft_dataset_ms
                config_name = args.sft_config
                is_sft = True
            else:
                hf_name = args.pretrain_dataset_hf
                ms_name = args.pretrain_dataset_ms
                config_name = args.pretrain_config
                is_sft = False
            dataset = HFStreamingDataset(
                dataset_name=hf_name,
                ms_dataset_name=ms_name,
                use_modelscope=args.use_modelscope,
                tokenizer=tokenizer,
                split="train",
                max_seq_len=args.seq_len,
                is_sft=is_sft,
                config_name=config_name,
                text_field=args.text_field,
                shuffle=args.shuffle,
            )
            prefetcher = _PrefetchBatcher(
                dataset, batch_size=args.batch_size, queues=all_queues,
            )
        else:
            # Non-owner: no dataset, no tokenizer, no worker thread.
            # The owner's prefetch worker is already pushing into our
            # queue; we just need an iterator that drains it.
            prefetcher = None
        dataloader = _QueueIterator(my_queue)

    # ---- TP model ----
    from src.models.tp_model import TPHippoModel
    from src.models.tp_layers import init_tp
    # Reuse the backend chosen at init_process_group time so
    # :func:`init_tp` records the right transport in the global TP
    # state. The init_process_group call itself is a no-op inside
    # init_tp (we already initialized above), so this only seeds the
    # world-size / group globals.
    init_tp(world_size=len(gpus), devices=gpus, backend=backend)
    torch.manual_seed(args.seed)
    model = TPHippoModel(config, devices=gpus, dtype=torch.float16)
    # Broadcast replicated params from rank 0 (devices gpus[0]) to
    # all others, so the KDA / embed / norm are identical on every
    # device. (No-op when world_size == 1.)
    dist.barrier()
    model.sync_replicated_from(gpus[0])
    if rank == 0:
        trainable = sum(
            p.numel() for p in model.trainable_parameters(gpus[0])
        )
        logger.info(f"TP model built. Per-device trainable params: {trainable:,}")

    # ---- FP16 GradScaler ----
    # V100 has FP16 tensor cores; the model runs in FP16
    # (V100 doesn't support BF16 tensor cores). Loss scaling
    # would normally be required to prevent FP16 gradient
    # underflow on small updates, but PyTorch's
    # ``GradScaler.unscale_`` refuses to unscale FP16 grads in
    # this version (``allow_fp16=False`` is hard-coded), so we
    # disable the scaler and run plain FP16. The risk of
    # underflow is mitigated by the FP16-mantissa headroom on
    # the magnitude range of the gradients (typical for
    # transformer training at lr≈1e-3). If we later observe
    # underflow, we can swap in a manual scale/unscale path
    # or keep FP32 master weights.
    scaler = torch.amp.GradScaler("cuda", enabled=False)

    # ---- Per-device optimizers ----
    from src.training.param_offload import (
        build_param_groups, accumulate_grads_to_cpu, zero_cpu_grad_accum,
    )
    muon_opt, adamw_opt = build_param_groups(
        model, device=gpus[rank],
        lr_muon=args.muon_lr,
        lr_adamw=args.learning_rate,
        weight_decay=args.weight_decay,
        muon_momentum=args.muon_momentum,
    )
    n_muon = sum(s.param.numel() for s in muon_opt.state.values())
    n_adamw = sum(s.param.numel() for s in adamw_opt.state.values())
    # Optimizer state: Muon int8 momentum (1 byte/elem) + FP16
    # per-row scale; AdamW BF16 m + v (2 bytes/elem each). All on GPU.
    logger.info(
        f"Device {gpus[rank]}: Muon params={n_muon:,}"
        f" ({n_muon * 1 / 1024**3:.2f} GB int8 momentum + FP16 scale),"
        f" AdamW params={n_adamw:,}"
        f" ({n_adamw * 2 * 2 / 1024**3:.2f} GB BF16 m+v)."
    )

    # ---- Training loop ----
    global_step = 0
    accumulated_loss = 0.0
    microbatch_in_cycle = 0

    try:
        for epoch in range(args.epochs):
            if rank == 0:
                logger.info(f"Epoch {epoch + 1}/{args.epochs}")
            for batch_idx, batch in enumerate(dataloader):
                if global_step >= args.max_steps:
                    if rank == 0:
                        logger.info(
                            f"Reached max_steps={args.max_steps}, stopping"
                        )
                    break

                # Heartbeat for the user: first batch handed off
                # means streaming + tokenize + prefetch are alive
                # and we are about to start stepping.
                if batch_idx == 0 and rank == 0:
                    logger.info(
                        f"First batch ready: "
                        f"input_ids={tuple(batch['input_ids'].shape)}, "
                        f"labels={tuple(batch['labels'].shape)} - "
                        f"starting training"
                    )

                input_ids = batch["input_ids"].to(gpus[rank], non_blocking=True)
                labels = batch["labels"].to(gpus[rank], non_blocking=True)

                # autocast is a no-op for already-FP16 inputs but
                # is needed in case any module casts back to FP32
                # internally (e.g. layernorm in fla kernels) -- it
                # catches that and downcasts to FP16 on the way out.
                with torch.amp.autocast(
                    device_type="cuda", dtype=torch.float16,
                ):
                    outputs = model(input_ids, labels=labels)
                    loss = outputs["loss"]
                # Scaler is disabled (see construction above), so
                # ``scale`` is a no-op. Direct backward in FP16.
                mb_loss = loss.detach().float().item()
                # Per-microbatch loss logging: always log for the
                # first ``_DIAG_STEPS`` accum cycles, and on any
                # cycle where the microbatch loss is non-finite.
                # Helps localize when/where nan/inf first appears.
                if rank == 0 and (
                    global_step < _DIAG_STEPS or not math.isfinite(mb_loss)
                ):
                    logger.info(
                        f"  [diag] mb={batch_idx} loss={mb_loss:.6e}"
                    )
                (loss / args.gradient_accumulation_steps).backward()
                accumulated_loss += mb_loss

                # Stream grads to CPU and free GPU copies.
                accumulate_grads_to_cpu(
                    [muon_opt, adamw_opt], sync_device=gpus[rank],
                )
                del loss, outputs, input_ids, labels
                microbatch_in_cycle += 1

                if microbatch_in_cycle >= args.gradient_accumulation_steps:
                    # Inf/nan check on the accumulated grads (now on
                    # the GPU as FP32). If any param's grad is
                    # non-finite we skip the optimizer step so the
                    # state (BF16 m/v; int8 momentum) doesn't get
                    # poisoned.
                    found_inf = False
                    for opt in (muon_opt, adamw_opt):
                        for s in opt.state.values():
                            g = s.param.grad
                            if g is not None and not torch.isfinite(g).all():
                                found_inf = True
                                break
                        if found_inf:
                            break
                    if not found_inf:
                        # Compute the TP-reduced L2 grad norm on the
                        # accumulated CPU grads and clip in place if
                        # it exceeds ``args.max_grad_norm``. This
                        # guards against FP16 overflow (esp. in
                        # early steps with large lr) and against
                        # spikey KDA grads at block boundaries.
                        total_norm = _compute_and_clip_grad_norm(
                            [muon_opt, adamw_opt], args.max_grad_norm,
                        )
                        # DIAG: detailed per-step telemetry for the
                        # first few steps, so we can localize which
                        # optimizer (or which param group) is the
                        # source of NaN/Inf if it appears. The fields
                        # we capture are the ones most likely to
                        # blow up in FP16:
                        #   - max abs of accum (the clipped grad)
                        #   - max abs of state (m for AdamW, momentum for Muon)
                        #   - max abs of param
                        #   - per-optimizer non-finite counts after step
                        if global_step < _DIAG_STEPS and rank == 0:
                            import math as _math
                            def _amax(t):
                                if t is None:
                                    return 0.0
                                return t.detach().abs().max().item()
                            # Muon: acc = max abs of param.grad (FP32 on GPU);
                            #       mom = max abs of the dequantized momentum
                            #       (int8_q * int8_scale, max per row).
                            muon_acc = max(
                                (_amax(s.param.grad) for s in muon_opt.state.values()
                                 if s.param.grad is not None),
                                default=0.0,
                            )
                            muon_mom_vals = []
                            for s in muon_opt.state.values():
                                if s.int8_q is None or s.int8_scale is None:
                                    continue
                                # Max per-row dequantized abs = int8_max * scale
                                row_mom_max = (
                                    s.int8_q.view(s.shape[0], s.shape[1])
                                    .abs().amax(dim=1).float()
                                    * s.int8_scale.float()
                                )
                                muon_mom_vals.append(row_mom_max.max().item())
                            muon_mom = max(muon_mom_vals, default=0.0)
                            muon_pmax = max(
                                (_amax(s.param.data) for s in muon_opt.state.values()),
                                default=0.0,
                            )
                            # AdamW: acc = max abs of param.grad; v = max abs of v (BF16).
                            adamw_acc = max(
                                (_amax(s.param.grad) for s in adamw_opt.state.values()
                                 if s.param.grad is not None),
                                default=0.0,
                            )
                            adamw_v = max(
                                (_amax(s.exp_avg_sq) for s in adamw_opt.state.values()),
                                default=0.0,
                            )
                            adamw_m = max(
                                (_amax(s.m) for s in adamw_opt.state.values()),
                                default=0.0,
                            )
                            adamw_pmax = max(
                                (_amax(s.param.data) for s in adamw_opt.state.values()),
                                default=0.0,
                            )
                            logger.info(
                                f"  [diag-step {global_step} pre]"
                                f" total_norm={total_norm:.3e}"
                                f" muon: acc_max={muon_acc:.3e}"
                                f" mom_max={muon_mom:.3e}"
                                f" pmax={muon_pmax:.3e}"
                                f" | adamw: acc_max={adamw_acc:.3e}"
                                f" m_max={adamw_m:.3e}"
                                f" v_max={adamw_v:.3e}"
                                f" pmax={adamw_pmax:.3e}"
                            )
                        muon_opt.step()
                        if global_step < _DIAG_STEPS and rank == 0:
                            _nan_pmax = 0.0
                            _nan_first = None
                            for s in muon_opt.state.values():
                                v = s.param.data
                                if not torch.isfinite(v).all():
                                    am = v.detach().abs().max().item()
                                    if _math.isnan(am) or am > _nan_pmax:
                                        _nan_pmax = am
                                        if _nan_first is None:
                                            _nan_first = (s.param.shape, am)
                            if _nan_first is not None:
                                logger.info(
                                    f"  [diag-step {global_step} post-muon]"
                                    f" NON-FINITE in muon params:"
                                    f" first_shape={_nan_first[0]}"
                                    f" pmax={_nan_first[1]:.3e}"
                                )
                            else:
                                _pmax = max(
                                    (_amax(s.param.data) for s in muon_opt.state.values()),
                                    default=0.0,
                                )
                                logger.info(
                                    f"  [diag-step {global_step} post-muon]"
                                    f" all finite, pmax={_pmax:.3e}"
                                )
                        adamw_opt.step()
                        if global_step < _DIAG_STEPS and rank == 0:
                            _nan_pmax = 0.0
                            _nan_first = None
                            for s in adamw_opt.state.values():
                                v = s.param.data
                                if not torch.isfinite(v).all():
                                    am = v.detach().abs().max().item()
                                    if _math.isnan(am) or am > _nan_pmax:
                                        _nan_pmax = am
                                        if _nan_first is None:
                                            _nan_first = (s.param.shape, am)
                            if _nan_first is not None:
                                logger.info(
                                    f"  [diag-step {global_step} post-adamw]"
                                    f" NON-FINITE in adamw params:"
                                    f" first_shape={_nan_first[0]}"
                                    f" pmax={_nan_first[1]:.3e}"
                                )
                            else:
                                _pmax = max(
                                    (_amax(s.param.data) for s in adamw_opt.state.values()),
                                    default=0.0,
                                )
                                logger.info(
                                    f"  [diag-step {global_step} post-adamw]"
                                    f" all finite, pmax={_pmax:.3e}"
                                )
                        zero_cpu_grad_accum([muon_opt, adamw_opt])
                    else:
                        total_norm = float("nan")
                    scaler.update()
                    torch.cuda.synchronize(gpus[rank])

                    global_step += 1
                    avg_loss = accumulated_loss / args.gradient_accumulation_steps
                    accumulated_loss = 0.0
                    microbatch_in_cycle = 0

                    if rank == 0 and global_step % args.log_interval == 0:
                        grad_norm_str = (
                            f"{total_norm:.2f}" if total_norm == total_norm
                            else "nan"
                        )
                        logger.info(
                            f"Step {global_step}/{args.max_steps} | "
                            f"Loss: {avg_loss:.4f} | "
                            f"grad_norm: {grad_norm_str}"
                        )
                        # VRAM snapshot for OOM debugging.
                        for d in gpus:
                            free, total = torch.cuda.mem_get_info(d)
                            used = total - free
                            logger.info(
                                f"  device {d} VRAM: {used / 1024**3:.2f} /"
                                f" {total / 1024**3:.2f} GB"
                            )

            if global_step >= args.max_steps:
                break
    finally:
        if prefetcher is not None:
            prefetcher.close()
        dist.destroy_process_group()


def train(args):
    """Top-level training entry point.

    For TP world size > 1, spawns one process per GPU and runs
    :func:`_train_worker`. For world size == 1, runs the worker
    inline (no distributed init needed since all collectives are
    no-ops when world_size == 1).

    TP simulation (``--tp_sim``) short-circuits the auto-detect
    step: the original ``get_available_gpus`` scan is kept
    verbatim (it still returns the largest viable contiguous run
    on multi-GPU boxes) but on a single-GPU dev box it returns
    ``[]``, in which case we fall back to ``cuda:0`` and then
    expand the list to ``tp_size`` copies so mp.spawn launches
    N children all bound to the same physical device.
    """
    run_dir = create_output_dir(args.output_dir)

    gpus = get_available_gpus(min_memory_mb=args.min_gpu_memory_mb)
    if not gpus and args.tp_sim:
        # ``get_available_gpus`` looks for a contiguous run of
        # 2/4/8 GPUs with >= 10 GB free. On a single-GPU dev box
        # (one 5060 Ti 16G) that scan returns []; sim mode
        # explicitly opts into "I'll fake it with one card", so
        # fall back to cuda:0 here. Keeps the auto-detect
        # algorithm intact while still making the script usable
        # in the new dev environment.
        if torch.cuda.is_available():
            gpus = [0]
            print(
                f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [TP-SIM] "
                f"auto-detect found no 2/4/8-GPU run;"
                f" falling back to cuda:0 for sim"
            )
        else:
            print(
                f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] ERROR: "
                f"--tp_sim requires CUDA",
                file=sys.stderr,
            )
            return run_dir
    if not gpus and not args.tp_sim:
        # No contiguous 2/4/8-GPU run. Single-GPU is still a valid
        # run configuration — the TP code paths (ColumnParallelLinear,
        # AttnRes all-reduce, lm_head narrow view) all no-op when
        # world_size == 1, and the worker already has a fast path for
        # ``len(gpus) == 1 and not args.tp_sim`` that runs inline
        # without spawning children. Fall back to cuda:0 so single-GPU
        # boxes get a clean error path (OOM, model too large) instead
        # of a misleading "no TP group" failure.
        if torch.cuda.is_available() and torch.cuda.device_count() >= 1:
            gpus = [0]
            print(
                f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] WARNING: "
                f"no contiguous 2/4/8-GPU run is available; "
                f"falling back to single-GPU training on cuda:0 "
                f"(TP will no-op, world_size=1)."
            )
        else:
            print(
                f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] ERROR: "
                f"no CUDA device is available.",
                file=sys.stderr,
            )
            return run_dir

    # TP-simulation override: collapse the detected GPU list to a
    # single physical device replicated ``tp_size`` times. The TP
    # sharding code paths then see world_size == tp_size; gloo
    # handles the cross-process collectives via host staging.
    # This is the "simple splitting" path: no synthetic
    # framework, just N copies of the same rank.
    if args.tp_sim:
        if args.tp_size is None or args.tp_size < 2:
            print(
                f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] WARNING: "
                f"--tp_size={args.tp_size}; --tp_sim requires >= 2,"
                f" disabling sim",
                file=sys.stderr,
            )
            args.tp_sim = False
        else:
            if args.tp_size > 4:
                # Replicated params (KDA, embed, RMSNorm, AttnRes)
                # are duplicated in every child process. For the
                # default HippoConfig the per-process replicated
                # weight is ~900 MB in FP16; with optimizer state
                # and activations a 4-way sim is already tight on
                # 16 GB, and 8-way will OOM. Warn but don't block
                # so the user can still try with a shrunk config.
                print(
                    f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] WARNING: "
                    f"--tp_size={args.tp_size} > 4 may OOM on 16 GB"
                    f" (replicated weights duplicated per process)."
                    f" Try --use_dummy_data first.",
                    file=sys.stderr,
                )
            first_gpu = gpus[0]
            print(
                f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [TP-SIM] "
                f"simulating TP={args.tp_size} ranks on cuda:{first_gpu}"
                f" (gloo backend)"
            )
            gpus = [first_gpu] * args.tp_size

    print(
        f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Selected contiguous"
        f" TP group: {gpus} (world_size={len(gpus)}, sim={args.tp_sim})"
    )

    if len(gpus) == 1 and not args.tp_sim:
        # No TP plumbing needed; just run the worker inline. The
        # worker will allocate its own in-process ``queue.Queue``
        # for the dataloader (no cross-process sharing needed).
        # Skip this fast path in sim mode: even with one physical
        # GPU we still need N child processes to exercise the
        # sharded forward / all-reduce paths.
        _train_worker(
            rank=0, args=args, gpus=gpus, port=0,
            run_dir_path=str(run_dir), shared_batch_queues=None,
        )
    else:
        import torch.multiprocessing as mp
        port = 29500 + (os.getpid() % 1000)
        # TP shards weights but REPLICATES the input. Therefore all
        # ranks must consume the exact same batches on every step.
        # We centralise the dataset iterator on rank 0 (the owner)
        # and fan each batch out to every rank via per-rank queues
        # — one ``mp.Queue`` per rank, allocated in the parent and
        # passed to every child. The owner's prefetch worker
        # broadcasts each batch into every queue; each rank
        # consumes from ``shared_batch_queues[rank]`` via a
        # uniform :class:`_QueueIterator`. Building an independent
        # producer per rank is incorrect (different iterators,
        # different timings, no cross-rank synchronisation on the
        # input).
        ctx = mp.get_context("spawn")
        shared_batch_queues: list = [
            ctx.Queue(maxsize=2) for _ in range(len(gpus))
        ]
        mp.spawn(
            _train_worker,
            args=(args, gpus, port, str(run_dir), shared_batch_queues),
            nprocs=len(gpus),
            join=True,
        )

    return run_dir


def main():
    parser = argparse.ArgumentParser(description="Train HippoLM")

    # Config file
    parser.add_argument("--config", type=str, default="configs/base.yml")

    # Model args
    parser.add_argument("--vocab_size", type=int, default=248320)
    parser.add_argument("--hidden_size", type=int, default=1024)
    parser.add_argument("--tie_word_embeddings", type=bool, default=True)
    parser.add_argument("--use_bias", type=bool, default=False)
    parser.add_argument("--num_heads", type=int, default=16)
    parser.add_argument("--head_dim", type=int, default=64)
    parser.add_argument("--expand_v", type=float, default=1.0)
    parser.add_argument("--kda_mode", type=str, default="chunk",
                        choices=["chunk", "fused_recurrent"])
    parser.add_argument("--use_short_conv", type=bool, default=False)
    parser.add_argument("--allow_neg_eigval", type=bool, default=False)
    parser.add_argument("--safe_gate", type=bool, default=False)
    parser.add_argument("--lower_bound", type=float, default=None)
    parser.add_argument("--conv_size", type=int, default=4)
    parser.add_argument("--conv_bias", type=bool, default=False)
    parser.add_argument("--num_layers", type=int, default=32)
    parser.add_argument("--num_blocks", type=int, default=8)
    parser.add_argument("--intermediate_size", type=int, default=2736)
    parser.add_argument("--rms_norm_eps", type=float, default=1e-6)

    # Training args
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                        help="Max global (TP-reduced) L2 grad norm. "
                             "Set <= 0 to disable clipping. Default 1.0.")
    parser.add_argument("--fp16", action="store_true", default=True)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--checkpoint_interval", type=int, default=100)

    # Output
    parser.add_argument("--output_dir", type=str, default="output")

    # Data
    parser.add_argument("--stage", type=str, default="pretrain",
                        choices=["pretrain", "sft"],
                        help="Training stage: pretrain or sft")
    parser.add_argument("--use_modelscope", type=bool, default=True,
                        help="Prefer ModelScope (Aliyun CDN) over HF. Falls "
                             "back to HF only on network/connection errors.")
    parser.add_argument("--pretrain_dataset_hf", type=str,
                        default="openbmb/Ultra-FineWeb-L3",
                        help="HF dataset id for pretraining (fallback source).")
    parser.add_argument("--pretrain_dataset_ms", type=str,
                        default="OpenBMB/Ultra-FineWeb-L3",
                        help="ModelScope dataset id for pretraining (primary).")
    parser.add_argument("--pretrain_config", type=str,
                        default="Ultra-FineWeb-L3-en-QA-Synthetic")
    parser.add_argument("--sft_dataset_hf", type=str,
                        default="openbmb/UltraData-SFT-2605",
                        help="HF dataset id for SFT (fallback source).")
    parser.add_argument("--sft_dataset_ms", type=str,
                        default="OpenBMB/UltraData-SFT-2605",
                        help="ModelScope dataset id for SFT (primary).")
    parser.add_argument("--sft_config", type=str, default=None)
    parser.add_argument("--text_field", type=str, default="content")
    parser.add_argument("--tokenizer_path", type=str,
                        default="src/tokenizer")
    parser.add_argument("--use_dummy_data", action="store_true", default=False)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument(
        "--shuffle", type=bool, default=True,
        help="Shuffle the streaming dataset. Disable (--shuffle false) for "
             "faster first-batch on slow mirrors — sequential order is fine "
             "for short validation runs.",
    )

    # GPU
    parser.add_argument("--min_gpu_memory_mb", type=int, default=10240)
    parser.add_argument("--muon_lr", type=float, default=0.02,
                        help="Learning rate for Muon (2D weight matrices).")
    parser.add_argument("--muon_momentum", type=float, default=0.95,
                        help="SGD momentum for the Muon path.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Init seed for replicated params on each device.")

    # TP simulation (development on a single-GPU box, e.g. one
    # 5060 Ti 16G). When --tp_sim is set we run N child processes
    # all bound to the same physical GPU and use the gloo backend
    # to transport CUDA tensors (host-staged; much slower than
    # NVLink NCCL but the sharding math is byte-for-byte the
    # same). Use this to verify the TP code paths without owning
    # N physical GPUs. --tp_size is only consulted when --tp_sim
    # is set.
    parser.add_argument("--tp_sim", action="store_true", default=False,
                        help="Simulate TP on a single GPU with gloo.")
    parser.add_argument("--tp_size", type=int, default=2,
                        help="Simulated TP world size when --tp_sim is set."
                             " Ignored otherwise. Default 2. Practical max"
                             " is ~2-4 on a 16 GB GPU because replicated"
                             " weights are duplicated per process.")

    args = parser.parse_args()

    if Path(args.config).exists():
        config_dict = load_config(args.config)
        for key, value in config_dict.items():
            if hasattr(args, key):
                setattr(args, key, value)

    train(args)


if __name__ == "__main__":
    main()
