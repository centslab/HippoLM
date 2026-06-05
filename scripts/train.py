"""Training script for HippoLM v0.0.0.

Features:
- Auto GPU selection (VRAM >= 10GB)
- AMP FP16 training
- Gradient accumulation
- Gradient checkpointing every block_size layers
- max_step training limit
- Checkpoint saving to timestamped output directory
- Qwen3.5 tokenizer (local)
- Streaming datasets with ModelScope preferred (Aliyun CDN, fast)
  and HuggingFace (hf-mirror.com) as fallback on network errors only:
  - Pretraining: OpenBMB/Ultra-FineWeb-L3 (MS) / openbmb/Ultra-FineWeb-L3 (HF)
  - SFT: OpenBMB/UltraData-SFT-2605 (MS) / openbmb/UltraData-SFT-2605 (HF)
- Single-threaded background-thread prefetch with persistent ~/.cache

CPU offload (DeepSpeed-style) will be added in v0.0.1.
"""

# CRITICAL: HF_ENDPOINT and HF_TOKEN must be set BEFORE huggingface_hub is
# imported, because HfApi reads the env vars at construction time. Set them
# at the very top, before any other imports, so that any chain import of
# huggingface_hub (e.g. via `datasets` or `transformers`) picks them up.
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

import os
import sys
import argparse
import logging
import time
import random
import queue
import threading
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, IterableDataset

sys.path.insert(0, "/home/wlx/HippoLM")

from configs.base_config import HippoConfig
from src.models.model import HippoModel


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
    """Select GPUs with at least min_memory_mb free VRAM."""
    if not torch.cuda.is_available():
        return []

    try:
        import pynvml

        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()

        available = []
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            free_mb = info.free / (1024 ** 2)
            if free_mb >= min_memory_mb:
                available.append(i)

        pynvml.nvmlShutdown()
        return available

    except ImportError:
        import subprocess

        available = []
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
                        available.append(idx)
        except FileNotFoundError:
            for i in range(torch.cuda.device_count()):
                total = torch.cuda.get_device_properties(i).total_memory / (1024 ** 2)
                if total >= min_memory_mb:
                    available.append(i)

        return available


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
    """Try ModelScope first, fall back to HF on network/connection errors only.

    Behavior matrix:
      - ``use_ms=False`` or ``ms_name is None``  -> HF directly
      - MS load succeeds                          -> return MS iterator
      - MS raises ``_network_exceptions``         -> warn, fall back to HF
      - MS raises ``ImportError`` (not installed) -> warn, fall back to HF
      - MS raises anything else (HTTPError 4xx,
        ValueError on schema, KeyError on field)  -> re-raise unchanged
    """
    if use_ms and ms_name:
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


class _PrefetchBatcher:
    """Run a streaming dataset in a background thread and emit collated batches.

    Why: with ``num_workers=0`` PyTorch DataLoader does no prefetching, so the
    GPU stalls waiting on the next HTTP fetch + tokenize. Pushing the
    iteration / tokenize / collate into a daemon thread and pulling finished
    batches from a bounded queue overlaps IO with compute.

    The producer thread iterates the dataset, tokenizes, accumulates
    ``batch_size`` samples, collates, and ``put``s the batch. The main thread
    pulls batches via ``__iter__`` and processes them. When the dataset
    iterator is exhausted a ``None`` sentinel ends the stream; any producer
    exception is re-raised on the main thread.
    """

    _SENTINEL = None

    def __init__(self, dataset, batch_size: int, queue_size: int = 2):
        self._dataset = dataset
        self._batch_size = batch_size
        self._queue: queue.Queue = queue.Queue(maxsize=max(queue_size, 1))
        self._error: Optional[BaseException] = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()

    def _producer(self) -> None:
        try:
            batch: list = []
            for sample in self._dataset:
                if self._stop.is_set():
                    return
                batch.append(sample)
                if len(batch) >= self._batch_size:
                    self._put_with_stop(_collate_batch(batch))
                    if self._stop.is_set():
                        return
                    batch = []
            if batch and not self._stop.is_set():
                self._put_with_stop(_collate_batch(batch))
        except BaseException as e:  # noqa: BLE001 - surface to consumer
            self._error = e
        finally:
            # Put the sentinel in a non-blocking way: if the main thread has
            # already abandoned the iterator the queue may be full of items
            # nobody will consume, and blocking here would deadlock.
            try:
                self._queue.put_nowait(self._SENTINEL)
            except queue.Full:
                pass

    def _put_with_stop(self, item) -> None:
        """``queue.put`` with a short timeout so ``stop()`` is responsive
        even when the consumer stops reading and the queue is full."""
        while not self._stop.is_set():
            try:
                self._queue.put(item, timeout=0.5)
                return
            except queue.Full:
                continue

    def start(self) -> None:
        """Start the producer thread. Idempotent; safe to call multiple times."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._thread = threading.Thread(
            target=self._producer, name="hf-prefetch", daemon=True
        )
        self._thread.start()

    def __iter__(self):
        self.start()
        while True:
            item = self._queue.get()
            if item is self._SENTINEL:
                if self._error is not None:
                    raise self._error
                return
            yield item

    def stop(self) -> None:
        """Signal the producer to exit at the next sample boundary."""
        self._stop.set()

    def close(self, timeout: float = 2.0) -> None:
        """Stop the producer and wait briefly for the thread to join.
        Avoids a noisy GIL-state warning at interpreter finalization when
        the daemon producer is still mid-fetch when the process exits."""
        self.stop()
        if self._thread is not None:
            self._thread.join(timeout=timeout)


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


def train(args):
    """Main training loop."""
    run_dir = create_output_dir(args.output_dir)
    log_file = run_dir / "logs" / "train.log"

    global logger
    logger = setup_logging(log_file)
    logger.info(f"HippoLM Training - Output: {run_dir}")

    # GPU selection
    gpus = get_available_gpus(min_memory_mb=args.min_gpu_memory_mb)
    use_dp = False
    if not gpus:
        logger.warning("No GPUs with enough memory, using CPU")
        device = torch.device("cpu")
    else:
        # Clamp to visible devices to avoid device ordinal errors when CUDA_VISIBLE_DEVICES is set
        visible = torch.cuda.device_count()
        gpus = [g for g in gpus if g < visible]
        if not gpus:
            gpus = list(range(visible))
        device = torch.device(f"cuda:{gpus[0]}")
        if len(gpus) > 1:
            logger.info(f"Using {len(gpus)} GPUs: {gpus} (DataParallel)")
            use_dp = args.batch_size >= len(gpus)

    # Model config
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
    logger.info(f"Model config: {config}")

    # Data — build the streaming dataset and start the prefetch producer
    # BEFORE the model is constructed. The first-batch cold start
    # (HF metadata cascade + 3-hop redirect parquet fetch) is IO-bound and
    # can fully overlap with the ~12s the model takes to materialize on GPU.
    if args.use_dummy_data:
        logger.info("Using dummy data")
        dataloader = create_dummy_dataloader(args.batch_size, args.seq_len, config.vocab_size)
        dataloader_is_prefetched = False
    else:
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
        logger.info(
            f"Loading dataset: stage={args.stage}, hf={hf_name}, ms={ms_name},"
            f" use_modelscope={args.use_modelscope}, config={config_name}"
        )
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
        # Single-threaded streaming with background-thread prefetch. Bounded
        # queue keeps a few batches of tokenized data ready so the GPU does
        # not stall waiting on HTTP. The training loop breaks on max_steps
        # and signals the producer to drain & exit.
        dataloader = _PrefetchBatcher(
            dataset,
            batch_size=args.batch_size,
            queue_size=2,
        )
        dataloader_is_prefetched = True
        # Kick off the producer now so the IO overlaps with the model build
        # and optimizer/scaler setup below.
        dataloader.start()
        logger.info("Streaming prefetch producer started; building model in parallel.")

    # Model
    model = HippoModel(config).to(device)
    if use_dp:
        model = nn.DataParallel(model, device_ids=gpus)

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Total params: {total_params:,}")

    # Optimizer
    if args.offload_optimizer:
        logger.info("Using CPU AdamW optimizer (optimizer state offloaded to CPU)")
        from src.training.cpu_adamw import create_cpu_adamw_optimizer
        optimizer = create_cpu_adamw_optimizer(
            model,
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
            logger=logger.info,
        )
    else:
        logger.info("Using standard GPU AdamW optimizer")
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            weight_decay=args.weight_decay,
        )
    scaler = torch.amp.GradScaler("cuda", enabled=args.fp16)

    # Training loop
    model.train()
    global_step = 0
    accumulated_loss = 0.0
    optimizer.zero_grad()

    for epoch in range(args.epochs):
        logger.info(f"Epoch {epoch + 1}/{args.epochs}")

        for batch_idx, batch in enumerate(dataloader):
            if global_step >= args.max_steps:
                logger.info(f"Reached max_steps={args.max_steps}, stopping")
                if dataloader_is_prefetched:
                    dataloader.stop()
                break

            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            with torch.amp.autocast("cuda", enabled=args.fp16):
                outputs = model(input_ids, labels=labels)
                loss = outputs["loss"]
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

            scaler.scale(loss).backward()
            accumulated_loss += loss.item()

            if (batch_idx + 1) % args.gradient_accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

                global_step += 1
                avg_loss = accumulated_loss / args.gradient_accumulation_steps

                if global_step % args.log_interval == 0:
                    lr = optimizer.param_groups[0]['lr']
                    logger.info(
                        f"Step {global_step}/{args.max_steps} | "
                        f"Loss: {avg_loss:.4f} | "
                        f"LR: {lr:.2e}"
                    )

                if global_step % args.checkpoint_interval == 0:
                    path = save_checkpoint(
                        model.module if hasattr(model, "module") else model,
                        optimizer, scaler, global_step, avg_loss,
                        run_dir / "checkpoints",
                    )
                    logger.info(f"Checkpoint saved: {path}")

                accumulated_loss = 0.0

        if global_step >= args.max_steps:
            break

    # Final checkpoint
    if global_step > 0:
        path = save_checkpoint(
            model.module if hasattr(model, "module") else model,
            optimizer, scaler, global_step, accumulated_loss,
            run_dir / "checkpoints",
        )
        logger.info(f"Final checkpoint saved: {path}")

    if dataloader_is_prefetched:
        dataloader.close()

    logger.info(f"Training complete! Output: {run_dir}")
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
    parser.add_argument("--offload_optimizer", action="store_true", default=False)

    args = parser.parse_args()

    if Path(args.config).exists() and not args.use_dummy_data:
        config_dict = load_config(args.config)
        for key, value in config_dict.items():
            if hasattr(args, key):
                setattr(args, key, value)

    train(args)


if __name__ == "__main__":
    main()
