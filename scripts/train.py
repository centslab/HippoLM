"""Training script for HippoLM v0.0.0.

Features:
- Auto GPU selection (VRAM >= 10GB)
- AMP FP16 training
- Gradient accumulation
- Gradient checkpointing every block_size layers
- max_step training limit
- Checkpoint saving to timestamped output directory
- Qwen3.5 tokenizer
- SlimPajama dataset support (streaming, zstd-compressed)

CPU offload (DeepSpeed-style) will be added in v0.0.1.
"""
import os
import sys
import argparse
import logging
import io
import json
import math
import random
import zstandard as zstd
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, IterableDataset
from torch.amp import GradScaler

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
    """Select GPUs with at least min_memory_mb free VRAM.

    Returns:
        List of GPU device indices
    """
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

            props = torch.cuda.get_device_properties(i)
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
                props = torch.cuda.get_device_properties(i)
                total = props.total_memory / (1024 ** 2)
                if total >= min_memory_mb:
                    available.append(i)

        return available


def load_tokenizer(tokenizer_path: str):
    """Load Qwen3.5 tokenizer."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_config(config_path: str) -> dict:
    """Load training config from YAML file."""
    import yaml

    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def create_output_dir(output_dir: str) -> Path:
    """Create timestamped output directory."""
    import time

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = Path(output_dir) / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / "checkpoints").mkdir(exist_ok=True)
    (run_dir / "logs").mkdir(exist_ok=True)
    (run_dir / "eval").mkdir(exist_ok=True)

    return run_dir


import io
import json
import math
import os
import random
import zstandard as zstd
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import IterableDataset


def _resolve_split_path(data_path: str, split: str) -> str:
    """Resolve path for a dataset split."""
    split_path = os.path.join(data_path, split)
    if os.path.exists(split_path):
        return split_path
    if os.path.exists(data_path) and os.path.basename(os.path.normpath(data_path)) == split:
        return data_path
    raise ValueError(f"Path does not exist: {split_path}")


def _find_files(root_path: str, primary_pattern: str, fallback_pattern: str) -> list[str]:
    """Find files matching pattern, with fallback."""
    import glob
    files = sorted(glob.glob(os.path.join(root_path, primary_pattern)))
    if not files:
        files = sorted(glob.glob(os.path.join(root_path, fallback_pattern), recursive=True))
    if not files:
        raise ValueError(f"No files found under {root_path} matching {primary_pattern!r} or {fallback_pattern!r}")
    return files


def _shard_range(total_items: int, shard_id: int, num_shards: int) -> tuple[int, int]:
    """Calculate range for a specific shard."""
    total_items = max(int(total_items), 0)
    num_shards = max(int(num_shards), 1)
    shard_id = min(max(int(shard_id), 0), num_shards - 1)

    base = total_items // num_shards
    remainder = total_items % num_shards
    start = shard_id * base + min(shard_id, remainder)
    end = start + base + (1 if shard_id < remainder else 0)
    return start, end


class ZstdFileReader:
    """Helper class to stream zstd-compressed jsonl files."""

    @staticmethod
    def iter_zst_lines(file_path: str):
        with open(file_path, 'rb') as f:
            dctx = zstd.ZstdDecompressor()
            with dctx.stream_reader(f) as reader:
                with io.TextIOWrapper(reader, encoding='utf-8') as text_reader:
                    for line in text_reader:
                        yield line.rstrip('\r\n')

    @staticmethod
    def read_zst_file(file_path: str) -> list[str]:
        return list(ZstdFileReader.iter_zst_lines(file_path))


class SlimPajamaDataset(IterableDataset):
    """Stream text samples from SlimPajama-style jsonl.zst shards."""

    def __init__(
        self,
        data_path: str,
        tokenizer,
        split: str = "train",
        max_seq_len: int = 2048,
        num_samples: Optional[int] = None,
        chunk_pattern: str = "chunk1/*.jsonl.zst",
        seed: int = 42,
        shuffle: bool = True,
        rank: int = 0,
        world_size: int = 1,
    ):
        super().__init__()
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.split = split
        self.max_seq_len = max_seq_len
        self.num_samples = num_samples
        self.chunk_pattern = chunk_pattern
        self.seed = seed
        self.shuffle = shuffle
        self.rank = rank
        self.world_size = max(int(world_size), 1)

        split_path = _resolve_split_path(data_path, split)
        self.files = _find_files(split_path, chunk_pattern, "**/*.jsonl.zst")

        if self.shuffle:
            shuffled = list(self.files)
            random.Random(seed).shuffle(shuffled)
            self.files = shuffled

        # Try to load metadata first to avoid counting all files
        self.file_sample_counts: Optional[list[int]] = None
        metadata_path = Path(self.data_path) / 'metadata.json'
        if metadata_path.exists():
            try:
                meta = json.loads(metadata_path.read_text(encoding='utf-8'))
                split_meta = next((s for s in meta.get('splits', []) if s['split'] == self.split), None)
                if split_meta and 'shard_rows' in split_meta:
                    self.file_sample_counts = [int(x) for x in split_meta['shard_rows']]
                    print(f"[SlimPajamaDataset] Loaded {len(self.file_sample_counts)} sample counts from metadata.json")
            except Exception:
                pass

        # Use round-robin sharding for training - fast startup, no counting needed
        self._needs_sample_count = False

    def _iter_file_texts(self, file_path: str):
        for line in ZstdFileReader.iter_zst_lines(file_path):
            if not line.strip():
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = data.get('text')
            if text is not None:
                yield text

    def _count_file_samples(self) -> list[int]:
        """Count samples in each file. Stop early once we have enough samples."""
        total_files = len(self.files)
        # For training, we only need enough samples for num_samples * 1.5 (with margin)
        target_samples = self.num_samples * 3 // 2 if self.num_samples else None

        counts = []

        for i, file_path in enumerate(self.files):
            count = 0
            try:
                for _ in self._iter_file_texts(file_path):
                    count += 1
            except Exception:
                pass

            counts.append(count)

            # After counting at least 1 file, we can estimate
            if len(counts) >= 1 and target_samples:
                total_counted = sum(counts)
                avg = total_counted / len(counts)
                estimated_total = int(avg * total_files)

                # If our estimate exceeds target by a lot, we can stop
                # But only if we've counted enough to have confidence (at least 5 files or 50% of target)
                if estimated_total >= target_samples * 10 and total_counted >= target_samples:
                    counts.extend([int(avg)] * (total_files - len(counts)))
                    print(f"[SlimPajamaDataset] Counted {len(counts)}/{total_files} files, estimated total: {estimated_total:,}")
                    return counts

        # If we didn't stop early, return all counts
        print(f"[SlimPajamaDataset] Counted all {len(counts)} files, total samples: {sum(counts):,}")
        return counts

    def _iter_streaming_partition(self, global_shard_id: int, global_num_shards: int):
        total_samples = sum(self.file_sample_counts or [])
        if self.num_samples is not None:
            total_samples = min(total_samples, int(self.num_samples))

        sample_start, sample_end = _shard_range(total_samples, global_shard_id, global_num_shards)
        if sample_start >= sample_end:
            return

        file_global_start = 0
        for file_path, file_sample_count in zip(self.files, self.file_sample_counts or []):
            file_global_end = file_global_start + int(file_sample_count)
            local_start = max(sample_start, file_global_start) - file_global_start
            local_end = min(sample_end, file_global_end) - file_global_start
            if local_start < local_end:
                local_idx = 0
                try:
                    for text in self._iter_file_texts(file_path):
                        if local_idx < local_start:
                            local_idx += 1
                            continue
                        if local_idx >= local_end:
                            break

                        tokens = self.tokenizer(
                            text,
                            max_length=self.max_seq_len,
                            truncation=True,
                            return_tensors="pt",
                        )
                        input_ids = tokens["input_ids"].squeeze(0)

                        if len(input_ids) >= 2:
                            yield {"input_ids": input_ids, "labels": input_ids.clone()}

                        local_idx += 1
                except Exception:
                    pass

            file_global_start = file_global_end
            if file_global_start >= sample_end:
                break

    def _iter_streaming_roundrobin(self, global_shard_id: int, global_num_shards: int):
        """Fallback streaming using simple round-robin sharding."""
        sample_idx = 0
        for file_path in self.files:
            try:
                for text in self._iter_file_texts(file_path):
                    if sample_idx % global_num_shards == global_shard_id:
                        tokens = self.tokenizer(
                            text,
                            max_length=self.max_seq_len,
                            truncation=True,
                            return_tensors="pt",
                        )
                        input_ids = tokens["input_ids"].squeeze(0)
                        if len(input_ids) >= 2:
                            yield {"input_ids": input_ids, "labels": input_ids.clone()}
                    sample_idx += 1
                    if self.num_samples is not None and sample_idx >= self.num_samples:
                        return
            except Exception:
                pass

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            local_worker_id = 0
            local_num_workers = 1
        else:
            local_worker_id = worker_info.id
            local_num_workers = worker_info.num_workers

        global_num_shards = self.world_size * local_num_workers
        global_shard_id = self.rank * local_num_workers + local_worker_id

        # Count samples on first iteration for range-based sharding
        if self._needs_sample_count:
            self.file_sample_counts = self._count_file_samples()
            self._needs_sample_count = False

        if self.file_sample_counts is not None:
            yield from self._iter_streaming_partition(global_shard_id, global_num_shards)
        else:
            yield from self._iter_streaming_roundrobin(global_shard_id, global_num_shards)

    def __len__(self):
        if self.num_samples is not None:
            total_samples = int(self.num_samples)
        elif self.file_sample_counts is not None:
            total_samples = sum(self.file_sample_counts)
        else:
            total_samples = len(self.files) * 9980  # SlimPajama average
        return max(math.ceil(total_samples / self.world_size), 1)


class PretokenizedDataset(IterableDataset):
    """Read pretokenized fixed-length `.npy` shards with mmap for memory efficiency."""

    def __init__(
        self,
        data_path: str,
        split: str = "train",
        max_seq_len: int = 2048,
        num_samples: Optional[int] = None,
        shard_pattern: str = "*.npy",
        seed: int = 42,
        shuffle: bool = False,
        rank: int = 0,
        world_size: int = 1,
    ):
        super().__init__()
        self.data_path = data_path
        self.split = split
        self.max_seq_len = int(max_seq_len)
        self.num_samples = num_samples
        self.shard_pattern = shard_pattern
        self.seed = seed
        self.shuffle = shuffle
        self.rank = rank
        self.world_size = max(int(world_size), 1)

        split_path = _resolve_split_path(data_path, split)
        self.files = _find_files(split_path, shard_pattern, f"**/{shard_pattern}")
        if self.shuffle:
            shuffled = list(self.files)
            random.Random(seed).shuffle(shuffled)
            self.files = shuffled

        # Load shard sizes from metadata.json if available
        metadata_path = Path(data_path) / 'metadata.json'
        self.shard_sizes = []
        self.total_samples = 0

        if metadata_path.exists():
            try:
                meta = json.loads(metadata_path.read_text(encoding='utf-8'))
                split_meta = next((s for s in meta.get('splits', []) if s['split'] == split), None)
                if split_meta and int(meta.get('max_seq_len', 0)) == self.max_seq_len:
                    shard_rows = split_meta.get('shard_rows')
                    if isinstance(shard_rows, list) and len(shard_rows) == len(self.files):
                        self.shard_sizes = [int(x) for x in shard_rows]
                        self.total_samples = sum(self.shard_sizes)
                    elif 'num_samples' in split_meta:
                        self.total_samples = int(split_meta['num_samples'])
            except (json.JSONDecodeError, KeyError, StopIteration):
                pass

        if not self.shard_sizes:
            for file_path in self.files:
                shard = np.load(file_path, mmap_mode='r')
                if shard.ndim != 2:
                    raise ValueError(f"Pretokenized shard must be 2D: {file_path}")
                if shard.shape[1] != self.max_seq_len:
                    raise ValueError(f"Pretokenized shard shape mismatch for {file_path}")
                self.shard_sizes.append(int(shard.shape[0]))
            self.total_samples = sum(self.shard_sizes)

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            local_worker_id = 0
            local_num_workers = 1
        else:
            local_worker_id = worker_info.id
            local_num_workers = worker_info.num_workers

        global_num_shards = self.world_size * local_num_workers
        global_shard_id = self.rank * local_num_workers + local_worker_id
        total_samples = self.total_samples
        if self.num_samples is not None:
            total_samples = min(total_samples, int(self.num_samples))

        sample_start, sample_end = _shard_range(total_samples, global_shard_id, global_num_shards)
        if sample_start >= sample_end:
            return

        _CHUNK = 256
        file_global_start = 0
        for file_idx, file_path in enumerate(self.files):
            n_rows = int(self.shard_sizes[file_idx])
            file_global_end = file_global_start + n_rows

            row_start = max(sample_start, file_global_start) - file_global_start
            row_end = min(sample_end, file_global_end) - file_global_start
            if row_start < row_end:
                shard = np.load(file_path, mmap_mode='r')
                for chunk_start in range(row_start, row_end, _CHUNK):
                    chunk_end = min(chunk_start + _CHUNK, row_end)
                    chunk = np.array(shard[chunk_start:chunk_end])
                    base_idx = file_global_start + chunk_start
                    for local_idx in range(chunk.shape[0]):
                        yield {
                            'input_ids': chunk[local_idx],
                            'labels': chunk[local_idx].copy(),
                            'idx': base_idx + local_idx
                        }

            file_global_start = file_global_end
            if file_global_start >= sample_end:
                break

    def __len__(self):
        total_samples = self.total_samples
        if self.num_samples is not None:
            total_samples = min(total_samples, int(self.num_samples))
        return max(math.ceil(total_samples / self.world_size), 1)


def build_dataset(data_path: str, tokenizer, split: str = "train", max_seq_len: int = 512,
                  num_samples: Optional[int] = None, use_pretokenized: bool = False,
                  rank: int = 0, world_size: int = 1):
    """Build appropriate dataset based on data format."""
    pretokenized_path = Path(data_path).parent / f"{Path(data_path).name}_pretokenized"

    if use_pretokenized and pretokenized_path.exists():
        return PretokenizedDataset(
            data_path=str(pretokenized_path),
            split=split,
            max_seq_len=max_seq_len,
            num_samples=num_samples,
            rank=rank,
            world_size=world_size,
        )

    return SlimPajamaDataset(
        data_path=data_path,
        tokenizer=tokenizer,
        split=split,
        max_seq_len=max_seq_len,
        num_samples=num_samples,
        chunk_pattern="chunk1/*.jsonl.zst",
        seed=42,
        shuffle=(split == "train"),
        rank=rank,
        world_size=world_size,
    )


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
    # Create output directory
    run_dir = create_output_dir(args.output_dir)
    log_file = run_dir / "logs" / "train.log"

    global logger
    logger = setup_logging(log_file)

    logger.info(f"HippoLM Training - Output: {run_dir}")

    # GPU selection
    gpus = get_available_gpus(min_memory_mb=args.min_gpu_memory_mb)
    if not gpus:
        logger.warning("No GPUs with enough memory, using CPU")
        device = torch.device("cpu")
    else:
        device = torch.device(f"cuda:{gpus[0]}")
        if len(gpus) > 1:
            logger.info(f"Using DataParallel on GPUs: {gpus}")

    # Model config
    config = HippoConfig(
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        num_heads=args.num_heads,
        head_dim=args.head_dim,
        num_kv=args.num_kv,
        num_layers=args.num_layers,
        num_blocks=args.num_blocks,
        intermediate_size=args.intermediate_size,
        max_seq_len=args.max_seq_len,
    )
    logger.info(f"Model config: {config}")

    # Model
    model = HippoModel(config).to(device)
    if len(gpus) > 1:
        model = nn.DataParallel(model, device_ids=gpus)

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Total params: {total_params:,}")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=args.weight_decay,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=args.fp16)

    # Data
    if args.use_dummy_data:
        logger.info("Using dummy data")
        dataloader = create_dummy_dataloader(args.batch_size, args.seq_len, config.vocab_size)
    else:
        tokenizer = load_tokenizer(args.tokenizer_path)
        if Path(args.data_dir).exists():
            dataset = SlimPajamaDataset(
                data_path=args.data_dir,
                tokenizer=tokenizer,
                split="train",
                max_seq_len=args.seq_len,
                num_samples=args.max_steps * args.batch_size * args.gradient_accumulation_steps,
                rank=0,
                world_size=1,
            )
            dataloader = DataLoader(
                dataset,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                pin_memory=True,
            )
        else:
            logger.warning(f"Dataset not found at {args.data_dir}, using dummy data")
            dataloader = create_dummy_dataloader(args.batch_size, args.seq_len, config.vocab_size)

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
                    logger.info(
                        f"Step {global_step}/{args.max_steps} | "
                        f"Loss: {avg_loss:.4f} | "
                        f"LR: {optimizer.param_groups[0]['lr']:.2e}"
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
            optimizer, scaler, global_step, avg_loss,
            run_dir / "checkpoints",
        )
        logger.info(f"Final checkpoint saved: {path}")

    logger.info(f"Training complete! Output: {run_dir}")
    return run_dir


def main():
    parser = argparse.ArgumentParser(description="Train HippoLM")

    # Config file
    parser.add_argument("--config", type=str, default="configs/base.yml")

    # Model args
    parser.add_argument("--vocab_size", type=int, default=248320)
    parser.add_argument("--hidden_size", type=int, default=1024)
    parser.add_argument("--num_heads", type=int, default=16)
    parser.add_argument("--head_dim", type=int, default=64)
    parser.add_argument("--num_kv", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=32)
    parser.add_argument("--num_blocks", type=int, default=8)
    parser.add_argument("--intermediate_size", type=int, default=2736)
    parser.add_argument("--max_seq_len", type=int, default=-1)

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
    parser.add_argument("--data_dir", type=str, default="/mnt/6138/datasets/xpengx/SlimPajama-627B")
    parser.add_argument("--tokenizer_path", type=str, default="/home/wlx/Qwen3.5-9B")
    parser.add_argument("--use_dummy_data", action="store_true", default=False)
    parser.add_argument("--num_workers", type=int, default=2)

    # GPU
    parser.add_argument("--min_gpu_memory_mb", type=int, default=10240)

    args = parser.parse_args()

    # Load config from YAML if exists
    if Path(args.config).exists() and not args.use_dummy_data:
        config_dict = load_config(args.config)
        # Override defaults with config file values
        for key, value in config_dict.items():
            if hasattr(args, key):
                setattr(args, key, value)

    train(args)


if __name__ == "__main__":
    main()