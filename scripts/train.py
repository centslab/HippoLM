"""Training script for HippoLM v0.0.0.

Features:
- Auto GPU selection (VRAM >= 10GB)
- AMP FP16 training
- Gradient accumulation
- Gradient checkpointing every block_size layers
- max_step training limit
- Checkpoint saving to timestamped output directory
- Qwen3.5 tokenizer
- SlimPajama dataset support

CPU offload (DeepSpeed-style) will be added in v0.0.1.
"""
import os
import sys
import argparse
import logging
from pathlib import Path

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


class SlimPajamaDataset(IterableDataset):
    """Iterable dataset for SlimPajama JSONL files."""

    def __init__(self, data_dir: str, tokenizer, max_seq_len: int = 512):
        self.data_dir = Path(data_dir)
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.files = sorted(self.data_dir.glob("**/*.jsonl"))

    def __iter__(self):
        import json

        for file_path in self.files:
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        text = data.get("text", "")
                        if not text:
                            continue

                        tokens = self.tokenizer(
                            text,
                            max_length=self.max_seq_len,
                            truncation=True,
                            return_tensors="pt",
                        )
                        input_ids = tokens["input_ids"].squeeze(0)

                        if len(input_ids) < 2:
                            continue

                        yield {"input_ids": input_ids, "labels": input_ids.clone()}
                    except (json.JSONDecodeError, KeyError):
                        continue

    def __len__(self):
        return 1000000


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
            dataset = SlimPajamaDataset(args.data_dir, tokenizer, args.seq_len)
            dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers)
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