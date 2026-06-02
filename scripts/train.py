"""Training script for HippoLM v0.0.0.

Features:
- Auto GPU selection (VRAM >= 10GB)
- AMP FP16 training
- Gradient accumulation
- Gradient checkpointing every block_size layers
- max_step training limit
- Checkpoint saving to timestamped output directory
- Qwen3.5 tokenizer (local)
- Streaming HF datasets from hf-mirror.com:
  - Pretraining: openbmb/Ultra-FineWeb-L3
  - SFT: openbmb/UltraData-SFT-2605

CPU offload (DeepSpeed-style) will be added in v0.0.1.
"""

# CRITICAL: HF_ENDPOINT must be set BEFORE huggingface_hub is imported,
# because HfApi reads the env var at construction time. Set it at the
# very top, before any other imports, so that any chain import of
# huggingface_hub (e.g. via `datasets` or `transformers`) picks it up.
import os as _os
if "HF_ENDPOINT" not in _os.environ:
    _os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# All HF cache locations go to a fresh tempdir so nothing persists locally.
import tempfile as _tempfile
if "HF_HOME" not in _os.environ:
    _hf_home = _tempfile.mkdtemp(prefix="hippolm_hf_")
    _os.environ["HF_HOME"] = _hf_home
    _os.environ.setdefault("HF_DATASETS_CACHE", _os.path.join(_hf_home, "datasets"))
    _os.environ.setdefault("HUGGINGFACE_HUB_CACHE", _os.path.join(_hf_home, "hub"))

import os
import sys
import argparse
import logging
import time
import random
import tempfile
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
    """Stream text samples from a HuggingFace dataset via hf-mirror.com.

    Supports both pretraining (text/content field) and SFT (messages field) formats.
    """

    def __init__(
        self,
        dataset_name: str,
        tokenizer,
        split: str = "train",
        max_seq_len: int = 2048,
        num_samples: Optional[int] = None,
        text_field: str = "content",
        is_sft: bool = False,
        config_name: Optional[str] = None,
        seed: int = 42,
        shuffle: bool = True,
        rank: int = 0,
        world_size: int = 1,
    ):
        super().__init__()
        self.dataset_name = dataset_name
        self.tokenizer = tokenizer
        self.split = split
        self.max_seq_len = max_seq_len
        self.num_samples = num_samples
        self.text_field = text_field
        self.is_sft = is_sft
        self.config_name = config_name
        self.seed = seed
        self.shuffle = shuffle
        self.rank = rank
        self.world_size = max(int(world_size), 1)

        # Note: HF_ENDPOINT and HF_HOME are set at module import time
        # (top of this file) so that huggingface_hub picks them up before
        # HfApi() caches its default endpoint. Streaming keeps the dataset
        # off-disk; only small metadata is fetched to the temp HF_HOME.

        # Lazy init dataset
        self._ds = None
        self._iterator = None

    def _ensure_dataset(self):
        if self._ds is None:
            from datasets import load_dataset
            logging.getLogger(__name__).info(
                f"Streaming (no local parquet cache) dataset {self.dataset_name}"
                f"{f' ({self.config_name})' if self.config_name else ''}"
                f" (split={self.split}, endpoint={os.environ.get('HF_ENDPOINT')},"
                f" HF_HOME={os.environ.get('HF_HOME')})"
            )
            load_kwargs = dict(
                split=self.split,
                streaming=True,
                cache_dir=None,  # Do not persist parquet locally
            )
            if self.config_name:
                load_kwargs["name"] = self.config_name
            self._ds = load_dataset(self.dataset_name, **load_kwargs)
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

        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            local_worker_id = 0
            local_num_workers = 1
        else:
            local_worker_id = worker_info.id
            local_num_workers = worker_info.num_workers

        global_num_shards = self.world_size * local_num_workers
        global_shard_id = self.rank * local_num_workers + local_worker_id

        sample_idx = 0
        for example in self._ds:
            if sample_idx % global_num_shards != global_shard_id:
                sample_idx += 1
                continue

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

            sample_idx += 1
            if self.num_samples is not None and sample_idx >= self.num_samples:
                return

    def __len__(self):
        if self.num_samples is not None:
            return max(int(self.num_samples) // self.world_size, 1)
        return 1_000_000  # Streaming: arbitrary large number


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

    # Data
    if args.use_dummy_data:
        logger.info("Using dummy data")
        dataloader = create_dummy_dataloader(args.batch_size, args.seq_len, config.vocab_size)
    else:
        tokenizer = load_tokenizer(args.tokenizer_path)
        if args.stage == "sft":
            dataset_name = args.sft_dataset
            config_name = args.sft_config
            is_sft = True
        else:
            dataset_name = args.pretrain_dataset
            config_name = args.pretrain_config
            is_sft = False
        logger.info(f"Loading dataset: {dataset_name} (config={config_name}, stage={args.stage})")
        dataset = HFStreamingDataset(
            dataset_name=dataset_name,
            tokenizer=tokenizer,
            split="train",
            max_seq_len=args.seq_len,
            num_samples=args.max_steps * args.batch_size * args.gradient_accumulation_steps,
            is_sft=is_sft,
            config_name=config_name,
            text_field=args.text_field,
            rank=0,
            world_size=1,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=True,
        )

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
    parser.add_argument("--pretrain_dataset", type=str,
                        default="openbmb/Ultra-FineWeb-L3")
    parser.add_argument("--pretrain_config", type=str,
                        default="Ultra-FineWeb-L3-en-QA-Synthetic")
    parser.add_argument("--sft_dataset", type=str,
                        default="openbmb/UltraData-SFT-2605")
    parser.add_argument("--sft_config", type=str, default=None)
    parser.add_argument("--text_field", type=str, default="content")
    parser.add_argument("--tokenizer_path", type=str,
                        default="src/tokenizer")
    parser.add_argument("--use_dummy_data", action="store_true", default=False)
    parser.add_argument("--num_workers", type=int, default=2)

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
