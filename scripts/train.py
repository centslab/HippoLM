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
import sys
from pathlib import Path

# Make the repo root importable so ``from src...`` works. Must be
# set before any ``from src.training.env import ...`` line.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# All env-var shimming (HF_ENDPOINT, NCCL transport, datasets retry
# config, socket timeout) is delegated to src.training.env. Must be
# called BEFORE any heavy import — torch.distributed reads its env
# at init time, and huggingface_hub reads HF_TOKEN at HfApi()
# construction.
from src.training.env import configure_runtime_environment

configure_runtime_environment()

import logging
import os
import time
from pathlib import Path

import torch

from src.training.launch import select_tp_gpus, expand_for_tp_sim
from src.training.loop import _train_worker
from scripts.cli import parse_args

# Top-level logger for the main process (per-rank workers have their
# own logger from setup_logging() in _train_worker).
log = logging.getLogger(__name__)


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


def create_output_dir(output_dir: str) -> Path:
    """Create timestamped output directory."""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = Path(output_dir) / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / "checkpoints").mkdir(exist_ok=True)
    (run_dir / "logs").mkdir(exist_ok=True)
    (run_dir / "eval").mkdir(exist_ok=True)

    return run_dir


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

    gpus = select_tp_gpus(min_memory_mb=args.min_gpu_memory_mb)
    if not gpus and args.tp_sim:
        # ``select_tp_gpus`` looks for a contiguous run of 2/4/8
        # GPUs with >= min_memory_mb free. On a single-GPU dev
        # box the scan returns []; sim mode explicitly opts into
        # "I'll fake it with one card", so fall back to cuda:0.
        if torch.cuda.is_available():
            gpus = [0]
            log.warning(
                "[TP-SIM] auto-detect found no 2/4/8-GPU run;"
                " falling back to cuda:0 for sim"
            )
        else:
            log.error("--tp_sim requires CUDA")
            return run_dir
    if not gpus and not args.tp_sim:
        # No contiguous 2/4/8-GPU run. Single-GPU is still a valid
        # run configuration — the TP code paths all no-op when
        # world_size == 1. Fall back to cuda:0 so single-GPU
        # boxes get a clean error path (OOM, model too large)
        # instead of a misleading "no TP group" failure.
        if torch.cuda.is_available() and torch.cuda.device_count() >= 1:
            gpus = [0]
            log.warning(
                "no contiguous 2/4/8-GPU run is available;"
                " falling back to single-GPU training on cuda:0"
                " (TP will no-op, world_size=1)."
            )
        else:
            log.error("no CUDA device is available.")
            return run_dir

    # TP-simulation: collapse the detected GPU list to a single
    # physical device replicated ``tp_size`` times (or disable
    # sim if ``tp_size`` is invalid). All warnings go through
    # the logger at WARNING level.
    gpus, args.tp_sim = expand_for_tp_sim(gpus, args.tp_size, args.tp_sim)

    # ``log.warning`` (not ``log.info``) so the line is visible
    # even before :func:`setup_logging` wires up the root
    # handler — :func:`main` does not call it, so the default
    # root logger (level=WARNING, no handler) would silently
    # drop ``log.info``. Same reason :func:`log.warning` is
    # used for the "no contiguous run" fallback a few lines
    # above.
    log.warning(
        "Selected contiguous TP group: %s (world_size=%d, sim=%s)",
        gpus, len(gpus), args.tp_sim,
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
        # uniform :class:`QueueIterator`. Building an independent
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
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
