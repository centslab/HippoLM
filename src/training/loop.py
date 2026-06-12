"""Per-rank training worker: setup, run, teardown.

The training entry point spawns one :func:`_train_worker` per
GPU (one per TP rank). Each worker is split into three phases,
called from a thin orchestration:

  - :func:`_setup_worker`       — env vars, NCCL/gloo init,
    per-worker logging, dataset / dataloader / prefetcher, TP
    model, GradScaler, Muon + AdamW optimizer pair. Returns a
    plain dict that the next two phases consume.
  - :func:`_run_training_loop`  — the per-step forward / backward
    / grad-to-CPU / step / log / checkpoint sequence. This is
    the heart of the worker; everything else is plumbing.
  - :func:`_teardown_worker`    — close the prefetcher thread,
    destroy the distributed process group.

The orchestration :func:`_train_worker` is the public symbol
the entry point imports. The split into three phases keeps the
training loop readable (the per-step logic used to be embedded
in a 400-line monolith with NCCL init / data plumbing mixed in)
and makes each phase independently testable.
"""
from __future__ import annotations

import logging
import math
import os
import queue
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from src.models import HippoConfig
from src.models.tp_layers import init_tp
from src.models.tp_model import TPHippoModel
from src.training import save_checkpoint
from src.training.data import (
    MultiSourceStreamingDataset,
    PrefetchBatcher,
    QueueIterator,
    StreamingDataset,
    dummy_dataloader,
)
from src.training.diagnostics import log_post_opt_diag, log_pre_step_diag
from src.training.param_offload import (
    accumulate_grads_to_cpu,
    build_param_groups,
    flush_pending_grads,
    register_grad_offload_hooks,
    zero_cpu_grad_accum,
)
from src.training.precision_config import PrecisionConfig
from src.training.tokenizer import load_tokenizer

log = logging.getLogger(__name__)


# Number of early training steps for which to emit the detailed
# pre-/post-step diagnostic (acc_max, m_max, pmax, total_norm,
# per-optimizer finite check). Gated at the call site to keep
# steady-state logging overhead at zero.
_DIAG_STEPS = 3


def _compute_and_clip_grad_norm(opts, max_norm: float) -> float:
    """Compute the L2 norm across all per-param accumulators in
    the given optimizers, all-reduce across the TP world, and
    clip in place.

    Mirrors the pre-refactor helper that lived in
    ``scripts/train.py``. Scales each ``s.accum`` by
    ``clip_coef = max_norm / (total_norm + 1e-6)`` only when the
    norm exceeds the cap, so well-behaved steps are no-ops.
    """
    import torch.distributed as dist

    local_sq = torch.zeros(1)
    for opt in opts:
        for s in opt.state.values():
            local_sq += s.accum.detach().float().pow(2).sum()
    if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        dist.all_reduce(local_sq, op=dist.ReduceOp.SUM)
    total_norm = local_sq.sqrt().item()
    if total_norm > max_norm > 0.0:
        clip_coef = max_norm / (total_norm + 1e-6)
        for opt in opts:
            for s in opt.state.values():
                s.accum.mul_(clip_coef)
    if max_norm > 0.0:
        scale = max_norm / (total_norm + 1e-6)
        for opt in opts:
            for s in opt.state.values():
                s.accum.mul_(scale)
    return total_norm


# --------------------------------------------------------------------------- #
# Phase 1: setup.                                                             #
# --------------------------------------------------------------------------- #
def _setup_worker(
    rank: int,
    args,
    gpus: List[int],
    port: int,
    run_dir_path: str,
    shared_batch_queues: Optional[List[queue.Queue]] = None,
) -> Dict[str, Any]:
    """Build everything the training loop needs.

    Returns a dict with the keys:

    - ``rank``, ``args``, ``gpus``, ``port``, ``run_dir``,
      ``logger``: identity + plumbing
    - ``config``: the :class:`HippoConfig` this rank built
    - ``dataloader``, ``prefetcher``: the data path
    - ``model``: the per-rank TP model fragment
    - ``scaler``: the AMP :class:`GradScaler` (disabled)
    - ``muon_opt``, ``adamw_opt``: the per-device optimizer pair
    - ``backend``: the distributed backend (``"nccl"`` or
      ``"gloo"``) chosen at process-group init time

    The distributed process group (``dist.init_process_group``)
    is initialized as a side effect and the worker is bound to
    its training GPU via ``torch.cuda.set_device``. The caller
    is expected to call :func:`_teardown_worker` (which calls
    ``dist.destroy_process_group``) on every code path.
    """
    import torch.distributed as dist

    # Env vars for env:// init_method.
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(len(gpus))
    os.environ["LOCAL_RANK"] = str(rank)

    # Pin to this rank's training GPU before NCCL init. See the
    # long comment in scripts/train.py for the cudaErrorMemoryAllocation
    # scenario that motivates this; it has not been abbreviated here
    # because the next person to touch this file will need the
    # history. gloo doesn't have device_id, so we skip that
    # argument for the sim path.
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
    logger = logging.getLogger(f"train_rank{rank}")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handlers = [logging.StreamHandler(), logging.FileHandler(log_file)]
        logger.handlers = handlers
        logger.propagate = False
    logger.info(f"Worker {rank}/{len(gpus)} starting on cuda:{gpus[rank]}")

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

    # ---- Data path ----
    if args.use_dummy_data:
        # Dummy: each rank gets its own random DataLoader. The
        # batches are random per rank (so technically violate the
        # TP input-replication contract), but it's enough for a
        # smoke test of the sharded forward/backward path.
        dataloader = dummy_dataloader(
            args.batch_size, args.seq_len, config.vocab_size,
        )
        prefetcher = None
    else:
        if shared_batch_queues is None:
            my_queue: queue.Queue = queue.Queue(maxsize=2)
            all_queues = [my_queue]
        else:
            my_queue = shared_batch_queues[rank]
            all_queues = list(shared_batch_queues)
        if rank == 0:
            tokenizer = load_tokenizer(args.tokenizer_path)
            if args.stage == "sft":
                hf_name = args.sft_dataset_hf
                ms_name = args.sft_dataset_ms
                config_name = args.sft_config
                is_sft = True
                dataset = StreamingDataset(
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
            elif getattr(args, "pretrain_multi_source", False):
                # Ultra-FineWeb-L3 multi-source: 4 subsets in a
                # 2-phase schedule (multi-style first, then QA,
                # 2:1 en:zh). Each sub is its own StreamingDataset
                # so the per-subset network/cache config is
                # independent. ``MultiSourceStreamingDataset`` lazily
                # opens each sub only when its phase is reached.
                sub_specs = [
                    ("en_multi", args.pretrain_en_multi_config),
                    ("zh_multi", args.pretrain_zh_multi_config),
                    ("en_qa",    args.pretrain_en_qa_config),
                    ("zh_qa",    args.pretrain_zh_qa_config),
                ]
                subs = []
                for label, cfg in sub_specs:
                    logger.info(
                        f"Multi-source pretrain sub [{label}]:"
                        f" hf={args.pretrain_dataset_hf}"
                        f" ms={args.pretrain_dataset_ms}"
                        f" config={cfg}"
                    )
                    subs.append(StreamingDataset(
                        dataset_name=args.pretrain_dataset_hf,
                        ms_dataset_name=args.pretrain_dataset_ms,
                        use_modelscope=args.use_modelscope,
                        tokenizer=tokenizer,
                        split="train",
                        max_seq_len=args.seq_len,
                        is_sft=False,
                        config_name=cfg,
                        text_field=args.text_field,
                        shuffle=args.shuffle,
                    ))
                # Phase 0: multi-style, en *2 + zh *1.
                # Phase 1: QA, en *2 + zh *1.
                schedule = [
                    [(0, 2), (1, 1)],  # multi-style block
                    [(2, 2), (3, 1)],  # QA block
                ]
                dataset = MultiSourceStreamingDataset(subs, schedule)
                logger.info(
                    f"Multi-source pretrain: 4 subsets, "
                    f"phase_schedule={schedule} "
                    f"(multi-style first, then QA, en:zh=2:1)"
                )
            else:
                hf_name = args.pretrain_dataset_hf
                ms_name = args.pretrain_dataset_ms
                config_name = args.pretrain_config
                is_sft = False
                dataset = StreamingDataset(
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
            prefetcher = PrefetchBatcher(
                dataset, batch_size=args.batch_size, queues=all_queues,
            )
        else:
            prefetcher = None
        dataloader = QueueIterator(my_queue)

    # ---- TP model ----
    init_tp(world_size=len(gpus), devices=gpus, backend=backend)
    torch.manual_seed(args.seed)
    # Resolve the precision config: ``args.precision`` is a raw
    # yml-shaped dict (set by the CLI overlay in
    # ``scripts.cli.parse_args``); convert to a typed
    # ``PrecisionConfig`` so the model_weights dtype flows into
    # the model construction and the optimizer-state dtypes
    # flow into ``build_param_groups`` below. ``None`` /
    # missing means the canonical yml defaults.
    precision = PrecisionConfig.from_dict(
        getattr(args, "precision", None)
    )
    weight_dtype = precision.model_weights.dtype.to_torch()
    model = TPHippoModel(config, devices=gpus, dtype=weight_dtype)
    dist.barrier()
    model.sync_replicated_from(gpus[0])
    if rank == 0:
        trainable = sum(
            p.numel() for p in model.trainable_parameters(gpus[0])
        )
        logger.info(
            f"TP model built. Per-device trainable params: {trainable:,}"
        )

    # ---- GradScaler (disabled) ----
    # V100 has FP16 tensor cores; the model runs in FP16. We
    # disable the scaler (PyTorch's GradScaler refuses to unscale
    # FP16 grads in this version) and run plain FP16 backward.
    # See the long comment in scripts/train.py for the history.
    scaler = torch.amp.GradScaler("cuda", enabled=False)

    # ---- Optimizers (CPU offload) ----
    muon_opt, adamw_opt = build_param_groups(
        model, device=gpus[rank],
        lr_muon=args.muon_lr,
        lr_adamw=args.learning_rate,
        weight_decay=args.weight_decay,
        muon_momentum=args.muon_momentum,
        precision=precision,
    )
    # Install the per-param post-accumulate-grad hooks so each
    # param's grad is DMA'd to its CPU accumulator as soon as
    # autograd computes it (during ``backward()``), not after.
    # This keeps the peak GPU grad memory to "at most one
    # param's grad" rather than "the sum of all grads". The
    # training loop calls :func:`flush_pending_grads` once per
    # microbatch to sync CUDA and apply the CPU adds. See
    # :func:`register_grad_offload_hooks` for the design.
    register_grad_offload_hooks([muon_opt, adamw_opt])
    n_muon = sum(s.param.numel() for s in muon_opt.state.values())
    n_adamw = sum(s.param.numel() for s in adamw_opt.state.values())
    n_muon_rows = sum(s.shape[0] for s in muon_opt.state.values())
    muon_bytes = (
        n_muon * 1             # int8 momentum
        + n_muon_rows * 2      # BF16 per-row scale
        + n_muon * 2           # BF16 accum
    )
    adamw_bytes = n_adamw * 6  # BF16 m + BF16 v + BF16 accum
    logger.info(
        f"Device {gpus[rank]}: Muon params={n_muon:,}"
        f" ({muon_bytes / 1024**3:.2f} GB int8 momentum + BF16 scale on CPU),"
        f" AdamW params={n_adamw:,}"
        f" ({adamw_bytes / 1024**3:.2f} GB BF16 m+v on CPU)."
    )

    return {
        "rank": rank,
        "args": args,
        "gpus": gpus,
        "port": port,
        "run_dir": run_dir,
        "logger": logger,
        "config": config,
        "dataloader": dataloader,
        "prefetcher": prefetcher,
        "model": model,
        "scaler": scaler,
        "muon_opt": muon_opt,
        "adamw_opt": adamw_opt,
        "backend": backend,
    }


# --------------------------------------------------------------------------- #
# Phase 2: run.                                                               #
# --------------------------------------------------------------------------- #
def _run_training_loop(ctx: Dict[str, Any]) -> None:
    """Per-step training loop. Consumes the dict from
    :func:`_setup_worker` and never mutates any plumbing keys.
    """
    import torch.distributed as dist

    rank = ctx["rank"]
    args = ctx["args"]
    gpus = ctx["gpus"]
    run_dir = ctx["run_dir"]
    logger = ctx["logger"]
    dataloader = ctx["dataloader"]
    model = ctx["model"]
    scaler = ctx["scaler"]
    muon_opt = ctx["muon_opt"]
    adamw_opt = ctx["adamw_opt"]

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

                # First-batch heartbeat (data path is alive).
                if batch_idx == 0 and rank == 0:
                    logger.info(
                        f"First batch ready: "
                        f"input_ids={tuple(batch['input_ids'].shape)}, "
                        f"labels={tuple(batch['labels'].shape)} - "
                        f"starting training"
                    )

                input_ids = batch["input_ids"].to(gpus[rank], non_blocking=True)
                labels = batch["labels"].to(gpus[rank], non_blocking=True)

                with torch.amp.autocast(
                    device_type="cuda", dtype=torch.float16,
                ):
                    outputs = model(input_ids, labels=labels)
                    loss = outputs["loss"]
                mb_loss = loss.detach().float().item()
                if rank == 0 and (
                    global_step < _DIAG_STEPS or not math.isfinite(mb_loss)
                ):
                    logger.info(
                        f"  [diag] mb={batch_idx} loss={mb_loss:.6e}"
                    )
                (loss / args.gradient_accumulation_steps).backward()
                accumulated_loss += mb_loss

                # Sync CUDA and apply the CPU-side adds for the
                # D2H transfers that the per-param hooks issued
                # during ``backward()`` above. The hooks (see
                # :func:`register_grad_offload_hooks`) already
                # cleared each ``.grad`` as they fired, so the
                # peak GPU grad memory at this point is at most
                # one param's grad (whatever autograd is
                # currently processing — none, since backward
                # has returned).
                flush_pending_grads(sync_device=gpus[rank])
                del loss, outputs, input_ids, labels
                microbatch_in_cycle += 1

                if microbatch_in_cycle >= args.gradient_accumulation_steps:
                    # Inf/nan check on the accumulated CPU grads.
                    found_inf = False
                    for opt in (muon_opt, adamw_opt):
                        for s in opt.state.values():
                            if not torch.isfinite(s.accum).all():
                                found_inf = True
                                break
                        if found_inf:
                            break
                    if not found_inf:
                        total_norm = _compute_and_clip_grad_norm(
                            [muon_opt, adamw_opt], args.max_grad_norm,
                        )
                        if global_step < _DIAG_STEPS and rank == 0:
                            log_pre_step_diag(
                                logger,
                                step=global_step,
                                total_norm=total_norm,
                                muon_state=muon_opt.state,
                                adamw_state=adamw_opt.state,
                            )
                        muon_opt.step()
                        if global_step < _DIAG_STEPS and rank == 0:
                            log_post_opt_diag(
                                logger,
                                step=global_step,
                                opt_label="muon",
                                state=muon_opt.state,
                            )
                        adamw_opt.step()
                        if global_step < _DIAG_STEPS and rank == 0:
                            log_post_opt_diag(
                                logger,
                                step=global_step,
                                opt_label="adamw",
                                state=adamw_opt.state,
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
                        for d in gpus:
                            free, total = torch.cuda.mem_get_info(d)
                            used = total - free
                            logger.info(
                                f"  device {d} VRAM: {used / 1024**3:.2f} /"
                                f" {total / 1024**3:.2f} GB"
                            )

                    # Periodic checkpoint. Rank 0 only; the other
                    # ranks' sharded state is not gathered here.
                    # Best-effort: log and continue on failure.
                    if (
                        rank == 0
                        and args.checkpoint_interval > 0
                        and global_step % args.checkpoint_interval == 0
                    ):
                        try:
                            ckpt_path = save_checkpoint(
                                model,
                                optimizers={"muon": muon_opt, "adamw": adamw_opt},
                                scaler=scaler,
                                step=global_step,
                                loss=avg_loss,
                                checkpoint_dir=run_dir / "checkpoints",
                            )
                            logger.info(f"  checkpoint saved: {ckpt_path}")
                        except Exception as ckpt_err:
                            logger.warning(
                                f"  checkpoint save failed at step"
                                f" {global_step}: {ckpt_err!r};"
                                f" continuing training"
                            )

            if global_step >= args.max_steps:
                break
    finally:
        _teardown_worker(ctx)


# --------------------------------------------------------------------------- #
# Phase 3: teardown.                                                          #
# --------------------------------------------------------------------------- #
def _teardown_worker(ctx: Dict[str, Any]) -> None:
    """Close the prefetcher thread and destroy the distributed
    process group. Idempotent: safe to call multiple times (the
    finally clause in :func:`_run_training_loop` calls this, and
    a caller that bails out of setup_worker would call it
    directly with a partially-populated ctx).
    """
    import torch.distributed as dist

    prefetcher = ctx.get("prefetcher")
    if prefetcher is not None:
        try:
            prefetcher.close()
        except Exception as e:
            log.warning("prefetcher close failed: %r", e)
    if dist.is_available() and dist.is_initialized():
        try:
            dist.destroy_process_group()
        except Exception as e:
            log.warning("dist.destroy_process_group failed: %r", e)


# --------------------------------------------------------------------------- #
# Public orchestration.                                                       #
# --------------------------------------------------------------------------- #
def _train_worker(
    rank: int,
    args,
    gpus: List[int],
    port: int,
    run_dir_path: str,
    shared_batch_queues: Optional[List[queue.Queue]] = None,
) -> None:
    """Single-worker training entry point (one process per GPU).

    Thin orchestration: setup -> run (which owns the teardown
    via its finally clause).
    """
    ctx = _setup_worker(
        rank, args, gpus, port, run_dir_path, shared_batch_queues,
    )
    _run_training_loop(ctx)
