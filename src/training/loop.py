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
import time
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
from src.training.data.cache import purge_stale_cache_if_no_hit
from src.training.diagnostics import log_post_opt_diag, log_pre_step_diag
from src.training.param_offload import (
    accumulate_grads_to_cpu,
    build_param_groups,
    flush_manual_flush_params,
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


def wsd_lr(
    step: int,
    peak_lr: float,
    warmup_steps: int,
    decay_steps: int,
    max_steps: int,
    min_lr: float = 0.0,
) -> float:
    """WSD (Warmup-Stable-Decay) learning rate schedule.

    Three phases over the ``max_steps`` training horizon:

      1. **Warmup**  : steps ``[0, warmup_steps)`` —
         linear ramp from ``0`` to ``peak_lr``.
      2. **Stable**  : steps ``[warmup_steps, max_steps - decay_steps)`` —
         constant at ``peak_lr``.
      3. **Decay**   : steps ``[max_steps - decay_steps, max_steps)`` —
         linear ramp from ``peak_lr`` to ``min_lr``.

    Edge cases:

    - ``step < 0`` or ``step >= max_steps``  : ``0`` (clamped; the
      training loop never reaches these but tests / callers may).
    - ``warmup_steps == 0`` : no warmup phase; step 0 is already at
      ``peak_lr``.
    - ``decay_steps == 0``  : no decay phase; LR stays at ``peak_lr``
      to the end (matches the pre-WSD "constant LR" behavior).
    - ``warmup + decay > max_steps`` : stable phase shrinks to zero;
      the warmup phase is still walked first, then the decay phase
      kicks in at ``max_steps - decay_steps`` (overlap region is
      governed by phase priority: warmup > stable > decay).
    - ``peak_lr <= 0`` : returns ``peak_lr`` unchanged (no-op).

    Both optimizers in this codebase (``CPUAdamW`` and ``CPUMuon``)
    read ``self.lr`` at the start of :meth:`step`, so updating
    ``muon_opt.lr`` and ``adamw_opt.lr`` between optimizer
    constructions and each step is enough — no
    ``torch.optim.lr_scheduler`` wrapper needed.
    """
    if peak_lr <= 0.0:
        return peak_lr
    if step < 0 or step >= max_steps:
        return 0.0
    # Phase 1: warmup.
    if warmup_steps > 0 and step < warmup_steps:
        # step 0 -> 0, step warmup_steps-1 -> peak * (warmup-1)/warmup.
        # The very first stable step (== warmup_steps) hits peak.
        return peak_lr * step / warmup_steps
    # Phase 3: decay.
    decay_start = max_steps - decay_steps
    if decay_steps > 0 and step >= decay_start:
        # step decay_start -> peak_lr, step max_steps-1 -> min_lr.
        # Decay spans ``decay_steps`` steps [decay_start, decay_start +
        # decay_steps - 1]; divisor is ``decay_steps - 1`` so the
        # final step lands exactly on min_lr. With decay_steps == 1
        # the single decay step jumps straight to min_lr.
        denom = decay_steps - 1 if decay_steps > 1 else 1
        progress = (step - decay_start) / denom
        return peak_lr + (min_lr - peak_lr) * progress
    # Phase 2: stable.
    return peak_lr


def _slice_cu_seqlens(
    global_cu_seqlens: torch.Tensor,
    chunk_start: int,
    chunk_end: int,
) -> torch.Tensor:
    """Slice a global ``cu_seqlens`` tensor to a chunk-local one.

    The chunked-training invariant: each chunk covers tokens at
    global offsets ``[chunk_start, chunk_end)`` within the
    super-long packed sequence. Doc boundaries (encoded in
    ``global_cu_seqlens`` as cumulative end-offsets) that fall
    inside this range become boundaries in the chunk-local
    ``cu_seqlens``; boundaries outside are dropped.

    The returned tensor:

      - starts at ``0`` (chunk-local origin),
      - lists each global doc-end that lies in
        ``(chunk_start, chunk_end]`` shifted by ``-chunk_start``,
      - ends at ``chunk_end - chunk_start`` (chunk-local right edge).

    Edge cases:

      - A global boundary exactly at ``chunk_start`` becomes the
        chunk's local ``0`` (the chunk opens a new "doc" — same
        convention as :func:`pack_chunk_aligned` where the
        ``pack_end`` entry resets the KDA + ShortConv state).
      - A global boundary exactly at ``chunk_end`` becomes the
        chunk's local ``chunk_size`` (the chunk closes its last
        doc at the chunk's right edge).
      - Empty chunk (``chunk_start == chunk_end``) returns
        ``[0]`` (a degenerate single-entry sequence).

    This is what the model receives as ``cu_seqlens`` for the
    ShortConvolution call (depthwise conv kernel resets at doc
    boundaries). The KDA sub-layer itself does NOT receive
    ``cu_seqlens`` — see ``TPKDA.forward``.
    """
    if chunk_start == chunk_end:
        # Empty chunk: degenerate but well-defined.
        return global_cu_seqlens.new_zeros(1)

    in_range = (global_cu_seqlens >= chunk_start) & (
        global_cu_seqlens <= chunk_end
    )
    local = global_cu_seqlens[in_range] - chunk_start
    # Defensive: ensure 0 at the start and chunk_size at the end.
    chunk_size = chunk_end - chunk_start
    if local.numel() == 0 or local[0].item() != 0:
        local = torch.cat([
            local.new_zeros(1, dtype=local.dtype), local,
        ])
    if local[-1].item() != chunk_size:
        local = torch.cat([
            local, local.new_full((1,), chunk_size, dtype=local.dtype),
        ])
    return local


def _compute_and_clip_grad_norm(opts, max_norm: float) -> float:
    """Compute the L2 norm across all per-param accumulators in
    the given optimizers, all-reduce across the TP world, and
    clip in place.

    For each param the accumulator is whichever tensor holds
    the sum-of-microbatch-grads for that param at this point:
    ``s.m`` (AdamW), ``s.accum`` (quantized muon: int8 /
    mxfp8), or ``s.mom_buf`` (fp* muon, merged design).
    Scales the accumulator by ``clip_coef = max_norm /
    (total_norm + 1e-6)`` only when the norm exceeds the cap,
    so well-behaved steps are no-ops.

    The clip is always a plain ``.mul_(coef)`` because all
    three accumulators are floating-point (BF16 / FP16 / FP32);
    the previous FP8 ``mom_buf`` round-trip via
    ``_scale_mxfp8_mom_buf`` is gone — the separate-accumulator
    redesign moved the FP8 storage away from the clip path.
    """
    import torch.distributed as dist

    local_sq = torch.zeros(1)
    for opt in opts:
        for s in opt.state.values():
            accum = (
                s.m if s.kind == "adamw"
                else (s.accum if s.accum is not None else s.mom_buf)
            )
            local_sq += accum.detach().float().pow(2).sum()
    if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        dist.all_reduce(local_sq, op=dist.ReduceOp.SUM)
    total_norm = local_sq.sqrt().item()
    # Clip only when the norm exceeds the cap.  An earlier revision
    # applied a second unconditional ``s.accum.mul_(max_norm / (total_norm
    # + eps))`` below this guard, which had two bugs:
    #   (a) when total_norm < max_norm it AMPLIFIED small grads
    #       (``max_norm / total_norm > 1``), wrecking training stability;
    #   (b) when total_norm > max_norm the clip was applied twice
    #       (quadratic clip instead of linear).
    if max_norm > 0.0 and total_norm > max_norm:
        clip_coef = max_norm / (total_norm + 1e-6)
        from src.training.param_offload import _scale_accum
        for opt in opts:
            for s in opt.state.values():
                _scale_accum(s, clip_coef)
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
    # TF32 for FP32 tensor-core matmul (FP16 weights + FP16
    # activations still use tensor cores; this is for any leftover
    # FP32 matmuls — layer norms etc.). "high" picks the fastest of
    # the three TF32 modes (10-bit mantissa); "highest" is exact FP32
    # for numerics-sensitive work. The global default is "highest"
    # so "high" is a deliberate opt-in for speed.
    torch.set_float32_matmul_precision("high")
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
    # Note: ALL yml-driven fields must be passed here from args,
    # otherwise :class:`HippoConfig`'s dataclass ``repr`` (logged
    # below) will show the dataclass default instead of the
    # actual yml value — a recurring source of "did my yml
    # change take effect?" confusion. If you add a new yml key,
    # add the corresponding ``args.<name>=`` line here.
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
        pack_chunk_size=args.pack_chunk_size,
        pack_buffer_size=args.pack_buffer_size,
        kda_skip_aqk_akk_saved=getattr(args, "kda_skip_aqk_akk_saved", False),
        ffn_nvfp4=getattr(args, "ffn_nvfp4", False),
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
            # Pre-training cache integrity check: gather the expected
            # (ms_name, config_name) tuples for THIS run, then purge
            # the cache if NONE of them have a hit. Prevents stale
            # parquets from a *previous* dataset the dev box trained
            # on from accumulating disk forever. See
            # :func:`purge_stale_cache_if_no_hit` for the contract.
            if args.stage == "sft":
                expected_cache_keys = [
                    (args.sft_dataset_ms, args.sft_config),
                ]
            elif getattr(args, "pretrain_multi_source", False):
                expected_cache_keys = [
                    (args.pretrain_dataset_ms, args.pretrain_en_multi_config),
                    (args.pretrain_dataset_ms, args.pretrain_zh_multi_config),
                    (args.pretrain_dataset_ms, args.pretrain_en_qa_config),
                    (args.pretrain_dataset_ms, args.pretrain_zh_qa_config),
                ]
            else:
                expected_cache_keys = [
                    (args.pretrain_dataset_ms, args.pretrain_config),
                ]
            purge_stale_cache_if_no_hit(
                expected_cache_keys, log=logger,
            )

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
                dataset,
                batch_size=args.batch_size,
                queues=all_queues,
                seq_len=args.seq_len,
                chunk_size=config.pack_chunk_size,
                pad_id=tokenizer.pad_token_id,
                eos_id=tokenizer.eos_token_id,
                pack_buffer_size=args.pack_buffer_size,
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
    # The forward runs at the activation dtype configured in
    # ``precision.activations`` (fp16 by default, which matches
    # the canonical FP16 weights). We disable the scaler
    # (PyTorch's GradScaler refuses to unscale FP16 grads in this
    # version) and run plain backward in the configured dtype.
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
    #
    # ``manual_flush_params`` are the tied-embed params whose
    # grad is the sum of (a) the natural-path grad from
    # ``nn.Embedding`` and (b) the FusedLinearCE dw, which is
    # added by ``_TiedFusedLCEFunction.backward`` AFTER the
    # natural-path ``accumulate_grad_`` has fired. For these
    # params we skip the streaming hook and instead call
    # :func:`flush_manual_flush_params` per microbatch (after
    # ``backward()`` returns) so we read the *complete* grad.
    manual_flush = []
    # nn.ModuleDict has no ``.get()`` (only ``__getitem__``), so fall
    # back to an explicit ``in`` check. ``embed_tokens`` is always
    # present in the current TPHippoModel layout, so the branch is
    # only defensive against future layout changes.
    device_mods = model.replicated_per_device[str(gpus[0])]
    embed_param = device_mods["embed_tokens"] if "embed_tokens" in device_mods else None
    if embed_param is not None and hasattr(embed_param, "weight"):
        manual_flush.append(embed_param.weight)
    register_grad_offload_hooks(
        [muon_opt, adamw_opt],
        manual_flush_params=manual_flush or None,
    )
    n_muon = sum(s.param.numel() for s in muon_opt.state.values())
    n_adamw = sum(s.param.numel() for s in adamw_opt.state.values())
    n_muon_rows = sum(s.shape[0] for s in muon_opt.state.values())
    # Per-param byte breakdown, kept in named variables so the log
    # line below (and any future debugger) can see the components
    # separately.
    #
    # Accumulator layout: AdamW uses ``s.m`` (BF16) as both the
    # grad accumulator and the first moment — no separate
    # ``accum`` buffer. Muon uses the same merged design for fp*
    # storage (``s.mom_buf`` doubles as the accumulator). For
    # quantized muon (int8 / mxfp8) there is a SEPARATE bf16
    # ``s.accum`` buffer that holds the per-mb grad sum cheaply
    # (the old dequant-add-requant per-mb cycle was the CPU
    # bottleneck at 8+ seconds per microbatch — see
    # ``docs/optimizer_layout.md`` for the design). The byte
    # cost of the separate ``accum`` is 2 bytes/elt for the
    # quantized-muon params, paid back by ~10x faster per-mb
    # sync.
    #
    # Muon's momentum storage dtype is configurable
    # (``precision.muon_momentum``): int8 with a BF16 per-row
    # scale, mxfp8 with an E8M0 per-block scale (1 byte per
    # ``block_size`` elements, default 32), or full-precision
    # bf16/fp16/fp32 (no scale). We read the actual dtype off
    # the first state entry rather than hardcoding, so the log
    # message reflects whatever the user configured.
    muon_states = list(muon_opt.state.values())
    if muon_states:
        muon_mom_dtype = muon_states[0].mom_buf.dtype
        muon_mom_bytes_per_elt = muon_mom_dtype.itemsize
        # Scale tensor exists for int8 AND mxfp8 (None for fp*).
        muon_has_scale = muon_states[0].mom_scale is not None
        # mxfp8: per-block E8M0 scale; total scale bytes is
        # ``ceil(cols / block_size)`` per row, 1 byte per entry.
        mxfp8_block_size = muon_states[0].mxfp8_block_size
        # Separate ``accum`` buffer exists only for quantized
        # muon (int8 / mxfp8) — fp* muon keeps the merged-
        # accumulator design. Checked per-state (not just the
        # first) so a heterogeneous model still computes
        # correctly.
        quantized_muon_params = sum(
            s.param.numel() for s in muon_states if s.accum is not None
        )
    else:
        muon_mom_dtype = torch.int8
        muon_mom_bytes_per_elt = 1
        muon_has_scale = False
        mxfp8_block_size = None
        quantized_muon_params = 0
    muon_mom = n_muon * muon_mom_bytes_per_elt
    if muon_has_scale:
        if mxfp8_block_size is not None:
            # mxfp8: scale bytes = rows * ceil(cols / block_size)
            # summed over all muon params. We approximate by
            # summing per-state contributions; for the typical
            # case (all params share the same block_size and
            # similar cols) this is close to the true total.
            n_muon_scale_entries = sum(
                (s.shape[1] + mxfp8_block_size - 1) // mxfp8_block_size
                for s in muon_states
            )
            muon_scale = n_muon_scale_entries * 1  # 1 byte / E8M0
        else:
            # int8: per-row BF16 scale.
            muon_scale = n_muon_rows * 2
    else:
        muon_scale = 0
    # Separate ``accum`` cost: 2 bytes/elt (BF16) for every
    # quantized-muon param. Zero for fp* muon (no separate
    # buffer) and for AdamW (its ``s.m`` is the merged
    # accumulator).
    muon_accum = quantized_muon_params * 2
    adamw_m = n_adamw * 2
    adamw_v = n_adamw * 2
    muon_bytes = muon_mom + muon_scale + muon_accum
    adamw_bytes = adamw_m + adamw_v
    muon_mom_label = (
        f"{str(muon_mom_dtype).replace('torch.', '')}_mom"
    )
    if mxfp8_block_size is not None:
        muon_scale_str = (
            f" + e8m0_scale(bs={mxfp8_block_size})="
            f"{muon_scale / 1024**3:.3f} GB"
        )
    elif muon_has_scale:
        muon_scale_str = (
            f" + bf16_scale={muon_scale / 1024**3:.3f} GB"
        )
    else:
        muon_scale_str = ""
    muon_accum_str = (
        f" + bf16_accum={muon_accum / 1024**3:.3f} GB"
        if muon_accum > 0 else ""
    )
    logger.info(
        f"Device {gpus[rank]}: Muon params={n_muon:,}"
        f" ({muon_bytes / 1024**3:.2f} GB total:"
        f" {muon_mom_label}={muon_mom / 1024**3:.3f} GB"
        f"{muon_scale_str}{muon_accum_str}),"
        f" AdamW params={n_adamw:,}"
        f" ({adamw_bytes / 1024**3:.2f} GB total:"
        f" bf16_m={adamw_m / 1024**3:.3f} GB"
        f" + bf16_v={adamw_v / 1024**3:.3f} GB)"
        f" [separate-accum for quantized muon: cheap CPU bf16 add per mb]."
    )

    return {
        "rank": rank,
        "args": args,
        "gpus": gpus,
        "port": port,
        "run_dir": run_dir,
        "logger": logger,
        "config": config,
        "precision": precision,
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
    # WSD LR scheduler knobs (see :func:`wsd_lr`). The two peak LRs
    # (``muon_lr`` and ``learning_rate``) are scaled independently by
    # the same schedule — both peak at the start of the stable phase
    # and decay together. Defaults of 0/0 disable scheduling (constant
    # peak LR), preserving the pre-WSD behavior for short smoke runs
    # that don't opt in.
    warmup_steps = getattr(args, "lr_warmup_steps", 0)
    decay_steps = getattr(args, "lr_decay_steps", 0)
    peak_muon_lr = args.muon_lr
    peak_adamw_lr = args.learning_rate
    # Activation precision comes from the precision config:
    # fp16 / bf16 → ``torch.amp.autocast`` with that dtype
    # (tensor-core matmul); fp32 → autocast disabled (pure FP32
    # forward). Resolved once per worker because autocast is a
    # hot-path and the dtype never changes after setup.
    precision = ctx["precision"]
    autocast_enabled = precision.autocast_enabled
    autocast_dtype = precision.autocast_dtype

    # Chunked-training configuration. ``seq_len`` is now the TOTAL
    # tokens per step (super-long FFD-packed sequence); the
    # forward is split into ``n_chunks`` chunks of
    # ``micro_batch_size`` tokens each, with the KDA recurrent
    # state carried between chunks (full BPTT through the
    # carried state). ``n_chunks`` is derived as
    # ``seq_len // micro_batch_size`` and replaces the legacy
    # ``gradient_accumulation_steps`` knob — same number, but
    # now it controls the number of chunks per step rather than
    # the number of independent microbatches per step.
    micro_batch_size = getattr(args, "micro_batch_size", 0) or 0
    if micro_batch_size <= 0 or micro_batch_size > args.seq_len:
        # Default / test-config fallback: one chunk per step
        # (legacy single-chunk behavior). This makes quick.yml
        # and other small configs Just Work without forcing every
        # test to set ``micro_batch_size`` explicitly.
        micro_batch_size = args.seq_len
    n_chunks = args.seq_len // micro_batch_size
    if n_chunks * micro_batch_size != args.seq_len:
        raise ValueError(
            f"seq_len ({args.seq_len}) must be a multiple of"
            f" micro_batch_size ({micro_batch_size}); got"
            f" seq_len / micro_batch_size = {args.seq_len / micro_batch_size}"
        )

    global_step = 0
    accumulated_loss = 0.0

    try:
        for epoch in range(args.epochs):
            if rank == 0:
                logger.info(f"Epoch {epoch + 1}/{args.epochs}")
            for batch_idx, batch in enumerate(dataloader):
                _mb_t_loop_start = time.perf_counter()
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

                # One batch == one super-long packed sequence.
                # Shape: [batch_size=1, seq_len=T_total]. It holds
                # the FFD-packed contents of one step (many docs
                # back-to-back); we slice it into ``n_chunks``
                # chunks of ``micro_batch_size`` tokens each for
                # the per-chunk forward.
                input_ids = batch["input_ids"].to(gpus[rank], non_blocking=True)
                labels = batch["labels"].to(gpus[rank], non_blocking=True)
                cu_seqlens = batch.get("cu_seqlens")
                if cu_seqlens is not None:
                    cu_seqlens = cu_seqlens.to(gpus[rank], non_blocking=True)
                # End of "data wait + H2D" phase.
                _mb_t_data_end = time.perf_counter()

                # Per-chunk forward, KDA state flows between chunks
                # (carried as a list of per-layer final-state tensors
                # from the previous chunk's forward). For chunk 0
                # the state is ``None`` — each layer starts from
                # zeros. We accumulate all chunk losses and call
                # backward ONCE at the end of the step so the
                # autograd graph connects across chunks via the
                # carried state (full BPTT — gradient of later
                # chunks flows back to earlier chunks' params).
                kda_states: list[torch.Tensor | None] | None = None
                chunk_losses: list[torch.Tensor] = []
                for ci in range(n_chunks):
                    s = ci * micro_batch_size
                    e = s + micro_batch_size
                    chunk_ids = input_ids[:, s:e]
                    chunk_labels = labels[:, s:e]
                    chunk_cu = (
                        _slice_cu_seqlens(cu_seqlens, s, e)
                        if cu_seqlens is not None else None
                    )

                    with torch.amp.autocast(
                        device_type="cuda",
                        dtype=autocast_dtype,
                        enabled=autocast_enabled,
                    ):
                        outputs = model(
                            chunk_ids, labels=chunk_labels,
                            cu_seqlens=chunk_cu,
                            kda_states=kda_states,
                        )
                        chunk_loss = outputs["loss"]
                    chunk_losses.append(chunk_loss)
                    # Carry forward: the new per-layer KDA states
                    # become the next chunk's initial states. The
                    # state tensors are part of the autograd graph
                    # (they depend on the chunk's input + the prior
                    # state), so we must NOT detach — the chain rule
                    # must flow back through them during backward.
                    kda_states = outputs["kda_states"]
                    del outputs

                # Total loss = mean of chunk losses (equivalent to
                # the per-step mean cross-entropy because each
                # chunk has the same number of tokens).
                loss = torch.stack(chunk_losses).mean()
                torch.cuda.synchronize(gpus[rank])
                _mb_t_fwd_end = time.perf_counter()
                # Per-step loss for logging (detach to avoid
                # retaining the graph after this line).
                step_loss = loss.detach().float().item()
                accumulated_loss += step_loss
                if rank == 0 and (
                    global_step < _DIAG_STEPS or not math.isfinite(step_loss)
                ):
                    logger.info(
                        f"  [diag] step={global_step} loss={step_loss:.6e}"
                    )

                # Single backward at end of step: chain rule flows
                # through the carried KDA states from chunk
                # n_chunks-1 back to chunk 0 (full BPTT).
                loss.backward()
                torch.cuda.synchronize(gpus[rank])
                _mb_t_bwd_end = time.perf_counter()

                # Per-step flush. The per-param post-accumulate-grad
                # hooks (see :func:`register_grad_offload_hooks`)
                # issued D2H transfers during ``backward()``; we
                # sync the current device's stream and fold the CPU
                # src tensors into the per-param accumulator. Same
                # semantics as the per-microbatch flush in the
                # legacy path; just consolidated to once-per-step
                # because we now do a single backward.
                flush_pending_grads(sync_device=gpus[rank])
                flush_manual_flush_params([muon_opt, adamw_opt])

                if getattr(args, "empty_cache_between_mb", True):
                    torch.cuda.empty_cache()

                _mb_t_sync_end = time.perf_counter()

                # Per-step timing log (chunked path: one row per
                # step, not per chunk). The breakdown is the same
                # data_ms / fwd_ms / bwd_ms / sync_ms columns.
                mb_timing = getattr(args, "mb_timing", 0)
                if rank == 0 and mb_timing > 0 and batch_idx < mb_timing:
                    n_real_docs = (
                        cu_seqlens.numel() - args.batch_size
                        if cu_seqlens is not None else -1
                    )
                    logger.info(
                        f"  [mb-time] step={global_step}"
                        f" n_chunks={n_chunks}/{n_chunks}"
                        f" data_ms={(_mb_t_data_end - _mb_t_loop_start) * 1000:6.1f}"
                        f" fwd_ms={(_mb_t_fwd_end - _mb_t_data_end) * 1000:6.1f}"
                        f" bwd_ms={(_mb_t_bwd_end - _mb_t_fwd_end) * 1000:6.1f}"
                        f" sync_ms={(_mb_t_sync_end - _mb_t_bwd_end) * 1000:6.1f}"
                        f" shape={tuple(input_ids.shape)}"
                        f" cu_seqlens=[{cu_seqlens.tolist() if cu_seqlens is not None else 'None'}]"
                        f" real_docs={n_real_docs}"
                    )

                del loss, chunk_losses, input_ids, labels
                # Free the chunk-local cu_seqlens tensors and the
                # final KDA state list; they go out of scope when
                # we move past the next loop iteration, but
                # releasing the references here makes the
                # allocator's job easier on the next step.
                del kda_states

                # Per-step end-of-step: optimizer step + diagnostics
                # + checkpoint. The old "if microbatch_in_cycle >=
                # gradient_accumulation_steps" gate collapses to
                # "every step is a step" — the chunked forward IS
                # the gradient accumulation (n_chunks = gas).
                flush_pending_grads(sync_device=gpus[rank])
                # Inf/nan check on the accumulated CPU grads.
                found_inf = False
                n_nan_accum_total = 0
                for opt_name, opt in (("muon", muon_opt), ("adamw", adamw_opt)):
                    for sid, s in opt.state.items():
                        accum = (
                            s.m if s.kind == "adamw"
                            else (s.accum if s.accum is not None else s.mom_buf)
                        )
                        # Cast to BF16 before ``isfinite`` — FP8
                        # storage (mxfp8 muon) has no direct
                        # ``isfinite`` in PyTorch 2.9.1. The bf16
                        # round-trip is lossless for the non-NaN/Inf
                        # values we're trying to detect.
                        n_nan = (~torch.isfinite(accum.to(torch.bfloat16))).sum().item()
                        n_nan_accum_total += n_nan
                        if n_nan > 0:
                            found_inf = True
                            break
                    if found_inf:
                        break
                if rank == 0 and global_step < 5:
                    print(f"  [CHECK-DEBUG] step={global_step} n_nan_accum_total={n_nan_accum_total} found_inf={found_inf}")
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
                    cur_muon_lr = wsd_lr(
                        global_step, peak_muon_lr,
                        warmup_steps, decay_steps,
                        args.max_steps,
                    )
                    cur_adamw_lr = wsd_lr(
                        global_step, peak_adamw_lr,
                        warmup_steps, decay_steps,
                        args.max_steps,
                    )
                    muon_opt.lr = cur_muon_lr
                    adamw_opt.lr = cur_adamw_lr
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
                    if getattr(args, "ffn_nvfp4", False):
                        from src.models.ops.nvfp4_linear import repack_nvfp4_weights
                        n_repacked = repack_nvfp4_weights(model)
                        if global_step < _DIAG_STEPS and rank == 0:
                            logger.info(
                                "[nvfp4] step=%d repacked %d NVFP4 weight(s)",
                                global_step, n_repacked,
                            )
                else:
                    total_norm = float("nan")
                scaler.update()
                torch.cuda.synchronize(gpus[rank])

                global_step += 1
                avg_loss = accumulated_loss  # one loss value per step now
                accumulated_loss = 0.0

                if rank == 0 and global_step % args.log_interval == 0:
                    grad_norm_str = (
                        f"{total_norm:.2f}" if total_norm == total_norm
                        else "nan"
                    )
                    lr_str = (
                        f"lr_muon={muon_opt.lr:.2e}"
                        f" lr_adamw={adamw_opt.lr:.2e}"
                    )
                    logger.info(
                        f"Step {global_step}/{args.max_steps} | "
                        f"Loss: {avg_loss:.4f} | "
                        f"grad_norm: {grad_norm_str} | "
                        f"{lr_str}"
                    )
                    for d in gpus:
                        free, total = torch.cuda.mem_get_info(d)
                        used = total - free
                        logger.info(
                            f"  device {d} VRAM: {used / 1024**3:.2f} /"
                            f" {total / 1024**3:.2f} GB"
                        )

                # Periodic checkpoint.
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
                            keep_last_n=args.checkpoint_keep_last_n,
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
