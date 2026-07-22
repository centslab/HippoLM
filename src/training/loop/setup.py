"""Phase 1 of the training worker: setup.

:func:`_setup_worker` builds everything the training loop
needs from the rank's inputs:

  * Distributed process group (NCCL or gloo, depending on
    ``args.tp_sim``).
  * Per-worker logging (both stderr and a per-rank file under
    ``<run_dir>/logs/``).
  * :class:`HippoConfig` from the yml-driven ``args``.
  * Data path: dummy / queue-iterator / MultiSource /
    StreamingDataset depending on the config.
  * :class:`TPHippoModel` (the per-rank model fragment).
  * The disabled :class:`torch.amp.GradScaler` (FP16 / BF16
    runs refuse the scaler's unscale in this torch version; the
    forward runs in the configured activation dtype plain).
  * The Muon + AdamW optimizer pair via
    :func:`build_param_groups`.
  * The per-param grad-offload hooks via
    :func:`register_grad_offload_hooks`, plus the manual-flush
    marker for the tied-embed param whose grad arrives via a
    custom autograd Function.

Returns a plain :class:`dict` that :func:`_run_training_loop`
(and :func:`_teardown_worker`) consume. The ``rank``,
``args``, ``gpus``, ``port`` and ``run_dir`` keys are
preserved across the three phases; everything else is
plumbing the loop needs and teardown needs to clean up.
"""
from __future__ import annotations

import logging
import os
import queue
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from src.models import HippoConfig
from src.models.tp_model._primitives import init_tp
from src.models.tp_model import TPHippoModel
from src.training.data import (
    MultiSourceStreamingDataset,
    PrefetchBatcher,
    QueueIterator,
    StreamingDataset,
    dummy_dataloader,
)
from src.training.data.cache import purge_stale_cache_if_no_hit
from src.training.param_offload import (
    _start_cpu_add_worker,
    build_param_groups,
    register_grad_offload_hooks,
)
from src.training.precision_config import PrecisionConfig
from src.training.tokenizer import load_tokenizer

log = logging.getLogger(__name__)


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
        # Scheme-driven precision (2026-07-21, 5-scheme spec).
        # Each module gets one of "w16a16", "w8a16", "w8a8",
        # "w4a16", "w4a8". The legacy boolean flags
        # (``ffn_nvfp4``, ``ffn_nvfp4_marlin``,
        # ``ffn_nvfp4_no_bf16_master``, ``kda_fp8``, ``kda_mxfp8``)
        # were removed from the dataclass on 2026-07-21; the
        # scheme is the single source of truth.
        embedding_precision=getattr(args, "embedding_precision", "w16a16"),
        attention_precision=getattr(args, "attention_precision", "w16a16"),
        ffn_precision=getattr(args, "ffn_precision", "w16a16"),
        # Producer-side FP8 fusions (2026-07-22, full-FP8
        # productionization). Both default to False; yml sets
        # ``true`` in ``configs/test/fp8_full.yml`` for the
        # smoke-test scenario. End-to-end fwd+bwd STE-bwd probe
        # showed +0.27% loss delta at 300 steps vs BF16, well
        # inside the FP8 noise floor.
        fp8_residual=getattr(args, "fp8_residual", False),
        fp8_silu_mul=getattr(args, "fp8_silu_mul", False),
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
        # NVFP4 mode-3 modules register their weights as
        # ``register_buffer`` (FP4 packed bytes), NOT as
        # ``nn.Parameter`` — see :class:`NVFP4Linear` /
        # :class:`NVFP4ColumnParallelLinear`. So
        # ``trainable_parameters()`` (which iterates
        # ``nn.Parameter``) misses the SwiGLU weights entirely,
        # undercounting the model by ~604M elements at
        # base.yml (32 layers × 18.87M / layer for gate_up +
        # down). Those weights ARE trainable (the optimizer
        # updates them via the side-channel material/commit
        # API), so the log must include their element count to
        # match the per-optimizer Muon+AdamW param totals that
        # follow. We walk ``model.modules()`` and add the
        # ``out_features × in_features[_per_partition]`` for
        # every ``no_bf16_master`` NVFP4 module. Element
        # count, not byte count — each FP4 element is one
        # weight value (the 2-bits-per-element packing is a
        # storage detail, not a "fewer params" detail).
        nvfp4_params = 0
        for m in model.modules():
            if not getattr(m, "no_bf16_master", False):
                continue
            cols = getattr(
                m, "in_features_per_partition", m.in_features,
            )
            nvfp4_params += m.out_features * cols
        total_trainable = trainable + nvfp4_params
        logger.info(
            f"TP model built. Per-device trainable params:"
            f" {total_trainable:,}"
            f" (BF16 leaf={trainable:,}, NVFP4 FP4={nvfp4_params:,})"
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
        adamw_beta1=args.adamw_beta1,
        adamw_beta2=args.adamw_beta2,
        adamw_eps=args.adamw_eps,
        muon_momentum=args.muon_momentum,
        muon_weight_decay=args.muon_weight_decay,
        muon_exp_avg_storage=args.muon_exp_avg_storage,
        muon_block_size=args.muon_block_size,
        precision=precision,
    )
    # Install the per-param post-accumulate-grad hooks so each
    # param's grad is DMA'd to its CPU accumulator as soon as
    # autograd computes it (during ``backward()``), not after.
    # This keeps the peak GPU grad memory to "at most one
    # param's grad" rather than "the sum of all grads". The
    # training loop calls :func:`accumulate_grads_to_cpu` once per
    # microbatch to sync CUDA and apply the CPU adds — it
    # drains both the per-param hook queue (normal params) and
    # the NVFP4 mode-3 stash (``module._latest_grad_w``) in one
    # pass. See :func:`register_grad_offload_hooks` for the design.
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
    # Install the per-param streaming hooks. The per-mb
    # ``accumulate_grads_to_cpu`` + ``flush_manual_flush_params`` path
    # in :func:`_run_training_loop` syncs + CPU-adds the queued D2Hs.
    register_grad_offload_hooks(
        [muon_opt, adamw_opt],
        manual_flush_params=manual_flush or None,
    )
    # Start the async CPU-add worker (Priority 2 overlap).
    # After this call, the per-param streaming hook and the
    # manual-flush / NVFP4 stash paths push ``(event, target,
    # src)`` entries to a background queue instead of doing a
    # synchronous ``cuda.synchronize`` + CPU-add on the main
    # thread. The next chunk's fwd can launch while the worker
    # drains chunk i's adds — see
    # :func:`_drain_cpu_add_queue` for the end-of-step join.
    # Stopped in :func:`_teardown_worker`.
    _start_cpu_add_worker()
    # NVFP4 mode-3 entries have ``s.param is None`` (the
    # weight lives on the module as FP4 packed buffers, not as a
    # leaf Parameter). Fall back to ``s.nvfp4_n`` (cached at
    # :meth:`register_nvfp4_module` time) for those.
    n_muon = sum(
        s.param.numel() if s.param is not None else s.nvfp4_n
        for s in muon_opt.state.values()
    )
    n_adamw = sum(
        s.param.numel() if s.param is not None else s.nvfp4_n
        for s in adamw_opt.state.values()
    )
    # Per-param byte breakdown, kept in named variables so the log
    # line below (and any future debugger) can see the components
    # separately.
    #
    # Post-2026-07-15 layout (mu=1 merged-accumulator trick was
    # deleted; the grad accumulator and the optimizer's momentum
    # state are now separate buffers):
    #
    #   AdamW per param: ``s.grad`` (per-step accumulator,
    #                     dtype = adamw_m),
    #                     ``s.exp_avg`` (β1 EMA, dtype = adamw_m),
    #                     ``s.exp_avg_sq`` (β2 EMA, dtype = adamw_v).
    #   Muon per param:  ``s.grad`` (per-step accumulator),
    #                     ``s.exp_avg`` (SGD momentum).
    #                     Both share ``precision.muon_momentum``
    #                     dtype (bf16 / fp16 / fp32).
    #
    # Quantized storage formats (int8 per-row BF16 scale, mxfp8
    # per-block E8M0 scale) were removed on 2026-07-12 after
    # long-training runs showed quantization-error accumulation
    # destabilizing optimization. See
    # ``docs/optimizer_layout.md`` for the post-removal layout.
    adamw_m_dtype_t = precision.adamw_m.dtype.to_torch()
    adamw_v_dtype_t = precision.adamw_v.dtype.to_torch()
    adamw_m_bytes = n_adamw * adamw_m_dtype_t.itemsize
    adamw_exp_avg_bytes = n_adamw * adamw_m_dtype_t.itemsize
    adamw_v_bytes = n_adamw * adamw_v_dtype_t.itemsize
    muon_states = list(muon_opt.state.values())
    if muon_states:
        muon_storage_dtype = muon_states[0].grad.dtype
        muon_bytes_per_elt = muon_storage_dtype.itemsize * 2  # grad + exp_avg
    else:
        muon_storage_dtype = torch.bfloat16
        muon_bytes_per_elt = 4  # BF16 × 2 buffers
    muon_bytes = n_muon * muon_bytes_per_elt
    adamw_bytes = adamw_m_bytes + adamw_exp_avg_bytes + adamw_v_bytes
    muon_label = (
        f"{str(muon_storage_dtype).replace('torch.', '')}_m+ema"
    )
    accum_note = (
        "grad accumulator + EMA moments (post-2026-07-15 "
        "explicit-accumulator layout)"
    )
    logger.info(
        f"Device {gpus[rank]}: Muon params={n_muon:,}"
        f" ({muon_bytes / 1024**3:.2f} GB total:"
        f" {muon_label}={muon_bytes / 1024**3:.3f} GB),"
        f" AdamW params={n_adamw:,}"
        f" ({adamw_bytes / 1024**3:.2f} GB total:"
        f" bf16_grad={adamw_m_bytes / 1024**3:.3f} GB"
        f" + bf16_exp_avg={adamw_exp_avg_bytes / 1024**3:.3f} GB"
        f" + bf16_exp_avg_sq={adamw_v_bytes / 1024**3:.3f} GB)"
        f" [{accum_note}]."
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