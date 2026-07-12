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
    build_param_groups,
    register_grad_offload_hooks,
    setup_per_layer_gpu_accum,
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
        ffn_nvfp4=getattr(args, "ffn_nvfp4", False),
        ffn_nvfp4_marlin=getattr(args, "ffn_nvfp4_marlin", False),
        ffn_nvfp4_no_bf16_master=getattr(args, "ffn_nvfp4_no_bf16_master", False),
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
    # Offload-strategy dispatch. ``cpu_add`` (default, prod-proven)
    # installs the per-param streaming hooks; the per-mb
    # ``accumulate_grads_to_cpu`` + ``flush_manual_flush_params`` path
    # in :func:`_run_training_loop` syncs + CPU-adds the queued D2Hs.
    # ``per_layer_gpu`` (opt-in v4) replaces this with a shared
    # GPU accumulator per ``TPHippoLayer`` + per-layer async D2H to
    # a CPU pinned slot + worker-thread CPU add — amortizes the
    # D2H into the per-mb bwd tail (33-44% step-time win at the test
    # scale; see :mod:`.per_layer_gpu_accum` for the A/B numbers and
    # VRAM analysis). v4 installs its OWN per-param
    # ``register_post_accumulate_grad_hook`` for manual-flush-style
    # params (anything not under any ``TPHippoLayer`` — tied embed,
    # top-level norm, attn_res, lm_head), so we don't need the
    # ``manual_flush_params=`` list under v4.
    offload_strategy = getattr(args, "offload_strategy", "cpu_add")
    if offload_strategy == "per_layer_gpu":
        if rank == 0:
            logger.info(
                f"Offload strategy: per_layer_gpu (v4 — see "
                f"src.training.param_offload.per_layer_gpu_accum)"
            )
        setup_per_layer_gpu_accum(model, [muon_opt, adamw_opt])
    else:
        register_grad_offload_hooks(
            [muon_opt, adamw_opt],
            manual_flush_params=manual_flush or None,
        )
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
    n_muon_rows = sum(s.shape[0] for s in muon_opt.state.values())
    # Per-param byte breakdown, kept in named variables so the log
    # line below (and any future debugger) can see the components
    # separately.
    #
    # Accumulator layout: AdamW uses ``s.m`` (BF16) as both the
    # grad accumulator and the first moment — no separate
    # ``accum`` buffer. Muon uses the same merged design for fp*
    # storage (``s.mom_buf`` doubles as the accumulator). For
    # int8 muon, there is a separate bf16 ``s.accum`` buffer that
    # holds the per-mb grad sum cheaply (the int8
    # dequant-add-requant per-mb cycle was the CPU bottleneck at
    # 8+ seconds per microbatch — see ``docs/optimizer_layout.md``
    # for the design). For mxfp8 muon, there is NO separate
    # ``accum`` buffer: the per-mb accumulation flows through a
    # fused C++ kernel (:func:`fused_mxfp8_dequant_add_requant`)
    # that does dequant + add + requant in one pass directly into
    # ``s.mom_buf``. This is the mxfp8 design's whole point — to
    # save the 2 bytes/elt that a separate accumulator would have
    # cost (see test/test_mxfp8_no_accum.py for the regression).
    # Per-mb CPU sync is slower for mxfp8 (~1s/mb at the smoke
    # scale) than for int8 with bf16 accum (~50ms), but it's still
    # inside the per-mb sync budget because the D2H DMA dominates
    # the wall-clock anyway.
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
            (s.param.numel() if s.param is not None else s.nvfp4_n)
            for s in muon_states if s.accum is not None
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
    # Per-param log line describes the per-mb CPU sync cost:
    # int8 uses a separate bf16 accum (~50ms/mb); mxfp8 uses
    # the fused C++ kernel on mom_buf directly (~1s/mb at smoke
    # scale). Branches on mom_buf.dtype so the message reflects
    # the actual configured storage, not just "any quantized
    # muon".
    if muon_states and muon_mom_dtype == torch.float8_e4m3fn:
        accum_note = "mxfp8: fused C++ kernel per-mb (~1s/mb at smoke scale)"
    elif muon_accum > 0:
        accum_note = "int8: cheap CPU bf16 add per mb (~50ms)"
    else:
        accum_note = "merged-accumulator (mom_buf doubles as accumulator)"
    logger.info(
        f"Device {gpus[rank]}: Muon params={n_muon:,}"
        f" ({muon_bytes / 1024**3:.2f} GB total:"
        f" {muon_mom_label}={muon_mom / 1024**3:.3f} GB"
        f"{muon_scale_str}{muon_accum_str}),"
        f" AdamW params={n_adamw:,}"
        f" ({adamw_bytes / 1024**3:.2f} GB total:"
        f" bf16_m={adamw_m / 1024**3:.3f} GB"
        f" + bf16_v={adamw_v / 1024**3:.3f} GB)"
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