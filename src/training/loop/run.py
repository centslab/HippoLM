"""Phase 2 of the training worker: the per-step loop.

:func:`_run_training_loop` is the heart of the worker. Each
step:

  1. Pull one batch from the data path (the prefetcher's
     queue-iterator on rank > 0; rank 0 feeds the
     :class:`PrefetchBatcher` itself).
  2. Slice the super-long packed sequence into ``n_chunks``
     chunks of ``micro_batch_size`` tokens each. Per chunk:
     forward the model, accumulate the token-weighted loss,
     backward, fold the per-mb grads into the CPU accumulator
     via :func:`accumulate_grads_to_cpu`, and carry the KDA
     state detached (truncated BPTT).
  3. Step the optimizers (with the WSD schedule applied to
     ``muon_opt.lr`` / ``adamw_opt.lr``), with a
     :func:`_compute_and_clip_grad_norm` clip pass.
  4. Log / checkpoint on the canonical intervals.

The ``finally`` clause at the bottom delegates to
:func:`_teardown_worker` so every exit path (natural end of
training, exception in the loop body) tears down the
distributed process group and the prefetcher thread.
"""
from __future__ import annotations

import logging
import math
import time
from typing import Any, Dict

import torch

from src.training import save_checkpoint
from src.training.diagnostics import log_post_opt_diag, log_pre_step_diag
from src.training.param_offload import (
    accumulate_grads_to_cpu,
    flush_manual_flush_params,
    flush_per_layer_gpu_accum,
    transfer_per_layer_gpu_accum_join,
    zero_cpu_grad_accum,
)
from .grad_norm import _compute_and_clip_grad_norm
from .support import _slice_cu_seqlens, wsd_lr
from .teardown import _teardown_worker

log = logging.getLogger(__name__)


# Number of early training steps for which to emit the detailed
# pre-/post-step diagnostic (acc_max, m_max, pmax, total_norm,
# per-optimizer finite check). Gated at the call site to keep
# steady-state logging overhead at zero.
_DIAG_STEPS = 3


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
    # Offload strategy resolved once per worker. The per-mb flush
    # dispatch below and the per-step worker join both branch on
    # this; ``cpu_add`` is the legacy prod path, ``per_layer_gpu``
    # is the opt-in v4 alternative (see
    # :mod:`src.training.param_offload.per_layer_gpu_accum`).
    _offload_strategy = getattr(args, "offload_strategy", "cpu_add")
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
                # zeros.
                #
                # The state is DETACHED between chunks: chunk i+1's
                # forward receives the numerical final-state of chunk i
                # but does NOT extend the autograd graph back through it.
                # This is **truncated BPTT** — the standard pattern for
                # recurrent nets when full BPTT activation memory blows
                # the VRAM budget. Gradients still accumulate across
                # chunks at the optimizer state (the per-param CPU
                # accumulator in :func:`build_param_groups` folds each
                # chunk's bwd into the running total). The fix was
                # forced by the b173879 OOM: at 16 chunks of T=16384,
                # full BPTT retains all N chunks' saved-tensor activations
                # simultaneously (~93 GB); per-chunk bwd+detach bounds
                # the peak to one chunk (~12.6 GB at production). See
                # ``docs/gradient_checkpointing.md`` for the per-component
                # breakdown.
                #
                # After chunk i's bwd completes, the freed memory is
                # reusable for chunk i+1's fwd — the autograd graph is
                # severed.
                # Token-weighted per-chunk loss normalization.
                # ------------------------------------------------
                # The model returns each chunk's MEAN cross-entropy
                # over that chunk's valid (non-ignored) tokens. Real
                # FFD-packed data fills docs from position 0 and pads
                # the tail, so the trailing chunks are pure padding
                # (labels all -100) and the model returns 0 for them.
                # Dividing every chunk's mean by ``n_chunks`` (the old
                # code) assumed each chunk held an equal 1/n_chunks
                # share of the valid tokens — true only for the fully
                # packed dummy-data smoke test. On real data the empty
                # trailing chunks then diluted the step loss toward
                # zero (e.g. 5 real chunks of ~12.7 out of 16 reported
                # ~3.98 instead of ~12.7) and under-scaled the real
                # chunks' gradients.
                #
                # Fix: weight each chunk by its share of the step's
                # valid tokens, so ``step_loss`` equals the token-
                # weighted mean CE over the whole packed sequence and
                # the accumulated gradient matches a single full-
                # sequence backward. This reduces to the old
                # ``/n_chunks`` exactly when every chunk is equally
                # full. The model shifts labels by one (predict t+1
                # from t), so the per-chunk valid count is measured on
                # ``labels[:, s+1:e]`` — identical to the count the
                # FusedLinearCE kernel divides its mean by.
                chunk_valid = [
                    int(
                        (labels[:, ci * micro_batch_size + 1:
                                (ci + 1) * micro_batch_size] != -100)
                        .sum().item()
                    )
                    for ci in range(n_chunks)
                ]
                total_valid = max(sum(chunk_valid), 1)
                # Stop after the last chunk that has any valid token:
                # trailing all-pad chunks carry no state into any real
                # chunk, so running their (32-layer) forward would be
                # pure waste. Interior empty chunks (should not occur
                # with the trailing-pad FFD packer) still run forward
                # for state carry but contribute no loss/gradient.
                last_real = max(
                    (ci for ci, nv in enumerate(chunk_valid) if nv > 0),
                    default=-1,
                )

                kda_states: list[torch.Tensor | None] | None = None
                step_loss_chunk_sum = 0.0
                for ci in range(last_real + 1):
                    s = ci * micro_batch_size
                    e = s + micro_batch_size
                    chunk_ids = input_ids[:, s:e]
                    chunk_labels = labels[:, s:e]
                    chunk_cu = (
                        _slice_cu_seqlens(cu_seqlens, s, e)
                        if cu_seqlens is not None else None
                    )
                    n_valid_ci = chunk_valid[ci]

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
                        # ``outputs["loss"]`` is the mean over this
                        # chunk's ``n_valid_ci`` tokens, so
                        # ``loss * n_valid_ci`` is the chunk's summed
                        # CE and dividing by ``total_valid`` gives the
                        # chunk's contribution to the step's global
                        # token-weighted mean.
                        chunk_loss = (
                            outputs["loss"] * (n_valid_ci / total_valid)
                        )
                    # Carry forward, then detach. Detach preserves the
                    # numerical state (chunk i+1 sees the same h-vector
                    # values) but breaks the autograd graph chain so
                    # the backward of chunk i+1 does NOT traverse back
                    # into chunk i.
                    kda_states = [
                        st.detach() if st is not None else None
                        for st in outputs["kda_states"]
                    ]
                    del outputs
                    if n_valid_ci == 0:
                        # All-pad interior chunk: the forward already
                        # carried the KDA state; there is no loss to
                        # backward.
                        if getattr(args, "empty_cache_between_mb", True):
                            torch.cuda.empty_cache()
                        continue
                    # Accumulate the *detached* loss for logging.
                    # ``step_loss_chunk_sum`` is a plain float; safe to
                    # hold without retaining the graph.
                    step_loss_chunk_sum += chunk_loss.item()
                    # Per-chunk immediate backward. Per-param
                    # accumulate-grad hooks fire as the autograd graph
                    # walks for non-NVFP4 params (their grads land in
                    # ``_pending_grads``); ``accumulate_grads_to_cpu``
                    # drains that queue AND consumes the NVFP4 mode-3
                    # stash (``module._latest_grad_w``) in one pass.
                    # ``flush_manual_flush_params`` then handles the
                    # tied-embed param whose hook was deliberately
                    # skipped (idempotent: grad is already None after
                    # accumulate_grads_to_cpu drained it).
                    chunk_loss.backward()
                    if _offload_strategy == "per_layer_gpu":
                        # v4: per-layer hooks already issued async
                        # D2Hs and the worker drains them in parallel
                        # with the main thread's continued bwd. The
                        # per-mb flush is a no-op on the main thread;
                        # we still call ``accumulate_grads_to_cpu`` to
                        # drain NVFP4 mode-3 stashes (``module._latest_
                        # grad_w`` — idempotent for BF16 params whose
                        # ``.grad`` was already cleared by the v4
                        # hooks). No ``flush_manual_flush_params``
                        # needed: v4 installs its own per-param hook
                        # for the manual-flush-style params (anything
                        # not under any ``TPHippoLayer``).
                        flush_per_layer_gpu_accum()
                        accumulate_grads_to_cpu(
                            [muon_opt, adamw_opt], sync_device=gpus[rank],
                        )
                    else:
                        accumulate_grads_to_cpu(
                            [muon_opt, adamw_opt], sync_device=gpus[rank],
                        )
                        flush_manual_flush_params([muon_opt, adamw_opt])
                    if getattr(args, "empty_cache_between_mb", True):
                        torch.cuda.empty_cache()

                torch.cuda.synchronize(gpus[rank])
                _mb_t_fwd_end = time.perf_counter()
                # Per-step loss for logging (already detached; sum of
                # per-chunk losses divided by n_chunks = mean per-token
                # loss across the full step).
                step_loss = step_loss_chunk_sum
                accumulated_loss += step_loss
                if rank == 0 and (
                    global_step < _DIAG_STEPS or not math.isfinite(step_loss)
                ):
                    logger.info(
                        f"  [diag] step={global_step} loss={step_loss:.6e}"
                    )

                # With per-chunk backward (above) the fwd and bwd are
                # interleaved; the per-step timing breakdown collapses
                # them into a single "compute" interval measured by
                # ``_mb_t_fwd_end - _mb_t_data_end``. The legacy
                # ``fwd_ms / bwd_ms / sync_ms`` columns become
                # ``compute_ms = fwd+bwd_ms / cache_ms = empty_cache+flush``.
                _mb_t_bwd_end = _mb_t_fwd_end
                _mb_t_sync_end = _mb_t_fwd_end

                # Per-step timing log (chunked path: one row per
                # step, not per chunk). The breakdown is the same
                # data_ms / fwd_ms / bwd_ms / sync_ms columns but
                # with fwd+bwd fused because the per-chunk loop
                # interleaves them.
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
                        f" compute_ms={(_mb_t_fwd_end - _mb_t_data_end) * 1000:6.1f}"
                        f" shape={tuple(input_ids.shape)}"
                        f" cu_seqlens=[{cu_seqlens.tolist() if cu_seqlens is not None else 'None'}]"
                        f" real_docs={n_real_docs}"
                    )

                del input_ids, labels
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
                # Drain any remaining NVFP4 stashes (idempotent if
                # already drained per-chunk).
                accumulate_grads_to_cpu(
                    [muon_opt, adamw_opt], sync_device=gpus[rank],
                )
                # v4 step-end barrier: wait for the worker thread
                # to drain the per-layer + per-mf D2H + CPU-add
                # queue. Also calls ``torch.cuda.empty_cache()`` to
                # release cached ``.grad`` blocks back to the OS so
                # VRAM stays at "model + gpu_buf + mf_buf" between
                # steps. No-op under the cpu_add strategy.
                if _offload_strategy == "per_layer_gpu":
                    transfer_per_layer_gpu_accum_join()
                # Inf/nan check on the accumulated CPU grads.
                found_inf = False
                n_nan_accum_total = 0
                for opt_name, opt in (("muon", muon_opt), ("adamw", adamw_opt)):
                    for sid, s in opt.state.items():
                        accum = (
                            s.m if s.kind == "adamw"
                            else s.mom_buf
                        )
                        n_nan = (~torch.isfinite(accum)).sum().item()
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