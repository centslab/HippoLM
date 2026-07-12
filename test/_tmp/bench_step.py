"""Baseline timing benchmark for the per-step training loop.

Mimics production shape (chunked FFD forward+backward with detached
KDA state) but stays inside the 5060 Ti 16G ceiling. Runs N steps
and prints per-step wall-clock + a breakdown.

Usage:
    python test/_tmp/bench_step.py [--steps N] [--seq-len L]
                                    [--micro-batch-size M]
                                    [--num-layers L] [--num-blocks B]
                                    [--empty-cache {0,1}]
                                    [--tp-sim 0|1]

All flag-less defaults mirror a representative-but-tiny training
run; the script is meant to be tweaked by hand to compare loop
optimizations side-by-side.

The script writes a single summary line to stdout at the end
suitable for grepping / parsing::

    [BENCH] steps=10 mean_ms=X median_ms=Y min_ms=Z max_ms=W
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

# Repo-root import path.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch  # noqa: E402

from src.models import HippoConfig  # noqa: E402
from src.models.tp_model import TPHippoModel  # noqa: E402
from src.models.tp_model._primitives import init_tp  # noqa: E402
from src.training.loop.support import _slice_cu_seqlens  # noqa: E402
from src.training.param_offload import (  # noqa: E402
    accumulate_grads_to_cpu, build_param_groups, flush_manual_flush_params,
    register_grad_offload_hooks, zero_cpu_grad_accum,
)
from src.training.precision_config import PrecisionConfig  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--seq-len", type=int, default=8192)
    p.add_argument("--micro-batch-size", type=int, default=1024)
    p.add_argument("--num-layers", type=int, default=4)
    p.add_argument("--num-blocks", type=int, default=2)
    p.add_argument("--num-heads", type=int, default=4)
    p.add_argument("--head-dim", type=int, default=32)
    p.add_argument("--hidden-size", type=int, default=128)
    p.add_argument("--intermediate-size", type=int, default=256)
    p.add_argument("--vocab-size", type=int, default=1024)
    p.add_argument("--empty-cache", type=int, default=1,
                   help="1 = torch.cuda.empty_cache between chunks "
                        "(matches base.yml empty_cache_between_mb=true)")
    p.add_argument("--tp-sim", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    backend = "gloo" if args.tp_sim else "nccl"
    # Single-process benchmark: world_size=1 unless tp_sim forces N procs.
    if args.tp_sim:
        # Bring up a 2-rank gloo group on a single GPU.
        import torch.distributed as dist
        import os
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = "29501"
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "2"
        os.environ["LOCAL_RANK"] = "0"
        torch.cuda.set_device(0)
        dist.init_process_group(
            backend="gloo",
            init_method="env://",
            rank=0,
            world_size=2,
        )
        init_tp(world_size=2, devices=[0, 0], backend="gloo")
        rank, world, device = 0, 2, torch.device("cuda:0")
    else:
        torch.cuda.set_device(0)
        init_tp(world_size=1, devices=[0], backend="nccl")
        rank, world, device = 0, 1, torch.device("cuda:0")

    config = HippoConfig(
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        tie_word_embeddings=True,
        use_bias=False,
        num_heads=args.num_heads,
        head_dim=args.head_dim,
        expand_v=1.0,
        kda_mode="chunk",
        use_short_conv=True,
        allow_neg_eigval=False,
        safe_gate=True,
        lower_bound=-5.0,
        conv_size=4,
        conv_bias=False,
        num_layers=args.num_layers,
        num_blocks=args.num_blocks,
        intermediate_size=args.intermediate_size,
        rms_norm_eps=1e-6,
        pack_chunk_size=64,
        pack_buffer_size=2048,
        kda_skip_aqk_akk_saved=False,
        ffn_nvfp4=True,
        ffn_nvfp4_marlin=True,
        ffn_nvfp4_no_bf16_master=True,
    )

    precision = PrecisionConfig.from_dict(None)
    model = TPHippoModel(config, devices=[0], dtype=precision.model_weights.dtype.to_torch())
    if world > 1:
        import torch.distributed as dist
        dist.barrier()
    model.sync_replicated_from(0)

    muon_opt, adamw_opt = build_param_groups(
        model, device=0,
        lr_muon=0.02, lr_adamw=0.004,
        weight_decay=0.0, adamw_beta1=0.9, adamw_beta2=0.95,
        adamw_eps=1e-8, muon_momentum=0.95, muon_weight_decay=0.0,
        precision=precision,
    )
    register_grad_offload_hooks([muon_opt, adamw_opt])

    autocast_dtype = precision.autocast_dtype
    autocast_enabled = precision.autocast_enabled

    micro_batch_size = args.micro_batch_size
    seq_len = args.seq_len
    n_chunks = seq_len // micro_batch_size
    assert n_chunks * micro_batch_size == seq_len, (
        f"seq_len={seq_len} must be multiple of "
        f"micro_batch_size={micro_batch_size}"
    )

    # Pre-build N+1 random batches so the data path is not the
    # bottleneck we are measuring.
    batches = []
    for _ in range(args.steps + 2):
        input_ids = torch.randint(
            0, args.vocab_size, (1, seq_len), device=device,
        )
        labels = input_ids.clone()
        cu_seqlens = None  # KDA mode='chunk' tolerates None
        batches.append((input_ids, labels, cu_seqlens))

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    # ---- Warm-up step (not counted) ----
    input_ids, labels, cu_seqlens = batches[0]
    kda_states = None
    for ci in range(n_chunks):
        s = ci * micro_batch_size
        e = s + micro_batch_size
        with torch.amp.autocast(
            device_type="cuda", dtype=autocast_dtype,
            enabled=autocast_enabled,
        ):
            outputs = model(input_ids[:, s:e], labels=labels[:, s:e],
                            cu_seqlens=None, kda_states=kda_states)
            cl = outputs["loss"]
        kda_states = [
            st.detach() if st is not None else None
            for st in outputs["kda_states"]
        ]
        del outputs
        cl.backward()
        accumulate_grads_to_cpu([muon_opt, adamw_opt], sync_device=0)
        flush_manual_flush_params([muon_opt, adamw_opt])
        if args.empty_cache:
            torch.cuda.empty_cache()
    torch.cuda.synchronize()
    accumulate_grads_to_cpu([muon_opt, adamw_opt], sync_device=0)
    muon_opt.step()
    adamw_opt.step()
    zero_cpu_grad_accum([muon_opt, adamw_opt])

    # ---- Measured steps ----
    step_ms: list[float] = []
    for step in range(args.steps):
        input_ids, labels, cu_seqlens = batches[step + 1]
        t0 = time.perf_counter()

        kda_states = None
        step_loss_sum = 0.0
        for ci in range(n_chunks):
            s = ci * micro_batch_size
            e = s + micro_batch_size
            with torch.amp.autocast(
                device_type="cuda", dtype=autocast_dtype,
                enabled=autocast_enabled,
            ):
                outputs = model(input_ids[:, s:e], labels=labels[:, s:e],
                                cu_seqlens=None, kda_states=kda_states)
                cl = outputs["loss"]
            kda_states = [
                st.detach() if st is not None else None
                for st in outputs["kda_states"]
            ]
            del outputs
            cl.backward()
            accumulate_grads_to_cpu([muon_opt, adamw_opt], sync_device=0)
            flush_manual_flush_params([muon_opt, adamw_opt])
            if args.empty_cache:
                torch.cuda.empty_cache()
            step_loss_sum += cl.item()

        torch.cuda.synchronize()
        accumulate_grads_to_cpu([muon_opt, adamw_opt], sync_device=0)
        muon_opt.step()
        adamw_opt.step()
        zero_cpu_grad_accum([muon_opt, adamw_opt])
        torch.cuda.synchronize()

        t1 = time.perf_counter()
        step_ms.append((t1 - t0) * 1000.0)

    mean = statistics.mean(step_ms)
    median = statistics.median(step_ms)
    mn, mx = min(step_ms), max(step_ms)
    hwm_mb = torch.cuda.max_memory_allocated() / 1024**2
    print(
        f"[BENCH] steps={args.steps} n_chunks={n_chunks}"
        f" empty_cache={args.empty_cache} tp_sim={args.tp_sim}"
        f" mean_ms={mean:.1f} median_ms={median:.1f}"
        f" min_ms={mn:.1f} max_ms={mx:.1f}"
        f" hwm_mb={hwm_mb:.0f}"
    )
    print(f"[BENCH] per_step_ms={['%.1f' % x for x in step_ms]}")

    if world > 1:
        import torch.distributed as dist
        dist.destroy_process_group()


if __name__ == "__main__":
    main()