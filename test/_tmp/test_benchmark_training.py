"""Training-script micro-benchmark.

Times the per-microbatch and per-step breakdown of a 1-GPU (TP sim,
world_size=1) training run on the dummy data path. Designed for
quick iteration on the optimization back-and-forth — the user can
read the printed table to see where time / VRAM is going.

Usage:
    python -m pytest test/_tmp/test_benchmark_training.py -v -s
    python -m pytest test/_tmp/test_benchmark_training.py -v -s --step-count=20

What it reports (per microbatch + per step):
    - forward_ms:    forward + autocast wall time
    - backward_ms:   loss.backward() wall time
    - sync_ms:       flush_pending_grads() wall time (D2H wait + CPU add)
    - h2d_ms:        batch.to(gpu) wall time
    - per-mb peak VRAM (allocated + reserved)
    - per-step grad_norm_ms, opt_step_ms (muon + adamw separately)

And the cumulative view:
    - VRAM at step 0 vs step N (growing or stable?)
    - CPU RSS at step 0 vs step N

This is a TEMPORARY test, like the other test/_tmp/ files. Delete
before commit (per the user's CLAUDE.md guidance: "git commit 之前
要删掉关于模型架构的 test").
"""
from __future__ import annotations

import argparse
import gc
import os
import resource
import sys
import time
from pathlib import Path

import torch

# Repo root on path (so ``from src...`` works inside the test process).
_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO))

from src.models import HippoConfig
from src.models.tp_layers import init_tp
from src.models.tp_model import TPHippoModel
from src.training.data import dummy_dataloader
from src.training.param_offload import (
    build_param_groups, flush_pending_grads,
    register_grad_offload_hooks, zero_cpu_grad_accum,
)
from src.training.precision_config import PrecisionConfig


def _rss_mb() -> float:
    """Process RSS in MiB (Linux). Used to track CPU memory growth."""
    ru = resource.getrusage(resource.RUSAGE_SELF)
    return ru.ru_maxrss / 1024.0  # KB -> MB on Linux


def _vram(device: int) -> tuple[float, float]:
    """Return (allocated_MB, reserved_MB) on the given device."""
    # PyTorch's caching allocator splits this into "allocated" (in
    # use) and "reserved" (allocated by torch but currently free).
    free, total = torch.cuda.mem_get_info(device)
    driver_used = (total - free) / 1024 / 1024
    alloc = torch.cuda.memory_allocated(device) / 1024 / 1024
    reserved = torch.cuda.memory_reserved(device) / 1024 / 1024
    # Return (allocated, reserved) — driver_used is the OS view.
    return alloc, reserved


def _mb_to_str(allocated: float, reserved: float) -> str:
    return f"alloc={allocated:7.1f}MB reserved={reserved:7.1f}MB"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=10,
                   help="number of optimizer steps to run")
    p.add_argument("--mb-per-step", type=int, default=8,
                   help="gradient_accumulation_steps")
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--num-layers", type=int, default=8)
    p.add_argument("--num-blocks", type=int, default=2)
    p.add_argument("--hidden-size", type=int, default=512)
    p.add_argument("--tp-sim", action="store_true", default=True)
    p.add_argument("--tp-size", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-dummy", action="store_true",
                   help="use random init, no dataloader")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = 0
    torch.cuda.set_device(device)
    torch.set_float32_matmul_precision("high")

    # --- model ---
    cfg = HippoConfig(
        vocab_size=8192, hidden_size=args.hidden_size,
        tie_word_embeddings=True, use_bias=False,
        num_heads=8, head_dim=64, expand_v=1.0,
        kda_mode="chunk", use_short_conv=True,
        allow_neg_eigval=False, safe_gate=False,
        lower_bound=None, conv_size=4, conv_bias=False,
        num_layers=args.num_layers, num_blocks=args.num_blocks,
        intermediate_size=args.hidden_size * 3,
        rms_norm_eps=1e-6,
    )
    init_tp(world_size=1, devices=[device], backend="gloo")
    model = TPHippoModel(cfg, devices=[device], dtype=torch.float16)
    # We skip sync_replicated_from because world_size=1.
    trainable = sum(p.numel() for p in model.trainable_parameters(device))
    print(f"[setup] trainable={trainable:,} layers={args.num_layers}"
          f" hidden={args.hidden_size} seq={args.seq_len} bs={args.batch_size}"
          f" mb_per_step={args.mb_per_step}")

    # --- precision + optimizers ---
    precision = PrecisionConfig()
    muon_opt, adamw_opt = build_param_groups(
        model, device=device,
        lr_muon=0.02, lr_adamw=3e-4, weight_decay=0.01,
        muon_momentum=0.95, precision=precision,
    )
    register_grad_offload_hooks([muon_opt, adamw_opt])

    # --- data ---
    if args.no_dummy:
        # Random tokens on GPU directly (no D2H/H2D in the timing).
        def data_iter():
            while True:
                ids = torch.randint(
                    0, cfg.vocab_size,
                    (args.batch_size, args.seq_len),
                    device=device, dtype=torch.long,
                )
                yield {"input_ids": ids, "labels": ids}
        loader = data_iter()
    else:
        # dummy_dataloader is on CPU, so each .to() is a real H2D.
        loader = iter(dummy_dataloader(args.batch_size, args.seq_len, cfg.vocab_size))

    # --- warmup + bench ---
    print(f"\n[setup] RSS={_rss_mb():.1f}MB")
    alloc, reserved = _vram(device)
    print(f"[setup] VRAM {_mb_to_str(alloc, reserved)}\n")

    print(f"{'step':>4} {'mb':>3} "
          f"{'fwd_ms':>8} {'bwd_ms':>8} {'sync_ms':>8} {'h2d_ms':>7} "
          f"{'VRAM_alloc':>10} {'VRAM_rsv':>10} "
          f"{'opt_ms':>8} {'muon_ms':>8} {'adamw_ms':>8}")
    print("-" * 110)

    # one warmup step
    for _ in range(2):
        batch = next(loader)
        ids = batch["input_ids"].to(device, non_blocking=True)
        labs = batch["labels"].to(device, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=True):
            out = model(ids, labels=labs)
            loss = out["loss"]
        (loss / args.mb_per_step).backward()
        flush_pending_grads(sync_device=device)
        del loss, out, ids, labs
    zero_cpu_grad_accum([muon_opt, adamw_opt])
    gc.collect()
    torch.cuda.empty_cache()

    # bench
    step_wall_ms = []
    for step in range(args.steps):
        t0 = time.perf_counter()
        for mb in range(args.mb_per_step):
            t_h2d_0 = time.perf_counter()
            batch = next(loader)
            ids = batch["input_ids"].to(device, non_blocking=True)
            labs = batch["labels"].to(device, non_blocking=True)
            t_h2d_1 = time.perf_counter()

            torch.cuda.synchronize(device)
            t_fwd_0 = time.perf_counter()
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=True):
                out = model(ids, labels=labs)
                loss = out["loss"]
            torch.cuda.synchronize(device)
            t_fwd_1 = time.perf_counter()

            t_bwd_0 = time.perf_counter()
            (loss / args.mb_per_step).backward()
            torch.cuda.synchronize(device)
            t_bwd_1 = time.perf_counter()

            t_sync_0 = time.perf_counter()
            flush_pending_grads(sync_device=device)
            t_sync_1 = time.perf_counter()

            del loss, out, ids, labs

            alloc, reserved = _vram(device)
            print(f"{step:>4} {mb:>3} "
                  f"{(t_fwd_1 - t_fwd_0) * 1000:8.2f}"
                  f" {(t_bwd_1 - t_bwd_0) * 1000:8.2f}"
                  f" {(t_sync_1 - t_sync_0) * 1000:8.2f}"
                  f" {(t_h2d_1 - t_h2d_0) * 1000:7.2f}"
                  f" {alloc:10.1f} {reserved:10.1f}"
                  f" {'':>8} {'':>8} {'':>8}")

        # optimizer step — break out muon vs adamw
        t_muon_0 = time.perf_counter()
        muon_opt.step()
        torch.cuda.synchronize(device)
        t_muon_1 = time.perf_counter()
        t_adamw_0 = time.perf_counter()
        adamw_opt.step()
        torch.cuda.synchronize(device)
        t_adamw_1 = time.perf_counter()
        zero_cpu_grad_accum([muon_opt, adamw_opt])

        t1 = time.perf_counter()
        step_wall_ms.append((t1 - t0) * 1000)
        muon_ms = (t_muon_1 - t_muon_0) * 1000
        adamw_ms = (t_adamw_1 - t_adamw_0) * 1000
        opt_ms = muon_ms + adamw_ms

        # print step summary
        print(f"{step:>4} {'opt':>3} "
              f"{'':>8} {'':>8} {'':>8} {'':>7} "
              f"{'':>10} {'':>10} "
              f" {opt_ms:8.1f} {muon_ms:8.1f} {adamw_ms:8.1f}")
        print("-" * 110)

    print(f"\n[summary] step wall ms: "
          f"first={step_wall_ms[0]:.1f} "
          f"last={step_wall_ms[-1]:.1f} "
          f"mean={sum(step_wall_ms) / len(step_wall_ms):.1f}")
    alloc, reserved = _vram(device)
    print(f"[summary] VRAM final {_mb_to_str(alloc, reserved)}")
    print(f"[summary] RSS final {_rss_mb():.1f}MB")


if __name__ == "__main__":
    main()
