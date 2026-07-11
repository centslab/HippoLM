"""Step-time breakdown: forward+backward only vs full step with optimizer.

Measures:
  - Per-mb fwd+bwd+sync time
  - Per-step optimizer time (muon + adamw)
  - Per-step total = N_mb * per-mb + optimizer

Helps quantify: is the bottleneck the GPU work, the CPU
optimizer, or the gloo sync (in TP-sim)?

TEMPORARY in test/_tmp/. Delete before commit.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO))

from src.models import HippoConfig
from src.models.tp_model._primitives import init_tp
from src.models.tp_model import TPHippoModel
from src.training.data import dummy_dataloader
from src.training.param_offload import (
    build_param_groups, flush_pending_grads,
    register_grad_offload_hooks, zero_cpu_grad_accum,
)
from src.training.precision_config import PrecisionConfig


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=3)
    p.add_argument("--mb-per-step", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--num-layers", type=int, default=4)
    p.add_argument("--hidden-size", type=int, default=512)
    p.add_argument("--intermediate", type=int, default=1536)
    p.add_argument("--num-heads", type=int, default=8)
    p.add_argument("--head-dim", type=int, default=64)
    p.add_argument("--tp-size", type=int, default=1)
    args = p.parse_args()

    torch.manual_seed(0)
    device = 0
    torch.cuda.set_device(device)
    torch.set_float32_matmul_precision("high")
    init_tp(world_size=1, devices=[device], backend="gloo")

    cfg = HippoConfig(
        vocab_size=8192, hidden_size=args.hidden_size,
        tie_word_embeddings=True, use_bias=False,
        num_heads=args.num_heads, head_dim=args.head_dim,
        expand_v=1.0,
        kda_mode="chunk", use_short_conv=True,
        allow_neg_eigval=False, safe_gate=False,
        lower_bound=None, conv_size=4, conv_bias=False,
        num_layers=args.num_layers, num_blocks=2,
        intermediate_size=args.intermediate,
        rms_norm_eps=1e-6,
    )
    model = TPHippoModel(cfg, devices=[device], dtype=torch.float16)
    precision = PrecisionConfig()
    muon_opt, adamw_opt = build_param_groups(
        model, device=device,
        lr_muon=0.02, lr_adamw=3e-4, weight_decay=0.01,
        muon_momentum=0.95, precision=precision,
    )
    register_grad_offload_hooks([muon_opt, adamw_opt])
    loader = iter(dummy_dataloader(args.batch_size, args.seq_len, cfg.vocab_size))

    # warmup
    for _ in range(2):
        for _ in range(args.mb_per_step):
            batch = next(loader)
            ids = batch["input_ids"].to(device, non_blocking=True)
            labs = batch["labels"].to(device, non_blocking=True)
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=True):
                out = model(ids, labels=labs)
            (out["loss"] / args.mb_per_step).backward()
            flush_pending_grads(sync_device=device)
        zero_cpu_grad_accum([muon_opt, adamw_opt])
    torch.cuda.empty_cache()

    # Measure A: per-mb fwd+bwd+sync only (no optimizer)
    print(f"\n[bench A] {args.steps * args.mb_per_step} microbatches, no opt...")
    times_a = []
    for step in range(args.steps):
        for mb in range(args.mb_per_step):
            t0 = time.perf_counter()
            batch = next(loader)
            ids = batch["input_ids"].to(device, non_blocking=True)
            labs = batch["labels"].to(device, non_blocking=True)
            torch.cuda.synchronize(device)
            t_fwd_0 = time.perf_counter()
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=True):
                out = model(ids, labels=labs)
            torch.cuda.synchronize(device)
            t_bwd_0 = time.perf_counter()
            (out["loss"] / args.mb_per_step).backward()
            torch.cuda.synchronize(device)
            t_sync_0 = time.perf_counter()
            flush_pending_grads(sync_device=device)
            torch.cuda.synchronize(device)
            t1 = time.perf_counter()
            times_a.append((t1 - t0) * 1000)
            del out, ids, labs
        zero_cpu_grad_accum([muon_opt, adamw_opt])
    print(f"  per-mb: {sum(times_a)/len(times_a):.1f} ms (n={len(times_a)})")

    # repopulate grads for optimizer
    for _ in range(args.mb_per_step):
        batch = next(loader)
        ids = batch["input_ids"].to(device, non_blocking=True)
        labs = batch["labels"].to(device, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=True):
            out = model(ids, labels=labs)
        (out["loss"] / args.mb_per_step).backward()
        flush_pending_grads(sync_device=device)
    zero_cpu_grad_accum([muon_opt, adamw_opt])

    # Measure B: optimizer step only (with non-zero grads)
    print(f"\n[bench B] {args.steps} optimizer steps (with non-zero grads)...")
    times_muon = []
    times_adamw = []
    for step in range(args.steps):
        # repopulate grads FIRST (don't zero them out before measuring)
        for _ in range(args.mb_per_step):
            batch = next(loader)
            ids = batch["input_ids"].to(device, non_blocking=True)
            labs = batch["labels"].to(device, non_blocking=True)
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=True):
                out = model(ids, labels=labs)
            (out["loss"] / args.mb_per_step).backward()
            flush_pending_grads(sync_device=device)

        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        muon_opt.step()
        torch.cuda.synchronize(device)
        times_muon.append((time.perf_counter() - t0) * 1000)
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        adamw_opt.step()
        torch.cuda.synchronize(device)
        times_adamw.append((time.perf_counter() - t0) * 1000)

        # zero accum AFTER stepping (so next iteration starts fresh)
        zero_cpu_grad_accum([muon_opt, adamw_opt])
    print(f"  muon step:  {sum(times_muon)/len(times_muon):.1f} ms")
    print(f"  adamw step: {sum(times_adamw)/len(times_adamw):.1f} ms")

    # summary
    mb_ms = sum(times_a)/len(times_a)
    muon_ms = sum(times_muon)/len(times_muon)
    adamw_ms = sum(times_adamw)/len(times_adamw)
    step_ms = args.mb_per_step * mb_ms + muon_ms + adamw_ms

    print(f"\n[summary] per-step decomposition (B={args.batch_size} T={args.seq_len}"
          f" layers={args.num_layers} hidden={args.hidden_size}):")
    print(f"  per-mb fwd+bwd+sync  : {mb_ms:6.1f} ms  × {args.mb_per_step} = {args.mb_per_step * mb_ms:6.1f} ms")
    print(f"  muon step             : {muon_ms:6.1f} ms")
    print(f"  adamw step            : {adamw_ms:6.1f} ms")
    print(f"  ----------------------------------------")
    print(f"  total step            : {step_ms:6.1f} ms")
    print(f"  per-mb equiv          : {step_ms/args.mb_per_step:6.1f} ms")
    print(f"  fwd+bwd+sync fraction : {100 * args.mb_per_step * mb_ms / step_ms:5.1f}%")
    print(f"  optimizer fraction     : {100 * (muon_ms + adamw_ms) / step_ms:5.1f}%")
    print(f"    of which muon       : {100 * muon_ms / step_ms:5.1f}%")
    print(f"    of which adamw      : {100 * adamw_ms / step_ms:5.1f}%")


if __name__ == "__main__":
    main()