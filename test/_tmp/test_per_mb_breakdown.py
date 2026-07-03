"""Break per-mb into fwd_ms / bwd_ms / sync_ms at production scale.

Emulates the exact loop in src/training/loop.py (lines 440-516):
- autocast forward
- sync + loss.item()
- backward + divide-by-grad-accum
- sync + flush_pending_grads

Reports per-phase timing so we can identify which phase dominates
the per-mb cost.

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
from src.models.tp_layers import init_tp
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
    p.add_argument("--hidden-size", type=int, default=1024)
    p.add_argument("--intermediate", type=int, default=1536)
    p.add_argument("--num-heads", type=int, default=16)
    p.add_argument("--head-dim", type=int, default=64)
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

    # Measure per-mb fwd / bwd / sync separately, emulating loop.py
    n_mb = args.steps * args.mb_per_step
    print(f"\n[per-mb] measuring {n_mb} microbatches at"
          f" B={args.batch_size} T={args.seq_len} layers={args.num_layers}"
          f" hidden={args.hidden_size}...")
    data_ms, fwd_ms, bwd_ms, sync_ms, total_ms = [], [], [], [], []
    for step in range(args.steps):
        for mb_idx in range(args.mb_per_step):
            t_loop = time.perf_counter()
            batch = next(loader)
            ids = batch["input_ids"].to(device, non_blocking=True)
            labs = batch["labels"].to(device, non_blocking=True)
            t_data = time.perf_counter()

            torch.cuda.synchronize(device)
            t_fwd_start = time.perf_counter()
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=True):
                out = model(ids, labels=labs)
            torch.cuda.synchronize(device)
            t_fwd_end = time.perf_counter()
            mb_loss = out["loss"].detach().float().item()
            t_bwd_start = time.perf_counter()
            (out["loss"] / args.mb_per_step).backward()
            torch.cuda.synchronize(device)
            t_bwd_end = time.perf_counter()
            flush_pending_grads(sync_device=device)
            t_sync_end = time.perf_counter()

            del out, ids, labs
            data_ms.append((t_data - t_loop) * 1000)
            fwd_ms.append((t_fwd_end - t_fwd_start) * 1000)
            bwd_ms.append((t_bwd_end - t_bwd_start) * 1000)
            sync_ms.append((t_sync_end - t_bwd_end) * 1000)
            total_ms.append((t_sync_end - t_loop) * 1000)
        zero_cpu_grad_accum([muon_opt, adamw_opt])

    def stat(name, vals):
        avg = sum(vals) / len(vals)
        mn = min(vals)
        mx = max(vals)
        print(f"  {name:>12}: avg={avg:6.1f}  min={mn:6.1f}  max={mx:6.1f}  ms")

    print(f"\n[per-mb breakdown] (n={len(data_ms)}):")
    stat("data_ms", data_ms)
    stat("fwd_ms", fwd_ms)
    stat("bwd_ms", bwd_ms)
    stat("sync_ms", sync_ms)
    stat("total_ms", total_ms)
    fwd_avg = sum(fwd_ms) / len(fwd_ms)
    bwd_avg = sum(bwd_ms) / len(bwd_ms)
    sync_avg = sum(sync_ms) / len(sync_ms)
    total_avg = sum(total_ms) / len(total_ms)
    print(f"\n[phase fractions of total per-mb]:")
    print(f"  fwd:  {100 * fwd_avg / total_avg:5.1f}%")
    print(f"  bwd:  {100 * bwd_avg / total_avg:5.1f}%")
    print(f"  sync: {100 * sync_avg / total_avg:5.1f}%")
    print(f"\n[per-mb sums]:")
    print(f"  fwd+bwd (GPU work):       {fwd_avg + bwd_avg:6.1f} ms")
    print(f"  data+sync (CPU/IO work):  "
          f"{sum(data_ms)/len(data_ms) + sync_avg:6.1f} ms")
    print(f"  sum of phases:            {fwd_avg + bwd_avg + sync_avg:6.1f} ms")
    print(f"  measured total:           {total_avg:6.1f} ms")
    print(f"  python overhead per mb:   "
          f"{total_avg - fwd_avg - bwd_avg - sync_avg:6.1f} ms")


if __name__ == "__main__":
    main()