"""Benchmark production muon step (GPU) vs an inline CPU baseline.

Production :class:`CPUMuon.step` now runs the dequantize / SGD /
requantize cycle on the GPU. This test keeps an inline reference
implementation of the *pre-optimization* CPU path so we can
measure the before/after speedup on the same model + grad data,
without needing to revert the production change.

The CPU baseline (reconstructed from the test's prior docstring
description of the original step()) does:

  1. CPU dequant  mom_buf + mom_scale  -> m_fp32         (CPU DRAM)
  2. CPU SGD      m_fp32 = beta * m_fp32 + accum         (CPU DRAM)
  3. CPU requant  m_fp32 -> mom_buf, mom_scale           (CPU DRAM)
  4. GPU weight decay (one full-param mul_)
  5. For each chunk:
       CPU FP32 -> GPU FP16  (H2D per chunk)
       GPU NS
       GPU apply

The production step does (1)-(3) on the GPU (async H2D of the
quantized state, then bandwidth-bound work on HBM, then async
D2H of the new mom_buf), and (5) reads m_fp32 from GPU instead
of doing a per-chunk H2D.

Run::

    python test/test_muon_gpu_step.py --steps 3

Prints the per-step wall time for both paths and the delta.
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
    CPUMuon, build_param_groups, flush_pending_grads,
    register_grad_offload_hooks, zero_cpu_grad_accum,
)
from src.training.precision_config import PrecisionConfig


def muon_step_cpu(muon_opt: CPUMuon) -> None:
    """Reference implementation of the *pre-optimization* CPU muon
    step. Reuses the same per-param state and the chunked NS +
    apply pattern; only the dequant / requant run on the CPU
    (the bandwidth-bound part that the production path moves
    to the GPU).

    In the merged-accumulator design, ``mom_buf`` doubles as the
    grad accumulator (mu=1 accumulation). The step just reads
    mom_buf (dequantized), requantizes, orthogonalizes, applies,
    and resets mom_buf to zero.
    """
    lr = muon_opt.lr
    wd = muon_opt.weight_decay
    CHUNK_ROWS = CPUMuon._STREAM_CHUNK_ROWS
    for s in muon_opt.state.values():
        if s.mom_buf.abs().sum().item() == 0:
            continue
        rows, cols = s.shape[0], s.shape[1]
        device = s.param.device

        # ---- CPU dequant (the bandwidth-bound part) ----
        m_fp32 = muon_opt._dequantize(s)              # [rows, cols] FP32, CPU

        # ---- CPU requant ----
        muon_opt._requantize(m_fp32, s)

        # ---- GPU weight decay (full param, once) ----
        if wd != 0.0:
            s.param.data.mul_(1.0 - lr * wd)

        # ---- NS + apply per chunk (CPU->GPU H2D per chunk) ----
        for r_start in range(0, rows, CHUNK_ROWS):
            r_end = min(r_start + CHUNK_ROWS, rows)
            m_chunk = m_fp32[r_start:r_end].to(device, non_blocking=True)
            m_chunk = m_chunk.to(torch.float16)
            update = muon_opt._newton_schulz(m_chunk)
            update = update.to(s.param.dtype)
            s.param.data[r_start:r_end].add_(update, alpha=-lr)

        s.mom_buf.zero_()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=3)
    p.add_argument("--mb-per-step", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--num-layers", type=int, default=4)
    p.add_argument("--hidden-size", type=int, default=1024)
    p.add_argument("--intermediate", type=int, default=3072)
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
    mb = args.mb_per_step

    # warmup (use the production step() so the cuda allocator
    # caches are warm and the chunked NS path is exercised)
    for _ in range(2):
        for _ in range(mb):
            batch = next(loader)
            ids = batch["input_ids"].to(device, non_blocking=True)
            labs = batch["labels"].to(device, non_blocking=True)
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=True):
                out = model(ids, labels=labs)
            (out["loss"] / mb).backward()
            flush_pending_grads(sync_device=device)
        zero_cpu_grad_accum([muon_opt, adamw_opt])
        muon_opt.step()
        adamw_opt.step()
        zero_cpu_grad_accum([muon_opt, adamw_opt])
    torch.cuda.empty_cache()

    # ---------------------------------------------------------- #
    # Phase 1: CPU baseline (inline reference)                  #
    # ---------------------------------------------------------- #
    # Repopulate grads -> step -> zero. zero MUST happen AFTER
    # step or muon will skip params with zero accum.
    print(f"\n[baseline] CPU dequant/sgd/requant, {args.steps} muon steps...")
    base_times = []
    for _ in range(args.steps):
        for _ in range(mb):
            batch = next(loader)
            ids = batch["input_ids"].to(device, non_blocking=True)
            labs = batch["labels"].to(device, non_blocking=True)
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=True):
                out = model(ids, labels=labs)
            (out["loss"] / mb).backward()
            flush_pending_grads(sync_device=device)

        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        muon_step_cpu(muon_opt)
        torch.cuda.synchronize(device)
        base_times.append((time.perf_counter() - t0) * 1000)
        adamw_opt.step()
        zero_cpu_grad_accum([muon_opt, adamw_opt])
    base_avg = sum(base_times) / len(base_times)
    print(f"  CPU muon step: {base_avg:6.1f} ms (n={len(base_times)})")

    # ---------------------------------------------------------- #
    # Phase 2: GPU production step (CPUMuon.step())             #
    # ---------------------------------------------------------- #
    print(f"\n[gpu] GPU dequant/sgd/requant, {args.steps} muon steps...")
    gpu_times = []
    for _ in range(args.steps):
        for _ in range(mb):
            batch = next(loader)
            ids = batch["input_ids"].to(device, non_blocking=True)
            labs = batch["labels"].to(device, non_blocking=True)
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=True):
                out = model(ids, labels=labs)
            (out["loss"] / mb).backward()
            flush_pending_grads(sync_device=device)

        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        muon_opt.step()
        torch.cuda.synchronize(device)
        gpu_times.append((time.perf_counter() - t0) * 1000)
        adamw_opt.step()
        zero_cpu_grad_accum([muon_opt, adamw_opt])
    gpu_avg = sum(gpu_times) / len(gpu_times)
    print(f"  GPU muon step: {gpu_avg:6.1f} ms (n={len(gpu_times)})")

    delta = base_avg - gpu_avg
    pct = 100 * delta / base_avg
    print(f"\n[result] cpu={base_avg:.1f}  gpu={gpu_avg:.1f} ms"
          f"  delta={delta:+.1f} ms  ({pct:+.1f}%)")
    print(f"  projected per-step savings: {delta:+.1f} ms")


if __name__ == "__main__":
    main()
