"""Profile forward+backward in detail.

Uses torch.profiler to break down a single fwd+bwd pass into
op-level timings. Identifies which kernels / ops dominate.

TEMPORARY in test/_tmp/. Delete before commit.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parent.parent
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
    torch.manual_seed(0)
    device = 0
    torch.cuda.set_device(device)
    torch.set_float32_matmul_precision("high")
    init_tp(world_size=1, devices=[device], backend="gloo")

    cfg = HippoConfig(
        vocab_size=8192, hidden_size=512,
        tie_word_embeddings=True, use_bias=False,
        num_heads=8, head_dim=64, expand_v=1.0,
        kda_mode="chunk", use_short_conv=True,
        allow_neg_eigval=False, safe_gate=False,
        lower_bound=None, conv_size=4, conv_bias=False,
        num_layers=4, num_blocks=2,
        intermediate_size=1536,
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

    loader = iter(dummy_dataloader(2, 512, cfg.vocab_size))

    # warmup
    for _ in range(3):
        batch = next(loader)
        ids = batch["input_ids"].to(device, non_blocking=True)
        labs = batch["labels"].to(device, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=True):
            out = model(ids, labels=labs)
        (out["loss"] / 4).backward()
        flush_pending_grads(sync_device=device)
    zero_cpu_grad_accum([muon_opt, adamw_opt])
    torch.cuda.empty_cache()

    # profile a single fwd+bwd
    batch = next(loader)
    ids = batch["input_ids"].to(device, non_blocking=True)
    labs = batch["labels"].to(device, non_blocking=True)
    torch.cuda.synchronize(device)

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
    ) as prof:
        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=True):
            out = model(ids, labels=labs)
        out["loss"].backward()
        torch.cuda.synchronize(device)

    print(f"\n[profile] Top 30 CUDA kernels during fwd+bwd:")
    print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=30))

    print(f"\n[profile] Top 30 CPU ops during fwd+bwd:")
    print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=30))


if __name__ == "__main__":
    main()