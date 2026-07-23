"""Profile peak VRAM at production base.yml.

Walks the same path as scripts/train.py's single-GPU branch:
  - Build HippoConfig from production args
  - Construct TPHippoModel (TP=1, fp16)
  - build_param_groups + register_grad_offload_hooks
  - Run N steps of the chunked loop from src/training/loop.py
  - At each phase boundary (chunk fwd start/end, bwd start/end,
    grad-flush, optimizer step) record:
      * torch.cuda.memory_stats() snapshot
      * mem_get_info (driver-level)
      * max_memory_allocated diff vs the previous phase

Two outputs:
  1. human-readable per-phase log on stdout
  2. pickle dump of torch.cuda.memory._record_memory_history()
     for offline MemoryViz rendering

Usage::

    python -m tools.vram_profile.profile \\
        --steps 2 --n-chunks 16 --chunk-size 16384 \\
        --dump /tmp/vram.pickle --log /tmp/vram.log

See ``docs/vram_debugging.md`` for the full methodology.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path

import torch
from torch.cuda import nvtx

_REPO = Path(__file__).resolve().parents[2]  # vram_profile → tools → REPO
sys.path.insert(0, str(_REPO))

from src.models import HippoConfig
from src.models.tp_model._primitives import init_tp
from src.models.tp_model import TPHippoModel
from src.training.data import dummy_dataloader
from src.training.param_offload import (
    build_param_groups, flush_pending_grads, flush_manual_flush_params,
    register_grad_offload_hooks, zero_cpu_grad_accum,
)


# Per-phase measurement helper.
def _snap(label: str, peak_track: dict) -> None:
    free, total = torch.cuda.mem_get_info()
    stats = torch.cuda.memory_stats()
    cur = torch.cuda.memory_allocated()
    peak_track["max_allocated"] = max(
        peak_track["max_allocated"], cur
    )
    peak_track["max_reserved"] = max(
        peak_track["max_reserved"], stats["reserved_bytes.all.current"]
    )
    peak_track["max_driver"] = max(
        peak_track["max_driver"], total - free
    )
    peak_track["snaps"].append({
        "label": label,
        "alloc_cur": cur,
        "alloc_peak": stats["allocated_bytes.all.peak"],
        "reserved_cur": stats["reserved_bytes.all.current"],
        "reserved_peak": stats["reserved_bytes.all.peak"],
        "cached": stats.get("active_bytes.all.current", 0),
        "inactive_split": stats.get("inactive_split_bytes.all.current", 0),
        "driver_used": total - free,
        "driver_total": total,
        "num_alloc": stats.get("num_alloc_retries", 0),
        "num_ooms": stats.get("num_ooms", 0),
        "t": time.perf_counter(),
    })


def _phase_peaks(peak_track: dict) -> dict:
    return {
        "max_allocated_MiB": peak_track["max_allocated"] / 1024**2,
        "max_reserved_MiB": peak_track["max_reserved"] / 1024**2,
        "max_driver_MiB": peak_track["max_driver"] / 1024**2,
    }


def _live_total(snap) -> int:
    """Sum of all 'active_allocated' block sizes in a snapshot."""
    total = 0
    for seg in snap.get("segments", []):
        for blk in seg.get("blocks", []):
            if blk.get("state") == "active_allocated":
                total += blk["size"]
    return total


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=2,
                   help="training steps (chunked loop)")
    p.add_argument("--chunk-size", type=int, default=16384,
                   help="tokens per chunk (16384 = base.yml's micro_batch_size)")
    p.add_argument("--n-chunks", type=int, default=16,
                   help="number of chunks per step (base.yml: 16)")
    p.add_argument("--dump", type=str, default="/tmp/hippolm_vram.pickle",
                   help="output pickle path (MemoryViz-compatible)")
    p.add_argument("--log", type=str, default="/tmp/hippolm_vram.log",
                   help="per-phase text log")
    args = p.parse_args()

    if not torch.cuda.is_available():
        print("FATAL: CUDA not available", file=sys.stderr)
        sys.exit(1)

    torch.manual_seed(0)
    device = 0
    torch.cuda.set_device(device)
    torch.set_float32_matmul_precision("high")
    init_tp(world_size=1, devices=[device], backend="gloo")

    # Production config from base.yml (prod-shape).
    cfg = HippoConfig(
        vocab_size=248320, hidden_size=1536,
        tie_word_embeddings=True, use_bias=False,
        num_heads=12, head_dim=128,
        expand_v=1.0,
        kda_mode="chunk", use_short_conv=True,
        allow_neg_eigval=False, safe_gate=True,
        lower_bound=-5.0, conv_size=4, conv_bias=False,
        num_layers=32, num_blocks=8,
        intermediate_size=4096,
        rms_norm_eps=1e-6,
    )
    print(f"[init] config: {cfg}")
    print(f"[init] model build start (alloc={torch.cuda.memory_allocated()/1024**2:.1f} MiB)")
    t0 = time.perf_counter()
    model = TPHippoModel(cfg, devices=[device], dtype=torch.bfloat16)
    torch.cuda.synchronize()
    t_build = time.perf_counter() - t0
    print(f"[init] model built in {t_build:.1f}s, "
          f"alloc={torch.cuda.memory_allocated()/1024**2:.1f} MiB, "
          f"reserved={torch.cuda.memory_reserved()/1024**2:.1f} MiB")

    print(f"[init] param groups build start")
    muon_opt, adamw_opt = build_param_groups(
        model, device=device,
        lr_muon=0.02, lr_adamw=4e-3, weight_decay=0.01,
        muon_momentum=0.95,
    )
    torch.cuda.synchronize()
    print(f"[init] optimizers built, "
          f"alloc={torch.cuda.memory_allocated()/1024**2:.1f} MiB, "
          f"reserved={torch.cuda.memory_reserved()/1024**2:.1f} MiB")

    # Manual flush for tied embed.
    device_mods = model.replicated_per_device[str(device)]
    embed_param = device_mods["embed_tokens"] if "embed_tokens" in device_mods else None
    manual_flush = [embed_param.weight] if (embed_param is not None and hasattr(embed_param, "weight")) else None
    register_grad_offload_hooks([muon_opt, adamw_opt], manual_flush_params=manual_flush)
    print(f"[init] grad offload hooks registered")

    # Build a fixed random batch for the run (so we know shape).
    B = 1
    T = args.chunk_size * args.n_chunks
    print(f"[init] dummy batch build: B={B}, T={T}")
    loader = dummy_dataloader(B, T, cfg.vocab_size)
    batch = next(iter(loader))
    input_ids_full = batch["input_ids"].to(device, non_blocking=True)
    labels_full = batch["labels"].to(device, non_blocking=True)
    torch.cuda.synchronize()
    print(f"[init] input_ids on GPU: "
          f"alloc={torch.cuda.memory_allocated()/1024**2:.1f} MiB")

    # Begin memory recording. PyTorch 2.9 API: 'enabled' is a
    # string flag ('all' = alloc/free history + current state,
    # 'state' = current state only); 'stacks' controls per-allocation
    # Python frame recording (off here to keep the pickle small);
    # 'max_entries' bounds the ring buffer (we cap at 200k entries
    # to fit a 2-step run comfortably).
    print(f"[init] enabling memory history")
    torch.cuda.memory._record_memory_history(
        enabled="all",
        stacks="python",
        max_entries=200_000,
    )

    # Reset peak stats after init so we measure only the training
    # loop's peaks (model build allocates but does not measure peak
    # unless asked).
    torch.cuda.reset_peak_memory_stats()
    print(f"[init] reset peak stats; "
          f"alloc={torch.cuda.memory_allocated()/1024**2:.1f} MiB")

    log_lines = []

    def L(msg: str) -> None:
        print(msg)
        log_lines.append(msg)

    L(f"\n[run] starting {args.steps} steps, "
      f"n_chunks={args.n_chunks}, chunk_size={args.chunk_size}")
    L(f"[run] alloc baseline (post-reset): "
      f"{torch.cuda.memory_allocated()/1024**2:.1f} MiB")

    # Per-step peak tracking.
    step_peak_track = {"max_allocated": 0, "max_reserved": 0,
                       "max_driver": 0, "snaps": []}

    micro_batch_size = args.chunk_size
    n_chunks = args.n_chunks
    # We collect a snapshot of the allocator state at each chunk's
    # fwd_out (peak saved-tensor set) and bwd_out (residual grads).
    # The end-of-run snapshot only sees the residual model weights
    # because the peak saved-tensors have already been freed.
    peak_snapshots = []

    with nvtx.range(f"training_loop_{args.steps}_steps"):
        for step in range(args.steps):
            L(f"\n=== STEP {step} ===")
            _snap(f"step{step}_start", step_peak_track)

            kda_states = None
            for ci in range(n_chunks):
                with nvtx.range(f"chunk_{ci}_forward"):
                    s = ci * micro_batch_size
                    e = s + micro_batch_size
                    chunk_ids = input_ids_full[:, s:e]
                    chunk_labs = labels_full[:, s:e]
                    L(f"  [chunk {ci}] fwd in: "
                      f"alloc={torch.cuda.memory_allocated()/1024**2:.1f} MiB, "
                      f"reserved={torch.cuda.memory_reserved()/1024**2:.1f} MiB")
                    _snap(f"step{step}.chunk{ci}.fwd_in", step_peak_track)

                    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=True):
                        out = model(
                            chunk_ids, labels=chunk_labs,
                            cu_seqlens=None, kda_states=kda_states,
                        )

                    torch.cuda.synchronize()
                    L(f"  [chunk {ci}] fwd out: "
                      f"alloc={torch.cuda.memory_allocated()/1024**2:.1f} MiB, "
                      f"reserved={torch.cuda.memory_reserved()/1024**2:.1f} MiB")
                    _snap(f"step{step}.chunk{ci}.fwd_out", step_peak_track)

                    # Capture peak snapshots at every chunk's fwd_out
                    # AND bwd_out (the saved-tensor set lives
                    # through both). The TRUE peak might be a
                    # transient during bwd that we miss if we
                    # only sample once.
                    peak_snapshots.append(
                        (f"step{step}.chunk{ci}.fwd_out",
                         torch.cuda.memory._snapshot())
                    )

                    # Detach state for next chunk (truncated BPTT).
                    kda_states = [
                        st.detach() if st is not None else None
                        for st in out["kda_states"]
                    ]
                    chunk_loss = out["loss"]
                    del out
                    torch.cuda.synchronize()
                    L(f"  [chunk {ci}] post-detach: "
                      f"alloc={torch.cuda.memory_allocated()/1024**2:.1f} MiB, "
                      f"reserved={torch.cuda.memory_reserved()/1024**2:.1f} MiB")

                with nvtx.range(f"chunk_{ci}_backward"):
                    _snap(f"step{step}.chunk{ci}.bwd_in", step_peak_track)
                    chunk_loss.backward()
                    torch.cuda.synchronize()
                    _snap(f"step{step}.chunk{ci}.bwd_out", step_peak_track)
                    peak_snapshots.append(
                        (f"step{step}.chunk{ci}.bwd_out",
                         torch.cuda.memory._snapshot())
                    )
                    L(f"  [chunk {ci}] bwd out: "
                      f"alloc={torch.cuda.memory_allocated()/1024**2:.1f} MiB, "
                      f"reserved={torch.cuda.memory_reserved()/1024**2:.1f} MiB")
                    flush_pending_grads(sync_device=device)
                    flush_manual_flush_params([muon_opt, adamw_opt])
                    _snap(f"step{step}.chunk{ci}.flush_out", step_peak_track)
                    peak_snapshots.append(
                        (f"step{step}.chunk{ci}.flush_out",
                         torch.cuda.memory._snapshot())
                    )
                    L(f"  [chunk {ci}] flush out: "
                      f"alloc={torch.cuda.memory_allocated()/1024**2:.1f} MiB, "
                      f"reserved={torch.cuda.memory_reserved()/1024**2:.1f} MiB")
                    # Match the production loop's per-chunk empty_cache.
                    torch.cuda.empty_cache()
                    _snap(f"step{step}.chunk{ci}.cache_out", step_peak_track)

            # Per-step grad norm + optimizer step.
            _snap(f"step{step}.norm_in", step_peak_track)
            L(f"\n  [step {step}] norm in: "
              f"alloc={torch.cuda.memory_allocated()/1024**2:.1f} MiB")

            muon_opt.step()
            torch.cuda.synchronize()
            _snap(f"step{step}.muon_step_out", step_peak_track)
            L(f"  [step {step}] muon_step out: "
              f"alloc={torch.cuda.memory_allocated()/1024**2:.1f} MiB, "
              f"reserved={torch.cuda.memory_reserved()/1024**2:.1f} MiB")

            adamw_opt.step()
            torch.cuda.synchronize()
            _snap(f"step{step}.adamw_step_out", step_peak_track)
            L(f"  [step {step}] adamw_step out: "
              f"alloc={torch.cuda.memory_allocated()/1024**2:.1f} MiB, "
              f"reserved={torch.cuda.memory_reserved()/1024**2:.1f} MiB")

            zero_cpu_grad_accum([muon_opt, adamw_opt])
            torch.cuda.synchronize()
            _snap(f"step{step}.end", step_peak_track)
            L(f"  [step {step}] end: "
              f"alloc={torch.cuda.memory_allocated()/1024**2:.1f} MiB, "
              f"reserved={torch.cuda.memory_reserved()/1024**2:.1f} MiB")

    # End recording and dump.
    torch.cuda.memory._record_memory_history(enabled=False)
    print(f"\n[final] DUMPING memory history to {args.dump}")

    # Dump the peak snapshot (captured mid-fwd) AND the residual
    # end-of-run snapshot.
    try:
        with open(args.dump, "wb") as f:
            pickle.dump({
                "peak_snapshots": peak_snapshots,
                "end": torch.cuda.memory._snapshot(),
            }, f)
        print(f"[final] pickle written "
              f"({len(peak_snapshots)} peak snapshots + end)")
    except Exception as e:
        print(f"[final] pickle failed: {e}")

    # Write the per-phase log.
    with open(args.log, "w") as f:
        for line in log_lines:
            f.write(line + "\n")

    # Summary.
    print(f"\n=== SUMMARY ===")
    pp = _phase_peaks(step_peak_track)
    print(f"  per-step max allocated:  {pp['max_allocated_MiB']:.1f} MiB")
    print(f"  per-step max reserved:   {pp['max_reserved_MiB']:.1f} MiB")
    print(f"  per-step max driver:     {pp['max_driver_MiB']:.1f} MiB")
    print(f"  final alloc:             {torch.cuda.memory_allocated()/1024**2:.1f} MiB")
    print(f"  final reserved:          {torch.cuda.memory_reserved()/1024**2:.1f} MiB")
    print(f"  num_ooms:                "
          f"{torch.cuda.memory_stats().get('num_ooms', 0)}")

    # Per-snapshot live total (the "true peak" at each phase).
    print(f"\n=== Per-snapshot live-total (MiB) ===")
    for label, snap in peak_snapshots:
        total = _live_total(snap)
        n_seg = len(snap.get("segments", []))
        print(f"  {label:>40}: live={total/1024**2:8.1f}  "
              f"n_segs={n_seg:>3}")

    # Dump the per-phase JSON for downstream analysis.
    json.dump(step_peak_track, open(args.log + ".json", "w"), indent=2)


if __name__ == "__main__":
    main()