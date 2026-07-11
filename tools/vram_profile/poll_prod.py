"""High-frequency VRAM poll during production training.

Goal: catch the transient peak between fwd_end and bwd_end that the
synchronous snapshot might miss. Use a background thread polling
torch.cuda.mem_get_info() at ~100 Hz.
"""

import argparse, json, os, sys, time, threading
import torch

sys.path.insert(0, "/hy-tmp/HippoLM")
sys.path.insert(0, "/hy-tmp/HippoLM/test/_tmp")


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=1)
    p.add_argument("--chunk-size", type=int, default=16384)
    p.add_argument("--n-chunks", type=int, default=16)
    p.add_argument("--poll-hz", type=int, default=200)
    p.add_argument("--out", default="/tmp/vram_poll.json")
    return p.parse_args()


def main():
    args = parse()
    torch.manual_seed(0)

    # Polling state
    polls = []
    polling = threading.Event()
    polling.set()
    peak = {"alloc": 0, "reserved": 0, "driver": 0, "alloc_max_label": "",
            "reserved_max_label": "", "driver_max_label": ""}

    def poll_loop():
        interval = 1.0 / args.poll_hz
        next_t = time.perf_counter()
        while polling.is_set():
            now = time.perf_counter()
            if now < next_t:
                time.sleep(max(0, next_t - now - 0.001))
                continue
            next_t += interval
            try:
                free, total = torch.cuda.mem_get_info()
                a = torch.cuda.memory_allocated()
                r = torch.cuda.memory_reserved()
                d = total - free
                polls.append((now, a, r, d))
                if a > peak["alloc"]:
                    peak["alloc"] = a
                if r > peak["reserved"]:
                    peak["reserved"] = r
                if d > peak["driver"]:
                    peak["driver"] = d
            except Exception:
                pass

    # Build production runner.
    # Replicate the runner build from profile_vram.main() but only the setup.
    from src.models import HippoConfig
    from src.models.tp_model._primitives import init_tp
    from src.models.tp_model import TPHippoModel
    from src.training.data import dummy_dataloader
    from src.training.param_offload import (
        build_param_groups, flush_pending_grads, flush_manual_flush_params,
        register_grad_offload_hooks, zero_cpu_grad_accum,
    )
    from src.training.precision_config import PrecisionConfig

    device = 0
    torch.cuda.set_device(device)
    torch.set_float32_matmul_precision("high")
    init_tp(world_size=1, devices=[device], backend="gloo")
    cfg = HippoConfig(
        vocab_size=248320, hidden_size=1536, tie_word_embeddings=True,
        use_bias=False, num_heads=12, head_dim=128, expand_v=1.0,
        kda_mode="chunk", use_short_conv=True, allow_neg_eigval=False,
        safe_gate=True, lower_bound=-5.0, conv_size=4, conv_bias=False,
        num_layers=32, num_blocks=8, intermediate_size=4096,
        rms_norm_eps=1e-6, ffn_nvfp4=True,
    )
    print(f"[init] building model")
    model = TPHippoModel(cfg, devices=[device], dtype=torch.float16)
    precision = PrecisionConfig()
    muon_opt, adamw_opt = build_param_groups(
        model, device=device, lr_muon=0.02, lr_adamw=4e-3,
        weight_decay=0.01, muon_momentum=0.95, precision=precision,
    )
    device_mods = model.replicated_per_device[str(device)]
    embed_param = device_mods["embed_tokens"] if "embed_tokens" in device_mods else None
    manual_flush = [embed_param.weight] if (embed_param is not None and hasattr(embed_param, "weight")) else None
    register_grad_offload_hooks([muon_opt, adamw_opt], manual_flush_params=manual_flush)
    B, T = 1, args.chunk_size * args.n_chunks
    loader = dummy_dataloader(B, T, cfg.vocab_size)
    batch = next(iter(loader))
    input_ids_full = batch["input_ids"].to(device)
    labels_full = batch["labels"].to(device)

    torch.cuda.reset_peak_memory_stats()
    init_driver = torch.cuda.mem_get_info()[1] - torch.cuda.mem_get_info()[0]
    print(f"[init] driver used after init: {init_driver/1024**2:.1f} MiB")

    # Start poll thread
    poll_thread = threading.Thread(target=poll_loop, daemon=True)
    poll_thread.start()

    # Per-chunk labels for the polls (best-effort alignment)
    micro_batch_size = args.chunk_size
    n_chunks = args.n_chunks

    phase_labels = []

    def mark(label):
        # Force a synchronous flush so the next poll sees this phase
        torch.cuda.synchronize()
        # Take a snapshot AT this exact moment too
        free, total = torch.cuda.mem_get_info()
        a = torch.cuda.memory_allocated()
        r = torch.cuda.memory_reserved()
        d = total - free
        polls.append((time.perf_counter(), a, r, d))
        phase_labels.append((label, time.perf_counter(), a, r, d))

    try:
        for step in range(args.steps):
            kda_states = None
            for ci in range(n_chunks):
                mark(f"step{step}.chunk{ci}.fwd_in")
                s = ci * micro_batch_size
                e = s + micro_batch_size
                chunk_ids = input_ids_full[:, s:e]
                chunk_labs = labels_full[:, s:e]
                with torch.amp.autocast("cuda", dtype=torch.float16, enabled=True):
                    out = model(chunk_ids, labels=chunk_labs, cu_seqlens=None, kda_states=kda_states)
                mark(f"step{step}.chunk{ci}.fwd_out")
                kda_states = [st.detach() if st is not None else None for st in out["kda_states"]]
                chunk_loss = out["loss"]
                del out
                chunk_loss.backward()
                mark(f"step{step}.chunk{ci}.bwd_out")
                flush_pending_grads(sync_device=device)
                flush_manual_flush_params([muon_opt, adamw_opt])
                mark(f"step{step}.chunk{ci}.flush_out")
                torch.cuda.empty_cache()
                mark(f"step{step}.chunk{ci}.cache_out")
            mark(f"step{step}.norm_in")
            muon_opt.step()
            mark(f"step{step}.muon_step_out")
            adamw_opt.step()
            mark(f"step{step}.adamw_step_out")
            zero_cpu_grad_accum([muon_opt, adamw_opt])
            mark(f"step{step}.end")
    finally:
        polling.clear()
        poll_thread.join(timeout=2)

    # Write the poll log
    print(f"\n=== POLL SUMMARY ===")
    print(f"  poll Hz: {args.poll_hz}")
    print(f"  polls captured: {len(polls)}")
    print(f"  max allocated:  {peak['alloc']/1024**2:.1f} MiB")
    print(f"  max reserved:   {peak['reserved']/1024**2:.1f} MiB")
    print(f"  max driver:     {peak['driver']/1024**2:.1f} MiB")
    print(f"  init driver:    {init_driver/1024**2:.1f} MiB")

    # Find the chunk where max driver was hit
    max_d_poll = max(polls, key=lambda p: p[3])
    print(f"  max driver poll: t={max_d_poll[0]:.3f} alloc={max_d_poll[1]/1024**2:.1f}M reserved={max_d_poll[2]/1024**2:.1f}M driver={max_d_poll[3]/1024**2:.1f}M")

    # Find nearest phase label
    for label, t, a, r, d in phase_labels:
        if abs(t - max_d_poll[0]) < 0.5:
            print(f"    closest phase: {label} (dt={t-max_d_poll[0]:+.3f}s)")

    # Dump the polls for offline analysis
    with open(args.out, "w") as f:
        json.dump({
            "init_driver": init_driver,
            "max_alloc": peak["alloc"],
            "max_reserved": peak["reserved"],
            "max_driver": peak["driver"],
            "polls": [{"t": p[0], "alloc": p[1], "reserved": p[2], "driver": p[3]} for p in polls],
            "phase_labels": [{"label": l[0], "t": l[1], "alloc": l[2], "reserved": l[3], "driver": l[4]} for l in phase_labels],
        }, f)
    print(f"\n[done] wrote {args.out}")


if __name__ == "__main__":
    main()