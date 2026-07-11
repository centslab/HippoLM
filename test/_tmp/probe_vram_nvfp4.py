"""High-frequency VRAM probe at base.yml config — NVFP4 mode-3 path.

Differences from train.py:
- max_steps = 2 (we just need peak)
- Background thread samples torch.cuda memory stats at 200 Hz
  (nvitop's 1 Hz sampling misses transient spikes from the bwd
   graph build and chunked FP4 repack).
- After step 1 prints a per-module breakdown so we can see which
  classes of buffer are using what.

Long-lived diagnostic in test/_tmp/ (per the directory's README
convention). NOT a pytest test — run with ``python
test/_tmp/probe_vram_nvfp4.py``. Re-run after any model / precision
refactor that touches the FFN storage layout; diff the peak
``memory_allocated`` line against the recorded baseline at
``docs/vram_debugging.md`` §10.

The mode-3 storage-contract assertions (BF16 master absent, packed
buffers present, grad_w stash drained) used to live here. They are
now formal pytest tests in ``test/test_nvfp4_no_bf16_master.py``
(``TestStorageContract`` / ``TestOptimizerIntegration`` /
``TestBuildParamGroupsWiring``) so a regression now fails the test
suite instead of being silently hidden in a hand-rolled script.
"""
from __future__ import annotations

import argparse
import sys
import threading
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
    accumulate_grads_to_cpu,
    register_grad_offload_hooks, zero_cpu_grad_accum,
)
from src.training.precision_config import PrecisionConfig

# Load base.yml's precision block so we match what train.py does.
import yaml as _yaml
_BASE_YML = _yaml.safe_load(
    (_REPO / "configs" / "base.yml").read_text()
)
_BASE_PRECISION_DICT = _BASE_YML.get("precision", {})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=2)
    p.add_argument("--seq-len", type=int, default=16384)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--sample-hz", type=int, default=200)
    args = p.parse_args()

    torch.manual_seed(0)
    device = 0
    torch.cuda.set_device(device)
    torch.set_float32_matmul_precision("high")
    init_tp(world_size=1, devices=[device], backend="gloo")

    cfg = HippoConfig(
        vocab_size=248320,
        hidden_size=1536,
        tie_word_embeddings=True, use_bias=False,
        num_heads=12, head_dim=128,
        expand_v=1.0, kda_mode="chunk",
        use_short_conv=True, allow_neg_eigval=False,
        safe_gate=True, lower_bound=-5.0,
        conv_size=4, conv_bias=False,
        num_layers=32, num_blocks=8,
        intermediate_size=4096,
        rms_norm_eps=1e-6,
        ffn_nvfp4=True,
        ffn_nvfp4_marlin=True,
        ffn_nvfp4_no_bf16_master=True,
    )

    precision = PrecisionConfig.from_dict(_BASE_PRECISION_DICT)
    weight_dtype = precision.model_weights.dtype.to_torch()
    model = TPHippoModel(cfg, devices=[device], dtype=weight_dtype)
    model.sync_replicated_from(device)

    # ---- Optimizers ----
    muon_opt, adamw_opt = build_param_groups(
        model, device=device,
        lr_muon=0.02, lr_adamw=0.004,
        weight_decay=0.01, adamw_beta1=0.9, adamw_beta2=0.95,
        adamw_eps=1e-8, muon_momentum=0.95,
        muon_weight_decay=0.0, precision=precision,
    )
    manual_flush = []
    device_mods = model.replicated_per_device[str(device)]
    embed_param = device_mods["embed_tokens"] if "embed_tokens" in device_mods else None
    if embed_param is not None and hasattr(embed_param, "weight"):
        manual_flush.append(embed_param.weight)
    register_grad_offload_hooks([muon_opt, adamw_opt], manual_flush)

    # ---- High-frequency VRAM sampler ----
    sample_interval = 1.0 / args.sample_hz
    samples = []
    sampling = True
    peak_alloc = 0
    peak_reserved = 0

    def sample_loop():
        nonlocal peak_alloc, peak_reserved
        while sampling:
            try:
                a = torch.cuda.memory_allocated(device)
                r = torch.cuda.memory_reserved(device)
                samples.append((time.perf_counter(), a, r))
                peak_alloc = max(peak_alloc, a)
                peak_reserved = max(peak_reserved, r)
            except Exception:
                pass
            time.sleep(sample_interval)

    sampler = threading.Thread(target=sample_loop, daemon=True)
    sampler.start()

    dataloader = dummy_dataloader(args.batch_size, args.seq_len, cfg.vocab_size)

    print("=" * 78)
    print("Training step loop ({} steps)".format(args.steps))
    print("=" * 78)
    t_start = time.perf_counter()
    for step_idx, batch in enumerate(dataloader):
        if step_idx >= args.steps:
            break
        a_before = torch.cuda.memory_allocated(device)
        print(f"  [step {step_idx}] alloc before fwd: {a_before/1024**2:.1f} MiB")
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        outputs = model(input_ids, labels=labels)
        a_after_fwd = torch.cuda.memory_allocated(device)
        print(f"  [step {step_idx}] alloc after fwd:  {a_after_fwd/1024**2:.1f} MiB"
              f"  (Δ={a_after_fwd-a_before:+.1f} MiB)")
        loss = outputs["loss"]
        del outputs
        loss.backward()
        flush_pending_grads(sync_device=device)
        # flush_pending_grads only empties the per-param hook queue;
        # NVFP4 mode-3's stash is consumed by accumulate_grads_to_cpu
        # (the regression test for that contract lives in
        # test/test_nvfp4_no_bf16_master.py::TestOptimizerIntegration).
        accumulate_grads_to_cpu([muon_opt, adamw_opt], sync_device=device)
        a_after_bwd = torch.cuda.memory_allocated(device)
        print(f"  [step {step_idx}] alloc after bwd:  {a_after_bwd/1024**2:.1f} MiB"
              f"  (Δ={a_after_bwd-a_after_fwd:+.1f} MiB)")
        # step optimizers
        muon_opt.step()
        adamw_opt.step()
        zero_cpu_grad_accum([muon_opt, adamw_opt])
        from src.models.ops.nvfp4_linear import repack_nvfp4_weights
        repack_nvfp4_weights(model)
        a_after_step = torch.cuda.memory_allocated(device)
        print(f"  [step {step_idx}] alloc after step: {a_after_step/1024**2:.1f} MiB"
              f"  (Δ={a_after_step-a_after_bwd:+.1f} MiB)")
        torch.cuda.synchronize()

    # ---- Frozen-weight audit (delegated to formal tests) ----
    # The mode-3 storage-contract and build_param_groups wiring checks
    # used to live below this comment. They've been promoted to
    # test/test_nvfp4_no_bf16_master.py::TestBuildParamGroupsWiring
    # (regression test for the wiring-gap bug from
    # project_nvfp4_mode3_unwired.md, 2026-07-10). Run that file under
    # pytest instead of this probe to catch that class of regression.

    elapsed = time.perf_counter() - t_start
    sampling = False
    sampler.join(timeout=1.0)

    print()
    print("=" * 78)
    print(f"VRAM sampler: {len(samples)} samples at ~{args.sample_hz} Hz"
          f" over {elapsed:.2f}s")
    print(f"Peak torch.cuda.memory_allocated:   {peak_alloc/1024**2:.1f} MiB")
    print(f"Peak torch.cuda.memory_reserved:    {peak_reserved/1024**2:.1f} MiB")
    free, total = torch.cuda.mem_get_info(device)
    used = total - free
    print(f"torch.cuda.mem_get_info used:       {used/1024**2:.1f} MiB /"
          f" {total/1024**2:.1f} MiB")
    print()
    # Distribution: report top-10 spike samples so we can see if peak
    # is reached in fwd, bwd, or post-step.
    top = sorted(samples, key=lambda s: -s[1])[:10]
    print("Top-10 peak samples (alloc MiB / relative offset in s):")
    for t, a, r in top:
        print(f"  alloc={a/1024**2:7.1f} MiB  reserved={r/1024**2:7.1f} MiB"
              f"  t={t - samples[0][0]:6.3f}s")

    # ---- Per-module GPU footprint ----
    print()
    print("=" * 78)
    print("Per-module GPU tensor footprint (alloc bytes)")
    print("=" * 78)
    by_class = {}
    by_class_count = {}
    for name, p in model.named_parameters():
        if not isinstance(p, torch.nn.Parameter):
            continue
        if p.device.type != "cuda":
            continue
        cls = p.__class__.__name__
        # Use the owning module's class instead
        by_class.setdefault(cls, 0)
        by_class[cls] += p.numel() * p.element_size()
    # Buffers too
    for name, b in model.named_buffers():
        if not isinstance(b, torch.Tensor):
            continue
        if b.device.type != "cuda":
            continue
        key = f"buf[{b.dtype}]"
        by_class[key] = by_class.get(key, 0) + b.numel() * b.element_size()

    # Manual scan: for each module, count all tensors it owns on GPU.
    by_module_class = {}
    by_module_count = {}
    for mod_name, mod in model.named_modules():
        cls = mod.__class__.__name__
        mod_bytes = 0
        for n, p in mod.named_parameters(recurse=False):
            if isinstance(p, torch.Tensor) and p.device.type == "cuda":
                mod_bytes += p.numel() * p.element_size()
        for n, b in mod.named_buffers(recurse=False):
            if isinstance(b, torch.Tensor) and b.device.type == "cuda":
                mod_bytes += b.numel() * b.element_size()
        if mod_bytes > 0:
            by_module_class[cls] = by_module_class.get(cls, 0) + mod_bytes
            by_module_count[cls] = by_module_count.get(cls, 0) + 1

    print("Per-module-class GPU bytes:")
    for cls, b in sorted(by_module_class.items(), key=lambda kv: -kv[1]):
        n = by_module_count[cls]
        print(f"  {cls:40s}  {b/1024**2:8.1f} MiB  ({n} modules,"
              f"  ~{b/max(1,n)/1024**2:.1f} MiB/mod)")

    # ---- Detailed NVFP4 dump (mode-3 informational) ----
    # Diagnostic-only: shows the storage layout for the first 6 NVFP4
    # modules. The correctness assertions on this layout live in
    # test_nvfp4_no_bf16_master.py::TestStorageContract — this dump
    # just prints the byte counts so the VRAM peak can be attributed.
    print()
    print("=" * 78)
    print("NVFP4 module detail (first 6 modules, informational)")
    print("=" * 78)
    nvfp4_modules_for_dump = [
        (name, mod) for name, mod in model.named_modules()
        if mod.__class__.__name__ in (
            "NVFP4Linear",
            "NVFP4ColumnParallelLinear",
            "NVFP4RowParallelLinear",
        )
    ]
    total_bf16 = 0
    for name, mod in nvfp4_modules_for_dump[:6]:
        lines = []
        if hasattr(mod, "weight") and isinstance(mod.weight, torch.nn.Parameter):
            w_b = mod.weight.numel() * mod.weight.element_size()
            lines.append(f"weight(BF16)={w_b/1024**2:.1f}MiB")
            total_bf16 += w_b
        for buf_name in ("packed_weight", "scales", "global_scale",
                         "_scales_for_kernel", "_global_scale_adj"):
            buf = getattr(mod, buf_name, None)
            if isinstance(buf, torch.Tensor) and buf.numel() > 0:
                lines.append(f"{buf_name}({buf.dtype})={buf.numel()*buf.element_size()/1024**2:.2f}MiB")
        print(f"  {name}: {', '.join(lines) or '(empty)'}")
    if len(nvfp4_modules_for_dump) > 6:
        print(f"  ... ({len(nvfp4_modules_for_dump)-6} more)")
    print(f"Total NVFP4 modules in model: {len(nvfp4_modules_for_dump)}; "
          f"first-6 BF16-master bytes (informational): {total_bf16/1024**2:.1f} MiB")


if __name__ == "__main__":
    main()