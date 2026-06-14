"""Numerical correctness: CPU vs GPU muon step produce same mom_buf.

Compares:
  - mom_buf (the quantized momentum) after one muon step
  - mom_scale (per-row BF16 scale) after one muon step
  - s.param.data (the updated param)

The comparison is done against the production CPUMuon.step()
output. Differences are expected to be within FP16 / int8
quantization noise (~1 ULP per element).

TEMPORARY in test/_tmp/. Delete before commit.
"""
from __future__ import annotations

import argparse
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
    CPUMuon, build_param_groups, flush_pending_grads,
    register_grad_offload_hooks, zero_cpu_grad_accum,
)
from src.training.precision_config import PrecisionConfig


def muon_step_gpu(muon_opt: CPUMuon) -> None:
    """Same as in test_muon_gpu_step.py — duplicated here to keep
    this test independent of that file's edit history."""
    mom = muon_opt.momentum
    lr = muon_opt.lr
    wd = muon_opt.weight_decay
    CHUNK_ROWS = CPUMuon._STREAM_CHUNK_ROWS
    for s in muon_opt.state.values():
        if s.accum.abs().sum().item() == 0:
            continue
        shape = s.shape
        rows, cols = shape[0], shape[1]
        device = s.param.device

        mom_buf_gpu = s.mom_buf.to(device, non_blocking=True)
        accum_gpu = s.accum.to(device, non_blocking=True)
        scale_gpu = s.mom_scale.to(device, non_blocking=True)

        q_2d = mom_buf_gpu.float().view(rows, cols)
        scale_2d = scale_gpu.float().unsqueeze(1)
        m_fp32_gpu = q_2d * scale_2d

        g_2d = accum_gpu.float().view(rows, cols)
        m_fp32_gpu.mul_(mom).add_(g_2d, alpha=1.0 - mom)

        row_max = m_fp32_gpu.abs().amax(dim=1).clamp(min=1e-8)
        new_scale_fp32 = row_max / 127.0
        new_scale_bf16 = new_scale_fp32.to(torch.bfloat16)
        new_scale_2d = new_scale_bf16.float().unsqueeze(1)
        q_int8_gpu = (m_fp32_gpu / new_scale_2d).round() \
                        .clamp(-128, 127).to(torch.int8)

        s.mom_buf.copy_(q_int8_gpu.view(-1), non_blocking=True)
        s.mom_scale.copy_(new_scale_bf16, non_blocking=True)

        if wd != 0.0:
            s.param.data.mul_(1.0 - lr * wd)

        m_fp16 = m_fp32_gpu.to(torch.float16)
        for r_start in range(0, rows, CHUNK_ROWS):
            r_end = min(r_start + CHUNK_ROWS, rows)
            m_chunk = m_fp16[r_start:r_end]
            update = muon_opt._newton_schulz(m_chunk)
            update = update.to(s.param.dtype)
            s.param.data[r_start:r_end].add_(update, alpha=-lr)

        torch.cuda.current_stream(device).synchronize()
        s.accum.zero_()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--hidden-size", type=int, default=512)
    p.add_argument("--intermediate", type=int, default=1536)
    p.add_argument("--num-heads", type=int, default=8)
    p.add_argument("--head-dim", type=int, default=64)
    args = p.parse_args()

    torch.manual_seed(42)
    device = 0
    torch.cuda.set_device(device)
    torch.set_float32_matmul_precision("high")
    init_tp(world_size=1, devices=[device], backend="gloo")

    def build():
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
        return cfg, model, muon_opt, adamw_opt

    cfg_a, model_a, muon_a, adamw_a = build()
    cfg_b, model_b, muon_b, adamw_b = build()

    loader_a = iter(dummy_dataloader(args.batch_size, args.seq_len, cfg_a.vocab_size))
    loader_b = iter(dummy_dataloader(args.batch_size, args.seq_len, cfg_b.vocab_size))
    mb = 4

    # Forward + backward + flush on both models with the SAME inputs.
    for _ in range(mb):
        for model, muon_opt, adamw_opt, loader in (
            (model_a, muon_a, adamw_a, loader_a),
            (model_b, muon_b, adamw_b, loader_b),
        ):
            batch = next(loader)
            ids = batch["input_ids"].to(device, non_blocking=True)
            labs = batch["labels"].to(device, non_blocking=True)
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=True):
                out = model(ids, labels=labs)
            (out["loss"] / mb).backward()
            flush_pending_grads(sync_device=device)

    # Snapshot state pre-step.
    pre_mom_buf = {id(p): s.mom_buf.clone() for p, s in
                   [(s.param, s) for s in muon_a.state.values()]}
    pre_scale = {id(p): s.mom_scale.clone() for p, s in
                 [(s.param, s) for s in muon_a.state.values()]}
    pre_param = {id(p): p.data.clone() for p in
                 [s.param for s in muon_a.state.values()]}

    # Copy pre-step state from A to B so they start identical.
    for s_a, s_b in zip(muon_a.state.values(), muon_b.state.values()):
        assert s_a.param is not s_b.param
        s_b.mom_buf.copy_(s_a.mom_buf)
        s_b.mom_scale.copy_(s_a.mom_scale)
        s_b.param.data.copy_(s_a.param.data)
        s_b.accum.copy_(s_a.accum)
        s_b.step = s_a.step

    # Run CPU step on A, GPU step on B.
    muon_a.step()
    muon_step_gpu(muon_b)

    # Compare mom_buf, mom_scale, param.data for every param.
    print(f"\n[correctness] comparing {len(muon_a.state)} muon params:")
    mom_buf_max_diff = 0
    scale_max_diff = 0.0
    param_max_diff = 0.0
    mom_buf_total_diff = 0
    mom_buf_total_elt = 0
    scale_total_diff = 0.0
    scale_total_elt = 0
    param_total_diff = 0.0
    param_total_elt = 0
    for s_a, s_b in zip(muon_a.state.values(), muon_b.state.values()):
        # mom_buf: element-wise abs diff in [-255, 255]
        d_q = (s_a.mom_buf.to(torch.int32) - s_b.mom_buf.to(torch.int32)).abs()
        mom_buf_max_diff = max(mom_buf_max_diff, int(d_q.max()))
        mom_buf_total_diff += int(d_q.sum())
        mom_buf_total_elt += d_q.numel()

        # mom_scale: BF16, compare in FP32
        d_s = (s_a.mom_scale.float() - s_b.mom_scale.float()).abs()
        scale_max_diff = max(scale_max_diff, float(d_s.max()))
        scale_total_diff += float(d_s.sum())
        scale_total_elt += d_s.numel()

        # param.data: FP16, compare in FP32
        d_p = (s_a.param.data.float() - s_b.param.data.float()).abs()
        param_max_diff = max(param_max_diff, float(d_p.max()))
        param_total_diff += float(d_p.sum())
        param_total_elt += d_p.numel()

    print(f"  mom_buf:       max diff={mom_buf_max_diff:>3d}/255"
          f"  mean diff={mom_buf_total_diff / mom_buf_total_elt:.4f}/255"
          f"  ({mom_buf_total_elt:,} elts)")
    print(f"  mom_scale:     max diff={scale_max_diff:.4e}"
          f"  mean diff={scale_total_diff / scale_total_elt:.4e}")
    print(f"  param.data:    max diff={param_max_diff:.4e}"
          f"  mean diff={param_total_diff / param_total_elt:.4e}")

    # Sanity thresholds:
    #   mom_buf:   each value can differ by 1-2 (FP16 cast noise in
    #              the dequant/requant boundary).
    #   mom_scale: tiny (BF16 mantissa loss on the divide).
    #   param.data: tiny (small momentum update * FP16 NS noise).
    ok = (
        mom_buf_max_diff <= 5
        and scale_max_diff <= 1e-3
        and param_max_diff <= 5e-3
    )
    print(f"\n[verdict] {'PASS' if ok else 'FAIL'}"
          f"  (thresholds: mom_buf<=5, scale<=1e-3, param<=5e-3)")


if __name__ == "__main__":
    main()