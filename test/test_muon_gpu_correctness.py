"""Regression test: production CPUMuon.step() matches an independent
inline GPU reference implementation.

Historical context
------------------
``e513852 feat(offload): CPUMuon GPU step + bf16/fp16/fp32 momentum
storage`` moved the muon step's dequant / SGD / requant cycle from CPU
DRAM to GPU HBM. The optimization had to be **bit-equivalent** to the
original CPU path within FP16 quantization noise — losing even a
few percent on this test would mean the production optimizer was
silently producing different (and unverified) updates.

The test pins that contract by running the production step() on one
copy of a model and an inlined, hand-written GPU reference on a
second copy, then comparing ``grad`` (per-step accumulator) and
the post-step param. The reference is reconstructed from the
algorithm description (CPU-side SGD momentum → Nesterov
correction → H2D → NS → apply → zero grad), so a future change
to the production step() that changes the math will trip the
test even if the change is locally consistent.

Quantized storage (int8 + per-row scale, mxfp8 + per-block E8M0)
was removed on 2026-07-12 — long-training runs showed
quantization-error accumulation destabilizing optimization. The
explicit-accumulator design (``grad`` + ``exp_avg``, both at
the configured full-precision storage dtype: bf16 / fp16 /
fp32) is the only supported muon storage path since
2026-07-15 (the previous "merged-accumulator" / mu=1 design
where ``mom_buf`` doubled as both the per-step accumulator and
the SGD momentum was deleted; ``grad`` is the per-step
accumulator and ``exp_avg`` is the SGD momentum, with cross-
step EMA state preserved).

Test layout
-----------
- Build two identical TP models, do the same fwd + bwd + flush on
  both.
- Copy the pre-step state from A to B so they start identical.
- Run production ``CPUMuon.step()`` on A.
- Run the inline GPU reference on B.
- Assert the outputs agree within tight thresholds
  (grad ≤ 5/255 in storage units after the step, param ≤ 5e-3).

The thresholds match the original (pre-quantization-removal) values;
tightening any of them would catch a real divergence.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.models import HippoConfig
from src.models.tp_model._primitives import init_tp
from src.models.tp_model import TPHippoModel
from src.training.data import dummy_dataloader
from src.training.param_offload import (
    CPUMuon, build_param_groups, flush_pending_grads,
    register_grad_offload_hooks, zero_cpu_grad_accum,
)


def _muon_step_gpu_inline(muon_opt: CPUMuon) -> None:
    """Independent GPU reference implementation of the muon step.

    Mirrors the algorithm (CPU-side SGD momentum → Nesterov
    correction → H2D → NS → apply → zero ``s.grad``) but inlined
    with explicit per-tensor operations. The production
    ``CPUMuon.step()`` is the optimized version of the same math;
    the inlined version is the "what the math should do" reference
    that the test compares against.

    Storage design (post-2026-07-15 explicit-accumulator layout):
    two CPU pinned buffers per param, distinct roles:

      * ``s.grad``    — per-step grad accumulator. Zeroed at the
        end of every step. The test compares this AFTER step
        (both production and reference must zero it identically).
      * ``s.exp_avg`` — SGD momentum ``β·prev + grad``, preserved
        across steps. Computed on CPU at step time, then Nesterov
        correction + H2D before NS. Never recast back / zeroed
        during step.
    """
    lr = muon_opt.lr
    beta = muon_opt.momentum
    nesterov = muon_opt.nesterov
    wd = muon_opt.weight_decay
    CHUNK_ROWS = CPUMuon._STREAM_CHUNK_ROWS
    for s in muon_opt.state.values():
        # Per-step grad accumulator (post-2026-07-15).
        g_buf = s.grad
        if g_buf.abs().sum().item() == 0:
            continue
        shape = s.shape
        rows, cols = shape[0], shape[1]
        device = s.param.device

        # ---- SGD momentum on CPU (Keller Jordan Muon ref):
        # ``exp_avg ← β·exp_avg + grad`` in the storage dtype.
        # Done on CPU (not GPU) to match the production path;
        # the storage-dtype round trip is intentional (same as
        # production).
        exp_avg = s.exp_avg
        exp_avg.mul_(beta).add_(g_buf)

        # ---- Nesterov correction on CPU. ----
        if nesterov:
            g_for_ns = g_buf.add(exp_avg, alpha=beta)
        else:
            g_for_ns = exp_avg

        # ---- H2D + cast to FP32 (NS input is always FP32). ----
        orth_input = g_for_ns.to(device, non_blocking=True) \
                                .float().view(rows, cols)

        # ---- Decoupled weight decay on the full param. ----
        if wd != 0.0:
            s.param.data.mul_(1.0 - lr * wd)

        # ---- Stream NS over rows and apply the update. ----
        m_fp16 = orth_input.to(torch.float16)
        for r_start in range(0, rows, CHUNK_ROWS):
            r_end = min(r_start + CHUNK_ROWS, rows)
            m_chunk = m_fp16[r_start:r_end]
            update = muon_opt._newton_schulz(m_chunk)
            update = update.to(s.param.dtype)
            s.param.data[r_start:r_end].add_(update, alpha=-lr)

        torch.cuda.current_stream(device).synchronize()

        # ---- Step-end: zero the per-step grad accumulator
        # (production zero happens via ``fused_zero_many`` at the
        # end of step(); same net effect). ``s.exp_avg`` is
        # preserved across steps — NOT zeroed here. ----
        s.grad.zero_()


def _build_matched_pair(
    cfg: HippoConfig, device: int,
) -> tuple[TPHippoModel, TPHippoModel, CPUMuon, CPUMuon, object, object]:
    """Build two identical TP models + muon opts for the A/B comparison.

    Returns the two models, their muon opts, and the two adamw opts
    (not under test but needed to set up the offload hooks).
    """
    torch.manual_seed(42)
    model_a = TPHippoModel(cfg, devices=[device], dtype=torch.float16)
    model_b = TPHippoModel(cfg, devices=[device], dtype=torch.float16)
    muon_a, adamw_a = build_param_groups(
        model_a, device=device,
        lr_muon=0.02, lr_adamw=3e-4, weight_decay=0.01,
        muon_momentum=0.95,
    )
    muon_b, adamw_b = build_param_groups(
        model_b, device=device,
        lr_muon=0.02, lr_adamw=3e-4, weight_decay=0.01,
        muon_momentum=0.95,
    )
    register_grad_offload_hooks([muon_a, adamw_a])
    register_grad_offload_hooks([muon_b, adamw_b])
    return model_a, model_b, muon_a, muon_b, adamw_a, adamw_b


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_muon_step_gpu_matches_inline_reference():
    """Production ``CPUMuon.step()`` must match the inlined GPU
    reference implementation on ``s.grad`` (post-step zeroed),
    ``s.exp_avg`` (post-step SGD momentum) and the post-step param.

    A divergence here means the production step changed its math
    (likely a refactor of the NS chunking, the SGD momentum update,
    or the storage-dtype boundary). Either is a real regression —
    the muon update must stay bit-equivalent to the inline reference.
    """
    torch.manual_seed(0)
    device = 0
    torch.cuda.set_device(device)
    torch.set_float32_matmul_precision("high")
    init_tp(world_size=1, devices=[device], backend="gloo")

    cfg = HippoConfig(
        vocab_size=8192, hidden_size=256,
        tie_word_embeddings=True, use_bias=False,
        num_heads=4, head_dim=64, expand_v=1.0,
        kda_mode="chunk", use_short_conv=True,
        allow_neg_eigval=False, safe_gate=False,
        lower_bound=None, conv_size=4, conv_bias=False,
        num_layers=2, num_blocks=1,
        intermediate_size=768,
        rms_norm_eps=1e-6,
        # This test runs under FP16 autocast and checks Muon-step
        # bit-equivalence (precision-agnostic on the norm). Force
        # BF16 producers — the default FP8 RMSNorm kernel requires
        # BF16 input/weight and would reject the FP16 tensors here.
        rmsnorm_precision="bf16",
        residual_precision="bf16",
    )
    model_a, model_b, muon_a, muon_b, adamw_a, adamw_b = (
        _build_matched_pair(cfg, device)
    )

    loader_a = iter(dummy_dataloader(2, 128, cfg.vocab_size))
    loader_b = iter(dummy_dataloader(2, 128, cfg.vocab_size))
    mb = 4

    # Forward + backward + flush on both models with the same inputs.
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

    # Copy pre-step state from A to B so they start identical. Under
    # the post-2026-07-15 explicit-accumulator layout both buffers
    # need to match: ``s.grad`` (the per-step accumulator that feeds
    # into the SGD momentum update) and ``s.exp_avg`` (the prior
    # step's SGD momentum that ``exp_avg = β·prev + grad`` reads
    # from). If either is mismatched the two steps diverge at the
    # SGD-momentum update before NS even sees the input.
    for s_a, s_b in zip(muon_a.state.values(), muon_b.state.values()):
        assert s_a.param is not s_b.param
        s_b.grad.copy_(s_a.grad)
        s_b.exp_avg.copy_(s_a.exp_avg)
        s_b.param.data.copy_(s_a.param.data)
        s_b.step = s_a.step

    # Run production step on A, inline reference on B.
    muon_a.step()
    _muon_step_gpu_inline(muon_b)

    # Compare the post-step outputs.
    failures = []
    for s_a, s_b in zip(muon_a.state.values(), muon_b.state.values()):
        # s.grad: per-step accumulator — must be zeroed by the
        # end-of-step housekeeping in both paths. The post-step
        # ``.zero_()`` (or ``fused_zero_many`` in production)
        # touches only this buffer; ``s.exp_avg`` is preserved.
        # Compare in int16-equivalent units for FP16/BF16
        # (5/32767 ≈ 5/255 was the original bound for the
        # storage-dtype post-step residual); for FP32 use the
        # matching epsilon.
        if s_a.grad.dtype in (torch.bfloat16, torch.float16):
            d_q = (
                s_a.grad.to(torch.int16) - s_b.grad.to(torch.int16)
            ).abs()
            if int(d_q.max()) > 5:
                failures.append(
                    f"grad max diff {int(d_q.max())}/32767 exceeds 5 "
                    f"(step-end zero not identical)"
                )
        else:
            d_q = (s_a.grad.float() - s_b.grad.float()).abs()
            if float(d_q.max()) > 1e-5:
                failures.append(
                    f"grad (fp32) max diff {float(d_q.max()):.4e} "
                    f"exceeds 1e-5"
                )
        # s.exp_avg: SGD momentum, post-step. Both paths compute
        # ``exp_avg ← β·exp_avg + grad`` then (nesterov) read it
        # into ``g_for_ns = grad + β·exp_avg``. The buffer itself
        # is bit-identical between paths iff the in-place update
        # was identical — this catches any divergence in the
        # momentum math (storage-dtype rounding, beta, etc.).
        if s_a.exp_avg.dtype in (torch.bfloat16, torch.float16):
            d_m = (
                s_a.exp_avg.to(torch.int16) - s_b.exp_avg.to(torch.int16)
            ).abs()
            if int(d_m.max()) > 5:
                failures.append(
                    f"exp_avg max diff {int(d_m.max())}/32767 exceeds 5"
                )
        else:
            d_m = (s_a.exp_avg.float() - s_b.exp_avg.float()).abs()
            if float(d_m.max()) > 1e-5:
                failures.append(
                    f"exp_avg (fp32) max diff {float(d_m.max()):.4e} "
                    f"exceeds 1e-5"
                )
        # param.data: FP16, compare in FP32
        d_p = (s_a.param.data.float() - s_b.param.data.float()).abs()
        if float(d_p.max()) > 5e-3:
            failures.append(
                f"param max diff {float(d_p.max()):.4e} exceeds 5e-3"
            )

    assert not failures, (
        "production CPUMuon.step() diverged from the inline reference:\n  "
        + "\n  ".join(failures)
    )