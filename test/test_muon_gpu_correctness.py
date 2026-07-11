"""Regression test: production CPUMuon.step() matches an independent
inline GPU reference implementation.

Historical context
------------------
``e513852 feat(offload): CPUMuon GPU step + bf16/fp16/fp32 momentum
storage`` moved the muon step's dequant / SGD / requant cycle from CPU
DRAM to GPU HBM. The optimization had to be **bit-equivalent** to the
original CPU path within FP16 / int8 quantization noise — losing even
a few percent on this test would mean the production optimizer was
silently producing different (and unverified) updates.

The test pins that contract by running the production step() on one
copy of a model and an inlined, hand-written GPU reference on a
second copy, then comparing mom_buf, mom_scale, and the post-step
param. The reference is reconstructed from the algorithm
description (dequant → requant → NS → apply), so a future change to
the production step() that changes the math will trip the test
even if the change is locally consistent.

Test layout
-----------
- Build two identical TP models, do the same fwd + bwd + flush on
  both.
- Copy the pre-step state from A to B so they start identical.
- Run production ``CPUMuon.step()`` on A.
- Run the inline GPU reference on B.
- Assert the three outputs agree within tight thresholds
  (mom_buf ≤ 5/255, mom_scale ≤ 1e-3, param ≤ 5e-3).

The thresholds match the original (pre-conversion) values; tightening
any of them would catch a real divergence.
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
from src.training.precision_config import PrecisionConfig


def _muon_step_gpu_inline(muon_opt: CPUMuon) -> None:
    """Independent GPU reference implementation of the muon step.

    Mirrors the algorithm (dequant → NS → apply → requant) but inlined
    with explicit per-tensor operations. The production ``CPUMuon.step()``
    went through a sequence of optimizations after this test was
    written; the inlined version is the "what the math should do"
    reference that the test compares against.

    Storage design (production default = int8 + separate bf16 accum):
    the per-cycle grad sum lives in ``s.accum`` (bf16, no scale), NOT
    ``s.mom_buf`` (which is zero at step start). The step reads
    ``s.accum``, casts to FP32, orthogonalizes, applies the update,
    then requantizes ``s.accum`` into ``s.mom_buf`` / ``s.mom_scale``
    (per-row int8) — mirroring ``_quantize_accum_to_mom_buf`` — and
    zeros ``s.accum`` for the next cycle.

    For fp* muon (``s.accum is None``) the merged-accumulator design
    applies: ``s.mom_buf`` doubles as the accumulator, so the step
    reads it directly, recasts to storage dtype, and zeros it.
    """
    lr = muon_opt.lr
    wd = muon_opt.weight_decay
    CHUNK_ROWS = CPUMuon._STREAM_CHUNK_ROWS
    for s in muon_opt.state.values():
        # int8 muon: cycle grad is in ``s.accum``. fp* muon:
        # ``s.mom_buf`` is the merged accumulator.
        use_separate_accum = s.accum is not None
        g_buf = s.accum if use_separate_accum else s.mom_buf
        if g_buf.abs().sum().item() == 0:
            continue
        shape = s.shape
        rows, cols = shape[0], shape[1]
        device = s.param.device

        # ---- Dequantize the cycle's accumulated grad to FP32. ----
        # Both paths store the accumulator as a raw (unscaled) buffer:
        # ``accum`` is bf16, fp* ``mom_buf`` is its storage dtype.
        m_fp32_gpu = g_buf.to(device, non_blocking=True).float() \
                                    .view(rows, cols)

        # ---- Decoupled weight decay on the full param. ----
        if wd != 0.0:
            s.param.data.mul_(1.0 - lr * wd)

        # ---- Stream NS over rows and apply the update. ----
        m_fp16 = m_fp32_gpu.to(torch.float16)
        for r_start in range(0, rows, CHUNK_ROWS):
            r_end = min(r_start + CHUNK_ROWS, rows)
            m_chunk = m_fp16[r_start:r_end]
            update = muon_opt._newton_schulz(m_chunk)
            update = update.to(s.param.dtype)
            s.param.data[r_start:r_end].add_(update, alpha=-lr)

        torch.cuda.current_stream(device).synchronize()

        # ---- Step-end requantize + cycle reset. ----
        if use_separate_accum:
            # int8 per-row requant of accum into mom_buf / mom_scale
            # (mirrors _quantize_accum_to_mom_buf), then zero accum.
            row_max = m_fp32_gpu.abs().amax(dim=1).clamp(min=1e-8)
            new_scale_fp32 = row_max / 127.0
            new_scale_bf16 = new_scale_fp32.to(torch.bfloat16)
            new_scale_2d = new_scale_bf16.float().unsqueeze(1)
            q_int8_gpu = (m_fp32_gpu / new_scale_2d).round() \
                            .clamp(-128, 127).to(torch.int8)
            s.mom_buf.copy_(q_int8_gpu.view(-1), non_blocking=True)
            s.mom_scale.copy_(new_scale_bf16, non_blocking=True)
            s.accum.zero_()
        else:
            # fp* merged accumulator: recast to storage dtype, reset.
            s.mom_buf.copy_(
                m_fp32_gpu.to(s.mom_buf.dtype).view(-1), non_blocking=True
            )
            s.mom_buf.zero_()


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
    precision = PrecisionConfig()
    muon_a, adamw_a = build_param_groups(
        model_a, device=device,
        lr_muon=0.02, lr_adamw=3e-4, weight_decay=0.01,
        muon_momentum=0.95, precision=precision,
    )
    muon_b, adamw_b = build_param_groups(
        model_b, device=device,
        lr_muon=0.02, lr_adamw=3e-4, weight_decay=0.01,
        muon_momentum=0.95, precision=precision,
    )
    register_grad_offload_hooks([muon_a, adamw_a])
    register_grad_offload_hooks([muon_b, adamw_b])
    return model_a, model_b, muon_a, muon_b, adamw_a, adamw_b


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_muon_step_gpu_matches_inline_reference():
    """Production ``CPUMuon.step()`` must match the inlined GPU
    reference implementation on mom_buf, mom_scale, and post-step
    param.

    A divergence here means the production step changed its math
    (likely a refactor of the dequant/requant boundary or the NS
    chunking). Either is a real regression — the muon update must
    stay bit-equivalent to the inline reference, or the W4A16
    100-step sweep that validated the W4A16 path is invalidated.
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
        num_layers=2, num_blocks=2,
        intermediate_size=768,
        rms_norm_eps=1e-6,
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
    # the separate-accum design the cycle's grad lives in ``accum``
    # (bf16) for int8 muon — ``mom_buf`` is zero at step start — so
    # ``accum`` is the buffer that must match for the two steps to be
    # comparable. Copy it too (when present) alongside mom_buf/scale.
    for s_a, s_b in zip(muon_a.state.values(), muon_b.state.values()):
        assert s_a.param is not s_b.param
        s_b.mom_buf.copy_(s_a.mom_buf)
        s_b.mom_scale.copy_(s_a.mom_scale)
        if s_a.accum is not None:
            s_b.accum.copy_(s_a.accum)
        s_b.param.data.copy_(s_a.param.data)
        s_b.step = s_a.step

    # Run production step on A, inline reference on B.
    muon_a.step()
    _muon_step_gpu_inline(muon_b)

    # Compare the three post-step outputs.
    failures = []
    for s_a, s_b in zip(muon_a.state.values(), muon_b.state.values()):
        # mom_buf: int8, diff in [-255, 255]
        d_q = (s_a.mom_buf.to(torch.int32) - s_b.mom_buf.to(torch.int32)).abs()
        if int(d_q.max()) > 5:
            failures.append(
                f"mom_buf max diff {int(d_q.max())}/255 exceeds 5"
            )
        # mom_scale: BF16, compare in FP32
        d_s = (s_a.mom_scale.float() - s_b.mom_scale.float()).abs()
        if float(d_s.max()) > 1e-3:
            failures.append(
                f"mom_scale max diff {float(d_s.max()):.4e} exceeds 1e-3"
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
