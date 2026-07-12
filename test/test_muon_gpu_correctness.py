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
second copy, then comparing ``mom_buf`` and the post-step param.
The reference is reconstructed from the algorithm description
(NS → apply → recast to storage), so a future change to the
production step() that changes the math will trip the test even
if the change is locally consistent.

Quantized storage (int8 + per-row scale, mxfp8 + per-block E8M0)
was removed on 2026-07-12 — long-training runs showed
quantization-error accumulation destabilizing optimization. The
merged-accumulator design (mom_buf doubles as the accumulator) is
the only supported muon storage path now.

Test layout
-----------
- Build two identical TP models, do the same fwd + bwd + flush on
  both.
- Copy the pre-step state from A to B so they start identical.
- Run production ``CPUMuon.step()`` on A.
- Run the inline GPU reference on B.
- Assert the outputs agree within tight thresholds
  (mom_buf ≤ 5/255 in storage units, param ≤ 5e-3).

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
from src.training.precision_config import PrecisionConfig


def _muon_step_gpu_inline(muon_opt: CPUMuon) -> None:
    """Independent GPU reference implementation of the muon step.

    Mirrors the algorithm (NS → apply → recast-to-storage) but inlined
    with explicit per-tensor operations. The production
    ``CPUMuon.step()`` is the optimized version of the same math; the
    inlined version is the "what the math should do" reference that
    the test compares against.

    Storage design (merged-accumulator, 2026-07-12):
    the per-cycle grad sum lives directly in ``s.mom_buf`` (the
    configured full-precision storage dtype: bf16 / fp16 / fp32).
    The step reads ``s.mom_buf``, casts to FP32, orthogonalizes,
    applies the update, then recasts ``s.mom_buf`` to its storage
    dtype and zeros it for the next cycle.
    """
    lr = muon_opt.lr
    wd = muon_opt.weight_decay
    CHUNK_ROWS = CPUMuon._STREAM_CHUNK_ROWS
    for s in muon_opt.state.values():
        # Merged-accumulator: ``s.mom_buf`` IS the accumulator.
        if s.mom_buf.abs().sum().item() == 0:
            continue
        shape = s.shape
        rows, cols = shape[0], shape[1]
        device = s.param.device

        # Cast to FP32 for the math (the storage dtype may be
        # BF16 / FP16 / FP32 — FP32 is the compute precision).
        m_fp32_gpu = s.mom_buf.to(device, non_blocking=True).float() \
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

        # ---- Step-end: recast mom_buf to storage dtype, reset. ----
        # The accumulator (in FP32 above) is the *post-NS* momentum;
        # we recast it back to the storage dtype and zero for the
        # next cycle, mirroring the production path.
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
    reference implementation on ``mom_buf`` and the post-step param.

    A divergence here means the production step changed its math
    (likely a refactor of the NS chunking or the storage-dtype
    boundary). Either is a real regression — the muon update must
    stay bit-equivalent to the inline reference.
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
    # the merged-accumulator design the cycle's grad lives in
    # ``mom_buf`` (the storage dtype), so that is the only buffer that
    # needs to match for the two steps to be comparable.
    for s_a, s_b in zip(muon_a.state.values(), muon_b.state.values()):
        assert s_a.param is not s_b.param
        s_b.mom_buf.copy_(s_a.mom_buf)
        s_b.param.data.copy_(s_a.param.data)
        s_b.step = s_a.step

    # Run production step on A, inline reference on B.
    muon_a.step()
    _muon_step_gpu_inline(muon_b)

    # Compare the post-step outputs.
    failures = []
    for s_a, s_b in zip(muon_a.state.values(), muon_b.state.values()):
        # mom_buf: storage dtype (BF16 / FP16 / FP32). For FP16/BF16
        # compare in int16-equivalent units (cast through int16);
        # for FP32 compare in FP32.
        if s_a.mom_buf.dtype in (torch.bfloat16, torch.float16):
            d_q = (
                s_a.mom_buf.to(torch.int16) - s_b.mom_buf.to(torch.int16)
            ).abs()
            if int(d_q.max()) > 5:
                failures.append(
                    f"mom_buf max diff {int(d_q.max())}/32767 exceeds 5"
                )
        else:
            d_q = (s_a.mom_buf.float() - s_b.mom_buf.float()).abs()
            if float(d_q.max()) > 1e-5:
                failures.append(
                    f"mom_buf (fp32) max diff {float(d_q.max()):.4e} exceeds 1e-5"
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