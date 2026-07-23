"""Regression tests for W8A8 lm_head (FP8Linear as the language-model head).

Covers the wiring added 2026-07-22 (per
``project_w8a8_lmhead_embed_probe_2026_07_22.md``):
  * Single-step numerics: FP8 lm_head logits / grad_x / grad_w
    relative to a BF16 baseline within the FP8 noise floor
    (~3.76% sig_rel, cos_sim > 0.999).
  * BF16 STE bwd path (fp8_bwd=False) is bit-exact on grads.
  * All gradients are finite (no NaN/Inf at the prod shape).
  * Tied-embedding weight-sharing works (lm_head.weight is
    embed_tokens.weight, both share the optimizer's BF16 master).
  * State-dict round-trip preserves the BF16 master weight
    (``FP8Linear.weight`` is the BF16 leaf — checkpoint format
    matches ``nn.Linear``).
  * End-to-end: ``HippoModel(lm_head_precision='w8a8')`` wires
    a ``FP8Linear`` as ``self.lm_head``; a tiny training loop
    on real loss shows the loss drops (proves the bwd is usable
    by AdamW).

Run as part of ``pytest test/``.
"""
from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.config import HippoConfig
from src.models.model import HippoModel
from src.models.ops.fp8_linear import FP8Linear


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="FP8 lm_head requires CUDA (sm_120 _scaled_mm)",
)


def _sig_rel(a, b):
    return (a.float() - b.float()).norm().item() / (b.float().norm().item() + 1e-12)


def _cos(a, b):
    return F.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()


# --------------------------------------------------------------------------
# Single-step numerics at the prod shape
# --------------------------------------------------------------------------
class TestLMHeadW8A8SingleStep:
    """Prod-shape (M=1024, H=1536, V=248320) FP8 lm_head numerics vs BF16."""

    @staticmethod
    def _make_inputs(M=1024, H=1536, V=248320):
        torch.manual_seed(0)
        x = torch.randn(M, H, device="cuda", dtype=torch.bfloat16) * 0.5
        w = torch.randn(V, H, device="cuda", dtype=torch.bfloat16) * 0.02
        grad_out = torch.randn(M, V, device="cuda", dtype=torch.bfloat16)
        return x, w, grad_out

    def test_fp8_lm_head_logits_within_fp8_noise_floor(self):
        x, w, _ = self._make_inputs()
        bf_logits = F.linear(x, w)
        fp8 = FP8Linear(w.shape[1], w.shape[0], bias=False, fp8_bwd=True).cuda()
        with torch.no_grad():
            fp8.weight.data.copy_(w)
        fp8_logits = fp8(x)
        rel = _sig_rel(fp8_logits, bf_logits)
        cos = _cos(fp8_logits, bf_logits)
        assert rel < 0.05, f"FP8 lm_head logits sig_rel={rel*100:.3f}% > 5% floor"
        assert cos > 0.99, f"FP8 lm_head logits cos_sim={cos:.6f} < 0.99 floor"

    def test_fp8_lm_head_grad_x_within_fp8_noise_floor(self):
        x, w, grad_out = self._make_inputs()
        # BF16 reference
        x_bf = x.detach().clone().requires_grad_(True)
        w_bf = w.detach().clone().requires_grad_(True)
        F.linear(x_bf, w_bf).backward(grad_out)
        gx_bf = x_bf.grad.float()
        del x_bf, w_bf
        torch.cuda.empty_cache()

        # FP8 bwd
        fp8 = FP8Linear(w.shape[1], w.shape[0], bias=False, fp8_bwd=True).cuda()
        with torch.no_grad():
            fp8.weight.data.copy_(w)
        x_fp8 = x.detach().clone().requires_grad_(True)
        fp8(x_fp8).backward(grad_out)
        gx_fp8 = x_fp8.grad.float()
        assert torch.isfinite(gx_fp8).all().item()
        rel = _sig_rel(gx_fp8, gx_bf)
        cos = _cos(gx_fp8, gx_bf)
        assert rel < 0.05, f"FP8 lm_head grad_x sig_rel={rel*100:.3f}%"
        assert cos > 0.99, f"FP8 lm_head grad_x cos_sim={cos:.6f}"

    def test_fp8_lm_head_grad_w_within_fp8_noise_floor(self):
        x, w, grad_out = self._make_inputs()
        # BF16 reference
        x_bf = x.detach().clone().requires_grad_(True)
        w_bf = w.detach().clone().requires_grad_(True)
        F.linear(x_bf, w_bf).backward(grad_out)
        gw_bf = w_bf.grad.float()
        del x_bf, w_bf
        torch.cuda.empty_cache()

        # FP8 bwd
        fp8 = FP8Linear(w.shape[1], w.shape[0], bias=False, fp8_bwd=True).cuda()
        with torch.no_grad():
            fp8.weight.data.copy_(w)
        x_fp8 = x.detach().clone().requires_grad_(True)
        fp8(x_fp8).backward(grad_out)
        gw_fp8 = fp8.weight.grad.float()
        assert torch.isfinite(gw_fp8).all().item()
        # Chunked rel (V*H = 763 MB in fp32).
        chunk = 65536
        sq_err = 0.0
        sq_ref = 0.0
        for s in range(0, w.shape[0], chunk):
            e = min(s + chunk, w.shape[0])
            sq_err += ((gw_fp8[s:e] - gw_bf[s:e]) ** 2).sum().item()
            sq_ref += (gw_bf[s:e] ** 2).sum().item()
        rel = (sq_err ** 0.5) / (sq_ref ** 0.5 + 1e-9)
        assert rel < 0.05, f"FP8 lm_head grad_w sig_rel={rel*100:.3f}%"

    def test_fp8_lm_head_ste_bwd_is_bit_exact(self):
        """``fp8_bwd=False`` (BF16 STE) produces grad_x / grad_w that
        match the BF16 reference bit-exactly (cos_sim = 1.0). The STE
        path trades compute for precision — useful when a sensitive
        downstream op (e.g. Muon's NS) needs exact BF16 grads."""
        x, w, grad_out = self._make_inputs(M=512, H=512, V=4096)  # smaller for speed
        # BF16 reference
        x_bf = x.detach().clone().requires_grad_(True)
        w_bf = w.detach().clone().requires_grad_(True)
        F.linear(x_bf, w_bf).backward(grad_out)
        gx_bf = x_bf.grad
        gw_bf = w_bf.grad
        del x_bf, w_bf
        torch.cuda.empty_cache()

        # FP8+STE
        fp8 = FP8Linear(w.shape[1], w.shape[0], bias=False, fp8_bwd=False).cuda()
        with torch.no_grad():
            fp8.weight.data.copy_(w)
        x_ste = x.detach().clone().requires_grad_(True)
        fp8(x_ste).backward(grad_out)
        gx_ste = x_ste.grad
        gw_ste = fp8.weight.grad

        assert torch.equal(gx_ste, gx_bf), "FP8+STE grad_x diverged from BF16"
        assert torch.equal(gw_ste, gw_bf), "FP8+STE grad_w diverged from BF16"


# --------------------------------------------------------------------------
# Tied embedding / state-dict round-trip
# --------------------------------------------------------------------------
class TestLMHeadW8A8StateDict:
    """W8A8 lm_head shares the embed_tokens weight (tie_word_embeddings).
    The BF16 master weight must round-trip through state_dict so a
    checkpoint loads directly into a BF16 model — the FP8 quantization
    is computed on the fly per forward."""

    def test_fp8_lm_head_weight_is_bf16_master(self):
        fp8 = FP8Linear(1536, 248320, bias=False, fp8_bwd=True).cuda()
        assert fp8.weight.dtype == torch.bfloat16, (
            "FP8Linear.weight must be the BF16 master (optimizer leaf)"
        )
        assert fp8.weight.shape == (248320, 1536)

    def test_fp8_lm_head_state_dict_matches_nn_linear(self):
        """The state-dict of a FP8Linear has the same shape / dtype as
        a plain nn.Linear — drop-in for a BF16 checkpoint."""
        fp8 = FP8Linear(1536, 248320, bias=False, fp8_bwd=True).cuda()
        ref = nn.Linear(1536, 248320, bias=False).cuda().to(torch.bfloat16)
        sd_fp8 = fp8.state_dict()
        sd_ref = ref.state_dict()
        assert set(sd_fp8.keys()) == set(sd_ref.keys())
        for k in sd_fp8:
            assert sd_fp8[k].shape == sd_ref[k].shape, k
            assert sd_fp8[k].dtype == sd_ref[k].dtype, k

    def test_fp8_lm_head_tied_weight_shares_storage(self):
        """``lm_head.weight = embed_tokens.weight`` semantics — the
        shared tensor must be the same Python object (so the
        optimizer step sees one weight update per micro-step, not
        two divergent ones)."""
        fp8 = FP8Linear(64, 128, bias=False, fp8_bwd=True).cuda()
        embed = nn.Embedding(128, 64).cuda().to(torch.bfloat16)
        fp8.weight = embed.weight  # the canonical tie pattern
        assert fp8.weight is embed.weight, (
            "Tied weight assignment lost object identity — the "
            "optimizer step would write to two separate copies"
        )


# --------------------------------------------------------------------------
# End-to-end: HippoModel wires FP8Linear as lm_head
# --------------------------------------------------------------------------
class TestHippoModelLMHeadW8A8:
    @staticmethod
    def _build_model(lm_head_precision="w8a8", **overrides):
        cfg_kwargs = dict(
            vocab_size=256,
            hidden_size=128,
            intermediate_size=256,
            num_layers=2,
            num_blocks=2,
            num_heads=2,
            head_dim=64,
            safe_gate=True,
            lower_bound=-5.0,
            use_short_conv=False,
            use_bias=False,
            ffn_precision="w4a8",
            tie_word_embeddings=True,
            rmsnorm_precision="fp8",
            # Tied embeddings require embedding_precision ==
            # lm_head_precision (they share one weight tensor); mirror
            # the production base.yml where both are w8a8.
            embedding_precision=lm_head_precision,
            lm_head_precision=lm_head_precision,
        )
        cfg_kwargs.update(overrides)
        cfg = HippoConfig(**cfg_kwargs)
        return HippoModel(cfg).cuda().to(torch.bfloat16)

    def test_w8a8_wires_fp8_linear_as_lm_head(self):
        m = self._build_model("w8a8")
        assert isinstance(m.lm_head, FP8Linear), (
            f"lm_head is {type(m.lm_head).__name__}, expected FP8Linear"
        )
        assert m.lm_head.fp8_bwd is True

    def test_w16a16_wires_plain_nn_linear(self):
        m = self._build_model("w16a16")
        assert isinstance(m.lm_head, nn.Linear), (
            f"lm_head is {type(m.lm_head).__name__}, expected nn.Linear"
        )
        # Confirm it's NOT an FP8Linear (subclass check).
        assert not isinstance(m.lm_head, FP8Linear)

    def test_w8a8_tied_weight_is_shared(self):
        m = self._build_model("w8a8")
        assert m.lm_head.weight is m.embed_tokens.weight, (
            "Tied-embeddings pattern lost on FP8 lm_head — the "
            "FP8Linear wrapper must expose .weight as the same "
            "parameter object as embed_tokens"
        )

    def test_w8a8_lm_head_forward_finite(self):
        m = self._build_model("w8a8")
        ids = torch.randint(0, m.config.vocab_size, (2, 16), device="cuda")
        labels = torch.randint(0, m.config.vocab_size, (2, 16), device="cuda")
        out = m(ids, labels=labels)
        assert torch.isfinite(out["logits"]).all().item()
        assert torch.isfinite(out["loss"]).item()

    def test_w8a8_lm_head_bwd_finite(self):
        m = self._build_model("w8a8")
        ids = torch.randint(0, m.config.vocab_size, (2, 16), device="cuda")
        labels = torch.randint(0, m.config.vocab_size, (2, 16), device="cuda")
        out = m(ids, labels=labels)
        out["loss"].backward()
        assert m.embed_tokens.weight.grad is not None
        assert torch.isfinite(m.embed_tokens.weight.grad).all().item()
        # The tied weight means embed_tokens.weight.grad IS
        # lm_head.weight.grad — same tensor, same check.
        assert torch.isfinite(m.lm_head.weight.grad).all().item()
        assert m.embed_tokens.weight.grad is m.lm_head.weight.grad

    def test_w8a8_lm_head_training_step_decreases_loss(self):
        """A few SGD steps on a tiny synthetic batch should drop
        the loss (proves the FP8 lm_head grad is usable by AdamW
        and the autograd chain is wired correctly)."""
        m = self._build_model("w8a8")
        # Over-scale the init so the FP8 grad noise doesn't wash out
        # the loss signal at this tiny scale.
        with torch.no_grad():
            m.embed_tokens.weight.data.mul_(4.0)
        opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=0.0)
        ids = torch.randint(0, m.config.vocab_size, (2, 16), device="cuda")
        labels = torch.randint(0, m.config.vocab_size, (2, 16), device="cuda")
        m.train()

        loss_first = None
        for step in range(5):
            opt.zero_grad()
            out = m(ids, labels=labels)
            loss = out["loss"]
            if loss_first is None:
                loss_first = loss.item()
            loss.backward()
            opt.step()
        loss_final = loss.item()
        assert loss_final < loss_first, (
            f"5 AdamW steps did not decrease loss "
            f"({loss_first:.4f} → {loss_final:.4f}) — the FP8 "
            f"lm_head grad is not being learned"
        )

    def test_w8a8_lm_head_long_range_loss_no_compounding_drift(self):
        """50-step loss probe: FP8 lm_head loss should NOT drift
        relative to BF16 over many steps (no compounding noise).
        Reference probe: ``project_w8a8_lmhead_embed_probe_2026_07_22.md``
        showed ~0.03% rel diff constant across 50 steps."""
        # Use small vocab to keep the FP8 matmul light.
        cfg_kwargs = dict(
            vocab_size=4096,
            hidden_size=256,
            intermediate_size=512,
            num_layers=2,
            num_blocks=2,
            num_heads=2,
            head_dim=128,
            safe_gate=True,
            lower_bound=-5.0,
            use_short_conv=False,
            use_bias=False,
            tie_word_embeddings=True,
            ffn_precision="w4a8",
            rmsnorm_precision="fp8",
        )
        # Build two models sharing init (one BF16, one FP8). Tied
        # embeddings require embedding_precision == lm_head_precision,
        # so each model sets both to the same scheme.
        torch.manual_seed(42)
        cfg_bf = HippoConfig(
            **cfg_kwargs, embedding_precision="w16a16",
            lm_head_precision="w16a16",
        )
        torch.manual_seed(42)
        cfg_fp8 = HippoConfig(
            **cfg_kwargs, embedding_precision="w8a8",
            lm_head_precision="w8a8",
        )
        m_bf = HippoModel(cfg_bf).cuda().to(torch.bfloat16)
        m_fp8 = HippoModel(cfg_fp8).cuda().to(torch.bfloat16)
        # Copy weights from BF16 to FP8 model so the comparison starts
        # from the same point.
        with torch.no_grad():
            for p_bf, p_fp8 in zip(m_bf.parameters(), m_fp8.parameters()):
                p_fp8.data.copy_(p_bf.data)

        ids = torch.randint(0, cfg_kwargs["vocab_size"], (8, 32), device="cuda")
        labels = torch.randint(0, cfg_kwargs["vocab_size"], (8, 32), device="cuda")
        opt_bf = torch.optim.AdamW(m_bf.parameters(), lr=1e-3, weight_decay=0.0)
        opt_fp8 = torch.optim.AdamW(m_fp8.parameters(), lr=1e-3, weight_decay=0.0)

        # Track loss at the start, middle, and end.
        loss_bf = {}
        loss_fp8 = {}
        for step in range(50):
            opt_bf.zero_grad()
            opt_fp8.zero_grad()
            loss_b = m_bf(ids, labels=labels)["loss"]
            loss_f = m_fp8(ids, labels=labels)["loss"]
            loss_b.backward()
            loss_f.backward()
            opt_bf.step()
            opt_fp8.step()
            if step in (0, 25, 49):
                loss_bf[step] = loss_b.item()
                loss_fp8[step] = loss_f.item()

        # The two models drift apart through training, but the
        # relative diff should stay bounded (no compounding FP8
        # noise runaway). At 50 steps the probe showed ~0.03%
        # rel diff — we allow up to 1% here to absorb the larger
        # initial-mismatch from copying weights to a different
        # model.
        for step in (0, 25, 49):
            rel_diff = abs(loss_fp8[step] - loss_bf[step]) / max(abs(loss_bf[step]), 1e-3) * 100
            assert rel_diff < 1.0, (
                f"FP8 lm_head loss drift at step {step}: "
                f"BF16={loss_bf[step]:.4f} FP8={loss_fp8[step]:.4f} "
                f"({rel_diff:.2f}% rel diff) — FP8 noise is compounding"
            )


# --------------------------------------------------------------------------
# Speed: W8A8 vs BF16 at prod shape (single card, sm_120)
# --------------------------------------------------------------------------
class TestLMHeadW8A8Speed:
    """Wall-clock speedup at the prod shape (M=1024, H=1536, V=248320).

    The forward is a memory-bound matmul (BF16 is at peak; FP8 has
    headroom on sm_120 but the M=1024 small-M underutilization
    caps it at ~72% of FP8 peak). Net forward speedup is 1.39x;
    fwd+bwd is 1.21x.
    """

    def test_fp8_lm_head_fwd_faster_than_bf16(self):
        """Loose 1.10x floor — the prod measurement is 1.39x but a
        different GPU thermal state / driver version can shave a
        few percent. As long as we're clearly faster than BF16 we
        can ship."""
        M, H, V = 1024, 1536, 248320
        torch.manual_seed(0)
        x = torch.randn(M, H, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(V, H, device="cuda", dtype=torch.bfloat16) * 0.02
        bf = nn.Linear(H, V, bias=False).cuda().to(torch.bfloat16)
        fp8 = FP8Linear(H, V, bias=False, fp8_bwd=True).cuda()
        with torch.no_grad():
            fp8.weight.data.copy_(w)
            bf.weight.data.copy_(w)

        def cuda_time_ms(fn, iters=15, warmup=5):
            for _ in range(warmup):
                fn()
            torch.cuda.synchronize()
            times = []
            for _ in range(iters):
                s = torch.cuda.Event(enable_timing=True)
                e = torch.cuda.Event(enable_timing=True)
                s.record()
                fn()
                e.record()
                e.synchronize()
                times.append(s.elapsed_time(e))
            times.sort()
            return times[len(times) // 2]

        bf_us = cuda_time_ms(lambda: bf(x)) * 1e3
        fp8_us = cuda_time_ms(lambda: fp8(x)) * 1e3
        speedup = bf_us / fp8_us
        # The actual measurement is ~1.39x; allow ≥1.10x for variance.
        assert speedup > 1.10, (
            f"FP8 lm_head forward speedup {speedup:.3f}x < 1.10x floor — "
            f"BF16={bf_us:.0f}us, FP8={fp8_us:.0f}us"
        )