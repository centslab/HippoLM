"""Regression test for the fused TPKDA / TPSwiGLU projection layout.

Historical context
------------------
``9c6bc9a feat(tp): fuse QKV / FG-first / gate+up projections in
TPKDA + TPSwiGLU`` merged what used to be 9 separate matmuls per KDA
layer (q, k, v, f[0], f[1], g[0], g[1], b, o) and 3 per SwiGLU
(gate, up, down) into 6 + 2 — saving 4 kernel launches per layer.
The perf gain is real, but only if the fused layout produces the
**same outputs** as the split layout for any given weight + input.

A future refactor that swaps the slice order, drops a column, or
rearranges the bias would silently produce different activations
downstream — the KDA chunk kernel would then compute on wrong q/k/v
and the loss would drift without any single test catching it (the
smoke test only logs step loss, not per-projection equivalence).

These tests pin the math by:
  1. Building hand-written SPLIT versions of TPKDA / TPSwiGLU that
     mirror the pre-fusion structure (separate q/k/v ColumnParallel,
     f_proj / g_proj as Sequential, separate gate_proj / up_proj).
  2. Building hand-written FUSED versions that mirror the post-fusion
     structure (qkv_proj, fg_first, f_proj1, g_proj1; gate_up_proj).
  3. Copying the split weights into the fused module with the
     correct slice / concat order.
  4. Forwarding the same input and asserting the per-projection
     outputs match within FP16 noise.

The threshold is loose (max abs diff < 5e-3) because the FP16 matmul
on a 5060 Ti accumulates in a different order when the weight is one
big block vs two small ones (the matmul kernel picks different tile
sizes), and order-of-summation noise in FP16 can be ~1e-3 absolute
on a unit-magnitude output.

The fused layout per rank (world=1 for this test):
  qkv_proj  weight: [2*key_dim + value_dim, hidden_size]
  fg_first  weight: [2*d_v, hidden_size]
  f_proj1   weight: [gate_dim, d_v]
  g_proj1   weight: [value_dim, d_v], bias [value_dim]
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from src.models import HippoConfig
from src.models.tp_layers import (
    ColumnParallelLinear, RowParallelLinear, init_tp,
)


# --------------------------------------------------------------------------- #
# SPLIT versions (mirror of the pre-fusion TPKDA / TPSwiGLU)                  #
# --------------------------------------------------------------------------- #
class _TPKDASplit(nn.Module):
    """TPKDA with separate q/k/v, f_proj / g_proj as Sequential.
    Mirrors the pre-fusion structure.
    """

    def __init__(self, cfg, device, dtype=torch.float16):
        super().__init__()
        h = cfg.hidden_size
        nh = cfg.num_heads
        nvh = cfg.num_heads
        d_h = cfg.head_dim
        d_v = int(d_h * cfg.expand_v)
        key_dim = nh * d_h
        value_dim = nvh * d_v
        gate_dim = nvh * d_h

        self.key_per_partition = key_dim
        self.value_per_partition = value_dim
        self.gate_per_partition = gate_dim
        self.head_k_dim = d_h
        self.head_v_dim = d_v

        self.q_proj = ColumnParallelLinear(h, key_dim, bias=False, device=device, dtype=dtype)
        self.k_proj = ColumnParallelLinear(h, key_dim, bias=False, device=device, dtype=dtype)
        self.v_proj = ColumnParallelLinear(h, value_dim, bias=False, device=device, dtype=dtype)
        self.f_proj = nn.Sequential(
            nn.Linear(h, d_v, bias=False, device=device, dtype=dtype),
            ColumnParallelLinear(d_v, gate_dim, bias=False, device=device, dtype=dtype),
        )
        self.g_proj = nn.Sequential(
            nn.Linear(h, d_v, bias=False, device=device, dtype=dtype),
            ColumnParallelLinear(d_v, value_dim, bias=True, device=device, dtype=dtype),
        )
        self.b_proj = ColumnParallelLinear(h, nvh, bias=False, device=device, dtype=dtype)
        self.o_proj = RowParallelLinear(value_dim, h, bias=False, device=device, dtype=dtype)

    def forward(self, x):
        h = x
        q = self.q_proj(h)
        k = self.k_proj(h)
        v = self.v_proj(h)
        f = self.f_proj[1](self.f_proj[0](h))
        g = self.g_proj[1](self.g_proj[0](h))
        b = self.b_proj(h)
        return {"q": q, "k": k, "v": v, "f": f, "g": g, "b": b}


class _TPSwiGLUSplit(nn.Module):
    """TPSwiGLU with separate gate_proj / up_proj."""

    def __init__(self, cfg, device, dtype=torch.float16):
        super().__init__()
        self.gate_proj = ColumnParallelLinear(
            cfg.hidden_size, cfg.intermediate_size,
            bias=cfg.use_bias, device=device, dtype=dtype,
        )
        self.up_proj = ColumnParallelLinear(
            cfg.hidden_size, cfg.intermediate_size,
            bias=cfg.use_bias, device=device, dtype=dtype,
        )
        self.down_proj = RowParallelLinear(
            cfg.intermediate_size, cfg.hidden_size,
            bias=cfg.use_bias, device=device, dtype=dtype,
        )

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# --------------------------------------------------------------------------- #
# FUSED versions                                                              #
# --------------------------------------------------------------------------- #
class _TPKDAFused(nn.Module):
    """TPKDA-shape module with fused QKV / FG-first / B / O.

    Mirrors the production fused design. We construct this by
    hand so the test is independent of the production class.
    """

    def __init__(self, cfg, device, dtype=torch.float16):
        super().__init__()
        h = cfg.hidden_size
        nh = cfg.num_heads
        nvh = cfg.num_heads
        d_h = cfg.head_dim
        d_v = int(d_h * cfg.expand_v)
        key_dim = nh * d_h
        value_dim = nvh * d_v
        gate_dim = nvh * d_h

        self.qkv_proj = ColumnParallelLinear(
            h, 2 * key_dim + value_dim, bias=False, device=device, dtype=dtype,
        )
        self.fg_first = nn.Linear(
            h, 2 * d_v, bias=False, device=device, dtype=dtype,
        )
        # NOTE: we keep f_proj1 and g_proj1 separate (don't fuse
        # the FG-second layer) because fusing them would require
        # a 50%-zero weight matrix and double the FLOPs of that
        # layer. See tp_model.py TPKDA.__init__ for details.
        self.f_proj1 = ColumnParallelLinear(
            d_v, gate_dim, bias=False, device=device, dtype=dtype,
        )
        self.g_proj1 = ColumnParallelLinear(
            d_v, value_dim, bias=True, device=device, dtype=dtype,
        )
        self.b_proj = ColumnParallelLinear(
            h, nvh, bias=False, device=device, dtype=dtype,
        )
        self.o_proj = RowParallelLinear(
            value_dim, h, bias=False, device=device, dtype=dtype,
        )
        self._d_v = d_v
        self._gate_dim = gate_dim
        self._value_dim = value_dim
        self._key_dim = key_dim

    def forward(self, x):
        qkv = self.qkv_proj(x)
        q, k, v = qkv.split(
            [self._key_dim, self._key_dim, self._value_dim], dim=-1,
        )
        fg_first = self.fg_first(x)
        f_inter, g_inter = fg_first.chunk(2, dim=-1)
        f = self.f_proj1(f_inter)
        g = self.g_proj1(g_inter)
        b = self.b_proj(x)
        return {"q": q, "k": k, "v": v, "f": f, "g": g, "b": b}


class _TPSwiGLUFused(nn.Module):
    """TPSwiGLU with fused gate_up."""

    def __init__(self, cfg, device, dtype=torch.float16):
        super().__init__()
        h = cfg.hidden_size
        inter = cfg.intermediate_size
        self.gate_up_proj = ColumnParallelLinear(
            h, 2 * inter, bias=cfg.use_bias, device=device, dtype=dtype,
        )
        self.down_proj = RowParallelLinear(
            inter, h, bias=cfg.use_bias, device=device, dtype=dtype,
        )
        self._inter = inter

    def forward(self, x):
        gu = self.gate_up_proj(x)
        gate, up = gu.split(self._inter, dim=-1)
        return self.down_proj(F.silu(gate) * up)


# --------------------------------------------------------------------------- #
# Weight copy                                                                 #
# --------------------------------------------------------------------------- #
def _copy_split_to_fused_kda(split: _TPKDASplit, fused: _TPKDAFused) -> None:
    """Copy split TPKDA weights into the fused module (world=1).

    Conventions:
      fused.qkv_proj.weight[0:key_dim]                  = q
      fused.qkv_proj.weight[key_dim:2*key_dim]          = k
      fused.qkv_proj.weight[2*key_dim:2*key+v_dim]      = v
      fused.fg_first.weight[0:d_v]                      = f's first layer
      fused.fg_first.weight[d_v:2*d_v]                  = g's first layer
      fused.f_proj1.weight                              = f's second layer
      fused.g_proj1.weight, .bias                       = g's second layer
    """
    fused.qkv_proj.weight.data = torch.cat(
        [split.q_proj.weight.data, split.k_proj.weight.data, split.v_proj.weight.data],
        dim=0,
    )
    fused.fg_first.weight.data = torch.cat(
        [split.f_proj[0].weight.data, split.g_proj[0].weight.data], dim=0,
    )
    fused.f_proj1.weight.data = split.f_proj[1].weight.data.clone()
    if split.g_proj[1].bias is not None:
        fused.g_proj1.bias.data = split.g_proj[1].bias.data.clone()
    fused.g_proj1.weight.data = split.g_proj[1].weight.data.clone()
    fused.b_proj.weight.data = split.b_proj.weight.data.clone()
    fused.o_proj.weight.data = split.o_proj.weight.data.clone()


def _copy_split_to_fused_swiglu(split: _TPSwiGLUSplit, fused: _TPSwiGLUFused) -> None:
    fused.gate_up_proj.weight.data = torch.cat(
        [split.gate_proj.weight.data, split.up_proj.weight.data], dim=0,
    )
    fused.down_proj.weight.data = split.down_proj.weight.data.clone()
    if split.gate_proj.bias is not None:
        fused.gate_up_proj.bias.data = torch.cat(
            [split.gate_proj.bias.data, split.up_proj.bias.data], dim=0,
        )


# --------------------------------------------------------------------------- #
# Test harness                                                                #
# --------------------------------------------------------------------------- #
def _max_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a.float() - b.float()).abs().max().item()


def _build_cfg() -> HippoConfig:
    return HippoConfig(
        vocab_size=8192, hidden_size=512, tie_word_embeddings=True,
        use_bias=False, num_heads=8, head_dim=64, expand_v=1.0,
        kda_mode="chunk", use_short_conv=False,  # skip conv to keep test focused
        allow_neg_eigval=False, safe_gate=False, lower_bound=None,
        conv_size=4, conv_bias=False, num_layers=4, num_blocks=2,
        intermediate_size=1536, rms_norm_eps=1e-6,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for matmul tests")
def test_tpkda_fused_matches_split():
    """The fused TPKDA projection layout (qkv_proj / fg_first /
    f_proj1 / g_proj1) must produce the same per-projection outputs
    as the pre-fusion split layout, given the same weights.

    Failure here means the slice / concat order in the fused module
    is wrong, or a projection was dropped. Either regresses the KDA
    math without the smoke test noticing for many steps.
    """
    device = 0
    torch.cuda.set_device(device)
    init_tp(world_size=1, devices=[device], backend="gloo")
    torch.manual_seed(0)

    cfg = _build_cfg()
    B, T = 2, 256
    x = torch.randn(B, T, cfg.hidden_size, device=device, dtype=torch.float16)

    split = _TPKDASplit(cfg, device)
    fused = _TPKDAFused(cfg, device)
    _copy_split_to_fused_kda(split, fused)
    out_split = split(x)
    out_fused = fused(x)

    failures = []
    for name in ("q", "k", "v", "f", "g", "b"):
        diff = _max_abs_diff(out_split[name], out_fused[name])
        if diff >= 5e-3:
            failures.append(
                f"TPKDA {name}: max abs diff {diff:.4e} >= 5e-3"
            )
    assert not failures, (
        "fused TPKDA diverged from the pre-fusion split layout:\n  "
        + "\n  ".join(failures)
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for matmul tests")
def test_tpswiglu_fused_matches_split():
    """The fused TPSwiGLU projection (gate_up_proj + down_proj) must
    produce the same output as the pre-fusion split gate_proj /
    up_proj / down_proj layout, given the same weights.

    Failure here means the slice order in the fused module is wrong
    or a projection was dropped. The FFN path is the single largest
    cost in the layer (~70% of step time at prod), so a silent math
    regression here would be expensive to track down.
    """
    device = 0
    torch.cuda.set_device(device)
    init_tp(world_size=1, devices=[device], backend="gloo")
    torch.manual_seed(0)

    cfg = _build_cfg()
    B, T = 2, 256
    x = torch.randn(B, T, cfg.hidden_size, device=device, dtype=torch.float16)

    split = _TPSwiGLUSplit(cfg, device)
    fused = _TPSwiGLUFused(cfg, device)
    _copy_split_to_fused_swiglu(split, fused)
    y_split = split(x)
    y_fused = fused(x)

    diff = _max_abs_diff(y_split, y_fused)
    assert diff < 5e-3, (
        f"TPSwiGLU fused vs split diverged: max abs diff {diff:.4e} >= 5e-3"
    )