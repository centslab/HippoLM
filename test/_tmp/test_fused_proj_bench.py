"""Benchmark for fused vs split projections.

Measures forward+backward time of:
  - TPKDA (with KDA chunk kernel)
  - TPSwiGLU (FFN)
  - one TPHippoLayer (KDA + FFN + residuals)

For both the SPLIT (pre-fusion) and FUSED (post-fusion) variants,
so we can read off the speedup directly.

The SPLIT variants are reconstructed by hand here (matching the
old TPKDA / TPSwiGLU structure: separate q/k/v ColumnParallelLinear,
f_proj and g_proj as Sequential, separate gate_proj / up_proj)
because the production code is already fused.

TEMPORARY test in test/_tmp/. Delete before commit per the
user's CLAUDE.md guidance.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO))

from src.models import HippoConfig
from src.models.tp_model._primitives import ColumnParallelLinear, RowParallelLinear, init_tp
from src.models.tp_model import TPKDA, TPSwiGLU, TPHippoLayer
from src.models.norms import RMSNorm
from src.models.ops._vendored.fla.modules import ShortConvolution


# --------------------------------------------------------------------------- #
# SPLIT versions (mirror the pre-fusion TPKDA / TPSwiGLU)                     #
# --------------------------------------------------------------------------- #
class _TPKDASplit(nn.Module):
    """TPKDA with separate q/k/v, f_proj / g_proj as Sequential.

    Mirrors the pre-fusion structure: 9 matmuls per KDA layer
    (3 QKV + 4 FG + 1 b + 1 o).
    """

    def __init__(self, cfg, device, dtype=torch.float16):
        super().__init__()
        from src.models.ops._vendored.fla.ops.kda import chunk_kda
        self._chunk_kda = chunk_kda

        self.world = 1
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
        self.use_short_conv = cfg.use_short_conv
        self.allow_neg_eigval = cfg.allow_neg_eigval
        self.safe_gate = cfg.safe_gate
        self.lower_bound = cfg.lower_bound

        # Q / K / V (separate ColumnParallelLinear)
        self.q_proj = ColumnParallelLinear(h, key_dim, bias=False, device=device, dtype=dtype)
        self.k_proj = ColumnParallelLinear(h, key_dim, bias=False, device=device, dtype=dtype)
        self.v_proj = ColumnParallelLinear(h, value_dim, bias=False, device=device, dtype=dtype)

        if cfg.use_short_conv:
            self.q_conv1d = ShortConvolution(
                hidden_size=key_dim, kernel_size=cfg.conv_size,
                bias=cfg.conv_bias, activation="silu",
            ).to(device=device, dtype=dtype)
            self.k_conv1d = ShortConvolution(
                hidden_size=key_dim, kernel_size=cfg.conv_size,
                bias=cfg.conv_bias, activation="silu",
            ).to(device=device, dtype=dtype)
            self.v_conv1d = ShortConvolution(
                hidden_size=value_dim, kernel_size=cfg.conv_size,
                bias=cfg.conv_bias, activation="silu",
            ).to(device=device, dtype=dtype)

        # F / G (Sequential of Linear + ColumnParallelLinear)
        self.f_proj = nn.Sequential(
            nn.Linear(h, d_v, bias=False, device=device, dtype=dtype),
            ColumnParallelLinear(d_v, gate_dim, bias=False, device=device, dtype=dtype),
        )
        self.g_proj = nn.Sequential(
            nn.Linear(h, d_v, bias=False, device=device, dtype=dtype),
            ColumnParallelLinear(d_v, value_dim, bias=True, device=device, dtype=dtype),
        )

        self.b_proj = ColumnParallelLinear(h, nvh, bias=False, device=device, dtype=dtype)

        import math as _math
        if cfg.safe_gate:
            self.A_log = nn.Parameter(
                torch.zeros(nvh, dtype=torch.float32, device=device)
            )
        else:
            self.A_log = nn.Parameter(
                torch.log(
                    torch.empty(nvh, dtype=torch.float32, device=device).uniform_(1, 16)
                )
            )
        self.A_log._no_weight_decay = True
        dt = torch.exp(
            torch.rand(gate_dim, dtype=torch.float32, device=device) *
            (_math.log(0.1) - _math.log(0.001)) + _math.log(0.001)
        ).clamp(min=1e-4)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)
        self.dt_bias._no_weight_decay = True

        from src.models.ops._vendored.fla.modules import FusedRMSNormGated
        self.o_norm = FusedRMSNormGated(
            d_v, activation="sigmoid", eps=cfg.rms_norm_eps,
            device=device, dtype=dtype,
        )
        self.o_proj = RowParallelLinear(value_dim, h, bias=False, device=device, dtype=dtype)

    def forward(self, x):
        from einops import rearrange
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        if self.use_short_conv:
            q, _ = self.q_conv1d(q)
            k, _ = self.k_conv1d(k)
            v, _ = self.v_conv1d(v)
        else:
            q = F.silu(q); k = F.silu(k); v = F.silu(v)
        g = self.f_proj(x)
        beta = self.b_proj(x).sigmoid()
        g_for_norm = self.g_proj(x)
        q = rearrange(q, "... (h d) -> ... h d", d=self.head_k_dim)
        k = rearrange(k, "... (h d) -> ... h d", d=self.head_k_dim)
        g = rearrange(g, "... (h d) -> ... h d", d=self.head_k_dim)
        v = rearrange(v, "... (h d) -> ... h d", d=self.head_v_dim)
        if self.allow_neg_eigval:
            beta = beta * 2.0
        o, _ = self._chunk_kda(
            q=q, k=k, v=v, g=g, beta=beta,
            A_log=self.A_log, dt_bias=self.dt_bias,
            initial_state=None, output_final_state=False,
            use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True,
            safe_gate=self.safe_gate, lower_bound=self.lower_bound,
            cu_seqlens=None,
        )
        g_for_norm = rearrange(g_for_norm, "... (h d) -> ... h d", d=self.head_v_dim)
        o = self.o_norm(o, g_for_norm)
        o = rearrange(o, "b t h d -> b t (h d)")
        return self.o_proj(o)


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
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)


class _LayerSplit(nn.Module):
    """TPHippoLayer with split TPKDA / TPSwiGLU."""

    def __init__(self, cfg, device, dtype=torch.float16):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps).to(device=device, dtype=dtype)
        self.mlp_norm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps).to(device=device, dtype=dtype)
        self.kda = _TPKDASplit(cfg, device, dtype).to(device=device, dtype=dtype)
        self.ffn = _TPSwiGLUSplit(cfg, device, dtype).to(device=device, dtype=dtype)

    def forward(self, x):
        x = x + self.kda(self.attn_norm(x))
        x = x + self.ffn(self.mlp_norm(x))
        return x


# --------------------------------------------------------------------------- #
# Benchmark harness                                                           #
# --------------------------------------------------------------------------- #
def _time_cuda_event(fn, n_warmup=5, n_iter=30):
    """Time ``fn()`` using CUDA events. Avoids host-side jitter
    that ``time.perf_counter`` picks up from the GPU command
    queue depth."""
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]
    stops = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]
    for i in range(n_iter):
        starts[i].record()
        fn()
        stops[i].record()
    torch.cuda.synchronize()
    times_ms = [s.elapsed_time(t) for s, t in zip(starts, stops)]
    mean = sum(times_ms) / len(times_ms)
    var = sum((t - mean) ** 2 for t in times_ms) / len(times_ms)
    return mean, var ** 0.5


def _bench(name, layer, x, n_iter=30):
    def fwd_bwd():
        x_local = x.detach().requires_grad_(True)
        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=True):
            y = layer(x_local)
        y.sum().backward()
    return _time_cuda_event(fwd_bwd, n_iter=n_iter)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--seq", type=int, default=512)
    p.add_argument("--hidden", type=int, default=1024)
    p.add_argument("--num-heads", type=int, default=8)
    p.add_argument("--head-dim", type=int, default=128)
    p.add_argument("--intermediate", type=int, default=3072)
    p.add_argument("--n-iter", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--mode", type=str, default="all",
                   choices=["kda", "ffn", "layer", "all"],
                   help="Which case to benchmark. Use --mode <case> to "
                        "isolate one case in a fresh process (avoids "
                        "GPU-state contamination between cases).")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = 0
    torch.cuda.set_device(device)
    torch.set_float32_matmul_precision("high")
    init_tp(world_size=1, devices=[device], backend="gloo")

    cfg = HippoConfig(
        vocab_size=8192,
        hidden_size=args.hidden, tie_word_embeddings=True,
        use_bias=False, num_heads=args.num_heads,
        head_dim=args.head_dim, expand_v=1.0,
        kda_mode="chunk", use_short_conv=True,
        allow_neg_eigval=False, safe_gate=False, lower_bound=None,
        conv_size=4, conv_bias=False,
        num_layers=2, num_blocks=2,
        intermediate_size=args.intermediate,
        rms_norm_eps=1e-6,
    )

    print(f"[setup] cfg: hidden={args.hidden} B={args.batch} T={args.seq}"
          f" intermediate={args.intermediate}")
    free, total = torch.cuda.mem_get_info(device)
    print(f"[setup] VRAM free={free/1024**3:.1f}GB / total={total/1024**3:.1f}GB")
    print(f"[setup] {args.n_iter} timed iterations after 3 warmup runs\n")

    x_template = torch.randn(args.batch, args.seq, cfg.hidden_size,
                             device=device, dtype=torch.float16)

    if args.mode in ("kda", "all"):
        # ---- 1) TPKDA: split vs fused ---- #
        print("[1/3] TPKDA: SPLIT (9 matmuls) vs FUSED (6 matmuls)...")
        split_kda = _TPKDASplit(cfg, device).to(device=device)
        fused_kda = TPKDA(cfg, layer_idx=0, device=device, dtype=torch.float16)
        m_split, s_split = _bench("split-kda", split_kda, x_template, n_iter=args.n_iter)
        m_fused, s_fused = _bench("fused-kda", fused_kda, x_template, n_iter=args.n_iter)
        delta_pct = 100 * (m_split - m_fused) / m_split
        print(f"  SPLIT: {m_split:7.2f} ± {s_split:.2f} ms")
        print(f"  FUSED: {m_fused:7.2f} ± {s_fused:.2f} ms")
        print(f"  delta: {m_split - m_fused:+7.2f} ms ({delta_pct:+.1f}%)")

    if args.mode in ("ffn", "all"):
        # ---- 2) SwiGLU: split vs fused ---- #
        print("\n[2/3] TPSwiGLU: SPLIT (gate+up) vs FUSED (gate_up)...")
        split_ffn = _TPSwiGLUSplit(cfg, device).to(device=device)
        fused_ffn = TPSwiGLU(cfg, device=device, dtype=torch.float16)
        m_split, s_split = _bench("split-ffn", split_ffn, x_template, n_iter=args.n_iter)
        m_fused, s_fused = _bench("fused-ffn", fused_ffn, x_template, n_iter=args.n_iter)
        delta_pct = 100 * (m_split - m_fused) / m_split
        print(f"  SPLIT: {m_split:7.2f} ± {s_split:.2f} ms")
        print(f"  FUSED: {m_fused:7.2f} ± {s_fused:.2f} ms")
        print(f"  delta: {m_split - m_fused:+7.2f} ms ({delta_pct:+.1f}%)")

    if args.mode in ("layer", "all"):
        # ---- 3) full TPHippoLayer: split vs fused ---- #
        print("\n[3/3] full TPHippoLayer: SPLIT vs FUSED...")
        split_layer = _LayerSplit(cfg, device).to(device=device)
        fused_layer = TPHippoLayer(layer_idx=0, config=cfg, device=device, dtype=torch.float16)
        m_split, s_split = _bench("split-layer", split_layer, x_template, n_iter=args.n_iter)
        m_fused, s_fused = _bench("fused-layer", fused_layer, x_template, n_iter=args.n_iter)
        delta_pct = 100 * (m_split - m_fused) / m_split
        print(f"  SPLIT: {m_split:7.2f} ± {s_split:.2f} ms")
        print(f"  FUSED: {m_fused:7.2f} ± {s_fused:.2f} ms")
        print(f"  delta: {m_split - m_fused:+7.2f} ms ({delta_pct:+.1f}%)")

    if args.mode == "all":
        print("\n[summary]")
        print("  Kernel launches per layer:")
        print("    SPLIT  TPKDA:  9 matmuls (q, k, v, f[0], f[1], g[0], g[1], b, o)")
        print("    FUSED  TPKDA:  6 matmuls (qkv, fg_first, f[1], g[1], b, o)")
        print("    SPLIT  SwiGLU: 3 matmuls (gate, up, down)")
        print("    FUSED  SwiGLU: 2 matmuls (gate_up, down)")
        print("    Saved: 4 launches per layer (3 KDA + 1 SwiGLU)")
        print("\n  Note: tried fusing f_proj[1] + g_proj[1] into one")
        print("  ColumnParallelLinear with output=gate_dim+value_dim. That")
        print("  would save 1 more launch but the fused weight is 50% zeros")
        print("  (structural) which still execute FMAs, doubling the FLOPs of")
        print("  that layer. Empirically slower; kept f_proj[1]/g_proj[1]")
        print("  split.")


if __name__ == "__main__":
    main()