"""Benchmark the KDA backward pass (FLA vendored implementation).

End-to-end timing of fwd + bwd, then per-subkernel bwd breakdown.
The bwd path goes through:
  1. fwd recompute (w, u, qg, kg, h, v_new) inside chunk_kda_bwd
  2. chunk_kda_bwd_dAv (dA, dv)        - [BT,BT] bmm + [BT,BV] bmm
  3. chunk_gated_delta_rule_bwd_dhu    - dh recurrence (reverse of fwd_h)
  4. chunk_kda_bwd_wy_dqkg_fused       - dq, dk, dv, dA, db, dg
  5. chunk_kda_bwd_intra               - intra-chunk corrections
  6. chunk_local_cumsum                - reverse cumsum for dg
  7. (gate bwd if use_gate_in_kernel)

Subkernel-level timing reuses FLA's Python entry points with manual
CUDA-event boundaries.

Usage:
    python bench/fast_kda_bwd_bench.py
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

# FLA bwd entry points (mirror the order in chunk_kda_bwd's source)
from src.models.ops._vendored.fla.ops.kda import chunk_kda
from src.models.ops._vendored.fla.ops.kda.chunk_bwd import (
    chunk_kda_bwd,
    chunk_kda_bwd_dAv,
    chunk_kda_bwd_wy_dqkg_fused,
)
from src.models.ops._vendored.fla.ops.kda.wy_fast import recompute_w_u_fwd
from src.models.ops._vendored.fla.ops.common.chunk_delta_h import (
    chunk_gated_delta_rule_bwd_dhu,
    chunk_gated_delta_rule_fwd_h,
)
from src.models.ops._vendored.fla.ops.kda.chunk_intra import chunk_kda_bwd_intra
from src.models.ops._vendored.fla.ops.utils import chunk_local_cumsum


CHUNK = 64  # FLA's BT
PROD_CHUNK = 16  # fast_kda's CHUNK


def _make_inputs(B, T, H, K, V, device, dtype, seed):
    torch.manual_seed(seed)
    q = torch.randn(B, T, H, K, device=device, dtype=torch.float32)
    k = torch.randn(B, T, H, K, device=device, dtype=torch.float32)
    v = torch.randn(B, T, H, V, device=device, dtype=dtype)
    q = (q / q.norm(dim=-1, keepdim=True)).to(dtype)
    k = (k / k.norm(dim=-1, keepdim=True)).to(dtype)
    g = -torch.rand(B, T, H, K, device=device, dtype=dtype) * 2.0 - 0.5
    beta = torch.randn(B, T, H, device=device, dtype=dtype).sigmoid()
    return q, k, v, g, beta


class CudaTimer:
    def __init__(self):
        self._start = None
        self._end = None

    def __enter__(self):
        torch.cuda.synchronize()
        self._start = torch.cuda.Event(enable_timing=True)
        self._end = torch.cuda.Event(enable_timing=True)
        self._start.record()
        return self

    def __exit__(self, *exc):
        self._end.record()
        torch.cuda.synchronize()
        self.ms = self._start.elapsed_time(self._end)


def _bench(fn, iters):
    """Median ms per call."""
    times = []
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    for _ in range(iters):
        with CudaTimer() as t:
            fn()
        times.append(t.ms)
    times.sort()
    return times[len(times) // 2]


SHAPES = {
    "small":    (1,   256,   4, 128, 128),
    "medium":   (1,  1024,   8, 128, 128),
    "k128":     (1,  2048,   8, 128, 128),
    "prod":     (1, 16384,  12, 128, 128),
    "long":     (1, 32768,  12, 128, 128),
}


def _stage_breakdown(B, T, H, K, V, dtype, device, iters):
    """Time each bwd subkernel by manually running chunk_kda_bwd's internals.

    Reproduces the order of operations from chunk_kda_bwd() in
    src/models/ops/_vendored/fla/ops/kda/chunk_bwd.py, but with CUDA
    events between each step.

    This is a structural breakdown — the chunk_kda_bwd() function does
    all of these in sequence; we just split them with events.
    """
    NC = (T + CHUNK - 1) // CHUNK
    NT = NC
    scale = 1.0 / math.sqrt(K)

    q, k, v, g, beta = _make_inputs(B, T, H, K, V, device, dtype, seed=7)
    q = q.detach().requires_grad_(True)
    k = k.detach().requires_grad_(True)
    v = v.detach().requires_grad_(True)
    g = g.detach().requires_grad_(True)
    beta = beta.detach().requires_grad_(True)

    # Run fwd once to get intermediates
    o, final_state = chunk_kda(q, k, v, g, beta, scale=scale)
    do = torch.randn_like(o)
    o.backward(do, retain_graph=True)

    # Fwd recompute inputs needed by chunk_kda_bwd
    # These mirror the recompute branch in chunk_kda_bwd():
    #   w, u, qg, kg = recompute_w_u_fwd(q, k, v, beta, A=Akk, gk=g)
    #   h, v_new, _ = chunk_gated_delta_rule_fwd_h(k=kg, w=w, u=u, gk=g, initial_state=initial_state)
    # We need Aqk + Akk from the fwd graph — easiest to call chunk_kda_fwd_intra
    from src.models.ops._vendored.fla.ops.kda.chunk_intra import chunk_kda_fwd_intra
    g_cumsum = g.detach().clone()  # placeholder, FLA uses this when use_gate_in_kernel=False
    _, _, _, _, Aqk, Akk = chunk_kda_fwd_intra(
        q=q.detach(), k=k.detach(), v=v.detach(), gk=g_cumsum,
        beta=beta.detach(), scale=scale, chunk_size=CHUNK,
    )

    # Re-extract q,k,v,g,beta as detached (bwd kernels don't need grads)
    qd = q.detach()
    kd = k.detach()
    vd = v.detach()
    gd = g_cumsum
    betad = beta.detach()
    dod = do.detach()

    # Stage breakdown loop
    times = {k: [] for k in [
        "w_u_recomp", "h_recomp", "dAv", "dhu", "wy_dqkg", "intra", "local_cumsum", "total",
    ]}

    for _ in range(iters):
        # Reset grads (fwd recompute creates new tensors; this is a fresh call)
        for _t in times.values():
            _t.clear()
        torch.cuda.synchronize()

        # Stage 1: w, u, qg, kg recompute (fwd recompute)
        ev0 = torch.cuda.Event(enable_timing=True)
        ev1 = torch.cuda.Event(enable_timing=True)
        ev2 = torch.cuda.Event(enable_timing=True)
        ev3 = torch.cuda.Event(enable_timing=True)
        ev4 = torch.cuda.Event(enable_timing=True)
        ev5 = torch.cuda.Event(enable_timing=True)
        ev6 = torch.cuda.Event(enable_timing=True)
        ev7 = torch.cuda.Event(enable_timing=True)
        ev8 = torch.cuda.Event(enable_timing=True)

        ev0.record()
        w, u, qg, kg = recompute_w_u_fwd(
            q=qd, k=kd, v=vd, beta=betad, A=Akk, gk=gd,
        )
        ev1.record()
        # Stage 2: h, v_new recompute (fwd_h)
        h, v_new, _ = chunk_gated_delta_rule_fwd_h(
            k=kg, w=w, u=u, gk=gd, initial_state=None,
            output_final_state=False, use_exp2=True,
        )
        ev2.record()
        # Stage 3: dAv
        dA, dv = chunk_kda_bwd_dAv(
            q=qd, k=kd, v=v_new, do=dod, A=Aqk, scale=scale, chunk_size=CHUNK,
        )
        ev3.record()
        # Stage 4: dhu (dh recurrence, reverse of fwd_h)
        dh, dh0, dv2 = chunk_gated_delta_rule_bwd_dhu(
            q=qg, k=kg, w=w, do=dod, dv=dv, gk=gd, h0=None, dht=None,
            scale=scale, use_exp2=True,
        )
        ev4.record()
        # Stage 5: wy_dqkg_fused (dq, dk, dv, dA, db, dg)
        dq, dk, dv3, db, dg, dAkk = chunk_kda_bwd_wy_dqkg_fused(
            q=qd, k=kd, v=vd, v_new=v_new, g=gd, beta=betad, A=Akk, h=h,
            do=dod, dh=dh, dv=dv2, scale=scale, chunk_size=CHUNK,
        )
        ev5.record()
        # Stage 6: intra (chunk-local corrections)
        dq2, dk2, db2, dg2 = chunk_kda_bwd_intra(
            q=qd, k=kd, g=gd, beta=betad, dAqk=dA, dAkk=dAkk,
            dq=dq, dk=dk, db=db, dg=dg, chunk_size=CHUNK,
        )
        ev6.record()
        # Stage 7: local cumsum for dg
        dg3 = chunk_local_cumsum(dg2, chunk_size=CHUNK, reverse=True)
        ev7.record()
        torch.cuda.synchronize()

        times["w_u_recomp"].append(ev0.elapsed_time(ev1))
        times["h_recomp"].append(ev1.elapsed_time(ev2))
        times["dAv"].append(ev2.elapsed_time(ev3))
        times["dhu"].append(ev3.elapsed_time(ev4))
        times["wy_dqkg"].append(ev4.elapsed_time(ev5))
        times["intra"].append(ev5.elapsed_time(ev6))
        times["local_cumsum"].append(ev6.elapsed_time(ev7))
        # total bwd kernel time
        times["total"].append(ev0.elapsed_time(ev7))

    return {k: sorted(v)[len(v) // 2] for k, v in times.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--shape", default="all", choices=list(SHAPES.keys()) + ["all"])
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--no_breakdown", action="store_true", help="Skip per-subkernel breakdown")
    args = p.parse_args()

    device = torch.device(f"cuda:{args.device}")
    dtype = torch.bfloat16
    torch.cuda.set_device(args.device)

    shapes = SHAPES if args.shape == "all" else {args.shape: SHAPES[args.shape]}

    print(f"\n=== KDA bwd benchmark ===")
    print(f"{'shape':<8} {'T':>6} {'H':>3} {'NC':>5} "
          f"{'fwd ms':>10} {'bwd ms':>10} {'f+b ms':>10} "
          f"{'bwd/fwd':>10}")
    print("-" * 80)

    rows = []
    for name, (B, T, H, K, V) in shapes.items():
        NC = (T + CHUNK - 1) // CHUNK
        scale = 1.0 / math.sqrt(K)
        q, k, v, g, beta = _make_inputs(B, T, H, K, V, device, dtype, args.seed)
        # Fwd only
        def _fwd():
            return chunk_kda(q=q, k=k, v=v, g=g, beta=beta, scale=scale)
        fwd_ms = _bench(_fwd, args.iters)

        # Set up for bwd
        q_g = q.detach().requires_grad_(True)
        k_g = k.detach().requires_grad_(True)
        v_g = v.detach().requires_grad_(True)
        g_g = g.detach().requires_grad_(True)
        beta_g = beta.detach().requires_grad_(True)

        # Fwd + bwd (one full training step)
        def _fwd_bwd():
            o, _ = chunk_kda(q=q_g, k=k_g, v=v_g, g=g_g, beta=beta_g, scale=scale)
            do = torch.randn_like(o)
            o.backward(do, retain_graph=False)
            # zero grads for next iter
            for p in (q_g, k_g, v_g, g_g, beta_g):
                if p.grad is not None:
                    p.grad = None
        fwd_bwd_ms = _bench(_fwd_bwd, args.iters)
        bwd_ms = max(fwd_bwd_ms - fwd_ms, 0.0)

        ratio = bwd_ms / fwd_ms if fwd_ms > 0 else 0
        print(f"{name:<8} {T:>6} {H:>3} {NC:>5} "
              f"{fwd_ms:>10.3f} {bwd_ms:>10.3f} {fwd_bwd_ms:>10.3f} "
              f"{ratio:>9.2f}x")
        rows.append((name, T, fwd_ms, bwd_ms, fwd_bwd_ms, ratio))

    # Per-subkernel breakdown for prod
    if not args.no_breakdown and "prod" in shapes:
        B, T, H, K, V = shapes["prod"]
        print(f"\n=== Per-subkernel bwd breakdown (prod: T={T} H={H} K=V={K}) ===")
        print(f"{'stage':<18} {'time (ms)':>10} {'% of bwd':>10}")
        print("-" * 50)
        stages = _stage_breakdown(B, T, H, K, V, dtype, device, args.iters)
        total_bwd = stages["total"]
        for k in ["w_u_recomp", "h_recomp", "dAv", "dhu", "wy_dqkg", "intra", "local_cumsum"]:
            pct = stages[k] / total_bwd * 100 if total_bwd > 0 else 0
            print(f"  {k:<18} {stages[k]:>10.3f} {pct:>9.1f}%")
        print(f"  {'TOTAL':<18} {total_bwd:>10.3f} {'100.0%':>10}")


if __name__ == "__main__":
    main()
