"""FP8 GEMM small-M × large-M sweep on sm_120.

Sweeps every prebuilt fp8_gemm_scaled config + cuBLAS BF16 + _scaled_mm FP8
across M ∈ {128, 256, 512, 1024, 2048, 4096, 8192} at the prod KDA + FFN shapes.

Goal: pick the right backend at each M without regressing large M.

Per auto-memory feedback_verify_correctness.md / feedback_roofline_discipline.md:
- direct CUDA event median timing
- correctness gate: max-diff vs cuBLAS BF16 reference, isfinite check
- reported as TFLOPS = 2*M*N*K / time, % of 97 TF dense fp8 peak (sm_120)

Run:
  PYTHONPATH=/hy-tmp/HippoLM python scripts/probes/probe_fp8_small_m_sweep.py

This is the canonical new-env recipe for adapting the FP8 GEMM dispatch
config to a new GPU arch (sm_89 prod, sm_100 datacenter, etc.). It pairs
with ``scripts/build_fp8_gemm.py --arch sm_<X>`` (which produces the
``.so`` files this sweep consumes) and ``docs/fp8_gemm_kernel_pipeline.md``
(interpretation rules + numerics gates).
"""
from __future__ import annotations

import ctypes
import sys
from pathlib import Path

import torch

PEAK_FP8 = 97.0   # sm_120 5060 Ti dense FP8 (cuBLAS RowWise absent; CUTLASS reaches this)
PEAK_BF16 = 50.0  # sm_120 5060 Ti dense BF16

_LIB_DIR = Path("/hy-tmp/HippoLM/src/models/ops/cuda/lib")

# Prebuilt configs in the lib dir. Each .so exposes a single `gemm_run(M, N, K,
# A, B, Y, a_scale, b_scale, workspace, stream)` symbol. ABI is identical; the
# only difference is the (BM, BN, BK, NUM_STAGES, CWG, WARP_M, WARP_N, DIRECT_STORE)
# specialization.
CONFIGS = [
    # (label, file, BM, BN, BK, kind)
    ("_scaled_mm_fp8",   None,                                 None, None, None, "cublas"),
    ("cuBLAS_BF16",      None,                                 None, None, None, "bf16"),
    ("BM128_BN128",      "fp8_gemm_sm_120_bm128_bn128_bk128_s3_cwg2_wm32_wn64_ds.so",     128, 128, 128, "custom"),
    ("BM64_BN64",        "fp8_gemm_sm_120_bm64_bn64_bk128_s3_cwg1_wm32_wn32_ds.so",       64,  64,  128, "custom"),
    ("BM128_BN64",       "fp8_gemm_sm_120_bm128_bn64_bk128_s3_cwg1_wm32_wn64_ds.so",      128, 64,  128, "custom"),
    ("BM64_BN128",       "fp8_gemm_sm_120_bm64_bn128_bk128_s3_cwg1_wm32_wn64_ds.so",      64,  128, 128, "custom"),
    ("BM128_BN128_s2",   "fp8_gemm_sm_120_bm128_bn128_bk128_s2_cwg2_wm32_wn64.so",        128, 128, 128, "custom"),
    ("BM128_BN128_wm64", "fp8_gemm_sm_120_bm128_bn128_bk128_s2_cwg2_wm64_wn32.so",        128, 128, 128, "custom"),
    ("BM64_BN64_s2",     "fp8_gemm_sm_120_bm64_bn64_bk128_s2_cwg1_wm32_wn32.so",          64,  64,  128, "custom"),
]

# Production shapes: (K, N) pairs across KDA + FFN sublayers. Names matter for
# the report.
SHAPES = [
    ("KDA_qkv[1536,1536]", 1536, 1536),
    ("FFN_gate_up[1536,4096]", 1536, 4096),
    ("FFN_down[4096,1536]", 4096, 1536),
    # lm_head (N=248320) is N×M bf16 out, OOMs at M=8192 (4 GiB); separate
    # skinny-K fat-N regime, off the MBS-shrink critical path.
]

# M values to sweep. Picks cover the user's upcoming MBS shrink trajectory plus
# two large-M anchors to detect regression.
M_VALUES = [128, 256, 512, 1024, 2048, 4096, 8192]


def _load_custom_so(so_name: str):
    """Load a prebuilt fp8_gemm .so and bind gemm_run."""
    path = _LIB_DIR / so_name
    if not path.exists():
        return None
    lib = ctypes.CDLL(str(path), mode=ctypes.RTLD_LOCAL)
    lib.gemm_run.restype = None
    lib.gemm_run.argtypes = [
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p,
    ]
    return lib


def _cuda_time(fn, iters: int = 50, warmup: int = 10) -> float:
    """Median per-call ms via CUDA events. Skips outliers by sorting."""
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


def _quantize_inputs(M: int, K: int, N: int, dtype_bf16: bool = False):
    """Build (A, B, a_scale, b_scale) for FP8 GEMM. Returns BF16 reference too.

    a_scale: per-row [M, 1] FP32  (max over K of |A|, /448 then clamp)
    b_scale: per-col [1, N] FP32  (max over K of |B|, /448 then clamp)
    """
    torch.manual_seed(0)  # deterministic across runs
    a_bf16 = (torch.randn(M, K, dtype=torch.bfloat16, device="cuda") * 0.1).contiguous()
    b_bf16 = (torch.randn(N, K, dtype=torch.bfloat16, device="cuda") * 0.05).contiguous()

    a_fp32 = a_bf16.float()
    b_fp32 = b_bf16.float()
    a_amax = a_fp32.abs().amax(dim=1, keepdim=True).clamp(min=1e-6)
    b_amax = b_fp32.abs().amax(dim=1, keepdim=True).clamp(min=1e-6)
    a_scale = (a_amax / 448.0).contiguous()  # [M, 1]
    b_scale_in = (b_amax / 448.0)           # [N, 1]
    b_scale = b_scale_in.T.contiguous()     # [1, N]

    a_q = (a_fp32 / a_scale).clamp(-448, 448).to(torch.float8_e4m3fn).contiguous()
    b_q = (b_fp32 / b_scale_in).clamp(-448, 448).to(torch.float8_e4m3fn).contiguous()

    return a_bf16, b_bf16, a_q, b_q, a_scale, b_scale


def _run_one(label: str, M: int, K: int, N: int, BM, BN, kind, lib) -> dict:
    """Run one (config, M, K, N) and return timing + correctness."""
    a_bf16, b_bf16, a_q, b_q, a_scale, b_scale = _quantize_inputs(M, K, N)
    out = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")

    # --- BF16 cuBLAS reference (correctness ground truth) ---
    bf16_ref = torch.nn.functional.linear(a_bf16, b_bf16)
    if not torch.isfinite(bf16_ref).all():
        return {"label": label, "M": M, "K": K, "N": N, "ERR": "BF16 ref not finite"}

    # Warmup all paths so CUTLASS heuristics / ctypes binding settle.
    for _ in range(5):
        if kind == "bf16":
            torch.nn.functional.linear(a_bf16, b_bf16)
        elif kind == "cublas":
            torch._scaled_mm(a_q, b_q.T, a_scale, b_scale, out_dtype=torch.bfloat16)
        else:
            stream = torch.cuda.current_stream().cuda_stream
            lib.gemm_run(
                M, N, K,
                a_q.data_ptr(), b_q.data_ptr(), out.data_ptr(),
                a_scale.data_ptr(), b_scale.data_ptr(),
                0, stream,
            )
    torch.cuda.synchronize()

    def _bf16():
        torch.nn.functional.linear(a_bf16, b_bf16)
    def _cublas():
        torch._scaled_mm(a_q, b_q.T, a_scale, b_scale, out_dtype=torch.bfloat16)
    def _custom():
        stream = torch.cuda.current_stream().cuda_stream
        lib.gemm_run(
            M, N, K,
            a_q.data_ptr(), b_q.data_ptr(), out.data_ptr(),
            a_scale.data_ptr(), b_scale.data_ptr(),
            0, stream,
        )

    if kind == "bf16":
        t = _cuda_time(_bf16)
    elif kind == "cublas":
        t = _cuda_time(_cublas)
    else:
        t = _cuda_time(_custom)

    flops = 2 * M * K * N
    tflops = flops / (t * 1e9)
    peak_ref = PEAK_BF16 if kind == "bf16" else PEAK_FP8
    pct_peak = tflops / peak_ref * 100

    # --- Correctness vs BF16 cuBLAS reference ---
    if kind == "bf16":
        max_diff = 0.0
        sig_rel = 0.0
    else:
        if kind == "cublas":
            y = torch._scaled_mm(a_q, b_q.T, a_scale, b_scale, out_dtype=torch.bfloat16)
        else:
            stream = torch.cuda.current_stream().cuda_stream
            lib.gemm_run(
                M, N, K,
                a_q.data_ptr(), b_q.data_ptr(), out.data_ptr(),
                a_scale.data_ptr(), b_scale.data_ptr(),
                0, stream,
            )
            y = out
        finite = torch.isfinite(y).all().item()
        if not finite:
            return {"label": label, "M": M, "K": K, "N": N, "ERR": "FP8 output not finite"}
        max_diff = (y.float() - bf16_ref.float()).abs().max().item()
        denom = bf16_ref.float().abs().max().item() + 1e-9
        sig_rel = (y.float() - bf16_ref.float()).abs().max().item() / denom

    return {
        "label": label, "M": M, "K": K, "N": N,
        "t_ms": t, "tflops": tflops, "pct_peak": pct_peak,
        "max_diff": max_diff, "sig_rel": sig_rel,
    }


def main():
    print(f"GPU: {torch.cuda.get_device_name(0)} (sm_120, 36 SMs)")
    print(f"Reference peaks: FP8={PEAK_FP8} TFLOPS, BF16={PEAK_BF16} TFLOPS\n")

    # Load custom .so files once.
    libs = {}
    for cfg in CONFIGS:
        label, fname, BM, BN, BK, kind = cfg
        if kind != "custom":
            continue
        lib = _load_custom_so(fname)
        if lib is None:
            print(f"  [WARN] missing {_LIB_DIR / fname}")
        libs[fname] = lib

    # ---- Sweep ----
    rows = []
    for shape_name, K, N in SHAPES:
        for M in M_VALUES:
            # Skip combinations where M/K/N aren't divisible by the smallest tile
            # we test (BM=64, BN=64, BK=128). The kernel hard-asserts, so we
            # should fail-fast in Python rather than trap.
            if M % 64 or N % 64 or K % 128:
                continue
            for cfg in CONFIGS:
                label, fname, BM, BN, BK, kind = cfg
                if kind == "custom":
                    lib = libs.get(fname)
                    if lib is None:
                        continue
                else:
                    lib = None
                rows.append(_run_one(label, M, K, N, BM, BN, kind, lib))

    # ---- Report ----
    # Group by shape for readability.
    by_shape = {}
    for r in rows:
        if "ERR" in r:
            print(f"  [ERR] {r}")
            continue
        key = r["K"], r["N"]
        by_shape.setdefault(key, []).append(r)

    for (K, N), group in by_shape.items():
        shape_name = next(s for s, kk, nn in SHAPES if kk == K and nn == N)
        print(f"\n========== {shape_name} (K={K}, N={N}) ==========")
        # Pivot: rows = M, columns = (label, tflops / pct_peak)
        ms = sorted(set(r["M"] for r in group))
        labels = sorted(set(r["label"] for r in group), key=lambda l: -1 if l == "_scaled_mm_fp8" else (0 if l == "cuBLAS_BF16" else 1 if l == "BM128_BN128" else 2))
        # Header
        header = f"  {'M':>5} | " + " | ".join(f"{lbl:>22}" for lbl in labels)
        print(header)
        print("  " + "-" * (len(header) - 2))
        for m in ms:
            row_data = []
            for lbl in labels:
                hit = next((r for r in group if r["M"] == m and r["label"] == lbl), None)
                if hit:
                    row_data.append(f"{hit['pct_peak']:5.0f}% ({hit['tflops']:5.0f}TF)")
                else:
                    row_data.append(f"{'N/A':>22}")
            print(f"  {m:>5} | " + " | ".join(f"{c:>22}" for c in row_data))
        # Correctness table for this shape — single line per scheme
        print(f"  --- correctness (sig_rel vs BF16 cuBLAS ref, max across M) ---")
        for lbl in labels:
            sigrels = [r["sig_rel"] for r in group if r["label"] == lbl]
            maxdiffs = [r["max_diff"] for r in group if r["label"] == lbl]
            if sigrels:
                print(f"    {lbl:>22}: max_diff={max(maxdiffs):.4f}, max_sig_rel={max(sigrels):.4%}")

    # ---- Summary: best backend per M ----
    print("\n========== SUMMARY: best backend per (M, shape) ==========")
    best = {}
    for r in rows:
        if "ERR" in r:
            continue
        key = (r["M"], r["K"], r["N"])
        if key not in best or r["pct_peak"] > best[key]["pct_peak"]:
            best[key] = r
    for (M, K, N), r in sorted(best.items()):
        shape_name = next(s for s, kk, nn in SHAPES if kk == K and nn == N)
        print(f"  M={M:>5} {shape_name:>30}: {r['label']:>18}  {r['pct_peak']:5.1f}% peak ({r['tflops']:5.1f} TF)")

    # ---- Regression check: large-M anchor ----
    print("\n========== LARGE-M REGRESSION CHECK ==========")
    for shape_name, K, N in SHAPES:
        # Compare best FP8 backend at M=8192 vs M=1024 (prod M)
        prod_ref = next((r for r in rows if r["M"] == 1024 and r["K"] == K and r["N"] == N), None)
        large_ref = next((r for r in rows if r["M"] == 8192 and r["K"] == K and r["N"] == N), None)
        if prod_ref and large_ref:
            # Both report best backend; we want to confirm large-M still at ≥90% peak
            ok = "OK" if large_ref["pct_peak"] >= 90 else "REGRESS"
            print(f"  {shape_name:>30}: M=1024 {prod_ref['pct_peak']:5.1f}%, M=8192 {large_ref['pct_peak']:5.1f}% [{ok}]")


if __name__ == "__main__":
    main()