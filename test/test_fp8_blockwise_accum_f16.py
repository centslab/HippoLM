"""Regression coverage for the ACC_RAW_F16 (R10) blockwise FP8 GEMM.

The F16 in-block accumulator (mma...f16.e4m3.e4m3.f16) sustains ~2x
the F32-acc QMMA issue rate on sm_120 (97.9 -> 168 TF @ 4096^3, and
54 -> 169 TF at the M>=4096/K>=8192 shapes). These tests pin:

1. Numerics parity with the F32 path on bounded data (med_rel < 0.2%
   vs FP64 reference; F32 path is ~0.11%).
2. Finiteness for scale-normalized inputs (|values| ~ O(1)).
3. The documented overflow boundary: acc_raw sums 128 raw FP8
   products in F16 (max 65504) before the FP32 flush, so adversarial
   max-magnitude FP8 inputs overflow to Inf while the F32 path stays
   finite. If a future change removes the boundary, update the entry
   file comment + auto-memory instead of deleting this test.
4. Perf floor: R10 >= 1.3x R7 at 4096^3 (expected ~1.7x; loose bound
   to stay non-flaky on a shared box).
"""
from __future__ import annotations

import ctypes
from pathlib import Path

import pytest
import torch

_LIB_DIR = Path(__file__).resolve().parent.parent / "src" / "models" / "ops" / "cuda" / "lib"
_SM = f"sm_{torch.cuda.get_device_capability()[0]}{torch.cuda.get_device_capability()[1]}"
_R7_SO = _LIB_DIR / f"fp8_blockwise_gemm_{_SM}_bm128_bn128_bk128_s3_cwg2_wm32_wn64_ds_blockwise_accum_bsk128.so"
_R10_SO = _LIB_DIR / f"fp8_blockwise_gemm_{_SM}_bm128_bn128_bk128_s3_cwg2_wm32_wn64_ds_blockwise_accum_bsk128_f16.so"


def _load(path: Path, symbol: str):
    lib = ctypes.CDLL(str(path), mode=ctypes.RTLD_LOCAL)
    fn = getattr(lib, symbol)
    fn.restype = None
    fn.argtypes = [ctypes.c_int] * 3 + [ctypes.c_void_p] * 6 + [ctypes.c_void_p]
    return fn


def _run(fn, A, B, a_s, b_s):
    M, K = A.shape
    N = B.shape[0]
    out = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)
    fn(M, N, K, A.data_ptr(), B.data_ptr(), out.data_ptr(),
       a_s.data_ptr(), b_s.data_ptr(), 0, torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()
    return out


def _reference(A, B, a_s, b_s):
    M, K = A.shape
    N = B.shape[0]
    Af = A.double().view(M // 64, 64, K // 128, 128) * a_s.double().unsqueeze(1).unsqueeze(-1)
    Bf = B.double().view(N // 64, 64, K // 128, 128) * b_s.double().unsqueeze(1).unsqueeze(-1)
    return Af.view(M, K) @ Bf.view(N, K).T


def _med_rel(out, ref):
    d = (out.double() - ref).abs()
    return (d.median() / ref.abs().median().clamp_min(1e-30)).item() * 100


def _time_ms(fn, iters=20):
    for _ in range(5):
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


pytestmark = pytest.mark.skipif(
    not (_R7_SO.exists() and _R10_SO.exists()),
    reason="accum_bsk128 / accum_bsk128_f16 .so not built for this arch",
)


@pytest.fixture(scope="module")
def kernels():
    f32 = _load(_R7_SO, "blockwise_accum_bsk128_gemm_run")
    f16 = _load(_R10_SO, "blockwise_accum_bsk128_f16_gemm_run")
    return f32, f16


@pytest.fixture(scope="module")
def data():
    M, N, K = 1024, 1024, 4096
    g = torch.Generator(device="cuda").manual_seed(0)
    A = torch.randn(M, K, generator=g, device="cuda").clamp(-3, 3).to(torch.float8_e4m3fn)
    B = torch.randn(N, K, generator=g, device="cuda").clamp(-3, 3).to(torch.float8_e4m3fn)
    return A, B


def test_mild_scales_numerics(kernels, data):
    f32, f16 = kernels
    A, B = data
    M, K = A.shape
    N = B.shape[0]
    g = torch.Generator(device="cuda").manual_seed(1)
    a_s = (torch.rand(M // 64, K // 128, generator=g, device="cuda") * 0.5 + 0.75).bfloat16().contiguous()
    b_s = (torch.rand(N // 64, K // 128, generator=g, device="cuda") * 0.5 + 0.75).bfloat16().contiguous()
    ref = _reference(A, B, a_s, b_s)
    y32 = _run(f32, A, B, a_s, b_s)
    y16 = _run(f16, A, B, a_s, b_s)
    assert torch.isfinite(y16).all()
    r32 = _med_rel(y32, ref)
    r16 = _med_rel(y16, ref)
    assert r32 < 0.2, f"F32 path regressed: {r32:.4f}%"
    assert r16 < 0.2, f"F16 path above noise floor: {r16:.4f}% (F32 {r32:.4f}%)"


def test_sharp_scales_numerics(kernels, data):
    f32, f16 = kernels
    A, B = data
    M, K = A.shape
    N = B.shape[0]
    g = torch.Generator(device="cuda").manual_seed(2)
    a_s = (torch.rand(M // 64, K // 128, generator=g, device="cuda") * 8 - 4).exp2().bfloat16().contiguous()
    b_s = (torch.rand(N // 64, K // 128, generator=g, device="cuda") * 8 - 4).exp2().bfloat16().contiguous()
    ref = _reference(A, B, a_s, b_s)
    y16 = _run(f16, A, B, a_s, b_s)
    assert torch.isfinite(y16).all()
    assert _med_rel(y16, ref) < 0.2


def test_overflow_boundary_documented(kernels):
    f32, f16 = kernels
    M, N, K = 512, 512, 256
    A = torch.full((M, K), 448.0, device="cuda").to(torch.float8_e4m3fn)
    B = torch.full((N, K), 448.0, device="cuda").to(torch.float8_e4m3fn)
    a_s = torch.full((M // 64, K // 128), 1e-3, device="cuda", dtype=torch.bfloat16).contiguous()
    b_s = torch.full((N // 64, K // 128), 1e-3, device="cuda", dtype=torch.bfloat16).contiguous()
    y32 = _run(f32, A, B, a_s, b_s)
    y16 = _run(f16, A, B, a_s, b_s)
    # F32 path: finite (huge intermediate partials, scaled down at flush).
    assert torch.isfinite(y32).all()
    # F16 path: 448*448 = 200704 > 65504 per single product -> Inf.
    # This is the documented ACC_RAW_F16 boundary; if a future kernel
    # version avoids it, update this test + the entry comment.
    assert not torch.isfinite(y16).all()


def test_perf_floor_vs_f32(kernels):
    f32, f16 = kernels
    M = N = K = 4096
    g = torch.Generator(device="cuda").manual_seed(3)
    A = torch.randn(M, K, generator=g, device="cuda").clamp(-3, 3).to(torch.float8_e4m3fn)
    B = torch.randn(N, K, generator=g, device="cuda").clamp(-3, 3).to(torch.float8_e4m3fn)
    a_s = torch.ones(M // 64, K // 128, device="cuda", dtype=torch.bfloat16).contiguous()
    b_s = torch.ones(N // 64, K // 128, device="cuda", dtype=torch.bfloat16).contiguous()

    out32 = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)
    out16 = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)

    def call32():
        f32(M, N, K, A.data_ptr(), B.data_ptr(), out32.data_ptr(),
            a_s.data_ptr(), b_s.data_ptr(), 0, torch.cuda.current_stream().cuda_stream)

    def call16():
        f16(M, N, K, A.data_ptr(), B.data_ptr(), out16.data_ptr(),
            a_s.data_ptr(), b_s.data_ptr(), 0, torch.cuda.current_stream().cuda_stream)

    ms32 = _time_ms(call32)
    ms16 = _time_ms(call16)
    ratio = ms32 / ms16
    assert ratio >= 1.3, (
        f"F16-acc expected >=1.3x F32-acc at 4096^3 (measured {ratio:.2f}x; "
        f"F32 {2*M*N*K/(ms32*1e-3)/1e12:.1f} TF, F16 {2*M*N*K/(ms16*1e-3)/1e12:.1f} TF)"
    )
