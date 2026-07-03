"""Smoke test: does Triton compile a CPU kernel at all?

We need to know if Triton can be used as the fused-kernel
backend before committing to a Triton implementation. If it
fails, we fall back to bf16 (option B).
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import triton
import triton.language as tl


@triton.jit
def simple_add(x_ptr, y_ptr, out_ptr, n: tl.constexpr):
    offs = tl.arange(0, n)
    x = tl.load(x_ptr + offs)
    y = tl.load(y_ptr + offs)
    tl.store(out_ptr + offs, x + y)


def main():
    print("=== Triton CPU backend smoke test ===\n")
    n = 1024
    x = torch.randn(n, dtype=torch.float32)
    y = torch.randn(n, dtype=torch.float32)
    out = torch.empty(n, dtype=torch.float32)

    # Try a CPU compile
    try:
        simple_add[(1,)](x, y, out, n)
        diff = (out - x - y).abs().max().item()
        print(f"  PASS: simple_add max diff = {diff:.2e}")
    except Exception as e:
        print(f"  FAIL on CPU: {type(e).__name__}: {e}")
        return

    # Time it
    for _ in range(5):
        simple_add[(1,)](x, y, out, n)
    t0 = time.perf_counter()
    for _ in range(1000):
        simple_add[(1,)](x, y, out, n)
    elapsed = (time.perf_counter() - t0) / 1000 * 1e6
    print(f"  Per-call latency: {elapsed:.1f} us")

    # Check what target Triton compiled to
    print(f"  Triton version: {triton.__version__}")


if __name__ == "__main__":
    main()
