---
name: bench-flops
description: Measure FP8/BF16 GEMM throughput correctly on sm_120. Use when the user asks to "benchmark GEMM", "measure TFLOPS", "compare kernels", "report FLOPs", "measure peak efficiency", or any time a kernel-speedup claim involves absolute TFLOPS numbers. Gate: pre-flight torch.matmul cross-check + 3-method cross-validation + real-time clock polling. Triggered by 2026-07-27 audit finding that single-event cudaEvent timing under-reports ctypes-loaded .so kernels by 2x.
---

# Benchmark FP8/BF16 GEMM FLOPs

The 2026-07-27 audit found that **cudaEvent single-event timing
systematically under-reports kernel time by 2x** when measuring
ctypes-loaded custom .so kernels on sm_120. The R10 FP8 GEMM
"168 TF" memory entry was this artifact — true CUPTI ground truth
is **80 TF**. Numbers from R0–R9 and any prior kernel sweep are
suspect until re-measured with this methodology.

A FLOPs number is not a claim until three independent cross-checks
agree.

## Phase 1 — Pre-flight cross-validation (mandatory before any GEMM timing)

Before measuring any custom kernel, validate the methodology itself
on `torch.matmul` BF16 (known peak ~50 TF on RTX 5060 Ti, ~165 TF
on RTX 4090). If the methodology can't measure a built-in op
correctly, it cannot measure a custom op correctly.

```python
import torch
M = N = K = 4096
a = torch.randn(M, K, device='cuda', dtype=torch.bfloat16)
b = torch.randn(K, N, device='cuda', dtype=torch.bfloat16)

# Warmup
for _ in range(20):
    c = a @ b
torch.cuda.synchronize()

flops = 2.0 * M * N * K

# Method A (CUPTI ground truth)
with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
    for _ in range(100):
        c = a @ b
    torch.cuda.synchronize()
# Extract from prof.key_averages(): print(prof.key_averages().table(...))

# Method B (batched cudaEvent aggregate)
n = 500
s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
s.record()
for _ in range(n):
    c = a @ b
e.record(); e.synchronize()
batched_us = s.elapsed_time(e) * 1000 / n

# Gate: Methods A and B must agree within 5% on torch.matmul.
# If not, the measurement environment is broken — DO NOT report
# any custom kernel timing.
```

`torch.matmul` is the control: both methods should land at
~50% of theoretical BF16 peak (~25 TF on 5060 Ti, ~83 TF on 4090).
If Method A says 50 TF but Method B says 25 TF, something is
wrong with how Method B is reading events — fix the methodology
first.

## Phase 2 — Setup (pre-allocate everything)

```python
# WRONG: tensors allocated inside the per-iter function — HBM alloc
# overhead (~1 ms/iter on 5060 Ti) buries the real kernel time.
def run(M, N, K):
    A = torch.randn(M, K, ...).to(...)   # NEW tensor per call
    B = torch.randn(N, K, ...).to(...)
    out = torch.empty(M, N, ...)
    fn(M, N, K, A.data_ptr(), ...)        # ~0.8 ms kernel + ~1 ms alloc = 1.8 ms

# RIGHT: tensors allocated once, passed in.
A = torch.randn(M, K, ...).to(...)
B = torch.randn(N, K, ...).to(...)
out = torch.empty(M, N, ...)
def run():
    fn(M, N, K, A.data_ptr(), B.data_ptr(), out.data_ptr(), ...)  # ~0.8 ms pure kernel
```

Heavy warmup (50+ iterations) to ensure the GPU is at steady-state
clock before any timing measurement.

## Phase 3 — Three-method cross-validation

Use **all three** for every custom-kernel perf claim:

| Method | Pattern | Accuracy | When to trust |
|---|---|---|---|
| **A: torch.profiler (CUPTI)** | profile N iters, read `cuda_time_total / count` | Ground truth via hardware counters | Always — this is the citation |
| **B: cudaEvent batched** | `s.record(); for _ in range(N): fn(); e.record(); e.synchronize(); time = s.elapsed_time(e)/N` | Agrees with A | When A is unavailable; report as cross-check |
| **C: cudaEvent single-event** | `for _ in range(N): s.record(); fn(); e.record(); e.synchronize(); times.append(s.elapsed_time(e))` | **Under-reports 2x for ctypes .so kernels** | ⚠️ Never use alone for custom kernels |

**Cross-validation gates**:
- If A and B differ by >5% on the custom kernel → measurement bug,
  do not report the time. Investigate.
- C must first be shown to agree with A on the custom kernel (within
  10%, not 5% — C is empirically biased). If C disagrees with A by
  >10%, the kernel is in the bug regime; use A only.

## Phase 4 — Theoretical peak + clock polling

```python
import subprocess, time, threading, statistics

samples = []
stop = threading.Event()
def poll():
    while not stop.is_set():
        try:
            r = subprocess.check_output(
                ['nvidia-smi', '--query-gpu=clocks.current.graphics,power.draw',
                 '--format=csv,noheader,nounits'], timeout=2).decode().strip()
            clock_mhz, power_w = [p.strip() for p in r.split(',')]
            samples.append({'clock': int(clock_mhz), 'power': float(power_w)})
        except Exception:
            pass
        time.sleep(0.05)

# Run sustained (100+ iters) with polling
t = threading.Thread(target=poll); t.start()
s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
s.record()
for _ in range(N):
    fn()  # uses pre-allocated buffers from Phase 2
e.record(); e.synchronize()
stop.set(); t.join()

# Use AVERAGE clock during the run, NOT the idle clock or boost clock.
avg_clock_mhz = statistics.mean(s['clock'] for s in samples)
peak_tf = 512 * 144 * (avg_clock_mhz / 1000.0) / 1000.0  # F16-acc FP8 dense peak formula
achieved_tf = flops / (s.elapsed_time(e) / N * 1e-3) / 1e12
print(f'  achieved: {achieved_tf:.2f} TF  (clock avg {avg_clock_mhz:.0f} MHz, peak {peak_tf:.2f} TF)')
print(f'  achieved/peak: {achieved_tf/peak_tf*100:.1f}%')
```

**Why real-time clock matters**:
- Idle clock (no work): ~2.82 GHz on 5060 Ti, would over-state peak
- Boost clock (3.09 GHz on 5060 Ti): rarely achieved under sustained FP8 GEMM
- Sustained clock under load: ~2.77 GHz on 5060 Ti, this is the
  correct denominator for peak

## Phase 5 — Report format

Every FLOPs claim in a commit message or memory entry should include:

```
achieved TF  |  achieved/peak %  |  sustained clock  |  methodology
80.4 TF      |  39.6%           |  2777 MHz        |  torch.profiler (CUPTI)
```

If only one method was used, the claim is **not** substantiated. If
Method C alone was used on a ctypes-loaded kernel, the claim is
**likely 2x too high** and should be re-measured.

## Known measurement artifacts (avoid these)

1. **cudaEvent single-event on ctypes .so** — under-reports 2x.
   Always cross-validate against Method A.
2. **Per-iter HBM allocation inside the timing function** — adds
   ~1 ms/iter overhead on 5060 Ti, hides kernel time.
3. **Single sync inside a loop without batching** — same as #1.
4. **Trusting `torch.cuda.clock_rate()` idle reading** — uses idle
   clock, over-states peak.
5. **`torch.profiler` CPU time** — irrelevant; only `cuda_time_total`
   matters for GPU kernel time.
6. **Comparing across different shapes without normalizing** — TF
   varies with shape; always state the M/N/K when quoting TF.

## How to apply

- Before claiming any "X TF" in a commit message, memory entry, or
  PR description, run Phases 1–5 with the specific kernel and
  shape, and cite the achieved TF + methodology.
- If re-measuring an old claim that used Method C alone, **assume
  it is 2x too high** until verified.
- The R10 commit `c3b96bc` claimed "97.9 → 168.3 TF (1.72x)"; the
  `97.9 → 80 TF` revision is the ground-truth number. The "1.72x"
  ratio was correct (R10/R7 = 80/46 ≈ 1.74x), but the absolute TF
  was measured with the buggy single-event methodology.