---
paths:
  - "src/models/ops/**"
  - "configs/**"
---

# Don't reintroduce V100 (sm_70) support

V100 was dropped on 2026-07-08. Both production paths that depend
on sm_70 cannot run on it:

- **Marlin FP4** — prebuilt `.so` requires sm_80+ (BF16 MMA).
  The shipped artifacts cover sm120 only (sm80/sm89 templates
  compile-validate, but only sm120 has runtime smoke).
- **NVFP4 W4A16 FFN** — same BF16 MMA requirement.

Before 2026-07-08 we kept torch 2.9.1 / CUDA 12.8 / driver 570.x
specifically to compile-check on V100. Pinning to torch 2.12 /
CUDA 13.0 / driver 580.x freed us from that (see auto-memory
`project_hardware.md`).

## What this means for edits to these paths

- Don't add `sm_70` (or `(7, 0)`) template instantiations or
  `cuda.get_device_capability()` branches.
- Don't lower `bf16` master copies to `fp16` to "save V100 memory"
  — that path was the reason we dropped V100.
- Don't bump torch / CUDA / driver back to 2.9.x / 12.8 / 570.x
  to recover V100 compatibility — the production paths above
  won't compile there.
- `sm_75` (Turing) and `sm_100` (Blackwell datacenter / B200) are
  also out of scope. The marlin shipped `.so` covers sm120 only;
  new arch needs `scripts/build_marlin.py --arch <X>` before
  claiming it works.
- For yml flag changes in `configs/base.yml`, leave the existing
  `arch` / hardware-target values alone unless the change is
  explicitly a hardware-matrix update.
