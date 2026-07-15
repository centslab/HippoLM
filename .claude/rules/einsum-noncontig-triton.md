---
paths:
  - "src/models/ops/**"
---

# `torch.einsum` output is non-contiguous — call `.contiguous()` before stride-indexed Triton kernels

`torch.einsum` (and `tensor.view` of non-trivial mem-order, and
slicing with non-unit stride) returns tensors whose stride
layout is **not** the "obvious" one PyTorch's einsum heuristic
chose. If you pass one of these into a Triton kernel that
takes `stride_*` arguments and uses them to address memory,
the kernel reads the **wrong memory** — finite output, no NaN,
numerically garbage.

## The trap

`torch.einsum("hd,nbthd->nbth", q, K)` returns shape
`(N, B, T, H)` with strides `(T*H, T*H, 1, T*H*H)`: source
dim has stride 256, head dim has stride 65536. A Triton kernel
that received the "logical" strides `(256, 256, 1)` computed
addresses that read slot 0 from source 2's logit, slot 2 from
source 0's, etc. `dV` was still finite, just numerically wrong
(relative error ~30%).

This bit the AttnRes bwd probe on 2026-07-14 — see
auto-memory `project_einsum_noncontig.md` for the incident
writeup. The probe's standalone `torch.testing.assert_close`
caught it (rel > 5%), but only because the probe happened to
exist; a smoke test would not have.

## The rule

Any tensor going into a Triton kernel that uses `stride_*`
args (or any "raw" `ptr + offset` arithmetic that assumes the
stride matches the dim order) **must be `tensor.contiguous()`
first**. The same applies to:

- `tensor.view` results whose memory order doesn't match the
  view shape.
- Any slicing with a non-unit stride (e.g. `t[::2, :, :]`,
  `t[:, 1::3, :]`).
- `torch.einsum` outputs (any equation).
- `torch.matmul` / `torch.bmm` / `torch.mm` outputs that get
  transposed in-place via `t.t()` or `t.transpose(...)` and
  reused.

Cost of the extra `.contiguous()`: one HBM-to-HBM copy of the
tensor — usually negligible compared to the kernel work.
When correctness-vs-speed is ambiguous, prefer `.contiguous()`
and measure.

## How to identify the trap in an existing kernel

If a Triton kernel is failing a numerical correctness check
(`assert_close`, max-diff, or rel-diff against a PyTorch
reference) with **finite but wrong** outputs:

1. List the kernel's tensor args.
2. For each arg, check `arg.is_contiguous()` (and
   `arg.stride()` if you need to be sure).
3. For any arg where `is_contiguous()` is False and the
   kernel uses raw `ptr + stride * idx` arithmetic (rather
   than `tl.make_block_ptr`), the bug is here.

`tl.make_block_ptr` with explicit `order=` arg sidesteps the
trap — it computes addresses from the strides itself and
doesn't need the caller to have made the tensor contiguous.
If you're using `tl.make_block_ptr`, you can drop the
`.contiguous()`. If you're using raw stride-indexed loads
(typical for non-rectangular shapes where `tl.make_block_ptr`
is awkward), keep the `.contiguous()`.

## See also

- `docs/triton_kernel_playbook.md` §5 — `tl.make_block_ptr`
  pattern (the alternative to `.contiguous()`).
- Auto-memory `project_einsum_noncontig.md` — original
  incident + worked example.
- [`saved-tensors-not-hwm`](../rules/saved-tensors-not-hwm.md) —
  sibling rule on the same path glob.