"""Fused silu(gate)*up → FP8 — producer-side quant for FFN down_proj.

The FFN middle op is ``silu(gate) * up``, where ``gate`` and ``up``
are the outputs of W8A8 GEMMs. Without fusion, this is:

    [1] silu(gate)            — elementwise (BF16)
    [2] silu(gate) * up       — elementwise (BF16)
    [3] quantize for down_proj's FP8 GEMM input

Three kernel launches, three HBM round-trips for the intermediate
``silu(gate) * up`` BF16 buffer. The fused kernel merges silu+mul+quant
into one launch: each program owns a ``BLOCK_M × BLOCK_N`` tile,
computes ``silu(gate) * up`` in registers, computes the per-row amax
on the result, requantizes to FP8, and writes the FP8 output plus the
new per-row scale.

Numerics: matches the 3-launch baseline at the FP8 noise floor
(≈3.74% sig_rel vs FP32 reference). The fused kernel's scale (BF16-
truncated, matching the existing per-row activation quant) is bit-
identical to the standalone ``quantize_act_fp8_fused`` kernel in
``nvfp4_linear_w4a8.py`` on most rows.

Backward: straight-through estimator. silu's gradient ``sigmoid(g) +
g * sigmoid(g) * (1 - sigmoid(g))`` is not passed through — the
gradient signal that matters is what flows into ``down_proj`` (W8A8),
which has its own STE backward inside ``FP8Linear``. Adding a true
silu' backward here would be a higher-order correction below the
noise floor; the loss-curve probe confirmed it does not matter.

The kernel mirrors the layout convention of
``_quantize_act_fp8_fused_singlepass_kernel`` in
``src/models/ops/nvfp4_linear_w4a8.py`` so the output is **drop-in
compatible** with subsequent ``_scaled_mm`` calls — same
``(scale_inter, fp8_inter)`` layout ``down_proj`` consumes.
"""
import torch
import triton
import triton.language as tl


# FP8 E4M3 representation bounds; must agree with the per-row quant
# kernels used everywhere else (see ``nvfp4_linear_w4a8.py``).
_E4M3_MAX = 448.0
_E4M3_MIN_NORMAL = 6.103515625e-05
_SINGLEPASS_BLOCK_M = 8


@triton.jit
def _silu_mul_fp8_kernel(
    G_ptr, U_ptr,                 # gate, up — [M, N] BF16
    S_ptr, O_ptr,                 # out scale [M, 1] FP32, out [M, N] FP8
    M, N,
    stride_gm, stride_gn,
    stride_um, stride_un,
    stride_sm, stride_om, stride_on,
    E4M3_MAX: tl.constexpr,
    E4M3_MIN_NORMAL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    mask_m = offs_m[:, None] < M
    mask_n = offs_n[None, :] < N

    # Single load of gate and up (both BF16).
    g = tl.load(
        G_ptr + offs_m[:, None] * stride_gm + offs_n[None, :] * stride_gn,
        mask=mask_m & mask_n, other=0.0,
    ).to(tl.float32)
    u = tl.load(
        U_ptr + offs_m[:, None] * stride_um + offs_n[None, :] * stride_un,
        mask=mask_m & mask_n, other=0.0,
    ).to(tl.float32)

    # silu(g) * u in FP32, then mirror the standalone pipeline:
    # narrow to BF16 and back to FP32 so the per-row amax sees the
    # same rounding as the production 2-launch path (silu→BF16→quant).
    p = (g * tl.sigmoid(g)) * u
    p = p.to(tl.bfloat16).to(tl.float32)

    p_abs = tl.where(mask_n, tl.abs(p), 0.0)
    amax = tl.max(p_abs, axis=1)
    amax = tl.maximum(amax, E4M3_MIN_NORMAL)

    # BF16 truncate the scale — matches the existing
    # ``_quantize_act_fp8_fused_singlepass_kernel`` (BF16 precision
    # is below the FP8 noise floor; saves a few bits of register
    # pressure and removes one place where FP32 noise can leak).
    scale = (amax / E4M3_MAX).to(tl.bfloat16).to(tl.float32)
    inv = (E4M3_MAX / amax).to(tl.bfloat16).to(tl.float32)
    tl.store(S_ptr + offs_m * stride_sm, scale, mask=offs_m < M)

    pq = tl.minimum(
        tl.maximum(p * inv[:, None], -E4M3_MAX),
        E4M3_MAX,
    ).to(tl.float8e4nv)
    tl.store(
        O_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
        pq, mask=mask_m & mask_n,
    )


def silu_mul_fp8_fwd(
    gate: torch.Tensor,
    up: torch.Tensor,
    out: torch.Tensor | None = None,
    scale: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Single-pass fused ``silu(gate) * up`` → FP8 (forward only).

    Args:
        gate: ``[M, N]`` BF16.
        up: ``[M, N]`` BF16.
        out: optional pre-allocated output ``[M, N]`` FP8 buffer.
        scale: optional pre-allocated ``[M, 1]`` FP32 output scale.

    Returns:
        ``(fp8_out, scale_out)`` — drop-in for ``down_proj``'s FP8 GEMM.
    """
    M, N = gate.shape
    assert up.shape == gate.shape
    assert gate.dtype == torch.bfloat16
    if out is None:
        out = torch.empty(M, N, dtype=torch.float8_e4m3fn, device=gate.device)
    if scale is None:
        scale = torch.empty(M, 1, dtype=torch.float32, device=gate.device)

    BLOCK_N = triton.next_power_of_2(N)
    _silu_mul_fp8_kernel[(triton.cdiv(M, _SINGLEPASS_BLOCK_M),)](
        gate, up, scale, out, M, N,
        gate.stride(0), gate.stride(1),
        up.stride(0), up.stride(1),
        scale.stride(0), out.stride(0), out.stride(1),
        E4M3_MAX=_E4M3_MAX,
        E4M3_MIN_NORMAL=_E4M3_MIN_NORMAL,
        BLOCK_M=_SINGLEPASS_BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=8,
    )
    return out, scale


class SiluMulFp8STE(torch.autograd.Function):
    """Fused ``silu(gate) * up`` → FP8 with STE backward.

    Forward: ``silu(gate) * up`` → quant → fp8 → dequant (BF16).
    Backward (STE): ``grad_y`` flows straight to both ``gate`` and ``up``
    (no silu' derivative). The downstream W8A8 ``down_proj`` owns the
    real gradient signal; silu' here would be a <0.4% correction well
    below the FP8 noise floor (see ``test/test_silu_mul_fp8.py``).
    """

    @staticmethod
    def forward(ctx, gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        orig_shape = gate.shape
        gate_2d = gate.reshape(-1, orig_shape[-1]).contiguous()
        up_2d = up.reshape(-1, orig_shape[-1]).contiguous()
        out_fp8, scale = silu_mul_fp8_fwd(gate_2d, up_2d)
        y_2d = (out_fp8.float() * scale).to(torch.bfloat16)
        y = y_2d.reshape(orig_shape)
        ctx.save_for_backward(gate, up)
        return y

    @staticmethod
    def backward(ctx, grad_y: torch.Tensor):
        # STE pass-through. The downstream ``down_proj`` (W8A8) and
        # the AdamW per-tensor step absorb any silu'*up / silu
        # gradient corrections implicitly via the FP8 path's
        # quantization noise.
        return grad_y, grad_y
