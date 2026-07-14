"""Tests for the Triton-fused AttnRes path in BlockAttnRes.forward.

The Triton kernel is a hot-path replacement for the 5+ PyTorch ops
in BlockAttnRes.forward (stack → RMSNorm → 2 contig slices → 2 einsums).
We verify:

  1. Forward equivalence: the Triton path matches the PyTorch
     reference path on a representative set of shapes (different
     N, B, T, H, D_h).

  2. Backward equivalence: gradients on V, query, norm_weight
     match between the Triton and PyTorch paths (the autograd
     Function uses PyTorch ops for backward, but we want to verify
     the integration doesn't break the existing path).

  3. Numerical stability: at extreme inputs (very large/small
     values, exact RMSNorm boundaries), the Triton path matches
     the PyTorch path.

  4. Boundary cases: T not divisible by typical tile sizes, very
     small N, very small T, etc.

The numerical agreement between Triton and PyTorch is bounded by
BF16 reduction-order differences (typically <1% relative). This is
acceptable because:
  - The model's own tests (test_attn_res.py) bound the gap tighter
    via end-to-end training-step validation.
  - The reference PyTorch path itself has BF16 reduction noise.
"""
from __future__ import annotations

import pytest
import torch

from src.models.config import HippoConfig
from src.models.ops.attn_res import BlockAttnRes
from src.models.ops.attn_res_triton import (
    fused_attn_res_compute,
    fused_attn_res_forward,
    is_available,
)


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Triton kernels require CUDA",
)


# ---------------------------------------------------------------------------
# Reference: same math as BlockAttnRes.forward (post-stack).
# ---------------------------------------------------------------------------
def _reference_compute(
    V_full: torch.Tensor,    # [N, B, T, D] bf16
    V_local: torch.Tensor,   # [N, B, T, H, D_h] bf16 (slice, == V_full at TP=1)
    query: torch.Tensor,     # [H, D_h] bf16
    norm_weight: torch.Tensor,  # [D] bf16
    eps: float = 1e-6,
) -> torch.Tensor:
    N, B, T, D = V_full.shape
    _, _, _, H, D_h = V_local.shape
    var = V_full.float().pow(2).mean(dim=-1, keepdim=True)
    K = V_full.float() * torch.rsqrt(var + eps) * norm_weight.float()
    K = K.view(N, B, T, H, D_h).to(V_full.dtype)
    logits = torch.einsum("hd,nbthd->nbth", query, K)
    weights = torch.softmax(logits, dim=0)
    out = torch.einsum("nbth,nbthd->bthd", weights, V_local)
    return out


# Production chunk shapes from configs/base.yml + boundary cases.
_SHAPES = [
    # (N, B, T, H, D_h) — N is num_blocks, the block depth at AttnRes.
    (2, 1, 1, 2, 4),         # degenerate: 1 token, 1 batch, 2 blocks
    (2, 1, 16, 4, 8),        # small
    (4, 1, 128, 8, 16),      # medium
    (8, 1, 16384, 12, 128),  # prod (num_blocks=8)
    (4, 2, 4096, 12, 128),   # larger B
    (8, 1, 1, 12, 128),      # T=1 (degenerate — only 1 token)
    (2, 1, 7, 12, 128),      # T=7 (not power of 2; tests the
                             # BLOCK_T=1 program-per-token layout)
]


@pytest.mark.parametrize("N, B, T, H, D_h", _SHAPES)
def test_fused_forward_matches_reference(N, B, T, H, D_h):
    """Triton fused AttnRes matches the PyTorch reference on forward."""
    if not is_available():
        pytest.skip("Triton fused path not available")

    torch.manual_seed(N * 31 + B * 7 + T)
    D = H * D_h
    device = "cuda"

    blocks_data = [
        torch.randn(B, T, D, dtype=torch.bfloat16, device=device)
        for _ in range(N)
    ]
    # Stack once outside (the model does this internally).
    V_full = torch.stack(blocks_data, dim=0).contiguous()
    # At TP=1, V_local = V_full (the slice is identity).
    V_local = V_full.view(N, B, T, H, D_h)
    query = torch.randn(H, D_h, dtype=torch.bfloat16, device=device) * 0.02
    norm_weight = torch.ones(D, dtype=torch.bfloat16, device=device)

    ref_out = _reference_compute(V_full, V_local, query, norm_weight)
    fused_out = fused_attn_res_compute(V_full, V_local, query, norm_weight)

    assert fused_out.shape == ref_out.shape, (
        f"shape mismatch: {fused_out.shape} vs {ref_out.shape}"
    )
    # BF16 reduction order differs slightly; allow up to 2% relative.
    max_abs = (fused_out - ref_out).abs().max().item()
    max_rel = max_abs / (ref_out.abs().max().item() + 1e-9)
    assert max_rel < 0.02, (
        f"N={N} B={B} T={T} H={H} D_h={D_h}: "
        f"max abs diff = {max_abs:.4e}, max rel = {max_rel:.4e}"
    )


@pytest.mark.parametrize("N, B, T, H, D_h", _SHAPES)
def test_fused_forward_via_module_matches_reference(N, B, T, H, D_h):
    """BlockAttnRes.forward (which now uses Triton) matches a
    hand-written reference.

    This is the integration test: it goes through the full forward
    call path including the ``fused_attn_res_forward`` autograd
    wrapper. The all-gather is skipped (TP=1 → identity).

    Module is cast to bf16 to match the production TPHippoModel
    build path (``weight_dtype=bf16`` from ``precision_config``).
    Without this, ``self.query``/``self.norm.weight`` stay fp32 and
    mismatch the bf16 block tensors (PyTorch einsum rejects mixed
    fp32/bf16 in the same op).
    """
    if not is_available():
        pytest.skip("Triton fused path not available")

    torch.manual_seed(N * 17 + B * 5 + T * 3)
    D = H * D_h
    device = "cuda"

    config = HippoConfig(
        hidden_size=D,
        num_heads=H,
        head_dim=D_h,
        rms_norm_eps=1e-6,
    )
    module = BlockAttnRes(config).to(device).to(torch.bfloat16)

    # Build blocks and a hand-written reference.
    blocks_data = [
        torch.randn(B, T, D, dtype=torch.bfloat16, device=device)
        for _ in range(N)
    ]
    # Run the module (Triton path).
    out_module = module(list(blocks_data))
    # Hand reference (post-stack, post-slice — TP=1 so slice is identity).
    V_full = torch.stack(blocks_data, dim=0).contiguous()
    V_local_per_head = V_full.view(N, B, T, H, D_h)
    ref_out = _reference_compute(
        V_full, V_local_per_head, module.query, module.norm.weight,
    )

    # Module returns [B, T, D] (post all-gather; at TP=1 this is the
    # full hidden). The reference returns [B, T, H, D_h]. Reshape one
    # to the other for comparison.
    ref_out_flat = ref_out.reshape(B, T, D)
    assert out_module.shape == ref_out_flat.shape, (
        f"shape mismatch: module={out_module.shape} vs ref={ref_out_flat.shape}"
    )
    max_abs = (out_module - ref_out_flat).abs().max().item()
    max_rel = max_abs / (ref_out_flat.abs().max().item() + 1e-9)
    assert max_rel < 0.02, (
        f"N={N} B={B} T={T} H={H} D_h={D_h}: "
        f"module output diverges from reference, max rel = {max_rel:.4e}"
    )


@pytest.mark.parametrize("N, B, T, H, D_h", [
    (4, 1, 256, 8, 32),    # small enough that bwd cost is small
    (8, 1, 1024, 12, 64),  # medium
])
def test_fused_backward_matches_reference(N, B, T, H, D_h):
    """Backward gradients on V, query, norm_weight match between
    Triton-fused forward + PyTorch backward (via the autograd
    Function) and the pure-PyTorch reference.

    The autograd Function's backward re-runs the reference math in
    ``torch.enable_grad()`` and uses PyTorch autograd to compute the
    gradients. This test verifies the integration doesn't break the
    gradient flow.
    """
    if not is_available():
        pytest.skip("Triton fused path not available")

    torch.manual_seed(N * 41 + B * 11 + T * 5)
    D = H * D_h
    device = "cuda"

    # Build leaves.
    V_full = torch.randn(N, B, T, D, dtype=torch.bfloat16, device=device)
    V_full.requires_grad_(True)
    query = torch.randn(H, D_h, dtype=torch.bfloat16, device=device) * 0.02
    query.requires_grad_(True)
    norm_weight = torch.ones(D, dtype=torch.bfloat16, device=device)
    norm_weight.requires_grad_(True)
    V_local = V_full.view(N, B, T, H, D_h)

    grad_out = torch.randn(B, T, H, D_h, dtype=torch.bfloat16, device=device)

    # Triton path (uses the autograd Function).
    out_fused = fused_attn_res_forward(
        V_full, V_local, query, norm_weight,
    )
    out_fused.backward(grad_out)
    d_V_fused = V_full.grad.detach().clone()
    d_q_fused = query.grad.detach().clone()
    d_w_fused = norm_weight.grad.detach().clone()
    V_full.grad = None
    query.grad = None
    norm_weight.grad = None

    # PyTorch reference path.
    var = V_full.float().pow(2).mean(dim=-1, keepdim=True)
    K = V_full.float() * torch.rsqrt(var + 1e-6) * norm_weight.float()
    K = K.view(N, B, T, H, D_h).to(V_full.dtype)
    logits = torch.einsum("hd,nbthd->nbth", query, K)
    weights = torch.softmax(logits, dim=0)
    out_ref = torch.einsum("nbth,nbthd->bthd", weights, V_local)
    out_ref.backward(grad_out)
    d_V_ref = V_full.grad.detach().clone()
    d_q_ref = query.grad.detach().clone()
    d_w_ref = norm_weight.grad.detach().clone()

    # Compare. BF16 reduction order differs slightly; allow 5% rel.
    for name, fused, ref in [
        ("d_V", d_V_fused, d_V_ref),
        ("d_query", d_q_fused, d_q_ref),
        ("d_norm_weight", d_w_fused, d_w_ref),
    ]:
        max_abs = (fused - ref).abs().max().item()
        max_rel = max_abs / (ref.abs().max().item() + 1e-9)
        assert max_rel < 0.05, (
            f"N={N} B={B} T={T} H={H} D_h={D_h}: {name} "
            f"max abs diff = {max_abs:.4e}, max rel = {max_rel:.4e}"
        )


def test_uniform_with_zero_query_triton():
    """With zero pseudo-query, weights are uniform; output = mean(V)."""
    if not is_available():
        pytest.skip("Triton fused path not available")

    torch.manual_seed(0)
    N, B, T, H, D_h = 4, 1, 64, 4, 8
    D = H * D_h
    device = "cuda"

    blocks = [
        torch.randn(B, T, D, dtype=torch.bfloat16, device=device)
        for _ in range(N)
    ]
    config = HippoConfig(
        hidden_size=D, num_heads=H, head_dim=D_h, rms_norm_eps=1e-6,
    )
    module = BlockAttnRes(config).to(device).to(torch.bfloat16)
    module.query.data.zero_()

    out = module(blocks)
    expected = sum(blocks) / len(blocks)  # mean of V
    # The RMSNorm on K doesn't affect V, so the weighted sum with
    # uniform weights gives mean(V).
    assert torch.allclose(out, expected, atol=1e-2), (
        f"zero-query output != mean(V): "
        f"max diff = {(out - expected).abs().max().item():.4e}"
    )


def test_triton_path_handles_extreme_inputs():
    """Inputs near or beyond the RMSNorm's stable range should not
    crash and should match the PyTorch reference within BF16
    reduction noise.

    Verifies the kernel's IEEE division (default Triton fp32 div)
    survives the pathological-but-valid cases.
    """
    if not is_available():
        pytest.skip("Triton fused path not available")

    torch.manual_seed(99)
    N, B, T, H, D_h = 4, 1, 256, 4, 32
    D = H * D_h
    device = "cuda"

    # Crafted inputs: large magnitude, small magnitude, zero.
    base = torch.randn(B, T, D, dtype=torch.float32, device=device)
    blocks = []
    for n in range(N):
        scale = 10.0 ** (n - 2)  # spans 0.01 to 1000
        blocks.append((base * scale).to(torch.bfloat16))

    V_full = torch.stack(blocks, dim=0).contiguous()
    V_local = V_full.view(N, B, T, H, D_h)
    query = torch.randn(H, D_h, dtype=torch.bfloat16, device=device) * 0.02
    norm_weight = torch.ones(D, dtype=torch.bfloat16, device=device)

    ref_out = _reference_compute(V_full, V_local, query, norm_weight)
    fused_out = fused_attn_res_compute(V_full, V_local, query, norm_weight)

    max_abs = (fused_out - ref_out).abs().max().item()
    max_rel = max_abs / (ref_out.abs().max().item() + 1e-9)
    assert max_rel < 0.05, (
        f"extreme inputs diverged: max rel = {max_rel:.4e}"
    )


def test_fallback_works_when_triton_unavailable():
    """The PyTorch fallback inside ``fused_attn_res_forward`` works
    for non-CUDA inputs (simulates Triton-unavailable path).

    We force the fallback by calling with a CPU tensor — the
    integration checks ``is_cuda`` and routes to PyTorch.
    """
    if not torch.cuda.is_available():
        pytest.skip("Need CUDA to construct the bf16 input")

    torch.manual_seed(0)
    N, B, T, H, D_h = 4, 1, 64, 4, 8
    D = H * D_h

    V_full = torch.randn(N, B, T, D, dtype=torch.bfloat16, device="cpu")
    V_local = V_full.view(N, B, T, H, D_h)
    query = torch.randn(H, D_h, dtype=torch.bfloat16, device="cpu") * 0.02
    norm_weight = torch.ones(D, dtype=torch.bfloat16, device="cpu")

    # CPU input forces the PyTorch fallback (Triton requires CUDA).
    out = fused_attn_res_forward(V_full, V_local, query, norm_weight)
    assert out.shape == (B, T, H, D_h)
    assert out.dtype == torch.bfloat16
    # The fallback uses the same math as the reference; should be
    # bit-identical (we re-run the same ops).
    ref = _reference_compute(V_full, V_local, query, norm_weight)
    assert torch.equal(out, ref), "fallback diverged from reference"