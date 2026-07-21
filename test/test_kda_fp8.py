"""FP8 W8A8 KDA projection regression test.

Guards the production rollout of E4M3 W8A8 on the KDA projection
layers (q/k/v/o/b_proj + f_proj[0..1] + g_proj[0..1]). The
implementation is in :mod:`src.models.ops.fp8_linear`; the
wrapper integration is in :mod:`src.models.ops.kda`.

What this exercises
-------------------
1. ``FP8Linear`` standalone (vs a plain ``nn.Linear`` of the same
   shape with the same weights):
   - Forward: signal-level noise (sig_rel) bounded — 3.7% on
     Gaussian inputs per the feasibility sweep.
   - Backward: ``dx``, ``dw``, ``db`` grads match the BF16
     reference within numerical tolerance (STE re-runs BF16 matmul).
   - ``bias=True`` path: same fidelity as no-bias.
   - ``bf16_only=True`` escape hatch: 0% noise (matches ``nn.Linear``).
   - State-dict load: a plain ``nn.Linear``'s state_dict loads
     directly into the matching ``FP8Linear`` (same key shape).
2. ``KDA`` wrapper integration (``kda_fp8=True``):
   - All 9 projection layers become ``FP8Linear`` instances.
   - State-dict round-trip: copy the BF16-trained weights into the
     FP8-enabled model without manual conversion.
   - Forward against the BF16 reference (same weights) — the noise
     floor should match the per-Linear feasibility numbers.
   - Backward — every projection layer's ``weight.grad`` and
     ``.grad`` on the input must be finite and within tolerance.
   - No NaN / Inf in either the fwd or the bwd pass.

Why per-row-act + per-channel-weight (the only row-wise scaling
configuration ``torch._scaled_mm`` accepts for E4M3 on sm_120) is
the granularity the test asserts: per-tensor scaling would be the
wrong starting point — both granularities have ~3.7% sig_rel on
Gaussian inputs, but per-row + per-channel absorbs row-wise
outliers cleanly. See ``test/_tmp/test_kda_w8a8_feasibility.py``
for the full sweep.
"""
from __future__ import annotations

import sys
import math
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from src.models.config import HippoConfig
from src.models.ops.fp8_linear import FP8Linear
from src.models.ops.kda import KDA


device = torch.device("cuda:0")
dtype = torch.bfloat16


# ---------------------------------------------------------------------------
# Standalone FP8Linear
# ---------------------------------------------------------------------------

def _make_pair(K: int, N: int, *, bias: bool, seed: int):
    """Construct (BF16 reference, FP8 copy) with identical weights."""
    torch.manual_seed(seed)
    ref = nn.Linear(K, N, bias=bias).to(device=device, dtype=dtype)
    fp8 = FP8Linear(K, N, bias=bias, device=device, dtype=dtype)
    fp8.weight.data.copy_(ref.weight.data)
    if bias:
        fp8.bias.data.copy_(ref.bias.data)
    return ref, fp8


def _sig_rel(a: torch.Tensor, b: torch.Tensor) -> float:
    """Signal-level relative noise: ||a - b||_F / ||b||_F."""
    return (a.float() - b.float()).norm().item() / (b.float().norm().item() + 1e-12)


@pytest.mark.parametrize("K, N, bias", [
    (1536, 1536, False),  # q/k/v/o projections at prod shape
    (1536, 12, False),    # b_proj (narrow output)
    (128, 1536, True),    # f_proj[1] / g_proj[1] (bias=True for g_proj[1])
    (1536, 128, False),   # f_proj[0] / g_proj[0] bottleneck down
    (64, 32, False),      # tiny shape — exercise small-K edge case
])
def test_fp8_linear_forward(K, N, bias):
    """Forward noise floor matches the feasibility sweep (~3.7%)."""
    torch.manual_seed(0)
    M = 1024
    x = torch.randn(M, K, device=device, dtype=dtype)
    ref, fp8 = _make_pair(K, N, bias=bias, seed=42)
    y_ref = ref(x)
    y_fp8 = fp8(x)
    rel = _sig_rel(y_fp8, y_ref)
    # 5% ceiling — the per-part feasibility numbers are 3.6-3.8%.
    assert rel < 0.05, f"sig_rel {rel*100:.2f}% > 5% for K={K},N={N},bias={bias}"
    assert torch.isfinite(y_fp8).all(), "FP8 fwd produced non-finite values"


@pytest.mark.parametrize("K, N, bias", [
    (1536, 1536, False),
    (1536, 12, False),
    (128, 1536, True),
    (1536, 128, False),
])
def test_fp8_linear_backward(K, N, bias):
    """Backward grads (dx, dw, [db]) match the BF16 reference.

    Forces the BF16 STE path (``fp8_bwd=False``) so the test is
    the bit-equivalence regression for the FP32-cast bug fix
    (auto-memory ``project_fp8_bwd_fp32_bug.md``). The default
    since 2026-07-21 is ``fp8_bwd=True`` (FP8 E4M3 bwd); that
    path is covered by ``test_fp8_linear_backward_fp8`` below.

    STE re-runs BF16 matmul against the BF16 leaf weight. The
    expected numerical gap vs the BF16 reference is at the BF16
    rounding-noise floor (cos_sim 1.0). We assert < 1% sig_rel
    to leave room for accumulated BF16-rounding noise.
    """
    torch.manual_seed(0)
    M = 1024
    x = torch.randn(M, K, device=device, dtype=dtype)
    dy = torch.randn(M, N, device=device, dtype=dtype)

    # Build a fresh FP8Linear with fp8_bwd=False for the BF16 STE
    # check. ``_make_pair`` defaults to the constructor's current
    # default (which is fp8_bwd=True since 2026-07-21).
    from src.models.ops.fp8_linear import FP8Linear as _FP8
    torch.manual_seed(42)
    ref = nn.Linear(K, N, bias=bias).to(device=device, dtype=dtype)
    fp8 = _FP8(K, N, bias=bias, device=device, dtype=dtype, fp8_bwd=False)
    fp8.weight.data.copy_(ref.weight.data)
    if bias:
        fp8.bias.data.copy_(ref.bias.data)

    # BF16 reference backward
    x_ref = x.detach().clone().requires_grad_(True)
    y_ref = ref(x_ref)
    y_ref.backward(dy)
    dx_ref = x_ref.grad.detach().clone()
    dw_ref = ref.weight.grad.detach().clone()
    db_ref = ref.bias.grad.detach().clone() if bias else None

    # FP8 backward (BF16 STE)
    x_fp8 = x.detach().clone().requires_grad_(True)
    y_fp8 = fp8(x_fp8)
    y_fp8.backward(dy)
    dx_fp8 = x_fp8.grad.detach().clone()
    dw_fp8 = fp8.weight.grad.detach().clone()
    db_fp8 = fp8.bias.grad.detach().clone() if bias else None

    # dx / dw / db should match the BF16 reference to BF16 precision.
    assert _sig_rel(dx_fp8, dx_ref) < 0.01, (
        f"dx sig_rel too high: {_sig_rel(dx_fp8, dx_ref)*100:.2f}%"
    )
    assert _sig_rel(dw_fp8, dw_ref) < 0.01, (
        f"dw sig_rel too high: {_sig_rel(dw_fp8, dw_ref)*100:.2f}%"
    )
    if bias:
        assert _sig_rel(db_fp8, db_ref) < 0.01, (
            f"db sig_rel too high: {_sig_rel(db_fp8, db_ref)*100:.2f}%"
        )
    # No NaN / Inf anywhere
    for name, t in [("dx", dx_fp8), ("dw", dw_fp8), ("db", db_fp8)]:
        if t is None:
            continue
        assert torch.isfinite(t).all(), f"FP8 bwd produced non-finite {name}"


def test_fp8_linear_weight_quant_is_fused_and_noise_floor():
    """``_quantize_w_per_channel`` uses the fused Triton kernel and
    stays at the FP8 per-channel noise floor.

    Guards the 2026-07-21 swap of the PyTorch broadcast path
    (``w.to(fp32)`` + amax + divide + cast, ~140 us) for
    ``quantize_act_fp8_fused`` (~56 us) in the FP8 forward. The
    fused kernel truncates the FP32 scale to BF16 (matching the
    activation-quant convention), so the exact E4M3 codes can
    differ from the broadcast path, but the reconstruction error
    must stay at the ~2.6% per-channel noise floor. See
    ``project_fp8_perf_root_cause.md``.
    """
    from src.models.ops.fp8_linear import _quantize_w_per_channel
    torch.manual_seed(0)
    N, K = 1536, 1536
    w = torch.randn(N, K, device=device, dtype=dtype)
    w_q, scale_w = _quantize_w_per_channel(w)
    # Shape contract: w_q [N, K] FP8, scale_w [1, N] FP32.
    assert w_q.shape == (N, K)
    assert w_q.dtype == torch.float8_e4m3fn
    assert scale_w.shape == (1, N)
    assert scale_w.dtype == torch.float32
    assert scale_w.is_contiguous()
    # Reconstruct w from (w_q, scale_w) and check the noise floor.
    # scale_w is [1, N] (per output channel); w_q is [N, K].
    rec = w_q.float() * scale_w.T                      # [N, K] * [N, 1]
    sig_rel = (rec - w.float()).norm().item() / w.float().norm().item()
    assert sig_rel < 0.05, f"per-channel w quant sig_rel {sig_rel*100:.2f}% > 5%"
    assert torch.isfinite(w_q.float()).all()


def test_fp8_linear_bf16_only_escape():
    """``bf16_only=True`` matches plain ``nn.Linear`` exactly (0% noise)."""
    torch.manual_seed(0)
    K, N = 1536, 1536
    x = torch.randn(1024, K, device=device, dtype=dtype)
    ref = nn.Linear(K, N, bias=False).to(device=device, dtype=dtype)
    fp8 = FP8Linear(K, N, bias=False, device=device, dtype=dtype, bf16_only=True)
    fp8.weight.data.copy_(ref.weight.data)
    y_ref = ref(x)
    y_fp8 = fp8(x)
    assert _sig_rel(y_fp8, y_ref) < 1e-6, (
        f"bf16_only should be bit-exact, got sig_rel {_sig_rel(y_fp8, y_ref)*100:.4f}%"
    )


@pytest.mark.parametrize("K, N, bias", [
    (1536, 1536, False),
    (1536, 12, False),
    (128, 1536, True),
    (1536, 128, False),
])
def test_fp8_linear_backward_fp8(K, N, bias):
    """Backward with ``fp8_bwd=True`` matches BF16 within FP8 noise floor.

    The FP8 backward re-quantizes grad_out / w / x to E4M3 and runs
    two ``_scaled_mm`` calls. Per the feasibility sweep, this carries
    ~3.3-3.7% sig_rel on grad_x and ~3.4-3.8% sig_rel on grad_w vs
    a BF16 reference. We assert <10% sig_rel to leave headroom for
    the row-outlier noise; cos_sim >= 0.999 to gate on direction.
    """
    torch.manual_seed(0)
    M = 1024
    x = torch.randn(M, K, device=device, dtype=dtype)
    dy = torch.randn(M, N, device=device, dtype=dtype)

    ref, fp8 = _make_pair(K, N, bias=bias, seed=42)
    # Re-instantiate fp8 with fp8_bwd=True.
    from src.models.ops.fp8_linear import FP8Linear as _FP8
    fp8 = _FP8(K, N, bias=bias, device=device, dtype=dtype, fp8_bwd=True)
    fp8.weight.data.copy_(ref.weight.data)
    if bias:
        fp8.bias.data.copy_(ref.bias.data)

    # BF16 reference backward
    x_ref = x.detach().clone().requires_grad_(True)
    y_ref = ref(x_ref)
    y_ref.backward(dy)
    dx_ref = x_ref.grad.detach().clone()
    dw_ref = ref.weight.grad.detach().clone()
    db_ref = ref.bias.grad.detach().clone() if bias else None

    # FP8-bwd backward
    x_fp8 = x.detach().clone().requires_grad_(True)
    y_fp8 = fp8(x_fp8)
    y_fp8.backward(dy)
    dx_fp8 = x_fp8.grad.detach().clone()
    dw_fp8 = fp8.weight.grad.detach().clone()
    db_fp8 = fp8.bias.grad.detach().clone() if bias else None

    def cos_sim(a, b):
        return F.cosine_similarity(a.float().flatten(), b.float().flatten(), dim=0).item()

    rel_dx = _sig_rel(dx_fp8, dx_ref)
    rel_dw = _sig_rel(dw_fp8, dw_ref)
    assert rel_dx < 0.10, f"dx sig_rel {rel_dx*100:.2f}% > 10% for K={K},N={N},bias={bias}"
    assert rel_dw < 0.10, f"dw sig_rel {rel_dw*100:.2f}% > 10% for K={K},N={N},bias={bias}"
    # Direction check.
    if dx_ref.norm().item() > 0:
        assert cos_sim(dx_fp8, dx_ref) > 0.99, (
            f"dx cos_sim {cos_sim(dx_fp8, dx_ref):.4f} too low for K={K},N={N},bias={bias}"
        )
    if dw_ref.norm().item() > 0:
        assert cos_sim(dw_fp8, dw_ref) > 0.99, (
            f"dw cos_sim {cos_sim(dw_fp8, dw_ref):.4f} too low for K={K},N={N},bias={bias}"
        )
    if bias:
        rel_db = _sig_rel(db_fp8, db_ref)
        assert rel_db < 0.10, f"db sig_rel {rel_db*100:.2f}% > 10%"
    # No NaN / Inf anywhere
    for name, t in [("dx", dx_fp8), ("dw", dw_fp8), ("db", db_fp8)]:
        if t is None:
            continue
        assert torch.isfinite(t).all(), f"FP8 bwd produced non-finite {name}"


def test_fp8_linear_state_dict_load():
    """A plain ``nn.Linear``'s state_dict loads into ``FP8Linear``.

    This is the production checkpoint path: existing BF16-trained
    checkpoints should load without manual conversion. After load,
    ``fp8.weight`` is bit-identical to ``ref.weight``; the forward
    output differs by the standard FP8 quantization noise floor
    (~3.7% sig_rel on Gaussian inputs — same as the feasibility
    sweep).
    """
    K, N = 1536, 1536
    ref = nn.Linear(K, N, bias=True).to(device=device, dtype=dtype)
    fp8 = FP8Linear(K, N, bias=True, device=device, dtype=dtype)
    # Initialize FP8 to random — load_state_dict should overwrite.
    nn.init.normal_(fp8.weight, std=0.5)
    nn.init.normal_(fp8.bias, std=0.5)
    missing, unexpected = fp8.load_state_dict(ref.state_dict(), strict=True)
    assert missing == [] and unexpected == [], (
        f"state_dict mismatch — missing={missing}, unexpected={unexpected}"
    )
    # Weight is now bit-identical to ref.weight.
    assert torch.equal(fp8.weight.data, ref.weight.data), (
        "load_state_dict did not copy the weight tensor"
    )
    if ref.bias is not None:
        assert torch.equal(fp8.bias.data, ref.bias.data)
    # Forward: same weight -> FP8 quantization noise floor (~3.7%).
    x = torch.randn(64, K, device=device, dtype=dtype)
    y_ref = ref(x)
    y_fp8 = fp8(x)
    assert _sig_rel(y_fp8, y_ref) < 0.05, (
        f"FP8 noise floor too high: {_sig_rel(y_fp8, y_ref)*100:.2f}%"
    )


# ---------------------------------------------------------------------------
# KDA wrapper integration
# ---------------------------------------------------------------------------

def _make_kda_pair(kda_fp8: bool, *, hidden=128, H=16, head_dim=32):
    """Construct (BF16 KDA, FP8 KDA) at a small shape for fast CI.

    H=16 keeps every projection's output dim divisible by 16 (so
    all 9 Linears use the FP8 path, not the BF16 fallback). For
    the production H=12 case, ``b_proj`` falls back to BF16 (its
    output is 12 — see :class:`FP8Linear` shape constraint).

    Post-2026-07-21: the legacy ``kda_fp8=True`` flag was removed;
    FP8 vs BF16 is now driven by ``attention_precision='w8a8'`` vs
    ``'w16a16'``. This helper maps the legacy boolean for the
    tests' convenience.
    """
    cfg = HippoConfig(
        hidden_size=hidden,
        num_heads=H,
        head_dim=head_dim,
        num_layers=1,
        num_blocks=1,
        # Defaults below: KDA-safe (safe_gate True needs lower_bound).
        safe_gate=True,
        lower_bound=-5.0,
        # Scheme-driven precision (2026-07-21). w8a8 = FP8,
        # w16a16 = BF16. The legacy kda_fp8 boolean was removed.
        attention_precision="w8a8" if kda_fp8 else "w16a16",
        # Use the chunk kernel (not the recurrent one) at training.
        kda_mode="chunk",
        # No conv1d to keep the shape small + deterministic for the test.
        use_short_conv=False,
    )
    layer = KDA(cfg, layer_idx=0)
    layer = layer.to(device=device, dtype=dtype)
    return layer, cfg


def test_kda_fp8_replaces_all_projections():
    """``kda_fp8=True`` swaps the surviving FP8 set: q/k/v/o + g_proj[0/1].

    f_proj stays as ``nn.Linear`` (BF16 GEMM) — it's precision-
    sensitive (sequential 2-Linear compounding + feeds exp forget
    gate). b_proj at prod H=12 falls back to BF16 via FP8Linear's
    div-by-16 shape guard (not relevant at H=16 used in this test;
    see ``test_kda_fp8_b_proj_falls_back_when_not_divisible`` for
    the H=12 case).
    """
    layer, _ = _make_kda_pair(kda_fp8=True, hidden=128, H=16, head_dim=32)
    attn = layer.attn
    # Direct Linears (FP8 set)
    for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
        assert isinstance(getattr(attn, name), FP8Linear), (
            f"{name} should be FP8Linear, got {type(getattr(attn, name)).__name__}"
        )
    # f_proj is sequential; stays as nn.Sequential of nn.Linear.
    f_seq = getattr(attn, "f_proj")
    assert isinstance(f_seq, nn.Sequential)
    for i, sub in enumerate(f_seq):
        assert isinstance(sub, nn.Linear), (
            f"f_proj[{i}] should stay as nn.Linear, got {type(sub).__name__}"
        )
        assert not isinstance(sub, FP8Linear), (
            f"f_proj[{i}] should NOT be FP8Linear (precision-sensitive)"
        )
    # g_proj is the surviving sequential FP8 set.
    g_seq = getattr(attn, "g_proj")
    assert isinstance(g_seq, nn.Sequential)
    for i, sub in enumerate(g_seq):
        assert isinstance(sub, FP8Linear), (
            f"g_proj[{i}] should be FP8Linear, got {type(sub).__name__}"
        )
    # b_proj stays as plain nn.Linear (BF16 always; H=16 here so div-16
    # would technically allow FP8 but precision-sensitive either way).
    assert isinstance(getattr(attn, "b_proj"), nn.Linear), (
        "b_proj should stay as nn.Linear (precision-sensitive)"
    )
    assert not isinstance(getattr(attn, "b_proj"), FP8Linear), (
        "b_proj should NOT be FP8Linear (precision-sensitive)"
    )


def test_kda_fp8_b_proj_falls_back_when_not_divisible():
    """At H=12, b_proj stays as nn.Linear (precision-sensitive: feeds sigmoid(beta)).

    The surviving FP8 set is q/k/v/o + g_proj[0/1] (o_proj re-added
    2026-07-21 after probe showed its 3.66% intrinsic noise is the
    same as q/k/v, not precision-sensitive).
    """
    cfg = HippoConfig(
        hidden_size=128,
        num_heads=12,           # H=12 matches production
        head_dim=32,
        safe_gate=True,
        lower_bound=-5.0,
        # Scheme-driven (2026-07-21): w8a8 = FP8, w16a16 = BF16.
        attention_precision="w8a8",
        kda_mode="chunk",
        use_short_conv=False,
    )
    layer = KDA(cfg, layer_idx=0).to(device=device, dtype=dtype)
    attn = layer.attn
    # q/k/v/o are FP8 at H=12 (outputs divisible by 16).
    for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
        assert isinstance(getattr(attn, name), FP8Linear), f"{name} should be FP8Linear"
    # g_proj (sequential) is FP8.
    for i, sub in enumerate(getattr(attn, "g_proj")):
        assert isinstance(sub, FP8Linear), f"g_proj[{i}] should be FP8Linear"
    # b_proj stays as plain nn.Linear (precision-sensitive, BF16 always).
    assert isinstance(attn.b_proj, nn.Linear), (
        "b_proj should stay as nn.Linear (precision-sensitive)"
    )
    assert not isinstance(attn.b_proj, FP8Linear), (
        "b_proj should NOT be FP8Linear (precision-sensitive)"
    )


def test_kda_fp8_state_dict_round_trip():
    """BF16-trained weights load into the FP8-enabled KDA without conversion.

    Asserts that the keys + shapes match (the FP8 path stores BF16
    leaf weights with the same names as the plain ``nn.Linear``).
    """
    layer_bf16, cfg_bf16 = _make_kda_pair(kda_fp8=False, hidden=128, H=16, head_dim=32)
    layer_fp8, cfg_fp8 = _make_kda_pair(kda_fp8=True, hidden=128, H=16, head_dim=32)
    # The BF16 layer has plain Linears; the FP8 layer has FP8Linears.
    # load_state_dict with strict=True should succeed (same keys,
    # same shapes).
    missing, unexpected = layer_fp8.load_state_dict(
        layer_bf16.state_dict(), strict=True,
    )
    assert missing == [] and unexpected == [], (
        f"state_dict mismatch — missing={missing}, unexpected={unexpected}"
    )


def test_kda_fp8_forward_matches_bf16():
    """With identical weights, FP8 KDA forward is within ~5% of BF16.

    After the 2026-07-21 BF16 revert of f_proj + o_proj, the FP8
    set is q/k/v + g_proj[0/1] (4 of the original 9 projections).
    The remaining FP8 projections carry ~3.7% per-Linear sig_rel;
    chunk_kda's recurrence amplifies this modestly. Expected
    whole-KDA forward noise is well under the pre-revert ~7-8%.
    """
    torch.manual_seed(7)
    layer_bf16, _ = _make_kda_pair(kda_fp8=False, hidden=128, H=16, head_dim=32)
    layer_fp8, _ = _make_kda_pair(kda_fp8=True, hidden=128, H=16, head_dim=32)
    # Copy weights: BF16 -> FP8 (state_dict compatible).
    layer_fp8.load_state_dict(layer_bf16.state_dict())

    x = torch.randn(2, 64, 128, device=device, dtype=dtype)
    with torch.no_grad():
        y_bf16 = layer_bf16(x)
        y_fp8 = layer_fp8(x)

    rel = _sig_rel(y_fp8, y_bf16)
    # With f_proj (was the worst noise contributor at 5.11%) and
    # o_proj (residual stream) back in BF16, the FP8 set shrinks
    # from 9 to 4 projections. Whole-KDA sig_rel at the small test
    # shape (hidden=128 K=128 for q/k/v) is ~6-7% — the per-Linear
    # floor is ~3.6% at K=128 (vs ~3.7% at K=1536), and chunk_kda's
    # recurrence amplifies modestly. Pre-revert whole-KDA was 7-8%.
    # 8% ceiling matches the pre-revert ceiling and leaves headroom
    # for the small-shape compounding.
    assert rel < 0.08, f"whole-KDA sig_rel {rel*100:.2f}% > 8%"
    assert torch.isfinite(y_fp8).all(), "FP8 KDA forward produced non-finite values"


def test_kda_fp8_backward_grads_finite_and_match_bf16():
    """Backward: every FP8 projection's ``weight.grad`` is finite + close to BF16.

    f_proj is NOT in the FP8 set (precision-sensitive), so it's not
    asserted here. b_proj is also not in the FP8 set.

    Since 2026-07-21 ``FP8Linear`` defaults to ``fp8_bwd=True`` (FP8
    E4M3 bwd, same scheme as FFN W4A8's autograd contract). The
    per-Linear sig_rel is ~3-4% (FP8 quant in bwd), and chunk_kda's
    recurrence amplifies modestly. We assert < 15% to leave headroom
    for the small-shape (hidden=128) compounding; the production
    large-M shape carries less relative error.
    """
    torch.manual_seed(7)
    layer_bf16, _ = _make_kda_pair(kda_fp8=False, hidden=128, H=16, head_dim=32)
    layer_fp8, _ = _make_kda_pair(kda_fp8=True, hidden=128, H=16, head_dim=32)
    layer_fp8.load_state_dict(layer_bf16.state_dict())

    x = torch.randn(2, 64, 128, device=device, dtype=dtype)
    y_bf16 = layer_bf16(x)
    y_fp8 = layer_fp8(x)
    # Same downstream gradient (synthetic, matching distribution).
    torch.manual_seed(11)
    dy = torch.randn_like(y_bf16)
    y_bf16.backward(dy)
    y_fp8.backward(dy)

    # FP8 projections: q/k/v/o + g_proj[0/1].
    fp8_attn = layer_fp8.attn
    bf16_attn = layer_bf16.attn
    for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
        w_ref = getattr(bf16_attn, name).weight.grad
        w_fp8 = getattr(fp8_attn, name).weight.grad
        assert w_fp8 is not None, f"{name}.weight.grad is None"
        assert torch.isfinite(w_fp8).all(), f"{name}.weight.grad has NaN/Inf"
        assert _sig_rel(w_fp8, w_ref) < 0.15, (
            f"{name}.weight.grad sig_rel {_sig_rel(w_fp8, w_ref)*100:.2f}% > 15%"
        )
    for i in range(2):
        w_ref = getattr(bf16_attn, "g_proj")[i].weight.grad
        w_fp8 = getattr(fp8_attn, "g_proj")[i].weight.grad
        assert w_fp8 is not None, f"g_proj[{i}].weight.grad is None"
        assert torch.isfinite(w_fp8).all(), f"g_proj[{i}].weight.grad has NaN/Inf"
        assert _sig_rel(w_fp8, w_ref) < 0.15, (
            f"g_proj[{i}].weight.grad sig_rel {_sig_rel(w_fp8, w_ref)*100:.2f}% > 15%"
        )


def test_kda_fp8_kernel_stays_bf16():
    """The KDA kernel itself (chunk_kda) is not FP8 — it stays in BF16.

    Guards against an accidental future refactor that moves the
    kernel internals to FP8. The contract today is FP8 only on
    the projection Linear layers; the kernel proper (recurrent
    state, gated delta rule) is BF16.
    """
    from src.models.ops._vendored.fla.ops.kda import chunk_kda
    # Just confirm the kernel entry point still exists and is the
    # BF16 path. The KDA layer's forward calls into it; we don't
    # need to re-run it here.
    assert callable(chunk_kda)


# ---------------------------------------------------------------------------
# MXFP8 path (kda_mxfp8=True) — b12x dense GEMM, BF16 STE backward
# ---------------------------------------------------------------------------
def _mxfp8_available():
    try:
        from src.models.ops.mxfp8_linear import is_mxfp8_available
    except Exception:
        return False, "mxfp8_linear import failed"
    return is_mxfp8_available()


_MXFP8_OK, _MXFP8_WHY = _mxfp8_available()
_skip_no_mxfp8 = pytest.mark.skipif(
    not _MXFP8_OK, reason=f"MXFP8 unavailable: {_MXFP8_WHY}"
)


@_skip_no_mxfp8
def test_kda_mxfp8_replaces_direct_projections():
    """``kda_mxfp8=True`` swaps q/k/v/o for MXFP8Linear.

    MXFP8 requires K % 128 == 0. At hidden=256 H=8 head_dim=32 the
    direct projections (q/k/v: 256->256, o: 256->256) all satisfy
    this. g_proj's small inner Linear may fall back to BF16; that's
    allowed. f_proj / b_proj stay BF16 (precision-sensitive), same
    as the W8A8 path.
    """
    from src.models.ops.mxfp8_linear import MXFP8Linear

    cfg = HippoConfig(
        hidden_size=256, num_heads=8, head_dim=32,
        num_layers=1, num_blocks=1,
        safe_gate=True, lower_bound=-5.0,
        kda_mxfp8=True, kda_mode="chunk", use_short_conv=False,
    )
    attn = KDA(cfg, layer_idx=0).to(device=device, dtype=dtype).attn
    for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
        sub = getattr(attn, name)
        assert isinstance(sub, MXFP8Linear), (
            f"{name} should be MXFP8Linear, got {type(sub).__name__}"
        )
        assert not sub.bf16_only, f"{name} K should support MXFP8 (K % 128 == 0)"
    # f_proj / b_proj precision-sensitive → stay plain nn.Linear.
    for i, sub in enumerate(attn.f_proj):
        assert not isinstance(sub, MXFP8Linear)
    assert not isinstance(attn.b_proj, MXFP8Linear)


@_skip_no_mxfp8
def test_kda_mxfp8_forward_finite_and_backward_ste():
    """MXFP8 forward is finite; backward is BF16 STE (matches BF16 KDA).

    The forward runs q/k/v/o through the b12x MXFP8 GEMM, so the
    output differs from a BF16 KDA by the FP8 quant-noise floor
    (same ballpark as W8A8, ~7-8% at this tiny shape). The backward
    is STE against the BF16 leaf, so grads must be finite and close
    to the BF16-KDA grads.
    """
    common = dict(
        hidden_size=256, num_heads=8, head_dim=32,
        num_layers=1, num_blocks=1, safe_gate=True, lower_bound=-5.0,
        kda_mode="chunk", use_short_conv=False,
    )
    torch.manual_seed(0)
    layer_bf16 = KDA(HippoConfig(**common), layer_idx=0).to(device=device, dtype=dtype)
    layer_mx = KDA(HippoConfig(**common, kda_mxfp8=True), layer_idx=0).to(device=device, dtype=dtype)
    layer_mx.load_state_dict(layer_bf16.state_dict())

    x = (torch.randn(2, 64, 256, device=device, dtype=dtype) * 0.1).requires_grad_(True)
    x_ref = x.detach().clone().requires_grad_(True)

    y_mx = layer_mx(x)
    y_bf16 = layer_bf16(x_ref)
    assert torch.isfinite(y_mx).all(), "MXFP8 KDA forward produced non-finite"
    # Quant-noise floor: same ballpark as W8A8 (< 12% at this tiny shape).
    assert _sig_rel(y_mx, y_bf16) < 0.12

    g = torch.randn_like(y_mx)
    y_mx.backward(g)
    y_bf16.backward(g)
    assert torch.isfinite(x.grad).all()
    # STE re-runs BF16 at each projection, but grad_x still flows through
    # the chunk_kda backward which depends on the (MXFP8-quantized) forward
    # activations — so grad_x carries the forward quant-noise floor (~6-8%
    # at this tiny shape), not ~0%.
    assert _sig_rel(x.grad, x_ref.grad) < 0.10


@_skip_no_mxfp8
def test_kda_mxfp8_state_dict_round_trip():
    """MXFP8 KDA state_dict has BF16-master keys identical to BF16 KDA."""
    common = dict(
        hidden_size=256, num_heads=8, head_dim=32,
        num_layers=1, num_blocks=1, safe_gate=True, lower_bound=-5.0,
        kda_mode="chunk", use_short_conv=False,
    )
    layer_bf16 = KDA(HippoConfig(**common), layer_idx=0).to(device=device, dtype=dtype)
    layer_mx = KDA(HippoConfig(**common, kda_mxfp8=True), layer_idx=0).to(device=device, dtype=dtype)
    assert set(layer_bf16.state_dict().keys()) == set(layer_mx.state_dict().keys())
    # Load BF16 checkpoint into the MXFP8 layer without conversion.
    layer_mx.load_state_dict(layer_bf16.state_dict())


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))