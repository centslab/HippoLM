"""Tests for the block-boundary BlockAttnRes design.

The official Kimi design uses AttnRes only between blocks
(every ``block_size`` layers); within a block the connection
is the standard residual ``x = x + Sublayer(x)``. This file
covers the new ``BlockAttnRes`` interface (``blocks`` only,
no ``partial_block``) and the surrounding layer/model
structure changes that come with it.
"""
import sys
from pathlib import Path

import torch

# Allow running this file both as ``python -m pytest test/test_attn_res.py``
# (pytest rootdir resolves the repo root) and as ``python test/test_attn_res.py``
# (in which case we add the repo root to sys.path explicitly).
_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.models import HippoConfig
from src.models.model import HippoLayer, HippoModel
from src.models.ops.attn_res import BlockAttnRes


# ---------------------------------------------------------------------------
# Small config for fast test runs.
#
# The default HippoConfig (vocab_size=248320, hidden_size=1024,
# num_heads=16) makes ``HippoModel(...)`` CPU init cost ~9 s on the
# dev box (the 0.95 GB embedding weight alone accounts for ~1.86 s in
# ``nn.init.normal_``). Three tests in this file build a HippoModel,
# so the file spends ~27 s of its ~35 s wall clock on CPU init.
#
# None of the 7 tests in this file depend on those dimensions: they
# guard the *block-boundary BlockAttnRes design* — interface
# invariants (per-layer attr absence, single top-level attn_res,
# call-once-per-boundary), the zero-pseudo-query contract (uniform
# softmax → mean-of-blocks), the within-block standard residual,
# and gradient flow end-to-end. Verified in test/_tmp/
# check_small_config_attn_res.py that this small config catches
# every regression the default config catches (7/7 assertions PASS,
# 4/4 historical-bug mutations CATCH).
#
# Constraints preserved:
#   - hidden_size == num_heads * head_dim (required by BlockAttnRes
#     assert in __init__).
#   - num_layers % num_blocks == 0 (HippoConfig.__post_init__ check).
#   - num_heads % tp_world == 0 with tp_world=1 (default).
# ---------------------------------------------------------------------------
def _small_cfg(**overrides):
    base = dict(
        vocab_size=4096,
        hidden_size=128,
        head_dim=64,
        num_heads=2,
        intermediate_size=384,
    )
    base.update(overrides)
    return HippoConfig(**base)


def test_block_attn_res_takes_only_blocks():
    """BlockAttnRes.forward accepts a list of block tensors (no partial_block)."""
    if not torch.cuda.is_available():
        print("[SKIP] test_block_attn_res_takes_only_blocks (CUDA required for Triton kernel)")
        return
    config = _small_cfg()
    module = BlockAttnRes(config).cuda()

    B, T, D = 2, 8, config.hidden_size
    blocks = [
        torch.randn(B, T, D, device="cuda"),
        torch.randn(B, T, D, device="cuda"),
    ]

    # New interface: pass the blocks list directly.
    out = module(blocks)
    assert out.shape == (B, T, D), f"Expected {(B, T, D)}, got {out.shape}"
    print("[PASS] test_block_attn_res_takes_only_blocks")


def test_block_attn_res_uniform_with_zero_query():
    """With zero pseudo-query, attention weights are uniform, output is mean(blocks)."""
    if not torch.cuda.is_available():
        print("[SKIP] test_block_attn_res_uniform_with_zero_query (CUDA required for Triton kernel)")
        return
    config = _small_cfg()
    module = BlockAttnRes(config).cuda()

    B, T, D = 1, 4, config.hidden_size
    blocks = [
        torch.randn(B, T, D, device="cuda"),
        torch.randn(B, T, D, device="cuda"),
        torch.randn(B, T, D, device="cuda"),
    ]

    module.query.data.zero_()
    out = module(blocks)

    # With uniform attention weights: out = mean(V_n) = mean(blocks).
    # K = RMSNorm(V) does not affect the values, only the attention
    # logits, which become 0 (uniform softmax) when query is zero.
    expected = sum(blocks) / len(blocks)
    assert torch.allclose(out, expected, atol=1e-5), (
        f"Expected mean of blocks with zero query, got max diff "
        f"{(out - expected).abs().max().item()}"
    )
    print("[PASS] test_block_attn_res_uniform_with_zero_query")


def test_hippo_layer_has_no_attn_res():
    """HippoLayer no longer carries attn_res or mlp_res (AttnRes lives at model level)."""
    config = _small_cfg()
    layer = HippoLayer(0, config)

    assert not hasattr(layer, "attn_res"), (
        "HippoLayer should not have attn_res in the block-boundary design"
    )
    assert not hasattr(layer, "mlp_res"), (
        "HippoLayer should not have mlp_res in the block-boundary design"
    )
    print("[PASS] test_hippo_layer_has_no_attn_res")


def test_hippo_layer_standard_residual_within_block():
    """HippoLayer.forward(x) returns x + KDA(RMSNorm(x)) + FFN(RMSNorm(...))."""
    if not torch.cuda.is_available():
        print("[SKIP] test_hippo_layer_standard_residual_within_block (CUDA required)")
        return
    config = _small_cfg()
    layer = HippoLayer(0, config).cuda()
    layer.eval()

    B, T, D = 1, 4, config.hidden_size
    x = torch.randn(B, T, D, device="cuda")

    # Manual reference computation
    h = layer.attn_norm(x)
    attn_out = layer.kda(h)
    h_after_attn = x + attn_out
    h_norm = layer.mlp_norm(h_after_attn)
    ffn_out = layer.ffn(h_norm)
    expected = h_after_attn + ffn_out

    actual = layer(x)
    assert actual.shape == (B, T, D)
    assert torch.allclose(actual, expected, atol=1e-5), (
        f"HippoLayer is not a standard residual transformer; "
        f"max diff {(actual - expected).abs().max().item()}"
    )
    print("[PASS] test_hippo_layer_standard_residual_within_block")


def test_model_has_single_attn_res():
    """HippoModel exposes a single attn_res shared across block boundaries."""
    config = _small_cfg(num_layers=8, num_blocks=2)
    model = HippoModel(config)

    assert hasattr(model, "attn_res"), "HippoModel must have top-level attn_res"
    assert isinstance(model.attn_res, BlockAttnRes), (
        f"attn_res must be BlockAttnRes, got {type(model.attn_res).__name__}"
    )

    for li, layer in enumerate(model.layers):
        assert not hasattr(layer, "attn_res"), (
            f"Layer {li} should not have attn_res in the new design"
        )
        assert not hasattr(layer, "mlp_res"), (
            f"Layer {li} should not have mlp_res in the new design"
        )
    print("[PASS] test_model_has_single_attn_res")


def test_attn_res_called_once_per_block_boundary():
    """AttnRes is called only at block boundaries: ``num_blocks - 1`` times."""
    if not torch.cuda.is_available():
        print("[SKIP] test_attn_res_called_once_per_block_boundary (CUDA required)")
        return
    config = _small_cfg(num_layers=8, num_blocks=2)  # block_size = 4
    model = HippoModel(config).cuda()
    model.eval()

    # Wrap attn_res.forward to count invocations.
    call_count = [0]
    original_forward = model.attn_res.forward

    def counting_forward(blocks):
        call_count[0] += 1
        return original_forward(blocks)

    model.attn_res.forward = counting_forward

    B, T = 1, 8
    input_ids = torch.randint(0, config.vocab_size, (B, T), device="cuda")
    model(input_ids)

    # 2 blocks -> 1 boundary between them -> 1 attn_res call.
    assert call_count[0] == 1, (
        f"Expected 1 attn_res call (one boundary, two blocks), got {call_count[0]}"
    )
    print("[PASS] test_attn_res_called_once_per_block_boundary")


def test_gradient_flow_block_boundary_design():
    """Gradients flow through the new block-boundary design end-to-end."""
    config = _small_cfg(num_layers=4, num_blocks=2)  # block_size = 2
    model = HippoModel(config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    B, T = 1, 8
    input_ids = torch.randint(0, config.vocab_size, (B, T), device=device)
    labels = torch.randint(0, config.vocab_size, (B, T), device=device)
    labels[:, -2:] = -100

    outputs = model(input_ids, labels=labels)
    loss = outputs["loss"]
    assert loss.requires_grad
    loss.backward()

    # attn_res.query must receive a non-None gradient
    assert model.attn_res.query.grad is not None, (
        "attn_res.query has no gradient"
    )

    # Every trainable parameter must receive a gradient
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Parameter {name} has no gradient"
    print("[PASS] test_gradient_flow_block_boundary_design")


def run_all_tests():
    print("Running AttnRes block-boundary design tests...")
    test_block_attn_res_takes_only_blocks()
    test_block_attn_res_uniform_with_zero_query()
    test_hippo_layer_has_no_attn_res()
    test_hippo_layer_standard_residual_within_block()
    test_model_has_single_attn_res()
    test_attn_res_called_once_per_block_boundary()
    test_gradient_flow_block_boundary_design()
    print("\nAll AttnRes tests passed!")


if __name__ == "__main__":
    run_all_tests()
