"""Tests for HippoLM model architecture."""
import sys
from pathlib import Path

import torch

# Resolve repo root whether the file is run via pytest (auto-resolves
# rootdir) or directly via ``python test/test_model.py`` (which needs
# the repo on sys.path for ``from src...`` imports to work).
_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.models import HippoConfig
from src.models.model import HippoModel, HippoLayer
from src.models.kda import KDA
from src.models.ops.attn_res import BlockAttnRes


def test_kda_shape():
    """Test KDA produces correct output shape."""
    config = HippoConfig()
    module = KDA(config)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    module = module.to(device)

    B, T = 2, 16
    x = torch.randn(B, T, config.hidden_size, device=device)

    out = module(x)
    assert out.shape == (B, T, config.hidden_size), f"Expected {(B, T, config.hidden_size)}, got {out.shape}"
    print("[PASS] test_kda_shape")


def test_block_attn_res():
    """BlockAttnRes takes only ``blocks`` (no partial_block)."""
    if not torch.cuda.is_available():
        print("[SKIP] test_block_attn_res (CUDA required for Triton kernel in BlockAttnRes.norm)")
        return
    config = HippoConfig()
    module = BlockAttnRes(config).cuda()

    B, T, D = 2, 8, config.hidden_size
    blocks = [
        torch.randn(B, T, D, device="cuda"),
        torch.randn(B, T, D, device="cuda"),
    ]

    out = module(blocks)
    assert out.shape == (B, T, D)

    # With zero query, weights should be uniform -> mean of blocks.
    module.query.data.zero_()
    out_zero = module(blocks)
    expected = sum(blocks) / len(blocks)
    assert torch.allclose(out_zero, expected, atol=1e-5)

    print("[PASS] test_block_attn_res")


def test_model_forward():
    """Test full model forward pass."""
    # Use small config for fast testing
    config = HippoConfig(num_layers=4, num_blocks=2)
    model = HippoModel(config)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    B, T = 2, 32
    input_ids = torch.randint(0, config.vocab_size, (B, T), device=device)

    outputs = model(input_ids)
    logits = outputs["logits"]

    assert logits.shape == (B, T, config.vocab_size)
    print("[PASS] test_model_forward")


def test_gradient_flow():
    """Test that gradients flow through all layers."""
    config = HippoConfig(num_layers=4, num_blocks=2)
    model = HippoModel(config)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    B, T = 2, 16
    input_ids = torch.randint(0, config.vocab_size, (B, T), device=device)
    labels = torch.randint(0, config.vocab_size, (B, T), device=device)
    labels[:, -5:] = -100  # Test ignore_index

    outputs = model(input_ids, labels=labels)
    loss = outputs["loss"]

    assert loss.requires_grad
    loss.backward()

    # Check that all parameters have gradients (some may be zero at init)
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Parameter {name} has no gradient"

    print("[PASS] test_gradient_flow")


def test_tied_weights():
    """Test that embedding and lm_head share weights."""
    config = HippoConfig(tie_word_embeddings=True)
    model = HippoModel(config)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    assert model.lm_head.weight is model.embed_tokens.weight
    print("[PASS] test_tied_weights")


def test_kda_different_inputs():
    """Test that KDA produces different outputs for different inputs."""
    config = HippoConfig()
    module = KDA(config)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    module = module.to(device)

    B, T = 1, 4
    x = torch.randn(B, T, config.hidden_size, device=device)

    # Run forward
    out1 = module(x)

    # Run again with different input
    x2 = torch.randn(B, T, config.hidden_size, device=device)

    out2 = module(x2)

    # Outputs should differ due to different inputs
    assert not torch.allclose(out1, out2)
    print("[PASS] test_kda_different_inputs")


def test_block_boundary_attn_res_invoked():
    """AttnRes is invoked at every non-first block boundary in HippoModel.forward.

    With ``num_blocks=2`` there is one boundary (between block 0 and
    block 1). Block 0 starts from the embedding (no AttnRes); block 1
    starts from ``attn_res([b_0])`` — exactly one call.
    """
    if not torch.cuda.is_available():
        print("[SKIP] test_block_boundary_attn_res_invoked (CUDA required)")
        return
    config = HippoConfig(num_layers=4, num_blocks=2)  # block_size = 2
    model = HippoModel(config).cuda()
    model.eval()

    call_count = [0]
    original_forward = model.attn_res.forward

    def counting_forward(blocks):
        call_count[0] += 1
        return original_forward(blocks)

    model.attn_res.forward = counting_forward

    B, T = 1, 8
    input_ids = torch.randint(0, config.vocab_size, (B, T), device="cuda")
    model(input_ids)

    assert call_count[0] == 1, (
        f"Expected exactly 1 attn_res call (one boundary, two blocks), "
        f"got {call_count[0]}"
    )
    print("[PASS] test_block_boundary_attn_res_invoked")


def run_all_tests():
    print("Running HippoLM 0.0.0 tests...")
    test_kda_shape()
    test_block_attn_res()
    test_model_forward()
    test_gradient_flow()
    test_tied_weights()
    test_kda_different_inputs()
    test_block_boundary_attn_res_invoked()
    print("\nAll tests passed!")


if __name__ == "__main__":
    run_all_tests()
