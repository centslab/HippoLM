"""Tests for HippoLM model architecture."""
import sys
import torch

sys.path.insert(0, "/home/wlx/HippoLM")

from configs.base_config import HippoConfig
from src.models.model import HippoModel, HippoLayer
from src.models.kv_dla import kvDLA
from src.models.block_attn_res import BlockAttnRes


def test_kv_dla_shape():
    """Test kvDLA produces correct output shape."""
    config = HippoConfig()
    module = kvDLA(config)

    B, T = 2, 16
    x = torch.randn(B, T, config.hidden_size)

    out = module(x)
    assert out.shape == (B, T, config.hidden_size), f"Expected {(B, T, config.hidden_size)}, got {out.shape}"
    print("[PASS] test_kv_dla_shape")


def test_block_attn_res():
    """Test BlockAttnRes produces correct output shape and weights."""
    config = HippoConfig()
    module = BlockAttnRes(config)

    B, T, D = 2, 8, config.hidden_size
    blocks = [
        torch.randn(B, T, D),
        torch.randn(B, T, D),
    ]
    partial = torch.randn(B, T, D)

    out = module(blocks, partial)
    assert out.shape == (B, T, D)

    # With zero query, weights should be uniform
    module.query.data.zero_()
    out_zero = module(blocks, partial)
    # Should be close to average of inputs
    expected = (blocks[0] + blocks[1] + partial) / 3.0
    assert torch.allclose(out_zero, expected, atol=1e-5)

    print("[PASS] test_block_attn_res")


def test_model_forward():
    """Test full model forward pass."""
    config = HippoConfig()
    model = HippoModel(config)

    B, T = 2, 32
    input_ids = torch.randint(0, config.vocab_size, (B, T))

    outputs = model(input_ids)
    logits = outputs["logits"]

    assert logits.shape == (B, T, config.vocab_size)
    print("[PASS] test_model_forward")


def test_gradient_flow():
    """Test that gradients flow through all layers."""
    config = HippoConfig()
    model = HippoModel(config)

    B, T = 2, 16
    input_ids = torch.randint(0, config.vocab_size, (B, T))
    labels = torch.randint(0, config.vocab_size, (B, T))
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

    assert model.lm_head.weight is model.embed_tokens.weight
    print("[PASS] test_tied_weights")


def test_kv_dla_state_update():
    """Test that kvDLA states actually update during forward."""
    config = HippoConfig()
    module = kvDLA(config)

    B, T = 1, 4
    x = torch.randn(B, T, config.hidden_size)

    # Run forward
    out1 = module(x)

    # Run again with different input
    x2 = torch.randn(B, T, config.hidden_size)
    out2 = module(x2)

    # Outputs should differ due to different inputs
    assert not torch.allclose(out1, out2)
    print("[PASS] test_kv_dla_state_update")


def test_layer_boundary():
    """Test that block boundaries are handled correctly."""
    config = HippoConfig(num_layers=8, num_blocks=2)  # block_size = 4
    layer3 = HippoLayer(3, config)  # 4th layer -> boundary

    D = config.hidden_size
    B, T = 2, 8
    blocks = [torch.randn(B, T, D)]
    partial = torch.randn(B, T, D)

    out_blocks, out_partial = layer3(blocks, partial)

    # Layer 3 is a boundary (layer_number=4, block_size=4)
    # Before attention, partial should be appended to blocks
    assert len(out_blocks) == 2, f"Expected 2 blocks after boundary, got {len(out_blocks)}"
    # partial_block should be reset to None, then become attn+ffn output
    assert out_partial is not None
    print("[PASS] test_layer_boundary")


def run_all_tests():
    print("Running HippoLM 0.0.0 tests...")
    test_kv_dla_shape()
    test_block_attn_res()
    test_model_forward()
    test_gradient_flow()
    test_tied_weights()
    test_kv_dla_state_update()
    test_layer_boundary()
    print("\nAll tests passed!")


if __name__ == "__main__":
    run_all_tests()
