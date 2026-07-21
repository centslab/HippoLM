"""Class-level correctness test for NVFP4Linear packed state_dict.

Historical note (2026-07-21): this file used to hold the
"NVFP4 SwiGLU vs BF16" integration tests for the non-Marlin
dequant+cuBLAS path (``ffn_nvfp4=True`` without ``ffn_nvfp4_marlin``).
The 5-scheme precision migration removed that config->SwiGLU wiring:
``ffn_precision="w4a16"`` now always builds the Marlin mode-3 kernel
(no BF16 master), so a non-Marlin NVFP4 SwiGLU is no longer
reachable via config. Those integration tests were deleted; the
mode-3 SwiGLU integration is covered by
``test_nvfp4_no_bf16_master.py::TestSwiGLUMode3``.

The dequant+cuBLAS NVFP4Linear *class* still exists (used directly,
not via SwiGLU), so its packed-state_dict roundtrip is still worth
guarding here.

Run:
    python -m pytest test/test_ffn_nvfp4.py -v
"""
from __future__ import annotations

import torch

from src.models.ops.nvfp4_linear import NVFP4Linear


def test_nvfp4_packed_state_dict_roundtrip():
    """Saving and loading the state_dict of an NVFP4Linear should
    preserve the packed buffers (so checkpoints stay small).
    """
    torch.manual_seed(0)
    layer = NVFP4Linear(128, 64, bias=True).cuda()
    layer.repack_weights()

    sd = layer.state_dict()
    # Expected keys: weight (BF16 master), packed_weight, scales, bias.
    assert "weight" in sd
    assert "packed_weight" in sd
    assert "scales" in sd
    assert "bias" in sd
    assert sd["packed_weight"].dtype == torch.uint8
    assert sd["scales"].dtype == torch.float8_e4m3fn

    # Reconstruct from the dict and verify forward matches.
    layer2 = NVFP4Linear(128, 64, bias=True).cuda()
    layer2.load_state_dict(sd)

    x = torch.randn(2, 128, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        y1 = layer(x)
        y2 = layer2(x)
    assert torch.allclose(y1, y2), "state_dict roundtrip changed output"
