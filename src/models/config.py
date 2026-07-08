"""HippoLM model configuration.

Originally lived in :mod:`configs.base_config`; moved to
:mod:`src.models.config` in PR-7 (the model code lives under
:mod:`src.models`, so the config belongs with the model — not
in the YAML config directory which is reserved for the YAML
files driving a run).
"""
from dataclasses import dataclass
from typing import Optional


@dataclass
class HippoConfig:
    """HippoLM model configuration."""

    # Vocabulary and embeddings
    vocab_size: int = 248320
    hidden_size: int = 1024
    tie_word_embeddings: bool = True
    use_bias: bool = False

    # KDA (Kimi Delta Attention)
    num_heads: int = 16
    head_dim: int = 64
    expand_v: float = 1.0  # Value dimension expansion factor
    kda_mode: str = "chunk"  # "chunk" for training, "fused_recurrent" for inference
    use_short_conv: bool = False  # No local convolution (global KDA, NoPE)
    allow_neg_eigval: bool = False
    safe_gate: bool = False
    lower_bound: Optional[float] = None  # Required when safe_gate=True
    conv_size: int = 4
    conv_bias: bool = False

    # Architecture depth
    num_layers: int = 32
    num_blocks: int = 8  # Block AttnRes blocks

    # FFN (SwiGLU)
    intermediate_size: int = 2736  # hidden_size * 8 / 3, rounded to multiple of 8

    # Normalization
    rms_norm_eps: float = 1e-6

    # Chunk-aware FFD packing (see src/training/data/collate.py).
    # ``pack_chunk_size`` is the alignment granularity for doc
    # boundaries inside a pack: each doc is rounded up to a multiple
    # of this size so the KDA chunkwise kernel's state-reset lands
    # exactly at a doc boundary. The kernel's internal chunk_size is
    # independently fixed at 64 in the vendored fla code; the only
    # constraint is therefore that ``pack_chunk_size`` is a positive
    # multiple of 64. ``0`` is a sentinel meaning "use ``head_dim``",
    # which is the natural choice (one doc-state spans the head
    # dimension's worth of tokens per chunk on average).
    pack_chunk_size: int = 0
    # How many input docs the owner accumulates per packing window
    # before calling the packer. Higher = denser packs (FFD sees
    # more candidates) at the cost of one window's latency.
    pack_buffer_size: int = 8

    # KDA backward saved-tensor optimization. When True, the KDA
    # kernel does NOT save the per-chunk attention statistics
    # ``Aqk`` / ``Akk`` in the forward pass; instead, the backward
    # pass recomputes them via ``chunk_kda_fwd_intra``. Trades a
    # one-time intra recompute (small Triton kernel pair) for
    # 32 MiB of saved-tensor memory per KDA layer (Aqk 16 MiB +
    # Akk 16 MiB at production dims). With 30 layers in scope,
    # this saves ~1 GB of fwd peak at the cost of a few percent
    # of bwd wall-clock. Default False (legacy behavior).
    kda_skip_aqk_akk_saved: bool = False

    # W4A16 NVFP4 FFN (research path). When True, every FFN
    # linear (``gate_proj``, ``up_proj``, ``down_proj`` in the
    # single-GPU path; the column-/row-parallel equivalents in the
    # TP path) stores its weight in NVFP4 packed format
    # (E2M1 + FP8-e4m3fn 1x16 microblock scales). The forward
    # dequantizes to BF16 and runs a BF16 matmul (since PyTorch
    # 2.9.1's ``torch._scaled_mm`` NVFP4 path requires BOTH A
    # and B to be FP4-packed). The optimizer updates the BF16
    # master weight; ``repack_weights`` re-quantizes it after each
    # step. Memory saving on FFN weights alone: ~3.5x.
    # Default False (legacy BF16 path).
    ffn_nvfp4: bool = False

    # When ``ffn_nvfp4`` is True, use vLLM's Marlin FP4 kernel for
    # the forward matmul (BF16 MMA + register dequant + cp.async
    # double-buffered prefetch). ~3.5x speedup over the
    # dequant+cuBLAS path at FFN shapes on sm_120 (47-49 TFLOPS).
    # Backward stays as BF16 matmul (STE for the quantize noise).
    # Requires the prebuilt Marlin .so to be present at
    # ``src/models/ops/cuda/lib/`` (true on the 5060 Ti dev box).
    # Ignored when ``ffn_nvfp4`` is False. Default False (use the
    # safer dequant+cuBLAS path until the Marlin flow has been
    # smoke-tested on this config).
    ffn_nvfp4_marlin: bool = False

    def __post_init__(self):
        assert self.num_layers % self.num_blocks == 0, (
            f"num_layers ({self.num_layers}) must be divisible by num_blocks ({self.num_blocks})"
        )
        # Derived
        self.block_size: int = self.num_layers // self.num_blocks
        self.kv_channels: int = self.num_heads * self.head_dim
        # Validate KDA mode
        assert self.kda_mode in ("chunk", "fused_recurrent"), (
            f"kda_mode must be 'chunk' or 'fused_recurrent', got {self.kda_mode!r}"
        )
        # Packing
        if self.pack_chunk_size == 0:
            # Sentinel: defer to ``head_dim``.
            self.pack_chunk_size = self.head_dim
        assert self.pack_chunk_size > 0, (
            f"pack_chunk_size must be positive, got {self.pack_chunk_size}"
        )
        # The vendored KDA chunkwise solver is hardcoded to BT=64
        # (NC=4 sub-chunks of size 16, see
        # src/models/ops/_vendored/fla/ops/kda/chunk.py). Round
        # ``pack_chunk_size`` up to the nearest multiple of 64 so
        # the packer's alignment matches the kernel's chunking.
        if self.pack_chunk_size % 64 != 0:
            self.pack_chunk_size = ((self.pack_chunk_size + 63) // 64) * 64
        assert self.pack_buffer_size >= 1, (
            f"pack_buffer_size must be >= 1, got {self.pack_buffer_size}"
        )
        # NVFP4 FFN: currently supports only block_size=16 (the NVFP4
        # spec). The block_size is hardcoded in NVFP4Linear; the only
        # constraint here is a sanity check that the flag is a bool.
        assert isinstance(self.ffn_nvfp4, bool), (
            f"ffn_nvfp4 must be bool, got {type(self.ffn_nvfp4).__name__}"
        )
        # ffn_nvfp4_marlin must be bool, and is a no-op if ffn_nvfp4
        # is False.
        assert isinstance(self.ffn_nvfp4_marlin, bool), (
            f"ffn_nvfp4_marlin must be bool, got {type(self.ffn_nvfp4_marlin).__name__}"
        )
