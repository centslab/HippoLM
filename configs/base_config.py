"""Base configuration for HippoLM."""
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
