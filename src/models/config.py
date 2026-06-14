"""HippoLM model configuration.

Originally lived in :mod:`configs.base_config`; moved to
:mod:`src.models.config` in PR-7 (the model code lives under
:mod:`src.models`, so the config belongs with the model — not
in the YAML config directory which is reserved for the YAML
files driving a run).
"""
from dataclasses import dataclass


@dataclass
class HippoConfig:
    """HippoLM model configuration."""

    # Vocabulary and embeddings
    vocab_size: int = 248320
    hidden_size: int = 1024
    tie_word_embeddings: bool = True
    use_bias: bool = False

    # GDN2 (Gated DeltaNet 2): KDA's scalar beta is replaced with two
    # channel-wise gates (b on the key axis, w on the value axis).
    num_heads: int = 16
    head_dim: int = 64
    expand_v: float = 1.0  # Value dimension expansion factor
    gdn2_mode: str = "chunk"  # "chunk" for training, "fused_recurrent" for inference
    use_short_conv: bool = False  # No local convolution (global GDN2, NoPE)
    allow_neg_eigval: bool = False
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
        # Validate GDN2 mode
        assert self.gdn2_mode in ("chunk", "fused_recurrent"), (
            f"gdn2_mode must be 'chunk' or 'fused_recurrent', got {self.gdn2_mode!r}"
        )
