"""Base configuration for HippoLM."""
from dataclasses import dataclass


@dataclass
class HippoConfig:
    """HippoLM model configuration."""

    # Vocabulary and embeddings
    vocab_size: int = 248320
    hidden_size: int = 1024
    tie_word_embeddings: bool = True

    # Attention (KDA)
    num_heads: int = 16
    head_dim: int = 64
    num_kv: int = 64  # Number of value heads for grouped value attention

    # Architecture depth
    num_layers: int = 32
    num_blocks: int = 8  # Block AttnRes blocks

    # FFN
    intermediate_size: int = 2736  # hidden_size * 8 / 3, rounded to multiple of 8
    ffn_bias: bool = False

    # Normalization
    rms_norm_eps: float = 1e-6

    # Position encoding
    max_seq_len: int = -1  # -1 means unlimited context in code
    use_rope: bool = False  # NoPE by design

    # Linear layer bias
    use_bias: bool = False

    # Block AttnRes
    block_size: int = 4  # layers per block = num_layers / num_blocks

    # Memory optimization
    checkpoint_every_layer: bool = False  # If True, checkpoint every layer (saves more memory)

    def __post_init__(self):
        assert self.num_layers % self.num_blocks == 0, (
            f"num_layers ({self.num_layers}) must be divisible by num_blocks ({self.num_blocks})"
        )
        self.block_size = self.num_layers // self.num_blocks
