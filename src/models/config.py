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

    # EFKDA: KDA with EFLA closed-form alpha (Lei et al., 2025; arXiv
    # 2512.12602). Replaces the KDA Euler-step recurrence
    #     S_t = (I - β_t k_t kᵀ) Diag(exp(g_t)) S_{t-1} + β_t k_t v_tᵀ
    # with the rank-1 closed form
    #     α_t = (1 - exp(-β_t ||k_t||²)) / ||k_t||²
    #     S_t = (I - α_t k_t k_tᵀ) D S_{t-1} + α_t k_t v_tᵀ
    # The closed-form α absorbs the matrix exponential of the rank-1
    # dynamics exactly (vs the small-β Euler approximation), so the
    # recurrence is the KDA recurrence re-expressed — not a different
    # recurrence. The wrapper (src/models/ops/efkda.py) L2-normalises
    # q and k so ||k||=1 and α simplifies to (1 - exp(-β)).
    num_heads: int = 16
    head_dim: int = 64
    expand_v: float = 1.0  # Value dimension expansion factor
    # ``efkda_kernel`` selects the production kernel backend:
    #   "triton" — the custom Triton kernel for forward, PyTorch
    #              reference for backward (via _EFKDAChunkFn autograd).
    #              This is the default; it is the production path.
    #   "ref"    — pure-PyTorch reference (slow but stable; for
    #              debugging numerical regressions). Lives at
    #              src/models/ops/_vendored/fla/ops/kda/chunk_efla_naive.py.
    efkda_kernel: str = "triton"
    efkda_mode: str = "chunk"  # reserved; only "chunk" is implemented.
    # Legacy GDN2 fields kept for backward compat (the old GDN2 path
    # is no longer wired into the model, but downstream callers may
    # still read these via HippoConfig / base.yml). New code should
    # use ``efkda_kernel`` instead.
    gdn2_mode: str = "chunk"
    use_short_conv: bool = False  # No local convolution (global EFKDA, NoPE)
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

    # Chunk-aware FFD packing (see src/training/data/collate.py).
    # ``pack_chunk_size`` is the alignment granularity for doc
    # boundaries inside a pack: each doc is rounded up to a multiple
    # of this size so the GDN2 chunkwise kernel's state-reset lands
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

    def __post_init__(self):
        assert self.num_layers % self.num_blocks == 0, (
            f"num_layers ({self.num_layers}) must be divisible by num_blocks ({self.num_blocks})"
        )
        # Derived
        self.block_size: int = self.num_layers // self.num_blocks
        self.kv_channels: int = self.num_heads * self.head_dim
        # Validate GDN2 mode (legacy; kept for backward compat)
        assert self.gdn2_mode in ("chunk", "fused_recurrent"), (
            f"gdn2_mode must be 'chunk' or 'fused_recurrent', got {self.gdn2_mode!r}"
        )
        # Validate EFKDA backend selection
        assert self.efkda_kernel in ("triton", "ref"), (
            f"efkda_kernel must be 'triton' or 'ref', got {self.efkda_kernel!r}"
        )
        assert self.efkda_mode == "chunk", (
            f"efkda_mode must be 'chunk' (only the chunkwise path is implemented), "
            f"got {self.efkda_mode!r}"
        )
        # Packing
        if self.pack_chunk_size == 0:
            # Sentinel: defer to ``head_dim``.
            self.pack_chunk_size = self.head_dim
        assert self.pack_chunk_size > 0, (
            f"pack_chunk_size must be positive, got {self.pack_chunk_size}"
        )
        # The vendored GDN2 chunkwise solver is hardcoded to BT=64
        # (NC=4 sub-chunks of size 16, see
        # src/models/ops/_vendored/fla/ops/gdn2/chunk.py). Round
        # ``pack_chunk_size`` up to the nearest multiple of 64 so
        # the packer's alignment matches the kernel's chunking.
        if self.pack_chunk_size % 64 != 0:
            self.pack_chunk_size = ((self.pack_chunk_size + 63) // 64) * 64
        assert self.pack_buffer_size >= 1, (
            f"pack_buffer_size must be >= 1, got {self.pack_buffer_size}"
        )
