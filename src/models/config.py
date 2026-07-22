"""HippoLM model configuration.

Originally lived in :mod:`configs.base_config`; moved to
:mod:`src.models.config` in PR-7 (the model code lives under
:mod:`src.models`, so the config belongs with the model — not
in the YAML config directory which is reserved for the YAML
files driving a run).
"""
from dataclasses import dataclass, field
from typing import Literal, Optional


# The 5 quantization schemes supported by HippoConfig (2026-07-21).
# See the long comment block above the precision fields below for
# the per-scheme spec (weight storage + GEMM precision + grad
# precision + kernel choice). Listed once at module scope so the
# KDA / FFN / embedding wrappers can iterate over it (e.g. for
# the "all schemes except the default produce a warning" test).
SCHEMES: tuple[str, ...] = ("w16a16", "w8a16", "w8a8", "w4a16", "w4a8")


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

    # ---------------------------------------------------------------------
    # Scheme-driven precision config (2026-07-21).
    # ---------------------------------------------------------------------
    # Each module is assigned one of FIVE quantization schemes. The
    # scheme uniquely determines:
    #
    #   1. The persistent weight-storage format (BF16 / FP8 / NVFP4).
    #   2. The forward GEMM precision (BF16 / FP8 tensor cores).
    #   3. The backward GEMM precision (BF16 / FP8 tensor cores).
    #   4. The gradient-storage precision (BF16 / FP8 E4M3).
    #   5. The kernel choice (the kernel is fixed per scheme; no per-
    #      config knobs).
    #
    # The five schemes (W = weight storage, a = activation/GEMM/grad
    # precision):
    #
    #   * ``w16a16`` — pure BF16. No quantization anywhere. Default.
    #                  Kernel: ``nn.Linear`` (cuBLAS BF16 GEMM).
    #                  Storage: BF16 master weight. Bwd: BF16 STE.
    #                  Grad: BF16.
    #
    #   * ``w8a16``  — weights stored as FP8 (per-output-channel E4M3
    #                  with FP32 scale), forward AND backward use BF16
    #                  tensor cores (dequant-to-BF16 on the fly), grads
    #                  BF16. **No kernel on sm_120 today** — for now
    #                  this scheme falls back to ``w16a16`` (pure BF16)
    #                  with a logged warning in the wrapper. Listed in
    #                  the enum for completeness; the canonical
    #                  ``base.yml`` config does not use it.
    #
    #   * ``w8a8``   — W8A8. Weights stored as FP8 (per-row act +
    #                  per-output-channel weight E4M3 with FP32
    #                  scales), forward AND backward use FP8 tensor
    #                  cores via ``torch._scaled_mm``, grads FP8
    #                  E4M3. Kernel: ``FP8Linear(fp8_bwd=True)``
    #                  (``_FP8E4M3MatmulFP8Bwd``). The same scheme as
    #                  the FFN W4A8 autograd contract.
    #
    #   * ``w4a8``   — W4A8. Weights stored as NVFP4 packed (E2M1 +
    #                  FP8-e4m3fn 1x16 microblock scales + global_scale
    #                  scalar), forward dequant-to-FP8 + FP8 GEMM
    #                  (``torch._scaled_mm``), backward BF16 STE
    #                  (re-runs BF16 matmul against the BF16 leaf).
    #                  **Spec gap**: the w4a8 contract says "FP8 bwd +
    #                  FP8 grads" but the current implementation uses
    #                  BF16 STE bwd (BF16 grads). Tracked as TODO.
    #                  Kernel: ``NVFP4LinearW4A8`` (two-pass Triton
    #                  dequant + ``_scaled_mm``). MXFP4 is NOT
    #                  supported (different GEMM kernel — not
    #                  maintained).
    #
    #   * ``w4a16``  — W4A16. Weights stored as NVFP4 packed, forward
    #                  AND backward use BF16 tensor cores, grads BF16.
    #                  Kernel: ``NVFP4Linear`` with Marlin mode-3
    #                  (``use_marlin=True, no_bf16_master=True`` —
    #                  these are HARD-CODED in the wrapper, not user-
    #                  controllable). FP4 packed buffers are the
    #                  source of truth; the optimizer updates at the
    #                  FP4 level via chunked material/commit. MXFP4
    #                  is NOT supported.
    #
    # Module × scheme support matrix
    # -------------------------------
    #   embedding_precision: only ``w16a16`` is wired today. The other
    #                         four fall back to ``w16a16`` with a
    #                         warning (storage-only Linear layers
    #                         don't need quantization; the embed is
    #                         typically the biggest single buffer and
    #                         quantization of it isn't on the perf
    #                         critical path).
    #   attention_precision: ``w16a16`` ✓, ``w8a8`` ✓. ``w8a16``,
    #                         ``w4a8``, ``w4a16`` fall back: ``w8a16``
    #                         → ``w16a16`` (no BF16-GEMM W8A16 KDA
    #                         kernel), ``w4a8`` → ``w8a8`` (no NVFP4
    #                         KDA kernel), ``w4a16`` → ``w8a16``
    #                         (no NVFP4 KDA kernel). All fall-backs
    #                         emit a RuntimeWarning.
    #   ffn_precision:       all five wired:
    #                           w16a16 → nn.Linear
    #                           w8a8   → FP8Linear(fp8_bwd=True)
    #                           w8a16  → nn.Linear (no W8A16 FFN
    #                                    kernel; falls back with
    #                                    warning)
    #                           w4a8   → NVFP4LinearW4A8 (two-pass)
    #                           w4a16  → NVFP4Linear (Marlin mode-3)
    #
    # The legacy boolean flags (``kda_fp8``, ``kda_mxfp8``,
    # ``ffn_nvfp4``, ``ffn_nvfp4_marlin``, ``ffn_nvfp4_no_bf16_master``)
    # were removed on 2026-07-21; the scheme system is the single
    # source of truth.
    # ---------------------------------------------------------------------
    embedding_precision: Literal["w16a16", "w8a16", "w8a8", "w4a16", "w4a8"] = "w16a16"
    attention_precision: Literal["w16a16", "w8a16", "w8a8", "w4a16", "w4a8"] = "w16a16"
    ffn_precision: Literal["w16a16", "w8a16", "w8a8", "w4a16", "w4a8"] = "w16a16"

    # Producer-side quant fusions (2026-07-22, full-FP8 productionization).
    # Both default to False for backward compatibility; flipping on is
    # the "full FP8" production recipe (see auto-memory
    # ``project_fp8_full_e2e.md`` for the precondition that passed
    # end-to-end at 300 steps with +0.27% loss delta).
    #
    # ``fp8_residual=True`` replaces the ``x + sub`` elementwise add
    # with the fused FP8 quant-add-requant kernel in
    # ``src/models/ops/fp8_residual.py``. Numerics: per-row amax carries
    # the FP8 representation noise (≈3.76% sig_rel vs BF16 reference,
    # same as a single GEMM round; the ``sqrt(N)`` compounding through
    # L layers is the standard FP8 noise model).
    #
    # ``fp8_silu_mul=True`` replaces the FFN's silu(gate)*up (BF16)
    # with the fused ``silu_mul_fp8`` Triton kernel in
    # ``src/models/ops/silu_mul_fp8.py``. Numerics: same FP8 noise
    # floor; saves 2 kernel launches per FFN sublayer.
    fp8_residual: bool = False
    fp8_silu_mul: bool = False

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
        # ---- Scheme-driven precision (2026-07-21, 5-scheme spec) ----
        for field_name in (
            "embedding_precision",
            "attention_precision",
            "ffn_precision",
        ):
            value = getattr(self, field_name)
            assert value in SCHEMES, (
                f"{field_name} must be one of {SCHEMES!r}; got {value!r}"
            )