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
    #   lm_head_precision:    only ``w16a16`` and ``w8a8`` are wired
    #                         today. ``w8a8`` = ``FP8Linear(
    #                         fp8_bwd=True)`` — saves ~9.5 ms/step at
    #                         prod shape, 0.03% loss drift over 50
    #                         steps, no compounding noise. The other
    #                         three schemes fall back to ``w16a16``
    #                         with a warning.
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
    # lm_head precision (2026-07-22). Only the W8A8 wiring is real
    # today (``FP8Linear(fp8_bwd=True)`` — saves ~9.5 ms/step at prod
    # shape, 0.03% loss drift over 50 steps, no compounding noise);
    # the other schemes fall back to ``w16a16`` (plain nn.Linear)
    # with a logged warning. See
    # ``project_w8a8_lmhead_embed_probe_2026_07_22.md`` for the
    # full probe.
    lm_head_precision: Literal["w16a16", "w8a16", "w8a8", "w4a16", "w4a8"] = "w16a16"

    # ---------------------------------------------------------------------
    # Producer-side fused-op precision (2026-07-23).
    # ---------------------------------------------------------------------
    # These three knobs pick the precision for the fused element-wise
    # ops that sit BETWEEN the GEMMs (the "producers" that feed the
    # attention / FFN Linear layers and the residual adds). They are
    # independent of the four scheme fields above (which govern the
    # GEMMs themselves), because the fused kernels are not GEMMs —
    # they are RMSNorm / residual-add / block-attn-res.
    #
    #   * ``residual_precision`` — the inter-layer residual add
    #     (``x = x + sub``). ``"bf16"`` (default) is a plain BF16
    #     ``+``. ``"fp8"`` routes through ``Fp8ResidualSTE`` (fused
    #     FP8 add + STE bwd). Prod default is ``"bf16"``: the FP8
    #     residual path exists but the end-to-end A/B showed the
    #     residual must stay BF16 to keep the layer-out drift under
    #     budget (see ``project_fp8_mixed_e2e.md`` — full-FP8 residual
    #     compounds to 11.7% sig_rel, BF16 residual drops it to 4.4%).
    #
    #   * ``rmsnorm_precision`` — the ``attn_norm`` / ``mlp_norm``
    #     RMSNorm before each GEMM block. ``"bf16"`` is the plain
    #     ``RMSNorm`` module. ``"fp8"`` (default) routes through
    #     ``RmsNormFp8STE`` (fused RMSNorm + FP8 round + STE bwd);
    #     verified 2.43x faster with 0.42% sub-1% BF16-precision
    #     noise (see ``project_fp8_producer_fusions.md``).
    #
    #   * ``attn_res_precision`` — the block-level ``BlockAttnRes``
    #     residual aggregation. Only ``"bf16"`` is wired: an FP8
    #     BlockAttnRes kernel was never written (the online-softmax
    #     bwd is already Triton-fused at BF16 and the op is not on
    #     the FP8 critical path — not worth the kernel work). Anything
    #     other than ``"bf16"`` raises at construction.
    #
    # The fused kernels live in
    # ``src/models/ops/{rmsnorm_fp8,fp8_residual,silu_mul_fp8}.py``.
    # They are wired into :class:`src.models.model.HippoLayer` and
    # :class:`src.models.tp_model.layer.TPHippoLayer` gated on these
    # fields. ``silu_mul`` FP8 is folded into the FFN scheme path
    # (``ffn_precision``), so it has no separate knob here.
    residual_precision: Literal["bf16", "fp8"] = "bf16"
    rmsnorm_precision: Literal["bf16", "fp8"] = "fp8"
    attn_res_precision: Literal["bf16"] = "bf16"

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
            "lm_head_precision",
        ):
            value = getattr(self, field_name)
            assert value in SCHEMES, (
                f"{field_name} must be one of {SCHEMES!r}; got {value!r}"
            )
        # ---- Producer-side fused-op precision (2026-07-23) ----
        assert self.residual_precision in ("bf16", "fp8"), (
            f"residual_precision must be 'bf16' or 'fp8';"
            f" got {self.residual_precision!r}"
        )
        assert self.rmsnorm_precision in ("bf16", "fp8"), (
            f"rmsnorm_precision must be 'bf16' or 'fp8';"
            f" got {self.rmsnorm_precision!r}"
        )
        # attn_res has no FP8 kernel; only bf16 is valid.
        assert self.attn_res_precision == "bf16", (
            f"attn_res_precision only supports 'bf16' (no FP8"
            f" BlockAttnRes kernel exists); got"
            f" {self.attn_res_precision!r}"
        )
        # ---- Tied-embeddings precision consistency (2026-07-23) ----
        # When tie_word_embeddings=True, ``lm_head.weight`` aliases
        # ``embed_tokens.weight`` (see ``HippoModel.__init__`` line
        # ``self.lm_head.weight = self.embed_tokens.weight``). The
        # two roles cannot pick different precision schemes — one
        # side would have to quantize the shared tensor differently.
        # Reject this at config-parse time so the failure is loud,
        # not a silent "fp8 path on one side / bf16 on the other"
        # at first forward. Prod base.yml satisfies this by having
        # both at ``w16a16`` (the FP8 lm_head path was dead on TP
        # anyway).
        if self.tie_word_embeddings:
            assert self.embedding_precision == self.lm_head_precision, (
                f"tie_word_embeddings=True requires"
                f" embedding_precision == lm_head_precision;"
                f" got embedding_precision={self.embedding_precision!r},"
                f" lm_head_precision={self.lm_head_precision!r}."
                f" The two share one weight tensor at construction"
                f" (lm_head.weight = embed_tokens.weight); they"
                f" cannot be quantized under different schemes."
            )