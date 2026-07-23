"""HippoLM transformer layer and full model.

Block-boundary Block AttnRes (official Kimi design):
  - 32 layers are partitioned into 8 blocks of 4 layers each.
  - Within a block, every layer is a standard residual transformer
    ``x = x + KDA(RMSNorm(x)); x = x + FFN(RMSNorm(x))``. There
    is no per-layer AttnRes; the only AttnRes invocation is at
    the block boundary, where the next block's input is computed
    by softmax-attending over the completed block representations
    ``[b_0, b_1, ..., b_{n-1}]`` (Kimi Team, arXiv:2603.15031).
  - The pseudo-query is shared across all boundaries (one module
    per model, not per sub-layer). Initialised to 0 so the
    start-of-training attention is uniform.
  - Block 0's input is the embedding ``b_0``; subsequent blocks
    use ``attn_res([b_0, ..., b_{n-1}])`` as the input.
"""
import torch
import torch.nn as nn

from .norms import RMSNorm
from .ops.kda import KDA
from .ops.attn_res import BlockAttnRes
from .activation import SwiGLU
from .ops._vendored.fla.modules.fused_cross_entropy import FusedCrossEntropyLoss


class HippoLayer(nn.Module):
    """Single transformer layer with standard residual.

    Within a block, the residual connection is the standard
    ``x = x + SubLayer(x)`` form. AttnRes lives at the model
    level and is invoked only at block boundaries (see
    :class:`HippoModel`).

    Producer-side FP8 fusions are driven by the two per-op precision
    fields on ``HippoConfig`` (2026-07-23):

      - ``rmsnorm_precision='fp8'`` → ``attn_norm`` / ``mlp_norm`` use
        the fused ``RmsNormFp8STE`` kernel
        (``src/models/ops/rmsnorm_fp8.py``): RMSNorm + per-row FP8
        quant in one launch (2 launches saved per layer). STE on the
        FP8 round, proper RMSNorm bwd on the math (``dL/dweight`` is
        preserved). When the FFN is also FP8/NVFP4 with an inner quant,
        the fused kernel returns ``(bf16, fp8, scale)`` so the FFN's
        internal ``quantize_act_fp8_fused`` can be skipped (see
        :meth:`_norm_with_passthrough`).
      - ``residual_precision='fp8'`` → both sub-layer residuals use
        ``Fp8ResidualSTE`` (quant-add-requant in one Triton launch,
        STE bwd) instead of ``x + sub``. Production keeps this
        ``'bf16'`` — the residual stream's precision is the
        load-bearing path for the L-layer FP8 noise compounding
        (see ``project_fp8_mixed_e2e.md``).

    ``silu_mul`` FP8 is folded into the FFN scheme (``ffn_precision``),
    so it is handled inside :class:`SwiGLU`, not here.
    """

    def __init__(self, layer_idx: int, config):
        super().__init__()
        self.layer_idx = layer_idx
        self.config = config

        # Pre-attention and pre-FFN RMSNorms.
        self.attn_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Core sub-layers.
        self.kda = KDA(config, layer_idx=layer_idx)
        self.ffn = SwiGLU(config)

        # Producer-side precision toggles (resolved once at build time;
        # the autograd Functions lazy-import their Triton kernels on
        # first call, so there is no eager JIT cost here).
        self._use_fp8_rmsnorm: bool = (
            getattr(config, "rmsnorm_precision", "bf16") == "fp8"
        )
        self._use_fp8_residual: bool = (
            getattr(config, "residual_precision", "bf16") == "fp8"
        )

    def _norm(self, layer: RMSNorm, x: torch.Tensor) -> torch.Tensor:
        """RMSNorm, optionally through the fused FP8 kernel (BF16 out)."""
        if self._use_fp8_rmsnorm:
            from .ops.rmsnorm_fp8 import RmsNormFp8STE
            return RmsNormFp8STE.apply(x, layer.weight, layer.eps)
        return layer(x)

    def _norm_with_passthrough(
        self, layer: RMSNorm, x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Fused ``RMSNorm → FP8`` returning ``(y_bf16, out_fp8, scale)``.

        Used for ``mlp_norm`` when the FFN has an inner FP8 quant
        (``ffn_precision='w4a8'``) and ``rmsnorm_precision='fp8'``:
        the returned fp8 + scale route straight into
        :meth:`SwiGLU.forward_precomputed`, skipping the redundant
        per-FFN ``quantize_act_fp8_fused``. See
        :func:`src.models.ops.rmsnorm_fp8.rmsnorm_fp8_with_passthrough`.
        """
        from .ops.rmsnorm_fp8 import rmsnorm_fp8_with_passthrough
        return rmsnorm_fp8_with_passthrough(x, layer.weight, layer.eps)

    def _residual(self, x: torch.Tensor, sub: torch.Tensor) -> torch.Tensor:
        """Residual add, optionally through the fused FP8 kernel."""
        if self._use_fp8_residual:
            from .ops.fp8_residual import Fp8ResidualSTE
            return Fp8ResidualSTE.apply(x, sub)
        return x + sub

    def forward(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Standard residual transformer layer.

        Args:
            x: ``[B, T, hidden_size]`` input.
            cu_seqlens: optional varlen offsets passed through to
                the KDA sub-layer (FFN / RMSNorm ignore it).

        Returns:
            ``[B, T, hidden_size]`` output after KDA and FFN with
            standard residual connections. RMSNorm / residual precision
            follow ``config.rmsnorm_precision`` / ``residual_precision``.
        """
        sub = self.kda(self._norm(self.attn_norm, x), cu_seqlens=cu_seqlens)
        x = self._residual(x, sub)

        # FFN block. ``SwiGLU`` dispatches on ``config.ffn_precision``.
        # When both rmsnorm is FP8 and the FFN has an inner quant, use
        # the passthrough to skip the FFN's internal act_quant.
        if self._use_fp8_rmsnorm and getattr(self.ffn, "_uses_inner", False):
            x_norm_bf16, x_norm_fp8, x_norm_s = self._norm_with_passthrough(
                self.mlp_norm, x,
            )
            sub = self.ffn.forward_precomputed(x_norm_bf16, x_norm_fp8, x_norm_s)
        else:
            sub = self.ffn(self._norm(self.mlp_norm, x))
        x = self._residual(x, sub)
        return x


class HippoModel(nn.Module):
    """HippoLM model with block-boundary Block AttnRes."""

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            HippoLayer(i, config) for i in range(config.num_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Block-boundary attention residual. One module shared
        # across all ``num_blocks - 1`` boundaries. Initialised
        # to 0 (uniform attention) per the paper. Only BF16 is
        # wired (no FP8 BlockAttnRes kernel); config validation
        # already rejects anything else, but guard here too so a
        # future field change fails loudly at the construction site.
        if getattr(config, "attn_res_precision", "bf16") != "bf16":
            raise ValueError(
                f"attn_res_precision={config.attn_res_precision!r} is not "
                f"supported (only 'bf16' — no FP8 BlockAttnRes kernel)."
            )
        self.attn_res = BlockAttnRes(config)

        # Language modeling head. ``config.lm_head_precision`` drives
        # the choice:
        #   ``"w8a8"``  → ``FP8Linear(fp8_bwd=True)`` — saves ~9.5 ms
        #                 per step at prod shape (M=1024 H=1536
        #                 V=248320). 0.03% loss drift over 50 steps
        #                 (no compounding noise; cos_sim > 0.999 on
        #                 logits / grads). The BF16 master weight is
        #                 the optimizer leaf — the FP8 quantization is
        #                 computed on the fly per forward. Tied-
        #                 embeddings (``tie_word_embeddings=True``)
        #                 shares the same ``.weight`` tensor; see
        #                 ``project_w8a8_lmhead_embed_probe_2026_07_22.md``.
        #   ``"w16a16"`` (default) → plain ``nn.Linear``.
        #   other schemes fall back to ``nn.Linear`` with a logged
        #   warning (the only other candidate today is ``w8a16``
        #   which has no BF16-GEMM W8A16 head kernel on sm_120; the
        #   ``w4a16``/``w4a8`` paths are not implemented for lm_head).
        lm_head_precision = getattr(config, "lm_head_precision", "w16a16")
        if lm_head_precision == "w8a8":
            from .ops.fp8_linear import FP8Linear
            # Shape-constraint check (FP8Linear silently falls back to
            # BF16 when in/out not divisible by 16). For the prod
            # shape (V=248320, H=1536) both are divisible by 16, so
            # this branch always fires — the FP8 path is real.
            self.lm_head = FP8Linear(
                config.hidden_size, config.vocab_size,
                bias=config.use_bias, fp8_bwd=True,
            )
        else:
            if lm_head_precision != "w16a16":
                import warnings
                warnings.warn(
                    f"lm_head_precision={lm_head_precision!r} is not "
                    f"implemented for the lm_head module today (only "
                    f"'w16a16' and 'w8a8' are wired); falling back to "
                    f"'w16a16' (plain nn.Linear).",
                    RuntimeWarning,
                    stacklevel=2,
                )
            self.lm_head = nn.Linear(
                config.hidden_size, config.vocab_size, bias=config.use_bias
            )

        # Tie embeddings and output weights
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

        # Fused log-softmax + nll + softmax-derivative (Triton). Used by
        # the non-TP forward path; the TP path uses TPFusedLceLoss which
        # additionally folds the lm_head materialization. ~2x faster
        # than F.cross_entropy at prod shape (1024, 248320); saves the
        # 9.4 ms elementwise cluster on each step.
        self.fused_ce = FusedCrossEntropyLoss(ignore_index=-100, reduction="mean")

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        # Embedding: standard init
        nn.init.normal_(self.embed_tokens.weight, mean=0.0, std=0.02)

        # Linear layers: Xavier uniform
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            input_ids: ``[batch_size, seq_len]``
            labels: Optional ``[batch_size, seq_len]`` for loss
            cu_seqlens: Optional ``[total_docs + 1]`` global offset
                tensor for chunk-aligned FFD-packed inputs. When
                set the KDA sub-layer resets the recurrent state
                at each ``cu_seqlens`` boundary. BlockAttnRes,
                FFN, RMSNorm, embed, and lm_head all ignore it.

        Returns:
            Dict with ``logits`` and optionally ``loss``.
        """
        embeds = self.embed_tokens(input_ids)  # [B, T, D]

        # Block representation list. ``b_0`` is always the
        # embedding; ``b_n`` for ``n >= 1`` is the output of
        # the n-th completed block.
        blocks: list[torch.Tensor] = [embeds]
        x = embeds  # Block 0's input is the embedding.

        for block_idx in range(self.config.num_blocks):
            # At every non-first block boundary, AttnRes computes
            # the next block's input by softmax-attending over the
            # completed block representations.
            if block_idx > 0:
                x = self.attn_res(blocks)

            start = block_idx * self.config.block_size
            end = start + self.config.block_size
            block_layers = self.layers[start:end]
            # Last block: per-layer checkpointing for every
            # layer except the very last one (matches the TP
            # path's strategy; see tp_model.py for the full
            # rationale). 3 of the 4 layers are wrapped in
            # ``torch.utils.checkpoint.checkpoint`` with
            # ``use_reentrant=True`` so the KDA internals are
            # freed as soon as that layer's backward completes;
            # the final layer keeps its full cache to feed the
            # lm_head + fused CE backward.
            if block_idx == self.config.num_blocks - 1:
                for layer in block_layers[:-1]:
                    x = torch.utils.checkpoint.checkpoint(
                        layer, x, cu_seqlens,
                        use_reentrant=True, preserve_rng_state=False,
                    )
                x = block_layers[-1](x, cu_seqlens=cu_seqlens)
            else:
                for layer in block_layers:
                    x = layer(x, cu_seqlens=cu_seqlens)
            blocks.append(x)

        hidden_states = self.norm(blocks[-1])  # [B, T, D]
        logits = self.lm_head(hidden_states)  # [B, T, vocab_size]

        output = {"logits": logits}

        if labels is not None:
            # Shift for next-token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # Fused log-softmax + nll + softmax-derivative via FLA Triton
            # kernel. ~2x faster than F.cross_entropy at prod shape;
            # see test/test_fused_cross_entropy.py for the numerics
            # contract (cos_sim > 0.9999 on bwd grad).
            loss = self.fused_ce(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )
            output["loss"] = loss

        return output

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 1,
        temperature: float = 1.0,
        top_k: int = 0,
    ) -> torch.Tensor:
        """Greedy / sampling generation.

        Args:
            input_ids: ``[batch_size, seq_len]``
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling (0 = disabled)

        Returns:
            Generated token IDs ``[batch_size, seq_len + max_new_tokens]``.
        """
        self.eval()
        with torch.no_grad():
            for _ in range(max_new_tokens):
                outputs = self.forward(input_ids)
                logits = outputs["logits"][:, -1, :] / temperature  # [B, vocab]

                if top_k > 0:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = float("-inf")

                probs = nn.functional.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                input_ids = torch.cat([input_ids, next_token], dim=1)

        return input_ids
