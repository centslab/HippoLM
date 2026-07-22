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


class HippoLayer(nn.Module):
    """Single transformer layer with standard residual.

    Within a block, the residual connection is the standard
    ``x = x + SubLayer(x)`` form. AttnRes lives at the model
    level and is invoked only at block boundaries (see
    :class:`HippoModel`).

    Optional producer-side FP8 fusions (gated by
    ``config.fp8_rmsnorm`` / ``fp8_residual`` / ``fp8_silu_mul``):

      - ``fp8_rmsnorm=True``   → ``attn_norm(x)`` and ``mlp_norm(x)``
        run through :class:`RmsNormFp8STE` instead of the BF16 fla
        kernel. The fused kernel (``src/models/ops/rmsnorm_fp8.py``)
        does RMSNorm + per-row FP8 quant in one launch; saves 1
        kernel per RMSNorm (2 RMSNorms per layer → 2 launches saved).
        STE on the FP8 round, proper RMSNorm bwd on the math
        (``dL/dweight`` is preserved). The downstream KDA / FFN GEMMs
        consume the BF16 dequantized output via their own internal
        act_quant (the FP8 + scale output is currently discarded;
        a future KDA-side change could consume it directly and save
        the re-quant, but that requires KDA's act_quant to be plumbed
        to accept pre-quantized inputs).
      - ``fp8_residual=True``  → ``x = Fp8ResidualSTE(x, sub)`` instead
        of ``x + sub`` for both sublayer residuals. The fused kernel
        (``src/models/ops/fp8_residual.py``) does quant-add-requant
        in one Triton launch with STE backward. Numerics: per-row
        FP8 representation noise (3.76% sig_rel vs BF16 reference at
        a single boundary; compounds via ``sqrt(N)`` across layers as
        is standard for FP8). Production keeps this OFF — the residual
        stream's precision is the load-bearing path for the L-layer
        noise compounding.
      - ``fp8_silu_mul=True`` → FFN's middle op is
        ``SiluMulFp8STE(gate, up)`` instead of ``silu(gate)*up`` (BF16).
        The fused kernel (``src/models/ops/silu_mul_fp8.py``) does
        silu+mul+quant in one launch; saves 2 kernel launches per
        FFN sublayer.
    """

    def __init__(self, layer_idx: int, config):
        super().__init__()
        self.layer_idx = layer_idx

        # Pre-attention and pre-FFN RMSNorms.
        self.attn_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Core sub-layers.
        self.kda = KDA(config, layer_idx=layer_idx)
        self.ffn = SwiGLU(config)

        # Lazy imports: ops module is the only consumer, and the
        # autograd Functions depend on the Triton kernels which
        # JIT on first call (no eager cost).
        self._use_fp8_rmsnorm: bool = bool(getattr(config, "fp8_rmsnorm", False))
        self._use_fp8_residual: bool = bool(getattr(config, "fp8_residual", False))
        self._use_fp8_silu_mul: bool = bool(getattr(config, "fp8_silu_mul", False))

    def _norm(self, layer: RMSNorm, x: torch.Tensor) -> torch.Tensor:
        if self._use_fp8_rmsnorm:
            from .ops.rmsnorm_fp8 import RmsNormFp8STE
            return RmsNormFp8STE.apply(x, layer.weight, layer.eps)
        return layer(x)

    def forward(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Standard residual transformer layer (with optional FP8 fusions).

        Args:
            x: ``[B, T, hidden_size]`` input.
            cu_seqlens: optional varlen offsets passed through to
                the KDA sub-layer (FFN / RMSNorm ignore it).

        Returns:
            ``[B, T, hidden_size]`` output after KDA and FFN with
            standard residual connections (BF16 or fused FP8, see
            ``HippoConfig.fp8_rmsnorm`` / ``fp8_residual`` /
            ``fp8_silu_mul``).
        """
        sub = self.kda(self._norm(self.attn_norm, x), cu_seqlens=cu_seqlens)
        if self._use_fp8_residual:
            from .ops.fp8_residual import Fp8ResidualSTE
            x = Fp8ResidualSTE.apply(x, sub)
        else:
            x = x + sub

        x_norm = self._norm(self.mlp_norm, x)
        if self._use_fp8_silu_mul:
            # Fused silu(gate)*up → FP8 inside the FFN. We bypass
            # SwiGLU.forward so its internals don't re-fuse (avoid
            # double-quant). The FP8 output dequantises back to BF16
            # for ``down_proj`` (W8A8) to consume via its own quant.
            gate = self.ffn.gate_proj(x_norm)
            up = self.ffn.up_proj(x_norm)
            from .ops.silu_mul_fp8 import SiluMulFp8STE
            inter = SiluMulFp8STE.apply(gate, up)
            sub = self.ffn.down_proj(inter)
        else:
            sub = self.ffn(x_norm)
        if self._use_fp8_residual:
            from .ops.fp8_residual import Fp8ResidualSTE
            x = Fp8ResidualSTE.apply(x, sub)
        else:
            x = x + sub
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
        # to 0 (uniform attention) per the paper.
        self.attn_res = BlockAttnRes(config)

        # Language modeling head
        self.lm_head = nn.Linear(
            config.hidden_size, config.vocab_size, bias=config.use_bias
        )

        # Tie embeddings and output weights
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

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

            loss = nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
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
