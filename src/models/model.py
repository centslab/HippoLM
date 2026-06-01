"""HippoLM transformer layer and full model."""
import torch
import torch.nn as nn

from .norms import RMSNorm
from .kda import KDA
from .block_attn_res import BlockAttnRes
from .activation import SwiGLU


class HippoLayer(nn.Module):
    """Single transformer layer with Block AttnRes.

    Each layer contains:
    1. Pre-attention Block AttnRes -> kda -> add to partial_block
    2. Pre-FFN Block AttnRes -> SwiGLU -> add to partial_block

    Block boundaries are checked before the attention sub-layer.
    """

    def __init__(self, layer_idx: int, config):
        super().__init__()
        self.layer_idx = layer_idx
        self.block_size = config.block_size

        # Two AttnRes: one before kda, one before FFN
        self.attn_res = BlockAttnRes(config)
        self.mlp_res = BlockAttnRes(config)

        # Normalization
        self.attn_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Core sub-layers
        self.kda = KDA(config)
        self.ffn = SwiGLU(config)

    def forward(
        self,
        blocks: list[torch.Tensor],
        partial_block: torch.Tensor | None,
    ) -> tuple[list[torch.Tensor], torch.Tensor | None]:
        """Forward pass for one layer.

        Args:
            blocks: Completed block representations [b_0, b_1, ..., b_{n-1}]
            partial_block: Current block partial sum or None

        Returns:
            (updated_blocks, updated_partial_block)
        """
        # 1-indexed layer number for block boundary check
        layer_number = self.layer_idx + 1

        # Check block boundary BEFORE attention (per paper pseudocode)
        if layer_number % self.block_size == 0:
            if partial_block is not None:
                blocks = blocks + [partial_block]
            partial_block = None

        # ---- Attention sub-layer ----
        # Use checkpoint for attention to save activation memory
        # save (blocks, partial_block) only - recompute h_attn and attn_out during backward
        h_attn, attn_out = torch.utils.checkpoint.checkpoint(
            self._attn_forward, blocks, partial_block,
            use_reentrant=False, preserve_rng_state=False,
        )

        if partial_block is None:
            partial_block = attn_out
        else:
            partial_block = partial_block + attn_out

        # ---- FFN sub-layer ----
        # Use checkpoint for FFN
        ffn_out = torch.utils.checkpoint.checkpoint(
            self._ffn_forward, blocks, partial_block,
            use_reentrant=False, preserve_rng_state=False,
        )

        partial_block = partial_block + ffn_out

        return blocks, partial_block

    def _attn_forward(self, blocks, partial_block):
        """Attention sub-layer forward (used for checkpointing)."""
        h_attn = self.attn_res(blocks, partial_block)
        attn_out = self.kda(self.attn_norm(h_attn))
        return h_attn, attn_out

    def _ffn_forward(self, blocks, partial_block):
        """FFN sub-layer forward (used for checkpointing)."""
        h_mlp = self.mlp_res(blocks, partial_block)
        ffn_out = self.ffn(self.mlp_norm(h_mlp))
        return ffn_out


class HippoModel(nn.Module):
    """HippoLM model.

    Combines token embeddings, Block AttnRes layers with kda attention,
    and a language modeling head.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            HippoLayer(i, config) for i in range(config.num_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

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
    ) -> dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            input_ids: [batch_size, seq_len]
            labels: Optional [batch_size, seq_len] for loss computation

        Returns:
            Dict with 'logits' and optionally 'loss'
        """
        embeds = self.embed_tokens(input_ids)  # [B, T, D]

        # Block AttnRes state
        blocks: list[torch.Tensor] = [embeds]  # b_0 = embedding
        partial_block: torch.Tensor | None = embeds

        # Each layer uses checkpointing internally for kda and FFN sub-layers
        # This saves activation memory while keeping block-level gradient flow
        for layer in self.layers:
            blocks, partial_block = layer(blocks, partial_block)

        # Final normalization and projection
        hidden_states = self.norm(partial_block)  # [B, T, D]
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
            input_ids: [batch_size, seq_len]
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling (0 = disabled)

        Returns:
            Generated token IDs [batch_size, seq_len + max_new_tokens]
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
