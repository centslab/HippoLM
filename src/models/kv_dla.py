"""Key-Value Delta Linear Attention (kvDLA) module.

Replaces standard self-attention with explicit key-value pair maintenance
per head, using a gated delta-rule update mechanism.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class kvDLA(nn.Module):
    """Key-Value Delta Linear Attention.

    Maintains num_kv explicit key-value pairs per head instead of a state matrix.
    Updates follow a gated delta rule with learnable gates and step sizes.

    For v0.0.0, this uses a sequential token-by-token loop for correctness.
    Performance optimization (chunked parallel, CUDA kernel) is v0.0.3.
    """

    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.num_kv = config.num_kv
        self.hidden_size = config.hidden_size
        self.kv_init_std = config.kv_init_std

        # Projections to per-head keys and values
        self.W_K = nn.Linear(
            self.hidden_size,
            self.num_heads * self.head_dim,
            bias=config.use_bias,
        )
        self.W_V = nn.Linear(
            self.hidden_size,
            self.num_heads * self.head_dim,
            bias=config.use_bias,
        )

        # Output projection
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim,
            self.hidden_size,
            bias=config.use_bias,
        )

        # Learnable scalar gate coefficients
        self.alpha = nn.Parameter(torch.tensor(config.alpha_init))
        self.beta = nn.Parameter(torch.tensor(config.beta_init))
        self.gamma = nn.Parameter(torch.tensor(config.gamma_init))

        # Learnable vector step sizes [num_heads, head_dim]
        self.eta_v = nn.Parameter(
            torch.full((self.num_heads, self.head_dim), config.eta_init)
        )
        self.eta_k = nn.Parameter(
            torch.full((self.num_heads, self.head_dim), config.eta_init)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor [batch_size, seq_len, hidden_size]

        Returns:
            Output tensor [batch_size, seq_len, hidden_size]
        """
        B, T, D = x.shape
        H, N, d = self.num_heads, self.num_kv, self.head_dim

        # Project to per-head keys and values
        k = self.W_K(x).view(B, T, H, d)  # [B, T, H, d]
        v = self.W_V(x).view(B, T, H, d)  # [B, T, H, d]

        # Initialize K, V states per (batch, head, slot)
        K_state = torch.zeros(B, H, N, d, device=x.device, dtype=x.dtype)
        V_state = torch.zeros(B, H, N, d, device=x.device, dtype=x.dtype)
        nn.init.normal_(K_state, mean=0, std=self.kv_init_std)
        nn.init.normal_(V_state, mean=0, std=self.kv_init_std)

        outputs = []
        for t in range(T):
            k_t = k[:, t, :, :]  # [B, H, d]
            v_t = v[:, t, :, :]  # [B, H, d]

            # Compute attention scores over KV slots: K_i^T k_t
            scores = torch.einsum("bhnd,bhd->bhn", K_state, k_t) / math.sqrt(d)
            weights = F.softmax(scores, dim=-1)  # [B, H, N]

            # Weighted retrieval: sum_i softmax_i * V_i
            v_hat = torch.einsum("bhn,bhnd->bhd", weights, V_state)  # [B, H, d]

            # Prediction error
            e_t = v_t - v_hat  # [B, H, d]

            # Compute gates g_i = sigmoid(alpha * K_i^T k_t + beta * V_i^T e_t + gamma)
            kt_scores = torch.einsum("bhnd,bhd->bhn", K_state, k_t)  # [B, H, N]
            ve_scores = torch.einsum("bhnd,bhd->bhn", V_state, e_t)  # [B, H, N]
            gate_input = self.alpha * kt_scores + self.beta * ve_scores + self.gamma
            g = torch.sigmoid(gate_input)  # [B, H, N]

            # Update V: V_i += eta_v * g_i * e_t
            # g: [B, H, N] -> [B, H, N, 1]
            # e_t: [B, H, d] -> [B, H, 1, d]
            # eta_v: [H, d] -> [1, H, 1, d]
            delta_V = g.unsqueeze(-1) * e_t.unsqueeze(2) * self.eta_v.view(1, H, 1, d)
            V_state = V_state + delta_V  # [B, H, N, d]

            # Update K: K_i += eta_k * g_i * (k_t - K_i)
            diff_k = k_t.unsqueeze(2) - K_state  # [B, H, N, d]
            delta_K = g.unsqueeze(-1) * diff_k * self.eta_k.view(1, H, 1, d)
            K_state = K_state + delta_K

            outputs.append(v_hat)

        # Stack and project
        output = torch.stack(outputs, dim=1)  # [B, T, H, d]
        output = output.view(B, T, H * d)
        output = self.o_proj(output)
        return output
