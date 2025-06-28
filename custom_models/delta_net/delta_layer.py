import torch
import torch.nn as nn
from .delta_utils import chunk_delta_rule_forward

class DeltaAttention(nn.Module):
    """
    Custom Delta-rule multi-head self-attention.
    """
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.0,
        chunk_size: int = 16,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.chunk_size = chunk_size

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        # project to per-head beta weights
        self.beta_proj = nn.Linear(d_model, num_heads)

        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
        B, L, _ = hidden_states.shape
        # project and reshape to [B,H,L,D]
        q = self.q_proj(hidden_states).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        # compute beta per head, per position: [B,N,H] → [B,H,N]
        beta = torch.sigmoid(self.beta_proj(hidden_states)).transpose(1, 2)

        # delta-rule forward
        o, _ = chunk_delta_rule_forward(q, k, v, beta, self.chunk_size)

        # merge heads and final projection: [B,H,N,D] → [B,N,D_model]
        o = o.transpose(1, 2).reshape(B, L, -1)
        return self.out_proj(self.dropout(o))