import torch
import torch.nn as nn
from .delta_utils import chunk_delta_rule_forward

class DeltaAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1,
        chunk_size: int = 16,
        use_beta: bool = True,
        use_gate: bool = False,
        qk_activation: str = "silu",
        qk_norm: str = "l2",
        norm_eps: float = 1e-6,
    ):
        super().__init__()
        self.num_heads    = num_heads
        self.head_dim     = d_model // num_heads
        self.chunk_size   = chunk_size
        self.use_beta     = use_beta
        self.use_gate     = use_gate
        self.qk_activation= qk_activation
        self.qk_norm      = qk_norm

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)

        if use_beta:
            self.beta_proj = nn.Linear(d_model, num_heads, bias=False)
        if use_gate:
            self.g_proj = nn.Linear(d_model, d_model, bias=False)

        self.o_norm   = nn.LayerNorm(self.head_dim, eps=norm_eps)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout  = nn.Dropout(dropout)

        act = {
            "silu": torch.nn.functional.silu,
            "relu": torch.nn.functional.relu,
            "elu":  lambda x: (torch.nn.functional.elu(x) + 1.0),
        }.get(qk_activation, lambda x: x)
        self.qk_act = act

    def forward(self, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
        B, L, _ = hidden_states.shape

        q = self.qk_act(self.q_proj(hidden_states)) # use qk_act for non-linearity on projections
        k = self.qk_act(self.k_proj(hidden_states))
        v = self.qk_act(self.v_proj(hidden_states))

        def split(x):
            return x.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        q, k, v = map(split, (q, k, v))

        if self.use_beta:
            beta = torch.sigmoid(self.beta_proj(hidden_states)).transpose(1,2)  # [B,H,L]
        else:
            beta = q.new_ones(B, self.num_heads, L)

        o, _ = chunk_delta_rule_forward(q, k, v, beta, self.chunk_size)

        if self.use_gate:
            g = torch.sigmoid(self.g_proj(hidden_states))
            g = g.view(B, L, self.num_heads, self.head_dim).transpose(1,2)
            o = o * g

        o = self.o_norm(o)
        o = o.transpose(1,2).reshape(B, L, -1)
        return self.out_proj(self.dropout(o))
