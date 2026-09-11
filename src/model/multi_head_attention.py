import torch
import torch.nn as nn
from .causal_attention import CausalAttention


class MultiHeadAttentionWrapper(nn.Module):
    def __init__(
        self,
        d_in,
        d_out,
        context_length,
        dropout,
        num_heads,
        qkv_bias=False,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.heads = nn.ModuleList(
            [
                CausalAttention(d_in, d_out, context_length, dropout, qkv_bias)
                for i in range(num_heads)
            ]
        )

    def forward(self, x):
        return torch.cat([h(x) for h in self.heads], dim=-1)
