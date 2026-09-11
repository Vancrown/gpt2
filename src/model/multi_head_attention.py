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


class MultiHeadAttention(nn.Module):
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

        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask", torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        _key = self.W_key(x)
        _query = self.W_query(x)
        _value = self.W_value(x)

        _key = _key.view(b, num_tokens, self.num_heads, self.head_dim)
        _query = _query.view(b, num_tokens, self.num_heads, self.head_dim)
        _value = _value.view(b, num_tokens, self.num_heads, self.head_dim)

        _key = _key.transpose(1, 2)
        _query = _query.transpose(1, 2)
        _value = _value.transpose(1, 2)

        attn_score = _query @ _key.transpose(2, 3)
        mask_book = self.mask.bool()[:num_tokens, :num_tokens]

        attn_score.masked_fill_(mask_book, -torch.inf)

        attn_weight = torch.softmax(attn_score / _key.shape[-1] ** 0.5, dim=-1)
        attn_weight = self.dropout(attn_weight)

        context_vec = (attn_weight @ _value).transpose(1, 2)
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)

        return context_vec
