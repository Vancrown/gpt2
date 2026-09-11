import torch.nn as nn
import torch


class CausalAttention(nn.Module):
    def __init__(
        self, d_in, d_out, context_length, dropout, qkv_bias=False, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask", torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        _key = self.W_key(x)
        _query = self.W_query(x)
        _value = self.W_value(x)

        attn_score = _query @ _key.transpose(1, 2)
        attn_score.masked_fill_(self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)
        attn_weight = torch.softmax(attn_score / _key.shape[-1] ** 0.5, dim=-1)
        attn_weight = self.dropout(attn_weight)
        context_vec = attn_weight @ _value
        return context_vec
