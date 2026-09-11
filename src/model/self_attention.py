import torch.nn as nn
import torch


class SelfAttention_v1(nn.Module):
    def __init__(self, d_in, d_out, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.W_query = nn.Parameter(torch.rand(d_in, d_out))
        self.W_key = nn.Parameter(torch.rand(d_in, d_out))
        self.W_value = nn.Parameter(torch.rand(d_in, d_out))

    def forward(self, x):
        key = x @ self.W_key
        query = x @ self.W_query
        value = x @ self.W_value
        attn_score = query @ key.T
        attn_weight = torch.softmax(attn_score / key.shape[-1] ** 0.5, dim=-1)
        context_vec = attn_weight @ value
        return context_vec


class SelfAttention_v2(nn.Module):
    def __init__(self, d_in, d_out, qkv_bias=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, x):
        key = x @ self.W_key(x)
        query = x @ self.W_query(x)
        value = x @ self.W_value(x)
        attn_score = query @ key.T
        attn_weight = torch.softmax(attn_score / key.shape[-1] ** 0.5, dim=-1)
        context_vec = attn_weight @ value
        return context_vec


if __name__ == "__main__":

    print("done")
