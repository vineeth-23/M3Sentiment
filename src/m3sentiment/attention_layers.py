import copy

import torch.nn as nn


class InstrumentedTransformerEncoderLayer(nn.Module):
    def __init__(self, hidden_dim, n_heads, dropout):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.linear1 = nn.Linear(hidden_dim, hidden_dim * 2)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, src, return_attention=False):
        attn_out, attn_weights = self.self_attn(
            src,
            src,
            src,
            need_weights=return_attention,
            average_attn_weights=False,
        )
        src = self.norm1(src + self.dropout1(attn_out))
        ff_out = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = self.norm2(src + self.dropout2(ff_out))
        if return_attention:
            return src, attn_weights
        return src


class InstrumentedTransformerEncoder(nn.Module):
    def __init__(self, layer, num_layers):
        super().__init__()
        self.layers = nn.ModuleList(copy.deepcopy(layer) for _ in range(num_layers))

    def forward(self, src, return_attention=False):
        attentions = []
        output = src
        for layer in self.layers:
            if return_attention:
                output, attn_weights = layer(output, return_attention=True)
                attentions.append(attn_weights)
            else:
                output = layer(output)
        if return_attention:
            return output, attentions
        return output
