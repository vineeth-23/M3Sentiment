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

    def forward(self, sequence_states, return_attention=False):
        attention_output, attention_weights = self.self_attn(
            sequence_states,
            sequence_states,
            sequence_states,
            need_weights=return_attention,
            average_attn_weights=False,
        )
        sequence_states = self.norm1(sequence_states + self.dropout1(attention_output))
        feedforward_output = self.linear2(self.dropout(self.activation(self.linear1(sequence_states))))
        sequence_states = self.norm2(sequence_states + self.dropout2(feedforward_output))
        if return_attention:
            return sequence_states, attention_weights
        return sequence_states


class InstrumentedTransformerEncoder(nn.Module):
    def __init__(self, layer, num_layers):
        super().__init__()
        self.layers = nn.ModuleList(copy.deepcopy(layer) for _ in range(num_layers))

    def forward(self, sequence_states, return_attention=False):
        attention_maps = []
        output_states = sequence_states
        for layer in self.layers:
            if return_attention:
                output_states, attention_weights = layer(output_states, return_attention=True)
                attention_maps.append(attention_weights)
            else:
                output_states = layer(output_states)
        if return_attention:
            return output_states, attention_maps
        return output_states
