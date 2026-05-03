import math

import torch
import torch.nn as nn

from m3sentiment.attention_layers import InstrumentedTransformerEncoder, InstrumentedTransformerEncoderLayer


class PositionalEncoding(nn.Module):
    def __init__(self, hidden_dim: int, max_len: int = 50):
        super().__init__()
        position_encoding = torch.zeros(max_len, hidden_dim)
        position_index = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        frequency_terms = torch.exp(torch.arange(0, hidden_dim, 2).float() * (-math.log(10000.0) / hidden_dim))
        position_encoding[:, 0::2] = torch.sin(position_index * frequency_terms)
        position_encoding[:, 1::2] = torch.cos(position_index * frequency_terms)
        self.register_buffer("pe", position_encoding)

    def forward(self, sequence_states: torch.Tensor):
        return sequence_states + self.pe[: sequence_states.size(1)]


class ModalityTransformer(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, n_heads: int, n_layers: int, dropout: float, max_len: int = 50):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.pos_enc = PositionalEncoding(hidden_dim, max_len)
        encoder_layer = InstrumentedTransformerEncoderLayer(hidden_dim, n_heads, dropout)
        self.encoder = InstrumentedTransformerEncoder(encoder_layer, num_layers=n_layers)

    def forward(self, modality_sequence: torch.Tensor, return_attention: bool = False):
        projected_sequence = self.input_proj(modality_sequence)
        positioned_sequence = self.pos_enc(projected_sequence)
        if return_attention:
            encoded_sequence, attention_maps = self.encoder(positioned_sequence, return_attention=True)
            return encoded_sequence.mean(dim=1), attention_maps
        encoded_sequence = self.encoder(positioned_sequence)
        return encoded_sequence.mean(dim=1)


class CrossModalAttentionBlock(nn.Module):
    def __init__(self, hidden_dim: int, n_heads: int, dropout: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query_token: torch.Tensor, context_tokens: torch.Tensor, return_attention: bool = False):
        attention_output, attention_weights = self.attn(
            query_token,
            context_tokens,
            context_tokens,
            need_weights=return_attention,
            average_attn_weights=False,
        )
        updated_query = query_token + self.dropout(attention_output)
        updated_query = self.norm(updated_query).squeeze(1)
        if return_attention:
            return updated_query, attention_weights
        return updated_query


class LateFusionWithCrossModalOrtho(nn.Module):
    """Cross-modal model that also returns modality features for orthogonality loss."""

    def __init__(
        self,
        D_text: int,
        D_audio: int,
        D_vision: int,
        hidden_dim: int = 128,
        n_heads: int = 4,
        n_layers: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.text_enc = ModalityTransformer(D_text, hidden_dim, n_heads, n_layers, dropout)
        self.audio_enc = ModalityTransformer(D_audio, hidden_dim, n_heads, n_layers, dropout)
        self.vision_enc = ModalityTransformer(D_vision, hidden_dim, n_heads, n_layers, dropout)

        self.cross_attn = CrossModalAttentionBlock(hidden_dim, n_heads, dropout)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3),
        )

    def forward(self, text, audio, vision, return_attention: bool = False):
        if return_attention:
            text_summary, text_attention = self.text_enc(text, return_attention=True)
            audio_summary, audio_attention = self.audio_enc(audio, return_attention=True)
            vision_summary, vision_attention = self.vision_enc(vision, return_attention=True)
        else:
            text_summary = self.text_enc(text)
            audio_summary = self.audio_enc(audio)
            vision_summary = self.vision_enc(vision)

        text_query = text_summary.unsqueeze(1)
        audio_query = audio_summary.unsqueeze(1)
        vision_query = vision_summary.unsqueeze(1)

        text_context = torch.stack([audio_summary, vision_summary], dim=1)
        audio_context = torch.stack([text_summary, vision_summary], dim=1)
        vision_context = torch.stack([text_summary, audio_summary], dim=1)

        if return_attention:
            cross_attended_text, text_cross_attention = self.cross_attn(text_query, text_context, return_attention=True)
            cross_attended_audio, audio_cross_attention = self.cross_attn(audio_query, audio_context, return_attention=True)
            cross_attended_vision, vision_cross_attention = self.cross_attn(vision_query, vision_context, return_attention=True)
        else:
            cross_attended_text = self.cross_attn(text_query, text_context)
            cross_attended_audio = self.cross_attn(audio_query, audio_context)
            cross_attended_vision = self.cross_attn(vision_query, vision_context)

        fused_representation = torch.cat([cross_attended_text, cross_attended_audio, cross_attended_vision], dim=1)
        logits = self.classifier(fused_representation)

        if return_attention:
            analysis = {
                "features": {"text": text_summary, "audio": audio_summary, "vision": vision_summary},
                "self_attention": {
                    "text": text_attention,
                    "audio": audio_attention,
                    "vision": vision_attention,
                },
                "cross_attention": {
                    "text_query": {"weights": text_cross_attention, "keys": ["audio", "vision"]},
                    "audio_query": {"weights": audio_cross_attention, "keys": ["text", "vision"]},
                    "vision_query": {"weights": vision_cross_attention, "keys": ["text", "audio"]},
                },
            }
            return logits, text_summary, audio_summary, vision_summary, analysis
        return logits, text_summary, audio_summary, vision_summary
