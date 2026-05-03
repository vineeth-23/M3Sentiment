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
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        n_heads: int,
        n_layers: int,
        dropout: float,
        max_len: int = 50,
    ):
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


class LateFusionTransformer(nn.Module):
    def __init__(
        self,
        D_text: int,
        D_audio: int,
        D_vision: int,
        hidden_dim: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.text_enc = ModalityTransformer(D_text, hidden_dim, n_heads, n_layers, dropout, max_len=50)
        self.audio_enc = ModalityTransformer(D_audio, hidden_dim, n_heads, n_layers, dropout, max_len=50)
        self.vision_enc = ModalityTransformer(D_vision, hidden_dim, n_heads, n_layers, dropout, max_len=50)

        self.modality_tokens = nn.Parameter(torch.randn(3, hidden_dim))

        fusion_layer = InstrumentedTransformerEncoderLayer(hidden_dim, n_heads, dropout)
        self.fusion = InstrumentedTransformerEncoder(fusion_layer, num_layers=n_layers)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
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

        features = {"text": text_summary, "audio": audio_summary, "vision": vision_summary}

        tagged_text = text_summary + self.modality_tokens[0]
        tagged_audio = audio_summary + self.modality_tokens[1]
        tagged_vision = vision_summary + self.modality_tokens[2]

        fusion_tokens = torch.stack([tagged_text, tagged_audio, tagged_vision], dim=1)
        if return_attention:
            fused_tokens, fusion_attention = self.fusion(fusion_tokens, return_attention=True)
        else:
            fused_tokens = self.fusion(fusion_tokens)

        fused_summary = fused_tokens.mean(dim=1)
        logits = self.classifier(fused_summary)

        if return_attention:
            analysis = {
                "features": features,
                "cross_attention": {},
                "self_attention": {
                    "text": text_attention,
                    "audio": audio_attention,
                    "vision": vision_attention,
                },
                "fusion_attention": {
                    "weights": fusion_attention,
                    "tokens": ["text", "audio", "vision"],
                },
            }
            return logits, analysis
        return logits
