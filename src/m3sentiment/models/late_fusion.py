import torch
import torch.nn as nn
import math
from m3sentiment.attention_layers import InstrumentedTransformerEncoder, InstrumentedTransformerEncoderLayer


class PositionalEncoding(nn.Module):
    def __init__(self, hidden_dim: int, max_len: int = 50):
        super().__init__()
        pe = torch.zeros(max_len, hidden_dim)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, hidden_dim, 2).float() * (-math.log(10000.0) / hidden_dim))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)  # (max_len, hidden_dim)

    def forward(self, x: torch.Tensor):
        # x: (B, T, hidden_dim)
        return x + self.pe[: x.size(1)]


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
        # Token embedding
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        # Positional encoding
        self.pos_enc = PositionalEncoding(hidden_dim, max_len)
        # Transformer encoder
        layer = InstrumentedTransformerEncoderLayer(hidden_dim, n_heads, dropout)
        self.encoder = InstrumentedTransformerEncoder(layer, num_layers=n_layers)

    def forward(self, x: torch.Tensor, return_attention: bool = False):
        # x: (B, T, in_dim)
        h = self.input_proj(x)     # (B, T, H)
        h = self.pos_enc(h)        # Add positional encoding (B, T, H)
        if return_attention:
            h, attentions = self.encoder(h, return_attention=True)        # (B, T, H)
            return h.mean(dim=1), attentions       # (B, H), list[(B, heads, T, T)]
        h = self.encoder(h)        # (B, T, H)
        return h.mean(dim=1)       # (B, H)


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
        # Per-modality transformers
        self.text_enc = ModalityTransformer(D_text, hidden_dim, n_heads, n_layers, dropout, max_len=50)
        self.audio_enc = ModalityTransformer(D_audio, hidden_dim, n_heads, n_layers, dropout, max_len=50)
        self.vision_enc = ModalityTransformer(D_vision, hidden_dim, n_heads, n_layers, dropout, max_len=50)

        # Learned modality embeddings for late fusion
        # nn.Parameter makes them trainable parameters of the model
        self.modality_tokens = nn.Parameter(torch.randn(3, hidden_dim))

        # Late-fusion transformer (3 tokens)
        fuse_layer = InstrumentedTransformerEncoderLayer(hidden_dim, n_heads, dropout)
        self.fusion = InstrumentedTransformerEncoder(fuse_layer, num_layers=n_layers)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3),
        )

    def forward(self, text, audio, vision, return_attention: bool = False):
        """
        Args:
          text:  (B, 50, D_text)
          audio: (B, 50, D_audio)
          vision:(B, 50, D_vision)
        Returns:
          logits: (B, 3)
        """
        # 1) Per-modality encoding
        if return_attention:
            t, text_attn = self.text_enc(text, return_attention=True)       # (B, H)
            a, audio_attn = self.audio_enc(audio, return_attention=True)     # (B, H)
            v, vision_attn = self.vision_enc(vision, return_attention=True)  # (B, H)
        else:
            t = self.text_enc(text)       # (B, H)
            a = self.audio_enc(audio)     # (B, H)
            v = self.vision_enc(vision)   # (B, H)
        features = {"text": t, "audio": a, "vision": v}

        # 2) Add modality token embeddings
        t = t + self.modality_tokens[0]
        a = a + self.modality_tokens[1]
        v = v + self.modality_tokens[2]

        # 3) Stack and fuse
        x = torch.stack([t, a, v], dim=1)  # (B, 3, H)
        if return_attention:
            fused, fusion_attn = self.fusion(x, return_attention=True)       # (B, 3, H)
        else:
            fused = self.fusion(x)             # (B, 3, H)

        # 4) Pool and classify
        pooled = fused.mean(dim=1)         # (B, H)
        logits = self.classifier(pooled)     # (B, 3)
        if return_attention:
            analysis = {
                "features": features,
                "cross_attention": {},
                "self_attention": {
                    "text": text_attn,
                    "audio": audio_attn,
                    "vision": vision_attn,
                },
                "fusion_attention": {
                    "weights": fusion_attn,
                    "tokens": ["text", "audio", "vision"],
                },
            }
            return logits, analysis
        return logits
