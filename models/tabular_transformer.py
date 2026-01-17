#!/usr/bin/env python3
"""
Tabular Transformer classifier for cardiac diagnosis.

Treats feature groups as tokens, adds a learned phase token, runs a small
Transformer encoder, and applies attention pooling for classification.
"""
import torch
import torch.nn as nn


class TabularTransformerClassifier(nn.Module):
    """
    Args:
        token_dim: padded feature dimension per token (shared across groups)
        num_tokens: number of feature-group tokens (phase token is added internally)
        num_classes: output classes
        d_model: Transformer width
        nhead: attention heads
        depth: encoder layers
        dropout: dropout rate
    """
    def __init__(
        self,
        token_dim: int,
        num_tokens: int,
        num_classes: int,
        d_model: int = 256,
        nhead: int = 8,
        depth: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.num_tokens = num_tokens

        self.input_proj = nn.Linear(token_dim, d_model)
        self.token_type_embed = nn.Embedding(num_tokens + 1, d_model)  # +1 reserved for phase token
        self.phase_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.randn(1, num_tokens + 1, d_model))

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=depth)

        self.attn_pool = nn.Linear(d_model, 1)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tokens: [B, num_tokens, token_dim] padded group features
        Returns:
            logits: [B, num_classes]
        """
        B = tokens.size(0)

        x = self.input_proj(tokens)  # [B, T, d_model]
        token_ids = torch.arange(self.num_tokens, device=tokens.device)
        type_emb = self.token_type_embed(token_ids).unsqueeze(0)  # [1, T, d_model]
        x = x + type_emb

        phase = self.phase_token.expand(B, -1, -1)  # [B, 1, d_model]
        seq = torch.cat([phase, x], dim=1)  # [B, T+1, d_model]
        pos = self.pos_embed[:, : seq.size(1), :]
        seq = seq + pos

        enc = self.encoder(seq)  # [B, T+1, d_model]

        attn_scores = self.attn_pool(enc)  # [B, T+1, 1]
        attn_weights = torch.softmax(attn_scores, dim=1)
        pooled = (attn_weights * enc).sum(dim=1)  # [B, d_model]

        logits = self.head(self.norm(pooled))
        return logits
