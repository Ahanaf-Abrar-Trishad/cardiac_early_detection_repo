#!/usr/bin/env python3
"""
Graph Attention Network over grouped physiological tokens.

Inputs: [B, N, token_dim] where N = number of feature groups (e.g., LV, RV, MYO, EF, shape/other).
Applies stacked GAT layers with self-loops, residual connections, and attention pooling.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, heads: int = 4, dropout: float = 0.2):
        super().__init__()
        self.heads = heads
        self.out_dim = out_dim
        self.lin = nn.Linear(in_dim, heads * out_dim, bias=False)
        self.attn = nn.Parameter(torch.randn(heads, out_dim * 2))
        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        x:   [B, N, C]
        adj: [N, N] boolean/float mask (1 for allowed edges, incl. self-loop)
        """
        B, N, _ = x.shape
        h = self.lin(x)  # [B, N, H*out_dim]
        h = h.view(B, N, self.heads, self.out_dim)  # [B, N, H, d]

        # Prepare for attention scoring (broadcast over node pairs)
        h_i = h.unsqueeze(2)  # [B, N, 1, H, d]
        h_j = h.unsqueeze(1)  # [B, 1, N, H, d]
        h_i = h_i.expand(-1, -1, N, -1, -1)  # [B, N, N, H, d]
        h_j = h_j.expand(-1, N, -1, -1, -1)  # [B, N, N, H, d]
        a_input = torch.cat([h_i, h_j], dim=-1)  # [B, N, N, H, 2d]

        # Compute unnormalized attention
        attn_scores = (a_input * self.attn).sum(dim=-1)  # [B, N, N, H]
        attn_scores = self.leaky_relu(attn_scores)

        # Masked softmax over neighbors (include self-loops)
        mask = adj.bool().unsqueeze(-1)  # [N, N, 1]
        attn_scores = attn_scores.masked_fill(~mask, float("-inf"))
        attn_weights = torch.softmax(attn_scores, dim=2)  # neighbor dim
        attn_weights = self.dropout(attn_weights)

        # Weighted sum
        h_out = torch.einsum("bijh,bjhd->bihd", attn_weights, h)  # [B, N, H, d]
        h_out = h_out.reshape(B, N, self.heads * self.out_dim)  # concat heads
        return h_out


class GraphClassifier(nn.Module):
    def __init__(
        self,
        token_dim: int,
        num_tokens: int,
        num_classes: int,
        hidden_dim: int = 128,
        heads: int = 4,
        layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        # fully-connected graph with self-loops
        adj = torch.ones(num_tokens, num_tokens)
        self.register_buffer("adj", adj)

        self.input_proj = nn.Linear(token_dim, hidden_dim)
        self.gat_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        for i in range(layers):
            in_dim = hidden_dim if i == 0 else hidden_dim * heads
            self.gat_layers.append(GraphAttentionLayer(in_dim, hidden_dim, heads=heads, dropout=dropout))
            self.norms.append(nn.LayerNorm(hidden_dim * heads))
        self.attn_pool = nn.Linear(hidden_dim * heads, 1)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * heads, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        tokens: [B, N, token_dim]
        """
        x = self.input_proj(tokens)  # [B, N, hidden]
        adj = self.adj.to(tokens.device)
        for gat, norm in zip(self.gat_layers, self.norms):
            h = gat(x, adj)
            if h.shape == x.shape:
                h = h + x  # residual
            x = norm(h)
            x = F.elu(x)
        scores = self.attn_pool(x)  # [B, N, 1]
        weights = torch.softmax(scores, dim=1)
        pooled = (weights * x).sum(dim=1)  # [B, hidden*heads]
        logits = self.head(pooled)
        return logits
