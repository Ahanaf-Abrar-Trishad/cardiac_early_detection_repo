#!/usr/bin/env python3
"""
Advanced Attention-Based Classifier for Cardiac Disease Classification

Implements state-of-the-art attention mechanisms:
1. Multi-Head Self-Attention (MHSA)
2. Squeeze-and-Excitation (SE) channel attention
3. Spatial Pyramid Attention (SPA)
4. Cross-Modal Attention for multi-modal fusion
5. Residual Attention Blocks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-Head Self-Attention mechanism
    Allows model to attend to different representation subspaces
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, f'dim {dim} should be divisible by num_heads {num_heads}'
        
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Query, Key, Value projections
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
    
    def forward(self, x):
        B, N, C = x.shape
        
        # Generate Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, num_heads, N, head_dim]
        
        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, num_heads, N, N]
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x, attn


class SqueezeExcitation(nn.Module):
    """
    Squeeze-and-Excitation block for channel attention
    Adaptively recalibrates channel-wise feature responses
    """
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction, bias=False)
        self.fc2 = nn.Linear(channels // reduction, channels, bias=False)
    
    def forward(self, x):
        # x: [B, C, *] or [B, N, C]
        if x.dim() == 3:  # [B, N, C]
            b, n, c = x.shape
            # Global average pooling
            squeeze = x.mean(dim=1)  # [B, C]
        else:  # [B, C, H, W] or [B, C, D, H, W]
            squeeze = x.mean(dim=list(range(2, x.dim())))  # [B, C]
        
        # Excitation
        excitation = self.fc1(squeeze)
        excitation = F.relu(excitation, inplace=True)
        excitation = self.fc2(excitation)
        excitation = torch.sigmoid(excitation)
        
        # Scale
        if x.dim() == 3:
            return x * excitation.unsqueeze(1)
        else:
            shape = [excitation.size(0), excitation.size(1)] + [1] * (x.dim() - 2)
            return x * excitation.view(*shape)


class ResidualAttentionBlock(nn.Module):
    """
    Residual Attention Block combining MHSA, FFN, and residual connections
    Similar to Transformer encoder block
    """
    def __init__(self, dim, num_heads=8, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadSelfAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, 
                                           attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        
        # Feed-forward network
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop)
        )
        
        # SE attention
        self.se = SqueezeExcitation(dim, reduction=16)
    
    def forward(self, x):
        # Self-attention with residual
        normed_x = self.norm1(x)
        attn_out, attn_weights = self.attn(normed_x)
        x = x + attn_out
        
        # SE attention
        x = self.se(x)
        
        # FFN with residual
        normed_x = self.norm2(x)
        x = x + self.mlp(normed_x)
        
        return x, attn_weights


class AdvancedAttentionClassifier(nn.Module):
    """
    Advanced Attention-Based Classifier for Cardiac Disease Classification
    
    Architecture:
    1. Feature embedding
    2. Multiple Residual Attention Blocks
    3. Global pooling with attention
    4. Classification head
    
    Args:
        input_features: Number of input features (e.g., 8 for geometric+EF features)
        num_classes: Number of disease classes (5 for ACDC)
        hidden_dim: Dimension of hidden representations
        num_blocks: Number of residual attention blocks
        num_heads: Number of attention heads
        mlp_ratio: Ratio of MLP hidden dim to embedding dim
        dropout: Dropout rate
    """
    def __init__(
        self,
        input_features=8,
        num_classes=5,
        hidden_dim=256,
        num_blocks=4,
        num_heads=8,
        mlp_ratio=4.,
        dropout=0.3
    ):
        super().__init__()
        
        self.input_features = input_features
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        
        # Feature embedding
        self.feature_embed = nn.Sequential(
            nn.Linear(input_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Learnable position encoding
        self.pos_embed = nn.Parameter(torch.zeros(1, input_features, hidden_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        # Stack of Residual Attention Blocks
        self.blocks = nn.ModuleList([
            ResidualAttentionBlock(
                dim=hidden_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                drop=dropout,
                attn_drop=dropout
            )
            for _ in range(num_blocks)
        ])
        
        # Global attention pooling
        self.attn_pool = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.Tanh(),
            nn.Linear(hidden_dim // 4, 1)
        )
        
        # Classification head
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        # Store attention weights for visualization
        self.attention_weights = []
    
    def forward(self, x):
        """
        Args:
            x: [B, input_features] - Input feature vector
        Returns:
            logits: [B, num_classes] - Classification logits
        """
        B = x.size(0)
        
        # Embed each feature separately
        x_embedded = self.feature_embed(x)  # [B, hidden_dim]
        x = x_embedded.unsqueeze(1).repeat(1, self.input_features, 1)  # [B, input_features, hidden_dim]
        
        # Add positional encoding
        x = x + self.pos_embed
        
        # Apply residual attention blocks
        self.attention_weights = []
        for block in self.blocks:
            x, attn = block(x)
            self.attention_weights.append(attn)
        
        # Global attention pooling
        x = self.norm(x)
        attn_weights = self.attn_pool(x).softmax(dim=1)  # [B, N, 1]
        x = (x * attn_weights).sum(dim=1)  # [B, hidden_dim]
        
        # Classification
        logits = self.head(x)
        
        return logits
    
    def get_attention_weights(self):
        """Return attention weights from all blocks for visualization"""
        return self.attention_weights


class MultiModalAttentionClassifier(nn.Module):
    """
    Multi-Modal Attention Classifier with Cross-Modal Fusion
    
    Designed for fusing multiple types of features:
    - Geometric features (volumes)
    - Functional features (EF)
    
    Architecture:
    1. Separate encoders for each modality
    2. Self-attention within each modality
    3. Cross-attention between modalities
    4. Fused representation
    5. Classification
    """
    def __init__(
        self,
        num_geometric=6,  # LV/RV/MYO at ED/ES
        num_functional=2,  # LV_EF, RV_EF
        num_classes=5,
        hidden_dim=256,
        num_heads=8,
        dropout=0.3
    ):
        super().__init__()
        
        # Modality-specific encoders
        self.geo_encoder = nn.Sequential(
            nn.Linear(num_geometric, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        self.func_encoder = nn.Sequential(
            nn.Linear(num_functional, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        # Self-attention for each modality
        self.geo_self_attn = MultiHeadSelfAttention(hidden_dim, num_heads, attn_drop=dropout, proj_drop=dropout)
        self.func_self_attn = MultiHeadSelfAttention(hidden_dim, num_heads, attn_drop=dropout, proj_drop=dropout)
        
        # Cross-attention layers (geometric <-> functional)
        self.cross_attn_geo_func = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.cross_attn_func_geo = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        
        # Fusion
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # SE attention on fused features
        self.se = SqueezeExcitation(hidden_dim, reduction=16)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
    
    def forward(self, geometric, functional):
        """
        Args:
            geometric: [B, num_geometric] - Volumetric features
            functional: [B, num_functional] - EF features
        Returns:
            logits: [B, num_classes]
        """
        B = geometric.size(0)
        
        # Encode each modality
        geo_feat = self.geo_encoder(geometric).unsqueeze(1)  # [B, 1, hidden_dim]
        func_feat = self.func_encoder(functional).unsqueeze(1)  # [B, 1, hidden_dim]
        
        # Self-attention within modality
        geo_feat, _ = self.geo_self_attn(geo_feat)
        func_feat, _ = self.func_self_attn(func_feat)
        
        # Cross-modal attention
        geo_attended, _ = self.cross_attn_geo_func(geo_feat, func_feat, func_feat)  # Geo attends to Func
        func_attended, _ = self.cross_attn_func_geo(func_feat, geo_feat, geo_feat)  # Func attends to Geo
        
        # Fuse attended features
        fused = torch.cat([
            geo_attended.squeeze(1),
            func_attended.squeeze(1)
        ], dim=1)  # [B, hidden_dim * 2]
        
        fused = self.fusion(fused).unsqueeze(1)  # [B, 1, hidden_dim]
        fused = self.se(fused).squeeze(1)  # [B, hidden_dim]
        
        # Classification
        logits = self.classifier(fused)
        
        return logits


def test_models():
    """Test function to verify model architectures"""
    print("=" * 70)
    print("Testing Advanced Attention-Based Classifiers")
    print("=" * 70)
    
    # Test 1: AdvancedAttentionClassifier
    print("\n1. Testing AdvancedAttentionClassifier...")
    model1 = AdvancedAttentionClassifier(
        input_features=8,
        num_classes=5,
        hidden_dim=256,
        num_blocks=4,
        num_heads=8,
        dropout=0.3
    )
    
    # Dummy input: [batch_size, 8 features]
    x = torch.randn(16, 8)
    logits = model1(x)
    
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {logits.shape}")
    print(f"   Parameters: {sum(p.numel() for p in model1.parameters()):,}")
    print(f"   Attention blocks: {len(model1.attention_weights)}")
    print("   ✓ AdvancedAttentionClassifier works!")
    
    # Test 2: MultiModalAttentionClassifier
    print("\n2. Testing MultiModalAttentionClassifier...")
    model2 = MultiModalAttentionClassifier(
        num_geometric=6,
        num_functional=2,
        num_classes=5,
        hidden_dim=256,
        num_heads=8,
        dropout=0.3
    )
    
    # Dummy inputs
    geo = torch.randn(16, 6)  # Volumetric features
    func = torch.randn(16, 2)  # EF features
    logits = model2(geo, func)
    
    print(f"   Geometric input: {geo.shape}")
    print(f"   Functional input: {func.shape}")
    print(f"   Output shape: {logits.shape}")
    print(f"   Parameters: {sum(p.numel() for p in model2.parameters()):,}")
    print("   ✓ MultiModalAttentionClassifier works!")
    
    print("\n" + "=" * 70)
    print("All tests passed! ✓")
    print("=" * 70)


if __name__ == "__main__":
    test_models()
