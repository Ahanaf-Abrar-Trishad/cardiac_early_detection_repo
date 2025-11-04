#!/usr/bin/env python3
"""
Feature Fusion Classifier with RAP (Residual Attention Pooling) blocks.
Combines geometric features, EF features, and volume features for classification.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class RAPBlock(nn.Module):
    """
    Residual Attention Pooling Block.
    Applies attention mechanism with residual connection for feature refinement.
    """
    def __init__(self, in_features, reduction=4):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(in_features, in_features // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(in_features // reduction, in_features),
            nn.Sigmoid()
        )
        self.residual = nn.Sequential(
            nn.Linear(in_features, in_features),
            nn.BatchNorm1d(in_features),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Attention weights
        attn = self.attention(x)
        # Residual transformation
        res = self.residual(x)
        # Apply attention and add residual
        return x + (attn * res)


class CrossModalAttention(nn.Module):
    """
    Cross-modal attention for fusing different feature types.
    Uses query-key-value attention mechanism.
    """
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        
    def forward(self, query, key, value):
        B = query.shape[0]
        
        # Project and reshape for multi-head attention
        Q = self.q_proj(query).view(B, self.num_heads, self.head_dim)
        K = self.k_proj(key).view(B, self.num_heads, self.head_dim)
        V = self.v_proj(value).view(B, self.num_heads, self.head_dim)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)
        
        # Apply attention to values
        out = torch.matmul(attn_weights, V)
        out = out.view(B, -1)
        out = self.out_proj(out)
        
        return out


class FeatureFusionClassifier(nn.Module):
    """
    Multi-modal feature fusion classifier with RAP blocks and cross-modal attention.
    
    Args:
        geometric_dim: Dimension of geometric features
        ef_dim: Dimension of EF features
        vol_dim: Dimension of volume features
        num_classes: Number of output classes
        hidden_dim: Hidden dimension for fusion
        use_cross_attention: Whether to use cross-modal attention
        dropout: Dropout rate
    """
    def __init__(
        self,
        geometric_dim=10,
        ef_dim=2,
        vol_dim=4,
        num_classes=3,
        hidden_dim=128,
        use_cross_attention=True,
        dropout=0.3
    ):
        super().__init__()
        self.use_cross_attention = use_cross_attention
        
        # Feature projection layers with RAP blocks
        self.geo_proj = nn.Sequential(
            nn.Linear(geometric_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            RAPBlock(hidden_dim)
        )
        
        self.ef_proj = nn.Sequential(
            nn.Linear(ef_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            RAPBlock(hidden_dim)
        )
        
        self.vol_proj = nn.Sequential(
            nn.Linear(vol_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            RAPBlock(hidden_dim)
        )
        
        # Cross-modal attention (optional)
        if use_cross_attention:
            self.cross_attn_geo_ef = CrossModalAttention(hidden_dim, num_heads=4)
            self.cross_attn_geo_vol = CrossModalAttention(hidden_dim, num_heads=4)
            self.cross_attn_ef_vol = CrossModalAttention(hidden_dim, num_heads=4)
        
        # Fusion layer with RAP
        fusion_input_dim = hidden_dim * 3 if not use_cross_attention else hidden_dim * 6
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            RAPBlock(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            RAPBlock(hidden_dim)
        )
        
        # Classification head
        self.classifier = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, geometric, ef, volume):
        """
        Args:
            geometric: [B, geometric_dim]
            ef: [B, ef_dim]
            volume: [B, vol_dim]
        
        Returns:
            logits: [B, num_classes]
        """
        # Project features through RAP blocks
        geo_feat = self.geo_proj(geometric)
        ef_feat = self.ef_proj(ef)
        vol_feat = self.vol_proj(volume)
        
        if self.use_cross_attention:
            # Cross-modal attention fusion
            geo_ef = self.cross_attn_geo_ef(geo_feat, ef_feat, ef_feat)
            geo_vol = self.cross_attn_geo_vol(geo_feat, vol_feat, vol_feat)
            ef_vol = self.cross_attn_ef_vol(ef_feat, vol_feat, vol_feat)
            
            # Concatenate original and cross-attended features
            fused = torch.cat([geo_feat, ef_feat, vol_feat, geo_ef, geo_vol, ef_vol], dim=1)
        else:
            # Simple concatenation
            fused = torch.cat([geo_feat, ef_feat, vol_feat], dim=1)
        
        # Final fusion and classification
        fused = self.fusion(fused)
        logits = self.classifier(fused)
        
        return logits


class GatedFusionClassifier(nn.Module):
    """
    Alternative fusion strategy using gating mechanism.
    Learns to weight different feature modalities.
    """
    def __init__(
        self,
        geometric_dim=10,
        ef_dim=2,
        vol_dim=4,
        num_classes=3,
        hidden_dim=128,
        dropout=0.3
    ):
        super().__init__()
        
        # Feature encoders
        self.geo_encoder = nn.Sequential(
            nn.Linear(geometric_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            RAPBlock(hidden_dim)
        )
        
        self.ef_encoder = nn.Sequential(
            nn.Linear(ef_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            RAPBlock(hidden_dim)
        )
        
        self.vol_encoder = nn.Sequential(
            nn.Linear(vol_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            RAPBlock(hidden_dim)
        )
        
        # Gating network
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 3),  # 3 gates for 3 modalities
            nn.Softmax(dim=1)
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
    
    def forward(self, geometric, ef, volume):
        # Encode features
        geo_feat = self.geo_encoder(geometric)
        ef_feat = self.ef_encoder(ef)
        vol_feat = self.vol_encoder(volume)
        
        # Compute gates
        concat_feat = torch.cat([geo_feat, ef_feat, vol_feat], dim=1)
        gates = self.gate(concat_feat)  # [B, 3]
        
        # Weighted fusion
        fused = (
            gates[:, 0:1] * geo_feat +
            gates[:, 1:2] * ef_feat +
            gates[:, 2:3] * vol_feat
        )
        
        # Classify
        logits = self.classifier(fused)
        return logits


def create_fusion_classifier(
    feature_dims,
    num_classes=3,
    fusion_type="rap",
    **kwargs
):
    """
    Factory function for creating fusion classifiers.
    
    Args:
        feature_dims: Dict with keys 'geometric', 'ef', 'volume'
        num_classes: Number of output classes
        fusion_type: 'rap' or 'gated'
        **kwargs: Additional arguments for the classifier
    
    Returns:
        Fusion classifier model
    """
    if fusion_type == "rap":
        return FeatureFusionClassifier(
            geometric_dim=feature_dims.get('geometric', 10),
            ef_dim=feature_dims.get('ef', 2),
            vol_dim=feature_dims.get('volume', 4),
            num_classes=num_classes,
            **kwargs
        )
    elif fusion_type == "gated":
        return GatedFusionClassifier(
            geometric_dim=feature_dims.get('geometric', 10),
            ef_dim=feature_dims.get('ef', 2),
            vol_dim=feature_dims.get('volume', 4),
            num_classes=num_classes,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown fusion_type: {fusion_type}")