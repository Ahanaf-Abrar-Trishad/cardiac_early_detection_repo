#!/usr/bin/env python3
"""
UNETR: Transformer-based 3D segmentation model.
Combines Vision Transformer encoder with CNN decoder.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class PatchEmbedding3D(nn.Module):
    """Convert 3D volume into patches and project to embedding dimension."""
    def __init__(self, in_channels=1, patch_size=(16, 16, 16), embed_dim=768):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv3d(
            in_channels, embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
    
    def forward(self, x):
        # x: [B, C, D, H, W]
        x = self.proj(x)  # [B, embed_dim, D', H', W']
        # Flatten spatial dims
        x = rearrange(x, 'b c d h w -> b (d h w) c')
        return x


class TransformerBlock(nn.Module):
    """Standard Transformer encoder block with MSA and MLP."""
    def __init__(self, embed_dim=768, num_heads=12, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        mlp_hidden = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, embed_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        # Multi-head self-attention
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out
        
        # MLP
        x = x + self.mlp(self.norm2(x))
        return x


class TransformerEncoder(nn.Module):
    """Stack of Transformer blocks."""
    def __init__(self, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return self.norm(x)


class DecoderBlock3D(nn.Module):
    """CNN decoder block with skip connection."""
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch // 2 + skip_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x, skip):
        x = self.up(x)
        # Match spatial dimensions
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='trilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class UNETR(nn.Module):
    """
    UNETR: Transformer-based 3D U-Net.
    
    Args:
        in_channels: Input channels (default: 1)
        out_channels: Output channels (default: 4 for ACDC with BG)
        img_size: Input volume size (D, H, W)
        patch_size: Patch size for embedding
        embed_dim: Transformer embedding dimension
        depth: Number of Transformer blocks
        num_heads: Number of attention heads
        feature_size: CNN decoder feature dimension
    """
    def __init__(
        self,
        in_channels=1,
        out_channels=4,
        img_size=(64, 128, 128),
        patch_size=(16, 16, 16),
        embed_dim=768,
        depth=12,
        num_heads=12,
        feature_size=16
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        
        # Patch embedding
        self.patch_embed = PatchEmbedding3D(in_channels, patch_size, embed_dim)
        
        # Transformer encoder
        self.transformer = TransformerEncoder(embed_dim, depth, num_heads)
        
        # Calculate number of patches
        self.num_patches = (
            (img_size[0] // patch_size[0]) *
            (img_size[1] // patch_size[1]) *
            (img_size[2] // patch_size[2])
        )
        
        # Skip connection projections from transformer intermediate layers
        self.skip_proj1 = nn.Conv3d(embed_dim, feature_size * 8, kernel_size=1)
        self.skip_proj2 = nn.Conv3d(embed_dim, feature_size * 4, kernel_size=1)
        self.skip_proj3 = nn.Conv3d(embed_dim, feature_size * 2, kernel_size=1)
        
        # CNN decoder
        self.decoder4 = nn.Sequential(
            nn.Conv3d(embed_dim, feature_size * 16, kernel_size=3, padding=1),
            nn.BatchNorm3d(feature_size * 16),
            nn.ReLU(inplace=True)
        )
        
        self.decoder3 = DecoderBlock3D(feature_size * 16, feature_size * 8, feature_size * 8)
        self.decoder2 = DecoderBlock3D(feature_size * 8, feature_size * 4, feature_size * 4)
        self.decoder1 = DecoderBlock3D(feature_size * 4, feature_size * 2, feature_size * 2)
        
        # Final upsampling and output
        self.final_up = nn.Sequential(
            nn.ConvTranspose3d(feature_size * 2, feature_size, kernel_size=2, stride=2),
            nn.Conv3d(feature_size, out_channels, kernel_size=1)
        )
    
    def forward(self, x):
        B, C, D, H, W = x.shape
        original_size = (D, H, W)
        
        # Pad input to be divisible by patch size
        pad_d = (self.patch_size[0] - D % self.patch_size[0]) % self.patch_size[0]
        pad_h = (self.patch_size[1] - H % self.patch_size[1]) % self.patch_size[1]
        pad_w = (self.patch_size[2] - W % self.patch_size[2]) % self.patch_size[2]
        
        if pad_d > 0 or pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h, 0, pad_d), mode='constant', value=0)
        
        D_padded, H_padded, W_padded = x.shape[2:]
        
        # Patch embedding
        x_embed = self.patch_embed(x)  # [B, num_patches, embed_dim]
        
        # Transformer encoding (extract features at multiple depths)
        # For simplicity, we'll extract from layers 3, 6, 9, 12
        features = []
        x_trans = x_embed
        for i, block in enumerate(self.transformer.blocks):
            x_trans = block(x_trans)
            if i in [2, 5, 8]:  # Extract skip features
                features.append(x_trans)
        
        # Final transformer output
        z12 = self.transformer.norm(x_trans)  # [B, num_patches, embed_dim]
        
        # Reshape to 3D for decoder
        patch_dims = (
            D_padded // self.patch_size[0],
            H_padded // self.patch_size[1],
            W_padded // self.patch_size[2]
        )
        
        def reshape_to_3d(feat):
            feat = rearrange(
                feat, 'b (d h w) c -> b c d h w',
                d=patch_dims[0], h=patch_dims[1], w=patch_dims[2]
            )
            return feat
        
        z12_3d = reshape_to_3d(z12)
        z9_3d = self.skip_proj1(reshape_to_3d(features[2]))
        z6_3d = self.skip_proj2(reshape_to_3d(features[1]))
        z3_3d = self.skip_proj3(reshape_to_3d(features[0]))
        
        # Decoder with skip connections
        d4 = self.decoder4(z12_3d)
        d3 = self.decoder3(d4, z9_3d)
        d2 = self.decoder2(d3, z6_3d)
        d1 = self.decoder1(d2, z3_3d)
        
        # Final output
        out = self.final_up(d1)
        
        # Ensure output size matches original input
        if out.shape[2:] != original_size:
            out = F.interpolate(out, size=original_size, mode='trilinear', align_corners=False)
        
        return out


def create_unetr(in_channels=1, out_channels=4, img_size=(64, 128, 128)):
    """
    Factory function for UNETR model with adaptive patch size.
    For cardiac data with small depth (e.g., ACDC with ~7-10 slices),
    uses smaller depth patches to avoid kernel size errors.
    """
    # Adaptive patch size based on input dimensions
    # Use smaller patch in depth for small volumes (like ACDC)
    D, H, W = img_size
    if D <= 16:
        # Small depth (ACDC): use smaller depth patch
        patch_size = (4, 16, 16)  # 4 slices in depth, 16x16 spatial
    else:
        # Normal depth: use isotropic patches
        patch_size = (16, 16, 16)
    
    return UNETR(
        in_channels=in_channels,
        out_channels=out_channels,
        img_size=img_size,
        patch_size=patch_size,
        embed_dim=768,
        depth=12,
        num_heads=12,
        feature_size=16
    )