# filepath: /home/ahanaf/cardiac_early_detection_repo/models/attention_modules.py
"""
Advanced attention modules: CRAM, RAP blocks for cardiac segmentation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CRAM(nn.Module):
    """
    Context Recalibration Attention Module (CRAM)
    
    Recalibrates feature maps using spatial and channel attention.
    Useful for highlighting cardiac structures.
    
    Args:
        in_channels: number of input channels
        reduction: channel reduction ratio for efficiency
    """
    def __init__(self, in_channels, reduction=16):
        super(CRAM, self).__init__()
        
        # Channel attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        mid_channels = max(in_channels // reduction, 8)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, mid_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid_channels, in_channels, bias=False)
        )
        
        # Spatial attention
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm2d(1)
        )
        
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        b, c, h, w = x.size()
        
        # Channel attention
        avg_pool = self.avg_pool(x).view(b, c)
        max_pool = self.max_pool(x).view(b, c)
        
        avg_out = self.fc(avg_pool)
        max_out = self.fc(max_pool)
        
        channel_att = self.sigmoid(avg_out + max_out).view(b, c, 1, 1)
        x_channel = x * channel_att
        
        # Spatial attention
        avg_spatial = torch.mean(x_channel, dim=1, keepdim=True)
        max_spatial, _ = torch.max(x_channel, dim=1, keepdim=True)
        spatial_input = torch.cat([avg_spatial, max_spatial], dim=1)
        
        spatial_att = self.sigmoid(self.spatial_conv(spatial_input))
        x_out = x_channel * spatial_att
        
        return x_out


class CRAM3D(nn.Module):
    """3D version of CRAM for volumetric data (ACDC)"""
    def __init__(self, in_channels, reduction=16):
        super(CRAM3D, self).__init__()
        
        # Channel attention
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        
        mid_channels = max(in_channels // reduction, 8)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, mid_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid_channels, in_channels, bias=False)
        )
        
        # Spatial attention
        self.spatial_conv = nn.Sequential(
            nn.Conv3d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm3d(1)
        )
        
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        b, c = x.size()[:2]
        
        # Channel attention
        avg_pool = self.avg_pool(x).view(b, c)
        max_pool = self.max_pool(x).view(b, c)
        
        avg_out = self.fc(avg_pool)
        max_out = self.fc(max_pool)
        
        channel_att = self.sigmoid(avg_out + max_out).view(b, c, 1, 1, 1)
        x_channel = x * channel_att
        
        # Spatial attention
        avg_spatial = torch.mean(x_channel, dim=1, keepdim=True)
        max_spatial, _ = torch.max(x_channel, dim=1, keepdim=True)
        spatial_input = torch.cat([avg_spatial, max_spatial], dim=1)
        
        spatial_att = self.sigmoid(self.spatial_conv(spatial_input))
        x_out = x_channel * spatial_att
        
        return x_out


class RAPBlock(nn.Module):
    """
    Residual Attention Pooling (RAP) Block
    
    Combines residual connections with attention-weighted pooling.
    Useful for feature fusion in classification.
    
    Args:
        in_features: input feature dimension
        hidden_features: hidden layer dimension
        dropout: dropout rate
    """
    def __init__(self, in_features, hidden_features=None, dropout=0.1):
        super(RAPBlock, self).__init__()
        
        hidden_features = hidden_features or in_features // 2
        
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.dropout = nn.Dropout(dropout)
        
        # Attention weights
        self.attention = nn.Sequential(
            nn.Linear(in_features, in_features // 4),
            nn.ReLU(inplace=True),
            nn.Linear(in_features // 4, in_features),
            nn.Sigmoid()
        )
        
        self.norm = nn.LayerNorm(in_features)
    
    def forward(self, x):
        # Residual path
        residual = x
        
        # Attention-weighted transformation
        att_weights = self.attention(x)
        x = x * att_weights
        
        # MLP
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        
        # Residual connection + normalization
        x = self.norm(x + residual)
        
        return x