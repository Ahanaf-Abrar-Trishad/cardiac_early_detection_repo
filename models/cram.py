#!/usr/bin/env python3
"""
CRAM: Context Recalibration Attention Module.
Enhances U-Net with spatial and channel attention.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    """Squeeze-and-Excitation style channel attention."""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        B, C = x.shape[:2]
        
        # Average and max pooling
        avg_out = self.fc(self.avg_pool(x).view(B, C))
        max_out = self.fc(self.max_pool(x).view(B, C))
        
        # Combine
        out = self.sigmoid(avg_out + max_out).view(B, C, 1, 1, 1)
        return x * out


class SpatialAttention(nn.Module):
    """Spatial attention using avg and max pooling."""
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv3d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # Channel-wise pooling
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        
        # Concatenate and convolve
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.sigmoid(self.conv(out))
        return x * out


class CRAM(nn.Module):
    """
    Context Recalibration Attention Module.
    Combines channel and spatial attention for feature recalibration.
    """
    def __init__(self, channels, reduction=16, kernel_size=7):
        super().__init__()
        self.channel_att = ChannelAttention(channels, reduction)
        self.spatial_att = SpatialAttention(kernel_size)
    
    def forward(self, x):
        x = self.channel_att(x)
        x = self.spatial_att(x)
        return x


class DoubleConv3DWithCRAM(nn.Module):
    """Double 3D convolution with CRAM attention."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True)
        )
        self.cram = CRAM(out_ch)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.cram(x)
        return x


class UNet3DCRAM(nn.Module):
    """
    3D U-Net with CRAM attention blocks.
    Uses anisotropic pooling (pools only in H/W, not depth) for cardiac volumes.
    """
    def __init__(self, in_ch=1, out_ch=4, feat=(16, 32, 64, 128)):
        super().__init__()
        
        # Encoder
        self.enc1 = DoubleConv3DWithCRAM(in_ch, feat[0])
        self.enc2 = DoubleConv3DWithCRAM(feat[0], feat[1])
        self.enc3 = DoubleConv3DWithCRAM(feat[1], feat[2])
        self.enc4 = DoubleConv3DWithCRAM(feat[2], feat[3])
        
        # Bottleneck
        self.bottleneck = DoubleConv3DWithCRAM(feat[3], feat[3] * 2)
        
        # Decoder - anisotropic upsampling (1,2,2)
        self.up4 = nn.ConvTranspose3d(feat[3] * 2, feat[3], kernel_size=(1,2,2), stride=(1,2,2))
        self.dec4 = DoubleConv3DWithCRAM(feat[3] * 2, feat[3])
        
        self.up3 = nn.ConvTranspose3d(feat[3], feat[2], kernel_size=(1,2,2), stride=(1,2,2))
        self.dec3 = DoubleConv3DWithCRAM(feat[2] * 2, feat[2])
        
        self.up2 = nn.ConvTranspose3d(feat[2], feat[1], kernel_size=(1,2,2), stride=(1,2,2))
        self.dec2 = DoubleConv3DWithCRAM(feat[1] * 2, feat[1])
        
        self.up1 = nn.ConvTranspose3d(feat[1], feat[0], kernel_size=(1,2,2), stride=(1,2,2))
        self.dec1 = DoubleConv3DWithCRAM(feat[0] * 2, feat[0])
        
        # Output
        self.out_conv = nn.Conv3d(feat[0], out_ch, kernel_size=1)
    
    @staticmethod
    def _adaptive_pool(x):
        """Adaptive pooling: only pool in H/W if dimensions are large enough."""
        D, H, W = x.shape[2:]
        if H >= 2 and W >= 2:
            # Pool only in spatial dimensions (H, W), keep depth (D) unchanged
            return F.max_pool3d(x, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        else:
            return x
    
    def forward(self, x):
        # Encoder with anisotropic pooling
        e1 = self.enc1(x)
        e2 = self.enc2(self._adaptive_pool(e1))
        e3 = self.enc3(self._adaptive_pool(e2))
        e4 = self.enc4(self._adaptive_pool(e3))
        
        # Bottleneck
        b = self.bottleneck(self._adaptive_pool(e4))
        
        # Decoder with skip connections
        d4 = self.up4(b)
        if d4.shape[2:] != e4.shape[2:]:
            d4 = F.interpolate(d4, size=e4.shape[2:], mode='trilinear', align_corners=False)
        d4 = self.dec4(torch.cat([d4, e4], dim=1))
        
        d3 = self.up3(d4)
        if d3.shape[2:] != e3.shape[2:]:
            d3 = F.interpolate(d3, size=e3.shape[2:], mode='trilinear', align_corners=False)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        
        d2 = self.up2(d3)
        if d2.shape[2:] != e2.shape[2:]:
            d2 = F.interpolate(d2, size=e2.shape[2:], mode='trilinear', align_corners=False)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        
        d1 = self.up1(d2)
        if d1.shape[2:] != e1.shape[2:]:
            d1 = F.interpolate(d1, size=e1.shape[2:], mode='trilinear', align_corners=False)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        
        return self.out_conv(d1)
