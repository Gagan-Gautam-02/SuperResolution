import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class SRCNN(nn.Module):
    """Super-Resolution Convolutional Neural Network for dual images"""
    
    def __init__(self, num_channels=3, scale_factor=2):
        super(SRCNN, self).__init__()
        self.scale_factor = scale_factor
        
        # Feature extraction
        self.conv1 = nn.Conv2d(num_channels * 2, 64, kernel_size=9, padding=4)
        
        # Non-linear mapping
        self.conv2 = nn.Conv2d(64, 32, kernel_size=1, padding=0)
        
        # Reconstruction
        self.conv3 = nn.Conv2d(32, num_channels, kernel_size=5, padding=2)
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        # Upsample input images
        img1_up = F.interpolate(img1, scale_factor=self.scale_factor, mode='bicubic', align_corners=False)
        img2_up = F.interpolate(img2, scale_factor=self.scale_factor, mode='bicubic', align_corners=False)
        
        # Concatenate dual images
        x = torch.cat([img1_up, img2_up], dim=1)
        
        # Feature extraction
        x = self.relu(self.conv1(x))
        
        # Non-linear mapping
        x = self.relu(self.conv2(x))
        
        # Reconstruction
        x = self.conv3(x)
        
        # Add residual connection
        x = x + img1_up
        
        return x

class ResidualBlock(nn.Module):
    """Residual block for EDSR"""
    
    def __init__(self, num_channels=64):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        residual = x
        x = self.relu(self.conv1(x))
        x = self.conv2(x)
        x += residual
        return x

class EDSR(nn.Module):
    """Enhanced Deep Residual Networks for dual image super-resolution"""
    
    def __init__(self, num_channels=3, num_blocks=16, scale_factor=2):
        super(EDSR, self).__init__()
        self.scale_factor = scale_factor
        
        # Initial feature extraction
        self.conv_input = nn.Conv2d(num_channels * 2, 64, kernel_size=3, padding=1)
        
        # Residual blocks
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(64) for _ in range(num_blocks)
        ])
        
        # Upsampling
        self.upconv = nn.Conv2d(64, 64 * (scale_factor ** 2), kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(scale_factor)
        
        # Output layer
        self.conv_output = nn.Conv2d(64, num_channels, kernel_size=3, padding=1)
    
    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        # Concatenate dual images
        x = torch.cat([img1, img2], dim=1)
        
        # Initial feature extraction
        x = self.conv_input(x)
        residual = x
        
        # Residual blocks
        for block in self.residual_blocks:
            x = block(x)
        
        x += residual
        
        # Upsampling
        x = self.upconv(x)
        x = self.pixel_shuffle(x)
        
        # Output
        x = self.conv_output(x)
        
        return x

class ChannelAttention(nn.Module):
    """Channel attention mechanism"""
    
    def __init__(self, num_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(num_channels, num_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(num_channels // reduction, num_channels),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class RCAN_Block(nn.Module):
    """Residual Channel Attention Block"""
    
    def __init__(self, num_channels=64):
        super(RCAN_Block, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        self.ca = ChannelAttention(num_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        residual = x
        x = self.relu(self.conv1(x))
        x = self.conv2(x)
        x = self.ca(x)
        x += residual
        return x

class RCAN(nn.Module):
    """Residual Channel Attention Network for dual images"""
    
    def __init__(self, num_channels=3, num_blocks=20, scale_factor=2):
        super(RCAN, self).__init__()
        self.scale_factor = scale_factor
        
        # Initial feature extraction
        self.conv_input = nn.Conv2d(num_channels * 2, 64, kernel_size=3, padding=1)
        
        # Residual channel attention blocks
        self.rca_blocks = nn.ModuleList([
            RCAN_Block(64) for _ in range(num_blocks)
        ])
        
        # Upsampling
        self.upconv = nn.Conv2d(64, 64 * (scale_factor ** 2), kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(scale_factor)
        
        # Output layer
        self.conv_output = nn.Conv2d(64, num_channels, kernel_size=3, padding=1)
    
    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        # Concatenate dual images
        x = torch.cat([img1, img2], dim=1)
        
        # Initial feature extraction
        x = self.conv_input(x)
        residual = x
        
        # Residual channel attention blocks
        for block in self.rca_blocks:
            x = block(x)
        
        x += residual
        
        # Upsampling
        x = self.upconv(x)
        x = self.pixel_shuffle(x)
        
        # Output
        x = self.conv_output(x)
        
        return x

def create_model(model_type: str, num_channels: int = 3, scale_factor: int = 2) -> nn.Module:
    """Factory function to create super-resolution models"""
    
    models = {
        'srcnn': SRCNN,
        'edsr': EDSR,
        'rcan': RCAN
    }
    
    if model_type.lower() not in models:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return models[model_type.lower()](num_channels=num_channels, scale_factor=scale_factor)
