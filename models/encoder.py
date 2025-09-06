"""
ELA (Error Level Analysis) Encoder with HAFT integration
Handles frequency domain feature processing with learnable HAFT
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .haft import HAFT
from .blocks import ConvBlock, DeformableConv2d
from .attention import CBAM


class EncodeELA(nn.Module):
    """
    ELA Encoder with integrated HAFT processing
    Processes frequency features and grayscale images through learnable HAFT
    """
    
    def __init__(self, config):
        super(EncodeELA, self).__init__()
        
        # Get HAFT parameters from config
        haft_config = config.get('model.haft', {})
        num_haft_levels = haft_config.get('num_levels', 3)
        num_radial_bins = haft_config.get('num_radial_bins', 16)
        context_vector_dim = haft_config.get('context_vector_dim', 64)
        
        # Add HAFT as a learnable module for processing grayscale input
        self.haft_processor = HAFT(
            in_channels=1,  # Grayscale input
            num_haft_levels=num_haft_levels,
            num_radial_bins=num_radial_bins,
            context_vector_dim=context_vector_dim
        )
        
        # Get attention parameters from config
        attention_config = config.get('model.attention.cbam', {})
        reduction_ratio = attention_config.get('reduction_ratio', 16)
        kernel_size = attention_config.get('kernel_size', 7)
        
        # Ultra-speed optimization: Enable memory format optimization
        self.enable_channels_last = True
        
        # Updated frequency features: SWT(2) + DCT(1) + Laplacian(2) + SRM(3) = 8 channels
        # HAFT output: 1 channel (enhanced grayscale features)
        # Total: 8 + 1 = 9 channels
        self.conv1 = ConvBlock(9, 64)  # 8 frequency + 1 HAFT = 9 channels
        self.cbam1 = CBAM(64, reduction_ratio, kernel_size)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = ConvBlock(64, 128)
        self.cbam2 = CBAM(128, reduction_ratio, kernel_size)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv3 = DeformableConv2d(128, 256)
        self.cbam3 = CBAM(256, reduction_ratio, kernel_size)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv4 = DeformableConv2d(256, 512)
        self.cbam4 = CBAM(512, reduction_ratio, kernel_size)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv5 = DeformableConv2d(512, 512)
        self.cbam5 = CBAM(512, reduction_ratio, kernel_size)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Ultra-speed optimization: Pre-compute interpolation modes
        self.interpolation_mode = 'bilinear'
        self.align_corners = False

    def forward(self, freq_img, gray_img):
        """
        Ultra-optimized forward pass with learnable HAFT processing.
        
        Args:
            freq_img: Frequency features tensor (B, 8, H, W)
            gray_img: Grayscale image tensor (B, 1, H, W) for HAFT processing
            
        Returns:
            Tuple of feature maps at different scales (pool1, pool2, pool3, pool4, pool5)
        """
        # Convert to channels_last for better performance if enabled
        if self.enable_channels_last:
            freq_img = freq_img.to(memory_format=torch.channels_last)
            gray_img = gray_img.to(memory_format=torch.channels_last)
        
        # Process grayscale image through learnable HAFT
        haft_features = self.haft_processor(gray_img)  # (B, 1, H, W)
        
        # Resize HAFT features to match frequency features if needed
        if haft_features.shape[-2:] != freq_img.shape[-2:]:
            haft_features = F.interpolate(
                haft_features, 
                size=freq_img.shape[-2:], 
                mode=self.interpolation_mode, 
                align_corners=self.align_corners
            )
            print(f"🔄 Resized HAFT features to {freq_img.shape[-2:]} for concatenation")
        
        # Concatenate 8 frequency features with HAFT output: (B, 9, H, W)
        x = torch.cat([freq_img, haft_features], dim=1)
        
        # Continue with existing processing pipeline
        conv1 = self.conv1(x)  # Now processes 9 channels
        cbam1 = self.cbam1(conv1)
        pool1 = self.pool1(cbam1) 
        
        conv2 = self.conv2(pool1)
        cbam2 = self.cbam2(conv2)
        pool2 = self.pool2(cbam2)
        
        conv3 = self.conv3(pool2)
        cbam3 = self.cbam3(conv3)
        pool3 = self.pool3(cbam3)
        
        conv4 = self.conv4(pool3)
        cbam4 = self.cbam4(conv4)
        pool4 = self.pool4(cbam4)
        
        conv5 = self.conv5(pool4)
        cbam5 = self.cbam5(conv5)
        pool5 = self.pool5(cbam5)
        
        return pool1, pool2, pool3, pool4, pool5
