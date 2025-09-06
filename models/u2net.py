"""
Main U2NET model with integrated ELA encoder and HAFT processing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import EncodeELA
from .blocks import Classifier, RSU7, RSU6, RSU5, RSU4, RSU4F, _upsample_like
from .attention import AttentionGateV2


class U2NET(nn.Module):
    """
    U2NET model for fake face manipulation detection
    Integrates RGB processing, frequency domain analysis, and HAFT
    """

    def __init__(self, config, in_ch=3, out_ch=1):
        super(U2NET, self).__init__()

        # RGB processing stages (original U2NET architecture)
        self.stage1 = RSU7(in_ch, 32, 64)
        self.pool12 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage2 = RSU6(64, 32, 128)
        self.pool23 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage3 = RSU5(128, 64, 256)
        self.pool34 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage4 = RSU4(256, 128, 512)
        self.pool45 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage5 = RSU4F(512, 256, 512)
        self.pool56 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage6 = RSU4F(512, 256, 512)

        # ELA encoder with HAFT integration
        self.encode_ela = EncodeELA(config)
        
        # Classification head
        self.classifier = Classifier(config)
        
        # Attention gates for feature fusion
        self.gate5 = AttentionGateV2(512)
        self.gate4 = AttentionGateV2(512)
        self.gate3 = AttentionGateV2(256)
        self.gate2 = AttentionGateV2(128)
        self.gate1 = AttentionGateV2(64)

        # Decoder stages
        self.stage5d = RSU4F(1024, 256, 512)
        self.stage4d = RSU4(1024, 128, 256)
        self.stage3d = RSU5(512, 64, 128)
        self.stage2d = RSU6(256, 32, 64)
        self.stage1d = RSU7(128, 16, 64)

        # Side output convolutions for multi-scale supervision
        self.side1 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side2 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side3 = nn.Conv2d(128, out_ch, 3, padding=1)
        self.side4 = nn.Conv2d(256, out_ch, 3, padding=1)
        self.side5 = nn.Conv2d(512, out_ch, 3, padding=1)
        self.side6 = nn.Conv2d(512, out_ch, 3, padding=1)

        # Final output convolution
        self.outconv = nn.Conv2d(6 * out_ch, out_ch, 1)

    def forward(self, x, ela, gray_img):
        """
        Forward pass with separate RGB, frequency features, and grayscale input for HAFT.
        
        Args:
            x: RGB image features (B, 3, H, W)
            ela: Frequency features tensor (B, 8, H, W) - SWT(2) + DCT(1) + Laplacian(2) + SRM(3) = 8 channels
            gray_img: Grayscale image tensor (B, 1, H, W) for HAFT processing
            
        Returns:
            Tuple containing:
            - mask: Predicted manipulation mask (B, 1, H, W)
            - pred_label: Classification logits (B, num_classes)
        """

        # Encode ELA image with HAFT integration
        ela1, ela2, ela3, ela4, ela5 = self.encode_ela(ela, gray_img)

        hx = x

        # RGB processing stages (encoder)
        hx1 = self.stage1(hx)
        hx = self.pool12(hx1)

        hx2 = self.stage2(hx)
        hx = self.pool23(hx2)

        hx3 = self.stage3(hx)
        hx = self.pool34(hx3)

        hx4 = self.stage4(hx)
        hx = self.pool45(hx4)

        hx5 = self.stage5(hx)
        hx = self.pool56(hx5)

        hx6 = self.stage6(hx)
        hx6up = _upsample_like(hx6, hx5)

        # Attention-guided feature fusion
        attn1 = self.gate1(hx1, ela1)
        attn2 = self.gate2(hx2, ela2)
        attn3 = self.gate3(hx3, ela3)
        attn4 = self.gate4(hx4, ela4)
        attn5 = self.gate5(hx5, ela5)
        
        # Classification from the deepest ELA features
        pred_label = self.classifier(ela5)
        
        # Decoder stages with fused features
        hx5d = self.stage5d(torch.cat((hx6up, attn5), 1))
        hx5dup = _upsample_like(hx5d, hx4)
        
        hx4d = self.stage4d(torch.cat((hx5dup, attn4), 1))
        hx4dup = _upsample_like(hx4d, hx3)

        hx3d = self.stage3d(torch.cat((hx4dup, attn3), 1))
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = self.stage2d(torch.cat((hx3dup, attn2), 1))
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.stage1d(torch.cat((hx2dup, attn1), 1))

        # Side outputs for multi-scale supervision
        d1 = self.side1(hx1d)

        d2 = self.side2(hx2d)
        d2 = _upsample_like(d2, d1)

        d3 = self.side3(hx3d)
        d3 = _upsample_like(d3, d1)

        d4 = self.side4(hx4d)
        d4 = _upsample_like(d4, d1)

        d5 = self.side5(hx5d)
        d5 = _upsample_like(d5, d1)

        d6 = self.side6(hx6)
        d6 = _upsample_like(d6, d1)

        # Final output combining all scales
        d0 = self.outconv(torch.cat((d1, d2, d3, d4, d5, d6), 1))

        return F.sigmoid(d0), pred_label


def create_model(config):
    """
    Create U2NET model with configuration
    
    Args:
        config: Configuration object
        
    Returns:
        U2NET model instance
    """
    model = U2NET(config)
    return model
