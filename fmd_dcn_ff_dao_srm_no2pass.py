import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import os
import cv2
import pywt
import numbers
import numpy as np
import torchvision.ops
import matplotlib.pyplot as plt

from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve  # Added for custom EER implementation
from tqdm.auto import tqdm
from einops import rearrange
from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio
from torchmetrics.classification import BinaryAUROC, BinaryF1Score, BinaryPrecision, BinaryRecall, BinarySpecificity, BinaryAccuracy
from torchsummary import summary

"""
HAFT: Hierarchical Adaptive Frequency Transform Block
Implements sophisticated frequency-domain processing
"""


class FrequencyContextEncoder(nn.Module):
    """Small CNN heads for extracting context vectors from frequency patches."""
    
    def __init__(self, context_vector_dim=64):
        super().__init__()
        self.context_vector_dim = context_vector_dim
        
        # Dictionary to store encoders for different levels and types
        self.encoders = nn.ModuleDict()
        
        # Create encoders for different hierarchy levels (0-3) and types (mag/phase)
        for level in range(4):  # max possible levels
            for enc_type in ['mag', 'phase']:
                encoder_name = f'level_{level}_{enc_type}'
                # Small CNN that processes single-channel 2D patches
                self.encoders[encoder_name] = nn.Sequential(
                    nn.Conv2d(1, 16, kernel_size=3, padding=1),
                    nn.BatchNorm2d(16),
                    nn.ReLU(inplace=True),
                    nn.AdaptiveAvgPool2d(4),  # Reduce to 4x4
                    nn.Conv2d(16, 32, kernel_size=3, padding=1),
                    nn.BatchNorm2d(32),
                    nn.ReLU(inplace=True),
                    nn.AdaptiveAvgPool2d(1),  # Global average pooling
                    nn.Flatten(),
                    nn.Linear(32, context_vector_dim)
                )
    
    def forward(self, patch, level, enc_type):
        """
        Args:
            patch: (B, 1, patch_H, patch_W) single-channel frequency patch
            level: int, hierarchy level
            enc_type: str, 'mag' or 'phase'
            
        Returns:
            context_vector: (B, context_vector_dim)
        """
        encoder_name = f'level_{level}_{enc_type}'
        if encoder_name not in self.encoders:
            raise ValueError(f"Encoder {encoder_name} not found")
        
        return self.encoders[encoder_name](patch)


class FilterPredictor(nn.Module):
    """MLPs to predict 1D radial filter profiles."""
    
    def __init__(self, input_dim, num_radial_bins=16, hidden_dim=128):
        super().__init__()
        
        # Separate MLPs for magnitude and phase
        self.mag_mlp_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_radial_bins),
            nn.Softmax(dim=-1)  # Normalize the filter
        )
        
        self.phase_mlp_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_radial_bins),
            nn.Tanh()  # Phase adjustments in [-1, 1]
        )
    
    def forward(self, mag_vector, phase_vector):
        """
        Args:
            mag_vector: (B, input_dim)
            phase_vector: (B, input_dim)
            
        Returns:
            w_mag_profile: (B, num_radial_bins)
            w_phase_profile: (B, num_radial_bins)
        """
        w_mag_profile = self.mag_mlp_head(mag_vector)
        w_phase_profile = self.phase_mlp_head(phase_vector)
        
        return w_mag_profile, w_phase_profile


def reconstruct_radial_filter_vectorized(profile_1d, patch_size, batch_size):
    """
    Vectorized reconstruction of 2D radial filter from 1D profile using Chebyshev distance.
    
    Args:
        profile_1d: (B * num_patches, num_radial_bins) 1D filter profile
        patch_size: (H, W) size of the patch
        batch_size: B for reshaping
        
    Returns:
        filter_2d: (B * num_patches, 1, H, W) 2D radial filter
    """
    total_patches, num_bins = profile_1d.shape
    H, W = patch_size
    
    # Create coordinate grid once
    device = profile_1d.device
    u = torch.arange(H, dtype=torch.float32, device=device).view(-1, 1)
    v = torch.arange(W, dtype=torch.float32, device=device).view(1, -1)
    
    # Calculate Chebyshev distance from center
    center_u, center_v = H / 2.0, W / 2.0
    d = torch.maximum(torch.abs(u - center_u), torch.abs(v - center_v))
    
    # Normalize distance to [0, num_bins-1] range
    max_d = torch.maximum(torch.tensor(center_u, device=device), torch.tensor(center_v, device=device))
    d_normalized = d / max_d * (num_bins - 1)
    d_indices = torch.floor(d_normalized).long().clamp(0, num_bins - 1)
    
    # Vectorized indexing: create filter_2d by advanced indexing
    filter_2d = profile_1d[:, d_indices.reshape(-1)].reshape(total_patches, H, W).unsqueeze(1)
    
    return filter_2d


class HAFT(nn.Module):
    """Hierarchical Adaptive Frequency Transform Block - Optimized for Speed."""
    
    def __init__(self, in_channels, num_haft_levels=3, num_radial_bins=16, 
                 context_vector_dim=64):
        super().__init__()
        self.in_channels = in_channels
        self.num_haft_levels = num_haft_levels
        self.num_radial_bins = num_radial_bins
        self.context_vector_dim = context_vector_dim
        
        # Core components
        self.freq_context_encoder = FrequencyContextEncoder(context_vector_dim)
        
        # Filter predictor input dimension depends on the hierarchical context
        filter_input_dim = num_haft_levels * context_vector_dim
        self.filter_predictor = FilterPredictor(filter_input_dim, num_radial_bins)
        
        # Level embeddings for hierarchical processing - use Linear instead of Embedding
        self.level_projection = nn.Linear(1, 2 * context_vector_dim)
        
        # Output projection to maintain channel dimensions
        self.output_proj = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        
    def forward(self, F_in):
        """
        Args:
            F_in: (B, C, H, W) input feature map
            
        Returns:
            F_out: (B, C, H, W) enhanced feature map
        """
        B, C, H, W = F_in.shape
        
        # SPEED OPTIMIZATION: Process all channels in parallel
        # Instead of sequential processing, use vectorized operations
        
        # Process all channels at once through the HAFT mechanism
        enhanced_channels = self._process_all_channels_parallel(F_in)
        
        # Output projection
        F_out = self.output_proj(enhanced_channels)
        
        return F_out
    
    def _process_all_channels_parallel(self, channel_features):
        """Process all channels through the HAFT mechanism in parallel - HIGHLY OPTIMIZED."""
        B, C, H, W = channel_features.shape
        
        # SPEED OPTIMIZATION: Pre-compute all hierarchical contexts at once
        hierarchical_contexts = self._compute_hierarchical_contexts_batch(channel_features)
        
        if not hierarchical_contexts:
            return channel_features
        
        # SPEED OPTIMIZATION: Process all channels simultaneously
        enhanced_channels = self._apply_frequency_filtering_batch(channel_features, hierarchical_contexts)
        
        return enhanced_channels
    
    def _compute_hierarchical_contexts_batch(self, channel_features):
        """Compute all hierarchical contexts in one highly optimized pass."""
        B, C, H, W = channel_features.shape
        hierarchical_contexts = []
        
        for level in range(self.num_haft_levels):
            num_patches_per_side = 2 ** level
            patch_h = H // num_patches_per_side
            patch_w = W // num_patches_per_side
            
            if patch_h < 4 or patch_w < 4:
                continue
                
            # Extract ALL patches for ALL channels at once
            patches = channel_features.unfold(2, patch_h, patch_h).unfold(3, patch_w, patch_w)
            # patches shape: (B, C, num_patches_h, num_patches_w, patch_h, patch_w)
            
            num_patches_h, num_patches_w = patches.shape[2], patches.shape[3]
            total_patches = num_patches_h * num_patches_w
            
            # Reshape to process ALL patches and channels at once
            patches_reshaped = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
            patches_reshaped = patches_reshaped.view(B * total_patches * C, 1, patch_h, patch_w)
            # Shape: (B * total_patches * C, 1, patch_h, patch_w)
            
            # SPEED OPTIMIZATION: Single batched FFT for all patches and channels
            patches_float = patches_reshaped.squeeze(1).to(torch.float32)
            patch_fft = torch.fft.fft2(patches_float)  # (B * total_patches * C, patch_h, patch_w)
            
            # Extract magnitude and phase for all at once
            magnitude = torch.abs(patch_fft).unsqueeze(1)  # (B * total_patches * C, 1, patch_h, patch_w)
            phase = torch.atan2(patch_fft.imag, patch_fft.real).unsqueeze(1)
            
            # SPEED OPTIMIZATION: Process all encoders in parallel
            level_contexts = []
            for enc_type in ['mag', 'phase']:
                if enc_type == 'mag':
                    context_vectors = self.freq_context_encoder(magnitude, level, enc_type)
                else:
                    context_vectors = self.freq_context_encoder(phase, level, enc_type)
                level_contexts.append(context_vectors)
            
            # Concatenate magnitude and phase contexts
            cv_all = torch.cat(level_contexts, dim=1)  # (B * total_patches * C, 2*context_dim)
            
            # Reshape back to spatial and channel arrangement
            cv_all = cv_all.reshape(B, total_patches, C, 2 * self.context_vector_dim)
            cv_all = cv_all.permute(0, 2, 3, 1)  # (B, C, 2*context_dim, total_patches)
            cv_all = cv_all.reshape(B, C, 2 * self.context_vector_dim, num_patches_h, num_patches_w)
            
            hierarchical_contexts.append(cv_all)
        
        return hierarchical_contexts
    
    def _apply_frequency_filtering_batch(self, channel_features, hierarchical_contexts):
        """Apply frequency domain filtering to all channels in parallel."""
        B, C, H, W = channel_features.shape
        
        deepest_level = len(hierarchical_contexts) - 1
        if deepest_level < 0:
            return channel_features
        
        deepest_contexts = hierarchical_contexts[deepest_level]  # (B, C, 2*context_dim, num_patches_h, num_patches_w)
        num_patches_h, num_patches_w = deepest_contexts.shape[3], deepest_contexts.shape[4]
        num_patches_per_side = int((num_patches_h * num_patches_w) ** 0.5)
        patch_h = H // num_patches_per_side
        patch_w = W // num_patches_per_side
        
        # Extract all patches for all channels at once
        patches = channel_features.unfold(2, patch_h, patch_h).unfold(3, patch_w, patch_w)
        # patches shape: (B, C, num_patches_h, num_patches_w, patch_h, patch_w)
        
        # Flatten for batch processing
        patches_flat = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
        patches_flat = patches_flat.view(B * num_patches_h * num_patches_w * C, 1, patch_h, patch_w)
        
        # SPEED OPTIMIZATION: Gather ancestral contexts efficiently
        ancestral_contexts = self._gather_ancestral_contexts_batch(hierarchical_contexts, B, C, num_patches_h, num_patches_w)
        
        # Add level embeddings
        level_embeddings = self._compute_level_embeddings_batch(hierarchical_contexts, B, num_patches_h, num_patches_w, C)
        enriched_contexts = ancestral_contexts + level_embeddings
        
        # Split into magnitude and phase vectors
        mid_dim = enriched_contexts.shape[3] // 2
        fused_mag_vectors = enriched_contexts[:, :, :, :mid_dim]
        fused_phase_vectors = enriched_contexts[:, :, :, mid_dim:]
        
        # SPEED OPTIMIZATION: Predict all filter profiles at once
        # Use reshape instead of view to handle non-contiguous tensors
        mag_profiles, phase_profiles = self.filter_predictor(
            fused_mag_vectors.reshape(-1, fused_mag_vectors.shape[3]), 
            fused_phase_vectors.reshape(-1, fused_phase_vectors.shape[3])
        )
        # Note: mag_profiles and phase_profiles are (B * num_patches * C, num_radial_bins)
        
        # Apply vectorized radial filter reconstruction
        mag_filters_2d = reconstruct_radial_filter_vectorized(
            mag_profiles, (patch_h, patch_w), B
        )  # (B * num_patches * C, 1, patch_h, patch_w)
        
        phase_filters_2d = reconstruct_radial_filter_vectorized(
            phase_profiles, (patch_h, patch_w), B
        )  # (B * num_patches * C, 1, patch_h, patch_w)
        
        # SPEED OPTIMIZATION: Apply frequency domain filtering in one batch
        patches_float = patches_flat.to(torch.float32).squeeze(1)  # (B * num_patches * C, patch_h, patch_w)
        patch_fft = torch.fft.fft2(patches_float)
        
        magnitude = torch.abs(patch_fft)
        phase = torch.atan2(patch_fft.imag, patch_fft.real)
        
        # Apply learned filters
        enhanced_magnitude = magnitude * mag_filters_2d.squeeze(1)
        enhanced_phase = phase + phase_filters_2d.squeeze(1)
        
        # Reconstruct enhanced patches
        enhanced_fft = enhanced_magnitude * torch.exp(1j * enhanced_phase)
        enhanced_patches = torch.real(torch.fft.ifft2(enhanced_fft)).unsqueeze(1)
        
        # Convert back to original dtype
        enhanced_patches = enhanced_patches.to(channel_features.dtype)
        
        # Reshape back to original arrangement
        enhanced_patches = enhanced_patches.reshape(B, num_patches_h, num_patches_w, C, patch_h, patch_w)
        enhanced_patches = enhanced_patches.permute(0, 3, 1, 4, 2, 5).contiguous()
        enhanced_patches = enhanced_patches.reshape(B, C, H, W)
        
        return enhanced_patches
    
    def _gather_ancestral_contexts_batch(self, hierarchical_contexts, B, C, num_patches_h, num_patches_w):
        """Gather ancestral contexts for all patches and channels efficiently."""
        total_patches = num_patches_h * num_patches_w
        num_patches_per_side = int((num_patches_h * num_patches_w) ** 0.5)
        ancestral_contexts_all = []
        
        for level in range(len(hierarchical_contexts)):
            level_contexts = hierarchical_contexts[level]  # (B, C, 2*context_dim, level_patches_h, level_patches_w)
            level_patches_h, level_patches_w = level_contexts.shape[3], level_contexts.shape[4]
            
            # Flatten level contexts for all channels
            level_contexts_flat = level_contexts.reshape(B, C, 2 * self.context_vector_dim, -1)
            level_contexts_flat = level_contexts_flat.permute(0, 1, 3, 2)  # (B, C, num_level_patches, 2*context_dim)
            
            # SPEED OPTIMIZATION: Pre-compute mapping indices for all patches
            level_patches_per_side = 2 ** level
            scale_factor = num_patches_per_side // level_patches_per_side
            
            # Create mapping indices for all patches at once
            patch_indices = torch.arange(total_patches, device=level_contexts.device)
            row_indices = patch_indices // num_patches_per_side
            col_indices = patch_indices % num_patches_per_side
            
            if scale_factor <= 1:
                mapped_indices = torch.clamp(patch_indices, 0, level_contexts_flat.shape[2] - 1)
            else:
                mapped_row_indices = row_indices // scale_factor
                mapped_col_indices = col_indices // scale_factor
                mapped_indices = mapped_row_indices * level_patches_per_side + mapped_col_indices
                mapped_indices = torch.clamp(mapped_indices, 0, level_contexts_flat.shape[2] - 1)
            
            # Gather ancestral contexts for all channels at once
            ancestral_level = level_contexts_flat[:, :, mapped_indices, :]  # (B, C, total_patches, 2*context_dim)
            ancestral_contexts_all.append(ancestral_level)
        
        # Concatenate all ancestral contexts
        fused_context_vectors = torch.cat(ancestral_contexts_all, dim=3)  # (B, C, total_patches, total_context_dim)
        
        return fused_context_vectors
    
    def _compute_level_embeddings_batch(self, hierarchical_contexts, B, num_patches_h, num_patches_w, C):
        """Compute level embeddings for all patches and channels efficiently."""
        total_patches = num_patches_h * num_patches_w
        level_inputs = []
        
        for level in range(len(hierarchical_contexts)):
            # Create level tensor for all patches and channels at once
            level_tensor = torch.full((B, C, total_patches, 1), 
                                    float(level), device=next(self.parameters()).device, dtype=torch.float32)
            level_emb = self.level_projection(level_tensor)  # (B, C, total_patches, 2*context_dim)
            level_inputs.append(level_emb)
        
        level_embeddings = torch.cat(level_inputs, dim=3)  # (B, C, total_patches, total_context_dim)
        
        return level_embeddings


random.seed(42)

# def high_pass_filter_image(image_path, radius=30):
#     image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

#     h, w = image.shape[:2]

#     if h != 160 or w != 160:
#         image = cv2.resize(image, (160, 160))
    
#     f = np.fft.fft2(image)
#     fshift = np.fft.fftshift(f)
    
#     rows, cols = image.shape
#     crow, ccol = rows // 2 , cols // 2
#     mask = np.ones((rows, cols), np.uint8)
#     mask[crow - radius:crow + radius, ccol - radius:ccol + radius] = 0
    
#     fshift_filtered = fshift * mask
#     f_ishift = np.fft.ifftshift(fshift_filtered)
#     img_back = np.fft.ifft2(f_ishift)
#     img_back = np.abs(img_back)
    
#     img_back = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX)
#     img_back = img_back.astype(np.uint8)
    
#     return Image.fromarray(img_back)


# Stationary Wavelet Transform
def stationary_wavelet_transform(image_path, wavelet='haar', level=2, resize_to=None):
    try:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        if resize_to is not None:
            image = cv2.resize(image, resize_to, interpolation=cv2.INTER_CUBIC)

        coeffs = pywt.swt2(image, wavelet, level=level)

        FS = []
        for i in range(len(coeffs)):
            fs = coeffs[i][1][0] + coeffs[i][1][1] + coeffs[i][1][2]
            fs = np.where((fs / np.max(fs) * 255.0) < 255.0, (fs / np.max(fs) * 255.0), 255.0)
            fs = np.where(fs < 0, 0, fs)
            fs = np.expand_dims(fs, axis=2)
            FS.append(fs)

        FS = np.concatenate(FS, axis=2)
        return FS

    except Exception as e:
        print(f"[LỖI ẢNH] {image_path} -> {e}")
        raise

def dct_high_freq_only(img_path, keep_ratio=0.2, resize_to=None):
    """
    Giữ lại thông tin cao tần trong miền DCT.
    :param img_path: đường dẫn ảnh đầu vào
    :param keep_ratio: tỉ lệ phần cao tần giữ lại (0-1)
    :param resize_to: resize ảnh về kích thước (W,H) nếu cần
    :return: ảnh khôi phục từ cao tần
    """
    # 1. Đọc ảnh grayscale
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Không đọc được ảnh từ {img_path}")
    
    if resize_to is not None:
        img = cv2.resize(img, resize_to, interpolation=cv2.INTER_CUBIC)

    img = img.astype(np.float32) / 255.0

    # 2. DCT
    dct_coeff = cv2.dct(img)

    # 3. Lọc bỏ thấp tần (đặt thành 0 vùng top-left)
    h, w = dct_coeff.shape
    h_cut = int(h * keep_ratio)
    w_cut = int(w * keep_ratio)

    dct_low_removed = dct_coeff.copy()
    dct_low_removed[:h_cut, :w_cut] = 0   # xoá vùng thấp tần

    # 4. IDCT để khôi phục ảnh cao tần
    img_high = cv2.idct(dct_low_removed)
    img_high = np.clip(img_high, 0, 1)
    img_high = (img_high * 255).astype(np.uint8)

    return img_high

def laplacian_high_freq(img_path, resize_to=None):
    """
    Lọc cao tần bằng Laplacian.
    :param img_path: đường dẫn ảnh
    :param resize_to: (W,H) nếu cần resize
    :return: ảnh laplacian (edges/noise) và ảnh khôi phục khi chỉ giữ cao tần
    """
    # 1. Đọc ảnh grayscale
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Không đọc được ảnh từ {img_path}")
    
    if resize_to is not None:
        img = cv2.resize(img, resize_to, interpolation=cv2.INTER_CUBIC)

    # 2. Áp dụng Laplacian (bộ lọc high-pass)
    lap = cv2.Laplacian(img, ddepth=cv2.CV_32F, ksize=3)

    # Chuẩn hóa về [0,255]
    lap_norm = cv2.convertScaleAbs(lap)

    # 3. Nếu muốn "khôi phục" lại ảnh chỉ chứa cao tần
    high_freq_only = cv2.addWeighted(img.astype(np.float32), 0, lap, 1, 0)
    high_freq_only = np.clip(high_freq_only, 0, 255).astype(np.uint8)

    return lap_norm, high_freq_only


def apply_srm_filters(img_path, resize_to=None):
    """
    Apply SRM (Steganalysis Rich Model) filters for artifact detection.
    
    Args:
        img_path: Path to the image
        resize_to: Optional tuple (width, height) to resize image
        
    Returns:
        SRM feature maps of shape (H, W, 3) - last 3 individual filter responses
    """
    # Define the last 3 SRM filters (most effective ones)
    srm_filters = [
        # SQUARE 5x5
        np.array([[-1, 2, -2, 2, -1], [2, -6, 8, -6, 2], [-2, 8, -12, 8, -2], 
                  [2, -6, 8, -6, 2], [-1, 2, -2, 2, -1]], dtype=np.float32),
        # D3,3 4x4
        np.array([[1, -3, 3, -1], [-3, 9, -9, 3], [3, -9, 9, -3], [-1, 3, -3, 1]], dtype=np.float32),
        # D4,4 5x5
        np.array([[1, -4, 6, -4, 1], [-4, 16, -24, 16, -4], [6, -24, 36, -24, 6], 
                  [-4, 16, -24, 16, -4], [1, -4, 6, -4, 1]], dtype=np.float32)
    ]
    
    # Read the image
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not read image: {img_path}")
    
    # Resize if needed
    if resize_to is not None:
        img = cv2.resize(img, resize_to, interpolation=cv2.INTER_CUBIC)
    
    # Apply all SRM filters and keep all individual responses
    responses = []
    for filt in srm_filters:
        # Pad filter to 5x5 if smaller
        h, w = filt.shape
        if h < 5 or w < 5:
            pad_h = 5 - h
            pad_w = 5 - w
            pad_top = pad_h // 2
            pad_bottom = pad_h - pad_top
            pad_left = pad_w // 2
            pad_right = pad_w - pad_left
            filt = np.pad(filt, ((pad_top, pad_bottom), (pad_left, pad_right)), 'constant')
        
        # Apply filter using convolution
        response = cv2.filter2D(img, -1, filt)
        
        # Normalize each response to 0-255 range
        if np.max(response) > 0:
            norm_resp = cv2.normalize(response, None, 0, 255, cv2.NORM_MINMAX)
        else:
            norm_resp = np.zeros_like(response)
        
        # Add channel dimension and append
        responses.append(np.expand_dims(norm_resp.astype(np.uint8), axis=2))
    
    # Concatenate all filter responses along channel dimension
    # This gives us 10 channels, one for each SRM filter
    srm_features = np.concatenate(responses, axis=2)
    
    # Return all 10 SRM filter responses as separate channels
    return srm_features


# Global HAFT model for processing grayscale images
# Initialize once to avoid repeated model creation
_haft_model = None
_haft_device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_haft_model():
    """Get or create the HAFT model for processing."""
    global _haft_model
    if _haft_model is None:
        # Create HAFT model with appropriate parameters
        # Using 1 input channel for grayscale, typical parameters
        _haft_model = HAFT(
            in_channels=1,  # Grayscale input
            num_haft_levels=3,  # 3 levels for good frequency analysis
            num_radial_bins=16,
            context_vector_dim=64
        ).to(_haft_device)
        _haft_model.eval()  # Set to evaluation mode
    return _haft_model

# Note: HAFT processing is now handled by the model's HAFT module
# as a learnable component rather than preprocessing

# Note: HAFT processing is now handled by the model's HAFT module
# as a learnable component rather than preprocessing
    

# Read FF+ dataset
BASE_URL = 'datasets/FaceForensic_ImageData/FF+'

def read_ff_dataset(deepfake_type):
    """ Read deepfake dataset by deepfake type
    Args:
        deepfake_type (str): {deepfakes, face2face, faceshifter, faceswap, neuraltexture}
    Return:
        train_real_url (List): train real dataset
        test_real_url (List): test real dataset
        train_fake_url (List): train fake dataset
        test_fake_url (List): test fake dataset
    """
    train_real_url = []
    test_real_url = []
    train_fake_url = []
    test_fake_url = []

    # List all video id
    vid_ids = os.listdir(BASE_URL)
    vid_ids = [vid_id for vid_id in vid_ids if '.' not in vid_id]

    # Number training video (Train : Test = 8 : 2)
    num_vid_train = int(0.8 * len(vid_ids))

    # Read dataset
    for i, vid_id in enumerate(vid_ids):
        # Define deepfake and original path base on folder structure
        deepfake_folder_path = f"{BASE_URL}/{vid_id}/{deepfake_type}"
        original_folder_path = f"{BASE_URL}/{vid_id}/original"
        
        if i < num_vid_train:
            for image_name in os.listdir(deepfake_folder_path):
                train_fake_url.append(os.path.join(deepfake_folder_path, image_name))
                train_real_url.append(os.path.join(original_folder_path, image_name))
        else:
            for image_name in os.listdir(deepfake_folder_path):
                test_fake_url.append(os.path.join(deepfake_folder_path, image_name))
                test_real_url.append(os.path.join(original_folder_path, image_name))

    return train_real_url, test_real_url, train_fake_url, test_fake_url
    

# Train test split
# train_real_url, test_real_url, train_fake_url, test_fake_url = train_test_split(list_real_url, list_fake_url_filter, test_size=0.2, random_state=42)

# Load FF+ dataset
train_real_url, test_real_url, train_fake_url, test_fake_url = read_ff_dataset(deepfake_type='deepfakes')

train_list = train_real_url + train_fake_url 
test_list = test_real_url + test_fake_url

print(f"Train real images: {len(train_real_url)}")
print(f"Test real images: {len(test_real_url)}")
print(f"Train fake images: {len(train_fake_url)}")
print(f"Test fake images: {len(test_fake_url)}")

from torch.utils.data import Dataset
from torchvision import transforms


transform = transforms.Compose([
    transforms.ToTensor(),  
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.Resize((256, 256))
])


real_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(),
    transforms.Resize((256,256))
])


class FaceDataset(Dataset):
    def __init__(self, list_img_url, transform = None, real_transform = None):
        self.list_img_url = list_img_url
        self.transform = transform
        self.real_transform = real_transform
        
    def __len__(self):
        return len(self.list_img_url)
        
    def __getitem__(self, idx):
        img_url = self.list_img_url[idx]
        img = Image.open(img_url)

        # Real image
        if '/original/' in img_url:
            mask = np.zeros((256,256), dtype = 'float32')
            mask = torch.from_numpy(mask).unsqueeze(dim = 0)

            label = 0
            rgb_img = self.real_transform(img)

        # Fake image
        else:
            mask_url = img_url.replace('FF+', 'FF+Mask')
            mask = Image.open(mask_url)
            mask = self.transform(mask)

            label = 1
            rgb_img = self.transform(img)

        # # Original frequency features (without HAFT preprocessing)
        # # FFT
        # fft_img = high_pass_filter_image(img_url)
        # fft_img = np.expand_dims(fft_img, axis=2)
        # SWT
        swt_img = stationary_wavelet_transform(img_url, resize_to=(160, 160))
        # DCT
        dct_img = dct_high_freq_only(img_url, keep_ratio=0.2, resize_to=(160, 160))
        dct_img = np.expand_dims(dct_img, axis=2)
        # Laplacian
        lap_edges, high_freq_img = laplacian_high_freq(img_url, resize_to=(160, 160))
        lap_edges = np.expand_dims(lap_edges, axis=2)
        high_freq_img = np.expand_dims(high_freq_img, axis=2)
        # SRM Filters - now returns only last 3 individual filter responses
        srm_img = apply_srm_filters(img_url, resize_to=(160, 160))
        
        # # Concatenate all frequency features (now 9 channels: FFT(1) + SWT(2) + DCT(1) + Laplacian(2) + SRM(3))
        # freq_img = np.concatenate((fft_img, swt_img, dct_img, lap_edges, high_freq_img, srm_img), axis=2).astype(np.float32)
        # Concatenate all frequency features (now 8 channels: SWT(2) + DCT(1) + Laplacian(2) + SRM(3))
        freq_img = np.concatenate((swt_img, dct_img, lap_edges, high_freq_img, srm_img), axis=2).astype(np.float32)
        freq_img = self.transform(freq_img)
        
        # Prepare grayscale image for HAFT processing within the model
        # Read and resize grayscale image to match frequency features size
        gray_img = cv2.imread(img_url, cv2.IMREAD_GRAYSCALE)
        gray_img = cv2.resize(gray_img, (160, 160), interpolation=cv2.INTER_CUBIC)  # Match other features
        gray_img = gray_img.astype(np.float32) / 255.0  # Normalize to [0, 1]
        gray_img = torch.from_numpy(gray_img).unsqueeze(0)  # Add channel dimension: (1, H, W)
        
        return rgb_img, freq_img, gray_img, mask, label

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)
        
        # Ultra-speed optimization: Fuse activations
        self.activation = nn.GELU()

    def forward(self, x):
        # Fused projection and activation
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = self.activation(x1) * x2
        x = self.project_out(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.scale = (dim // num_heads) ** -0.5  # Pre-compute scale factor

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        
        # Ultra-speed optimization: Pre-allocate buffers
        self.register_buffer('attention_cache', None)

    def forward(self, x):
        b, c, h, w = x.shape

        # Fused QKV computation
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)   
        
        # Optimized reshape using view instead of rearrange
        head_dim = c // self.num_heads
        q = q.view(b, self.num_heads, head_dim, h * w)
        k = k.view(b, self.num_heads, head_dim, h * w)
        v = v.view(b, self.num_heads, head_dim, h * w)

        # Fast normalization
        q = F.normalize(q, dim=-2, p=2)
        k = F.normalize(k, dim=-2, p=2)

        # Optimized attention computation with temperature scaling
        attn = torch.matmul(q.transpose(-2, -1), k) * (self.temperature * self.scale)
        attn = F.softmax(attn, dim=-1)

        # Efficient matrix multiplication
        out = torch.matmul(attn, v.transpose(-2, -1)).transpose(-2, -1)
        
        # Fast reshape back
        out = out.contiguous().view(b, c, h, w)
        out = self.project_out(out)
        return out


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x

class REBNCONV(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, dirate=1):
        super(REBNCONV, self).__init__()

        # Depthwise convolution
        self.depthwise = nn.Conv2d(
            in_ch, in_ch, kernel_size=3, padding=1*dirate, dilation=1*dirate, groups=in_ch)
        # Pointwise convolution
        self.pointwise = nn.Conv2d(
            in_ch, out_ch, kernel_size=1)

        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

def _upsample_like(src,tar):

    src = F.interpolate(src,size=tar.shape[2:],mode='bilinear')

    return src

class RSU7(nn.Module):#Encoder 1

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU7,self).__init__()

        self.rebnconvin = REBNCONV(in_ch,out_ch,dirate=1)

        self.rebnconv1 = REBNCONV(out_ch,mid_ch,dirate=1)
        self.pool1 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool2 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool3 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv4 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool4 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv5 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool5 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv6 = REBNCONV(mid_ch,mid_ch,dirate=1)

        self.rebnconv7 = REBNCONV(mid_ch,mid_ch,dirate=2)

        self.rebnconv6d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv5d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv4d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv3d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch*2,out_ch,dirate=1)

    def forward(self,x):

        hx = x
        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)

        hx4 = self.rebnconv4(hx)
        hx = self.pool4(hx4)

        hx5 = self.rebnconv5(hx)
        hx = self.pool5(hx5)

        hx6 = self.rebnconv6(hx)

        hx7 = self.rebnconv7(hx6)

        hx6d =  self.rebnconv6d(torch.cat((hx7,hx6),1))
        hx6dup = _upsample_like(hx6d,hx5)

        hx5d =  self.rebnconv5d(torch.cat((hx6dup,hx5),1))
        hx5dup = _upsample_like(hx5d,hx4)

        hx4d = self.rebnconv4d(torch.cat((hx5dup,hx4),1))
        hx4dup = _upsample_like(hx4d,hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4dup,hx3),1))
        hx3dup = _upsample_like(hx3d,hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup,hx2),1))
        hx2dup = _upsample_like(hx2d,hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup,hx1),1))

        return hx1d + hxin

class RSU6(nn.Module): #Encoder 2
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU6,self).__init__()

        self.rebnconvin = REBNCONV(in_ch,out_ch,dirate=1)

        self.rebnconv1 = REBNCONV(out_ch,mid_ch,dirate=1)
        self.pool1 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool2 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool3 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv4 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool4 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv5 = REBNCONV(mid_ch,mid_ch,dirate=1)

        self.rebnconv6 = REBNCONV(mid_ch,mid_ch,dirate=2)

        self.rebnconv5d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv4d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv3d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch*2,out_ch,dirate=1)

    def forward(self,x):

        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)

        hx4 = self.rebnconv4(hx)
        hx = self.pool4(hx4)

        hx5 = self.rebnconv5(hx)

        hx6 = self.rebnconv6(hx5)


        hx5d =  self.rebnconv5d(torch.cat((hx6,hx5),1))
        hx5dup = _upsample_like(hx5d,hx4)

        hx4d = self.rebnconv4d(torch.cat((hx5dup,hx4),1))
        hx4dup = _upsample_like(hx4d,hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4dup,hx3),1))
        hx3dup = _upsample_like(hx3d,hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup,hx2),1))
        hx2dup = _upsample_like(hx2d,hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup,hx1),1))

        return hx1d + hxin

class RSU5(nn.Module):#Encoder 3

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU5,self).__init__()

        self.rebnconvin = REBNCONV(in_ch,out_ch,dirate=1)

        self.rebnconv1 = REBNCONV(out_ch,mid_ch,dirate=1)
        self.pool1 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool2 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool3 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv4 = REBNCONV(mid_ch,mid_ch,dirate=1)

        self.rebnconv5 = REBNCONV(mid_ch,mid_ch,dirate=2)

        self.rebnconv4d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv3d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch*2,out_ch,dirate=1)

    def forward(self,x):

        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)

        hx4 = self.rebnconv4(hx)

        hx5 = self.rebnconv5(hx4)

        hx4d = self.rebnconv4d(torch.cat((hx5,hx4),1))
        hx4dup = _upsample_like(hx4d,hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4dup,hx3),1))
        hx3dup = _upsample_like(hx3d,hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup,hx2),1))
        hx2dup = _upsample_like(hx2d,hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup,hx1),1))

        return hx1d + hxin


class RSU4(nn.Module):#Encoder 4

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU4,self).__init__()

        self.rebnconvin = REBNCONV(in_ch,out_ch,dirate=1)

        self.rebnconv1 = REBNCONV(out_ch,mid_ch,dirate=1)
        self.pool1 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool2 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_ch,mid_ch,dirate=1)

        self.rebnconv4 = REBNCONV(mid_ch,mid_ch,dirate=2)

        self.rebnconv3d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch*2,out_ch,dirate=1)

    def forward(self,x):

        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)

        hx4 = self.rebnconv4(hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4,hx3),1))
        hx3dup = _upsample_like(hx3d,hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup,hx2),1))
        hx2dup = _upsample_like(hx2d,hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup,hx1),1))

        return hx1d + hxin

class RSU4F(nn.Module):#UNet04FRES(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU4F,self).__init__()

        self.rebnconvin = REBNCONV(in_ch,out_ch,dirate=1)

        self.rebnconv1 = REBNCONV(out_ch,mid_ch,dirate=1)
        self.rebnconv2 = REBNCONV(mid_ch,mid_ch,dirate=2)
        self.rebnconv3 = REBNCONV(mid_ch,mid_ch,dirate=4)

        self.rebnconv4 = REBNCONV(mid_ch,mid_ch,dirate=8)

        self.rebnconv3d = REBNCONV(mid_ch*2,mid_ch,dirate=4)
        self.rebnconv2d = REBNCONV(mid_ch*2,mid_ch,dirate=2)
        self.rebnconv1d = REBNCONV(mid_ch*2,out_ch,dirate=1)

    def forward(self,x):

        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx2 = self.rebnconv2(hx1)
        hx3 = self.rebnconv3(hx2)

        hx4 = self.rebnconv4(hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4,hx3),1))
        hx2d = self.rebnconv2d(torch.cat((hx3d,hx2),1))
        hx1d = self.rebnconv1d(torch.cat((hx2d,hx1),1))

        return hx1d + hxin

import torch
import torch.nn as nn
import torch.nn.functional as F

# Channel Attention Module
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # Output shape: (B, C, 1, 1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.mlp = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


# Spatial Attention Module
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = kernel_size // 2  # Maintain size

        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Channel-wise average and max pooling
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        concat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(concat)
        return self.sigmoid(out)


# CBAM Block
class CBAM(nn.Module):
    def __init__(self, channels, reduction_ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size)
        
        # Ultra-speed optimization: Enable fusion
        self.fuse_multiply = True

    def forward(self, x):
        # Fused attention computation
        if self.fuse_multiply:
            # Single-pass attention computation
            channel_att = self.channel_attention(x)
            out = x * channel_att
            spatial_att = self.spatial_attention(out)
            out = out * spatial_att
        else:
            out = x * self.channel_attention(x)
            out = out * self.spatial_attention(out)
        return out

# class ConvBlock(nn.Module):
#     def __init__(self, in_channel, out_channel):
#         super(ConvBlock, self).__init__()
#         self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size = 3, stride = 1, padding = 1)
#         self.bn1 = nn.BatchNorm2d(out_channel)

#         self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size = 3, stride = 1, padding = 1)
#         self.bn2 = nn.BatchNorm2d(out_channel)

#         self.skip = nn.Sequential(
#             nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, padding=0),
#             nn.BatchNorm2d(out_channel)
#         )

#         self.relu = nn.ReLU()
        
#     def forward(self, x):
#         identity = self.skip(x)

#         x = self.relu(self.bn1(self.conv1(x)))
#         x = self.bn2(self.conv2(x))

#         x += identity
#         x = self.relu(x)
        
#         return x 

class DeformableConv2d(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 dilation=1,
                 bias=False):
        super(DeformableConv2d, self).__init__()

        assert type(kernel_size) == tuple or type(kernel_size) == int

        kernel_size = kernel_size if type(kernel_size) == tuple else (kernel_size, kernel_size)
        self.stride = stride if type(stride) == tuple else (stride, stride)
        self.padding = padding
        self.dilation = dilation

        self.offset_conv = nn.Conv2d(in_channels,
                                     2 * kernel_size[0] * kernel_size[1],
                                     kernel_size=kernel_size,
                                     stride=stride,
                                     padding=self.padding,
                                     dilation=self.dilation,
                                     bias=True)

        nn.init.constant_(self.offset_conv.weight, 0.)
        nn.init.constant_(self.offset_conv.bias, 0.)

        self.modulator_conv = nn.Conv2d(in_channels,
                                        1 * kernel_size[0] * kernel_size[1],
                                        kernel_size=kernel_size,
                                        stride=stride,
                                        padding=self.padding,
                                        dilation=self.dilation,
                                        bias=True)

        nn.init.constant_(self.modulator_conv.weight, 0.)
        nn.init.constant_(self.modulator_conv.bias, 0.)

        self.regular_conv = nn.Conv2d(in_channels=in_channels,
                                      out_channels=out_channels,
                                      kernel_size=kernel_size,
                                      stride=stride,
                                      padding=self.padding,
                                      dilation=self.dilation,
                                      bias=bias)

    def forward(self, x):
        # h, w = x.shape[2:]
        # max_offset = max(h, w)/4.

        offset = self.offset_conv(x)  # .clamp(-max_offset, max_offset)
        modulator = 2. * torch.sigmoid(self.modulator_conv(x))
        # op = (n - (k * d - 1) + 2p / s)
        x = torchvision.ops.deform_conv2d(input=x,
                                          offset=offset,
                                          weight=self.regular_conv.weight,
                                          bias=self.regular_conv.bias,
                                          padding=self.padding,
                                          mask=modulator,
                                          stride=self.stride,
                                          dilation=self.dilation)
        return x
    

class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ConvBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
            nn.Conv2d(out_channel, out_channel, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
        )
    def forward(self, x):
        return self.conv_block(x)
        
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.fc1 = nn.Linear(channels, channels // reduction, bias=False)
        self.fc2 = nn.Linear(channels // reduction, channels, bias=False)

    def forward(self, x):
        b, c, h, w = x.size()
        # Squeeze
        y = F.adaptive_avg_pool2d(x, 1).view(b, c)
        # Excitation
        y = F.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y))
        # Scale
        y = y.view(b, c, 1, 1)
        return x * y.expand_as(x)

# class EncodeELA(nn.Module):
#     def __init__(self, in_ch=3, base_ch=64, use_se=True):
#         super(EncodeELA, self).__init__()

#         # Multi-scale stem
#         self.stem3 = nn.Conv2d(in_ch, base_ch//2, 3, padding=1)
#         self.stem5 = nn.Conv2d(in_ch, base_ch//2, 5, stride=2, padding=2)
#         self.stem7 = nn.Conv2d(in_ch, base_ch//2, 7, stride=2, padding=3)
#         self.stem_out = nn.Conv2d(base_ch*3//2, base_ch, 1)

#         # Encoder blocks
#         self.conv1 = ConvBlock(base_ch, base_ch)
#         self.cbam1 = CBAM(base_ch, 16)
#         self.down1 = nn.Conv2d(base_ch, base_ch*2, 3, stride=2, padding=1)

#         self.conv2 = ConvBlock(base_ch*2, base_ch*2)
#         self.cbam2 = CBAM(base_ch*2, 16)
#         self.down2 = nn.Conv2d(base_ch*2, base_ch*4, 3, stride=2, padding=1)

#         self.conv3 = ConvBlock(base_ch*4, base_ch*4)
#         self.cbam3 = CBAM(base_ch*4, 16)
#         self.se3 = SEBlock(base_ch*4) if use_se else nn.Identity()
#         self.down3 = nn.Conv2d(base_ch*4, base_ch*8, 3, stride=2, padding=1)

#         self.conv4 = ConvBlock(base_ch*8, base_ch*8)
#         self.cbam4 = CBAM(base_ch*8, 16)
#         self.down4 = nn.Conv2d(base_ch*8, base_ch*8, 3, stride=2, padding=1)

#         self.conv5 = ConvBlock(base_ch*8, base_ch*8)
#         self.cbam5 = CBAM(base_ch*8, 16)
#         self.se5 = SEBlock(base_ch*8) if use_se else nn.Identity()

class EncodeELA(nn.Module):
    def __init__(self):
        super(EncodeELA, self).__init__()
        
        # HAFT parameters from config
        num_haft_levels = 3
        num_radial_bins = 16  
        context_vector_dim = 64
        
        # Add HAFT as a learnable module for processing grayscale input
        self.haft_processor = HAFT(
            in_channels=1,  # Grayscale input
            num_haft_levels=num_haft_levels,
            num_radial_bins=num_radial_bins,
            context_vector_dim=context_vector_dim
        )
        
        # Ultra-speed optimization: Enable memory format optimization
        self.enable_channels_last = True
        
        # # Updated frequency features: FFT(1) + SWT(2) + DCT(1) + Laplacian(2) + SRM(3) = 9 channels
        # # HAFT output: 1 channel (enhanced grayscale features)
        # # Total: 9 + 1 = 10 channels
        # self.conv1 = ConvBlock(10, 64)  # 9 frequency + 1 HAFT = 10 channels
        # Updated frequency features: SWT(2) + DCT(1) + Laplacian(2) + SRM(3) = 8 channels
        # HAFT output: 1 channel (enhanced grayscale features)
        # Total: 8 + 1 = 9 channels
        self.conv1 = ConvBlock(9, 64)  # 8 frequency + 1 HAFT = 9 channels
        self.cbam1 = CBAM(64, 16)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = ConvBlock(64, 128)
        self.cbam2 = CBAM(128, 16)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv3 = DeformableConv2d(128, 256)
        self.cbam3 = CBAM(256, 16)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv4 = DeformableConv2d(256, 512)
        self.cbam4 = CBAM(512, 16)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv5 = DeformableConv2d(512, 512)
        self.cbam5 = CBAM(512, 16)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Ultra-speed optimization: Pre-compute interpolation modes
        self.interpolation_mode = 'bilinear'
        self.align_corners = False

    def forward(self, freq_img, gray_img):
        """
        Ultra-optimized forward pass with learnable HAFT processing.
        Maximum speed without memory constraints.
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
        
        # Concatenate 9 frequency features with HAFT output: (B, 10, H, W)
        x = torch.cat([freq_img, haft_features], dim=1)
        
        # Continue with existing processing pipeline
        conv1 = self.conv1(x)  # Now processes 10 channels
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


class AttentionGateV2(nn.Module):
    def __init__(self,channel = 3):
        super(AttentionGateV2, self).__init__()
        self.conv1 = nn.Conv2d(channel, channel, kernel_size = 1, stride = 1)
        self.conv2 = nn.ConvTranspose2d(channel, channel, kernel_size = 2, stride = 2)
        self.conv_block = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(channel, channel, kernel_size = 3, stride = 1, padding = 1),
            nn.Sigmoid()
        )
    def forward(self, x, ela):
        x1 = self.conv1(x)
        ela = self.conv2(ela)
        attn_gate = self.conv_block(x1 + ela)
        return x * attn_gate


class CMF_Block(nn.Module):
    def __init__(self, in_channel, hidden_channel, out_channel):
        super(CMF_Block, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channel, hidden_channel, kernel_size=1, stride=1, padding=0
        )
        self.conv2 = nn.Conv2d(
            in_channel, hidden_channel, kernel_size=1, stride=1, padding=0
        )
        self.conv3 = nn.Conv2d(
            in_channel, hidden_channel, kernel_size=1, stride=1, padding=0
        )

        self.scale = hidden_channel ** -0.5

        self.conv4 = nn.Sequential(
            nn.Conv2d(
                hidden_channel, out_channel, kernel_size=1, stride=1, padding=0
            ),
            nn.BatchNorm2d(out_channel),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, rgb, freq):
        _, _, h, w = rgb.size()

        q = self.conv1(rgb)
        k = self.conv2(freq)
        v = self.conv3(freq)

        q = q.view(q.size(0), q.size(1), q.size(2) * q.size(3)).transpose(
            -2, -1
        )
        k = k.view(k.size(0), k.size(1), k.size(2) * k.size(3))

        attn = torch.matmul(q, k) * self.scale
        m = attn.softmax(dim=-1)

        v = v.view(v.size(0), v.size(1), v.size(2) * v.size(3)).transpose(
            -2, -1
        )
        z = torch.matmul(m, v)
        z = z.view(z.size(0), h, w, -1)
        z = z.permute(0, 3, 1, 2).contiguous()

        output = rgb + self.conv4(z)

        return output


class Classifier(nn.Module):
    def __init__(self, num_classes=2):
        super(Classifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(),
            nn.Linear(in_features=512, out_features=256),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=256, out_features = 64),
            nn.ReLU(),
            nn.Dropout(p = 0.5),
            nn.Linear(in_features=64, out_features=num_classes)
        )
    def forward(self, x):
        x = self.classifier(x)
        return x

class U2NET(nn.Module):

    def __init__(self,in_ch=3,out_ch=1):
        super(U2NET,self).__init__()

        self.stage1 = RSU7(in_ch,32,64)
        self.pool12 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage2 = RSU6(64,32,128)
        self.pool23 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage3 = RSU5(128,64,256)
        self.pool34 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage4 = RSU4(256,128,512)
        self.pool45 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage5 = RSU4F(512,256,512)
        self.pool56 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage6 = RSU4F(512,256,512)

        self.encode_ela = EncodeELA()
        self.classifier = Classifier()
        self.gate5 = AttentionGateV2(512)
        self.gate4 = AttentionGateV2(512)
        self.gate3 = AttentionGateV2(256)
        self.gate2 = AttentionGateV2(128)
        self.gate1 = AttentionGateV2(64)

        # self.cmf1 = CMF_Block(64, 16, 64)
        # self.cmf2 = CMF_Block(128, 32, 128)
        # self.cmf3 = CMF_Block(256, 64, 256)
        # self.cmf4 = CMF_Block(512, 128, 512)
        # self.cmf5 = CMF_Block(512, 128, 512)

        # decoder
        self.stage5d = RSU4F(1024, 256, 512)
        self.stage4d = RSU4(1024, 128, 256)
        self.stage3d = RSU5(512, 64, 128)
        self.stage2d = RSU6(256, 32, 64)
        self.stage1d = RSU7(128, 16, 64)

        self.side1 = nn.Conv2d(64,out_ch,3,padding=1)
        self.side2 = nn.Conv2d(64,out_ch,3,padding=1)
        self.side3 = nn.Conv2d(128,out_ch,3,padding=1)
        self.side4 = nn.Conv2d(256,out_ch,3,padding=1)
        self.side5 = nn.Conv2d(512,out_ch,3,padding=1)
        self.side6 = nn.Conv2d(512,out_ch,3,padding=1)

        self.outconv = nn.Conv2d(6*out_ch,out_ch,1)

    def forward(self, x, ela, gray_img):
        """
        Forward pass with separate frequency features and grayscale input for HAFT.
        
        Args:
            x: RGB image features (B, 3, H, W)
            ela: Frequency features tensor (B, 9, H, W) - FFT(1) + SWT(2) + DCT(1) + Laplacian(2) + SRM(3) = 9 channels
            gray_img: Grayscale image tensor (B, 1, H, W) for HAFT processing
        """

        # Encode ELA image with HAFT integration
        ela1, ela2, ela3, ela4, ela5 = self.encode_ela(ela, gray_img)

        hx = x

        #stage 1
        hx1 = self.stage1(hx)
        hx = self.pool12(hx1)

        #stage 2
        hx2 = self.stage2(hx)
        hx = self.pool23(hx2)

        #stage 3
        hx3 = self.stage3(hx)
        hx = self.pool34(hx3)

        #stage 4
        hx4 = self.stage4(hx)
        hx = self.pool45(hx4)

        #stage 5
        hx5 = self.stage5(hx)
        hx = self.pool56(hx5)

        #stage 6
        hx6 = self.stage6(hx)
        hx6up = _upsample_like(hx6,hx5)

        attn1 = self.gate1(hx1, ela1)
        attn2 = self.gate2(hx2, ela2)
        attn3 = self.gate3(hx3, ela3)
        attn4 = self.gate4(hx4, ela4)
        attn5 = self.gate5(hx5, ela5)
        
        pred_label = self.classifier(ela5)
        
        #-------------------- decoder --------------------
        
        hx5d = self.stage5d(torch.cat((hx6up,attn5),1))
        hx5dup = _upsample_like(hx5d,hx4)
        
        hx4d = self.stage4d(torch.cat((hx5dup,attn4),1))
        hx4dup = _upsample_like(hx4d,hx3)

        hx3d = self.stage3d(torch.cat((hx4dup,attn3),1))
        hx3dup = _upsample_like(hx3d,hx2)

        hx2d = self.stage2d(torch.cat((hx3dup,attn2),1))
        hx2dup = _upsample_like(hx2d,hx1)

        hx1d = self.stage1d(torch.cat((hx2dup,attn1),1))


        #side output
        d1 = self.side1(hx1d)

        d2 = self.side2(hx2d)
        d2 = _upsample_like(d2,d1)

        d3 = self.side3(hx3d)
        d3 = _upsample_like(d3,d1)

        d4 = self.side4(hx4d)
        d4 = _upsample_like(d4,d1)

        d5 = self.side5(hx5d)
        d5 = _upsample_like(d5,d1)

        d6 = self.side6(hx6)
        d6 = _upsample_like(d6,d1)

        d0 = self.outconv(torch.cat((d1,d2,d3,d4,d5,d6),1))

        return F.sigmoid(d0), pred_label


device = 'cuda' if torch.cuda.is_available() else 'cpu'


def visualize_results(images: torch.Tensor, recon_images: torch.Tensor):
    
    images = images.detach().cpu().numpy()
    recon_images = recon_images.detach().cpu().numpy()
    
    fig, axes = plt.subplots(4, 2, figsize=(12, 24))
    axes = axes.flatten()
    
    for i in range(4):
        img = np.transpose(images[i], (1, 2, 0))
        recon_img = np.transpose(recon_images[i], (1, 2, 0))
        
        axes[2*i].imshow(img, cmap = 'gray')
        axes[2*i].set_title(f"Target mask")
        axes[2*i].axis("off")
        
        axes[2*i + 1].imshow(recon_img, cmap = 'gray')
        axes[2*i + 1].set_title(f"Mask")
        axes[2*i + 1].axis("off")

    
    plt.tight_layout()
    plt.show()


from torchmetrics.classification import BinaryAUROC
# Note: BinaryEER might not be available in all torchmetrics versions
# We'll use a custom EER implementation or skip it for now

def train_step(model, train_dataloader, loss_seg, loss_clf, optimizer, scheduler, 
               psnr, ssim, auc, f1, precision, recall, specificity, accuracy, eer, epoch):
    model.train()
    
    # Ultra-speed optimization: Enable mixed precision
    scaler = torch.cuda.amp.GradScaler()
    
    train_loss = 0
    
    for batch, (img, freq_img, gray_img, mask, label) in enumerate(train_dataloader):
        # Ultra-fast data transfer with non_blocking and memory format optimization
        img = img.to(device, non_blocking=True, memory_format=torch.channels_last)
        freq_img = freq_img.to(device, non_blocking=True, memory_format=torch.channels_last)
        gray_img = gray_img.to(device, non_blocking=True, memory_format=torch.channels_last)
        mask = mask.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True)
        
        # Mixed precision forward pass
        with torch.cuda.amp.autocast():
            pred_mask, pred_label = model(img, freq_img, gray_img)
            
            loss_segment = loss_seg(pred_mask, mask)
            loss_classifier = loss_clf(pred_label, label)
            loss = 0.7 * loss_segment + 0.3 * loss_classifier
        
        train_loss += loss.item()

        # Optimized softmax computation for classification metrics
        with torch.no_grad():
            pred_label_softmax = F.softmax(pred_label.detach(), dim=1)[:, 1]
            pred_label_binary = torch.argmax(pred_label.detach(), dim=1)
            
            # Update all metrics
            # Segmentation metrics
            psnr.update(pred_mask.detach(), mask)
            ssim.update(pred_mask.detach(), mask)
            
            # Classification metrics
            auc.update(pred_label_softmax, label)
            f1.update(pred_label_binary, label)
            precision.update(pred_label_binary, label)
            recall.update(pred_label_binary, label)
            specificity.update(pred_label_binary, label)
            accuracy.update(pred_label_binary, label)
            eer.update(pred_label_softmax, label)
        
        # Optimized backward pass with gradient scaling
        optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # OneCycleLR scheduler needs to be called every batch
        scheduler.step()
    
    # Compute all metrics
    train_loss /= len(train_dataloader)
    train_psnr = psnr.compute()
    train_ssim = ssim.compute()
    train_auc = auc.compute()
    train_f1 = f1.compute()
    train_precision = precision.compute()
    train_recall = recall.compute()
    train_specificity = specificity.compute()
    train_accuracy = accuracy.compute()
    train_eer = eer.compute()

    # Reset all metrics
    psnr.reset()
    ssim.reset()
    auc.reset()
    f1.reset()
    precision.reset()
    recall.reset()
    specificity.reset()
    accuracy.reset()
    eer.reset()
    
    return (train_loss, train_psnr, train_ssim, train_auc, train_f1, 
            train_precision, train_recall, train_specificity, train_accuracy, train_eer)


def test_step(model, test_dataloader, loss_seg, loss_clf, 
              psnr, ssim, auc, f1, precision, recall, specificity, accuracy, eer):
    model.eval()
    
    test_loss = 0
    with torch.no_grad():
        for batch, (img, freq_img, gray_img, mask, label) in enumerate(test_dataloader):
            img, freq_img, gray_img, mask, label = img.to(device), freq_img.to(device), gray_img.to(device), mask.to(device), label.to(device)
            
            pred_mask, pred_label = model(img, freq_img, gray_img)

            loss_segment = loss_seg(pred_mask, mask)
            loss_classifier = loss_clf(pred_label, label)
            loss = 0.7 * loss_segment + 0.3 * loss_classifier
            
            test_loss += loss.item()
            
            # Get predictions
            pred_label_softmax = F.softmax(pred_label, dim=1)[:, 1]
            pred_label_binary = torch.argmax(pred_label, dim=1)

            # Update all metrics
            # Segmentation metrics
            psnr.update(pred_mask, mask)
            ssim.update(pred_mask.detach(), mask)
            
            # Classification metrics
            auc.update(pred_label_softmax, label)
            f1.update(pred_label_binary, label)
            precision.update(pred_label_binary, label)
            recall.update(pred_label_binary, label)
            specificity.update(pred_label_binary, label)
            accuracy.update(pred_label_binary, label)
            eer.update(pred_label_softmax, label)
            
        # Compute all metrics
        test_loss = test_loss / len(test_dataloader)
        test_psnr = psnr.compute()
        test_ssim = ssim.compute()
        test_auc = auc.compute()
        test_f1 = f1.compute()
        test_precision = precision.compute()
        test_recall = recall.compute()
        test_specificity = specificity.compute()
        test_accuracy = accuracy.compute()
        test_eer = eer.compute()

    # Reset all metrics
    psnr.reset()
    ssim.reset()
    auc.reset()
    f1.reset()
    precision.reset()
    recall.reset()
    specificity.reset()
    accuracy.reset()
    eer.reset()
    
    return (test_loss, test_psnr, test_ssim, test_auc, test_f1, 
            test_precision, test_recall, test_specificity, test_accuracy, test_eer)


from tqdm.auto import tqdm

def load_checkpoint(model, optimizer, scheduler, checkpoint_path, device='cuda'):
    """
    Load checkpoint and restore model, optimizer, and scheduler states.
    
    Args:
        model: PyTorch model
        optimizer: PyTorch optimizer  
        scheduler: PyTorch scheduler
        checkpoint_path: Path to checkpoint file
        device: Device to load the model on
        
    Returns:
        Dictionary containing checkpoint information
    """
    print(f"📂 Loading checkpoint from: {checkpoint_path}")
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Load model state
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ Model state loaded successfully")
        
        # Load optimizer state  
        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print(f"✅ Optimizer state loaded successfully")
            
        # Load scheduler state
        if scheduler and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            print(f"✅ Scheduler state loaded successfully")
        
        # Print checkpoint info
        epoch = checkpoint.get('epoch', 'Unknown')
        test_loss = checkpoint.get('test_loss', 'Unknown')
        print(f"📊 Checkpoint info:")
        print(f"   Epoch: {epoch}")
        print(f"   Test Loss: {test_loss}")
        
        # Print available metrics
        metric_keys = [k for k in checkpoint.keys() if k.startswith('test_')]
        if metric_keys:
            print(f"📈 Available metrics: {', '.join(metric_keys)}")
            
        return checkpoint
        
    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")
        return None

def train(model, train_dataloader, test_dataloader, optimizer, scheduler, loss_seg, loss_clf, 
          psnr, ssim, auc, f1, precision, recall, specificity, accuracy, eer, epochs):
    
    # ===== CREATE CHECKPOINT DIRECTORY =====
    import os
    os.makedirs('dao_ckpt_srm_no2pass', exist_ok=True)
    print(f"📁 Checkpoint directory 'dao_ckpt_srm_no2pass' created/verified")
    
    # ===== INITIALIZE BEST METRICS FOR CHECKPOINT SAVING =====
    best_metrics = {
        'loss': float('inf'),
        'psnr': 0.0,
        'ssim': 0.0,
        'auc': 0.0,
        'f1': 0.0,
        'precision': 0.0,
        'recall': 0.0,
        'specificity': 0.0,
        'accuracy': 0.0,
        'eer': float('inf')  # Lower is better for EER
    }
    
    for epoch in tqdm(range(epochs)):
        # Train model
        (train_loss, train_psnr, train_ssim, train_auc, train_f1, 
         train_precision, train_recall, train_specificity, train_accuracy, train_eer) = train_step(
            model, train_dataloader, loss_seg, loss_clf, optimizer, scheduler, 
            psnr, ssim, auc, f1, precision, recall, specificity, accuracy, eer, epoch)

        # Test model
        (test_loss, test_psnr, test_ssim, test_auc, test_f1, 
         test_precision, test_recall, test_specificity, test_accuracy, test_eer) = test_step(
            model, test_dataloader, loss_seg, loss_clf, 
            psnr, ssim, auc, f1, precision, recall, specificity, accuracy, eer)

        # ===== SAVE BEST CHECKPOINTS FOR EACH METRIC =====
        checkpoint_base = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),  # ✅ Thêm scheduler state
            'train_loss': train_loss,
            'test_loss': test_loss,
            # All test metrics
            'test_psnr': test_psnr,
            'test_ssim': test_ssim,
            'test_auc': test_auc,
            'test_f1': test_f1,
            'test_precision': test_precision,
            'test_recall': test_recall,
            'test_specificity': test_specificity,
            'test_accuracy': test_accuracy,
            'test_eer': test_eer
        }
        
        # Check and save best checkpoints with notifications
        if test_loss < best_metrics['loss']:
            best_metrics['loss'] = test_loss
            torch.save(checkpoint_base, 'dao_ckpt_srm_no2pass/best_loss_dcn_dao.pt')
            print(f"✅ New best Loss: {test_loss:.6f} - Checkpoint saved!")
            
        if test_psnr > best_metrics['psnr']:
            best_metrics['psnr'] = test_psnr
            torch.save(checkpoint_base, 'dao_ckpt_srm_no2pass/best_psnr_dcn_dao.pt')
            print(f"✅ New best PSNR: {test_psnr:.4f} - Checkpoint saved!")
            
        if test_ssim > best_metrics['ssim']:
            best_metrics['ssim'] = test_ssim
            torch.save(checkpoint_base, 'dao_ckpt_srm_no2pass/best_ssim_dcn_dao.pt')
            print(f"✅ New best SSIM: {test_ssim:.4f} - Checkpoint saved!")
            
        if test_auc > best_metrics['auc']:
            best_metrics['auc'] = test_auc
            torch.save(checkpoint_base, 'dao_ckpt_srm_no2pass/best_auc_dcn_dao.pt')
            print(f"✅ New best AUC: {test_auc:.4f} - Checkpoint saved!")
            
        if test_f1 > best_metrics['f1']:
            best_metrics['f1'] = test_f1
            torch.save(checkpoint_base, 'dao_ckpt_srm_no2pass/best_f1_dcn_dao.pt')
            print(f"✅ New best F1: {test_f1:.4f} - Checkpoint saved!")
            
        if test_precision > best_metrics['precision']:
            best_metrics['precision'] = test_precision
            torch.save(checkpoint_base, 'dao_ckpt_srm_no2pass/best_precision_dcn_dao.pt')
            print(f"✅ New best Precision: {test_precision:.4f} - Checkpoint saved!")
            
        if test_recall > best_metrics['recall']:
            best_metrics['recall'] = test_recall
            torch.save(checkpoint_base, 'dao_ckpt_srm_no2pass/best_recall_dcn_dao.pt')
            print(f"✅ New best Recall: {test_recall:.4f} - Checkpoint saved!")
            
        if test_specificity > best_metrics['specificity']:
            best_metrics['specificity'] = test_specificity
            torch.save(checkpoint_base, 'dao_ckpt_srm_no2pass/best_specificity_dcn_dao.pt')
            print(f"✅ New best Specificity: {test_specificity:.4f} - Checkpoint saved!")
            
        if test_accuracy > best_metrics['accuracy']:
            best_metrics['accuracy'] = test_accuracy
            torch.save(checkpoint_base, 'dao_ckpt_srm_no2pass/best_accuracy_dcn_dao.pt')
            print(f"✅ New best Accuracy: {test_accuracy:.4f} - Checkpoint saved!")
            
        if test_eer < best_metrics['eer']:  # Lower is better for EER
            best_metrics['eer'] = test_eer
            torch.save(checkpoint_base, 'dao_ckpt_srm_no2pass/best_eer_dcn_dao.pt')
            print(f"✅ New best EER: {test_eer:.4f} - Checkpoint saved!")

        # ===== COMPREHENSIVE LOGGING =====
        print(f"\n{'='*80}")
        print(f"EPOCH {epoch+1}/{epochs}")
        print(f"{'='*80}")
        print(f"📊 TRAINING METRICS:")
        print(f"   Loss: {train_loss:.6f} | PSNR: {train_psnr:.4f} | SSIM: {train_ssim:.4f}")
        print(f"   AUC: {train_auc:.4f} | F1: {train_f1:.4f} | Precision: {train_precision:.4f}")
        print(f"   Recall: {train_recall:.4f} | Specificity: {train_specificity:.4f}")
        print(f"   Accuracy: {train_accuracy:.4f} | EER: {train_eer:.4f}")
        
        print(f"\n🎯 VALIDATION METRICS:")
        print(f"   Loss: {test_loss:.6f} | PSNR: {test_psnr:.4f} | SSIM: {test_ssim:.4f}")
        print(f"   AUC: {test_auc:.4f} | F1: {test_f1:.4f} | Precision: {test_precision:.4f}")
        print(f"   Recall: {test_recall:.4f} | Specificity: {test_specificity:.4f}")
        print(f"   Accuracy: {test_accuracy:.4f} | EER: {test_eer:.4f}")
        
        print(f"\n🏆 BEST METRICS SO FAR:")
        print(f"   Loss: {best_metrics['loss']:.6f} | PSNR: {best_metrics['psnr']:.4f} | SSIM: {best_metrics['ssim']:.4f}")
        print(f"   AUC: {best_metrics['auc']:.4f} | F1: {best_metrics['f1']:.4f} | Precision: {best_metrics['precision']:.4f}")
        print(f"   Recall: {best_metrics['recall']:.4f} | Specificity: {best_metrics['specificity']:.4f}")
        print(f"   Accuracy: {best_metrics['accuracy']:.4f} | EER: {best_metrics['eer']:.4f}")
        print(f"{'='*80}\n")
    
    # ===== SAVE FINAL BEST METRICS SUMMARY =====
    import json
    summary_path = 'dao_ckpt_srm_no2pass/best_metrics_summary.json'
    with open(summary_path, 'w') as f:
        # Convert tensor values to float for JSON serialization
        serializable_metrics = {}
        for key, value in best_metrics.items():
            if hasattr(value, 'item'):  # If it's a tensor
                serializable_metrics[key] = float(value.item())
            else:
                serializable_metrics[key] = float(value)
        
        json.dump(serializable_metrics, f, indent=4)
    print(f"📋 Best metrics summary saved to: {summary_path}")
    
    return best_metrics


from torch.utils.data import DataLoader

train_dataset = FaceDataset(train_list, transform=transform, real_transform=real_transform)
test_dataset = FaceDataset(test_list, transform=transform, real_transform=transform)
# Ultra-optimized DataLoader settings for maximum speed
train_dataloader = DataLoader(
    train_dataset, 
    batch_size=32,  # Increased batch size for better GPU utilization
    num_workers=2,  # Reduced to avoid worker creation warning
    shuffle=True,
    pin_memory=True,  # Faster GPU transfer
    persistent_workers=True,  # Keep workers alive between epochs
    prefetch_factor=4,  # Prefetch more batches
    drop_last=True  # Consistent batch sizes for better optimization
)

test_dataloader = DataLoader(
    test_dataset, 
    batch_size=32,  # Consistent batch size
    num_workers=2,  # Reduced to avoid worker creation warning
    shuffle=False,  # No need to shuffle test data
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=4,
    drop_last=False
)


def accuracy_fn(logits, labels):

    # Get predicted class by taking the index of the max logit
    preds = torch.argmax(logits, dim=1)  # Get class index (0 or 1)

    # Compute accuracy
    accuracy = (preds == labels).float().mean().item() * 100  # Convert to percentage

    return accuracy


model = U2NET().to(device)

# Summary model
# Updated to match new U2NET signature: SWT(2) + DCT(1) + Laplacian(2) + SRM(3) = 8 channels
summary(model, input_size=[(3, 256, 256), (8, 256, 256), (1, 256, 256)])

from torchmetrics.classification import BinaryAUROC

torch.manual_seed(42)
torch.cuda.manual_seed(42)

# ===== ULTRA-SPEED OPTIMIZATION SETTINGS =====
# Maximum performance without memory constraints

# Enable cuDNN benchmarking for consistent input sizes
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False  # Slightly faster
torch.backends.cudnn.allow_tf32 = True  # Enable TF32 for Ampere GPUs

# Enable optimal memory format
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True

# Check CUDA capability before compilation
cuda_capability = None
if torch.cuda.is_available():
    cuda_capability = torch.cuda.get_device_capability()
    print(f"🔍 CUDA Capability: {cuda_capability}")

# Compile model for maximum speed (PyTorch 2.0+) - only for modern GPUs
try:
    # Only compile if CUDA capability >= 7.0 (required for Triton backend)
    if cuda_capability and cuda_capability[0] >= 7:
        model = torch.compile(model, mode="max-autotune")
        print("✅ Model compiled with max-autotune for ultra speed!")
    else:
        print(f"⚠️ Skipping model compilation - CUDA capability {cuda_capability} < 7.0 (Triton requires >= 7.0)")
        print("🔧 Running in standard mode for compatibility with older GPUs")
except Exception as e:
    print(f"⚠️ Model compilation failed: {e}")
    print("🔧 Using standard mode")

# Convert model to optimal memory format
model = model.to(memory_format=torch.channels_last)

# Custom EER implementation compatible with torchmetrics interface
class CustomEER:
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset the metric state"""
        self.preds = []
        self.targets = []
    
    def update(self, preds, target):
        """Update the metric state with new predictions and targets"""
        # Store predictions and targets for batch computation
        if torch.is_tensor(preds):
            if preds.dim() > 1 and preds.shape[1] > 1:
                # Multi-class case: take the probability for the positive class
                preds = preds[:, 1]  # Assuming binary classification with class 1 as positive
            self.preds.extend(preds.detach().cpu().numpy())
        else:
            self.preds.extend(preds)
            
        if torch.is_tensor(target):
            self.targets.extend(target.detach().cpu().numpy())
        else:
            self.targets.extend(target)
    
    def compute(self):
        """Compute the EER from accumulated predictions and targets"""
        if len(self.preds) == 0 or len(self.targets) == 0:
            return 0.0
        return self.calculate_eer(self.targets, self.preds)
    
    def __call__(self, preds, target):
        """Direct computation without state update"""
        return self.calculate_eer(target.cpu().numpy(), preds.cpu().numpy())
    
    def calculate_eer(self, y_true, y_scores):
        """Calculate Equal Error Rate"""
        from sklearn.metrics import roc_curve
        import numpy as np
        try:
            fpr, tpr, thresholds = roc_curve(y_true, y_scores)
            fnr = 1 - tpr
            eer_idx = np.nanargmin(np.absolute(fnr - fpr))
            return float((fpr[eer_idx] + fnr[eer_idx]) / 2.0)
        except Exception as e:
            print(f"Warning: EER calculation failed: {e}")
            return 0.0

loss_seg = nn.L1Loss()
loss_clf = nn.CrossEntropyLoss()

# ===== COMPREHENSIVE METRICS INITIALIZATION =====
# Segmentation metrics
psnr = PeakSignalNoiseRatio().to(device)
ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

# Classification metrics
auc = BinaryAUROC(thresholds=None).to(device)  # Area Under ROC Curve
f1 = BinaryF1Score().to(device)               # F1 Score
precision = BinaryPrecision().to(device)      # Precision
recall = BinaryRecall().to(device)           # Recall (Sensitivity/TPR)
specificity = BinarySpecificity().to(device) # Specificity (TNR)
accuracy = BinaryAccuracy().to(device)       # Accuracy
eer = CustomEER()                            # Equal Error Rate

# Ultra-optimized optimizer and scheduler settings
optimizer = torch.optim.AdamW(  # AdamW is generally faster than Adam
    params=model.parameters(), 
    lr=0.001,  # Slightly higher learning rate for faster convergence
    weight_decay=0.01,  # L2 regularization
    betas=(0.9, 0.999),
    eps=1e-8,
    foreach=True  # Faster multi-tensor operations if available
)

# More aggressive learning rate schedule for faster training
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=0.001,
    epochs=50,
    steps_per_epoch=len(train_dataloader),
    pct_start=0.1,  # 10% warm-up
    anneal_strategy='cos'  # Cosine annealing
)

model_results = train(
    model=model,
    train_dataloader=train_dataloader, 
    test_dataloader=test_dataloader, 
    optimizer=optimizer, 
    scheduler=scheduler, 
    loss_seg=loss_seg, 
    loss_clf=loss_clf, 
    psnr=psnr, 
    ssim=ssim, 
    auc=auc, 
    f1=f1,
    precision=precision,
    recall=recall,
    specificity=specificity,
    accuracy=accuracy,
    eer=eer,
    epochs=50
)

print(f"\n🎉 TRAINING COMPLETED!")
print(f"📁 Best checkpoints saved:")
print(f"   • dao_ckpt_srm_no2pass/best_loss_dcn_dao.pt - Best Loss: {model_results['loss']:.6f}")
print(f"   • dao_ckpt_srm_no2pass/best_psnr_dcn_dao.pt - Best PSNR: {model_results['psnr']:.4f}")
print(f"   • dao_ckpt_srm_no2pass/best_ssim_dcn_dao.pt - Best SSIM: {model_results['ssim']:.4f}")
print(f"   • dao_ckpt_srm_no2pass/best_auc_dcn_dao.pt - Best AUC: {model_results['auc']:.4f}")
print(f"   • dao_ckpt_srm_no2pass/best_f1_dcn_dao.pt - Best F1: {model_results['f1']:.4f}")
print(f"   • dao_ckpt_srm_no2pass/best_precision_dcn_dao.pt - Best Precision: {model_results['precision']:.4f}")
print(f"   • dao_ckpt_srm_no2pass/best_recall_dcn_dao.pt - Best Recall: {model_results['recall']:.4f}")
print(f"   • dao_ckpt_srm_no2pass/best_specificity_dcn_dao.pt - Best Specificity: {model_results['specificity']:.4f}")
print(f"   • dao_ckpt_srm_no2pass/best_accuracy_dcn_dao.pt - Best Accuracy: {model_results['accuracy']:.4f}")
print(f"   • dao_ckpt_srm_no2pass/best_eer_dcn_dao.pt - Best EER: {model_results['eer']:.4f}")