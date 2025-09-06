"""
HAFT (Hierarchical Adaptive Frequency Transform) Block Implementation
Implements sophisticated frequency-domain processing for fake face detection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


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
