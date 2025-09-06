"""
Data processing utilities for frequency domain features
"""

import cv2
import numpy as np
import pywt
from PIL import Image
import torch


def stationary_wavelet_transform(image_path, wavelet='haar', level=2, resize_to=None):
    """
    Apply Stationary Wavelet Transform to extract frequency features
    
    Args:
        image_path: Path to input image
        wavelet: Wavelet type (default: 'haar')
        level: Decomposition level (default: 2)
        resize_to: Tuple (width, height) to resize image
        
    Returns:
        numpy.ndarray: SWT features of shape (H, W, num_levels)
    """
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
        print(f"[ERROR SWT] {image_path} -> {e}")
        raise


def dct_high_freq_only(img_path, keep_ratio=0.2, resize_to=None):
    """
    Extract high frequency components using DCT
    
    Args:
        img_path: Path to input image
        keep_ratio: Ratio of high frequency components to keep (0-1)
        resize_to: Tuple (width, height) to resize image
        
    Returns:
        numpy.ndarray: High frequency DCT components
    """
    # Read grayscale image
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Cannot read image from {img_path}")
    
    if resize_to is not None:
        img = cv2.resize(img, resize_to, interpolation=cv2.INTER_CUBIC)

    img = img.astype(np.float32) / 255.0

    # Apply DCT
    dct_coeff = cv2.dct(img)

    # Filter out low frequencies (set top-left region to 0)
    h, w = dct_coeff.shape
    h_cut = int(h * keep_ratio)
    w_cut = int(w * keep_ratio)

    dct_low_removed = dct_coeff.copy()
    dct_low_removed[:h_cut, :w_cut] = 0   # Remove low frequency region

    # IDCT to recover high frequency image
    img_high = cv2.idct(dct_low_removed)
    img_high = np.clip(img_high, 0, 1)
    img_high = (img_high * 255).astype(np.uint8)

    return img_high


def laplacian_high_freq(img_path, resize_to=None):
    """
    Extract high frequency components using Laplacian filter
    
    Args:
        img_path: Path to input image
        resize_to: Tuple (width, height) to resize image
        
    Returns:
        Tuple[numpy.ndarray, numpy.ndarray]: (laplacian_edges, high_freq_only)
    """
    # Read grayscale image
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Cannot read image from {img_path}")
    
    if resize_to is not None:
        img = cv2.resize(img, resize_to, interpolation=cv2.INTER_CUBIC)

    # Apply Laplacian (high-pass filter)
    lap = cv2.Laplacian(img, ddepth=cv2.CV_32F, ksize=3)

    # Normalize to [0,255]
    lap_norm = cv2.convertScaleAbs(lap)

    # Create high frequency only image
    high_freq_only = cv2.addWeighted(img.astype(np.float32), 0, lap, 1, 0)
    high_freq_only = np.clip(high_freq_only, 0, 255).astype(np.uint8)

    return lap_norm, high_freq_only


def apply_srm_filters(img_path, resize_to=None):
    """
    Apply SRM (Steganalysis Rich Model) filters for artifact detection
    
    Args:
        img_path: Path to input image
        resize_to: Tuple (width, height) to resize image
        
    Returns:
        numpy.ndarray: SRM feature maps of shape (H, W, 3) - last 3 individual filter responses
    """
    # Define the last 3 most effective SRM filters
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
    
    # Read image
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not read image: {img_path}")
    
    # Resize if needed
    if resize_to is not None:
        img = cv2.resize(img, resize_to, interpolation=cv2.INTER_CUBIC)
    
    # Apply all SRM filters
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
    srm_features = np.concatenate(responses, axis=2)
    
    return srm_features


def extract_all_frequency_features(img_path, config):
    """
    Extract all frequency domain features for an image
    
    Args:
        img_path: Path to input image
        config: Configuration object containing feature extraction parameters
        
    Returns:
        numpy.ndarray: Combined frequency features of shape (H, W, 8)
                      SWT(2) + DCT(1) + Laplacian(2) + SRM(3) = 8 channels
    """
    freq_size = tuple(config.get('data_processing.frequency_size', [160, 160]))
    
    # Extract SWT features
    swt_img = stationary_wavelet_transform(
        img_path, 
        wavelet=config.get('data_processing.swt.wavelet', 'haar'),
        level=config.get('data_processing.swt.level', 2),
        resize_to=freq_size
    )
    
    # Extract DCT features
    dct_img = dct_high_freq_only(
        img_path, 
        keep_ratio=config.get('data_processing.dct.keep_ratio', 0.2),
        resize_to=freq_size
    )
    dct_img = np.expand_dims(dct_img, axis=2)
    
    # Extract Laplacian features
    lap_edges, high_freq_img = laplacian_high_freq(img_path, resize_to=freq_size)
    lap_edges = np.expand_dims(lap_edges, axis=2)
    high_freq_img = np.expand_dims(high_freq_img, axis=2)
    
    # Extract SRM features
    srm_img = apply_srm_filters(img_path, resize_to=freq_size)
    
    # Concatenate all frequency features: SWT(2) + DCT(1) + Laplacian(2) + SRM(3) = 8 channels
    freq_features = np.concatenate((swt_img, dct_img, lap_edges, high_freq_img, srm_img), axis=2).astype(np.float32)
    
    return freq_features


def prepare_grayscale_for_haft(img_path, target_size=(160, 160)):
    """
    Prepare grayscale image for HAFT processing
    
    Args:
        img_path: Path to input image
        target_size: Target size for the image
        
    Returns:
        torch.Tensor: Grayscale image tensor of shape (1, H, W)
    """
    # Read and resize grayscale image
    gray_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    gray_img = cv2.resize(gray_img, target_size, interpolation=cv2.INTER_CUBIC)
    gray_img = gray_img.astype(np.float32) / 255.0  # Normalize to [0, 1]
    gray_img = torch.from_numpy(gray_img).unsqueeze(0)  # Add channel dimension: (1, H, W)
    
    return gray_img
