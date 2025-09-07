"""
Data processing utilities for frequency domain features
"""

import cv2
import numpy as np
import pywt
from PIL import Image
import torch
import torch.nn.functional as F


def stationary_wavelet_transform(image, wavelet='haar', level=2, resize_to=None):
    """
    Apply Stationary Wavelet Transform for frequency analysis.
    
    Args:
        image: Grayscale numpy array (expects grayscale input directly)
        wavelet: Wavelet type (default: 'haar')
        level: Decomposition level (default: 2)
        resize_to: Optional tuple (width, height) to resize output
        
    Returns:
        SWT coefficients as numpy array
    """
    try:
        # Assume input is already grayscale numpy array
        # No internal conversion needed
        
        # Ensure minimum size for SWT decomposition
        divisor = 2 ** level
        current_h, current_w = image.shape
        pad_h = current_h % divisor
        pad_w = current_w % divisor
        
        if pad_h > 0 or pad_w > 0:
            pad_h = divisor - pad_h if pad_h > 0 else 0
            pad_w = divisor - pad_w if pad_w > 0 else 0
            pad_top    = pad_h // 2
            pad_bottom = pad_h - pad_top
            pad_left   = pad_w // 2
            pad_right  = pad_w - pad_left
            image = np.pad(image, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='edge')

        coeffs = pywt.swt2(image, wavelet, level=level)

        FS = []
        for i in range(len(coeffs)):
            fs = coeffs[i][1][0] + coeffs[i][1][1] + coeffs[i][1][2]
            # Handle potential division by zero
            max_val = np.max(fs)
            if max_val > 0:
                fs = np.where((fs / max_val * 255.0) < 255.0, (fs / max_val * 255.0), 255.0)
            else:
                fs = np.zeros_like(fs)
            fs = np.where(fs < 0, 0, fs)
            
            fs = np.expand_dims(fs, axis=2)
            FS.append(fs)

        FS = np.concatenate(FS, axis=2)
        
        if resize_to is not None:
            resized_channels = []
            for c in range(FS.shape[2]):
                ch_resized = cv2.resize(FS[:,:,c], resize_to, interpolation=cv2.INTER_CUBIC)
                resized_channels.append(ch_resized[..., np.newaxis])
            FS = np.concatenate(resized_channels, axis=2)

        
        return FS

    except Exception as e:
        print(f"[LỖI ẢNH] SWT processing -> {e}")
        raise


def dct_high_freq_only(image, keep_ratio=0.2, resize_to=None):
    """
    Giữ lại thông tin cao tần trong miền DCT.
    :param image: Grayscale numpy array (expects grayscale input directly)
    :param keep_ratio: tỉ lệ phần cao tần giữ lại (0-1)
    :param resize_to: resize ảnh về kích thước (W,H) nếu cần
    :return: ảnh khôi phục từ cao tần
    """
    # Assume input is already grayscale numpy array
    # No internal conversion needed
    img = image
    
    if img is None:
        raise ValueError(f"Invalid image input")

    if img.dtype != np.float32:
        img = img.astype(np.float32)
    if img.max() > 1.0:
        img /= 255.0


    # 2. DCT
    dct_coeff = cv2.dct(img)

    # 3. Lọc bỏ thấp tần (đặt thành 0 vùng top-left)
    h, w = dct_coeff.shape
    h_cut = max(1, int(h * keep_ratio))  # Ensure at least 1 pixel
    w_cut = max(1, int(w * keep_ratio))  # Ensure at least 1 pixel

    dct_low_removed = dct_coeff.copy()
    dct_low_removed[:h_cut, :w_cut] = 0   # xoá vùng thấp tần

    # 4. IDCT để khôi phục ảnh cao tần
    img_high = cv2.idct(dct_low_removed)
    img_high = np.clip(img_high, 0, 1)
    img_high = (img_high * 255).astype(np.uint8)

    if resize_to is not None:
        img_high = cv2.resize(img_high, resize_to, interpolation=cv2.INTER_CUBIC)

    return img_high


def laplacian_high_freq(image, resize_to=None):
    """
    Lọc cao tần bằng Laplacian.
    :param image: Grayscale numpy array (expects grayscale input directly)
    :param resize_to: (W,H) nếu cần resize
    :return: ảnh laplacian (edges/noise) và ảnh khôi phục khi chỉ giữ cao tần
    """
    # Assume input is already grayscale numpy array
    # No internal conversion needed
    img = image
    
    if img is None:
        raise ValueError(f"Invalid image input")

    # 2. Áp dụng Laplacian (bộ lọc high-pass)
    lap = cv2.Laplacian(img, ddepth=cv2.CV_32F, ksize=3)

    # Chuẩn hóa về [0,255]
    lap_norm = cv2.convertScaleAbs(lap)

    # 3. Nếu muốn "khôi phục" lại ảnh chỉ chứa cao tần
    high_freq_only = cv2.addWeighted(img.astype(np.float32), 0, lap, 1, 0)
    high_freq_only = np.clip(high_freq_only, 0, 255).astype(np.uint8)

    if resize_to is not None:
        lap_norm = cv2.resize(lap_norm, resize_to, interpolation=cv2.INTER_CUBIC)
        high_freq_only = cv2.resize(high_freq_only, resize_to, interpolation=cv2.INTER_CUBIC)

    return lap_norm, high_freq_only


def apply_srm_filters(image, resize_to=None):
    """
    Apply SRM (Steganalysis Rich Model) filters for artifact detection
    
    Args:
        image: Grayscale numpy array (expects grayscale input directly)
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
    
    # Assume input is already grayscale numpy array
    # No internal conversion needed
    img = image
    
    if img is None:
        raise ValueError(f"Invalid image input")
    
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
        
        # Resize if needed
        if resize_to is not None:
            norm_resp = cv2.resize(norm_resp, resize_to, interpolation=cv2.INTER_CUBIC)
        
        # Add channel dimension and append
        responses.append(np.expand_dims(norm_resp.astype(np.uint8), axis=2))
    
    # Concatenate all filter responses along channel dimension
    srm_features = np.concatenate(responses, axis=2)
    
    return srm_features


def extract_all_frequency_features(image, config):
    """
    Extract all frequency domain features for an image
    
    Args:
        image: Grayscale numpy array (expects grayscale input directly)
        config: Configuration object containing feature extraction parameters
        
    Returns:
        numpy.ndarray: Combined frequency features of shape (H, W, 8)
                      SWT(2) + DCT(1) + Laplacian(2) + SRM(3) = 8 channels
    """
    freq_size = tuple(config.get('data_processing.frequency_size', [160, 160]))
    
    # Extract SWT features
    swt_img = stationary_wavelet_transform(
        image, 
        wavelet=config.get('data_processing.swt.wavelet', 'haar'),
        level=config.get('data_processing.swt.level', 2),
        resize_to=freq_size
    )
    
    # Extract DCT features
    dct_img = dct_high_freq_only(
        image, 
        keep_ratio=config.get('data_processing.dct.keep_ratio', 0.2),
        resize_to=freq_size
    )
    dct_img = np.expand_dims(dct_img, axis=2)
    
    # Extract Laplacian features
    lap_edges, high_freq_img = laplacian_high_freq(image, resize_to=freq_size)
    lap_edges = np.expand_dims(lap_edges, axis=2)
    high_freq_img = np.expand_dims(high_freq_img, axis=2)
    
    # Extract SRM features
    srm_img = apply_srm_filters(image, resize_to=freq_size)
    
    # Concatenate all frequency features: SWT(2) + DCT(1) + Laplacian(2) + SRM(3) = 8 channels
    freq_features = np.concatenate((swt_img, dct_img, lap_edges, high_freq_img, srm_img), axis=2).astype(np.float32)
    
    # Convert frequency features and rgb image to tensor and normalize
    freq_img = torch.from_numpy(freq_features).permute(2, 0, 1)  # (C,H,W)
    # Normalize per channel
    freq_img = (freq_img - freq_img.mean(dim=(1,2), keepdim=True)) / (freq_img.std(dim=(1,2), keepdim=True) + 1e-6)

    return freq_img


def prepare_grayscale_for_haft(image, config):
    """
    Prepare grayscale image for HAFT processing
    
    Args:
        image: Grayscale numpy array (expects grayscale input directly)
        target_size: Target size for the image
        
    Returns:
        torch.Tensor: Grayscale image tensor of shape (1, H, W)
    """
    # Assume input is already grayscale numpy array
    # Resize image to target size
    gray_img = torch.from_numpy(image).unsqueeze(0)  # (1, H, W) format
    gray_img = F.interpolate(gray_img.unsqueeze(0), size=config.get('data_processing.grayscale_image_size', (160, 160)), mode='bilinear', align_corners=False).squeeze(0)
    # Normalize gray_img per channel
    gray_img = (gray_img - gray_img.mean(dim=(1,2), keepdim=True)) / (gray_img.std(dim=(1,2), keepdim=True) + 1e-6)
    
    return gray_img
