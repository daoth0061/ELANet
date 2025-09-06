"""
ELANet Model Architecture Documentation

This document provides a comprehensive overview of the ELANet model architecture for fake face manipulation detection. Use this as a reference for understanding and modifying the model components.

## Table of Contents
1. High-Level Architecture Overview
2. Model Components
   - HAFT (Hierarchical Adaptive Frequency Transform)
   - U2NET Architecture
   - ELA Encoder
   - Attention Mechanisms
   - Building Blocks
   - Frequency Feature Extraction
   - Classification Components
3. Data Flow
4. Configuration Parameters
5. Performance Optimizations

## High-Level Architecture Overview

ELANet is a multi-modal deep learning model designed for fake face manipulation detection. The model processes three input streams:
- RGB images (original face images)
- Frequency domain features (SWT, DCT, Laplacian, SRM filters)
- Grayscale images

The architecture combines these streams using hierarchical frequency processing and attention mechanisms to detect manipulated faces.

**Implementation:** Main model implementation in `models/u2net.py`

## Model Components

### HAFT (Hierarchical Adaptive Frequency Transform)

A sophisticated frequency domain processing module that adaptively processes image frequencies at multiple scales:

- **FrequencyContextEncoder**: Encodes frequency information at different hierarchical levels
  - **Implementation:** `models/haft.py`
  - **Purpose:** Extract contextual features from frequency patches

- **FilterPredictor**: Predicts adaptive filters for frequency domain manipulation
  - **Implementation:** `models/haft.py`
  - **Purpose:** Generate radial filters based on contextual information

- **HAFT Module**: Main component that integrates hierarchical contexts and applies adaptive filtering
  - **Implementation:** `models/haft.py`
  - **Key Methods:**
    - `_compute_hierarchical_contexts_batch`: Generates contexts at multiple scales
    - `_gather_ancestral_contexts_batch`: Aggregates context information across scales
    - `_apply_frequency_filtering_batch`: Applies predicted filters to feature maps

### U2NET Architecture

The backbone network that processes and combines multi-modal features:

- **Main U2NET Class**: Overall model architecture
  - **Implementation:** `models/u2net.py`
  - **Purpose:** Integrates RGB, frequency, and grayscale features for fake detection

- **Encoder-Decoder Structure**: Multi-scale feature extraction and reconstruction
  - **Implementation:** `models/u2net.py`
  - **Purpose:** Extract features at multiple scales and generate segmentation mask

### ELA Encoder

Specialized encoder for error level analysis features:

- **EncodeELA**: Multi-scale feature extraction with HAFT integration
  - **Implementation:** `models/encoder.py`
  - **Key Components:**
    - Multi-scale stem for initial feature extraction
    - Convolutional blocks with CBAM attention
    - Integration with HAFT for frequency domain processing

### Attention Mechanisms

Various attention mechanisms used throughout the model:

- **CBAM (Convolutional Block Attention Module)**: Channel and spatial attention
  - **Implementation:** `models/attention.py`
  - **Components:**
    - `ChannelAttention`: Focuses on important channels
    - `SpatialAttention`: Highlights important spatial regions

- **SEBlock (Squeeze-and-Excitation)**: Channel-wise attention
  - **Implementation:** `models/attention.py`
  - **Purpose:** Recalibrate channel-wise feature responses

- **AttentionGateV2**: Feature fusion between RGB and frequency domains
  - **Implementation:** `models/attention.py`
  - **Purpose:** Guided attention for feature fusion

### Building Blocks

Basic building blocks for the model architecture:

- **RSU Blocks**: Residual U-blocks of varying depths (RSU7, RSU6, RSU5, RSU4, RSU4F)
  - **Implementation:** `models/blocks.py`
  - **Purpose:** Multi-scale feature extraction at different network stages
  - **Variants:**
    - `RSU7`: 7-layer block used in early stages
    - `RSU6`: 6-layer block
    - `RSU5`: 5-layer block
    - `RSU4`: 4-layer block
    - `RSU4F`: 4-layer block with dilated convolutions

- **REBNCONV**: Residual Enhanced Batch Norm Convolution
  - **Implementation:** `models/blocks.py`
  - **Purpose:** Basic convolution block with residual connection

- **ConvBlock**: Basic convolutional block with skip connections
  - **Implementation:** `models/blocks.py`
  - **Purpose:** Feature extraction with residual connections

- **DeformableConv2d**: Deformable convolution for adaptive receptive fields
  - **Implementation:** `models/blocks.py`
  - **Purpose:** Adapt convolution sampling locations based on input features

### Frequency Feature Extraction

Preprocessing methods for extracting frequency domain features:

- **SWT (Stationary Wavelet Transform)**
  - **Implementation:** `data/frequency_processing.py`
  - **Purpose:** Multi-resolution analysis without downsampling

- **DCT (Discrete Cosine Transform)**
  - **Implementation:** `data/frequency_processing.py`
  - **Purpose:** Extract high-frequency components in frequency domain

- **Laplacian Filter**
  - **Implementation:** `data/frequency_processing.py`
  - **Purpose:** Edge detection for artifact identification

- **SRM (Steganalysis Rich Model) Filters**
  - **Implementation:** `data/frequency_processing.py`
  - **Purpose:** Detect subtle manipulation artifacts

### Classification Components

Components for binary classification of real/fake faces:

- **Classifier**: Final classification head
  - **Implementation:** `models/blocks.py`
  - **Purpose:** Convert features to binary classification logits

## Data Flow

1. **Input Processing**:
   - RGB images: Standard CNN processing
   - Frequency features: SWT, DCT, Laplacian, SRM preprocessing
   - Grayscale: Intensity channel processing

2. **Feature Extraction**:
   - EncodeELA extracts features from RGB inputs
   - HAFT processes frequency domain features
   - Attention mechanisms highlight important features

3. **Feature Fusion**:
   - AttentionGateV2 fuses RGB and frequency features

4. **Decoder Processing**:
   - U2NET decoder generates segmentation mask
   - Multi-scale feature integration

5. **Output Generation**:
   - Segmentation mask for pixel-level manipulation detection
   - Binary classification for image-level real/fake decision

## Configuration Parameters

Key configurable parameters (defined in `config.yaml`):

```yaml
dataset:
  base_url: "datasets/FaceForensic_ImageData/FF+"
  deepfake_type: "deepfakes"
  train_test_split: 0.8

model:
  input_size: [256, 256]
  base_channels: 64
  
  haft:
    num_levels: 3
    num_radial_bins: 16
    context_vector_dim: 64
  
  encoder:
    use_se: true
    use_cbam: true
    use_deformable: true
  
  frequency:
    swt_level: 2
    dct_keep_ratio: 0.2
    use_srm: true

training:
  batch_size: 32
  epochs: 50
  learning_rate: 0.001
  weight_decay: 0.01
  seg_weight: 0.3
  clf_weight: 0.7
  
  scheduler:
    type: "onecycle"
    max_lr: 0.001
    pct_start: 0.1
    
  checkpoints:
    save_dir: "checkpoints"
    save_best_only: true
    metrics: ["auc", "f1", "accuracy"]
    
optimization:
  use_amp: true
  compile_model: true
  channels_last: true
  cudnn_benchmark: true
```

## Performance Optimizations

The model includes several performance optimizations:

- **Mixed Precision Training**: Uses AMP for faster training
  - **Implementation:** `utils/optimization.py`

- **Model Compilation**: Uses PyTorch 2.0+ compilation
  - **Implementation:** `utils/optimization.py`

- **Memory Format Optimization**: Channels-last memory format
  - **Implementation:** `utils/optimization.py`

- **Vectorized Operations**: Optimized batch processing
  - **Implementation:** Throughout the code, especially in HAFT

- **cuDNN Optimizations**: Benchmarking and TF32 precision
  - **Implementation:** `utils/optimization.py`

---

To modify the model architecture, edit this documentation with your desired changes, and then request implementation of the updated architecture based on the modified specifications.
"""
