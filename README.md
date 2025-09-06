# ELANet: Fake Face Manipulation Detection

A professional implementation of ELANet for detecting fake face manipulations using hierarchical adaptive frequency transform (HAFT) and U2NET architecture.

## Features

- **HAFT (Hierarchical Adaptive Frequency Transform)**: Advanced frequency domain processing for artifact detection
- **U2NET Architecture**: Powerful encoder-decoder network for pixel-level manipulation localization
- **Multi-modal Processing**: RGB + frequency domain features + learnable frequency transforms
- **Professional Code Structure**: Modular, configurable, and maintainable codebase
- **Configuration Management**: YAML-based configuration for all parameters
- **Comprehensive Evaluation**: Multiple metrics and visualization tools

## Project Structure

```
ELANet/
├── config.yaml                 # Main configuration file
├── train.py                   # Training script
├── evaluate.py                # Evaluation script
├── config/
│   └── config_manager.py      # Configuration management
├── data/
│   ├── __init__.py
│   ├── dataset.py             # Dataset classes and utilities
│   └── frequency_processing.py # Frequency domain feature extraction
├── models/
│   ├── __init__.py
│   ├── u2net.py              # Main U2NET model
│   ├── encoder.py            # ELA encoder with HAFT
│   ├── haft.py               # HAFT implementation
│   ├── attention.py          # Attention mechanisms (CBAM, etc.)
│   └── blocks.py             # Basic building blocks
├── training/
│   ├── __init__.py
│   └── train_utils.py        # Training utilities and functions
├── evaluation/
│   ├── __init__.py
│   └── eval_utils.py         # Evaluation utilities
└── utils/
    ├── __init__.py
    └── optimization.py       # Performance optimization utilities
```

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd ELANet
   ```

2. **Install dependencies**:
   ```bash
   pip install torch torchvision torchaudio
   pip install torchmetrics opencv-python pillow
   pip install PyWavelets scikit-learn matplotlib seaborn
   pip install tqdm einops pyyaml
   ```

## Configuration

All parameters are configurable through `config.yaml`:

### Dataset Configuration
```yaml
dataset:
  base_url: "datasets/FaceForensic_ImageData/FF+"
  deepfake_type: "deepfakes"  # deepfakes, face2face, faceshifter, faceswap, neuraltexture
  train_test_split: 0.8
```

### Model Configuration
```yaml
model:
  haft:
    num_levels: 3
    num_radial_bins: 16
    context_vector_dim: 64
  attention:
    cbam:
      reduction_ratio: 16
      kernel_size: 7
```

### Training Configuration
```yaml
training:
  epochs: 50
  batch_size: 32
  learning_rate: 0.001
  loss_weights:
    segmentation: 0.7
    classification: 0.3
```

## Usage

### Training

```bash
python train.py
```

You can also specify a custom configuration file:
```bash
python train.py --config custom_config.yaml
```

### Evaluation

```bash
python evaluate.py --checkpoint checkpoints/best_auc.pt --save_visualizations
```

Options:
- `--checkpoint`: Path to model checkpoint
- `--output_dir`: Directory to save evaluation results (default: eval_results)
- `--save_visualizations`: Save visualization plots

### Custom Training Script

```python
from config import load_config
from data import create_datasets
from models import create_model
from training import train
from utils import setup_device

# Load configuration
config = load_config('config.yaml')

# Setup device and create datasets
device = setup_device(config)
train_dataset, test_dataset = create_datasets(config)

# Create model
model = create_model(config).to(device)

# Train model
# ... (see train.py for complete example)
```

## Model Architecture

### HAFT (Hierarchical Adaptive Frequency Transform)
- Processes frequency patches at multiple hierarchical levels
- Learns adaptive filters for magnitude and phase components
- Integrates contextual information across different scales

### U2NET Encoder-Decoder
- RSU blocks for multi-scale feature extraction
- Attention-guided feature fusion between RGB and frequency domains
- Multi-scale supervision for improved training

### Feature Processing
- **SWT**: Stationary Wavelet Transform for multi-resolution analysis
- **DCT**: Discrete Cosine Transform for frequency domain features
- **Laplacian**: High-frequency edge detection
- **SRM**: Steganalysis Rich Model filters for artifact detection

## Performance Optimization

The implementation includes several performance optimizations:

- **Mixed Precision Training**: Automatic mixed precision for faster training
- **Model Compilation**: PyTorch 2.0+ compilation for optimized execution
- **Memory Format Optimization**: Channels-last memory format for better performance
- **Vectorized Operations**: Efficient batch processing for HAFT

## Evaluation Metrics

The model is evaluated using comprehensive metrics:

- **Classification**: Accuracy, Precision, Recall, F1-Score, AUC, EER
- **Segmentation**: PSNR, SSIM for mask quality
- **Visualization**: Confusion matrices, ROC curves, sample predictions

## Checkpoints

The training automatically saves best checkpoints for each metric:
- `best_loss.pt` - Best validation loss
- `best_auc.pt` - Best AUC score
- `best_f1.pt` - Best F1 score
- `best_accuracy.pt` - Best accuracy
- And more...

## Dataset Support

Currently supports FaceForensics++ dataset with the following deepfake types:
- Deepfakes
- Face2Face
- FaceShifter
- FaceSwap
- NeuralTextures

## Contributing

1. Follow the existing code structure and style
2. Add appropriate documentation and comments
3. Test your changes thoroughly
4. Update configuration files as needed

## License

[Add your license information here]

## Citation

If you use this code in your research, please cite:

```bibtex
@article{elanet2024,
  title={ELANet: Hierarchical Adaptive Frequency Transform for Fake Face Detection},
  author={[Your Name]},
  journal={[Journal Name]},
  year={2024}
}
```
