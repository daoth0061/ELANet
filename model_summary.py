"""
Model Summary Script for ELANet
Displays detailed model architecture and parameter information
"""

import torch
from torchsummary import summary

# Import from modular structure
from config import load_config
from models import create_model
from utils import setup_device

def main():
    """Display model summary"""

    # Load configuration
    config = load_config('config.yaml')

    # Setup device
    device = setup_device(config)

    # Create model
    model = create_model(config)
    model = model.to(device)

    print("🔍 ELANet Model Summary")
    print("=" * 80)

    # Display model summary with correct input sizes
    # Based on original implementation: RGB(3) + Freq(8) + Gray(1) = 11 total channels
    # But wait, let me check the actual input sizes from the original code...

    # From original: [(3, 256, 256), (8, 256, 256), (1, 256, 256)]
    # RGB: (3, 256, 256)
    # Frequency features: (8, 256, 256) - SWT(2) + DCT(1) + Laplacian(2) + SRM(3)
    # Grayscale: (1, 256, 256)

    try:
        # Convert device to string for torchsummary
        device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
        summary(
            model,
            input_size=[(3, 256, 256), (8, 256, 256), (1, 160, 160)],
            device=device_str
        )
    except Exception as e:
        print(f"❌ Error displaying model summary: {e}")
        print("💡 Make sure the model is properly initialized and all dependencies are installed")

if __name__ == "__main__":
    main()
