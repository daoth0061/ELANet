"""
Example usage script for ELANet
This script demonstrates how to use the modular ELANet implementation
"""

import torch
from config import load_config
from data import create_datasets
from models import create_model
from utils import setup_device, print_model_info


def main():
    """Demonstrate basic usage of ELANet components"""
    
    print("🚀 ELANet Example Usage")
    print("="*50)
    
    # 1. Load configuration
    print("1. Loading configuration...")
    config = load_config('config.yaml')
    print(f"   ✅ Config loaded from config.yaml")
    
    # 2. Setup device
    print("2. Setting up device...")
    device = setup_device(config)
    
    # 3. Create datasets (this will show dataset statistics)
    print("3. Creating datasets...")
    try:
        train_dataset, test_dataset = create_datasets(config)
        print(f"   ✅ Datasets created successfully")
        print(f"   📊 Train samples: {len(train_dataset)}")
        print(f"   📊 Test samples: {len(test_dataset)}")
    except Exception as e:
        print(f"   ⚠️ Dataset creation failed: {e}")
        print(f"   💡 Please check your dataset path in config.yaml")
    
    # 4. Create model
    print("4. Creating model...")
    model = create_model(config)
    model = model.to(device)
    print(f"   ✅ Model created and moved to {device}")
    
    # 5. Print model information
    print("5. Model information:")
    print_model_info(model)
    
    # 6. Test forward pass with dummy data
    print("6. Testing forward pass...")
    try:
        # Create dummy inputs matching the expected format
        batch_size = 2
        rgb_img = torch.randn(batch_size, 3, 256, 256).to(device)
        freq_img = torch.randn(batch_size, 8, 160, 160).to(device)  # 8 frequency channels
        gray_img = torch.randn(batch_size, 1, 160, 160).to(device)
        
        model.eval()
        with torch.no_grad():
            pred_mask, pred_label = model(rgb_img, freq_img, gray_img)
        
        print(f"   ✅ Forward pass successful!")
        print(f"   📐 Predicted mask shape: {pred_mask.shape}")
        print(f"   📐 Predicted label shape: {pred_label.shape}")
        
    except Exception as e:
        print(f"   ⚠️ Forward pass failed: {e}")
    
    print("\n🎉 Example completed!")
    print("Next steps:")
    print("1. Prepare your dataset according to the FaceForensics++ format")
    print("2. Adjust config.yaml for your specific setup")
    print("3. Run: python train.py")
    print("4. Evaluate with: python evaluate.py --checkpoint checkpoints/best_auc.pt")


if __name__ == "__main__":
    main()
