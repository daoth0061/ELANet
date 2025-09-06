"""
Evaluation script for ELANet fake face manipulation detection
"""

import torch
import argparse
import os

from config import load_config
from data import create_datasets
from models import create_model
from evaluation import evaluate_model, compare_models
from utils import setup_device


def main():
    """Main evaluation function"""
    
    parser = argparse.ArgumentParser(description='Evaluate ELANet model')
    parser.add_argument('--config', type=str, default='config.yaml', 
                       help='Path to configuration file')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--output_dir', type=str, default='eval_results',
                       help='Directory to save evaluation results')
    parser.add_argument('--save_visualizations', action='store_true',
                       help='Save visualization plots')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup device
    device = setup_device(config)
    
    print(f"🎯 Starting ELANet evaluation:")
    print(f"   Checkpoint: {args.checkpoint}")
    print(f"   Output directory: {args.output_dir}")
    
    # Create datasets
    print("📂 Creating test dataset...")
    _, test_dataset = create_datasets(config)
    
    # Create data loader
    from torch.utils.data import DataLoader
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=config.get('training.batch_size', 32),
        num_workers=2,
        shuffle=False,
        pin_memory=True
    )
    
    # Create and load model
    print("🏗️ Loading model...")
    model = create_model(config)
    model = model.to(device)
    
    # Load checkpoint
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"✅ Model loaded from epoch {checkpoint.get('epoch', 'unknown')}")
    
    # Evaluate model
    print("📊 Evaluating model...")
    results = evaluate_model(
        model=model,
        test_dataloader=test_dataloader,
        device=device,
        save_visualizations=args.save_visualizations,
        output_dir=args.output_dir
    )
    
    print(f"\n🎉 EVALUATION COMPLETED!")
    print(f"📁 Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
