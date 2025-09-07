"""
Main training script for ELANet fake face manipulation detection
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Try to import torchmetrics, provide fallback if not available
try:
    from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio
    from torchmetrics.classification import BinaryAUROC, BinaryF1Score, BinaryPrecision, BinaryRecall, BinarySpecificity, BinaryAccuracy
    TORCHMETRICS_AVAILABLE = True
except ImportError:
    print("Warning: torchmetrics not available. Please install with: pip install torchmetrics")
    TORCHMETRICS_AVAILABLE = False

from config import load_config
from data import create_datasets
from models import create_model
from training import train, CustomEER
from utils import setup_device, optimize_model, print_model_info


def create_optimizer_and_scheduler(model, config, train_dataloader):
    """Create optimizer and scheduler based on configuration"""
    
    # Optimizer configuration
    optimizer_config = config.get('training.optimizer', {})
    optimizer_type = optimizer_config.get('type', 'AdamW')
    lr = config.get('training.learning_rate', 0.001)
    weight_decay = config.get('training.weight_decay', 0.01)
    
    if optimizer_type == 'AdamW':
        optimizer = torch.optim.AdamW(
            params=model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=optimizer_config.get('betas', [0.9, 0.999]),
            eps=optimizer_config.get('eps', 1e-8),
            foreach=optimizer_config.get('foreach', True)
        )
    elif optimizer_type == 'Adam':
        optimizer = torch.optim.Adam(
            params=model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=optimizer_config.get('betas', [0.9, 0.999]),
            eps=optimizer_config.get('eps', 1e-8)
        )
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")
    
    # Scheduler configuration
    scheduler_config = config.get('training.scheduler', {})
    scheduler_type = scheduler_config.get('type', 'OneCycleLR')
    epochs = config.get('training.epochs', 50)
    
    if scheduler_type == 'OneCycleLR':
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=scheduler_config.get('max_lr', lr),
            epochs=epochs,
            steps_per_epoch=len(train_dataloader),
            pct_start=scheduler_config.get('pct_start', 0.1),
            anneal_strategy=scheduler_config.get('anneal_strategy', 'cos')
        )
    elif scheduler_type == 'CosineAnnealingLR':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=epochs,
            eta_min=scheduler_config.get('eta_min', 0)
        )
    else:
        raise ValueError(f"Unsupported scheduler type: {scheduler_type}")
    
    return optimizer, scheduler


def create_metrics(device):
    """Create metric objects for training and evaluation"""
    
    if not TORCHMETRICS_AVAILABLE:
        print("Warning: torchmetrics not available. Using fallback metrics.")
        # Return a dictionary with CustomEER only
        return {
            'eer': CustomEER()
        }
    
    # Segmentation metrics
    psnr = PeakSignalNoiseRatio().to(device)
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    # Classification metrics
    auc = BinaryAUROC(thresholds=None).to(device)
    f1 = BinaryF1Score().to(device)
    precision = BinaryPrecision().to(device)
    recall = BinaryRecall().to(device)
    specificity = BinarySpecificity().to(device)
    accuracy = BinaryAccuracy().to(device)
    eer = CustomEER()
    
    metrics = {
        'psnr': psnr,
        'ssim': ssim,
        'auc': auc,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'specificity': specificity,
        'accuracy': accuracy,
        'eer': eer
    }
    
    return metrics


def create_dataloaders(config):
    """Create training and testing data loaders"""
    
    # Create datasets
    train_dataset, test_dataset = create_datasets(config)
    
    # DataLoader configuration
    dataloader_config = config.get('training.dataloader', {})
    batch_size = config.get('training.batch_size', 16)
    num_workers = dataloader_config.get('num_workers', 2)
    pin_memory = dataloader_config.get('pin_memory', True)
    persistent_workers = dataloader_config.get('persistent_workers', True)
    prefetch_factor = dataloader_config.get('prefetch_factor', 4)
    
    # Training data loader
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=dataloader_config.get('shuffle_train', True),
        pin_memory=pin_memory,
        persistent_workers=persistent_workers and num_workers > 0,
        prefetch_factor=prefetch_factor if num_workers > 0 else 2,
        drop_last=dataloader_config.get('drop_last', True)
    )

    # Testing data loader
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers and num_workers > 0,
        prefetch_factor=prefetch_factor if num_workers > 0 else 2,
        drop_last=False
    )
    
    return train_dataloader, test_dataloader


def main():
    """Main training function"""
    
    # Load configuration
    config = load_config('config.yaml')
    
    # Setup device
    device = setup_device(config)
    
    # Set random seeds for reproducibility
    random_seed = config.get('dataset.random_seed', 42)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    
    print(f"🎯 Starting ELANet training with configuration:")
    print(f"   Dataset: {config.get('dataset.deepfake_types', ['deepfakes'])}")
    print(f"   Epochs: {config.get('training.epochs', 50)}")
    print(f"   Batch size: {config.get('training.batch_size', 16)}")
    print(f"   Learning rate: {config.get('training.learning_rate', 0.001)}")
    
    # Create data loaders
    print("📂 Creating datasets and data loaders...")
    train_dataloader, test_dataloader = create_dataloaders(config)
    
    # Create model
    print("🏗️ Creating model...")
    model = create_model(config)
    model = model.to(device)
    
    # Print model information
    print_model_info(model)
    
    # Optimize model
    print("⚡ Optimizing model...")
    model = optimize_model(model, config, device)
    
    # Create optimizer and scheduler
    print("🔧 Setting up optimizer and scheduler...")
    optimizer, scheduler = create_optimizer_and_scheduler(model, config, train_dataloader)
    
    # Create loss functions
    loss_seg = nn.L1Loss()
    loss_clf = nn.CrossEntropyLoss()
    
    # Create metrics
    print("📊 Setting up metrics...")
    train_metrics = create_metrics(device)
    test_metrics = create_metrics(device)
    
    metrics = {
        'train': train_metrics,
        'test': test_metrics
    }
    
    # Start training
    print("🚀 Starting training...")
    print("="*80)
    
    best_metrics = train(
        model=model,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_seg=loss_seg,
        loss_clf=loss_clf,
        metrics=metrics,
        config=config
    )
    
    # Print final results
    print(f"\n🎉 TRAINING COMPLETED!")
    print(f"📁 Best checkpoints saved to: {config.get('checkpoints.save_dir', 'checkpoints')}")
    print(f"🏆 Best metrics achieved:")
    for metric, value in best_metrics.items():
        print(f"   {metric.upper()}: {value:.4f}")


if __name__ == "__main__":
    main()
