"""
Training utilities and functions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import os
from tqdm.auto import tqdm
from sklearn.metrics import roc_curve
import numpy as np


class CustomEER:
    """Custom EER implementation compatible with torchmetrics interface"""
    
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
        try:
            fpr, tpr, thresholds = roc_curve(y_true, y_scores)
            fnr = 1 - tpr
            eer_idx = np.nanargmin(np.absolute(fnr - fpr))
            return float((fpr[eer_idx] + fnr[eer_idx]) / 2.0)
        except Exception as e:
            print(f"Warning: EER calculation failed: {e}")
            return 0.0


def train_step(model, train_dataloader, loss_seg, loss_clf, optimizer, scheduler, 
               metrics, config, epoch):
    """
    Perform one training step
    
    Args:
        model: The model to train
        train_dataloader: Training data loader
        loss_seg: Segmentation loss function
        loss_clf: Classification loss function
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        metrics: Dictionary of metric objects
        config: Configuration object
        epoch: Current epoch number
        
    Returns:
        Dictionary of computed metrics
    """
    model.train()
    
    # Get loss weights from config
    seg_weight = config.get('training.loss_weights.segmentation', 0.7)
    clf_weight = config.get('training.loss_weights.classification', 0.3)
    
    # Mixed precision training if enabled
    use_amp = config.get('optimization.use_amp', True)
    if use_amp:
        scaler = torch.cuda.amp.GradScaler()
    
    train_loss = 0
    device = next(model.parameters()).device
    
    for batch, (img, freq_img, gray_img, mask, label) in enumerate(train_dataloader):
        # Ultra-fast data transfer with non_blocking and memory format optimization
        img = img.to(device, non_blocking=True, memory_format=torch.channels_last)
        freq_img = freq_img.to(device, non_blocking=True, memory_format=torch.channels_last)
        gray_img = gray_img.to(device, non_blocking=True, memory_format=torch.channels_last)
        mask = mask.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True)
        
        # Mixed precision forward pass
        if use_amp:
            with torch.cuda.amp.autocast():
                pred_mask, pred_label = model(img, freq_img, gray_img)
                
                loss_segment = loss_seg(pred_mask, mask)
                loss_classifier = loss_clf(pred_label, label)
                loss = seg_weight * loss_segment + clf_weight * loss_classifier
        else:
            pred_mask, pred_label = model(img, freq_img, gray_img)
            
            loss_segment = loss_seg(pred_mask, mask)
            loss_classifier = loss_clf(pred_label, label)
            loss = seg_weight * loss_segment + clf_weight * loss_classifier
        
        train_loss += loss.item()

        # Update metrics
        with torch.no_grad():
            pred_label_softmax = F.softmax(pred_label.detach(), dim=1)[:, 1]
            pred_label_binary = torch.argmax(pred_label.detach(), dim=1)
            
            # Update all metrics
            for name, metric in metrics.items():
                if name in ['psnr', 'ssim']:
                    metric.update(pred_mask.detach(), mask)
                elif name == 'eer':
                    metric.update(pred_label_softmax, label)
                elif name in ['auc']:
                    metric.update(pred_label_softmax, label)
                else:  # binary classification metrics
                    metric.update(pred_label_binary, label)
        
        # Optimized backward pass
        optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()
        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        
        # Update scheduler if it requires step-wise updates
        if hasattr(scheduler, 'step') and 'OneCycle' in str(type(scheduler)):
            scheduler.step()
    
    # Compute all metrics
    train_loss /= len(train_dataloader)
    results = {'loss': train_loss}
    
    for name, metric in metrics.items():
        results[name] = metric.compute()
        metric.reset()
    
    return results


def test_step(model, test_dataloader, loss_seg, loss_clf, metrics, config):
    """
    Perform one testing step
    
    Args:
        model: The model to evaluate
        test_dataloader: Testing data loader
        loss_seg: Segmentation loss function
        loss_clf: Classification loss function
        metrics: Dictionary of metric objects
        config: Configuration object
        
    Returns:
        Dictionary of computed metrics
    """
    model.eval()
    
    # Get loss weights from config
    seg_weight = config.get('training.loss_weights.segmentation', 0.7)
    clf_weight = config.get('training.loss_weights.classification', 0.3)
    
    test_loss = 0
    device = next(model.parameters()).device
    
    with torch.no_grad():
        for batch, (img, freq_img, gray_img, mask, label) in enumerate(test_dataloader):
            img = img.to(device, non_blocking=True)
            freq_img = freq_img.to(device, non_blocking=True)
            gray_img = gray_img.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)
            
            pred_mask, pred_label = model(img, freq_img, gray_img)

            loss_segment = loss_seg(pred_mask, mask)
            loss_classifier = loss_clf(pred_label, label)
            loss = seg_weight * loss_segment + clf_weight * loss_classifier
            
            test_loss += loss.item()
            
            # Get predictions
            pred_label_softmax = F.softmax(pred_label, dim=1)[:, 1]
            pred_label_binary = torch.argmax(pred_label, dim=1)

            # Update all metrics
            for name, metric in metrics.items():
                if name in ['psnr', 'ssim']:
                    metric.update(pred_mask, mask)
                elif name == 'eer':
                    metric.update(pred_label_softmax, label)
                elif name in ['auc']:
                    metric.update(pred_label_softmax, label)
                else:  # binary classification metrics
                    metric.update(pred_label_binary, label)
            
    # Compute all metrics
    test_loss = test_loss / len(test_dataloader)
    results = {'loss': test_loss}
    
    for name, metric in metrics.items():
        results[name] = metric.compute()
        metric.reset()
    
    return results


def save_checkpoint(model, optimizer, scheduler, epoch, metrics, save_path, config):
    """
    Save model checkpoint
    
    Args:
        model: Model to save
        optimizer: Optimizer state
        scheduler: Scheduler state
        epoch: Current epoch
        metrics: Metrics dictionary
        save_path: Path to save checkpoint
        config: Configuration object
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        **{f'test_{name}': value for name, value in metrics.items()}
    }
    
    torch.save(checkpoint, save_path)


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


def train(model, train_dataloader, test_dataloader, optimizer, scheduler, 
          loss_seg, loss_clf, metrics, config):
    """
    Main training loop
    
    Args:
        model: Model to train
        train_dataloader: Training data loader
        test_dataloader: Testing data loader
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        loss_seg: Segmentation loss function
        loss_clf: Classification loss function
        metrics: Dictionary of metric objects (train and test sets)
        config: Configuration object
        
    Returns:
        Dictionary of best metrics achieved
    """
    
    epochs = config.get('training.epochs', 50)
    checkpoint_dir = config.get('checkpoints.save_dir', 'checkpoints')
    
    # Create checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)
    print(f"📁 Checkpoint directory '{checkpoint_dir}' created/verified")
    
    # Initialize best metrics for checkpoint saving
    metrics_to_track = config.get('checkpoints.metrics_to_track', ['loss', 'auc', 'f1'])
    best_metrics = {metric: float('inf') if metric in ['loss', 'eer'] else 0.0 for metric in metrics_to_track}
    
    train_metrics = metrics['train']
    test_metrics = metrics['test']
    
    for epoch in tqdm(range(epochs)):
        # Train model
        train_results = train_step(
            model, train_dataloader, loss_seg, loss_clf, optimizer, scheduler, 
            train_metrics, config, epoch)

        # Test model
        test_results = test_step(
            model, test_dataloader, loss_seg, loss_clf, test_metrics, config)

        # Save best checkpoints
        for metric_name in metrics_to_track:
            if metric_name in test_results:
                current_value = test_results[metric_name]
                
                # Check if this is a new best value
                is_better = False
                if metric_name in ['loss', 'eer']:  # Lower is better
                    if current_value < best_metrics[metric_name]:
                        is_better = True
                        best_metrics[metric_name] = current_value
                else:  # Higher is better
                    if current_value > best_metrics[metric_name]:
                        is_better = True
                        best_metrics[metric_name] = current_value
                
                if is_better:
                    checkpoint_path = os.path.join(checkpoint_dir, f'best_{metric_name}.pt')
                    save_checkpoint(model, optimizer, scheduler, epoch, test_results, checkpoint_path, config)
                    print(f"✅ New best {metric_name}: {current_value:.4f} - Checkpoint saved!")

        # Comprehensive logging
        print(f"\n{'='*80}")
        print(f"EPOCH {epoch+1}/{epochs}")
        print(f"{'='*80}")
        print(f"📊 TRAINING METRICS:")
        for name, value in train_results.items():
            print(f"   {name.upper()}: {value:.4f}")
        
        print(f"\n🎯 VALIDATION METRICS:")
        for name, value in test_results.items():
            print(f"   {name.upper()}: {value:.4f}")
        
        print(f"\n🏆 BEST METRICS SO FAR:")
        for name, value in best_metrics.items():
            print(f"   {name.upper()}: {value:.4f}")
        print(f"{'='*80}\n")
    
    # Save final best metrics summary
    summary_path = os.path.join(checkpoint_dir, 'best_metrics_summary.json')
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
