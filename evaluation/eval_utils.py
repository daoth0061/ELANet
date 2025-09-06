"""
Evaluation utilities and metrics
"""

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns


def visualize_results(images: torch.Tensor, recon_images: torch.Tensor, save_path: str = None):
    """
    Visualize original images and reconstructed masks
    
    Args:
        images: Original mask images tensor
        recon_images: Reconstructed mask images tensor
        save_path: Optional path to save the visualization
    """
    images = images.detach().cpu().numpy()
    recon_images = recon_images.detach().cpu().numpy()
    
    fig, axes = plt.subplots(4, 2, figsize=(12, 24))
    axes = axes.flatten()
    
    for i in range(4):
        img = np.transpose(images[i], (1, 2, 0))
        recon_img = np.transpose(recon_images[i], (1, 2, 0))
        
        axes[2*i].imshow(img, cmap='gray')
        axes[2*i].set_title(f"Target mask {i+1}")
        axes[2*i].axis("off")
        
        axes[2*i + 1].imshow(recon_img, cmap='gray')
        axes[2*i + 1].set_title(f"Predicted mask {i+1}")
        axes[2*i + 1].axis("off")

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to: {save_path}")
    else:
        plt.show()


def plot_confusion_matrix(y_true, y_pred, class_names=['Real', 'Fake'], save_path=None):
    """
    Plot confusion matrix
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Names of the classes
        save_path: Optional path to save the plot
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to: {save_path}")
    else:
        plt.show()


def evaluate_model(model, test_dataloader, device='cuda', save_visualizations=False, output_dir='eval_results'):
    """
    Comprehensive model evaluation
    
    Args:
        model: Trained model
        test_dataloader: Test data loader
        device: Device to run evaluation on
        save_visualizations: Whether to save visualization plots
        output_dir: Directory to save evaluation results
        
    Returns:
        Dictionary containing evaluation metrics
    """
    model.eval()
    
    all_predictions = []
    all_labels = []
    all_pred_probs = []
    sample_masks = []
    sample_pred_masks = []
    
    with torch.no_grad():
        for batch_idx, (img, freq_img, gray_img, mask, label) in enumerate(test_dataloader):
            img = img.to(device)
            freq_img = freq_img.to(device)
            gray_img = gray_img.to(device)
            mask = mask.to(device)
            label = label.to(device)
            
            pred_mask, pred_label = model(img, freq_img, gray_img)
            
            # Get classification predictions
            pred_probs = F.softmax(pred_label, dim=1)[:, 1]  # Probability for class 1 (fake)
            pred_binary = torch.argmax(pred_label, dim=1)
            
            all_predictions.extend(pred_binary.cpu().numpy())
            all_labels.extend(label.cpu().numpy())
            all_pred_probs.extend(pred_probs.cpu().numpy())
            
            # Save some samples for visualization
            if batch_idx == 0 and save_visualizations:
                sample_masks = mask[:4]
                sample_pred_masks = pred_mask[:4]
    
    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_pred_probs = np.array(all_pred_probs)
    
    # Calculate metrics
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    
    results = {
        'accuracy': accuracy_score(all_labels, all_predictions),
        'precision': precision_score(all_labels, all_predictions),
        'recall': recall_score(all_labels, all_predictions),
        'f1_score': f1_score(all_labels, all_predictions),
        'auc': roc_auc_score(all_labels, all_pred_probs),
    }
    
    # Calculate EER
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(all_labels, all_pred_probs)
    fnr = 1 - tpr
    eer_idx = np.nanargmin(np.absolute(fnr - fpr))
    results['eer'] = (fpr[eer_idx] + fnr[eer_idx]) / 2.0
    
    # Print classification report
    print("Classification Report:")
    print(classification_report(all_labels, all_predictions, target_names=['Real', 'Fake']))
    
    # Print metrics
    print(f"\nEvaluation Results:")
    for metric, value in results.items():
        print(f"{metric.upper()}: {value:.4f}")
    
    if save_visualizations:
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Plot confusion matrix
        plot_confusion_matrix(
            all_labels, all_predictions, 
            save_path=os.path.join(output_dir, 'confusion_matrix.png')
        )
        
        # Visualize sample results
        if len(sample_masks) > 0:
            visualize_results(
                sample_masks, sample_pred_masks,
                save_path=os.path.join(output_dir, 'sample_results.png')
            )
        
        # Plot ROC curve
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {results["auc"]:.4f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, 'roc_curve.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Evaluation results saved to: {output_dir}")
    
    return results


def compare_models(model_paths, test_dataloader, device='cuda', model_names=None):
    """
    Compare multiple trained models
    
    Args:
        model_paths: List of paths to model checkpoints
        test_dataloader: Test data loader
        device: Device to run evaluation on
        model_names: Optional list of model names for display
        
    Returns:
        Dictionary containing comparison results
    """
    if model_names is None:
        model_names = [f"Model_{i+1}" for i in range(len(model_paths))]
    
    results = {}
    
    for i, (model_path, model_name) in enumerate(zip(model_paths, model_names)):
        print(f"\nEvaluating {model_name}...")
        
        # Load model
        from ..models import create_model
        from ..config import load_config
        
        config = load_config()  # Load default config, you might want to pass specific configs
        model = create_model(config).to(device)
        
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Evaluate
        model_results = evaluate_model(model, test_dataloader, device, save_visualizations=False)
        results[model_name] = model_results
    
    # Print comparison table
    print(f"\n{'='*80}")
    print("MODEL COMPARISON")
    print(f"{'='*80}")
    
    # Print header
    metrics = list(results[model_names[0]].keys())
    header = f"{'Model':<15}"
    for metric in metrics:
        header += f"{metric.upper():<10}"
    print(header)
    print("-" * len(header))
    
    # Print results for each model
    for model_name in model_names:
        row = f"{model_name:<15}"
        for metric in metrics:
            row += f"{results[model_name][metric]:<10.4f}"
        print(row)
    
    return results
