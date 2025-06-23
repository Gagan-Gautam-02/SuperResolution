import matplotlib.pyplot as plt
import numpy as np
from typing import List, Optional
import seaborn as sns

def plot_training_curves(train_losses: List[float], 
                        val_losses: List[float], 
                        val_psnrs: List[float],
                        save_path: str = 'training_curves.png') -> None:
    """Plot training curves"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss curves
    epochs = range(1, len(train_losses) + 1)
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # PSNR curve
    ax2.plot(epochs, val_psnrs, 'g-', label='Validation PSNR', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('PSNR (dB)')
    ax2.set_title('Validation PSNR')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_image_comparison(images: List[np.ndarray], 
                         titles: List[str],
                         save_path: Optional[str] = None,
                         figsize: tuple = (15, 5)) -> None:
    """Plot image comparison"""
    n_images = len(images)
    fig, axes = plt.subplots(1, n_images, figsize=figsize)
    
    if n_images == 1:
        axes = [axes]
    
    for i, (img, title) in enumerate(zip(images, titles)):
        if len(img.shape) == 3:
            axes[i].imshow(img)
        else:
            axes[i].imshow(img, cmap='gray')
        
        axes[i].set_title(title)
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_metrics_comparison(metrics_dict: dict, save_path: Optional[str] = None) -> None:
    """Plot comparison of different metrics"""
    methods = list(metrics_dict.keys())
    metrics = list(metrics_dict[methods[0]].keys())
    
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 4))
    
    if n_metrics == 1:
        axes = [axes]
    
    for i, metric in enumerate(metrics):
        values = [metrics_dict[method][metric] for method in methods]
        
        bars = axes[i].bar(methods, values)
        axes[i].set_title(f'{metric.upper()}')
        axes[i].set_ylabel('Value')
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            axes[i].text(bar.get_x() + bar.get_width()/2., height,
                        f'{value:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save
