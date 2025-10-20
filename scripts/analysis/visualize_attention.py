#!/usr/bin/env python3
"""
Attention Heatmap Visualization Tool.

This script visualizes the attention weights from attention-enhanced models,
showing which spatial locations the model focuses on for different tasks.

Usage:
  # Visualize attention from trained model
  python visualize_attention.py \
    --checkpoint runs/wm_attention_mtmf/checkpoints/best_*.pt \
    --config configs/attention_mtmf.yaml
  
  # Visualize specific trial
  python visualize_attention.py \
    --checkpoint runs/wm_attention_mtmf/checkpoints/best_*.pt \
    --task location --n 2 --num_samples 5
"""

import sys
from pathlib import Path
import argparse
import torch
import numpy as np
from typing import Dict, Tuple, Optional
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from models import create_model
from data.nback_generator import TaskFeature, create_sample_stimulus_data
from data.dataset import NBackDataModule

try:
    import yaml
except ImportError:
    yaml = None


def load_model_from_checkpoint(
    checkpoint_path: Path,
    device: str = 'cpu',
) -> Tuple[torch.nn.Module, Dict]:
    """
    Load attention model from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model on
    
    Returns:
        model: Loaded model
        config: Model configuration
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']
    
    # Create model
    model = create_model(
        model_type=config['model_type'],
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers'],
        dropout=config['dropout'],
        pretrained_backbone=config['pretrained_backbone'],
        freeze_backbone=config['freeze_backbone'],
        attention_hidden_dim=config.get('attention_hidden_dim'),
        attention_dropout=config.get('attention_dropout', 0.1),
    )
    
    # Load weights
    model.load_state_dict(checkpoint['model_state'])
    model.to(device)
    model.eval()
    
    return model, config


def visualize_attention_heatmap(
    attention_weights: torch.Tensor,
    images: torch.Tensor,
    task_name: str,
    predictions: Optional[torch.Tensor] = None,
    targets: Optional[torch.Tensor] = None,
    save_path: Optional[Path] = None,
):
    """
    Visualize attention heatmaps overlaid on input images.
    
    Args:
        attention_weights: Attention weights (T, 1, H, W)
        images: Input images (T, 3, H, W)
        task_name: Name of the task
        predictions: Predicted responses (T,)
        targets: Target responses (T,)
        save_path: Path to save visualization
    """
    T = attention_weights.shape[0]
    
    # Create figure
    fig = plt.figure(figsize=(15, 4))
    gs = gridspec.GridSpec(2, T, height_ratios=[1, 0.3], hspace=0.3, wspace=0.1)
    
    response_names = ['No Action', 'Non-Match', 'Match']
    
    for t in range(T):
        # Get attention map and image
        attn_map = attention_weights[t, 0].cpu().numpy()  # (H, W)
        img = images[t].cpu().numpy().transpose(1, 2, 0)  # (H, W, 3)
        
        # Denormalize image (assuming ImageNet normalization)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = img * std + mean
        img = np.clip(img, 0, 1)
        
        # Resize attention map to match image size
        from scipy.ndimage import zoom
        h_img, w_img = img.shape[:2]
        h_attn, w_attn = attn_map.shape
        zoom_factors = (h_img / h_attn, w_img / w_attn)
        attn_map_resized = zoom(attn_map, zoom_factors, order=1)
        
        # Plot image with attention overlay
        ax_img = fig.add_subplot(gs[0, t])
        ax_img.imshow(img)
        ax_img.imshow(attn_map_resized, cmap='hot', alpha=0.5)
        ax_img.axis('off')
        
        # Add title with prediction/target
        title = f"t={t}"
        if predictions is not None and targets is not None:
            pred_name = response_names[predictions[t].item()]
            target_name = response_names[targets[t].item()]
            correct = "✓" if predictions[t] == targets[t] else "✗"
            title += f"\nPred: {pred_name}\nTarget: {target_name} {correct}"
        ax_img.set_title(title, fontsize=8)
        
        # Plot attention histogram
        ax_hist = fig.add_subplot(gs[1, t])
        ax_hist.bar(range(len(attn_map.flatten())), np.sort(attn_map.flatten())[::-1][:50])
        ax_hist.set_xlim([0, 50])
        ax_hist.set_ylim([0, attn_map.max()])
        ax_hist.set_xlabel('Rank', fontsize=7)
        ax_hist.set_ylabel('Weight', fontsize=7)
        ax_hist.tick_params(labelsize=6)
        ax_hist.set_title('Top 50 Weights', fontsize=7)
    
    plt.suptitle(f'Attention Heatmaps: {task_name} Task', fontsize=12, fontweight='bold')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
    else:
        plt.show()
    
    plt.close()


def visualize_attention_comparison(
    attention_weights: Dict[str, torch.Tensor],
    images: torch.Tensor,
    save_path: Optional[Path] = None,
):
    """
    Compare attention patterns across different tasks.
    
    Args:
        attention_weights: Dictionary mapping task names to attention weights
        images: Input images (T, 3, H, W)
        save_path: Path to save visualization
    """
    tasks = list(attention_weights.keys())
    T = images.shape[0]
    
    fig, axes = plt.subplots(len(tasks), T, figsize=(15, 3 * len(tasks)))
    
    if len(tasks) == 1:
        axes = axes.reshape(1, -1)
    
    for task_idx, task_name in enumerate(tasks):
        attn = attention_weights[task_name]  # (T, 1, H, W)
        
        for t in range(T):
            ax = axes[task_idx, t]
            
            # Get attention map and image
            attn_map = attn[t, 0].cpu().numpy()
            img = images[t].cpu().numpy().transpose(1, 2, 0)
            
            # Denormalize
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img = img * std + mean
            img = np.clip(img, 0, 1)
            
            # Resize attention
            from scipy.ndimage import zoom
            h_img, w_img = img.shape[:2]
            h_attn, w_attn = attn_map.shape
            zoom_factors = (h_img / h_attn, w_img / w_attn)
            attn_map_resized = zoom(attn_map, zoom_factors, order=1)
            
            # Plot
            ax.imshow(img)
            ax.imshow(attn_map_resized, cmap='hot', alpha=0.5)
            ax.axis('off')
            
            if task_idx == 0:
                ax.set_title(f't={t}', fontsize=10)
            
            if t == 0:
                ax.set_ylabel(task_name.capitalize(), fontsize=10, rotation=0, ha='right', va='center')
    
    plt.suptitle('Attention Patterns Across Tasks', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
    else:
        plt.show()
    
    plt.close()


def analyze_attention_statistics(
    attention_weights: torch.Tensor,
    task_name: str,
) -> Dict:
    """
    Compute statistics about attention patterns.
    
    Args:
        attention_weights: Attention weights (T, 1, H, W)
        task_name: Task name
    
    Returns:
        Statistics dictionary
    """
    T, _, H, W = attention_weights.shape
    
    stats = {
        'task': task_name,
        'num_timesteps': T,
        'spatial_dims': (H, W),
    }
    
    # Per-timestep statistics
    timestep_stats = []
    for t in range(T):
        attn_map = attention_weights[t, 0].cpu().numpy()
        
        timestep_stats.append({
            'timestep': t,
            'max_weight': float(attn_map.max()),
            'mean_weight': float(attn_map.mean()),
            'std_weight': float(attn_map.std()),
            'entropy': float(-np.sum(attn_map * np.log(attn_map + 1e-10))),
            'sparsity': float((attn_map < attn_map.mean()).sum() / attn_map.size),
        })
    
    stats['per_timestep'] = timestep_stats
    
    # Overall statistics
    all_weights = attention_weights.cpu().numpy().flatten()
    stats['overall'] = {
        'max_weight': float(all_weights.max()),
        'mean_weight': float(all_weights.mean()),
        'std_weight': float(all_weights.std()),
        'median_weight': float(np.median(all_weights)),
    }
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Visualize attention heatmaps from attention-enhanced model"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to config file (optional, extracted from checkpoint if not provided)"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=3,
        help="Number of samples to visualize"
    )
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        choices=["location", "identity", "category"],
        help="Specific task to visualize"
    )
    parser.add_argument(
        "--n",
        type=int,
        default=2,
        help="N-back value"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/attention_viz",
        help="Output directory for visualizations"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device (cpu or cuda)"
    )
    
    args = parser.parse_args()
    
    checkpoint_path = Path(args.checkpoint)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*70)
    print("ATTENTION VISUALIZATION")
    print("="*70)
    print(f"\nCheckpoint: {checkpoint_path}")
    print(f"Output directory: {output_dir}")
    
    # Load model
    print("\nLoading model...")
    model, config = load_model_from_checkpoint(checkpoint_path, device=args.device)
    print(f"✓ Model loaded: {config['model_type']}")
    
    # Check if model has attention
    if not config['model_type'].startswith('attention_'):
        print("✗ Error: This model doesn't have attention mechanism!")
        return
    
    # Create data module
    print("\nCreating data...")
    stimulus_data = create_sample_stimulus_data()
    
    task_features = [TaskFeature.LOCATION, TaskFeature.IDENTITY, TaskFeature.CATEGORY]
    if args.task:
        task_map = {
            'location': TaskFeature.LOCATION,
            'identity': TaskFeature.IDENTITY,
            'category': TaskFeature.CATEGORY,
        }
        task_features = [task_map[args.task]]
    
    data_module = NBackDataModule(
        stimulus_data=stimulus_data,
        n_values=[args.n],
        task_features=task_features,
        sequence_length=6,
        batch_size=args.num_samples,
        num_train=0,
        num_val=args.num_samples,
        num_test=0,
    )
    
    val_loader = data_module.val_dataloader()
    
    # Get a batch
    batch = next(iter(val_loader))
    images = batch['images'].to(args.device)  # (B, T, 3, H, W)
    task_vectors = batch['task_vector'].to(args.device)  # (B, 3)
    targets = batch['responses'].argmax(dim=-1)  # (B, T)
    
    print(f"✓ Data loaded: {images.shape[0]} samples")
    
    # Run model with attention
    print("\nGenerating attention heatmaps...")
    with torch.no_grad():
        preds, probs, hidden_seq, attention_weights = model.predict(
            images, task_vectors, return_attention=True
        )
    
    if attention_weights is None:
        print("✗ Error: Model didn't return attention weights!")
        return
    
    print(f"✓ Attention weights shape: {attention_weights.shape}")
    
    # Visualize each sample
    for i in range(images.shape[0]):
        task_idx = task_vectors[i].argmax().item()
        task_names = ['Location', 'Identity', 'Category']
        task_name = task_names[task_idx]
        
        print(f"\nSample {i+1}/{images.shape[0]}: {task_name} Task")
        
        # Visualize attention
        save_path = output_dir / f"attention_sample{i+1}_{task_name.lower()}.png"
        visualize_attention_heatmap(
            attention_weights=attention_weights[i],  # (T, 1, H', W')
            images=images[i],  # (T, 3, H, W)
            task_name=task_name,
            predictions=preds[i],
            targets=targets[i],
            save_path=save_path,
        )
        
        # Compute statistics
        stats = analyze_attention_statistics(attention_weights[i], task_name)
        print(f"  Attention statistics:")
        print(f"    Mean weight: {stats['overall']['mean_weight']:.6f}")
        print(f"    Max weight: {stats['overall']['max_weight']:.6f}")
        print(f"    Std weight: {stats['overall']['std_weight']:.6f}")
        
        # Check for high sparsity (focused attention)
        mean_sparsity = np.mean([ts['sparsity'] for ts in stats['per_timestep']])
        print(f"    Mean sparsity: {mean_sparsity:.3f}")
        if mean_sparsity > 0.7:
            print(f"    → Highly focused attention!")
        elif mean_sparsity < 0.5:
            print(f"    → Distributed attention")
    
    # Summary
    print("\n" + "="*70)
    print("VISUALIZATION COMPLETE")
    print("="*70)
    print(f"\nGenerated {images.shape[0]} attention visualizations")
    print(f"Saved to: {output_dir}")
    print("\nUse these visualizations to:")
    print("  • Understand which spatial regions the model attends to")
    print("  • Compare attention patterns across different tasks")
    print("  • Identify if attention correlates with correct/incorrect predictions")
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    try:
        import scipy.ndimage
    except ImportError:
        print("Warning: scipy not found. Attention heatmap resizing may not work.")
        print("Install with: pip install scipy")
    
    main()
