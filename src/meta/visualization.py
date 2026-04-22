"""Visualization utilities for meta-learning experiments."""

import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from typing import Dict, Optional, List
import numpy as np
from PIL import Image


def visualize_meta_sequence(
    batch: Dict[str, torch.Tensor],
    logits: torch.Tensor,
    sample_idx: int = 0,
    save_path: Optional[Path] = None,
    task_name: str = "unknown",
    method: str = "unknown",
    epoch: int = 0,
    denormalize: bool = True
):
    """
    Visualize a meta-learning sequence with predictions.
    
    Args:
        batch: Batch dictionary containing images, targets, task_vector, n
        logits: Model predictions (B, T, 3)
        sample_idx: Which sample in batch to visualize
        save_path: Where to save the visualization
        task_name: Name of the novel task
        method: Adaptation method name
        epoch: Current epoch (0 = before training)
        denormalize: Whether to denormalize images for display
    """
    images = batch["images"][sample_idx]
    targets = batch["targets"][sample_idx]
    preds = logits[sample_idx].argmax(dim=-1)
    n_back = batch["n"][sample_idx].item()
    
    T = images.shape[0]
    
    response_names = ["No Action", "Non-Match", "Match"]
    
    fig = plt.figure(figsize=(18, 8))
    
    epoch_label = "Before Training" if epoch == 0 else f"Epoch {epoch}"
    fig.suptitle(
        f"Meta-Learning | Task: {task_name} | Method: {method} | {epoch_label}",
        fontsize=16, fontweight='bold'
    )
    
    gs = fig.add_gridspec(3, T, height_ratios=[3, 0.8, 0.5], hspace=0.3, wspace=0.2)
    
    if denormalize:
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    
    for t in range(T):
        ax_img = fig.add_subplot(gs[0, t])
        
        img = images[t].cpu()
        if denormalize:
            img = img * std + mean
        img = img.clamp(0, 1)
        img = img.permute(1, 2, 0).numpy()
        
        ax_img.imshow(img)
        ax_img.axis('off')
        ax_img.set_title(f"t={t}", fontsize=12, fontweight='bold')
        
        target = targets[t].item()
        pred = preds[t].item()
        is_correct = (target == pred)
        
        color = 'green' if is_correct else 'red'
        rect = patches.Rectangle((0, 0), img.shape[1], img.shape[0],
                                  linewidth=3, edgecolor=color, facecolor='none')
        ax_img.add_patch(rect)
        
        ax_meta = fig.add_subplot(gs[1, t])
        ax_meta.axis('off')
        ax_meta.text(0.5, 0.5, f"N={n_back}", ha='center', va='center', fontsize=9,
                    family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        ax_pred = fig.add_subplot(gs[2, t])
        ax_pred.axis('off')
        
        target_name = response_names[target] if target < len(response_names) else str(target)
        pred_name = response_names[pred] if pred < len(response_names) else str(pred)
        
        pred_text = f"Target: {target_name}\nPred: {pred_name}"
        text_color = 'green' if is_correct else 'red'
        
        ax_pred.text(0.5, 0.5, pred_text, ha='center', va='center', fontsize=9,
                    fontweight='bold', color=text_color, family='monospace',
                    bbox=dict(boxstyle='round', 
                             facecolor='lightgreen' if is_correct else 'lightcoral', alpha=0.5))
    
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='green', linewidth=3, label='Correct'),
        Line2D([0], [0], color='red', linewidth=3, label='Incorrect')
    ]
    fig.legend(handles=legend_elements, loc='upper right', fontsize=11)
    
    accuracy = (targets == preds).float().mean().item()
    fig.text(0.5, 0.02, f"Sequence Accuracy: {accuracy:.1%}", ha='center', fontsize=12, fontweight='bold')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def save_meta_visualization(
    model,
    dataloader,
    device: torch.device,
    save_dir: Path,
    task_name: str,
    method: str,
    epoch: int = 0,
    num_samples: int = 2
):
    """Save visualizations for meta-learning experiment.
    
    Args:
        model: The model
        dataloader: Test dataloader
        device: Device
        save_dir: Directory to save visualizations
        task_name: Name of the novel task
        method: Adaptation method name
        epoch: Current epoch (0 = before training)
        num_samples: Number of samples to visualize
    """
    model.eval()
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= num_samples:
                break
            
            images = batch["images"].to(device)
            task_vector = batch["task_vector"].to(device)
            
            logits, _, _ = model(images, task_vector)
            
            visualize_meta_sequence(
                batch=batch,
                logits=logits.cpu(),
                sample_idx=0,
                save_path=save_dir / f"epoch_{epoch:03d}_sample_{batch_idx:02d}.png",
                task_name=task_name,
                method=method,
                epoch=epoch
            )
    
    model.train()
    return save_dir
