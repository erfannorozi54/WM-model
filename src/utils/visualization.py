"""
Visualization utilities for training monitoring.

This module provides functions to visualize N-back sequences with predictions.
"""

import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from typing import Dict, Optional
import numpy as np
from PIL import Image


def visualize_sequence_prediction(
    batch: Dict[str, torch.Tensor],
    logits: torch.Tensor,
    sample_idx: int = 0,
    save_path: Optional[Path] = None,
    denormalize: bool = True
):
    """
    Visualize a single sequence with task info, targets, and predictions.
    
    Args:
        batch: Batch dictionary containing:
            - images: (B, T, 3, H, W)
            - responses: (B, T, 3) one-hot targets
            - task_vector: (B, 3) one-hot task
            - n: (B,) n-back value
            - locations: (B, T)
            - categories: List of lists
            - identities: List of lists
        logits: Model predictions (B, T, 3)
        sample_idx: Which sample in batch to visualize
        save_path: Where to save the visualization
        denormalize: Whether to denormalize images for display
    """
    # Extract data for the selected sample
    images = batch["images"][sample_idx]  # (T, 3, H, W)
    targets = batch["responses"][sample_idx].argmax(dim=-1)  # (T,)
    task_vec = batch["task_vector"][sample_idx]  # (3,)
    n_back = batch["n"][sample_idx].item()
    locations = batch["locations"][sample_idx]  # (T,)
    categories = batch["categories"][sample_idx]
    identities = batch["identities"][sample_idx]
    preds = logits[sample_idx].argmax(dim=-1)  # (T,)
    
    T = images.shape[0]
    
    # Task name mapping
    task_names = ["Location", "Identity", "Category"]
    task_idx = task_vec.argmax().item()
    task_name = task_names[task_idx]
    
    # Response mapping
    response_names = ["No Action", "Non-Match", "Match"]
    
    # Create figure
    fig = plt.figure(figsize=(18, 8))
    
    # Title with task information
    fig.suptitle(
        f"N-Back Sequence | Task: {task_name} | N={n_back}",
        fontsize=16,
        fontweight='bold'
    )
    
    # Create grid: top row for images, bottom row for info
    gs = fig.add_gridspec(3, T, height_ratios=[3, 0.8, 0.5], hspace=0.3, wspace=0.2)
    
    # Denormalization constants (ImageNet)
    if denormalize:
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    
    for t in range(T):
        # --- Image subplot ---
        ax_img = fig.add_subplot(gs[0, t])
        
        # Prepare image for display
        img = images[t].cpu()
        if denormalize:
            img = img * std + mean
        img = img.clamp(0, 1)
        img = img.permute(1, 2, 0).numpy()
        
        ax_img.imshow(img)
        ax_img.axis('off')
        
        # Add timestep label
        ax_img.set_title(f"t={t}", fontsize=12, fontweight='bold')
        
        # Highlight based on correctness
        target = targets[t].item()
        pred = preds[t].item()
        is_correct = (target == pred)
        
        color = 'green' if is_correct else 'red'
        linewidth = 3
        
        rect = patches.Rectangle(
            (0, 0), img.shape[1], img.shape[0],
            linewidth=linewidth,
            edgecolor=color,
            facecolor='none',
            transform=ax_img.transData
        )
        ax_img.add_patch(rect)
        
        # --- Metadata subplot ---
        ax_meta = fig.add_subplot(gs[1, t])
        ax_meta.axis('off')
        
        # Prepare metadata text
        meta_text = f"Loc: {locations[t].item()}\n"
        meta_text += f"Cat: {categories[t][:10]}\n"
        meta_text += f"ID: {identities[t][-3:]}"  # Show last 3 chars (the ID number)
        
        ax_meta.text(
            0.5, 0.5, meta_text,
            ha='center', va='center',
            fontsize=9,
            family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3)
        )
        
        # --- Prediction subplot ---
        ax_pred = fig.add_subplot(gs[2, t])
        ax_pred.axis('off')
        
        # Prepare prediction text
        target_name = response_names[target]
        pred_name = response_names[pred]
        
        pred_text = f"Target: {target_name}\n"
        pred_text += f"Pred: {pred_name}"
        
        # Color based on correctness
        text_color = 'green' if is_correct else 'red'
        
        ax_pred.text(
            0.5, 0.5, pred_text,
            ha='center', va='center',
            fontsize=9,
            fontweight='bold',
            color=text_color,
            family='monospace',
            bbox=dict(boxstyle='round', 
                     facecolor='lightgreen' if is_correct else 'lightcoral',
                     alpha=0.5)
        )
    
    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='green', linewidth=3, label='Correct'),
        Line2D([0], [0], color='red', linewidth=3, label='Incorrect')
    ]
    fig.legend(handles=legend_elements, loc='upper right', fontsize=11)
    
    # Calculate overall accuracy for this sequence
    accuracy = (targets == preds).float().mean().item()
    fig.text(
        0.5, 0.02,
        f"Sequence Accuracy: {accuracy:.1%}",
        ha='center',
        fontsize=12,
        fontweight='bold'
    )
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    
    # Save or show
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def save_training_sample(
    model,
    batch: Dict[str, torch.Tensor],
    device: torch.device,
    save_dir: Path,
    epoch: int,
    batch_idx: int = 0,
    split_name: str = "sample",
    sample_idx: int = 0
):
    """
    Convenience function to save a visualization during training.
    
    Args:
        model: The model to get predictions from
        batch: Batch from dataloader
        device: Device to run inference on
        save_dir: Directory to save visualizations
        epoch: Current epoch number
        batch_idx: Visualization index for filename
        split_name: Name of the data split (e.g., 'train', 'val_novel_angle', 'val_novel_identity')
        sample_idx: Index within the batch to visualize
    """
    model.eval()
    
    with torch.no_grad():
        images = batch["images"].to(device)
        task_vec = batch["task_vector"].to(device)
        
        # Get predictions (without CNN activations to save memory)
        logits, _, _ = model(images, task_vec)
    
    # Create save path with split name and batch index
    save_path = save_dir / f"epoch_{epoch:03d}_{split_name}_{batch_idx:02d}.png"
    
    # Visualize
    visualize_sequence_prediction(
        batch=batch,
        logits=logits.cpu(),
        sample_idx=sample_idx,
        save_path=save_path,
        denormalize=True
    )
    
    model.train()
    
    return save_path


if __name__ == "__main__":
    # Demo: Create a fake batch for testing
    print("Testing visualization function...")
    
    B, T, H, W = 2, 6, 224, 224
    
    fake_batch = {
        "images": torch.randn(B, T, 3, H, W),
        "responses": torch.eye(3)[torch.randint(0, 3, (B, T))],
        "task_vector": torch.eye(3)[[0, 1]],
        "n": torch.tensor([2, 3]),
        "locations": torch.randint(0, 4, (B, T)),
        "categories": [["airplane"] * T, ["car"] * T],
        "identities": [["airplane_001"] * T, ["car_002"] * T]
    }
    
    fake_logits = torch.randn(B, T, 3)
    
    visualize_sequence_prediction(
        batch=fake_batch,
        logits=fake_logits,
        sample_idx=0,
        save_path=None  # Show instead of save
    )
    
    print("Visualization test complete!")
