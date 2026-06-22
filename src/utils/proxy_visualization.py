"""
Visualization utilities for proxy task pre-training.
"""

import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from typing import Dict, Optional
import numpy as np
from PIL import Image


FEATURE_DISPLAY = {
    "location": "Location",
    "identity": "Identity",
    "category": "Category",
    "match_binary": "Match (1-back)",
    "consecutive": "Consecutive",
    "alternating": "Alternating",
}

LOCATION_NAMES = ["Top-Left", "Top-Right", "Bottom-Left", "Bottom-Right"]


def visualize_proxy_sequence(batch: Dict[str, torch.Tensor],
                              hidden_seq: torch.Tensor,
                              proxy_heads,
                              sample_idx: int = 0,
                              save_path: Optional[Path] = None,
                              denormalize: bool = True):
    images = batch["images"][sample_idx]
    proxy_targets = batch["proxy_targets"][sample_idx]
    task_vector = batch["task_vector"][sample_idx]
    n_back = batch["n"][sample_idx].item()
    locations = batch["locations"][sample_idx]
    categories = batch["categories"][sample_idx]
    identities = batch["identities"][sample_idx]
    task_feature = batch["task_feature"][sample_idx] if isinstance(batch["task_feature"], list) else batch["task_feature"]

    T = images.shape[0]

    head = proxy_heads.get_head_for_feature(task_feature)
    logits = head(hidden_seq[sample_idx])
    preds = logits.argmax(dim=-1)

    feature_display = FEATURE_DISPLAY.get(task_feature, task_feature)

    fig = plt.figure(figsize=(18, 9))
    fig.suptitle(
        f"Proxy Task | Feature: {feature_display} | N={n_back} | "
        f"Task Vector: [{', '.join(f'{v:.0f}' for v in task_vector)}]",
        fontsize=14, fontweight='bold'
    )

    gs = fig.add_gridspec(3, T, height_ratios=[3, 0.8, 0.6], hspace=0.3, wspace=0.2)

    if denormalize:
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    correct_count = 0
    valid_count = 0

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

        target = proxy_targets[t].item()
        pred = preds[t].item()

        if target >= 0:
            is_correct = (target == pred)
            correct_count += int(is_correct)
            valid_count += 1
            color = 'green' if is_correct else 'red'
        else:
            is_correct = None
            color = 'gray'

        linewidth = 3
        rect = patches.Rectangle(
            (0, 0), img.shape[1], img.shape[0],
            linewidth=linewidth, edgecolor=color, facecolor='none',
            transform=ax_img.transData
        )
        ax_img.add_patch(rect)

        ax_meta = fig.add_subplot(gs[1, t])
        ax_meta.axis('off')
        meta_text = f"Loc: {locations[t].item()}\n"
        meta_text += f"Cat: {categories[t][:10]}\n"
        meta_text += f"ID: {identities[t][-3:]}"
        ax_meta.text(0.5, 0.5, meta_text, ha='center', va='center',
                     fontsize=9, family='monospace',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

        ax_pred = fig.add_subplot(gs[2, t])
        ax_pred.axis('off')

        if target >= 0:
            target_str = _format_target(target, task_feature, locations, categories, identities, t - n_back if t >= n_back else 0)
            pred_str = _format_pred(pred, task_feature)
            text_color = 'green' if is_correct else 'red'
            bg_color = 'lightgreen' if is_correct else 'lightcoral'
            pred_text = f"Target: {target_str}\nPred: {pred_str}"
        else:
            pred_text = f"(no target\nt < N={n_back})"
            text_color = 'gray'
            bg_color = 'lightgray'

        ax_pred.text(0.5, 0.5, pred_text, ha='center', va='center',
                     fontsize=9, fontweight='bold', color=text_color,
                     family='monospace',
                     bbox=dict(boxstyle='round', facecolor=bg_color, alpha=0.5))

    if valid_count > 0:
        accuracy = correct_count / valid_count
        fig.text(0.5, 0.02, f"Proxy Accuracy: {accuracy:.1%} ({correct_count}/{valid_count})",
                 ha='center', fontsize=12, fontweight='bold')

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def _format_target(target, task_feature, locations, categories, identities, ref_t):
    if task_feature == "location":
        return f"Loc {target}"
    elif task_feature == "identity":
        return f"ID {target}"
    elif task_feature == "category":
        return f"Cat {target}"
    elif task_feature in ("match_binary", "consecutive", "alternating"):
        return "Match" if target == 1 else "Non-Match"
    return str(target)


def _format_pred(pred, task_feature):
    if task_feature == "location":
        return f"Loc {pred}"
    elif task_feature == "identity":
        return f"ID {pred}"
    elif task_feature == "category":
        return f"Cat {pred}"
    elif task_feature in ("match_binary", "consecutive", "alternating"):
        return "Match" if pred == 1 else "Non-Match"
    return str(pred)


def save_proxy_training_sample(model, proxy_heads, batch, device, save_dir,
                                epoch, batch_idx=0, split_name="sample",
                                sample_idx=0):
    model.eval()
    with torch.no_grad():
        images = batch["images"].to(device)
        task_vec = batch["task_vector"].to(device)
        hidden_seq, _ = model(images, task_vec)

    save_path = save_dir / f"epoch_{epoch:03d}_{split_name}_{batch_idx:02d}.png"

    visualize_proxy_sequence(
        batch=batch,
        hidden_seq=hidden_seq.cpu(),
        proxy_heads=proxy_heads,
        sample_idx=sample_idx,
        save_path=save_path,
        denormalize=True,
    )

    model.train()
    return save_path
