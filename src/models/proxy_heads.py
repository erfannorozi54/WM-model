"""
Multi-head classification module for proxy task pre-training.

Instead of a single 3-class classifier (match/non_match/no_action),
proxy heads provide separate classification heads for each feature type:
- Location head: predicts screen location (4 classes)
- Identity head: predicts object identity (N classes)
- Category head: predicts object category (4 classes)

The appropriate head is selected based on the task vector's feature encoding.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class ProxyHeads(nn.Module):
    def __init__(self, hidden_size: int, num_identities: int,
                 num_locations: int = 4, num_categories: int = 4):
        super().__init__()
        self.num_locations = num_locations
        self.num_identities = num_identities
        self.num_categories = num_categories

        self.location_head = nn.Linear(hidden_size, num_locations)
        self.identity_head = nn.Linear(hidden_size, num_identities)
        self.category_head = nn.Linear(hidden_size, num_categories)

        self.match_head = nn.Linear(hidden_size, 2)
        self.consecutive_head = nn.Linear(hidden_size, 2)

    @property
    def num_classes(self) -> Dict[str, int]:
        return {
            "location": self.num_locations,
            "identity": self.num_identities,
            "category": self.num_categories,
            "match_binary": 2,
            "consecutive": 2,
        }

    def forward(self, hidden_states: torch.Tensor,
                task_vectors: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {
            "location": self.location_head(hidden_states),
            "identity": self.identity_head(hidden_states),
            "category": self.category_head(hidden_states),
            "match_binary": self.match_head(hidden_states),
            "consecutive": self.consecutive_head(hidden_states),
        }

    def get_head_for_feature(self, task_feature: str) -> nn.Module:
        if task_feature == "location":
            return self.location_head
        elif task_feature == "identity":
            return self.identity_head
        elif task_feature == "category":
            return self.category_head
        elif task_feature in ("match_binary", "consecutive", "alternating"):
            return self.match_head
        else:
            return self.location_head


def compute_proxy_loss(proxy_heads: ProxyHeads,
                       hidden_seq: torch.Tensor,
                       task_vectors: torch.Tensor,
                       proxy_targets: torch.Tensor,
                       n_values: torch.Tensor,
                       task_features: list,
                       device: torch.device) -> Tuple[torch.Tensor, Dict[str, float]]:
    B, T, H = hidden_seq.shape
    total_loss = torch.tensor(0.0, device=device)
    total_correct = 0
    total_valid = 0

    per_task_stats = {}

    for b in range(B):
        tf = task_features[b]
        n = n_values[b].item()
        targets = proxy_targets[b]

        head = proxy_heads.get_head_for_feature(tf)
        logits = head(hidden_seq[b])

        valid_mask = targets >= 0
        if not valid_mask.any():
            continue

        valid_logits = logits[valid_mask]
        valid_targets = targets[valid_mask]

        nc = valid_logits.shape[-1]
        loss = F.cross_entropy(valid_logits, valid_targets.clamp(0, nc - 1))
        total_loss = total_loss + loss

        preds = valid_logits.argmax(dim=-1)
        correct = (preds == valid_targets.clamp(0, nc - 1)).sum().item()
        count = valid_targets.numel()
        total_correct += correct
        total_valid += count

        key = f"{tf}_n{n}"
        if key not in per_task_stats:
            per_task_stats[key] = {"correct": 0, "total": 0}
        per_task_stats[key]["correct"] += correct
        per_task_stats[key]["total"] += count

    total_loss = total_loss / max(B, 1)

    metrics = {
        "proxy_loss": total_loss.item(),
        "proxy_accuracy": total_correct / max(total_valid, 1),
    }
    for key, stats in per_task_stats.items():
        metrics[f"{key}_acc"] = stats["correct"] / max(stats["total"], 1)
        metrics[f"{key}_count"] = stats["total"]

    return total_loss, metrics


def compute_proxy_loss_batched(proxy_heads: ProxyHeads,
                                hidden_seq: torch.Tensor,
                                task_vectors: torch.Tensor,
                                proxy_targets: torch.Tensor,
                                n_values: torch.Tensor,
                                task_features: list,
                                device: torch.device) -> Tuple[torch.Tensor, Dict[str, float]]:
    B, T, H = hidden_seq.shape

    groups = {}
    for b in range(B):
        tf = task_features[b]
        if tf not in groups:
            groups[tf] = []
        groups[tf].append(b)

    total_loss = torch.tensor(0.0, device=device)
    total_correct = 0
    total_valid = 0
    per_task_stats = {}

    for tf, batch_indices in groups.items():
        head = proxy_heads.get_head_for_feature(tf)

        indices = torch.tensor(batch_indices, device=device)
        group_hidden = hidden_seq[indices]
        group_targets = proxy_targets[indices]

        logits = head(group_hidden)

        valid_mask = group_targets >= 0
        if not valid_mask.any():
            continue

        valid_logits = logits[valid_mask]
        valid_targets = group_targets[valid_mask]

        nc = valid_logits.shape[-1]
        valid_targets_clamped = valid_targets.clamp(0, nc - 1)
        loss = F.cross_entropy(valid_logits, valid_targets_clamped)
        total_loss = total_loss + loss

        preds = valid_logits.argmax(dim=-1)
        correct = (preds == valid_targets_clamped).sum().item()
        count = valid_targets.numel()
        total_correct += correct
        total_valid += count

        for b_local, b_global in enumerate(batch_indices):
            n = n_values[b_global].item()
            key = f"{tf}_n{n}"
            b_targets = group_targets[b_local]
            b_valid = b_targets >= 0
            if b_valid.any():
                b_preds = logits[b_local][b_valid].argmax(dim=-1)
                b_tgts = b_targets[b_valid].clamp(0, nc - 1)
                b_correct = (b_preds == b_tgts).sum().item()
                b_count = b_tgts.numel()
                if key not in per_task_stats:
                    per_task_stats[key] = {"correct": 0, "total": 0}
                per_task_stats[key]["correct"] += b_correct
                per_task_stats[key]["total"] += b_count

    total_loss = total_loss / max(len(groups), 1)

    metrics = {
        "proxy_loss": total_loss.item(),
        "proxy_accuracy": total_correct / max(total_valid, 1),
    }
    for key, stats in per_task_stats.items():
        metrics[f"{key}_acc"] = stats["correct"] / max(stats["total"], 1)
        metrics[f"{key}_count"] = stats["total"]

    return total_loss, metrics
