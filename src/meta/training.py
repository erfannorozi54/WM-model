"""Training and evaluation utilities for meta-learning."""

from typing import Dict
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}", leave=False)
    for batch in pbar:
        images = batch["images"].to(device)
        targets = batch["targets"].to(device)
        task_vector = batch["task_vector"].to(device)
        n_values = batch["n"].to(device)
        
        B, T = images.shape[:2]
        logits, _, _ = model(images, task_vector)
        
        loss = torch.tensor(0.0, device=device)
        for b in range(B):
            n = n_values[b].item()
            valid_start = min(n, T - 1)
            if valid_start < T:
                seq_logits = logits[b, valid_start:]
                seq_targets = targets[b, valid_start:]
                loss += criterion(seq_logits.reshape(-1, 3), seq_targets.reshape(-1))
        
        loss = loss / B
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item() * B
        
        for b in range(B):
            n = n_values[b].item()
            valid_start = min(n, T - 1)
            if valid_start < T:
                preds = logits[b, valid_start:].argmax(dim=-1)
                targs = targets[b, valid_start:]
                total_correct += (preds == targs).sum().item()
                total_samples += targs.numel()
        
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})
    
    return {
        "loss": total_loss / len(dataloader.dataset),
        "accuracy": total_correct / max(total_samples, 1),
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate model."""
    model.eval()
    total_correct = 0
    total_samples = 0
    total_loss = 0.0
    
    criterion = nn.CrossEntropyLoss(ignore_index=-1, reduction='sum')
    
    for batch in dataloader:
        images = batch["images"].to(device)
        targets = batch["targets"].to(device)
        task_vector = batch["task_vector"].to(device)
        n_values = batch["n"].to(device)
        
        B, T = images.shape[:2]
        logits, _, _ = model(images, task_vector)
        
        for b in range(B):
            n = n_values[b].item()
            valid_start = min(n, T - 1)
            if valid_start < T:
                seq_logits = logits[b, valid_start:]
                seq_targets = targets[b, valid_start:]
                
                loss = criterion(seq_logits.reshape(-1, 3), seq_targets.reshape(-1))
                total_loss += loss.item()
                
                preds = seq_logits.argmax(dim=-1)
                total_correct += (preds == seq_targets).sum().item()
                total_samples += seq_targets.numel()
    
    return {
        "loss": total_loss / max(total_samples, 1),
        "accuracy": total_correct / max(total_samples, 1),
    }
