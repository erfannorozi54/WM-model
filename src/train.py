#!/usr/bin/env python3
"""
Training script for the two-stage Working Memory model (Perceptual + Cognitive).
- Perceptual: ResNet50 feature extractor with 1x1 conv to RNN hidden size
- Cognitive: RNN/GRU/LSTM
- Classifier: 3-way response (no_action, non_match, match)

Features:
- AdamW optimizer
- MultiStepLR scheduler
- YAML/CLI configuration
- Validation hook to save hidden states per timestep

Usage:
    python -m src.train --config configs/mtmf.yaml
    python -m src.train --model_type gru --hidden_size 512
"""

import os
from pathlib import Path
import argparse
import time
import json
from typing import List, Dict, Any, Optional

import torch
import torch.nn as nn
import torch.optim as optim

# Import from same package
from .data.dataset import NBackDataModule
from .data.nback_generator import TaskFeature
from .data.shapenet_downloader import create_sample_stimulus_data
from .models import (
    create_model,
    print_model_summary,
    get_model_info,
)

try:
    import yaml
except Exception:
    yaml = None


def load_real_stimulus_data(data_dir: str = "data/stimuli") -> Dict[str, Dict]:
    """Load rendered stimuli if available; otherwise return empty dict."""
    p = Path(data_dir)
    if not p.exists():
        return {}
    stimulus_data: Dict[str, Dict[str, List[str]]] = {}
    for file_path in p.glob("stimulus_*.png"):
        parts = file_path.stem.split('_')
        if len(parts) >= 4:
            category = parts[1]
            identity = f"{parts[1]}_{parts[2]}"
            stimulus_data.setdefault(category, {}).setdefault(identity, []).append(str(file_path))
    for category in stimulus_data:
        for identity in stimulus_data[category]:
            stimulus_data[category][identity].sort()
    return stimulus_data


def parse_task_features(names: List[str]) -> List[TaskFeature]:
    name_map = {
        "location": TaskFeature.LOCATION,
        "identity": TaskFeature.IDENTITY,
        "category": TaskFeature.CATEGORY,
    }
    return [name_map[n.lower()] for n in names]


# Deprecated: Use model_factory.create_model instead
# Kept for backward compatibility
def build_cognitive(rnn_type: str, input_size: int, hidden_size: int, num_layers: int, dropout: float):
    from .models import create_cognitive_module
    return create_cognitive_module(rnn_type, input_size, hidden_size, num_layers, dropout)


def accuracy_from_logits(logits: torch.Tensor, targets_idx: torch.Tensor) -> float:
    # logits: (B, T, 3), targets_idx: (B, T)
    preds = logits.argmax(dim=-1)
    correct = (preds == targets_idx).float().sum().item()
    total = targets_idx.numel()
    return correct / max(total, 1)


def save_hidden_states(hidden_seq: torch.Tensor, batch: Dict[str, torch.Tensor], logits: torch.Tensor, out_dir: Path, epoch: int, batch_idx: int):
    out_dir.mkdir(parents=True, exist_ok=True)
    # Derive additional fields for analyses
    task_index = batch["task_vector"].argmax(dim=-1).cpu()  # (B,)
    payload = {
        "hidden": hidden_seq.cpu(),              # (B, T, H)
        "logits": logits.cpu(),                  # (B, T, 3)
        "task_vector": batch["task_vector"].cpu(),  # (B, 3)
        "task_index": task_index,                # (B,)
        "n": batch["n"].cpu(),                  # (B,)
        "targets": batch["responses"].argmax(dim=-1).cpu(),  # (B, T)
        # Object properties per timestep
        "locations": batch.get("locations"),     # (B, T) LongTensor
        "categories": batch.get("categories"),   # List[List[str]]
        "identities": batch.get("identities"),   # List[List[str]]
    }
    torch.save(payload, out_dir / f"epoch{epoch:03d}_batch{batch_idx:04d}.pt")


def get_default_config() -> Dict[str, Any]:
    return {
        "experiment_name": "wm_default",
        "output_dir": "runs",
        "seed": 42,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        # Data
        "use_real_stimuli": True,
        "n_values": [2],
        "task_features": ["location"],
        "sequence_length": 6,
        "batch_size": 8,
        "num_train": 200,
        "num_val": 50,
        "num_test": 50,
        "num_workers": 2,
        # Model
        "hidden_size": 512,
        "model_type": "gru",  # gru|lstm|rnn|attention_gru|attention_lstm|attention_rnn
        "num_layers": 1,
        "dropout": 0.0,
        "attention_hidden_dim": None,  # For attention models
        "attention_dropout": 0.1,      # For attention models
        "pretrained_backbone": True,
        "freeze_backbone": True,
        "capture_exact_layer42_relu": True,
        "classifier_layers": [],
        # Optim
        "epochs": 10,
        "lr": 3e-4,
        "weight_decay": 1e-2,
        "milestones": [6, 8],
        "gamma": 0.1,
        "grad_clip": 1.0,
        # Validation hooks
        "save_hidden": True,
        # Scenario label (for bookkeeping only)
        "scenario": "STSF",
    }


def load_config(path: Optional[str]) -> Dict[str, Any]:
    cfg = get_default_config()
    if path is None:
        return cfg
    if yaml is None:
        print("PyYAML not available; using default config.")
        return cfg
    with open(path, "r") as f:
        user = yaml.safe_load(f)
    if user:
        cfg.update(user)
    return cfg


def main():
    parser = argparse.ArgumentParser(description="Train Working Memory model")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config")
    parser.add_argument("--save_hidden", action="store_true", help="Force save hidden states during validation")
    parser.add_argument("--no_save_hidden", action="store_true", help="Disable saving hidden states during validation")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.save_hidden:
        cfg["save_hidden"] = True
    if args.no_save_hidden:
        cfg["save_hidden"] = False

    # Seeding
    torch.manual_seed(cfg["seed"]) 
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg["seed"]) 

    device = torch.device(cfg["device"]) 

    # Data
    if cfg["use_real_stimuli"]:
        stimulus_data = load_real_stimulus_data()
        if not stimulus_data:
            print("No real stimuli found; falling back to sample data (images will be black placeholders).")
            stimulus_data = create_sample_stimulus_data()
    else:
        stimulus_data = create_sample_stimulus_data()

    data_module = NBackDataModule(
        stimulus_data=stimulus_data,
        n_values=cfg["n_values"],
        task_features=parse_task_features(cfg["task_features"]),
        sequence_length=cfg["sequence_length"],
        batch_size=cfg["batch_size"],
        num_train=cfg["num_train"],
        num_val=cfg["num_val"],
        num_test=cfg["num_test"],
        num_workers=cfg["num_workers"],
    )

    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()

    # Model - using model factory for flexibility
    H = cfg["hidden_size"]
    model = create_model(
        model_type=cfg["model_type"],
        hidden_size=H,
        num_layers=cfg["num_layers"],
        dropout=cfg["dropout"],
        pretrained_backbone=cfg["pretrained_backbone"],
        freeze_backbone=cfg["freeze_backbone"],
        capture_exact_layer42_relu=cfg.get("capture_exact_layer42_relu", True),
        attention_hidden_dim=cfg.get("attention_hidden_dim"),
        attention_dropout=cfg.get("attention_dropout", 0.1),
        classifier_layers=cfg.get("classifier_layers"),
    )
    model.to(device)
    
    # Print model summary
    print_model_summary(model)
    model_info = get_model_info(model)
    print(f"\nModel configuration:")
    print(f"  Type: {cfg['model_type']}")
    print(f"  Is Attention: {model_info['is_attention']}")
    print(f"  Hidden Size: {H}")
    print(f"  Num Layers: {cfg['num_layers']}")

    # Optimizer & Scheduler
    optimizer = optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]) 
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg["milestones"], gamma=cfg["gamma"]) 

    criterion = nn.CrossEntropyLoss()

    # Output dirs
    run_dir = Path(cfg["output_dir"]) / cfg["experiment_name"]
    ckpt_dir = run_dir / "checkpoints"
    hidden_dir = run_dir / "hidden_states"
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_val_acc = -1.0
    training_log = []  # Track metrics for analysis

    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        epoch_loss = 0.0
        epoch_acc = 0.0
        n_batches = 0
        for batch_idx, batch in enumerate(train_loader):
            images = batch["images"].to(device)               # (B, T, 3, H, W)
            task_vec = batch["task_vector"].to(device)        # (B, 3)
            targets_oh = batch["responses"].to(device)        # (B, T, 3)
            targets_idx = targets_oh.argmax(dim=-1)           # (B, T)

            optimizer.zero_grad(set_to_none=True)
            logits, hidden_seq, _ = model(images, task_vec)   # logits: (B, T, 3)
            loss = criterion(logits.reshape(-1, 3), targets_idx.reshape(-1))
            loss.backward()
            if cfg["grad_clip"] and cfg["grad_clip"] > 0:
                nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"]) 
            optimizer.step()

            with torch.no_grad():
                acc = accuracy_from_logits(logits, targets_idx)
            epoch_loss += loss.item()
            epoch_acc += acc
            n_batches += 1

            if (batch_idx + 1) % 20 == 0:
                print(f"Epoch {epoch} [{batch_idx+1}/{len(train_loader)}] loss={epoch_loss/n_batches:.4f} acc={epoch_acc/n_batches:.3f}")

        scheduler.step()
        print(f"Epoch {epoch} TRAIN loss={epoch_loss/max(n_batches,1):.4f} acc={epoch_acc/max(n_batches,1):.3f}")

        # Validation
        model.eval()
        val_loss = 0.0
        val_acc = 0.0
        val_batches = 0
        with torch.no_grad():
            for b_idx, batch in enumerate(val_loader):
                images = batch["images"].to(device)
                task_vec = batch["task_vector"].to(device)
                targets_oh = batch["responses"].to(device)
                targets_idx = targets_oh.argmax(dim=-1)

                logits, hidden_seq, _ = model(images, task_vec)
                loss = criterion(logits.reshape(-1, 3), targets_idx.reshape(-1))
                acc = accuracy_from_logits(logits, targets_idx)
                val_loss += loss.item()
                val_acc += acc
                val_batches += 1

                if cfg["save_hidden"]:
                    save_hidden_states(hidden_seq, batch, logits, hidden_dir / f"epoch_{epoch:03d}", epoch, b_idx)

        val_loss /= max(val_batches, 1)
        val_acc /= max(val_batches, 1)
        print(f"Epoch {epoch} VAL   loss={val_loss:.4f} acc={val_acc:.3f}")
        
        # Log metrics for analysis pipeline
        training_log.append({
            "epoch": epoch,
            "train_loss": epoch_loss / max(n_batches, 1),
            "train_acc": epoch_acc / max(n_batches, 1),
            "val_loss": val_loss,
            "val_acc": val_acc,
        })

        # Save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_path = ckpt_dir / f"best_epoch{epoch:03d}_acc{val_acc:.3f}.pt"
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "config": cfg,
                "val_acc": val_acc,
            }, best_path)
            print(f"Saved new best checkpoint to {best_path}")

    # Final save
    final_path = ckpt_dir / f"final_epoch{cfg['epochs']:03d}.pt"
    torch.save({
        "epoch": cfg["epochs"],
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "config": cfg,
        "val_acc": best_val_acc,
    }, final_path)
    print(f"Training complete. Final model saved to {final_path}")
    
    # Save training log for analysis pipeline
    log_path = run_dir / "training_log.json"
    with open(log_path, 'w') as f:
        json.dump(training_log, f, indent=2)
    print(f"Training log saved to {log_path}")


if __name__ == "__main__":
    main()
