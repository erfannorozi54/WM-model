#!/usr/bin/env python3
"""
Training script with proper generalization evaluation.

Implements:
1. Training on standard data (angles 0,1,2 | identities 0-2)
2. Validation on Novel Angles (angle 3 | same identities 0-2)
3. Validation on Novel Identities (all angles | new identities 3-4)

This matches the methodology described in the paper for testing generalization.
"""

import os
from pathlib import Path
import argparse
import time
import json
from typing import Dict, Any
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# Import from same package
from .data.validation_splits import load_and_split_stimuli
from .data.dataset import NBackDataModule
from .data.nback_generator import TaskFeature
from .models import create_model, print_model_summary
from .utils.visualization import save_training_sample

try:
    import yaml
except Exception:
    yaml = None


def parse_task_features(names: list) -> list:
    """Parse task feature names to TaskFeature enum."""
    name_map = {
        "location": TaskFeature.LOCATION,
        "identity": TaskFeature.IDENTITY,
        "category": TaskFeature.CATEGORY,
    }
    return [name_map[n.lower()] for n in names]


def accuracy_from_logits(logits: torch.Tensor, targets_idx: torch.Tensor) -> float:
    """Calculate accuracy from logits and target indices."""
    preds = logits.argmax(dim=-1)
    correct = (preds == targets_idx).float().sum().item()
    total = targets_idx.numel()
    return correct / max(total, 1)


def per_class_metrics(logits: torch.Tensor, targets_idx: torch.Tensor) -> Dict[str, float]:
    """Calculate per-class accuracy and counts."""
    preds = logits.argmax(dim=-1).flatten()
    targets = targets_idx.flatten()
    
    class_names = ["No_Action", "Non_Match", "Match"]
    metrics = {}
    
    for class_idx, class_name in enumerate(class_names):
        mask = targets == class_idx
        if mask.sum() > 0:
            class_correct = ((preds == targets) & mask).float().sum().item()
            class_total = mask.sum().item()
            metrics[f"{class_name}_acc"] = class_correct / class_total
            metrics[f"{class_name}_count"] = class_total
        else:
            metrics[f"{class_name}_acc"] = 0.0
            metrics[f"{class_name}_count"] = 0
    
    return metrics


def save_states_and_activations(
    hidden_seq: torch.Tensor,
    cnn_activations: torch.Tensor,
    batch: Dict,
    logits: torch.Tensor,
    save_dir: Path,
    epoch: int,
    batch_idx: int,
    split_name: str
):
    """
    Save RNN hidden states and CNN activations for analysis.
    
    Args:
        hidden_seq: RNN hidden states (B, T, H)
        cnn_activations: CNN penultimate layer activations (B, T, H)
        batch: Batch dictionary with metadata
        logits: Model output logits (B, T, 3)
        save_dir: Base directory for saving
        epoch: Current epoch number
        batch_idx: Batch index
        split_name: Name of data split (e.g., "val_novel_angle")
    """
    # Create split-specific subdirectory
    split_dir = save_dir / f"epoch_{epoch:03d}" / split_name
    split_dir.mkdir(parents=True, exist_ok=True)
    
    # Derive task index
    task_index = batch["task_vector"].argmax(dim=-1).cpu()  # (B,)
    
    # Prepare payload
    payload = {
        "hidden": hidden_seq.cpu(),                      # (B, T, H) - RNN encoding space
        "cnn_activations": cnn_activations.cpu() if cnn_activations is not None else None,  # (B, T, H) - CNN perceptual space
        "logits": logits.cpu(),                          # (B, T, 3)
        "task_vector": batch["task_vector"].cpu(),       # (B, 3)
        "task_index": task_index,                        # (B,)
        "n": batch["n"].cpu(),                           # (B,)
        "targets": batch["responses"].argmax(dim=-1).cpu(),  # (B, T)
        # Object properties per timestep
        "locations": batch.get("locations"),             # (B, T) LongTensor
        "categories": batch.get("categories"),           # List[List[str]]
        "identities": batch.get("identities"),           # List[List[str]]
        "split": split_name,                             # Track which validation split
    }
    
    # Save to file
    filename = split_dir / f"batch_{batch_idx:04d}.pt"
    torch.save(payload, filename)


def evaluate_model(model, dataloader, criterion, device, save_dir=None, epoch=None, split_name="val", save_activations=True):
    """
    Evaluate model on a given dataloader.
    
    Args:
        model: Model to evaluate
        dataloader: Data to evaluate on
        criterion: Loss function
        device: Device to use
        save_dir: Directory to save hidden states and CNN activations (optional)
        epoch: Current epoch number (for saved filename)
        split_name: Name of split (e.g., "val_novel_angle", "val_novel_identity")
        save_activations: Whether to save hidden states and CNN activations
    
    Returns:
        Dictionary with loss and accuracy
    """
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    num_batches = 0
    
    with torch.no_grad():
        val_pbar = tqdm(dataloader, desc=f"Validating {split_name}", leave=False, position=2)
        for batch_idx, batch in enumerate(val_pbar):
            images = batch["images"].to(device)
            task_vec = batch["task_vector"].to(device)
            targets_oh = batch["responses"].to(device)
            targets_idx = targets_oh.argmax(dim=-1)
            
            # Forward pass with optional CNN activation capture
            if save_dir is not None and save_activations:
                forward_out = model(images, task_vec, return_cnn_activations=True)
                logits, hidden_seq, _, cnn_activations = forward_out
            else:
                logits, hidden_seq, _ = model(images, task_vec)
                cnn_activations = None
            
            loss = criterion(logits.reshape(-1, 3), targets_idx.reshape(-1))
            acc = accuracy_from_logits(logits, targets_idx)
            
            total_loss += loss.item()
            total_acc += acc
            num_batches += 1
            
            # Update validation progress bar
            val_pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{acc:.4f}'
            })
            
            # Save hidden states and CNN activations
            if save_dir is not None and save_activations and epoch is not None:
                save_states_and_activations(
                    hidden_seq=hidden_seq,
                    cnn_activations=cnn_activations,
                    batch=batch,
                    logits=logits,
                    save_dir=save_dir,
                    epoch=epoch,
                    batch_idx=batch_idx,
                    split_name=split_name
                )
    
    return {
        'loss': total_loss / max(num_batches, 1),
        'accuracy': total_acc / max(num_batches, 1)
    }


def main():
    parser = argparse.ArgumentParser(description="Train Working Memory Model with Generalization Evaluation")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--output_dir", type=str, default="experiments", help="Output directory")
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory with unique experiment ID
    exp_name = cfg.get("experiment_name", "wm_experiment")
    
    # Generate unique experiment ID (timestamp)
    exp_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir_name = f"{exp_name}_{exp_id}"
    
    out_dir = Path(args.output_dir) / exp_dir_name
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"EXPERIMENT ID: {exp_id}")
    print(f"Output Directory: {out_dir}")
    print(f"{'='*70}")
    
    # Create hidden states directory if save_hidden is enabled
    if cfg.get("save_hidden", True):
        hidden_dir = out_dir / "hidden_states"
        hidden_dir.mkdir(exist_ok=True)
    else:
        hidden_dir = None
    
    # Save config and experiment metadata
    with open(out_dir / "config.yaml", 'w') as f:
        yaml.dump(cfg, f)
    
    # Save experiment metadata
    metadata = {
        "experiment_id": exp_id,
        "experiment_name": exp_name,
        "config_file": args.config,
        "start_time": datetime.now().isoformat(),
        "device": str(device),
        "config": cfg
    }
    with open(out_dir / "experiment_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("\n" + "="*70)
    print("PHASE 6: TRAINING WITH GENERALIZATION EVALUATION")
    print("="*70)
    
    # Load and split data
    print("\n1. Loading Stimuli and Creating Validation Splits...")
    train_data, val_novel_angle_data, val_novel_identity_data, split_stats = load_and_split_stimuli(
        stimuli_dir="data/stimuli",
        train_angles=[0, 1, 2],  # Training uses angles 0, 1, 2
        val_angles=[3],           # Novel-angle validation uses angle 3
        train_identities_per_category=3,  # 3 identities for training
        val_identities_per_category=2     # 2 identities for novel-identity validation
    )
    
    print("\nData Split Summary:")
    print(f"  Training:             {split_stats['training']['num_stimuli']} stimuli, "
          f"{split_stats['training']['num_identities']} identities")
    print(f"  Val (Novel Angles):   {split_stats['val_novel_angle']['num_stimuli']} stimuli, "
          f"{split_stats['val_novel_angle']['num_identities']} identities")
    print(f"  Val (Novel IDs):      {split_stats['val_novel_identity']['num_stimuli']} stimuli, "
          f"{split_stats['val_novel_identity']['num_identities']} identities")
    
    # Create data module with three splits
    print("\n2. Creating Data Module...")
    data_module = NBackDataModule(
        stimulus_data=train_data,
        val_novel_angle_data=val_novel_angle_data,
        val_novel_identity_data=val_novel_identity_data,
        n_values=cfg["n_values"],
        task_features=parse_task_features(cfg["task_features"]),
        sequence_length=cfg["sequence_length"],
        batch_size=cfg["batch_size"],
        num_train=cfg["num_train"],
        num_val_novel_angle=cfg.get("num_val_novel_angle", cfg["num_val"]),
        num_val_novel_identity=cfg.get("num_val_novel_identity", cfg["num_val"]),
        num_workers=cfg["num_workers"],
    )
    
    train_loader = data_module.train_dataloader()
    val_novel_angle_loader = data_module.val_novel_angle_dataloader()
    val_novel_identity_loader = data_module.val_novel_identity_dataloader()
    
    # Create model
    print("\n3. Creating Model...")
    # Construct model_type string (e.g., "gru" or "attention_gru")
    rnn_type = cfg.get("rnn_type", "gru")
    model_arch = cfg.get("model_type", "baseline")
    if model_arch == "attention":
        model_type_str = f"attention_{rnn_type}"
    else:
        model_type_str = rnn_type
    
    model = create_model(
        model_type=model_type_str,
        hidden_size=cfg["hidden_size"],
        num_layers=cfg["num_layers"],
        dropout=cfg.get("dropout", 0.0),
        pretrained_backbone=cfg.get("pretrained_backbone", True),
        freeze_backbone=cfg.get("freeze_backbone", True),
        classifier_layers=cfg.get("classifier_layers"),
    )
    model = model.to(device)
    
    print_model_summary(model)
    
    # Log detailed trainable parameters breakdown
    print("\n" + "="*70)
    print("TRAINABLE PARAMETERS BREAKDOWN")
    print("="*70)
    
    perceptual_trainable = 0
    cognitive_trainable = 0
    classifier_trainable = 0
    
    print("\nPerceptual Module (trainable layers):")
    for name, param in model.perceptual.named_parameters():
        if param.requires_grad:
            print(f"  {name}: {param.numel():,} parameters")
            perceptual_trainable += param.numel()
    
    print(f"\nCognitive Module (trainable layers):")
    for name, param in model.cognitive.named_parameters():
        if param.requires_grad:
            print(f"  {name}: {param.numel():,} parameters")
            cognitive_trainable += param.numel()
    
    print(f"\nClassifier (trainable layers):")
    for name, param in model.classifier.named_parameters():
        if param.requires_grad:
            print(f"  {name}: {param.numel():,} parameters")
            classifier_trainable += param.numel()
    
    print("\n" + "-"*70)
    print(f"Perceptual (trainable):  {perceptual_trainable:,}")
    print(f"Cognitive (trainable):   {cognitive_trainable:,}")
    print(f"Classifier (trainable):  {classifier_trainable:,}")
    print(f"TOTAL TRAINABLE:         {perceptual_trainable + cognitive_trainable + classifier_trainable:,}")
    print("="*70 + "\n")
    
    # Optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=cfg["milestones"], gamma=cfg["gamma"]
    )
    
    # Calculate balanced class weights from training data
    print("\nCalculating class weights from training data...")
    class_counts = torch.zeros(3)
    for batch in train_loader:
        targets = batch["responses"].argmax(dim=-1)
        for i in range(3):
            class_counts[i] += (targets == i).sum().item()
    
    # Inverse frequency weighting: weight = total / (n_classes * class_count)
    total_samples = class_counts.sum()
    class_weights = total_samples / (3 * class_counts)
    class_weights = class_weights.to(device)
    
    print(f"Class distribution: No_Action={class_counts[0]:.0f}, Non_Match={class_counts[1]:.0f}, Match={class_counts[2]:.0f}")
    print(f"Calculated class weights: {[f'{w:.3f}' for w in class_weights.cpu().tolist()]}")
    
    # Create loss with class weights and optional label smoothing
    label_smoothing = cfg.get("label_smoothing", 0.0)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
    if label_smoothing > 0:
        print(f"Using label smoothing: {label_smoothing}")
    
    # Training loop
    print("\n4. Training...")
    best_val_novel_angle_acc = 0.0
    results_log = []
    
    for epoch in tqdm(range(cfg["epochs"]), desc="Training Epochs", position=0):
        # Training
        model.train()
        train_loss = 0.0
        train_acc = 0.0
        num_train_batches = 0
        
        # Collect all predictions and targets for per-class metrics
        all_train_logits = []
        all_train_targets = []
        
        # Progress bar for batches
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg['epochs']}", position=1, leave=False)
        for batch in train_pbar:
            images = batch["images"].to(device)
            task_vec = batch["task_vector"].to(device)
            targets_oh = batch["responses"].to(device)
            targets_idx = targets_oh.argmax(dim=-1)
            
            optimizer.zero_grad()
            logits, _, _ = model(images, task_vec)
            loss = criterion(logits.reshape(-1, 3), targets_idx.reshape(-1))
            loss.backward()
            
            if cfg.get("grad_clip"):
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            
            optimizer.step()
            
            acc = accuracy_from_logits(logits, targets_idx)
            train_loss += loss.item()
            train_acc += acc
            num_train_batches += 1
            
            # Store for per-class metrics
            all_train_logits.append(logits.detach().cpu())
            all_train_targets.append(targets_idx.cpu())
            
            # Update progress bar with current metrics
            train_pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{acc:.4f}'
            })
        
        train_loss /= max(num_train_batches, 1)
        train_acc /= max(num_train_batches, 1)
        
        # Calculate per-class metrics for training
        train_logits_all = torch.cat(all_train_logits, dim=0)
        train_targets_all = torch.cat(all_train_targets, dim=0)
        train_class_metrics = per_class_metrics(train_logits_all, train_targets_all)
        
        # Validation on Novel Angles (with CNN activation saving)
        val_novel_angle_results = evaluate_model(
            model, val_novel_angle_loader, criterion, device,
            save_dir=hidden_dir, epoch=epoch+1, split_name="val_novel_angle",
            save_activations=cfg.get("save_hidden", True)
        )
        
        # Validation on Novel Identities (with CNN activation saving)
        val_novel_identity_results = evaluate_model(
            model, val_novel_identity_loader, criterion, device,
            save_dir=hidden_dir, epoch=epoch+1, split_name="val_novel_identity",
            save_activations=cfg.get("save_hidden", True)
        )
        
        # Log results
        epoch_results = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_novel_angle_loss': val_novel_angle_results['loss'],
            'val_novel_angle_acc': val_novel_angle_results['accuracy'],
            'val_novel_identity_loss': val_novel_identity_results['loss'],
            'val_novel_identity_acc': val_novel_identity_results['accuracy'],
            'lr': optimizer.param_groups[0]['lr']
        }
        results_log.append(epoch_results)
        
        # Print epoch summary
        tqdm.write(f"\nEpoch {epoch+1}/{cfg['epochs']} Summary:")
        tqdm.write(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        tqdm.write(f"  Val (Novel Angle) Loss: {val_novel_angle_results['loss']:.4f}, "
              f"Acc: {val_novel_angle_results['accuracy']:.4f}")
        tqdm.write(f"  Val (Novel Identity) Loss: {val_novel_identity_results['loss']:.4f}, "
              f"Acc: {val_novel_identity_results['accuracy']:.4f}")
        
        # Print per-class training accuracy
        tqdm.write(f"\n  Per-Class Training Accuracy:")
        tqdm.write(f"    No_Action:  {train_class_metrics['No_Action_acc']:.3f} ({train_class_metrics['No_Action_count']} samples)")
        tqdm.write(f"    Non_Match:  {train_class_metrics['Non_Match_acc']:.3f} ({train_class_metrics['Non_Match_count']} samples)")
        tqdm.write(f"    Match:      {train_class_metrics['Match_acc']:.3f} ({train_class_metrics['Match_count']} samples)")
        
        # Visualize sample sequences from all three datasets
        if cfg.get("save_visualizations", True):
            vis_dir = out_dir / "visualizations"
            
            # Save sample from training set
            train_batch = next(iter(train_loader))
            save_training_sample(
                model=model,
                batch=train_batch,
                device=device,
                save_dir=vis_dir,
                epoch=epoch + 1,
                batch_idx=0,
                split_name="train"
            )
            
            # Save sample from novel angle validation
            val_angle_batch = next(iter(val_novel_angle_loader))
            save_training_sample(
                model=model,
                batch=val_angle_batch,
                device=device,
                save_dir=vis_dir,
                epoch=epoch + 1,
                batch_idx=0,
                split_name="val_novel_angle"
            )
            
            # Save sample from novel identity validation
            val_identity_batch = next(iter(val_novel_identity_loader))
            save_training_sample(
                model=model,
                batch=val_identity_batch,
                device=device,
                save_dir=vis_dir,
                epoch=epoch + 1,
                batch_idx=0,
                split_name="val_novel_identity"
            )
            
            if epoch == 0:  # Only print once
                tqdm.write(f"  ✓ Saved sequence visualizations to: {vis_dir}/")
                tqdm.write(f"    - Training sample: epoch_XXX_train.png")
                tqdm.write(f"    - Novel Angle sample: epoch_XXX_val_novel_angle.png")
                tqdm.write(f"    - Novel Identity sample: epoch_XXX_val_novel_identity.png")
        
        # Save best model based on novel-angle validation
        if val_novel_angle_results['accuracy'] > best_val_novel_angle_acc:
            best_val_novel_angle_acc = val_novel_angle_results['accuracy']
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_novel_angle_acc': val_novel_angle_results['accuracy'],
                'val_novel_identity_acc': val_novel_identity_results['accuracy'],
                'config': cfg,
            }, out_dir / "best_model.pt")
            tqdm.write(f"  ✓ Saved best model (val_novel_angle_acc={val_novel_angle_results['accuracy']:.4f})")
        
        scheduler.step()
    
    # Save final results
    with open(out_dir / "training_log.json", 'w') as f:
        json.dump(results_log, f, indent=2)
    
    # Update metadata with final results
    metadata["end_time"] = datetime.now().isoformat()
    metadata["best_val_novel_angle_acc"] = best_val_novel_angle_acc
    metadata["total_epochs"] = cfg["epochs"]
    with open(out_dir / "experiment_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("\n" + "="*70)
    print("TRAINING COMPLETED")
    print("="*70)
    print(f"\nExperiment ID: {exp_id}")
    print(f"Best Val (Novel Angle) Accuracy: {best_val_novel_angle_acc:.4f}")
    print(f"\nResults saved to: {out_dir}")
    print("\nKey Outputs:")
    print(f"  - Experiment Metadata: {out_dir / 'experiment_metadata.json'}")
    print(f"  - Model: {out_dir / 'best_model.pt'}")
    print(f"  - Training Log: {out_dir / 'training_log.json'}")
    print(f"  - Config: {out_dir / 'config.yaml'}")
    

if __name__ == "__main__":
    main()
