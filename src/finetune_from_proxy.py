#!/usr/bin/env python3
"""
Fine-tuning from Proxy-Pretrained Model.

Loads a model that was pre-trained on the proxy task (feature recall),
transfers the learned perceptual + attention + cognitive weights to a
standard N-back model (with 3-class classifier), and fine-tunes on the
real N-back match/non-match task.

The resulting model has the same architecture as models trained from
scratch, enabling direct comparison.

Usage:
    python -m src.finetune_from_proxy \\
        --proxy_exp_dir experiments/proxy_mtmf_20250101_120000 \\
        --config configs/mtmf.yaml
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

from .data.validation_splits import load_and_split_stimuli
from .data.dataset import NBackDataModule
from .data.nback_generator import TaskFeature
from .models import create_model, print_model_summary
from .models.proxy_model import ProxyWorkingMemoryModel
from .utils.visualization import save_training_sample
from .utils.logger import get_logger, log_to_file_only

try:
    import yaml
except Exception:
    yaml = None


def parse_task_features(names: list) -> list:
    name_map = {
        "location": TaskFeature.LOCATION,
        "identity": TaskFeature.IDENTITY,
        "category": TaskFeature.CATEGORY,
    }
    return [name_map[n.lower()] for n in names]


def accuracy_from_logits(logits, targets_idx):
    preds = logits.argmax(dim=-1)
    correct = (preds == targets_idx).float().sum().item()
    total = targets_idx.numel()
    return correct / max(total, 1)


def per_class_metrics(logits, targets_idx):
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


def build_mask(n_vals, T, device):
    t_idx = torch.arange(T, device=device).unsqueeze(0).expand(n_vals.shape[0], T)
    return t_idx >= n_vals.unsqueeze(1)


def confusion_matrix_from_logits(logits, targets_idx, num_classes=3):
    preds = logits.argmax(dim=-1).flatten()
    t = targets_idx.flatten()
    cm = torch.zeros(num_classes, num_classes, dtype=torch.long)
    for i in range(num_classes):
        for j in range(num_classes):
            cm[i, j] = (((t == i) & (preds == j)).long()).sum().item()
    return cm


def save_states_and_activations(hidden_seq, cnn_activations, batch, logits,
                                 save_dir, epoch, batch_idx, split_name):
    split_dir = save_dir / f"epoch_{epoch:03d}" / split_name
    split_dir.mkdir(parents=True, exist_ok=True)

    task_index = batch["task_vector"].argmax(dim=-1).cpu()

    payload = {
        "hidden": hidden_seq.cpu(),
        "cnn_activations": cnn_activations.cpu() if cnn_activations is not None else None,
        "logits": logits.cpu(),
        "task_vector": batch["task_vector"].cpu(),
        "task_index": task_index,
        "n": batch["n"].cpu(),
        "targets": batch["responses"].argmax(dim=-1).cpu(),
        "locations": batch.get("locations"),
        "categories": batch.get("categories"),
        "identities": batch.get("identities"),
        "split": split_name,
    }

    filename = split_dir / f"batch_{batch_idx:04d}.pt"
    torch.save(payload, filename)


def transfer_proxy_weights(proxy_checkpoint: Dict, standard_model: nn.Module,
                            device: torch.device, log_fn=print) -> Dict[str, Any]:
    proxy_state = proxy_checkpoint.get("model_state_dict", proxy_checkpoint)

    transferred = {}
    skipped = []

    transfer_prefixes = ["perceptual", "cognitive"]
    if hasattr(standard_model, "attention"):
        transfer_prefixes.append("attention")

    for key, value in proxy_state.items():
        should_transfer = any(key.startswith(p) for p in transfer_prefixes)
        if should_transfer and key in standard_model.state_dict():
            transferred[key] = value

    standard_state = standard_model.state_dict()
    for key in transferred:
        if key in standard_state:
            if standard_state[key].shape == transferred[key].shape:
                standard_state[key] = transferred[key]
            else:
                skipped.append(f"{key} (shape mismatch: {transferred[key].shape} vs {standard_state[key].shape})")
        else:
            skipped.append(f"{key} (not in standard model)")

    standard_model.load_state_dict(standard_state)

    classifier_keys = [k for k in standard_model.state_dict() if "classifier" in k]
    new_classifier = {k: standard_state[k] for k in classifier_keys}

    info = {
        "transferred": len(transferred),
        "skipped": len(skipped),
        "skipped_details": skipped,
        "classifier_initialized_fresh": len(classifier_keys),
    }

    log_fn(f"\nWeight Transfer Summary:")
    log_fn(f"  Transferred: {info['transferred']} parameters")
    log_fn(f"  Skipped: {info['skipped']} parameters")
    log_fn(f"  Classifier (fresh init): {info['classifier_initialized_fresh']} parameters")
    if skipped:
        log_fn(f"  Skipped details: {skipped[:5]}...")

    return info


def evaluate_model(model, dataloader, criterion, device, save_dir=None, epoch=None,
                    split_name="val", save_activations=True):
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    num_batches = 0
    masked_correct_sum = 0.0
    masked_total_sum = 0.0
    noaction_correct_sum = 0.0
    noaction_total_sum = 0.0

    with torch.no_grad():
        val_pbar = tqdm(dataloader, desc=f"Validating {split_name}", leave=False, position=2)
        for batch_idx, batch in enumerate(val_pbar):
            images = batch["images"].to(device)
            task_vec = batch["task_vector"].to(device)
            targets_oh = batch["responses"].to(device)
            targets_idx = targets_oh.argmax(dim=-1)

            if save_dir is not None and save_activations:
                forward_out = model(images, task_vec, return_cnn_activations=True)
                logits, hidden_seq, _, cnn_activations = forward_out
            else:
                logits, hidden_seq, _ = model(images, task_vec)
                cnn_activations = None

            loss = criterion(logits.reshape(-1, 3), targets_idx.reshape(-1))
            acc = accuracy_from_logits(logits, targets_idx)
            B, T = targets_idx.shape
            n_vals = batch["n"].to(device).view(-1)
            mask = build_mask(n_vals, T, device)
            preds = logits.argmax(dim=-1)
            masked_correct_sum += ((preds == targets_idx) & mask).float().sum().item()
            masked_total_sum += mask.sum().item()
            mask_na = ~mask
            noaction_correct_sum += ((preds == targets_idx) & mask_na).float().sum().item()
            noaction_total_sum += mask_na.sum().item()

            total_loss += loss.item()
            total_acc += acc
            num_batches += 1

            val_pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{acc:.4f}'})

            if save_dir is not None and save_activations and epoch is not None:
                save_states_and_activations(
                    hidden_seq=hidden_seq, cnn_activations=cnn_activations,
                    batch=batch, logits=logits, save_dir=save_dir,
                    epoch=epoch, batch_idx=batch_idx, split_name=split_name
                )

    return {
        'loss': total_loss / max(num_batches, 1),
        'accuracy': total_acc / max(num_batches, 1),
        'accuracy_masked': (masked_correct_sum / max(masked_total_sum, 1)) if masked_total_sum > 0 else 0.0,
        'accuracy_no_action': (noaction_correct_sum / max(noaction_total_sum, 1)) if noaction_total_sum > 0 else 0.0,
    }


def main():
    parser = argparse.ArgumentParser(description="Fine-tune from Proxy-Pretrained Model")
    parser.add_argument("--proxy_exp_dir", type=str, required=True,
                        help="Path to proxy pre-training experiment directory")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to fine-tuning config YAML (standard experiment config)")
    parser.add_argument("--output_dir", type=str, default="experiments",
                        help="Output directory")
    parser.add_argument("--finetune_epochs", type=int, default=None,
                        help="Override epochs for fine-tuning (default: use config value)")
    parser.add_argument("--finetune_lr", type=float, default=None,
                        help="Override learning rate for fine-tuning")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    proxy_dir = Path(args.proxy_exp_dir)
    proxy_model_path = proxy_dir / "best_model.pt"
    proxy_config_path = proxy_dir / "config.yaml"

    if not proxy_model_path.exists():
        raise FileNotFoundError(f"Proxy model not found: {proxy_model_path}")

    log_fn_prefix = "finetune_proxy"
    exp_name = cfg.get("experiment_name", "wm_mtmf")
    exp_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir_name = f"{log_fn_prefix}_{exp_name}_{exp_id}"

    out_dir = Path(args.output_dir) / exp_dir_name
    out_dir.mkdir(parents=True, exist_ok=True)

    logger = get_logger(log_file=out_dir / "training.log")

    def log(msg: str) -> None:
        logger.info(msg)

    def log_epoch(msg: str) -> None:
        tqdm.write(msg)
        log_to_file_only(msg)

    log(f"Using device: {device}")
    log("\n" + "="*70)
    log(f"FINE-TUNING FROM PROXY-PRETRAINED MODEL")
    log(f"EXPERIMENT ID: {exp_id}")
    log(f"Proxy Model: {proxy_model_path}")
    log(f"Output Directory: {out_dir}")
    log("="*70)

    if cfg.get("save_hidden", True):
        hidden_dir = out_dir / "hidden_states"
        hidden_dir.mkdir(exist_ok=True)
    else:
        hidden_dir = None

    with open(out_dir / "config.yaml", 'w') as f:
        yaml.dump(cfg, f)

    metadata = {
        "experiment_id": exp_id,
        "experiment_name": exp_name,
        "experiment_type": "finetune_from_proxy",
        "proxy_exp_dir": str(proxy_dir),
        "config_file": args.config,
        "start_time": datetime.now().isoformat(),
        "device": str(device),
        "config": cfg,
    }
    with open(out_dir / "experiment_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    log("\n" + "="*70)
    log("PHASE 1: LOADING PROXY-PRETRAINED WEIGHTS")
    log("="*70)

    log("\n1. Loading proxy checkpoint...")
    proxy_checkpoint = torch.load(proxy_model_path, map_location=device)
    proxy_cfg = proxy_checkpoint.get("config", {})
    log(f"  Proxy experiment type: {proxy_checkpoint.get('experiment_type', 'unknown')}")
    log(f"  Proxy best epoch: {proxy_checkpoint.get('epoch', 'unknown')}")
    log(f"  Proxy val accuracy: {proxy_checkpoint.get('val_novel_angle_acc', 'unknown')}")

    log("\n2. Loading Stimuli and Creating Validation Splits...")
    train_data, val_novel_angle_data, val_novel_identity_data, split_stats = load_and_split_stimuli(
        stimuli_dir="data/stimuli",
        train_angles=[0, 1, 2],
        val_angles=[3],
        train_identity_ratio=0.6,
    )

    log("\nData Split Summary:")
    log(f"  Training:             {split_stats['training']['num_stimuli']} stimuli, "
        f"{split_stats['training']['num_identities']} identities")
    log(f"  Val (Novel Angles):   {split_stats['val_novel_angle']['num_stimuli']} stimuli, "
        f"{split_stats['val_novel_angle']['num_identities']} identities")
    log(f"  Val (Novel IDs):      {split_stats['val_novel_identity']['num_stimuli']} stimuli, "
        f"{split_stats['val_novel_identity']['num_identities']} identities")

    log("\n3. Creating Data Module...")
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
        match_probability=cfg.get("match_probability", 0.3),
        cache_train_sequences=cfg.get("cache_train_sequences", False),
        cache_val_sequences=cfg.get("cache_val_sequences", True),
    )

    train_loader = data_module.train_dataloader()
    val_novel_angle_loader = data_module.val_novel_angle_dataloader()
    val_novel_identity_loader = data_module.val_novel_identity_dataloader()

    log("\n4. Creating Standard Model and Transferring Proxy Weights...")
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
        attention_hidden_dim=cfg.get("attention_hidden_dim"),
        attention_dropout=cfg.get("attention_dropout", 0.1),
        attention_mode=cfg.get("attention_mode", "task_only"),
        classifier_layers=cfg.get("classifier_layers"),
    )

    transfer_info = transfer_proxy_weights(proxy_checkpoint, model, device, log)
    model = model.to(device)

    print_model_summary(model)

    log("\n" + "="*70)
    log("TRAINABLE PARAMETERS BREAKDOWN")
    log("="*70)

    perceptual_trainable = 0
    cognitive_trainable = 0
    classifier_trainable = 0
    attention_trainable = 0

    for name, param in model.perceptual.named_parameters():
        if param.requires_grad:
            perceptual_trainable += param.numel()

    for name, param in model.cognitive.named_parameters():
        if param.requires_grad:
            cognitive_trainable += param.numel()

    if hasattr(model, "attention"):
        for name, param in model.attention.named_parameters():
            if param.requires_grad:
                attention_trainable += param.numel()

    for name, param in model.classifier.named_parameters():
        if param.requires_grad:
            classifier_trainable += param.numel()

    total_trainable = perceptual_trainable + cognitive_trainable + attention_trainable + classifier_trainable
    log(f"Perceptual (trainable):  {perceptual_trainable:,}")
    log(f"Cognitive (trainable):   {cognitive_trainable:,}")
    log(f"Attention (trainable):   {attention_trainable:,}")
    log(f"Classifier (trainable):  {classifier_trainable:,}")
    log(f"TOTAL TRAINABLE:         {total_trainable:,}")
    log("="*70 + "\n")

    fine_tune_lr = args.finetune_lr if args.finetune_lr is not None else cfg["lr"]
    optimizer = optim.AdamW(model.parameters(), lr=fine_tune_lr, weight_decay=cfg["weight_decay"])
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=cfg["milestones"], gamma=cfg["gamma"]
    )

    log("\nCalculating class weights from training data (t>=n)...")
    class_counts = torch.zeros(3, device=device)
    for batch in train_loader:
        targets = batch["responses"].argmax(dim=-1).to(device)
        n_vals = batch["n"].to(device).view(-1)
        B, T = targets.shape
        mask = build_mask(n_vals, T, device)
        masked_targets = targets[mask]
        binc = torch.bincount(masked_targets, minlength=3).float()
        class_counts += binc
    total_samples = class_counts.sum()
    present = class_counts > 0
    K = int(present.sum().item()) if present.any() else 1
    class_weights = torch.zeros(3, device=device)
    for c in range(3):
        if class_counts[c] > 0:
            class_weights[c] = total_samples / (K * class_counts[c])
        else:
            class_weights[c] = 0.0
    log(f"Class distribution (t>=n): No_Action={class_counts[0].item():.0f}, Non_Match={class_counts[1].item():.0f}, Match={class_counts[2].item():.0f}")
    log(f"Calculated class weights (t>=n): {[f'{w:.3f}' for w in class_weights.cpu().tolist()]}")

    label_smoothing = cfg.get("label_smoothing", 0.0)
    criterion_main = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
    criterion_no_action = nn.CrossEntropyLoss(weight=None, label_smoothing=label_smoothing)
    no_action_loss_weight = cfg.get("no_action_loss_weight", 0.1)

    log("\n" + "="*70)
    log("PHASE 2: FINE-TUNING ON REAL N-BACK TASK")
    log("="*70)

    fine_tune_epochs = args.finetune_epochs if args.finetune_epochs is not None else cfg["epochs"]
    best_val_novel_angle_acc = 0.0
    results_log = []

    mask_trivial_steps = cfg.get("mask_trivial_steps", True)
    for epoch in tqdm(range(fine_tune_epochs), desc="Fine-tuning Epochs", position=0):
        model.train()
        train_loss = 0.0
        train_acc = 0.0
        num_train_batches = 0

        all_train_logits_m = []
        all_train_targets_m = []
        all_train_logits_na = []
        all_train_targets_na = []

        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{fine_tune_epochs}", position=1, leave=False)
        for batch in train_pbar:
            images = batch["images"].to(device)
            task_vec = batch["task_vector"].to(device)
            targets_oh = batch["responses"].to(device)
            targets_idx = targets_oh.argmax(dim=-1)

            optimizer.zero_grad()
            logits, _, _ = model(images, task_vec)
            if mask_trivial_steps:
                B, T = targets_idx.shape
                n_vals = batch["n"].to(device).view(-1)
                mask = build_mask(n_vals, T, device)
                logits_flat = logits.reshape(-1, 3)
                targets_flat = targets_idx.reshape(-1)
                loss_main = criterion_main(logits_flat[mask.reshape(-1)], targets_flat[mask.reshape(-1)])
                mask_na = (~mask).reshape(-1)
                if mask_na.any() and no_action_loss_weight > 0:
                    loss_na = criterion_no_action(logits_flat[mask_na], targets_flat[mask_na])
                    loss = loss_main + no_action_loss_weight * loss_na
                else:
                    loss = loss_main
            else:
                loss = criterion_main(logits.reshape(-1, 3), targets_idx.reshape(-1))
            loss.backward()

            if cfg.get("grad_clip"):
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])

            optimizer.step()

            acc = accuracy_from_logits(logits, targets_idx)
            train_loss += loss.item()
            train_acc += acc
            num_train_batches += 1

            if mask_trivial_steps:
                all_train_logits_m.append(logits_flat[mask.reshape(-1)].detach().cpu())
                all_train_targets_m.append(targets_flat[mask.reshape(-1)].cpu())
                if mask_na.any():
                    all_train_logits_na.append(logits_flat[mask_na].detach().cpu())
                    all_train_targets_na.append(targets_flat[mask_na].cpu())
            else:
                all_train_logits_m.append(logits.detach().cpu().reshape(-1, 3))
                all_train_targets_m.append(targets_idx.cpu().reshape(-1))

            train_pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{acc:.4f}'})

        train_loss /= max(num_train_batches, 1)
        train_acc /= max(num_train_batches, 1)

        train_logits_all_m = torch.cat(all_train_logits_m, dim=0)
        train_targets_all_m = torch.cat(all_train_targets_m, dim=0)
        train_class_metrics = per_class_metrics(train_logits_all_m, train_targets_all_m)
        cm_train = confusion_matrix_from_logits(train_logits_all_m, train_targets_all_m)
        train_masked_acc = (train_logits_all_m.argmax(dim=-1) == train_targets_all_m).float().mean().item()
        if len(all_train_logits_na) > 0:
            train_logits_all_na = torch.cat(all_train_logits_na, dim=0)
            train_targets_all_na = torch.cat(all_train_targets_na, dim=0)
            train_no_action_acc = (train_logits_all_na.argmax(dim=-1) == train_targets_all_na).float().mean().item()
            train_no_action_count = train_targets_all_na.numel()
        else:
            train_no_action_acc = 0.0
            train_no_action_count = 0

        val_novel_angle_results = evaluate_model(
            model, val_novel_angle_loader, criterion_main, device,
            save_dir=hidden_dir, epoch=epoch+1, split_name="val_novel_angle",
            save_activations=cfg.get("save_hidden", True)
        )

        val_novel_identity_results = evaluate_model(
            model, val_novel_identity_loader, criterion_main, device,
            save_dir=hidden_dir, epoch=epoch+1, split_name="val_novel_identity",
            save_activations=cfg.get("save_hidden", True)
        )

        epoch_results = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'train_masked_acc': train_masked_acc,
            'train_no_action_acc': train_no_action_acc,
            'train_per_class': train_class_metrics,
            'train_confusion_matrix': cm_train.tolist(),
            'val_novel_angle_loss': val_novel_angle_results['loss'],
            'val_novel_angle_acc': val_novel_angle_results['accuracy'],
            'val_novel_angle_acc_masked': val_novel_angle_results['accuracy_masked'],
            'val_novel_angle_acc_no_action': val_novel_angle_results['accuracy_no_action'],
            'val_novel_identity_loss': val_novel_identity_results['loss'],
            'val_novel_identity_acc': val_novel_identity_results['accuracy'],
            'val_novel_identity_acc_masked': val_novel_identity_results['accuracy_masked'],
            'val_novel_identity_acc_no_action': val_novel_identity_results['accuracy_no_action'],
            'lr': optimizer.param_groups[0]['lr'],
        }
        results_log.append(epoch_results)

        log_epoch(
            f"[Epoch {epoch+1:03d}/{fine_tune_epochs}] "
            f"lr={optimizer.param_groups[0]['lr']:.2e} | "
            f"train: loss={train_loss:.4f}, acc={train_acc:.3f}, masked={train_masked_acc:.3f}, no_act={train_no_action_acc:.3f} | "
            f"val_angle: loss={val_novel_angle_results['loss']:.4f}, acc={val_novel_angle_results['accuracy']:.3f}, "
            f"masked={val_novel_angle_results['accuracy_masked']:.3f}, no_act={val_novel_angle_results['accuracy_no_action']:.3f} | "
            f"val_id: loss={val_novel_identity_results['loss']:.4f}, acc={val_novel_identity_results['accuracy']:.3f}, "
            f"masked={val_novel_identity_results['accuracy_masked']:.3f}, no_act={val_novel_identity_results['accuracy_no_action']:.3f}"
        )

        log_epoch(
            "  train per-class (t>=n) acc: "
            f"NA={train_class_metrics['No_Action_acc']:.3f} ({train_class_metrics['No_Action_count']}), "
            f"NM={train_class_metrics['Non_Match_acc']:.3f} ({train_class_metrics['Non_Match_count']}), "
            f"M={train_class_metrics['Match_acc']:.3f} ({train_class_metrics['Match_count']})"
        )

        if cfg.get("save_visualizations", True):
            vis_dir = out_dir / "visualizations"
            num_vis = cfg.get("num_visualizations", 1)
            task_names = cfg["task_features"]

            def save_task_visualizations(data_loader, split_name):
                task_samples = {t: [] for t in task_names}
                for batch in data_loader:
                    task_idx = batch["task_vector"].argmax(dim=-1)
                    for b in range(len(task_idx)):
                        t_idx = task_idx[b].item()
                        if t_idx < len(task_names):
                            t_name = task_names[t_idx]
                            if len(task_samples[t_name]) < num_vis:
                                task_samples[t_name].append((batch, b))
                    if all(len(v) >= num_vis for v in task_samples.values()):
                        break
                for t_name, samples in task_samples.items():
                    for vis_idx, (batch, sample_idx) in enumerate(samples[:num_vis]):
                        save_training_sample(
                            model=model, batch=batch, device=device,
                            save_dir=vis_dir, epoch=epoch+1, batch_idx=vis_idx,
                            split_name=f"{split_name}_{t_name}", sample_idx=sample_idx
                        )

            save_task_visualizations(train_loader, "train")
            save_task_visualizations(val_novel_angle_loader, "val_novel_angle")
            save_task_visualizations(val_novel_identity_loader, "val_novel_identity")

            if epoch == 0:
                log_epoch(f"  Saved sequence visualizations to: {vis_dir}/")

        if val_novel_angle_results['accuracy'] > best_val_novel_angle_acc:
            best_val_novel_angle_acc = val_novel_angle_results['accuracy']
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_novel_angle_acc': val_novel_angle_results['accuracy'],
                'val_novel_identity_acc': val_novel_identity_results['accuracy'],
                'config': cfg,
                'pretrained_from': str(proxy_dir),
                'experiment_type': 'finetune_from_proxy',
                'transfer_info': transfer_info,
            }, out_dir / "best_model.pt")
            log_epoch(f"  Saved best model (val_novel_angle_acc={val_novel_angle_results['accuracy']:.4f})")

        scheduler.step()

    with open(out_dir / "training_log.json", 'w') as f:
        json.dump(results_log, f, indent=2)

    metadata["end_time"] = datetime.now().isoformat()
    metadata["best_val_novel_angle_acc"] = best_val_novel_angle_acc
    metadata["total_epochs"] = fine_tune_epochs
    metadata["transfer_info"] = transfer_info
    with open(out_dir / "experiment_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    log("\n" + "="*70)
    log("FINE-TUNING COMPLETED")
    log("="*70)
    log(f"\nExperiment ID: {exp_id}")
    log(f"Proxy Pre-trained Model: {proxy_dir}")
    log(f"Best Val (Novel Angle) Accuracy: {best_val_novel_angle_acc:.4f}")
    log(f"\nResults saved to: {out_dir}")
    log(f"\nKey Outputs:")
    log(f"  - Experiment Metadata: {out_dir / 'experiment_metadata.json'}")
    log(f"  - Model: {out_dir / 'best_model.pt'}")
    log(f"  - Training Log: {out_dir / 'training_log.json'}")
    log(f"  - Config: {out_dir / 'config.yaml'}")


if __name__ == "__main__":
    main()
