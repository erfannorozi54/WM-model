#!/usr/bin/env python3
"""
Proxy Task Pre-training for Working Memory Models.

Trains the model on a proxy task (feature recall N-back) instead of the
standard match/non-match classification. The proxy task asks the model
to predict the feature value from N steps back, which provides a richer
training signal and builds foundational WM skills.

After proxy pre-training, the model can be fine-tuned on the real N-back
task using finetune_from_proxy.py.

Usage:
    python -m src.train_proxy --config configs/proxy/proxy_mtmf.yaml
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
from .data.proxy_dataset import ProxyDataModule
from .data.proxy_generator import build_identity_mapping
from .models import create_proxy_model, print_model_summary
from .models.proxy_heads import compute_proxy_loss_batched
from .utils.proxy_visualization import save_proxy_training_sample
from .utils.logger import get_logger, log_to_file_only

try:
    import yaml
except Exception:
    yaml = None


def parse_task_features(names: list) -> list:
    return [n.lower() for n in names]


def evaluate_proxy_model(model, proxy_heads, dataloader, device,
                          save_dir=None, epoch=None, split_name="val",
                          save_activations=True):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_valid = 0
    num_batches = 0
    per_task_stats = {}

    with torch.no_grad():
        val_pbar = tqdm(dataloader, desc=f"Validating {split_name}", leave=False, position=2)
        for batch_idx, batch in enumerate(val_pbar):
            images = batch["images"].to(device)
            task_vec = batch["task_vector"].to(device)
            proxy_targets = batch["proxy_targets"].to(device)
            n_values = batch["n"].to(device).view(-1)
            task_features = batch["task_feature"]

            hidden_seq, _ = model(images, task_vec)

            loss, metrics = compute_proxy_loss_batched(
                proxy_heads, hidden_seq, task_vec, proxy_targets,
                n_values, task_features, device
            )

            total_loss += metrics["proxy_loss"]
            total_correct += int(metrics["proxy_accuracy"] * sum(
                batch["proxy_targets"][b].ge(0).sum().item()
                for b in range(len(task_features))
            ))
            total_valid += sum(
                batch["proxy_targets"][b].ge(0).sum().item()
                for b in range(len(task_features))
            )

            for key, val in metrics.items():
                if key.endswith("_acc") and key not in per_task_stats:
                    per_task_stats[key] = {"correct": 0, "total": 0}
                if key.endswith("_acc"):
                    per_task_stats[key]["correct"] += int(val * metrics.get(key.replace("_acc", "_count"), 0))
                if key.endswith("_count"):
                    acc_key = key.replace("_count", "_acc")
                    if acc_key in per_task_stats:
                        per_task_stats[acc_key]["total"] += int(val)

            num_batches += 1

            val_pbar.set_postfix({
                'loss': f'{metrics["proxy_loss"]:.4f}',
                'acc': f'{metrics["proxy_accuracy"]:.4f}'
            })

            if save_dir is not None and save_activations and epoch is not None:
                _save_proxy_states(hidden_seq, batch, save_dir, epoch, batch_idx, split_name)

    return {
        'loss': total_loss / max(num_batches, 1),
        'accuracy': total_correct / max(total_valid, 1),
        'per_task': {k: v["correct"] / max(v["total"], 1) for k, v in per_task_stats.items()},
    }


def _save_proxy_states(hidden_seq, batch, save_dir, epoch, batch_idx, split_name):
    split_dir = save_dir / f"epoch_{epoch:03d}" / split_name
    split_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "hidden": hidden_seq.cpu(),
        "task_vector": batch["task_vector"].cpu(),
        "n": batch["n"].cpu(),
        "proxy_targets": batch["proxy_targets"].cpu(),
        "task_feature": batch["task_feature"],
        "locations": batch.get("locations"),
        "categories": batch.get("categories"),
        "identities": batch.get("identities"),
        "split": split_name,
    }

    filename = split_dir / f"batch_{batch_idx:04d}.pt"
    torch.save(payload, filename)


def main():
    parser = argparse.ArgumentParser(description="Proxy Task Pre-training for WM Models")
    parser.add_argument("--config", type=str, required=True, help="Path to proxy config YAML")
    parser.add_argument("--output_dir", type=str, default="experiments", help="Output directory")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    exp_name = cfg.get("experiment_name", "proxy_mtmf")
    exp_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir_name = f"{exp_name}_{exp_id}"

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
    log(f"PROXY TASK PRE-TRAINING")
    log(f"EXPERIMENT ID: {exp_id}")
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
        "experiment_type": "proxy_pretraining",
        "config_file": args.config,
        "start_time": datetime.now().isoformat(),
        "device": str(device),
        "config": cfg,
    }
    with open(out_dir / "experiment_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    log("\n" + "="*70)
    log("PHASE 1: PROXY TASK PRE-TRAINING")
    log("="*70)

    log("\n1. Loading Stimuli and Creating Validation Splits...")
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

    identity_mapping, num_identities = build_identity_mapping(train_data)
    log(f"\nIdentity mapping: {num_identities} unique identities")
    log(f"Mapping: {identity_mapping}")

    log("\n2. Creating Proxy Data Module...")
    task_features = parse_task_features(cfg["task_features"])
    data_module = ProxyDataModule(
        stimulus_data=train_data,
        val_novel_angle_data=val_novel_angle_data,
        val_novel_identity_data=val_novel_identity_data,
        n_values=cfg["n_values"],
        task_features=task_features,
        sequence_length=cfg["sequence_length"],
        batch_size=cfg["batch_size"],
        num_train=cfg["num_train"],
        num_val_novel_angle=cfg.get("num_val_novel_angle", cfg.get("num_val", 400)),
        num_val_novel_identity=cfg.get("num_val_novel_identity", cfg.get("num_val", 400)),
        num_workers=cfg["num_workers"],
        match_probability=cfg.get("match_probability", 0.5),
        cache_train_sequences=cfg.get("cache_train_sequences", False),
        cache_val_sequences=cfg.get("cache_val_sequences", True),
        identity_mapping=identity_mapping,
    )

    train_loader = data_module.train_dataloader()
    val_novel_angle_loader = data_module.val_novel_angle_dataloader()
    val_novel_identity_loader = data_module.val_novel_identity_dataloader()

    log("\n3. Creating Proxy Model...")
    rnn_type = cfg.get("rnn_type", "gru")
    model_arch = cfg.get("model_type", "baseline")
    if model_arch == "attention":
        model_type_str = f"attention_{rnn_type}"
    else:
        model_type_str = rnn_type

    model = create_proxy_model(
        model_type=model_type_str,
        hidden_size=cfg["hidden_size"],
        num_layers=cfg["num_layers"],
        dropout=cfg.get("dropout", 0.0),
        pretrained_backbone=cfg.get("pretrained_backbone", True),
        freeze_backbone=cfg.get("freeze_backbone", True),
        attention_hidden_dim=cfg.get("attention_hidden_dim"),
        attention_dropout=cfg.get("attention_dropout", 0.1),
        attention_mode=cfg.get("attention_mode", "task_only"),
        num_identities=num_identities,
        num_locations=cfg.get("num_locations", 4),
        num_categories=cfg.get("num_categories", 4),
    )
    model = model.to(device)
    proxy_heads = model.proxy_heads

    print_model_summary(model)

    log("\n" + "="*70)
    log("TRAINABLE PARAMETERS BREAKDOWN")
    log("="*70)

    perceptual_trainable = 0
    cognitive_trainable = 0
    proxy_heads_trainable = 0
    attention_trainable = 0

    log("\nPerceptual Module (trainable layers):")
    for name, param in model.perceptual.named_parameters():
        if param.requires_grad:
            log(f"  {name}: {param.numel():,} parameters")
            perceptual_trainable += param.numel()

    log(f"\nCognitive Module (trainable layers):")
    for name, param in model.cognitive.named_parameters():
        if param.requires_grad:
            log(f"  {name}: {param.numel():,} parameters")
            cognitive_trainable += param.numel()

    if model.is_attention:
        log(f"\nAttention Module (trainable layers):")
        for name, param in model.attention.named_parameters():
            if param.requires_grad:
                log(f"  {name}: {param.numel():,} parameters")
                attention_trainable += param.numel()

    log(f"\nProxy Heads (trainable layers):")
    for name, param in model.proxy_heads.named_parameters():
        if param.requires_grad:
            log(f"  {name}: {param.numel():,} parameters")
            proxy_heads_trainable += param.numel()

    total_trainable = perceptual_trainable + cognitive_trainable + attention_trainable + proxy_heads_trainable
    log("\n" + "-"*70)
    log(f"Perceptual (trainable):  {perceptual_trainable:,}")
    log(f"Cognitive (trainable):   {cognitive_trainable:,}")
    log(f"Attention (trainable):   {attention_trainable:,}")
    log(f"Proxy Heads (trainable): {proxy_heads_trainable:,}")
    log(f"TOTAL TRAINABLE:         {total_trainable:,}")
    log("="*70 + "\n")

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg["lr"], weight_decay=cfg["weight_decay"]
    )
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=cfg["milestones"], gamma=cfg["gamma"]
    )

    log("\n4. Training on Proxy Task...")
    best_val_acc = 0.0
    results_log = []

    for epoch in tqdm(range(cfg["epochs"]), desc="Training Epochs", position=0):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_valid = 0
        num_train_batches = 0
        all_per_task = {}

        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg['epochs']}", position=1, leave=False)
        for batch in train_pbar:
            images = batch["images"].to(device)
            task_vec = batch["task_vector"].to(device)
            proxy_targets = batch["proxy_targets"].to(device)
            n_values = batch["n"].to(device).view(-1)
            task_features = batch["task_feature"]

            optimizer.zero_grad()
            hidden_seq, _ = model(images, task_vec)

            loss, metrics = compute_proxy_loss_batched(
                proxy_heads, hidden_seq, task_vec, proxy_targets,
                n_values, task_features, device
            )

            loss.backward()
            if cfg.get("grad_clip"):
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            optimizer.step()

            train_loss += metrics["proxy_loss"]
            batch_valid = sum(
                batch["proxy_targets"][b].ge(0).sum().item()
                for b in range(len(task_features))
            )
            train_correct += int(metrics["proxy_accuracy"] * batch_valid)
            train_valid += batch_valid
            num_train_batches += 1

            for key, val in metrics.items():
                if key.endswith("_acc") or key.endswith("_count"):
                    if key not in all_per_task:
                        all_per_task[key] = []
                    all_per_task[key].append(val)

            train_pbar.set_postfix({
                'loss': f'{metrics["proxy_loss"]:.4f}',
                'acc': f'{metrics["proxy_accuracy"]:.4f}'
            })

        train_loss /= max(num_train_batches, 1)
        train_acc = train_correct / max(train_valid, 1)

        per_task_avg = {}
        for key, vals in all_per_task.items():
            per_task_avg[key] = sum(vals) / len(vals)

        val_novel_angle_results = {'loss': 0, 'accuracy': 0, 'per_task': {}}
        if val_novel_angle_loader is not None:
            val_novel_angle_results = evaluate_proxy_model(
                model, proxy_heads, val_novel_angle_loader, device,
                save_dir=hidden_dir, epoch=epoch+1, split_name="val_novel_angle",
                save_activations=cfg.get("save_hidden", True)
            )

        val_novel_identity_results = {'loss': 0, 'accuracy': 0, 'per_task': {}}
        if val_novel_identity_loader is not None:
            val_novel_identity_results = evaluate_proxy_model(
                model, proxy_heads, val_novel_identity_loader, device,
                save_dir=hidden_dir, epoch=epoch+1, split_name="val_novel_identity",
                save_activations=cfg.get("save_hidden", True)
            )

        epoch_results = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_proxy_accuracy': train_acc,
            'train_per_task': per_task_avg,
            'val_novel_angle_loss': val_novel_angle_results['loss'],
            'val_novel_angle_acc': val_novel_angle_results['accuracy'],
            'val_novel_angle_per_task': val_novel_angle_results['per_task'],
            'val_novel_identity_loss': val_novel_identity_results['loss'],
            'val_novel_identity_acc': val_novel_identity_results['accuracy'],
            'val_novel_identity_per_task': val_novel_identity_results['per_task'],
            'lr': optimizer.param_groups[0]['lr'],
        }
        results_log.append(epoch_results)

        log_epoch(
            f"[Epoch {epoch+1:03d}/{cfg['epochs']}] "
            f"lr={optimizer.param_groups[0]['lr']:.2e} | "
            f"train: loss={train_loss:.4f}, proxy_acc={train_acc:.3f} | "
            f"val_angle: loss={val_novel_angle_results['loss']:.4f}, acc={val_novel_angle_results['accuracy']:.3f} | "
            f"val_id: loss={val_novel_identity_results['loss']:.4f}, acc={val_novel_identity_results['accuracy']:.3f}"
        )

        per_task_str = "  per-task acc: "
        for key in sorted(per_task_avg.keys()):
            if key.endswith("_acc"):
                per_task_str += f"{key}={per_task_avg[key]:.3f}  "
        log_epoch(per_task_str)

        if cfg.get("save_visualizations", True):
            vis_dir = out_dir / "visualizations"
            num_vis = cfg.get("num_visualizations", 1)

            def save_proxy_vis(data_loader, split_name):
                task_samples = {tf: [] for tf in task_features}
                for batch in data_loader:
                    for b in range(len(batch["task_feature"])):
                        tf = batch["task_feature"][b]
                        if tf in task_samples and len(task_samples[tf]) < num_vis:
                            task_samples[tf].append((batch, b))
                    if all(len(v) >= num_vis for v in task_samples.values()):
                        break
                for tf, samples in task_samples.items():
                    for vis_idx, (batch, sample_idx) in enumerate(samples[:num_vis]):
                        save_proxy_training_sample(
                            model=model, proxy_heads=proxy_heads, batch=batch,
                            device=device, save_dir=vis_dir, epoch=epoch+1,
                            batch_idx=vis_idx, split_name=f"{split_name}_{tf}",
                            sample_idx=sample_idx,
                        )

            save_proxy_vis(train_loader, "train")
            if val_novel_angle_loader:
                save_proxy_vis(val_novel_angle_loader, "val_novel_angle")
            if val_novel_identity_loader:
                save_proxy_vis(val_novel_identity_loader, "val_novel_identity")

            if epoch == 0:
                log_epoch(f"  Saved proxy visualizations to: {vis_dir}/")

        val_acc = val_novel_angle_results['accuracy']
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'proxy_heads_state_dict': proxy_heads.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_novel_angle_acc': val_novel_angle_results['accuracy'],
                'val_novel_identity_acc': val_novel_identity_results['accuracy'],
                'identity_mapping': identity_mapping,
                'num_identities': num_identities,
                'config': cfg,
                'experiment_type': 'proxy_pretraining',
            }, out_dir / "best_model.pt")
            log_epoch(f"  Saved best proxy model (val_acc={val_acc:.4f})")

        scheduler.step()

    with open(out_dir / "training_log.json", 'w') as f:
        json.dump(results_log, f, indent=2)

    metadata["end_time"] = datetime.now().isoformat()
    metadata["best_val_acc"] = best_val_acc
    metadata["total_epochs"] = cfg["epochs"]
    with open(out_dir / "experiment_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    log("\n" + "="*70)
    log("PROXY PRE-TRAINING COMPLETED")
    log("="*70)
    log(f"\nExperiment ID: {exp_id}")
    log(f"Best Val Accuracy: {best_val_acc:.4f}")
    log(f"\nResults saved to: {out_dir}")
    log(f"\nTo fine-tune this model on the real N-back task:")
    log(f"  python -m src.finetune_from_proxy --proxy_exp_dir {out_dir} --config configs/mtmf.yaml")


if __name__ == "__main__":
    main()
