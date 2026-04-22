"""Main experiment runner for meta-learning."""

import json
import yaml
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image

from ..train import load_real_stimulus_data as load_stimulus_data
from ..models.model_factory import create_model
from .tasks import NOVEL_TASKS, generate_novel_sequences
from .adaptation import ADAPTATION_METHODS
from .training import train_epoch, evaluate
from .visualization import save_meta_visualization


class SimpleNBackDataset(Dataset):
    """Simple dataset for meta-learning sequences."""
    
    def __init__(self, sequences: List[Dict], stimulus_data: Dict):
        self.sequences = sequences
        self.stimulus_data = stimulus_data
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        images = []
        targets = []
        
        for trial in seq["trials"]:
            img = Image.open(trial["stimulus_path"]).convert('RGB')
            images.append(self.transform(img))
            targets.append(trial["target"])
        
        return {
            "images": torch.stack(images),
            "targets": torch.tensor(targets, dtype=torch.long),
            "task_vector": seq["task_vector"],
            "n": seq["n"],
        }


def custom_collate(batch):
    """Custom collate function for variable-length sequences."""
    return {
        "images": torch.stack([item["images"] for item in batch]),
        "targets": torch.stack([item["targets"] for item in batch]),
        "task_vector": torch.stack([item["task_vector"] for item in batch]),
        "n": torch.tensor([item["n"] for item in batch]),
    }


def prepare_dataloaders(
    sequences: List[Dict],
    stimulus_data: Dict,
    batch_size: int = 16,
) -> DataLoader:
    """Convert sequences to DataLoader."""
    dataset = SimpleNBackDataset(sequences=sequences, stimulus_data=stimulus_data)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate)


def run_meta_learning_experiment(
    exp_dir: Optional[str],
    task_name: str,
    method: str,
    num_shots: int,
    num_test: int = 200,
    epochs: int = 20,
    batch_size: int = 16,
    lr: float = 0.0001,
    device: str = "cuda",
    output_dir: Optional[str] = None,
    task_feature: str = "category",
    num_visualizations: int = 5,
) -> Dict[str, Any]:
    """Run a meta-learning experiment.
    
    Args:
        exp_dir: Path to experiment directory containing config.yaml and best_model.pt.
                 If None, creates a new model from scratch.
        task_name: Name of the novel task
        method: Adaptation method
        num_shots: Number of training examples
        num_test: Number of test examples
        epochs: Number of training epochs
        batch_size: Batch size
        lr: Learning rate
        device: Device to use
        output_dir: Directory to save results
        task_feature: Feature to use for the task
        num_visualizations: Number of visualizations to save per epoch
    """
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    stimulus_data = load_stimulus_data()
    
    if task_name not in NOVEL_TASKS:
        raise ValueError(f"Unknown task: {task_name}. Choose from {list(NOVEL_TASKS.keys())}")
    
    task_config = NOVEL_TASKS[task_name]
    
    print(f"\n{'='*70}")
    print(f"META-LEARNING EXPERIMENT")
    print(f"{'='*70}")
    print(f"Task: {task_name} - {task_config['description']}")
    print(f"Method: {method} - {ADAPTATION_METHODS[method]}")
    print(f"Shots: {num_shots}")
    print(f"Device: {device}")
    print(f"{'='*70}\n")
    
    # Generate sequences
    print("Generating sequences...")
    seq_length = 8 if task_config["task_type"] == "pattern" else 6
    train_sequences = generate_novel_sequences(
        task_name, stimulus_data, num_shots // seq_length + 1, task_feature, seq_length
    )
    test_sequences = generate_novel_sequences(
        task_name, stimulus_data, num_test // seq_length + 1, task_feature, seq_length
    )
    
    train_sequences = train_sequences[:num_shots // seq_length + 1]
    test_sequences = test_sequences[:num_test // seq_length + 1]
    
    print(f"Train sequences: {len(train_sequences)}")
    print(f"Test sequences: {len(test_sequences)}")
    
    train_loader = prepare_dataloaders(train_sequences, stimulus_data, batch_size)
    test_loader = prepare_dataloaders(test_sequences, stimulus_data, batch_size)
    
    # Load or create model
    if method == "scratch" or exp_dir is None:
        print("Creating new model from scratch...")
        model = create_model(
            model_type="attention_gru",
            hidden_size=256,
            num_layers=1,
            pretrained_backbone=True,
            freeze_backbone=True,
            attention_mode="task_only",
        )
    else:
        # Load config and model from experiment directory
        exp_path = Path(exp_dir)
        config_path = exp_path / "config.yaml"
        model_path = exp_path / "best_model.pt"
        
        if not config_path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        print(f"Loading config from {config_path}...")
        with open(config_path) as f:
            config = yaml.safe_load(f)
        
        # Extract architecture parameters (same as train_with_generalization.py)
        rnn_type = config.get("rnn_type", "gru")
        model_arch = config.get("model_type", "baseline")
        
        # Construct model_type string (same logic as train_with_generalization.py)
        if model_arch == "attention":
            model_type_str = f"attention_{rnn_type}"
        else:
            model_type_str = rnn_type
        
        print(f"Creating model with architecture: {model_type_str}")
        
        # Create model with same architecture using the same create_model function
        model = create_model(
            model_type=model_type_str,
            hidden_size=config.get("hidden_size", 256),
            num_layers=config.get("num_layers", 1),
            dropout=config.get("dropout", 0.0),
            pretrained_backbone=config.get("pretrained_backbone", True),
            freeze_backbone=config.get("freeze_backbone", True),
            attention_hidden_dim=config.get("attention_hidden_dim"),
            attention_dropout=config.get("attention_dropout", 0.1),
            attention_mode=config.get("attention_mode", "task_only"),
            classifier_layers=config.get("classifier_layers"),
        )
        
        # Load pretrained weights
        print(f"Loading pretrained weights from {model_path}...")
        checkpoint = torch.load(model_path, map_location=device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state_dict, strict=False)
    
    model = model.to(device)
    
    # Apply adaptation
    from .adaptation import apply_adaptation_method
    adapt_info = apply_adaptation_method(model, method)
    print(f"\nAdaptation: {adapt_info['trainable_params']:,} trainable, {adapt_info['frozen_params']:,} frozen")
    
    # Setup visualization directory
    vis_dir = Path(output_dir) / "visualizations" if output_dir else None
    if vis_dir:
        vis_dir.mkdir(parents=True, exist_ok=True)
    
    # Evaluate before training (epoch 0)
    print("\nEvaluating before adaptation...")
    before_metrics = evaluate(model, test_loader, device)
    print(f"Before: Loss={before_metrics['loss']:.4f}, Acc={before_metrics['accuracy']:.4f}")
    
    # Save visualization before training
    if vis_dir:
        save_meta_visualization(model, test_loader, device, vis_dir, task_name, method, epoch=0, num_samples=num_visualizations)
        print(f"  Saved {num_visualizations} visualizations: epoch_000 (before training)")
    
    # Train
    print(f"\nTraining for {epochs} epochs...")
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    
    results = {
        "task": task_name,
        "method": method,
        "num_shots": num_shots,
        "epochs": epochs,
        "before": before_metrics,
        "training_history": [],
    }
    
    best_acc = 0.0
    best_epoch = 0
    
    for epoch in range(1, epochs + 1):
        train_metrics = train_epoch(model, train_loader, optimizer, device, epoch)
        val_metrics = evaluate(model, test_loader, device)
        
        results["training_history"].append({
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_acc": train_metrics["accuracy"],
            "val_loss": val_metrics["loss"],
            "val_acc": val_metrics["accuracy"],
        })
        
        if val_metrics["accuracy"] > best_acc:
            best_acc = val_metrics["accuracy"]
            best_epoch = epoch
        
        print(f"Epoch {epoch}: Train Loss={train_metrics['loss']:.4f}, "
              f"Train Acc={train_metrics['accuracy']:.4f}, "
              f"Val Acc={val_metrics['accuracy']:.4f}")
        
        # Save visualization every epoch
        if vis_dir:
            save_meta_visualization(model, test_loader, device, vis_dir, task_name, method, epoch=epoch, num_samples=num_visualizations)
    
    # Final evaluation
    final_metrics = evaluate(model, test_loader, device)
    results["after"] = final_metrics
    results["best_accuracy"] = best_acc
    results["best_epoch"] = best_epoch
    results["improvement"] = final_metrics["accuracy"] - before_metrics["accuracy"]
    
    print(f"\n{'='*70}")
    print(f"RESULTS")
    print(f"{'='*70}")
    print(f"Before adaptation: {before_metrics['accuracy']:.4f}")
    print(f"After adaptation:  {final_metrics['accuracy']:.4f}")
    print(f"Improvement:       {results['improvement']:.4f}")
    print(f"Best accuracy:     {best_acc:.4f} (epoch {best_epoch})")
    print(f"{'='*70}\n")
    
    # Save results
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_file = output_dir / f"meta_learning_{task_name}_{method}_{timestamp}.json"
        
        with open(result_file, "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to {result_file}")
    
    return results
