#!/usr/bin/env python3
"""Plot meta-learning experiment results.

Usage:
    python -m src.scripts.plot_meta_learning --exp_dir experiments/meta_learning --output_dir plots/meta_learning
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import argparse
import numpy as np
from collections import defaultdict

METRIC_LABELS = {
    "train_loss": "Training Loss",
    "train_acc": "Training Accuracy",
    "val_loss": "Validation Loss",
    "val_acc": "Validation Accuracy",
}

STYLES = [
    {"color": "#1f77b4", "linestyle": "-", "marker": "o"},
    {"color": "#ff7f0e", "linestyle": "--", "marker": "s"},
    {"color": "#2ca02c", "linestyle": "-.", "marker": "^"},
    {"color": "#d62728", "linestyle": ":", "marker": "d"},
    {"color": "#9467bd", "linestyle": "-", "marker": "v"},
    {"color": "#8c564b", "linestyle": "--", "marker": "p"},
]

METHOD_NAMES = {
    "scratch": "From Scratch",
    "full_finetune": "Full Fine-tuning",
    "attention_only": "Attention Only",
    "attention_classifier": "Attention + Classifier",
    "cognitive_only": "Cognitive Only",
    "classifier_only": "Classifier Only",
}


def load_meta_experiments(exp_dir: Path):
    """Load meta-learning results, grouping by task and method."""
    experiments = defaultdict(lambda: defaultdict(list))
    
    for result_file in sorted(exp_dir.glob("meta_learning_*.json")):
        with open(result_file) as f:
            data = json.load(f)
        
        task = data.get("task", "unknown")
        method = data.get("method", "unknown")
        experiments[task][method].append(data)
    
    return experiments


def plot_learning_curves(experiments, output_dir):
    """Plot learning curves for each task, comparing methods."""
    for task, methods in experiments.items():
        plt.figure(figsize=(14, 5))
        
        # Plot 1: Training curves
        plt.subplot(1, 2, 1)
        for i, (method, runs) in enumerate(methods.items()):
            style = STYLES[i % len(STYLES)]
            label = f"{METHOD_NAMES.get(method, method)} (n={len(runs)})"
            
            # Aggregate across runs
            all_epochs = []
            all_values_by_epoch = defaultdict(list)
            
            for run in runs:
                # Add epoch 0 (before training) - use before metrics loss
                before_loss = run.get("before", {}).get("loss")
                if before_loss is not None:
                    all_epochs.append(0)
                    all_values_by_epoch[0].append(before_loss)
                
                # Add training history
                history = run.get("training_history", [])
                for entry in history:
                    epoch = entry.get("epoch", 0)
                    all_epochs.append(epoch)
                    all_values_by_epoch[epoch].append(entry.get("train_loss", 0))
            
            if not all_epochs:
                continue
            
            epochs = sorted(set(all_epochs))
            mean_values = [np.mean(all_values_by_epoch[e]) for e in epochs]
            
            plt.plot(epochs, mean_values,
                     color=style["color"],
                     linestyle=style["linestyle"],
                     marker=style["marker"],
                     markersize=4,
                     linewidth=2,
                     label=label,
                     markevery=max(1, len(epochs)//10))
        
        plt.xlabel("Epoch", fontsize=11)
        plt.ylabel("Training Loss", fontsize=11)
        plt.title(f"Task: {task} - Training Loss", fontsize=12, fontweight="bold")
        plt.legend(loc="best", fontsize=9)
        plt.grid(True, alpha=0.3, linestyle="--")
        
        # Plot 2: Validation accuracy
        plt.subplot(1, 2, 2)
        for i, (method, runs) in enumerate(methods.items()):
            style = STYLES[i % len(STYLES)]
            label = f"{METHOD_NAMES.get(method, method)} (n={len(runs)})"
            
            all_epochs = []
            all_values_by_epoch = defaultdict(list)
            
            for run in runs:
                # Add epoch 0 (before training) - use before metrics accuracy
                before_acc = run.get("before", {}).get("accuracy")
                if before_acc is not None:
                    all_epochs.append(0)
                    all_values_by_epoch[0].append(before_acc)
                
                # Add training history
                history = run.get("training_history", [])
                for entry in history:
                    epoch = entry.get("epoch", 0)
                    all_epochs.append(epoch)
                    all_values_by_epoch[epoch].append(entry.get("val_acc", 0))
            
            if not all_epochs:
                continue
            
            epochs = sorted(set(all_epochs))
            mean_values = [np.mean(all_values_by_epoch[e]) for e in epochs]
            
            plt.plot(epochs, mean_values,
                     color=style["color"],
                     linestyle=style["linestyle"],
                     marker=style["marker"],
                     markersize=4,
                     linewidth=2,
                     label=label,
                     markevery=max(1, len(epochs)//10))
        
        plt.xlabel("Epoch", fontsize=11)
        plt.ylabel("Validation Accuracy", fontsize=11)
        plt.title(f"Task: {task} - Validation Accuracy", fontsize=12, fontweight="bold")
        plt.legend(loc="best", fontsize=9)
        plt.grid(True, alpha=0.3, linestyle="--")
        
        plt.tight_layout()
        plt.savefig(output_dir / f"{task}_learning_curves.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved {task}_learning_curves.png")


def plot_few_shot_comparison(experiments, output_dir):
    """Plot final accuracy vs number of shots for each method."""
    for task, methods in experiments.items():
        plt.figure(figsize=(10, 6))
        
        for i, (method, runs) in enumerate(methods.items()):
            style = STYLES[i % len(STYLES)]
            
            # Group by num_shots
            shots_to_accs = defaultdict(list)
            for run in runs:
                shots = run.get("num_shots", 0)
                best_acc = run.get("best_accuracy", 0)
                shots_to_accs[shots].append(best_acc)
            
            if not shots_to_accs:
                continue
            
            shots = sorted(shots_to_accs.keys())
            mean_accs = [np.mean(shots_to_accs[s]) for s in shots]
            # Use standard error instead of standard deviation
            stderr_accs = [np.std(shots_to_accs[s]) / np.sqrt(len(shots_to_accs[s])) for s in shots]
            
            plt.errorbar(shots, mean_accs, yerr=stderr_accs,
                        color=style["color"],
                        linestyle=style["linestyle"],
                        marker=style["marker"],
                        markersize=6,
                        linewidth=2,
                        capsize=3,
                        label=METHOD_NAMES.get(method, method))
        
        plt.xlabel("Number of Training Examples", fontsize=11)
        plt.ylabel("Best Validation Accuracy", fontsize=11)
        plt.title(f"Task: {task} - Few-Shot Learning", fontsize=12, fontweight="bold")
        plt.legend(loc="best", fontsize=9)
        plt.grid(True, alpha=0.3, linestyle="--")
        plt.tight_layout()
        plt.savefig(output_dir / f"{task}_few_shot.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved {task}_few_shot.png")


def plot_method_comparison(experiments, output_dir):
    """Plot bar chart comparing methods across tasks."""
    all_methods = set()
    for methods in experiments.values():
        all_methods.update(methods.keys())
    
    for method in sorted(all_methods):
        tasks = []
        before_accs = []
        after_accs = []
        
        for task, methods in experiments.items():
            if method in methods:
                runs = methods[method]
                tasks.append(task)
                before_accs.append(np.mean([r.get("before", {}).get("accuracy", 0) for r in runs]))
                after_accs.append(np.mean([r.get("best_accuracy", 0) for r in runs]))
        
        if not tasks:
            continue
        
        x = np.arange(len(tasks))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(x - width/2, before_accs, width, label="Before Adaptation", color="#ff7f0e", alpha=0.8)
        ax.bar(x + width/2, after_accs, width, label="After Adaptation", color="#1f77b4", alpha=0.8)
        
        ax.set_xlabel("Task", fontsize=11)
        ax.set_ylabel("Accuracy", fontsize=11)
        ax.set_title(f"Method: {METHOD_NAMES.get(method, method)}", fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(tasks, rotation=15, ha="right")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3, linestyle="--", axis="y")
        
        plt.tight_layout()
        plt.savefig(output_dir / f"method_{method}.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved method_{method}.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", type=str, default="experiments/meta_learning")
    parser.add_argument("--output_dir", type=str, default="plots/meta_learning")
    args = parser.parse_args()
    
    exp_dir = Path(args.exp_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not exp_dir.exists():
        print(f"Error: Directory {exp_dir} does not exist")
        return
    
    experiments = load_meta_experiments(exp_dir)
    
    print(f"Found {len(experiments)} task(s):")
    for task, methods in experiments.items():
        print(f"  {task}: {sum(len(runs) for runs in methods.values())} experiment(s) across {len(methods)} method(s)")
    
    print("\nGenerating plots...")
    plot_learning_curves(experiments, output_dir)
    plot_few_shot_comparison(experiments, output_dir)
    plot_method_comparison(experiments, output_dir)
    
    print(f"\nPlots saved to {output_dir}/")


if __name__ == "__main__":
    main()
