#!/usr/bin/env python3
"""Plot training metrics across all experiments for comparison."""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import argparse
import numpy as np

# Better labels for metrics
METRIC_LABELS = {
    "train_loss": "Training Loss",
    "train_acc": "Training Accuracy",
    "train_masked_acc": "Training Accuracy (Masked)",
    "train_no_action_acc": "Training No-Action Accuracy",
    "val_novel_angle_loss": "Validation Loss (Novel Angle)",
    "val_novel_angle_acc": "Validation Accuracy (Novel Angle)",
    "val_novel_angle_acc_masked": "Validation Accuracy Masked (Novel Angle)",
    "val_novel_angle_acc_no_action": "Val No-Action Acc (Novel Angle)",
    "val_novel_identity_loss": "Validation Loss (Novel Identity)",
    "val_novel_identity_acc": "Validation Accuracy (Novel Identity)",
    "val_novel_identity_acc_masked": "Validation Accuracy Masked (Novel Identity)",
    "val_novel_identity_acc_no_action": "Val No-Action Acc (Novel Identity)",
    "lr": "Learning Rate",
}

# Distinct colors and line styles
STYLES = [
    {"color": "#1f77b4", "linestyle": "-", "marker": "o"},
    {"color": "#ff7f0e", "linestyle": "--", "marker": "s"},
    {"color": "#2ca02c", "linestyle": "-.", "marker": "^"},
    {"color": "#d62728", "linestyle": ":", "marker": "d"},
    {"color": "#9467bd", "linestyle": "-", "marker": "v"},
    {"color": "#8c564b", "linestyle": "--", "marker": "p"},
]

# Friendly experiment names
EXP_NAMES = {
    "wm_stsf": "STSF (Baseline)",
    "wm_stmf": "STMF",
    "wm_mtmf": "MTMF",
    "wm_attention_stmf": "STMF + Attention",
    "wm_dual_attention_stmf": "STMF + Dual Attention",
    "wm_attention_mtmf": "MTMF + Attention",
}


def load_experiments(exp_dir: Path):
    """Load all training logs from experiments directory."""
    experiments = {}
    for exp_path in sorted(exp_dir.iterdir()):
        log_file = exp_path / "training_log.json"
        if log_file.exists():
            with open(log_file) as f:
                data = json.load(f)
            name = "_".join(exp_path.name.split("_")[:-2])
            experiments[name] = data
    return experiments


def get_all_metrics(experiments):
    """Get all numeric metrics available across experiments."""
    metrics = set()
    for data in experiments.values():
        if data:
            for k, v in data[0].items():
                if isinstance(v, (int, float)) and k != "epoch":
                    metrics.add(k)
    return sorted(metrics)


def plot_metric(experiments, metric, output_dir):
    """Plot a single metric across all experiments."""
    plt.figure(figsize=(12, 7))
    
    for i, (name, data) in enumerate(experiments.items()):
        epochs = [d["epoch"] for d in data if metric in d]
        values = [d[metric] for d in data if metric in d]
        if epochs:
            style = STYLES[i % len(STYLES)]
            label = EXP_NAMES.get(name, name)
            plt.plot(epochs, values, 
                     color=style["color"],
                     linestyle=style["linestyle"],
                     marker=style["marker"],
                     markersize=4,
                     linewidth=2,
                     label=label,
                     markevery=max(1, len(epochs)//15))
    
    plt.xlabel("Epoch", fontsize=12)
    ylabel = METRIC_LABELS.get(metric, metric.replace("_", " ").title())
    plt.ylabel(ylabel, fontsize=12)
    plt.title(ylabel, fontsize=14, fontweight="bold")
    plt.legend(loc="best", fontsize=10, framealpha=0.9)
    plt.grid(True, alpha=0.3, linestyle="--")
    plt.tight_layout()
    plt.savefig(output_dir / f"{metric}.png", dpi=150, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", type=str, default="experiments")
    parser.add_argument("--output_dir", type=str, default="plots")
    parser.add_argument("--metrics", nargs="*", help="Specific metrics to plot")
    args = parser.parse_args()

    exp_dir = Path(args.exp_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    experiments = load_experiments(exp_dir)
    print(f"Found {len(experiments)} experiments: {list(experiments.keys())}")

    all_metrics = get_all_metrics(experiments)
    metrics_to_plot = args.metrics if args.metrics else all_metrics
    print(f"Plotting {len(metrics_to_plot)} metrics")

    for metric in metrics_to_plot:
        if metric in all_metrics:
            plot_metric(experiments, metric, output_dir)
            print(f"  Saved {metric}.png")

    print(f"\nPlots saved to {output_dir}/")


if __name__ == "__main__":
    main()
