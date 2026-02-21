#!/usr/bin/env python3
"""Plot training metrics across all experiments for comparison.

Modified to compute mean across multiple runs of the same experiment.
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import argparse
import numpy as np
from collections import defaultdict

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
    "wm_dual_attention_mtmf": "MTMF + Dual Attention",
}


def load_experiments(exp_dir: Path):
    """Load all training logs, grouping multiple runs of same experiment."""
    experiments_raw = defaultdict(list)
    
    for exp_path in sorted(exp_dir.iterdir()):
        if not exp_path.is_dir():
            continue
        log_file = exp_path / "training_log.json"
        if log_file.exists():
            with open(log_file) as f:
                data = json.load(f)
            # Extract experiment name (remove timestamp suffix)
            name = "_".join(exp_path.name.split("_")[:-2])
            experiments_raw[name].append(data)
    
    # Aggregate multiple runs
    experiments = {}
    for name, runs in experiments_raw.items():
        experiments[name] = {
            'runs': runs,
            'n_runs': len(runs)
        }
    
    return experiments


def get_all_metrics(experiments):
    """Get all numeric metrics available across experiments."""
    metrics = set()
    for exp_data in experiments.values():
        for run in exp_data['runs']:
            if run and isinstance(run, list):
                first_entry = next((e for e in run if isinstance(e, dict)), None)
                if not first_entry:
                    continue
                for k, v in first_entry.items():
                    if isinstance(v, (int, float)) and k != "epoch":
                        metrics.add(k)
    return sorted(metrics)


def plot_metric(experiments, metric, output_dir):
    """Plot a single metric across all experiments using mean of multiple runs."""
    plt.figure(figsize=(12, 7))
    
    for i, (name, exp_data) in enumerate(experiments.items()):
        runs = exp_data['runs']
        n_runs = exp_data['n_runs']
        contributing_runs = 0
        
        # Collect all epochs and values from all runs
        all_epochs = []
        all_values_by_epoch = defaultdict(list)
        
        for run in runs:
            run_has_metric = False
            for entry in run:
                if metric in entry:
                    epoch = entry["epoch"]
                    all_epochs.append(epoch)
                    all_values_by_epoch[epoch].append(entry[metric])
                    run_has_metric = True
            if run_has_metric:
                contributing_runs += 1
        
        if not all_epochs:
            continue
        
        # Get unique sorted epochs
        epochs = sorted(set(all_epochs))
        
        # Compute mean and std for each epoch
        mean_values = []
        std_values = []
        for epoch in epochs:
            values = all_values_by_epoch[epoch]
            mean_values.append(np.mean(values))
            std_values.append(np.std(values) if len(values) > 1 else 0)
        
        # Plot
        style = STYLES[i % len(STYLES)]
        label = f"{EXP_NAMES.get(name, name)} (n={contributing_runs}/{n_runs})"
        
        line = plt.plot(epochs, mean_values, 
                 color=style["color"],
                 linestyle=style["linestyle"],
                 marker=style["marker"],
                 markersize=4,
                 linewidth=2,
                 label=label,
                 markevery=max(1, len(epochs)//15))
        
        # Add shaded error region if multiple runs
        if contributing_runs > 1:
            plt.fill_between(epochs, 
                           np.array(mean_values) - np.array(std_values),
                           np.array(mean_values) + np.array(std_values),
                           color=style["color"], alpha=0.2)
    
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
    
    print(f"Found {len(experiments)} experiment types:")
    for name, exp_data in experiments.items():
        print(f"  {name}: {exp_data['n_runs']} run(s)")

    all_metrics = get_all_metrics(experiments)
    metrics_to_plot = args.metrics if args.metrics else all_metrics
    print(f"\nPlotting {len(metrics_to_plot)} metrics (using mean across runs)")

    for metric in metrics_to_plot:
        if metric in all_metrics:
            plot_metric(experiments, metric, output_dir)
            print(f"  Saved {metric}.png")

    print(f"\nPlots saved to {output_dir}/")
    print("\nNote: Plots show mean values across multiple runs.")
    print("      Shaded regions indicate ±1 standard deviation.")


if __name__ == "__main__":
    main()
