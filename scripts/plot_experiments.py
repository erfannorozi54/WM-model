#!/usr/bin/env python3
"""Plot training metrics across all experiments for comparison."""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import argparse


def load_experiments(exp_dir: Path):
    """Load all training logs from experiments directory."""
    experiments = {}
    for exp_path in sorted(exp_dir.iterdir()):
        log_file = exp_path / "training_log.json"
        if log_file.exists():
            with open(log_file) as f:
                data = json.load(f)
            # Extract experiment name (remove timestamp)
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
    plt.figure(figsize=(10, 6))
    for name, data in experiments.items():
        epochs = [d["epoch"] for d in data if metric in d]
        values = [d[metric] for d in data if metric in d]
        if epochs:
            plt.plot(epochs, values, marker="o", markersize=3, label=name)
    
    plt.xlabel("Epoch")
    plt.ylabel(metric)
    plt.title(metric.replace("_", " ").title())
    plt.legend(loc="best", fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / f"{metric}.png", dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", type=str, default="experiments")
    parser.add_argument("--output_dir", type=str, default="plots")
    parser.add_argument("--metrics", nargs="*", help="Specific metrics to plot (default: all)")
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
