#!/usr/bin/env python3
"""
Meta-Learning for Rapid Task Adaptation in Working Memory Models.

Usage:
    python -m src.meta_learning --help
    python -m src.meta_learning --list-models
    python -m src.meta_learning --exp-dir experiments/wm_mtmf_20501224_154125 --method attention_only --task nback_4 --shots 50
"""

import argparse
import sys
import yaml
from pathlib import Path
from typing import Optional, List, Dict, Any

import torch
import torch.nn as nn

from .models.model_factory import create_model
from .meta import (
    NOVEL_TASKS,
    ADAPTATION_METHODS,
    run_meta_learning_experiment,
)


def list_available_models():
    """List available pretrained models in experiments directory."""
    experiments_dir = Path("experiments")
    
    if not experiments_dir.exists():
        print("No experiments directory found.")
        return []
    
    models = []
    for exp_dir in experiments_dir.iterdir():
        if exp_dir.is_dir():
            model_path = exp_dir / "best_model.pt"
            if model_path.exists():
                config_path = exp_dir / "config.yaml"
                config_info = ""
                if config_path.exists():
                    try:
                        import yaml
                        with open(config_path) as f:
                            config = yaml.safe_load(f)
                        config_info = f" (N={config.get('n_values', [])}, features={config.get('task_features', [])})"
                    except:
                        pass
                
                models.append({
                    "name": exp_dir.name,
                    "path": str(model_path),
                    "config": config_info,
                })
    
    if not models:
        print("No pretrained models found in experiments directory.")
        return []
    
    print("\nAvailable Pretrained Models:")
    print("-" * 70)
    for i, m in enumerate(models, 1):
        print(f"  [{i}] {m['name']}{m['config']}")
        print(f"      Path: {m['path']}")
    print("-" * 70)
    
    return models


def main():
    parser = argparse.ArgumentParser(
        description="Meta-Learning for Rapid Task Adaptation in Working Memory Models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List available pretrained models
  python -m src.meta_learning --list-models

  # Run few-shot adaptation using experiment directory
  python -m src.meta_learning --exp-dir experiments/wm_mtmf_20501224_154125 \\
      --method attention_only --task nback_4 --shots 50

  # Train from scratch for comparison
  python -m src.meta_learning --method scratch --task nback_4 --shots 50
        """,
    )
    
    parser.add_argument("--exp-dir", type=str, default=None,
        help="Path to experiment directory containing config.yaml and best_model.pt")
    parser.add_argument("--list-models", action="store_true",
        help="List available pretrained models and exit.")
    parser.add_argument("--task", "-t", type=str, default="nback_4",
        choices=list(NOVEL_TASKS.keys()), help="Novel task to learn. Default: nback_4")
    parser.add_argument("--task-feature", type=str, default="category",
        choices=["location", "identity", "category"], help="Feature to use. Default: category")
    parser.add_argument("--method", type=str, default=None,
        choices=list(ADAPTATION_METHODS.keys()), help="Adaptation method. If not specified, runs all methods.")
    parser.add_argument("--shots", type=int, default=50,
        help="Number of training examples. Default: 50")
    parser.add_argument("--test-samples", type=int, default=200,
        help="Number of test examples. Default: 200")
    parser.add_argument("--epochs", type=int, default=20,
        help="Number of training epochs. Default: 20")
    parser.add_argument("--batch-size", type=int, default=16,
        help="Batch size. Default: 16")
    parser.add_argument("--lr", type=float, default=0.0001,
        help="Learning rate. Default: 0.0001")
    parser.add_argument("--output-dir", type=str, default="experiments/meta_learning",
        help="Directory to save results. Default: experiments/meta_learning")
    parser.add_argument("--device", type=str, default="cuda",
        help="Device to use (cuda/cpu). Default: cuda")
    parser.add_argument("--num-visualizations", type=int, default=5,
        help="Number of visualizations to save per epoch. Default: 5")
    parser.add_argument("--val-seed", type=int, default=42,
        help="Random seed for validation data generation (ensures same validation across runs). Default: 42")
    
    args = parser.parse_args()
    
    if args.list_models:
        list_available_models()
        return
    
    # Determine which methods to run
    methods_to_run = [args.method] if args.method else list(ADAPTATION_METHODS.keys())
    
    # For scratch method, no pretrained model needed
    if "scratch" in methods_to_run and len(methods_to_run) == 1:
        exp_dir = None
    elif args.exp_dir:
        exp_dir = args.exp_dir
    else:
        # Auto-detect latest experiment
        models = list_available_models()
        if models:
            # Get the directory (parent of best_model.pt)
            mtmf_models = [m for m in models if "mtmf" in m["name"]]
            model_path = mtmf_models[0]["path"] if mtmf_models else models[0]["path"]
            exp_dir = str(Path(model_path).parent)
            print(f"\nUsing experiment: {exp_dir}")
        else:
            print("Error: No pretrained models found. Use --method scratch or specify --exp-dir.")
            sys.exit(1)
    
    print(f"\n{'='*70}")
    print(f"Running {len(methods_to_run)} method(s): {', '.join(methods_to_run)}")
    print(f"{'='*70}\n")
    
    for i, method in enumerate(methods_to_run, 1):
        print(f"\n[{i}/{len(methods_to_run)}] Method: {method}")
        print("-" * 50)
        
        run_meta_learning_experiment(
            exp_dir=exp_dir,
            task_name=args.task,
            method=method,
            num_shots=args.shots,
            num_test=args.test_samples,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device=args.device,
            output_dir=args.output_dir,
            task_feature=args.task_feature,
            num_visualizations=args.num_visualizations,
            val_seed=args.val_seed,
        )


if __name__ == "__main__":
    main()
