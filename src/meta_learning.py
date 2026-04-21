#!/usr/bin/env python3
"""
Meta-Learning for Rapid Task Adaptation in Working Memory Models.

Usage:
    python -m src.meta_learning --help
    python -m src.meta_learning --list-models
    python -m src.meta_learning --method attention_only --task nback_4 --shots 50
"""

import argparse
import sys
from pathlib import Path

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

  # Run few-shot adaptation with attention-only method
  python -m src.meta_learning --model experiments/wm_mtmf_20501224_154125/best_model.pt \
      --method attention_only --task nback_4 --shots 50

  # Train from scratch for comparison
  python -m src.meta_learning --method scratch --task nback_4 --shots 50
        """,
    )
    
    parser.add_argument("--model", "-m", type=str, default=None,
        help="Path to pretrained model checkpoint. If not specified, uses latest MTMF model.")
    parser.add_argument("--list-models", action="store_true",
        help="List available pretrained models and exit.")
    parser.add_argument("--task", "-t", type=str, default="nback_4",
        choices=list(NOVEL_TASKS.keys()), help="Novel task to learn. Default: nback_4")
    parser.add_argument("--task-feature", type=str, default="category",
        choices=["location", "identity", "category"], help="Feature to use. Default: category")
    parser.add_argument("--method", type=str, default="attention_only",
        choices=list(ADAPTATION_METHODS.keys()), help="Adaptation method. Default: attention_only")
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
    
    args = parser.parse_args()
    
    if args.list_models:
        list_available_models()
        return
    
    model_path = args.model
    if model_path is None and args.method != "scratch":
        models = list_available_models()
        if models:
            mtmf_models = [m for m in models if "mtmf" in m["name"]]
            model_path = (mtmf_models[0]["path"] if mtmf_models else models[0]["path"])
            print(f"\nUsing model: {model_path}")
        else:
            print("Error: No pretrained models found. Use --method scratch or specify --model path.")
            sys.exit(1)
    
    run_meta_learning_experiment(
        pretrained_model_path=model_path,
        task_name=args.task,
        method=args.method,
        num_shots=args.shots,
        num_test=args.test_samples,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        output_dir=args.output_dir,
        task_feature=args.task_feature,
    )


if __name__ == "__main__":
    main()
