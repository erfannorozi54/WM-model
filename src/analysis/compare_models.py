#!/usr/bin/env python3
"""
Comparative Analysis Tool for Working Memory Models.

This script compares baseline and attention-enhanced models across multiple metrics:
1. Behavioral Performance (accuracy, training curves)
2. Representational Geometry (decoding, orthogonalization)
3. Temporal Dynamics (Procrustes analysis)

Usage:
  # Compare baseline GRU vs attention GRU
  python -m src.analysis.compare_models \
    --baseline runs/wm_mtmf/hidden_states \
    --attention runs/wm_attention_mtmf/hidden_states \
    --output_dir results/comparison
  
  # Specific property and task
  python -m src.analysis.compare_models \
    --baseline runs/wm_mtmf/hidden_states \
    --attention runs/wm_attention_mtmf/hidden_states \
    --property identity --n 2 --task location
"""

import sys
from pathlib import Path
import argparse
import json
import numpy as np
from typing import Dict, Any, List, Optional
from tqdm import tqdm

# Import from same package
from .decoding import evaluate as decoding_evaluate
from .orthogonalization import evaluate as orthogonalization_evaluate
from .procrustes import procrustes_analysis, swap_hypothesis_test
from .activations import load_payloads


def compare_decoding(
    baseline_root: Path,
    attention_root: Path,
    property_name: str,
    train_time: int,
    test_times: List[int],
    task: Optional[str] = None,
    n_value: Optional[int] = None,
    baseline_epochs: Optional[List[int]] = None,
    attention_epochs: Optional[List[int]] = None,
    split: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Compare decoding performance between baseline and attention models.
    
    Args:
        baseline_root: Path to baseline hidden states
        attention_root: Path to attention hidden states
        property_name: Property to decode
        train_time: Training time point
        test_times: Test time points
        task: Task filter
        n_value: N-back filter
    
    Returns:
        Comparison dictionary with accuracies for both models
    """
    print(f"\n  Decoding comparison: {property_name}")
    
    # Baseline model
    baseline_result = decoding_evaluate(
        hidden_root=baseline_root,
        property_name=property_name,
        train_time=train_time,
        test_times=test_times,
        train_task=task,
        test_task=task,
        train_n=[n_value] if n_value else None,
        test_n=[n_value] if n_value else None,
        epochs=baseline_epochs,
        split=split,
    )
    
    # Attention model
    attention_result = decoding_evaluate(
        hidden_root=attention_root,
        property_name=property_name,
        train_time=train_time,
        test_times=test_times,
        train_task=task,
        test_task=task,
        train_n=[n_value] if n_value else None,
        test_n=[n_value] if n_value else None,
        epochs=attention_epochs,
        split=split,
    )
    
    # Extract accuracies
    baseline_accs = {t: baseline_result['test'][str(t)]['acc'] for t in test_times}
    attention_accs = {t: attention_result['test'][str(t)]['acc'] for t in test_times}
    
    # Compute improvements
    improvements = {}
    for t in test_times:
        if baseline_accs[t] is not None and attention_accs[t] is not None:
            improvements[t] = attention_accs[t] - baseline_accs[t]
        else:
            improvements[t] = None
    
    # Chance level, so a "decodability dropped" read can be checked against the
    # floor rather than only against the other model (identity pools ~72 classes
    # across both splits, so chance is ~1.4%, not 50%).
    n_classes_baseline = len(baseline_result.get('classes') or {})
    n_classes_attention = len(attention_result.get('classes') or {})

    return {
        'property': property_name,
        'train_time': train_time,
        'test_times': test_times,
        'baseline_accuracies': baseline_accs,
        'attention_accuracies': attention_accs,
        'n_classes_baseline': n_classes_baseline,
        'n_classes_attention': n_classes_attention,
        'chance_baseline': 1.0 / n_classes_baseline if n_classes_baseline else None,
        'chance_attention': 1.0 / n_classes_attention if n_classes_attention else None,
        'note_train_time': (
            f"test_times includes train_time={train_time}; that entry is in-sample "
            "(same rows the decoder was fit on) and is not a held-out score."
        ),
        'improvements': improvements,
        'mean_baseline': np.mean([v for v in baseline_accs.values() if v is not None]),
        'mean_attention': np.mean([v for v in attention_accs.values() if v is not None]),
        'mean_improvement': np.mean([v for v in improvements.values() if v is not None]),
    }


def compare_orthogonalization(
    baseline_root: Path,
    attention_root: Path,
    property_name: str,
    time: int,
    task: Optional[str] = None,
    n_value: Optional[int] = None,
    baseline_epochs: Optional[List[int]] = None,
    attention_epochs: Optional[List[int]] = None,
    split: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Compare orthogonalization indices between models.
    
    Args:
        baseline_root: Path to baseline hidden states
        attention_root: Path to attention hidden states
        property_name: Property to analyze
        time: Time point
        task: Task filter
        n_value: N-back filter
    
    Returns:
        Comparison dictionary
    """
    print(f"\n  Orthogonalization comparison: {property_name} at t={time}")
    
    # Baseline model
    baseline_result = orthogonalization_evaluate(
        hidden_root=baseline_root,
        property_name=property_name,
        time=time,
        task=task,
        n_value=n_value,
        epochs=baseline_epochs,
        split=split,
    )
    
    # Attention model
    attention_result = orthogonalization_evaluate(
        hidden_root=attention_root,
        property_name=property_name,
        time=time,
        task=task,
        n_value=n_value,
        epochs=attention_epochs,
        split=split,
    )
    
    return {
        'property': property_name,
        'time': time,
        'baseline_orthogonalization': baseline_result['orthogonalization'],
        'attention_orthogonalization': attention_result['orthogonalization'],
        'improvement': attention_result['orthogonalization'] - baseline_result['orthogonalization'],
        'baseline_n_classes': baseline_result['n_classes'],
        'attention_n_classes': attention_result['n_classes'],
    }


def compare_procrustes(
    baseline_root: Path,
    attention_root: Path,
    property_name: str,
    source_time: int,
    target_time: int,
    task: Optional[str] = None,
    n_value: Optional[int] = None,
    baseline_epochs: Optional[List[int]] = None,
    attention_epochs: Optional[List[int]] = None,
    split: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Compare Procrustes analysis between models.
    
    Args:
        baseline_root: Path to baseline hidden states
        attention_root: Path to attention hidden states
        property_name: Property to analyze
        source_time: Source time
        target_time: Target time
        task: Task filter
        n_value: N-back filter
    
    Returns:
        Comparison dictionary
    """
    print(f"\n  Procrustes comparison: {property_name} t={source_time}→{target_time}")
    
    # Baseline model
    baseline_result = procrustes_analysis(
        hidden_root=baseline_root,
        property_name=property_name,
        source_time=source_time,
        target_time=target_time,
        task=task,
        n_value=n_value,
        epochs=baseline_epochs,
        split=split,
    )
    
    # Attention model
    attention_result = procrustes_analysis(
        hidden_root=attention_root,
        property_name=property_name,
        source_time=source_time,
        target_time=target_time,
        task=task,
        n_value=n_value,
        epochs=attention_epochs,
        split=split,
    )
    
    return {
        'property': property_name,
        'source_time': source_time,
        'target_time': target_time,
        'baseline_disparity': baseline_result['procrustes_disparity'],
        'attention_disparity': attention_result['procrustes_disparity'],
        'disparity_difference': baseline_result['procrustes_disparity'] - attention_result['procrustes_disparity'],
        'baseline_reconstruction_acc': baseline_result['reconstruction_accuracy'],
        'attention_reconstruction_acc': attention_result['reconstruction_accuracy'],
        'reconstruction_improvement': attention_result['reconstruction_accuracy'] - baseline_result['reconstruction_accuracy'],
    }


def compare_swap_test(
    baseline_root: Path,
    attention_root: Path,
    property_name: str,
    encoding_time: int,
    k_offset: int = 1,
    task: Optional[str] = None,
    n_value: Optional[int] = None,
    baseline_epochs: Optional[List[int]] = None,
    attention_epochs: Optional[List[int]] = None,
    split: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Compare swap hypothesis test between models.
    
    Args:
        baseline_root: Path to baseline hidden states
        attention_root: Path to attention hidden states
        property_name: Property to analyze
        encoding_time: Encoding time
        k_offset: Temporal offset
        task: Task filter
        n_value: N-back filter
    
    Returns:
        Comparison dictionary
    """
    print(f"\n  Swap test comparison: {property_name} encoding_t={encoding_time}")
    
    # Baseline model
    baseline_result = swap_hypothesis_test(
        hidden_root=baseline_root,
        property_name=property_name,
        encoding_time=encoding_time,
        k_offset=k_offset,
        task=task,
        n_value=n_value,
        epochs=baseline_epochs,
        split=split,
    )
    
    # Attention model
    attention_result = swap_hypothesis_test(
        hidden_root=attention_root,
        property_name=property_name,
        encoding_time=encoding_time,
        k_offset=k_offset,
        task=task,
        n_value=n_value,
        epochs=attention_epochs,
        split=split,
    )
    
    # swap_hypothesis_test always decodes location (identity/category labels are
    # unique per trial and cannot be aligned across the two disjoint stimulus
    # groups the test needs). Reporting property_name here would present a
    # location result as evidence about property_name - the exact mislabelling
    # that put a location number in an identity-suppression table in the deck.
    decoded = baseline_result.get('decoded_property', 'location')
    return {
        'decoded_property': decoded,
        'grouping_property': property_name,
        'interpretation_warning': (
            f"This test decodes '{decoded}', not '{property_name}'. It measures whether a "
            f"rotation fitted at one time/stimulus group transfers to another; it is NOT "
            f"evidence about '{property_name}' decodability or suppression."
        ),
        'encoding_time': encoding_time,
        'baseline': {
            'correct_acc': baseline_result['correct_accuracy'],
            'swap1_acc': baseline_result['swap1_accuracy'],
            'swap2_acc': baseline_result['swap2_accuracy'],
            'hypothesis_confirmed': baseline_result['hypothesis_confirmed'],
            'verdict': baseline_result.get('verdict'),
        },
        'attention': {
            'correct_acc': attention_result['correct_accuracy'],
            'swap1_acc': attention_result['swap1_accuracy'],
            'swap2_acc': attention_result['swap2_accuracy'],
            'hypothesis_confirmed': attention_result['hypothesis_confirmed'],
            'verdict': attention_result.get('verdict'),
        },
        'improvements': {
            'correct': attention_result['correct_accuracy'] - baseline_result['correct_accuracy'],
            'swap1': attention_result['swap1_accuracy'] - baseline_result['swap1_accuracy'],
            'swap2': attention_result['swap2_accuracy'] - baseline_result['swap2_accuracy'],
        },
    }


def find_best_epoch(hidden_root: Path, metric: str = "val_novel_identity_acc") -> Optional[int]:
    """Best epoch for an experiment, read from its training_log.json.

    Mirrors ComprehensiveAnalysis._find_best_epoch so this tool selects the same
    checkpoint the single-model pipeline reports on. Returns None if the log is
    missing or unreadable, in which case the caller falls back to all epochs.
    """
    log_path = Path(hidden_root).parent / "training_log.json"
    if not log_path.exists():
        return None
    try:
        with open(log_path) as f:
            log = json.load(f)
        if not log:
            return None
        return int(max(log, key=lambda x: x.get(metric, 0))["epoch"])
    except (KeyError, IndexError, ValueError, TypeError):
        return None


def comprehensive_comparison(
    baseline_root: Path,
    attention_root: Path,
    property_name: str = 'identity',
    task: Optional[str] = None,
    n_value: Optional[int] = None,
    baseline_epochs: Optional[List[int]] = None,
    attention_epochs: Optional[List[int]] = None,
    split: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run comprehensive comparison across all analyses.
    
    Args:
        baseline_root: Path to baseline hidden states
        attention_root: Path to attention hidden states
        property_name: Property to analyze
        task: Task filter
        n_value: N-back filter
    
    Returns:
        Comprehensive comparison dictionary
    """
    print("\n" + "="*70)
    print("COMPREHENSIVE MODEL COMPARISON")
    print("="*70)
    print(f"\nBaseline: {baseline_root}")
    print(f"Attention: {attention_root}")
    print(f"Property: {property_name}")
    print(f"Task: {task or 'all'}")
    print(f"N-back: {n_value or 'all'}")
    print(f"Baseline epochs: {baseline_epochs or 'ALL (pooled across training)'}")
    print(f"Attention epochs: {attention_epochs or 'ALL (pooled across training)'}")
    print(f"Split: {split or 'all'}")
    if baseline_epochs is None or attention_epochs is None:
        print("  ! Pooling every saved epoch mixes untrained early checkpoints into the\n"
              "    comparison and does not match the two runs' training trajectories.\n"
              "    Pass --best_epoch (or --epochs) for a checkpoint-matched comparison.")

    results = {
        'baseline_root': str(baseline_root),
        'attention_root': str(attention_root),
        'property': property_name,
        'task': task,
        'n_value': n_value,
        'baseline_epochs': baseline_epochs,
        'attention_epochs': attention_epochs,
        'split': split,
        'epochs_pooled': baseline_epochs is None or attention_epochs is None,
    }
    
    # 1. Decoding comparison
    print("\n" + "-"*70)
    print("1. DECODING ANALYSIS")
    print("-"*70)
    try:
        decoding_comp = compare_decoding(
            baseline_root, attention_root,
            property_name=property_name,
            train_time=2,
            test_times=[2, 3, 4, 5],
            task=task,
            n_value=n_value,
            baseline_epochs=baseline_epochs,
            attention_epochs=attention_epochs,
            split=split,
        )
        results['decoding'] = decoding_comp
        print(f"  Mean baseline accuracy: {decoding_comp['mean_baseline']:.3f}")
        print(f"  Mean attention accuracy: {decoding_comp['mean_attention']:.3f}")
        print(f"  Mean improvement: {decoding_comp['mean_improvement']:+.3f}")
        if decoding_comp['chance_baseline'] is not None:
            print(f"  Chance level: {decoding_comp['chance_baseline']:.4f} "
                  f"({decoding_comp['n_classes_baseline']} classes, baseline) / "
                  f"{decoding_comp['chance_attention']:.4f} "
                  f"({decoding_comp['n_classes_attention']} classes, attention)")
    except Exception as e:
        print(f"  ✗ Decoding comparison failed: {e}")
        results['decoding'] = None
    
    # 2. Orthogonalization comparison
    print("\n" + "-"*70)
    print("2. ORTHOGONALIZATION ANALYSIS")
    print("-"*70)
    try:
        ortho_comp = compare_orthogonalization(
            baseline_root, attention_root,
            property_name=property_name,
            time=3,
            task=task,
            n_value=n_value,
            baseline_epochs=baseline_epochs,
            attention_epochs=attention_epochs,
            split=split,
        )
        results['orthogonalization'] = ortho_comp
        print(f"  Baseline O-index: {ortho_comp['baseline_orthogonalization']:.3f}")
        print(f"  Attention O-index: {ortho_comp['attention_orthogonalization']:.3f}")
        print(f"  Improvement: {ortho_comp['improvement']:+.3f}")
    except Exception as e:
        print(f"  ✗ Orthogonalization comparison failed: {e}")
        results['orthogonalization'] = None
    
    # 3. Procrustes comparison
    print("\n" + "-"*70)
    print("3. PROCRUSTES ANALYSIS")
    print("-"*70)
    try:
        procrustes_comp = compare_procrustes(
            baseline_root, attention_root,
            property_name=property_name,
            source_time=2,
            target_time=3,
            task=task,
            n_value=n_value,
            baseline_epochs=baseline_epochs,
            attention_epochs=attention_epochs,
            split=split,
        )
        results['procrustes'] = procrustes_comp
        print(f"  Baseline disparity: {procrustes_comp['baseline_disparity']:.4f}")
        print(f"  Attention disparity: {procrustes_comp['attention_disparity']:.4f}")
        print(f"  Disparity difference: {procrustes_comp['disparity_difference']:+.4f}")
        print(f"  Reconstruction improvement: {procrustes_comp['reconstruction_improvement']:+.3f}")
    except Exception as e:
        print(f"  ✗ Procrustes comparison failed: {e}")
        results['procrustes'] = None
    
    # 4. Swap test comparison
    print("\n" + "-"*70)
    print("4. SWAP HYPOTHESIS TEST")
    print("-"*70)
    try:
        swap_comp = compare_swap_test(
            baseline_root, attention_root,
            property_name=property_name,
            encoding_time=2,
            k_offset=1,
            task=task,
            n_value=n_value,
            baseline_epochs=baseline_epochs,
            attention_epochs=attention_epochs,
            split=split,
        )
        results['swap_test'] = swap_comp
        print(f"  NOTE: {swap_comp['interpretation_warning']}")
        print(f"  Baseline verdict: {swap_comp['baseline']['verdict']}")
        print(f"  Attention verdict: {swap_comp['attention']['verdict']}")
        print(f"  Correct accuracy improvement: {swap_comp['improvements']['correct']:+.3f}")
    except Exception as e:
        print(f"  ✗ Swap test comparison failed: {e}")
        results['swap_test'] = None
    
    return results


def save_comparison_results(results: Dict[str, Any], output_dir: Path, name: str = "comparison"):
    """Save comparison results to JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{name}.json"
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare baseline and attention-enhanced models"
    )
    parser.add_argument(
        "--baseline",
        type=str,
        required=True,
        help="Path to baseline model hidden states"
    )
    parser.add_argument(
        "--attention",
        type=str,
        required=True,
        help="Path to attention model hidden states"
    )
    parser.add_argument(
        "--property",
        type=str,
        default="identity",
        choices=["location", "identity", "category"],
        help="Property to analyze"
    )
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        choices=["location", "identity", "category"],
        help="Task filter"
    )
    parser.add_argument(
        "--n",
        type=int,
        default=None,
        help="N-back value filter"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/comparison",
        help="Output directory for results"
    )
    parser.add_argument(
        "--best_epoch",
        action="store_true",
        help="Restrict each model to its own best epoch (highest val_novel_identity_acc "
             "in its training_log.json). Without this, every saved epoch is pooled, "
             "which mixes untrained early checkpoints into the comparison."
    )
    parser.add_argument(
        "--baseline_epochs", type=int, nargs="*", default=None,
        help="Explicit epochs for the baseline model (overrides --best_epoch)"
    )
    parser.add_argument(
        "--attention_epochs", type=int, nargs="*", default=None,
        help="Explicit epochs for the attention model (overrides --best_epoch)"
    )
    parser.add_argument(
        "--split", type=str, default=None,
        choices=["val_novel_angle", "val_novel_identity"],
        help="Restrict to one validation split (default: pool both)"
    )

    args = parser.parse_args()

    baseline_root = Path(args.baseline)
    attention_root = Path(args.attention)
    output_dir = Path(args.output_dir)

    baseline_epochs, attention_epochs = args.baseline_epochs, args.attention_epochs
    if args.best_epoch:
        if baseline_epochs is None:
            be = find_best_epoch(baseline_root)
            baseline_epochs = [be] if be is not None else None
            print(f"Baseline best epoch: {be}")
        if attention_epochs is None:
            ae = find_best_epoch(attention_root)
            attention_epochs = [ae] if ae is not None else None
            print(f"Attention best epoch: {ae}")
    
    # Check paths exist
    if not baseline_root.exists():
        print(f"✗ Baseline path not found: {baseline_root}")
        return
    
    if not attention_root.exists():
        print(f"✗ Attention path not found: {attention_root}")
        return
    
    # Run comprehensive comparison
    results = comprehensive_comparison(
        baseline_root=baseline_root,
        attention_root=attention_root,
        property_name=args.property,
        task=args.task,
        n_value=args.n,
        baseline_epochs=baseline_epochs,
        attention_epochs=attention_epochs,
        split=args.split,
    )
    
    # Save results
    save_comparison_results(results, output_dir)
    
    # Summary
    print("\n" + "="*70)
    print("COMPARISON SUMMARY")
    print("="*70)
    
    if results.get('decoding'):
        print(f"\nDecoding: {results['decoding']['mean_improvement']:+.3f} improvement")
    
    if results.get('orthogonalization'):
        print(f"Orthogonalization: {results['orthogonalization']['improvement']:+.3f} improvement")
    
    if results.get('procrustes'):
        print(f"Procrustes reconstruction: {results['procrustes']['reconstruction_improvement']:+.3f} improvement")
    
    if results.get('swap_test'):
        st = results['swap_test']
        print(f"Swap test correct: {st['improvements']['correct']:+.3f} improvement "
              f"(decoded on '{st['decoded_property']}', NOT '{st['grouping_property']}')")
    
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()
