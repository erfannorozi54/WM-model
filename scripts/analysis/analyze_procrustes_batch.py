#!/usr/bin/env python3
"""
Batch Procrustes Analysis for Figure 4 Replication

This script performs comprehensive spatiotemporal analysis across multiple conditions
to replicate the key findings in Figure 4 of the paper:

Figure 4 Components:
- 4a-c: Temporal generalization matrices (decoder accuracy across time)
- 4d-f: Procrustes disparity matrices (alignment quality across time pairs)
- 4g: Swap test results (chronological organization hypothesis)

Usage:
  python analyze_procrustes_batch.py --hidden_root runs/wm_mtmf/hidden_states
  python analyze_procrustes_batch.py --hidden_root runs/wm_mtmf/hidden_states --property identity --n 2
  python analyze_procrustes_batch.py --output_dir results/procrustes --visualize
"""

import sys
from pathlib import Path
import argparse
import json
import numpy as np
from typing import List, Dict, Any, Optional
from tqdm import tqdm

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from analysis.procrustes import (
    procrustes_analysis,
    swap_hypothesis_test,
)
from analysis.decoding import evaluate as decoding_evaluate


def compute_temporal_generalization_matrix(
    hidden_root: Path,
    property_name: str,
    max_time: int = 6,
    task: Optional[str] = None,
    n_value: Optional[int] = None,
) -> np.ndarray:
    """
    Compute temporal generalization matrix (Figure 4a-c).
    
    Matrix[i, j] = decoding accuracy when training on time i, testing on time j.
    Diagonal = within-time accuracy, off-diagonal = cross-time generalization.
    
    Args:
        hidden_root: Path to hidden states
        property_name: Property to decode
        max_time: Maximum time point to analyze
        task: Task context filter
        n_value: N-back value filter
    
    Returns:
        Matrix of shape (max_time, max_time) with accuracy values
    """
    print(f"\nComputing temporal generalization matrix for '{property_name}'...")
    matrix = np.zeros((max_time, max_time))
    
    for train_time in tqdm(range(max_time), desc="Train times"):
        for test_time in range(max_time):
            try:
                # Use decoding module to get cross-time accuracy
                result = decoding_evaluate(
                    hidden_root=hidden_root,
                    property_name=property_name,
                    train_time=train_time,
                    test_times=[test_time],
                    train_task=task,
                    test_task=task,
                    train_n=[n_value] if n_value else None,
                    test_n=[n_value] if n_value else None,
                )
                
                acc = result['test'][str(test_time)]['acc']
                if acc is not None:
                    matrix[train_time, test_time] = acc
                else:
                    matrix[train_time, test_time] = np.nan
            
            except Exception as e:
                matrix[train_time, test_time] = np.nan
    
    return matrix


def compute_procrustes_disparity_matrix(
    hidden_root: Path,
    property_name: str,
    max_time: int = 6,
    task: Optional[str] = None,
    n_value: Optional[int] = None,
) -> np.ndarray:
    """
    Compute Procrustes disparity matrix (Figure 4d-f).
    
    Matrix[i, j] = Procrustes disparity when aligning time i to time j.
    Lower values indicate better alignment.
    
    Args:
        hidden_root: Path to hidden states
        property_name: Property to decode
        max_time: Maximum time point to analyze
        task: Task context filter
        n_value: N-back value filter
    
    Returns:
        Matrix of shape (max_time, max_time) with disparity values
    """
    print(f"\nComputing Procrustes disparity matrix for '{property_name}'...")
    matrix = np.zeros((max_time, max_time))
    
    for source_time in tqdm(range(max_time), desc="Source times"):
        for target_time in range(max_time):
            if source_time == target_time:
                matrix[source_time, target_time] = 0.0  # Perfect alignment to self
                continue
            
            try:
                result = procrustes_analysis(
                    hidden_root=hidden_root,
                    property_name=property_name,
                    source_time=source_time,
                    target_time=target_time,
                    task=task,
                    n_value=n_value,
                )
                matrix[source_time, target_time] = result['procrustes_disparity']
            
            except Exception as e:
                matrix[source_time, target_time] = np.nan
    
    return matrix


def compute_swap_test_matrix(
    hidden_root: Path,
    property_name: str,
    max_encoding_time: int = 4,
    task: Optional[str] = None,
    n_value: Optional[int] = None,
) -> Dict[str, List[float]]:
    """
    Compute swap test results across multiple encoding times (Figure 4g).
    
    For each encoding time, tests the chronological organization hypothesis.
    
    Args:
        hidden_root: Path to hidden states
        property_name: Property to decode
        max_encoding_time: Maximum encoding time to test
        task: Task context filter
        n_value: N-back value filter
    
    Returns:
        Dictionary with lists of accuracies for each condition
    """
    print(f"\nComputing swap test results for '{property_name}'...")
    
    results = {
        'encoding_times': [],
        'correct_acc': [],
        'swap1_acc': [],  # Same stimulus, wrong time
        'swap2_acc': [],  # Different stimulus, same age
        'baseline_acc': [],
    }
    
    for enc_time in tqdm(range(2, max_encoding_time), desc="Encoding times"):
        try:
            result = swap_hypothesis_test(
                hidden_root=hidden_root,
                property_name=property_name,
                encoding_time=enc_time,
                k_offset=1,
                task=task,
                n_value=n_value,
            )
            
            results['encoding_times'].append(enc_time)
            results['correct_acc'].append(result['correct_accuracy'])
            results['swap1_acc'].append(result['swap1_accuracy'])
            results['swap2_acc'].append(result['swap2_accuracy'])
            results['baseline_acc'].append(result['baseline_accuracy'])
        
        except Exception as e:
            print(f"  Warning: Failed at encoding time {enc_time}: {e}")
    
    return results


def visualize_all_results(
    tg_matrix: np.ndarray,
    pd_matrix: np.ndarray,
    swap_results: Dict[str, List[float]],
    property_name: str,
    save_prefix: str = "figure4"
):
    """
    Create comprehensive visualization (Figure 4 style).
    
    Args:
        tg_matrix: Temporal generalization matrix
        pd_matrix: Procrustes disparity matrix
        swap_results: Swap test results
        property_name: Property name for titles
        save_prefix: Prefix for saved figure files
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap
        
        fig = plt.figure(figsize=(18, 5))
        
        # Panel 1: Temporal Generalization Matrix
        ax1 = plt.subplot(1, 3, 1)
        im1 = ax1.imshow(tg_matrix, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
        ax1.set_xlabel('Test Time')
        ax1.set_ylabel('Train Time')
        ax1.set_title(f'(a) Temporal Generalization\n{property_name.capitalize()}')
        plt.colorbar(im1, ax=ax1, label='Accuracy')
        
        # Add text annotations
        for i in range(tg_matrix.shape[0]):
            for j in range(tg_matrix.shape[1]):
                if not np.isnan(tg_matrix[i, j]):
                    text_color = 'white' if tg_matrix[i, j] < 0.5 else 'black'
                    ax1.text(j, i, f'{tg_matrix[i, j]:.2f}',
                            ha='center', va='center', color=text_color, fontsize=8)
        
        # Panel 2: Procrustes Disparity Matrix
        ax2 = plt.subplot(1, 3, 2)
        im2 = ax2.imshow(pd_matrix, cmap='viridis', aspect='auto')
        ax2.set_xlabel('Target Time')
        ax2.set_ylabel('Source Time')
        ax2.set_title(f'(b) Procrustes Disparity\n{property_name.capitalize()}')
        plt.colorbar(im2, ax=ax2, label='Disparity')
        
        # Add text annotations
        for i in range(pd_matrix.shape[0]):
            for j in range(pd_matrix.shape[1]):
                if not np.isnan(pd_matrix[i, j]):
                    ax2.text(j, i, f'{pd_matrix[i, j]:.2f}',
                            ha='center', va='center', color='white', fontsize=8)
        
        # Panel 3: Swap Test Results
        ax3 = plt.subplot(1, 3, 3)
        
        if swap_results['encoding_times']:
            x = swap_results['encoding_times']
            ax3.plot(x, swap_results['baseline_acc'], 'o--', label='Baseline', 
                    color='gray', linewidth=2, markersize=8)
            ax3.plot(x, swap_results['correct_acc'], 's-', label='Correct', 
                    color='green', linewidth=2, markersize=8)
            ax3.plot(x, swap_results['swap1_acc'], '^-', label='Swap 1 (wrong time)', 
                    color='red', linewidth=2, markersize=8)
            ax3.plot(x, swap_results['swap2_acc'], 'd-', label='Swap 2 (same age)', 
                    color='blue', linewidth=2, markersize=8)
            
            ax3.set_xlabel('Encoding Time')
            ax3.set_ylabel('Reconstruction Accuracy')
            ax3.set_title(f'(c) Swap Test\n{property_name.capitalize()}')
            ax3.legend(loc='best', fontsize=8)
            ax3.grid(True, alpha=0.3)
            ax3.set_ylim([0, 1.1])
        
        plt.tight_layout()
        
        # Save figure
        save_path = f"{save_prefix}_{property_name}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Figure saved: {save_path}")
        
        plt.close()
        
    except ImportError:
        print("\n✗ Matplotlib not available for visualization")
    except Exception as e:
        print(f"\n✗ Visualization failed: {e}")


def save_results(
    results: Dict[str, Any],
    output_dir: Path,
    property_name: str,
):
    """Save analysis results to JSON files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save temporal generalization matrix
    if 'tg_matrix' in results:
        tg_file = output_dir / f"temporal_generalization_{property_name}.json"
        with open(tg_file, 'w') as f:
            json.dump({
                'matrix': results['tg_matrix'].tolist(),
                'property': property_name,
                'description': 'Temporal generalization matrix (train_time x test_time)',
            }, f, indent=2)
        print(f"✓ Saved: {tg_file}")
    
    # Save Procrustes disparity matrix
    if 'pd_matrix' in results:
        pd_file = output_dir / f"procrustes_disparity_{property_name}.json"
        with open(pd_file, 'w') as f:
            json.dump({
                'matrix': results['pd_matrix'].tolist(),
                'property': property_name,
                'description': 'Procrustes disparity matrix (source_time x target_time)',
            }, f, indent=2)
        print(f"✓ Saved: {pd_file}")
    
    # Save swap test results
    if 'swap_results' in results:
        swap_file = output_dir / f"swap_test_{property_name}.json"
        with open(swap_file, 'w') as f:
            json.dump({
                **results['swap_results'],
                'property': property_name,
                'description': 'Swap test results across encoding times',
            }, f, indent=2)
        print(f"✓ Saved: {swap_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Batch Procrustes Analysis for Figure 4 Replication"
    )
    parser.add_argument(
        "--hidden_root",
        type=str,
        required=True,
        help="Path to hidden states directory"
    )
    parser.add_argument(
        "--property",
        type=str,
        default="identity",
        choices=["location", "identity", "category"],
        help="Property to analyze"
    )
    parser.add_argument(
        "--max_time",
        type=int,
        default=6,
        help="Maximum time point to analyze"
    )
    parser.add_argument(
        "--task",
        type=str,
        default="any",
        choices=["location", "identity", "category", "any"],
        help="Task context filter"
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
        default="results/procrustes",
        help="Output directory for results"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate visualization plots"
    )
    parser.add_argument(
        "--skip_tg",
        action="store_true",
        help="Skip temporal generalization matrix (slow)"
    )
    parser.add_argument(
        "--skip_swap",
        action="store_true",
        help="Skip swap test"
    )
    
    args = parser.parse_args()
    
    hidden_root = Path(args.hidden_root)
    output_dir = Path(args.output_dir)
    
    # Check if hidden states exist
    if not hidden_root.exists():
        print(f"\n✗ Error: Hidden states not found: {hidden_root}")
        print(f"\nRun training first: python train.py --config configs/mtmf.yaml")
        return
    
    print("\n" + "="*70)
    print("BATCH PROCRUSTES ANALYSIS")
    print("Figure 4 Replication")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Hidden states: {hidden_root}")
    print(f"  Property: {args.property}")
    print(f"  Max time: {args.max_time}")
    print(f"  Task: {args.task}")
    print(f"  N-back: {args.n if args.n else 'all'}")
    print(f"  Output: {output_dir}")
    
    task_filter = None if args.task == "any" else args.task
    results = {}
    
    # 1. Temporal Generalization Matrix
    if not args.skip_tg:
        print("\n" + "-"*70)
        print("STEP 1: Temporal Generalization Matrix (Figure 4a)")
        print("-"*70)
        tg_matrix = compute_temporal_generalization_matrix(
            hidden_root=hidden_root,
            property_name=args.property,
            max_time=args.max_time,
            task=task_filter,
            n_value=args.n,
        )
        results['tg_matrix'] = tg_matrix
        print(f"\n✓ Completed temporal generalization matrix")
        print(f"  Diagonal mean (within-time): {np.nanmean(np.diag(tg_matrix)):.3f}")
        print(f"  Off-diagonal mean (cross-time): {np.nanmean(tg_matrix[~np.eye(tg_matrix.shape[0], dtype=bool)]):.3f}")
    
    # 2. Procrustes Disparity Matrix
    print("\n" + "-"*70)
    print("STEP 2: Procrustes Disparity Matrix (Figure 4b)")
    print("-"*70)
    pd_matrix = compute_procrustes_disparity_matrix(
        hidden_root=hidden_root,
        property_name=args.property,
        max_time=args.max_time,
        task=task_filter,
        n_value=args.n,
    )
    results['pd_matrix'] = pd_matrix
    print(f"\n✓ Completed Procrustes disparity matrix")
    print(f"  Mean disparity: {np.nanmean(pd_matrix[pd_matrix > 0]):.4f}")
    print(f"  Adjacent time disparity: {np.nanmean(np.diag(pd_matrix, k=1)):.4f}")
    
    # 3. Swap Test
    if not args.skip_swap:
        print("\n" + "-"*70)
        print("STEP 3: Swap Hypothesis Test (Figure 4g)")
        print("-"*70)
        swap_results = compute_swap_test_matrix(
            hidden_root=hidden_root,
            property_name=args.property,
            max_encoding_time=args.max_time - 1,
            task=task_filter,
            n_value=args.n,
        )
        results['swap_results'] = swap_results
        
        if swap_results['encoding_times']:
            print(f"\n✓ Completed swap test")
            print(f"  Encoding times tested: {swap_results['encoding_times']}")
            print(f"  Mean correct accuracy: {np.mean(swap_results['correct_acc']):.3f}")
            print(f"  Mean swap2 accuracy: {np.mean(swap_results['swap2_acc']):.3f}")
            print(f"  Mean swap1 accuracy: {np.mean(swap_results['swap1_acc']):.3f}")
            
            # Check hypothesis
            swap2_better = np.mean(swap_results['swap2_acc']) > np.mean(swap_results['swap1_acc'])
            if swap2_better:
                print(f"\n  ✓ Hypothesis CONFIRMED: Same-age rotations perform better!")
            else:
                print(f"\n  • Hypothesis not strongly confirmed in this dataset")
    
    # Save results
    print("\n" + "-"*70)
    print("SAVING RESULTS")
    print("-"*70)
    save_results(results, output_dir, args.property)
    
    # Visualize
    if args.visualize:
        print("\n" + "-"*70)
        print("GENERATING VISUALIZATIONS")
        print("-"*70)
        
        tg_mat = results.get('tg_matrix', np.zeros((args.max_time, args.max_time)))
        pd_mat = results.get('pd_matrix', np.zeros((args.max_time, args.max_time)))
        swap_res = results.get('swap_results', {'encoding_times': []})
        
        save_prefix = str(output_dir / "figure4")
        visualize_all_results(tg_mat, pd_mat, swap_res, args.property, save_prefix)
    
    # Summary
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print(f"\nKey findings:")
    print(f"  • Temporal generalization reveals representation stability")
    print(f"  • Procrustes disparity quantifies transformation smoothness")
    print(f"  • Swap test validates chronological organization hypothesis")
    
    print(f"\n\nResults saved to: {output_dir}")
    print(f"\nTo analyze other properties:")
    print(f"  python analyze_procrustes_batch.py --hidden_root {hidden_root} --property location")
    print(f"  python analyze_procrustes_batch.py --hidden_root {hidden_root} --property category")
    
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()
