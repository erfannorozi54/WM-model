#!/usr/bin/env python3
"""
Procrustes Analysis for Spatiotemporal Transformations in Working Memory.

This module implements Orthogonal Procrustes alignment to study how neural 
representations transform over time. Key analyses include:

1. Computing decoder weights at different time points
2. Finding optimal rotation matrices between time points using Procrustes
3. Testing chronological memory subspace hypothesis via "swap" experiments

The core finding (replicating Figure 4g in the paper):
- Representations transform in a chronologically-organized manner
- Rotation matrices preserve relative temporal structure
- Swapping with same-age rotations maintains accuracy

Usage:

# Compute Procrustes alignment between time points
python -m src.analysis.procrustes \
  --hidden_root runs/wm_mtmf/hidden_states \
  --property identity \
  --source_time 2 --target_time 3 \
  --task location --n 2

# Run swap hypothesis test
python -m src.analysis.procrustes \
  --hidden_root runs/wm_mtmf/hidden_states \
  --property identity \
  --swap_test \
  --encoding_time 2 \
  --k_offset 1 \
  --task location --n 2
"""

from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
import argparse
import numpy as np
import torch
from scipy.linalg import orthogonal_procrustes
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

from .activations import load_payloads, build_matrix, TASK_INDEX_TO_NAME
from .orthogonalization import one_vs_rest_weights

PROPERTY_CHOICES = ["location", "identity", "category"]
TASK_CHOICES = ["location", "identity", "category", "any"]


def _task_name_to_index(name: Optional[str]) -> Optional[int]:
    """Convert task name to index."""
    if name is None or name == "any":
        return None
    for k, v in TASK_INDEX_TO_NAME.items():
        if v == name:
            return k
    raise ValueError(f"Unknown task name: {name}")


def compute_procrustes_alignment(
    W_source: Dict[int, np.ndarray],
    W_target: Dict[int, np.ndarray],
) -> Tuple[np.ndarray, float]:
    """
    Compute Orthogonal Procrustes alignment between two sets of decoder weights.
    
    Finds the optimal rotation matrix R that aligns W_source to W_target:
        W_target ≈ W_source @ R
    
    Args:
        W_source: Dictionary mapping class labels to weight vectors (source time)
        W_target: Dictionary mapping class labels to weight vectors (target time)
    
    Returns:
        R: Optimal rotation matrix (d, d)
        disparity: Procrustes disparity (measure of alignment quality)
    
    Notes:
        - Only uses classes present in both source and target
        - Weight vectors should be unit-normalized
    """
    # Find common classes
    common_classes = sorted(set(W_source.keys()) & set(W_target.keys()))
    
    if len(common_classes) < 2:
        raise ValueError(f"Need at least 2 common classes for Procrustes, found {len(common_classes)}")
    
    # Stack weight vectors into matrices (n_classes, d)
    A = np.stack([W_source[c] for c in common_classes], axis=0)  # source
    B = np.stack([W_target[c] for c in common_classes], axis=0)  # target
    
    # Compute optimal rotation: B ≈ A @ R
    R, disparity = orthogonal_procrustes(A, B)
    
    return R, float(disparity)


def reconstruct_weights(
    W_source: Dict[int, np.ndarray],
    R: np.ndarray,
) -> Dict[int, np.ndarray]:
    """
    Reconstruct target weights by applying rotation matrix to source weights.
    
    Args:
        W_source: Source decoder weights
        R: Rotation matrix
    
    Returns:
        W_reconstructed: Reconstructed weights (W_source @ R)
    """
    W_recon = {}
    for c, w in W_source.items():
        w_rot = w @ R
        # Re-normalize
        w_rot = w_rot / (np.linalg.norm(w_rot) + 1e-12)
        W_recon[c] = w_rot
    return W_recon


def evaluate_reconstruction(
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    W_reconstructed: Dict[int, np.ndarray],
    normalize: bool = True,
) -> float:
    """
    Evaluate reconstruction quality by building classifiers from reconstructed weights.
    
    Args:
        X_test: Test features (N, d)
        y_test: Test labels (N,)
        W_reconstructed: Reconstructed decoder weights
        normalize: Whether to standardize features before prediction
    
    Returns:
        accuracy: Classification accuracy using reconstructed weights
    """
    X_np = X_test.numpy()
    y_np = y_test.numpy()
    
    # Standardize features if requested
    if normalize:
        scaler = StandardScaler(with_mean=True, with_std=True)
        X_np = scaler.fit_transform(X_np)
    
    # For each sample, compute scores with all class weights
    classes = sorted(W_reconstructed.keys())
    scores = np.zeros((len(X_np), len(classes)))
    
    for i, c in enumerate(classes):
        w = W_reconstructed[c]
        scores[:, i] = X_np @ w  # Linear decision function
    
    # Predict class with highest score
    y_pred = np.array([classes[i] for i in scores.argmax(axis=1)])
    
    # Compute accuracy
    acc = accuracy_score(y_np, y_pred)
    return float(acc)


def procrustes_analysis(
    hidden_root: Path,
    property_name: str,
    source_time: int,
    target_time: int,
    task: Optional[str] = None,
    n_value: Optional[int] = None,
    epochs: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """
    Perform Procrustes analysis between two time points.
    
    Args:
        hidden_root: Path to saved hidden states
        property_name: Property to decode (location|identity|category)
        source_time: Source time point
        target_time: Target time point
        task: Task context filter
        n_value: N-back value filter
        epochs: Specific epochs to analyze
    
    Returns:
        Dictionary with rotation matrix, disparity, and reconstruction accuracy
    """
    payloads = load_payloads(Path(hidden_root), epochs=epochs)
    ti = _task_name_to_index(task)
    
    # Build matrices for source and target times
    X_source, y_source, label2idx_source = build_matrix(
        payloads, property_name, time=source_time, task_index=ti, n_value=n_value
    )
    X_target, y_target, label2idx_target = build_matrix(
        payloads, property_name, time=target_time, task_index=ti, n_value=n_value
    )
    
    if X_source.numel() == 0 or X_target.numel() == 0:
        raise RuntimeError("Insufficient samples for Procrustes analysis")
    
    # Compute decoder weights at both time points
    W_source = one_vs_rest_weights(X_source, y_source)
    W_target = one_vs_rest_weights(X_target, y_target)
    
    # Compute Procrustes alignment
    R, disparity = compute_procrustes_alignment(W_source, W_target)
    
    # Reconstruct target weights using rotation
    W_reconstructed = reconstruct_weights(W_source, R)
    
    # Evaluate reconstruction on target data
    recon_acc = evaluate_reconstruction(X_target, y_target, W_reconstructed)
    
    # Baseline: direct decoding accuracy at target time
    baseline_acc = evaluate_reconstruction(X_target, y_target, W_target)
    
    return {
        "property": property_name,
        "source_time": source_time,
        "target_time": target_time,
        "task": task,
        "n": n_value,
        "n_classes": len(W_source),
        "rotation_shape": list(R.shape),
        "procrustes_disparity": disparity,
        "reconstruction_accuracy": recon_acc,
        "baseline_accuracy": baseline_acc,
        "accuracy_ratio": recon_acc / baseline_acc if baseline_acc > 0 else 0.0,
    }


def _split_payloads_by_stimulus(
    payloads: List[Dict], property_name: str, time: int
) -> Tuple[List[Dict], List[Dict]]:
    """Split payloads into two groups based on stimulus identity hash at a given time.
    
    Uses the stimulus identity (or property value) at the specified time to assign
    each trial to group A or B via hash parity. This enables cross-stimulus
    generalization tests for Procrustes swap analysis.
    
    Returns:
        (payloads_group_a, payloads_group_b) — each with the same structure
        but with only the assigned trials kept.
    """
    import copy
    
    def _trial_hash(trial_identities, b, t, prop):
        if prop == "location" and trial_identities.get("locations") is not None:
            locs = trial_identities["locations"]
            if torch.is_tensor(locs) and locs.dim() == 2:
                val = int(locs[b, t]) if t < locs.shape[1] else 0
            else:
                val = 0
            return hash(str(val))
        elif prop == "identity" and trial_identities.get("identities") is not None:
            ids = trial_identities["identities"]
            val = ids[b][t] if b < len(ids) and t < len(ids[b]) else ""
            return hash(str(val))
        elif prop == "category" and trial_identities.get("categories") is not None:
            cats = trial_identities["categories"]
            val = cats[b][t] if b < len(cats) and t < len(cats[b]) else ""
            return hash(str(val))
        return hash(b)
    
    result_a, result_b = [], []
    for payload in payloads:
        B = payload["hidden"].shape[0]
        T = payload["hidden"].shape[1]
        mask_a = torch.zeros(B, dtype=torch.bool)
        mask_b = torch.zeros(B, dtype=torch.bool)
        for b in range(B):
            h = _trial_hash(payload, b, min(time, T - 1), property_name)
            if h % 2 == 0:
                mask_a[b] = True
            else:
                mask_b[b] = True
        
        if mask_a.any():
            pa = copy.deepcopy(payload)
            for key in ("hidden", "logits", "task_vector", "task_index", "n", "targets"):
                if key in pa and torch.is_tensor(pa[key]):
                    pa[key] = pa[key][mask_a]
            for key in ("locations",):
                if key in pa and pa[key] is not None and torch.is_tensor(pa[key]):
                    pa[key] = pa[key][mask_a]
            for key in ("identities", "categories"):
                if key in pa and pa[key] is not None:
                    pa[key] = [pa[key][i] for i in range(len(pa[key])) if mask_a[i]]
            pa["split"] = "group_a"
            result_a.append(pa)
        
        if mask_b.any():
            pb = copy.deepcopy(payload)
            for key in ("hidden", "logits", "task_vector", "task_index", "n", "targets"):
                if key in pb and torch.is_tensor(pb[key]):
                    pb[key] = pb[key][mask_b]
            for key in ("locations",):
                if key in pb and pb[key] is not None and torch.is_tensor(pb[key]):
                    pb[key] = pb[key][mask_b]
            for key in ("identities", "categories"):
                if key in pb and pb[key] is not None:
                    pb[key] = [pb[key][i] for i in range(len(pb[key])) if mask_b[i]]
            pb["split"] = "group_b"
            result_b.append(pb)
    
    if not result_a or not result_b:
        print("  ⚠ Could not split payloads into two groups, using random split")
        mid = len(payloads) // 2
        return payloads[:mid], payloads[mid:]
    
    return result_a, result_b


def swap_hypothesis_test(
    hidden_root: Path,
    property_name: str,
    encoding_time: int,
    k_offset: int = 1,
    task: Optional[str] = None,
    n_value: Optional[int] = None,
    epochs: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """
    Test chronological memory subspace hypothesis using rotation matrix swaps.

    Paper's Full Test (Figure 4g, Equations 2-3):
    -----------------------------------------------
    - Eq. 2 (Time Swap): R(T=j+1→j+2) applied to W(T=j)
      Tests if rotation from a different time interval generalizes
    - Eq. 3 (Stimulus Swap): R_A(T=j→j+1) applied to W_B(T=j)
      Tests if rotation from different stimuli (group A) at same age generalizes
      to held-out stimuli (group B)

    KEY FINDING: Eq. 3 > Eq. 2 (chronological organization dominates)

    Implementation:
    ---------------
    - For label alignment, decodes on `location` (4 fixed classes) — identity labels
      differ per trial and would not align between stimulus groups
    - Splits trials by identity hash (group A vs group B) for cross-stimulus effect
    - "Correct" rotation: R(T=j → T=j+1) on pooled data → test on held-out B
    - "Swap 1" (wrong time): R(T=j+k → T=j+k+1) applied to T=j weights → test on B
    - "Swap 2" (same age, diff stimuli): R_A(T=j → T=j+1) applied to W_B(T=j) → test on B
    - Baseline: direct decoding at T=j+1 on B

    Args:
        hidden_root: Path to saved hidden states
        property_name: Original property requested (used for label alignment via location)
        encoding_time: Initial encoding time j (typically 0-2)
        k_offset: Temporal offset for swap comparison (default=1)
        task: Task context filter (location/identity/category or None for all)
        n_value: N-back value filter (1/2/3 or None for all)
        epochs: Specific epochs to analyze (None for all)

    Returns:
        Dictionary with reconstruction accuracies and disparities
    """
    payloads = load_payloads(Path(hidden_root), epochs=epochs)
    ti = _task_name_to_index(task)

    # Use location (4 fixed classes) for label alignment between groups.
    # Identity labels are unique per trial, so they cannot be aligned between
    # disjoint stimulus groups (A and B have different identity sets).
    swap_property = "location"

    # Split payloads by identity (disjoint stimulus groups) — required for
    # cross-stimulus test, even though we decode on location
    print(f"  Splitting payloads by identity at time {encoding_time}...")
    payloads_a, payloads_b = _split_payloads_by_stimulus(payloads, "identity", encoding_time)
    n_a = sum(p["hidden"].shape[0] for p in payloads_a)
    n_b = sum(p["hidden"].shape[0] for p in payloads_b)
    print(f"    Group A: {n_a} trials, Group B: {n_b} trials")

    # Define time points
    j = encoding_time
    k = k_offset
    times = [j, j + 1, j + k, j + k + 1]

    # Build matrices for full data (used for correct and swap1)
    weights_full = {}
    for t in times:
        X, y, label2idx = build_matrix(
            payloads, swap_property, time=t, task_index=ti, n_value=n_value
        )
        if X.numel() == 0:
            raise RuntimeError(f"No samples at time {t}")
        weights_full[t] = one_vs_rest_weights(X, y)

    # Build matrices for group A (used for swap2 rotation)
    weights_a = {}
    matrices_a = {}
    for t in [j, j + 1]:
        X, y, label2idx = build_matrix(
            payloads_a, swap_property, time=t, task_index=ti, n_value=n_value
        )
        if X.numel() == 0:
            raise RuntimeError(f"No samples in group A at time {t}")
        matrices_a[t] = (X, y, label2idx)
        weights_a[t] = one_vs_rest_weights(X, y)

    # Build matrices for group B (used as held-out test set)
    matrices_b = {}
    weights_b = {}
    for t in [j, j + 1]:
        X, y, label2idx = build_matrix(
            payloads_b, swap_property, time=t, task_index=ti, n_value=n_value
        )
        if X.numel() == 0:
            raise RuntimeError(f"No samples in group B at time {t}")
        matrices_b[t] = (X, y, label2idx)
        weights_b[t] = one_vs_rest_weights(X, y)

    # ===== CORRECT ROTATION (pooled data) =====
    R_correct, disp_correct = compute_procrustes_alignment(
        weights_full[j], weights_full[j + 1]
    )

    # ===== SWAP 1: Different time interval (wrong time) =====
    R_swap1, disp_swap1 = compute_procrustes_alignment(
        weights_full[j + k], weights_full[j + k + 1]
    )

    # ===== SWAP 2: Different stimuli, same age =====
    # Compute R on group A data at the same time pair (j→j+1)
    R_swap2, disp_swap2 = compute_procrustes_alignment(
        weights_a[j], weights_a[j + 1]
    )

    # Test on group B held-out data
    X_target, y_target, _ = matrices_b[j + 1]
    W_target = weights_b[j + 1]
    W_source_b = weights_b[j]

    # Correct: apply pooled R to group B source weights → test on group B target
    W_recon_correct = reconstruct_weights(W_source_b, R_correct)
    acc_correct = evaluate_reconstruction(X_target, y_target, W_recon_correct)

    # Swap 1: apply wrong-time R to group B source → test on group B target
    W_recon_swap1 = reconstruct_weights(W_source_b, R_swap1)
    acc_swap1 = evaluate_reconstruction(X_target, y_target, W_recon_swap1)

    # Swap 2: apply group A's same-age R to group B source → test on group B target
    W_recon_swap2 = reconstruct_weights(W_source_b, R_swap2)
    acc_swap2 = evaluate_reconstruction(X_target, y_target, W_recon_swap2)

    # Baseline: direct decoding at T=j+1 on group B
    acc_baseline = evaluate_reconstruction(X_target, y_target, W_target)

    return {
        "property": swap_property,
        "original_property": property_name,
        "encoding_time": j,
        "k_offset": k,
        "task": task,
        "n": n_value,
        "n_classes": len(weights_full[j]),
        "n_group_a": n_a,
        "n_group_b": n_b,
        # Correct transformation
        "correct_disparity": disp_correct,
        "correct_accuracy": acc_correct,
        # Swap 1: Same stimulus, different time
        "swap1_disparity": disp_swap1,
        "swap1_accuracy": acc_swap1,
        # Swap 2: Different stimulus, same age
        "swap2_disparity": disp_swap2,
        "swap2_accuracy": acc_swap2,
        # Baseline
        "baseline_accuracy": acc_baseline,
        # Normalized accuracies (relative to correct)
        "swap1_relative": acc_swap1 / acc_correct if acc_correct > 0 else 0.0,
        "swap2_relative": acc_swap2 / acc_correct if acc_correct > 0 else 0.0,
        # Key result: swap2 should be closer to correct than swap1
        "hypothesis_confirmed": bool(abs(acc_swap2 - acc_correct) < abs(acc_swap1 - acc_correct)),
        "note": "Decoded on location (aligned labels). Groups split by identity hash for cross-stimulus effect."
    }


def main():
    parser = argparse.ArgumentParser(
        description="Procrustes analysis for spatiotemporal transformations"
    )
    parser.add_argument("--hidden_root", type=str, required=True,
                       help="Path to runs/<exp>/hidden_states")
    parser.add_argument("--property", type=str, choices=PROPERTY_CHOICES, required=True,
                       help="Property to decode")
    
    # Mode selection
    parser.add_argument("--swap_test", action="store_true",
                       help="Run swap hypothesis test (Figure 4g)")
    
    # Standard Procrustes parameters
    parser.add_argument("--source_time", type=int,
                       help="Source time point")
    parser.add_argument("--target_time", type=int,
                       help="Target time point")
    
    # Swap test parameters
    parser.add_argument("--encoding_time", type=int,
                       help="Encoding time for swap test")
    parser.add_argument("--k_offset", type=int, default=1,
                       help="Temporal offset for swap comparison")
    
    # Filtering parameters
    parser.add_argument("--task", type=str, choices=TASK_CHOICES, default="any",
                       help="Task context")
    parser.add_argument("--n", type=int, default=None,
                       help="N-back value")
    parser.add_argument("--epochs", type=int, nargs="*",
                       help="Specific epochs to analyze")
    
    args = parser.parse_args()
    
    if args.swap_test:
        # Run swap hypothesis test
        if args.encoding_time is None:
            parser.error("--encoding_time required for swap test")
        
        result = swap_hypothesis_test(
            hidden_root=Path(args.hidden_root),
            property_name=args.property,
            encoding_time=args.encoding_time,
            k_offset=args.k_offset,
            task=None if args.task == "any" else args.task,
            n_value=args.n,
            epochs=args.epochs,
        )
    else:
        # Run standard Procrustes analysis
        if args.source_time is None or args.target_time is None:
            parser.error("--source_time and --target_time required for Procrustes")
        
        result = procrustes_analysis(
            hidden_root=Path(args.hidden_root),
            property_name=args.property,
            source_time=args.source_time,
            target_time=args.target_time,
            task=None if args.task == "any" else args.task,
            n_value=args.n,
            epochs=args.epochs,
        )
    
    import json
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
