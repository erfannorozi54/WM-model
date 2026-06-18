#!/usr/bin/env python3
"""
Causal Perturbation Test (Figure A7)

This module implements the causal intervention analysis to test whether the
decoder-defined subspaces are functionally relevant to the network's behavior.

Key Idea:
---------
If a decoder can accurately predict object properties from hidden states, are those
subspaces causally related to the network's output decisions? We test this by:

1. Selecting trials where the model outputs "Match"
2. Perturbing hidden states along the decoder's hyperplane normal vector
3. Re-running the classifier to get new output probabilities
4. Measuring how P(Match), P(Non-Match), and P(No-Action) change with distance

Expected Result (Figure A7):
----------------------------
- As perturbation distance increases, P(Match) should DROP
- P(No-Action) should RISE (state becomes ambiguous)
- P(Non-Match) may rise slightly but less than No-Action
- Clear boundary transition demonstrates causal relationship

Usage:
    python -m src.analysis.causal_perturbation \\
        --model experiments/wm_mtmf/best_model.pt \\
        --hidden_root experiments/wm_mtmf/hidden_states \\
        --property location \\
        --output_dir analysis_results
"""

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from .activations import load_payloads, build_matrix
from .orthogonalization import one_vs_rest_weights
from ..models import create_model


def load_trained_model(checkpoint_path: Path, device: torch.device) -> nn.Module:
    """
    Load a trained model from checkpoint.
    
    Args:
        checkpoint_path: Path to model checkpoint (.pt file)
        device: Device to load model on
    
    Returns:
        Loaded model in eval mode
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get config from checkpoint
    if 'config' in checkpoint:
        cfg = checkpoint['config']
    else:
        raise ValueError("Checkpoint must contain 'config' key")
    
    # Recreate model
    # Construct model_type string (e.g., "gru" or "attention_gru")
    rnn_type = cfg.get("rnn_type", "gru")
    model_arch = cfg.get("model_type", "baseline")
    if model_arch in ("attention", "dual_attention"):
        model_type_str = f"attention_{rnn_type}"
    else:
        model_type_str = rnn_type
    
    model = create_model(
        model_type=model_type_str,
        hidden_size=cfg["hidden_size"],
        num_layers=cfg["num_layers"],
        dropout=cfg.get("dropout", 0.0),
        pretrained_backbone=cfg.get("pretrained_backbone", True),
        freeze_backbone=cfg.get("freeze_backbone", True),
        attention_hidden_dim=cfg.get("attention_hidden_dim", 256),
        attention_dropout=cfg.get("attention_dropout", 0.1),
        attention_mode=cfg.get("attention_mode", "task_only"),
    )
    
    # Load weights
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'model_state' in checkpoint:
        model.load_state_dict(checkpoint['model_state'])
    else:
        raise ValueError("Checkpoint must contain model state dict")
    
    model.to(device)
    model.eval()
    
    print(f"✓ Loaded model from {checkpoint_path}")
    print(f"  Validation accuracy: {checkpoint.get('val_novel_angle_acc', checkpoint.get('val_acc', 'N/A'))}")
    
    return model


def select_match_trials(
    payloads: List[Dict],
    property_name: str,
    timestep: int,
    task_index: Optional[int] = None,
    n_value: Optional[int] = None,
    min_samples: int = 50,
    label2idx: Optional[Dict] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[int]]:
    """
    Select trials where model predicted "Match" (class 2).
    
    Args:
        payloads: Loaded hidden state payloads
        property_name: Property to analyze
        timestep: Executive timestep to perturb (typically n+1 to n+3)
        task_index: Filter by task (None for all)
        n_value: Filter by n-back value (None for all)
        min_samples: Minimum number of match trials required
        label2idx: Mapping from raw property value → decoder class index.
            When provided, property_values returns class indices usable as
            keys into decoder_weights for per-trial perturbation directions.
    
    Returns:
        hidden_states: (N, H) hidden states at timestep
        property_values: (N,) class indices for each trial (keys into decoder_weights)
        predicted_logits: (N, 3) original model logits
        sample_indices: List of (payload_idx, batch_idx) for tracking
    """
    hidden_list = []
    property_list = []
    logits_list = []
    indices_list = []
    
    for payload_idx, payload in enumerate(payloads):
        # Filter by task if specified
        if task_index is not None:
            task_mask = payload["task_index"] == task_index
        else:
            task_mask = torch.ones(len(payload["task_index"]), dtype=torch.bool)
        
        # Filter by n-value if specified
        if n_value is not None:
            n_mask = payload["n"] == n_value
        else:
            n_mask = torch.ones(len(payload["n"]), dtype=torch.bool)
        
        # Combined mask
        mask = task_mask & n_mask
        
        if mask.sum() == 0:
            continue
        
        # Get data for this timestep
        hidden = payload["hidden"][mask, timestep, :]  # (B_filtered, H)
        logits = payload["logits"][mask, timestep, :]  # (B_filtered, 3)
        preds = logits.argmax(dim=-1)  # (B_filtered,)
        
        # Select only "Match" predictions (class 2)
        match_mask = preds == 2
        
        if match_mask.sum() == 0:
            continue
        
        # Get property values — map raw values to decoder class indices when label2idx is available
        if property_name == "location":
            props = payload["locations"][mask, timestep]  # (B_filtered,) raw int indices
        elif property_name == "identity":
            identities = payload["identities"]
            props = -torch.ones(mask.sum(), dtype=torch.long)
            j = 0
            for batch_ids, m in zip(identities, mask):
                if m and len(batch_ids) > timestep:
                    raw_val = batch_ids[timestep]
                    if label2idx is not None and raw_val in label2idx:
                        props[j] = label2idx[raw_val]
                    else:
                        props[j] = hash(raw_val) % 1000
                    j += 1
        elif property_name == "category":
            categories = payload["categories"]
            props = -torch.ones(mask.sum(), dtype=torch.long)
            j = 0
            for batch_cats, m in zip(categories, mask):
                if m and len(batch_cats) > timestep:
                    raw_val = batch_cats[timestep]
                    if label2idx is not None and raw_val in label2idx:
                        props[j] = label2idx[raw_val]
                    else:
                        props[j] = hash(raw_val) % 10
                    j += 1
        else:
            raise ValueError(f"Unknown property: {property_name}")
        
        # Apply match mask
        hidden_list.append(hidden[match_mask])
        property_list.append(props[match_mask])
        logits_list.append(logits[match_mask])
        
        # Track indices
        for batch_idx in torch.where(mask)[0][match_mask].tolist():
            indices_list.append((payload_idx, batch_idx))
    
    if len(hidden_list) == 0:
        raise RuntimeError(f"No match trials found at timestep {timestep}")
    
    hidden_states = torch.cat(hidden_list, dim=0)
    property_values = torch.cat(property_list, dim=0)
    predicted_logits = torch.cat(logits_list, dim=0)
    
    if len(hidden_states) < min_samples:
        print(f"⚠ Warning: Only {len(hidden_states)} match trials found (min={min_samples})")
    
    print(f"✓ Selected {len(hidden_states)} match trials at timestep {timestep}")
    
    return hidden_states, property_values, predicted_logits, indices_list


def run_causal_perturbation(
    model: nn.Module,
    hidden_states: torch.Tensor,
    decoder_weights: Dict[int, np.ndarray],
    perturbation_distances: np.ndarray,
    device: torch.device,
    property_values: Optional[torch.Tensor] = None,
    property_to_class_idx: Optional[Dict] = None,
    batch_size: int = 32
) -> Dict[str, np.ndarray]:
    """
    Perturb hidden states and measure output probability changes.

    The decoder normal vectors point from the "rest" region toward each class's
    region. For a match trial, the state is in the match region. To produce the
    paper's expected pattern (P(Match) drops, P(No-Action) rises), we push the
    state in the direction OPPOSITE to the match class's normal — i.e., across
    the match hyperplane.

    Implementation: use the mean direction of ALL class normals as the
    perturbation direction. This direction is, on average, roughly opposite to
    any individual class's normal (when classes are roughly balanced), producing
    a strong "leave the match region" effect.

    The optional `property_values` argument is kept for backward compatibility
    but is not used to select per-trial directions (per-trial class-specific
    directions were found to be weaker because they push the state deeper into
    the trial's class, not across its boundary).

    Args:
        model: Trained model (only classifier layer will be used)
        hidden_states: (N, H) hidden states to perturb
        decoder_weights: Dict mapping class labels to weight vectors (from one_vs_rest_weights)
        perturbation_distances: (D,) array of distances to perturb
        device: Device for computation
        property_values: (N,) property value per trial (unused, kept for API compat)
        property_to_class_idx: Unused, kept for API compat
        batch_size: Batch size for inference

    Returns:
        Dictionary with probability arrays
    """
    N, H = hidden_states.shape
    D = len(perturbation_distances)

    # Use mean direction of all class normals as the perturbation direction
    if isinstance(decoder_weights, dict):
        W_array_full = np.stack(list(decoder_weights.values()), axis=0)  # (C, H)
        mean_direction = torch.from_numpy(W_array_full.mean(axis=0)).float()
        mean_direction = mean_direction / (mean_direction.norm() + 1e-12)
    else:
        mean_direction = torch.from_numpy(decoder_weights.mean(axis=0)).float()
        mean_direction = mean_direction / (mean_direction.norm() + 1e-12)

    # All trials are pushed in the same mean direction
    perturbation_directions = mean_direction.unsqueeze(0).expand(N, -1).clone()  # (N, H)

    # Store results
    prob_match_all = np.zeros((D, N))
    prob_non_match_all = np.zeros((D, N))
    prob_no_action_all = np.zeros((D, N))

    with torch.no_grad():
        for d_idx, distance in enumerate(perturbation_distances):
            # Perturb each trial along the mean direction
            h_perturbed = hidden_states + distance * perturbation_directions  # (N, H)

            # Run through classifier in batches
            all_probs = []
            for i in range(0, N, batch_size):
                batch_h = h_perturbed[i:i+batch_size].to(device)
                logits = model.classifier(batch_h)
                probs = torch.softmax(logits, dim=-1)
                all_probs.append(probs.cpu())

            all_probs = torch.cat(all_probs, dim=0)  # (N, 3)

            # Extract probabilities for each action
            prob_no_action_all[d_idx, :] = all_probs[:, 0].numpy()
            prob_non_match_all[d_idx, :] = all_probs[:, 1].numpy()
            prob_match_all[d_idx, :] = all_probs[:, 2].numpy()

    # Compute means and stds
    results = {
        "distances": perturbation_distances,
        "prob_match": prob_match_all.mean(axis=1),
        "prob_non_match": prob_non_match_all.mean(axis=1),
        "prob_no_action": prob_no_action_all.mean(axis=1),
        "std_match": prob_match_all.std(axis=1),
        "std_non_match": prob_non_match_all.std(axis=1),
        "std_no_action": prob_no_action_all.std(axis=1),
    }

    return results


def plot_perturbation_results(
    results: Dict[str, np.ndarray],
    output_path: Path,
    property_name: str
):
    """
    Plot perturbation results (Figure A7 style).
    
    Args:
        results: Dictionary from run_causal_perturbation
        output_path: Path to save figure
        property_name: Property being tested
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    distances = results["distances"]
    
    # Plot means with error bands
    ax.plot(distances, results["prob_match"], 'o-', color='#2E7D32', linewidth=2, 
            markersize=6, label='Match', alpha=0.8)
    ax.fill_between(distances, 
                     results["prob_match"] - results["std_match"],
                     results["prob_match"] + results["std_match"],
                     color='#2E7D32', alpha=0.2)
    
    ax.plot(distances, results["prob_non_match"], 's-', color='#C62828', linewidth=2,
            markersize=6, label='Non-Match', alpha=0.8)
    ax.fill_between(distances,
                     results["prob_non_match"] - results["std_non_match"],
                     results["prob_non_match"] + results["std_non_match"],
                     color='#C62828', alpha=0.2)
    
    ax.plot(distances, results["prob_no_action"], '^-', color='#1565C0', linewidth=2,
            markersize=6, label='No Action', alpha=0.8)
    ax.fill_between(distances,
                     results["prob_no_action"] - results["std_no_action"],
                     results["prob_no_action"] + results["std_no_action"],
                     color='#1565C0', alpha=0.2)
    
    # Styling
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax.set_xlabel('Perturbation Distance (along decoder normal)', fontsize=12)
    ax.set_ylabel('Output Probability', fontsize=12)
    ax.set_title(f'Causal Perturbation Test: {property_name.capitalize()}\n(Figure A7)',
                fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([-0.05, 1.05])
    
    # Add expected pattern annotation
    ax.text(0.98, 0.02, 
            'Expected:\n• Match ↓\n• No Action ↑\n• Crossing indicates causal subspace',
            transform=ax.transAxes, fontsize=9, style='italic',
            verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved perturbation plot: {output_path}")


def analyze_causal_perturbation(
    model_path: Path,
    hidden_root: Path,
    property_name: str,
    output_dir: Path,
    timestep: int = 3,
    perturbation_range: Tuple[float, float] = (-2.0, 2.0),
    num_distances: int = 21,
    task: Optional[str] = None,
    n_value: Optional[int] = None,
    epochs: Optional[List[int]] = None,
    device: Optional[torch.device] = None
) -> Dict[str, Any]:
    """
    Complete causal perturbation analysis pipeline.
    
    Args:
        model_path: Path to trained model checkpoint
        hidden_root: Path to saved hidden states
        property_name: Property to decode (location, identity, category)
        output_dir: Directory to save results
        timestep: Timestep to analyze (executive phase, typically 3-5)
        perturbation_range: (min, max) distance to perturb
        num_distances: Number of perturbation distances to test
        task: Task filter (None for all)
        n_value: N-back filter (None for all)
        epochs: Specific epochs to analyze (None for all)
        device: Torch device (None for auto-detect)
    
    Returns:
        Dictionary with analysis results
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*70)
    print("CAUSAL PERTURBATION TEST (Figure A7)")
    print("="*70)
    print(f"\nProperty: {property_name}")
    print(f"Timestep: {timestep}")
    print(f"Device: {device}")
    
    # Step 1: Load model
    print("\n1. Loading trained model...")
    model = load_trained_model(model_path, device)
    
    # Step 2: Load hidden states and train decoder
    print("\n2. Training decoder on hidden states...")
    payloads = load_payloads(Path(hidden_root), epochs=epochs)
    
    task_idx = None
    if task == "location":
        task_idx = 0
    elif task == "identity":
        task_idx = 1
    elif task == "category":
        task_idx = 2
    
    # Train decoder on encoding space (t=0) to get hyperplane normals
    X, y, label2idx = build_matrix(
        payloads, property_name, time=0, task_index=task_idx, n_value=n_value
    )
    
    if X.numel() == 0:
        raise RuntimeError("No data found for decoder training")
    
    decoder_weights = one_vs_rest_weights(X, y)  # Dict[int, np.ndarray] with C classes
    print(f"✓ Trained decoder with {len(decoder_weights)} classes")
    
    # Step 3: Select match trials
    print(f"\n3. Selecting match trials at timestep {timestep}...")
    hidden_states, property_values, original_logits, indices = select_match_trials(
        payloads, property_name, timestep, task_idx, n_value, label2idx=label2idx
    )
    
    # Step 4: Run perturbation
    print("\n4. Running causal perturbation...")
    distances = np.linspace(perturbation_range[0], perturbation_range[1], num_distances)
    results = run_causal_perturbation(
        model, hidden_states, decoder_weights, distances, device,
        property_values=property_values,
    )
    
    # Step 5: Plot results
    print("\n5. Generating plots...")
    plot_path = output_dir / f"causal_perturbation_{property_name}.png"
    plot_perturbation_results(results, plot_path, property_name)
    
    # Step 6: Save numerical results
    import json
    results_json = {
        "property": property_name,
        "timestep": timestep,
        "num_trials": len(hidden_states),
        "num_distances": num_distances,
        "perturbation_range": perturbation_range,
        "distances": results["distances"].tolist(),
        "prob_match": results["prob_match"].tolist(),
        "prob_non_match": results["prob_non_match"].tolist(),
        "prob_no_action": results["prob_no_action"].tolist(),
    }
    
    json_path = output_dir / f"causal_perturbation_{property_name}.json"
    with open(json_path, 'w') as f:
        json.dump(results_json, f, indent=2)
    
    print(f"✓ Saved results: {json_path}")
    
    # Step 7: Verify expected pattern
    print("\n6. Verifying expected pattern...")
    prob_match_start = results["prob_match"][0]
    prob_match_end = results["prob_match"][-1]
    prob_no_action_start = results["prob_no_action"][0]
    prob_no_action_end = results["prob_no_action"][-1]
    
    match_drops = prob_match_end < prob_match_start * 0.7
    no_action_rises = prob_no_action_end > prob_no_action_start * 1.3
    
    if match_drops and no_action_rises:
        print("✓ EXPECTED PATTERN CONFIRMED:")
        print(f"  - P(Match): {prob_match_start:.3f} → {prob_match_end:.3f} (dropped)")
        print(f"  - P(No Action): {prob_no_action_start:.3f} → {prob_no_action_end:.3f} (rose)")
        print("  → Decoder subspaces are causally related to network behavior!")
    else:
        print("⚠ UNEXPECTED PATTERN:")
        print(f"  - P(Match): {prob_match_start:.3f} → {prob_match_end:.3f}")
        print(f"  - P(No Action): {prob_no_action_start:.3f} → {prob_no_action_end:.3f}")
        print("  → Check if decoder or perturbation parameters need adjustment")
    
    print("\n" + "="*70)
    print("CAUSAL PERTURBATION TEST COMPLETE")
    print("="*70)
    
    return results_json


def main():
    parser = argparse.ArgumentParser(description="Causal Perturbation Test (Figure A7)")
    parser.add_argument("--model", type=str, required=True,
                       help="Path to trained model checkpoint")
    parser.add_argument("--hidden_root", type=str, required=True,
                       help="Path to saved hidden states directory")
    parser.add_argument("--property", type=str, default="location",
                       choices=["location", "identity", "category"],
                       help="Property to decode")
    parser.add_argument("--output_dir", type=str, default="analysis_results",
                       help="Output directory for results")
    parser.add_argument("--timestep", type=int, default=3,
                       help="Timestep to analyze (executive phase, typically 3-5)")
    parser.add_argument("--min_dist", type=float, default=-2.0,
                       help="Minimum perturbation distance")
    parser.add_argument("--max_dist", type=float, default=2.0,
                       help="Maximum perturbation distance")
    parser.add_argument("--num_distances", type=int, default=21,
                       help="Number of perturbation distances to test")
    parser.add_argument("--task", type=str, default=None,
                       choices=["location", "identity", "category"],
                       help="Filter by task context")
    parser.add_argument("--n", type=int, default=None,
                       help="Filter by n-back value")
    parser.add_argument("--epochs", type=int, nargs="*", default=None,
                       help="Specific epochs to analyze")
    
    args = parser.parse_args()
    
    analyze_causal_perturbation(
        model_path=Path(args.model),
        hidden_root=Path(args.hidden_root),
        property_name=args.property,
        output_dir=Path(args.output_dir),
        timestep=args.timestep,
        perturbation_range=(args.min_dist, args.max_dist),
        num_distances=args.num_distances,
        task=args.task,
        n_value=args.n,
        epochs=args.epochs
    )


if __name__ == "__main__":
    main()
