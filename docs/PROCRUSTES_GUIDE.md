# Procrustes Analysis Guide

## Phase 4: Advanced Spatiotemporal Transformation Analysis

This guide provides comprehensive documentation for the Procrustes analysis implementation, which is the most sophisticated analysis in the working memory model project.

---

## Table of Contents

1. [Overview](#overview)
2. [Theoretical Background](#theoretical-background)
3. [Implementation Details](#implementation-details)
4. [Usage Examples](#usage-examples)
5. [Interpreting Results](#interpreting-results)
6. [Troubleshooting](#troubleshooting)
7. [Advanced Topics](#advanced-topics)

---

## Overview

### What is Procrustes Analysis?

**Orthogonal Procrustes analysis** finds the optimal rotation matrix that aligns two sets of points in high-dimensional space. In the context of working memory:

- **Points**: Decoder weight vectors for different object classes
- **Spaces**: Neural representation spaces at different time points
- **Goal**: Understand how representations transform as memories age

### Why is This Important?

The paper's key finding is that neural representations in working memory transform in a **chronologically-organized** manner:

1. Representations at time T can be aligned to time T+1 via rotation
2. These rotations preserve **temporal structure** more than stimulus identity
3. This suggests memory maintenance follows universal geometric trajectories

---

## Theoretical Background

### Mathematical Formulation

Given two sets of decoder weight vectors:

- **W_source**: Weights at time t (n_classes × d matrix)
- **W_target**: Weights at time t+1 (n_classes × d matrix)

Find rotation matrix **R** (d × d) that minimizes:

```math
||W_target - W_source @ R||²_F
```

Subject to: `R^T @ R = I` (orthogonality constraint)

### Solution via SVD

The closed-form solution uses Singular Value Decomposition:

```python
U, _, Vt = svd(W_source^T @ W_target)
R = U @ Vt
```

This is implemented in `scipy.linalg.orthogonal_procrustes`.

### Chronological Organization Hypothesis

**Core prediction**: Memory subspaces are organized by temporal age, not stimulus identity.

**Test via "swap" experiment**:

1. Compute correct rotation for stimulus S at time T
2. Try swapping with:
   - **Swap 1**: Rotation for same stimulus, different time (wrong!)
   - **Swap 2**: Rotation for different stimulus, same age (should work!)

**Expected result**: Swap 2 maintains accuracy better than Swap 1.

---

## Implementation Details

### Module Structure

```text
src/analysis/procrustes.py
├── compute_procrustes_alignment()  # Core Procrustes computation
├── reconstruct_weights()            # Apply rotation to weights
├── evaluate_reconstruction()        # Test reconstructed weights
├── procrustes_analysis()           # Full analysis pipeline
└── swap_hypothesis_test()          # Swap experiment (Figure 4g)
```

### Key Functions

#### 1. `compute_procrustes_alignment(W_source, W_target)`

Computes optimal rotation matrix between weight sets.

**Input:**

- `W_source`: Dict[int, np.ndarray] - Source decoder weights
- `W_target`: Dict[int, np.ndarray] - Target decoder weights

**Output:**

- `R`: np.ndarray (d, d) - Optimal rotation matrix
- `disparity`: float - Procrustes disparity (alignment quality)

**Disparity interpretation:**

- 0.0: Perfect alignment
- < 0.1: Excellent alignment
- 0.1-0.3: Good alignment
- > 0.5: Poor alignment

#### 2. `swap_hypothesis_test(hidden_root, property_name, encoding_time, k_offset)`

Tests chronological organization via rotation swaps.

**Input:**

- `hidden_root`: Path to saved hidden states
- `property_name`: 'location' | 'identity' | 'category'
- `encoding_time`: Initial encoding time (j)
- `k_offset`: Temporal offset for comparison (typically 1)

**Output dictionary contains:**

- `correct_accuracy`: Accuracy with correct rotation
- `swap1_accuracy`: Same stimulus, wrong time
- `swap2_accuracy`: Different stimulus, same age
- `hypothesis_confirmed`: bool - Whether swap2 > swap1

---

## Usage Examples

### Example 1: Basic Procrustes Analysis

```bash
# Compute alignment between two consecutive time points
python -m src.analysis.procrustes \
  --hidden_root runs/wm_mtmf/hidden_states \
  --property identity \
  --source_time 2 --target_time 3 \
  --task location --n 2
```

**Output:**

```json
{
  "property": "identity",
  "source_time": 2,
  "target_time": 3,
  "procrustes_disparity": 0.15,
  "reconstruction_accuracy": 0.82,
  "baseline_accuracy": 0.88,
  "accuracy_ratio": 0.93
}
```

**Interpretation:**

- Low disparity (0.15) indicates representations maintain geometric structure
- High reconstruction accuracy (0.82) shows linear transformability
- Accuracy ratio (93%) suggests minimal information loss via rotation

### Example 2: Swap Hypothesis Test

```bash
# Test chronological organization
python -m src.analysis.procrustes \
  --hidden_root runs/wm_mtmf/hidden_states \
  --property identity \
  --swap_test \
  --encoding_time 2 \
  --k_offset 1 \
  --task location --n 2
```

**Output:**

```json
{
  "encoding_time": 2,
  "k_offset": 1,
  "baseline_accuracy": 0.88,
  "correct_accuracy": 0.82,
  "swap1_accuracy": 0.45,
  "swap2_accuracy": 0.78,
  "swap1_relative": 0.55,
  "swap2_relative": 0.95,
  "hypothesis_confirmed": true
}
```

**Interpretation:**

- ✓ Hypothesis confirmed: swap2 (0.78) >> swap1 (0.45)
- Same-age rotation preserves 95% of correct performance
- Wrong-time rotation drops to 55% of correct performance
- This validates chronological organization of memory subspaces

### Example 3: Interactive Demo

```bash
# Run all analyses with visualization
python demo_procrustes.py \
  --hidden_root runs/wm_mtmf/hidden_states \
  --property identity \
  --demo all \
  --visualize
```

**Outputs:**

1. Basic Procrustes results
2. Temporal trajectory analysis
3. Swap hypothesis test
4. Visualization: `procrustes_analysis.png`

### Example 4: Batch Analysis (Full Figure 4)

```bash
# Complete analysis with all matrices
python analyze_procrustes_batch.py \
  --hidden_root runs/wm_mtmf/hidden_states \
  --property identity \
  --n 2 \
  --visualize \
  --output_dir results/procrustes
```

**Outputs:**

- `temporal_generalization_identity.json` - Cross-time decoding matrix
- `procrustes_disparity_identity.json` - Alignment quality matrix
- `swap_test_identity.json` - Swap experiment results
- `figure4_identity.png` - Publication-ready visualization

---

## Interpreting Results

### Temporal Generalization Matrix

Matrix[i, j] = accuracy when training at time i, testing at time j

**Patterns to look for:**

- **Strong diagonal**: Good within-time decoding
- **Broad diagonal band**: Representations stable across nearby times
- **Off-diagonal decay**: Information loss over time
- **Asymmetry**: Different encoding vs. retrieval dynamics

**Example interpretation:**

```math
     T0   T1   T2   T3   T4   T5
T0 [0.92 0.85 0.75 0.62 0.48 0.35]
T1 [0.80 0.94 0.88 0.78 0.65 0.52]
T2 [0.65 0.82 0.95 0.90 0.82 0.70]
T3 [0.48 0.68 0.86 0.96 0.91 0.85]
T4 [0.35 0.52 0.72 0.88 0.97 0.93]
T5 [0.25 0.40 0.58 0.78 0.90 0.98]
```

- Strong diagonal: Excellent within-time decoding
- Asymmetry: Better to train early, test late (row 2) than vice versa
- Suggests progressive refinement of representations

### Procrustes Disparity Matrix

Matrix[i, j] = disparity when aligning time i to time j

**Patterns to look for:**

- **Low adjacent values**: Smooth temporal transitions
- **Symmetric structure**: Bidirectional alignment quality
- **Temporal gradient**: Increasing disparity with time separation

**Example interpretation:**

```math
     T0   T1   T2   T3   T4   T5
T0 [0.00 0.12 0.25 0.38 0.52 0.68]
T1 [0.12 0.00 0.10 0.22 0.36 0.52]
T2 [0.25 0.10 0.00 0.11 0.24 0.40]
T3 [0.38 0.22 0.11 0.00 0.13 0.28]
T4 [0.52 0.36 0.24 0.13 0.00 0.15]
T5 [0.68 0.52 0.40 0.28 0.15 0.00]
```

- Adjacent times (off-diagonal): Low disparity (~0.1)
- Farther times: Increasing disparity
- Suggests gradual transformation trajectory

### Swap Test Results

**Critical comparisons:**

1. **Correct vs. Baseline**:
   - Close values: Rotations preserve information well
   - Large gap: Substantial representational drift

2. **Swap2 vs. Swap1**:
   - Swap2 > Swap1: ✓ Chronological organization
   - Swap1 > Swap2: ✗ Stimulus-specific organization

3. **Swap2 vs. Correct**:
   - Close values: Strong generalization across stimuli
   - Large gap: Stimulus-specific transformations

**Ideal results (supporting paper's hypothesis):**

```math
Baseline: 0.88
Correct:  0.82  (93% of baseline)
Swap2:    0.78  (95% of correct) ✓
Swap1:    0.45  (55% of correct) ✓
```

---

## Troubleshooting

### Problem: "No samples for the specified context"

**Cause**: Insufficient data at requested time/task/n combination

**Solutions:**

1. Check if hidden states exist: `ls runs/*/hidden_states/`
2. Verify training saved hidden states: `--save_hidden` flag
3. Try broader filters: `--task any` or remove `--n` filter
4. Train longer: More epochs = more validation data

### Problem: "Need at least 2 common classes for Procrustes"

**Cause**: Not enough shared classes between time points

**Solutions:**

1. Use property with more classes (e.g., 'identity' > 'category')
2. Train with more stimuli in dataset
3. Reduce filtering (don't restrict to specific task/n)

### Problem: Very high disparity values (> 0.5)

**Possible causes:**

1. **Insufficient training**: Model hasn't learned stable representations
2. **High noise**: Need more training data
3. **Rapid drift**: True representational change

**Solutions:**

1. Train longer (more epochs)
2. Increase validation set size (`num_val` in config)
3. Check if model is learning: Look at training accuracy

### Problem: Swap test not confirming hypothesis

**Possible causes:**

1. **Insufficient training**: Random representations
2. **Too few time points**: Need longer sequences
3. **Model architecture**: Try different RNN type

**Diagnostics:**

```bash
# Check baseline decoding accuracy
python -m src.analysis.decoding \
  --hidden_root runs/wm_mtmf/hidden_states \
  --property identity --train_time 2 --test_times 2 3 4

# If accuracy < 0.7, model needs more training
```

### Problem: "Failed to load hidden states"

**Cause**: Corrupted or missing .pt files

**Solutions:**

1. Re-run training: `python train.py --config configs/mtmf.yaml`
2. Check disk space
3. Verify file permissions

---

## Advanced Topics

### Custom Time Windows

Analyze specific temporal windows:

```python
from src.analysis.procrustes import procrustes_analysis

# Analyze encoding → early maintenance (0→1)
result_early = procrustes_analysis(
    hidden_root=Path("runs/wm_mtmf/hidden_states"),
    property_name="identity",
    source_time=0,
    target_time=1,
)

# Analyze late maintenance (4→5)
result_late = procrustes_analysis(
    hidden_root=Path("runs/wm_mtmf/hidden_states"),
    property_name="identity",
    source_time=4,
    target_time=5,
)

# Compare transformation rates
print(f"Early disparity: {result_early['procrustes_disparity']:.3f}")
print(f"Late disparity: {result_late['procrustes_disparity']:.3f}")
```

### Cross-Architecture Comparison

Compare different RNN types:

```bash
# Train with different architectures
python train.py --config configs/mtmf.yaml  # GRU (default)
# Edit config to use LSTM/RNN, then:
python train.py --config configs/mtmf_lstm.yaml

# Compare Procrustes results
for arch in gru lstm rnn; do
  python analyze_procrustes_batch.py \
    --hidden_root runs/wm_mtmf_${arch}/hidden_states \
    --property identity --visualize
done
```

### N-back Comparison

Analyze how memory load affects transformations:

```bash
# 1-back: Low memory load
python -m src.analysis.procrustes \
  --hidden_root runs/wm_mtmf/hidden_states \
  --property identity --n 1 \
  --source_time 2 --target_time 3

# 3-back: High memory load
python -m src.analysis.procrustes \
  --hidden_root runs/wm_mtmf/hidden_states \
  --property identity --n 3 \
  --source_time 2 --target_time 3
```

### Programmatic Analysis

Use as a library for custom analyses:

```python
from pathlib import Path
from src.analysis.procrustes import (
    compute_procrustes_alignment,
    swap_hypothesis_test,
)
from src.analysis.orthogonalization import one_vs_rest_weights
from src.analysis.activations import load_payloads, build_matrix

# Load data
hidden_root = Path("runs/wm_mtmf/hidden_states")
payloads = load_payloads(hidden_root, epochs=[10])

# Get weights at multiple times
times = [2, 3, 4, 5]
weights = {}
for t in times:
    X, y, label2idx = build_matrix(payloads, "identity", time=t)
    weights[t] = one_vs_rest_weights(X, y)

# Compute rotation chain: 2→3→4→5
disparities = []
for t in range(len(times) - 1):
    R, disp = compute_procrustes_alignment(
        weights[times[t]], 
        weights[times[t+1]]
    )
    disparities.append(disp)
    print(f"T{times[t]}→T{times[t+1]}: disparity = {disp:.4f}")

# Analyze trajectory smoothness
import numpy as np
print(f"\nMean disparity: {np.mean(disparities):.4f}")
print(f"Std disparity: {np.std(disparities):.4f}")
```

---

## Quick Reference

### Command Cheatsheet

```bash
# Basic Procrustes
python -m src.analysis.procrustes --hidden_root <path> --property <prop> \
  --source_time <t1> --target_time <t2>

# Swap test
python -m src.analysis.procrustes --hidden_root <path> --property <prop> \
  --swap_test --encoding_time <t>

# Demo (interactive)
python demo_procrustes.py --hidden_root <path> [--demo all|basic|trajectory|swap]

# Batch analysis
python analyze_procrustes_batch.py --hidden_root <path> --property <prop> \
  [--visualize] [--skip_tg] [--skip_swap]
```

### Property Choices

- **location**: 4 classes (spatial positions) - fastest
- **identity**: 8 classes (specific objects) - recommended
- **category**: 4 classes (object types) - intermediate

### Typical Hyperparameters

- **encoding_time**: 2 (after initial no-action trials)
- **k_offset**: 1 (compare adjacent time differences)
- **max_time**: 6 (full sequence length)
- **n_value**: 2 (standard working memory load)

---

## References

### Paper Findings

The original paper (Figure 4) shows:

1. Temporal generalization matrices reveal diagonal structure
2. Procrustes disparity increases with time separation
3. Same-age rotations outperform same-stimulus rotations
4. This pattern holds across all three properties (L/I/C)

### Mathematical Background

- Procrustes, G. (Greek mythology): Forced travelers to fit his bed
- Orthogonal Procrustes: Minimizes ||B - AR||² with orthogonal R
- Solution: R = U @ V^T where U, S, V = svd(A^T @ B)

### Related Analyses

- **Representational Similarity Analysis (RSA)**: Compare full similarity matrices
- **Canonical Correlation Analysis (CCA)**: Find correlated subspaces
- **Linear Discriminant Analysis (LDA)**: Alternative to one-vs-rest SVM
- **Dynamic Time Warping (DTW)**: Compare temporal sequences

---

## Conclusion

Procrustes analysis reveals the **geometric structure** of neural representations in working memory:

1. **Representations transform smoothly** over time (low adjacent disparity)
2. **Transformations preserve temporal structure** (chronological organization)
3. **Memory maintenance follows universal trajectories** (consistent across stimuli)

These findings suggest that working memory implements a **temporal coordinate system** where representations are organized by age rather than content.

---

**For questions or issues**, please refer to:

- Main README: `README.md`
- Source code: `src/analysis/procrustes.py`
- Demo scripts: `demo_procrustes.py`, `analyze_procrustes_batch.py`
