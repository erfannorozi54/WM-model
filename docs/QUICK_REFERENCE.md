# Quick Reference Guide
## Working Memory Model - Command Cheatsheet

---

## Training

### Baseline Models
```bash
# Single-Task Single-Feature (Location only, 2-back)
python train.py --config configs/stsf.yaml

# Single-Task Multi-Feature (All features, 2-back)
python train.py --config configs/stmf.yaml

# Multi-Task Multi-Feature (All features, 1/2/3-back)
python train.py --config configs/mtmf.yaml
```

### Attention Models
```bash
# Attention-enhanced variants
python train.py --config configs/attention_stsf.yaml
python train.py --config configs/attention_stmf.yaml
python train.py --config configs/attention_mtmf.yaml
```

---

## Analysis

### Decoding Analysis
```bash
# Basic decoding
python -m src.analysis.decoding \
  --hidden_root runs/wm_mtmf/hidden_states \
  --property identity \
  --train_time 2 \
  --test_times 2 3 4 5

# With filters
python -m src.analysis.decoding \
  --hidden_root runs/wm_mtmf/hidden_states \
  --property location \
  --train_time 2 --test_times 3 4 \
  --train_task location --test_task identity \
  --train_n 2 --test_n 2
```

### Orthogonalization Analysis
```bash
# Basic orthogonalization
python -m src.analysis.orthogonalization \
  --hidden_root runs/wm_mtmf/hidden_states \
  --property identity \
  --time 3

# With filters
python -m src.analysis.orthogonalization \
  --hidden_root runs/wm_mtmf/hidden_states \
  --property location \
  --time 3 --task location --n 2
```

### Procrustes Analysis
```bash
# Basic Procrustes
python -m src.analysis.procrustes \
  --hidden_root runs/wm_mtmf/hidden_states \
  --property identity \
  --source_time 2 --target_time 3

# Swap hypothesis test
python -m src.analysis.procrustes \
  --hidden_root runs/wm_mtmf/hidden_states \
  --property identity \
  --swap_test \
  --encoding_time 2 --k_offset 1

# With filters
python -m src.analysis.procrustes \
  --hidden_root runs/wm_mtmf/hidden_states \
  --property identity \
  --source_time 2 --target_time 3 \
  --task location --n 2
```

---

## Demos

### Procrustes Demo
```bash
# All analyses
python demo_procrustes.py \
  --hidden_root runs/wm_mtmf/hidden_states

# Specific demo
python demo_procrustes.py \
  --hidden_root runs/wm_mtmf/hidden_states \
  --demo basic

python demo_procrustes.py \
  --hidden_root runs/wm_mtmf/hidden_states \
  --demo trajectory

python demo_procrustes.py \
  --hidden_root runs/wm_mtmf/hidden_states \
  --demo swap --visualize
```

### Data Pipeline Demo
```bash
python demo_pipeline.py
```

---

## Batch Analysis

### Full Figure 4 Replication
```bash
# Complete analysis
python analyze_procrustes_batch.py \
  --hidden_root runs/wm_mtmf/hidden_states \
  --property identity \
  --n 2 \
  --visualize

# Fast version (skip temporal generalization)
python analyze_procrustes_batch.py \
  --hidden_root runs/wm_mtmf/hidden_states \
  --property identity \
  --skip_tg \
  --visualize

# Skip swap test
python analyze_procrustes_batch.py \
  --hidden_root runs/wm_mtmf/hidden_states \
  --property identity \
  --skip_swap \
  --visualize
```

---

## Model Comparison (Phase 5)

### Compare Baseline vs. Attention
```bash
# Full comparison
python compare_models.py \
  --baseline runs/wm_mtmf/hidden_states \
  --attention runs/wm_attention_mtmf/hidden_states \
  --property identity

# With filters
python compare_models.py \
  --baseline runs/wm_mtmf/hidden_states \
  --attention runs/wm_attention_mtmf/hidden_states \
  --property identity \
  --task location --n 2 \
  --output_dir results/comparison
```

---

## Attention Visualization (Phase 5)

### Visualize Attention Heatmaps
```bash
# Basic visualization
python visualize_attention.py \
  --checkpoint runs/wm_attention_mtmf/checkpoints/best_*.pt \
  --num_samples 5

# Specific task
python visualize_attention.py \
  --checkpoint runs/wm_attention_mtmf/checkpoints/best_*.pt \
  --task location \
  --num_samples 10 \
  --output_dir results/attention_viz

# Custom config
python visualize_attention.py \
  --checkpoint runs/wm_attention_mtmf/checkpoints/best_*.pt \
  --config configs/attention_mtmf.yaml \
  --n 2 \
  --device cuda
```

---

## Common Workflows

### Workflow 1: Train and Analyze Single Model
```bash
# 1. Train
python train.py --config configs/mtmf.yaml

# 2. Decoding
python -m src.analysis.decoding \
  --hidden_root runs/wm_mtmf/hidden_states \
  --property identity --train_time 2 --test_times 2 3 4 5

# 3. Orthogonalization
python -m src.analysis.orthogonalization \
  --hidden_root runs/wm_mtmf/hidden_states \
  --property identity --time 3

# 4. Procrustes
python demo_procrustes.py \
  --hidden_root runs/wm_mtmf/hidden_states --demo all
```

### Workflow 2: Compare Baseline vs. Attention
```bash
# 1. Train both
python train.py --config configs/mtmf.yaml
python train.py --config configs/attention_mtmf.yaml

# 2. Compare
python compare_models.py \
  --baseline runs/wm_mtmf/hidden_states \
  --attention runs/wm_attention_mtmf/hidden_states \
  --property identity

# 3. Visualize attention
python visualize_attention.py \
  --checkpoint runs/wm_attention_mtmf/checkpoints/best_*.pt \
  --num_samples 10
```

### Workflow 3: Full Publication Analysis
```bash
# 1. Train all models
python train.py --config configs/mtmf.yaml
python train.py --config configs/attention_mtmf.yaml

# 2. Full Procrustes for both
python analyze_procrustes_batch.py \
  --hidden_root runs/wm_mtmf/hidden_states \
  --property identity --visualize

python analyze_procrustes_batch.py \
  --hidden_root runs/wm_attention_mtmf/hidden_states \
  --property identity --visualize

# 3. Comparative analysis
python compare_models.py \
  --baseline runs/wm_mtmf/hidden_states \
  --attention runs/wm_attention_mtmf/hidden_states \
  --property identity

# 4. Attention visualization
python visualize_attention.py \
  --checkpoint runs/wm_attention_mtmf/checkpoints/best_*.pt \
  --num_samples 20 --output_dir figures/attention
```

---

## Parameter Reference

### Common Parameters

**Data Filters:**
- `--property`: `location`, `identity`, `category`
- `--task`: `location`, `identity`, `category`, `any`
- `--n`: N-back value (1, 2, 3)
- `--epochs`: Specific epochs to analyze (e.g., `10 11 12`)

**Time Parameters:**
- `--time`: Single time point (0-5)
- `--train_time`: Training time point
- `--test_times`: Multiple test time points (e.g., `2 3 4 5`)
- `--source_time`: Source time for Procrustes
- `--target_time`: Target time for Procrustes
- `--encoding_time`: Encoding time for swap test
- `--k_offset`: Temporal offset for swap test

**Output:**
- `--output_dir`: Output directory for results
- `--visualize`: Generate visualizations (flag)

**Model Comparison:**
- `--baseline`: Path to baseline hidden states
- `--attention`: Path to attention hidden states
- `--checkpoint`: Path to model checkpoint

---

## File Paths

### Training Outputs
```
runs/
└── wm_mtmf/                      # Experiment directory
    ├── checkpoints/
    │   ├── best_epoch010_acc0.850.pt
    │   └── final_epoch015.pt
    └── hidden_states/
        ├── epoch_010/
        │   ├── epoch010_batch0000.pt
        │   └── ...
        └── ...
```

### Analysis Outputs
```
results/
├── comparison/
│   └── comparison.json           # Model comparison results
├── procrustes/
│   ├── temporal_generalization_identity.json
│   ├── procrustes_disparity_identity.json
│   ├── swap_test_identity.json
│   └── figure4_identity.png
└── attention_viz/
    ├── attention_sample1_location.png
    └── ...
```

---

## Configuration Files

### Baseline Configs
- `configs/stsf.yaml` - Single-Task Single-Feature
- `configs/stmf.yaml` - Single-Task Multi-Feature  
- `configs/mtmf.yaml` - Multi-Task Multi-Feature

### Attention Configs
- `configs/attention_stsf.yaml` - Attention STSF
- `configs/attention_stmf.yaml` - Attention STMF
- `configs/attention_mtmf.yaml` - Attention MTMF

### Key Config Parameters
```yaml
# Model
model_type: "gru"                # gru|lstm|rnn|attention_gru|attention_lstm|attention_rnn
hidden_size: 512
num_layers: 1
dropout: 0.0

# Attention-specific (for attention models)
attention_hidden_dim: 512
attention_dropout: 0.1

# Data
n_values: [1, 2, 3]              # N-back values
task_features: ["location", "identity", "category"]
sequence_length: 6
batch_size: 8

# Training
epochs: 15
lr: 3e-4
weight_decay: 1e-2
milestones: [10, 13]
gamma: 0.1
```

---

## Python API

### Create Models
```python
from src.models import create_model, create_baseline_model, create_attention_model

# Factory method
model = create_model('attention_gru', hidden_size=512)

# Convenience methods
baseline = create_baseline_model('gru', hidden_size=512)
attention = create_attention_model('gru', hidden_size=512)
```

### Load Analysis Results
```python
import json
from pathlib import Path

# Load comparison results
with open('results/comparison/comparison.json') as f:
    results = json.load(f)

print(f"Decoding improvement: {results['decoding']['mean_improvement']:.3f}")
print(f"Ortho improvement: {results['orthogonalization']['improvement']:.3f}")

# Load Procrustes results
with open('results/procrustes/swap_test_identity.json') as f:
    swap = json.load(f)

print(f"Hypothesis confirmed: {swap['hypothesis_confirmed']}")
```

### Use Model Factory
```python
from src.models import create_model, get_model_info, print_model_summary

# Create model
model = create_model('attention_gru', hidden_size=512)

# Get info
info = get_model_info(model)
print(f"Parameters: {info['num_parameters']:,}")
print(f"Is attention: {info['is_attention']}")

# Print summary
print_model_summary(model)
```

---

## Troubleshooting Quick Fixes

### No hidden states found
```bash
# Ensure save_hidden: true in config
grep save_hidden configs/mtmf.yaml

# Re-train with hidden state saving
python train.py --config configs/mtmf.yaml --save_hidden
```

### Analysis fails with "No samples"
```bash
# Check hidden states exist
ls runs/wm_mtmf/hidden_states/

# Try broader filters (remove --task, --n)
python -m src.analysis.decoding \
  --hidden_root runs/wm_mtmf/hidden_states \
  --property identity --train_time 2 --test_times 3
```

### Attention weights are uniform
```bash
# Train longer
# Increase epochs in config, then re-train

# Reduce attention dropout
# Edit config: attention_dropout: 0.05
python train.py --config configs/attention_mtmf.yaml
```

### Comparison script fails
```bash
# Ensure both models trained with same config
# Check hidden states exist for both
ls runs/wm_mtmf/hidden_states/
ls runs/wm_attention_mtmf/hidden_states/

# Use matching filters
python compare_models.py \
  --baseline runs/wm_mtmf/hidden_states \
  --attention runs/wm_attention_mtmf/hidden_states \
  --property identity --n 2
```

---

## Performance Tips

### Speed up training
- Use GPU: Edit config `device: cuda`
- Reduce num_workers if CPU bottleneck
- Use smaller batch_size if memory constrained

### Speed up analysis
- Use `--skip_tg` flag for batch analysis (skips slow temporal generalization)
- Limit epochs: `--epochs 10 11 12`
- Filter by specific task/n

### Reduce memory usage
- Lower batch_size in config
- Reduce num_val in config
- Use CPU instead of GPU for analysis

---

## Documentation Files

- `README.md` - Main overview
- `QUICK_REFERENCE.md` - This file (commands)
- `PROCRUSTES_GUIDE.md` - Detailed Procrustes guide
- `PHASE4_SUMMARY.md` - Phase 4 summary
- `PHASE5_SUMMARY.md` - Phase 5 summary (attention)
- `PROJECT_COMPLETE.md` - Complete project summary

---

## Help Commands

```bash
# Training help
python train.py --help

# Analysis help
python -m src.analysis.decoding --help
python -m src.analysis.orthogonalization --help
python -m src.analysis.procrustes --help

# Demo help
python demo_procrustes.py --help
python analyze_procrustes_batch.py --help

# Comparison help
python compare_models.py --help
python visualize_attention.py --help
```

---

**Quick Reference Complete!** 🚀

For detailed documentation, see the full guides in the repository.
