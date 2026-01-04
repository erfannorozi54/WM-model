# Working Memory Model

PyTorch implementation of a two-stage neural network for N-back working memory tasks, based on paper 2411.02685.

## Quick Start

```bash
# 1. Activate environment
source venv/bin/activate
export PYTHONPATH="${PWD}/src:${PYTHONPATH}"

# 2. Train a model (stimuli already generated)
python -m src.train_with_generalization --config configs/stsf.yaml

# 3. View results
ls experiments/
```

## Project Structure

```
WM-model/
├── configs/                    # Training configurations
│   ├── stsf.yaml              # Single-Task Single-Feature
│   ├── stmf.yaml              # Single-Task Multi-Feature
│   ├── mtmf.yaml              # Multi-Task Multi-Feature
│   └── attention_*.yaml       # Attention-enhanced variants
├── src/
│   ├── train.py               # Basic training script
│   ├── train_with_generalization.py  # Training with validation splits (recommended)
│   ├── models/                # Neural network modules
│   ├── data/                  # Data pipeline
│   └── analysis/              # Analysis tools
├── data/stimuli/              # Rendered stimulus images (320 images)
├── experiments/               # Training outputs
└── configs/                   # YAML configurations
```

## Training

### Option 1: With Generalization Validation (Recommended)

Tests on novel angles and novel identities:

```bash
python -m src.train_with_generalization --config configs/stsf.yaml
```

### Option 2: Basic Training

```bash
python -m src.train --config configs/stsf.yaml
```

### Training Scenarios

| Config | Description | N-values | Tasks |
|--------|-------------|----------|-------|
| `stsf.yaml` | Single-Task Single-Feature | [2] | category |
| `stmf.yaml` | Single-Task Multi-Feature | [2] | location, identity, category |
| `mtmf.yaml` | Multi-Task Multi-Feature | [1,2,3] | location, identity, category |
| `attention_*.yaml` | With task-guided attention | varies | varies |

### Key Config Parameters

```yaml
# Model
hidden_size: 256        # RNN hidden dimension
rnn_type: "gru"         # rnn|gru|lstm
num_layers: 1           # RNN layers

# Data
n_values: [2]           # N-back levels
task_features: ["category"]  # location|identity|category
sequence_length: 6      # Trials per sequence
batch_size: 16

# Training
epochs: 16
lr: 0.0003
save_hidden: true       # Save hidden states for analysis
```

## Architecture

```
Input Images (B,T,3,224,224)
    ↓
ResNet50 (frozen) → 1×1 Conv → GAP → Visual Embedding (B,T,H)
    ↓
Concat with Task Vector (B,T,H+3)
    ↓
RNN/GRU/LSTM → Hidden States (B,T,H)
    ↓
Linear Classifier → Logits (B,T,3)
    ↓
Predictions: no_action | non_match | match
```

## Analysis

After training, analyze hidden states:

```bash
# Decoding analysis
python -m src.analysis.decoding \
  --hidden_root experiments/<exp_name>/hidden_states \
  --property identity \
  --train_time 2 --test_times 3 4 5

# Procrustes analysis (temporal dynamics)
python -m src.analysis.procrustes \
  --hidden_root experiments/<exp_name>/hidden_states \
  --property identity \
  --source_time 2 --target_time 3
```

## Data Pipeline

Stimuli are already generated in `data/stimuli/` (320 images: 4 categories × 5 identities × 4 locations × 4 angles).

To regenerate:

```bash
# Generate placeholder data
python -m src.data.download_shapenet --placeholder

# Generate stimuli
python -m src.data.generate_stimuli
```

## Output Structure

```
experiments/<exp_name>/
├── config.yaml              # Saved configuration
├── training.log             # Training logs
├── training_log.json        # Metrics per epoch
├── best_model.pt            # Best checkpoint
└── hidden_states/           # Saved activations
    └── epoch_XXX/
        └── batch_XXXX.pt
```

## Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA GPU (recommended)

Install dependencies:
```bash
pip install -r requirements.txt
```

## Troubleshooting

**CUDA out of memory**: Reduce `batch_size` in config

**No stimuli found**: Run `python -m src.data.generate_stimuli`

**Import errors**: Ensure `PYTHONPATH` includes `src/`:
```bash
export PYTHONPATH="${PWD}/src:${PYTHONPATH}"
```
