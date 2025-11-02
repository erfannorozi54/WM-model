# Working Memory Model

A comprehensive PyTorch framework for studying working memory using N-back tasks with neural network models and advanced spatiotemporal analysis.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## Overview

This project implements a two-stage neural network architecture (Perceptual + Cognitive modules) for working memory experiments. It replicates key neuroscience findings about temporal dynamics in working memory and extends the work with task-guided attention mechanisms.

**Key Features:**

- Two-stage architecture: ResNet50 perceptual encoder + RNN/GRU/LSTM cognitive module
- Support for multiple N-back tasks: Location, Identity, and Category matching
- 3D stimulus rendering from ShapeNet objects with multiple viewing angles
- **Novel validation splits**: Novel-angle and novel-identity generalization testing
- **Comprehensive analysis pipeline**: 5 analyses matching paper methodology
- Advanced analysis tools: decoding, orthogonalization, Procrustes alignment
- Task-guided attention mechanism for enhanced performance
- Complete experimental pipeline from data generation to figure replication

## Project Structure

```text
WM-model/
├── configs/                        # YAML configuration files
│   ├── stsf.yaml                  # Single-Task Single-Feature
│   ├── stmf.yaml                  # Single-Task Multi-Feature
│   ├── mtmf.yaml                  # Multi-Task Multi-Feature
│   └── attention_*.yaml           # Attention-enhanced variants
├── data/                           # Data storage (gitignored)
│   ├── shapenet/                  # 3D object files
│   └── stimuli/                   # Rendered 2D images
├── docs/                           # Documentation
│   ├── ANALYSIS_METHODOLOGY.md
│   ├── PROCRUSTES_GUIDE.md
│   ├── PHASE4_SUMMARY.md
│   ├── PHASE5_SUMMARY.md
│   ├── PROJECT_COMPLETE.md
│   └── QUICK_REFERENCE.md
├── scripts/                        # Utility scripts
│   ├── verify_analysis_setup.py   # Verify analysis pipeline setup
│   └── util/                      # Environment activation
│       ├── activate_env.sh
│       └── activate_env.bat
├── src/                            # Source code
│   ├── train.py                   # Main training script
│   ├── train_with_generalization.py  # Training with validation splits (Phase 6)
│   ├── setup_environment.py       # Environment setup guide
│   ├── analysis/                  # Analysis modules & tools
│   │   ├── comprehensive_analysis.py   # Master analysis pipeline (NEW)
│   │   ├── activations.py         # Hidden state utilities
│   │   ├── decoding.py            # Linear decoding analysis
│   │   ├── orthogonalization.py   # Representational geometry
│   │   ├── procrustes.py          # Temporal dynamics
│   │   ├── analyze_procrustes_batch.py  # Figure 4 replication CLI
│   │   ├── compare_models.py      # Model comparison CLI
│   │   └── visualize_attention.py # Attention visualization CLI
│   ├── data/                      # Data pipeline
│   │   ├── shapenet_downloader.py # Core downloader
│   │   ├── download_shapenet.py   # CLI for downloads
│   │   ├── generate_stimuli.py    # Stimulus generation CLI
│   │   ├── renderer.py            # 3D→2D rendering (4 locations, 4 angles)
│   │   ├── nback_generator.py     # N-back sequences
│   │   ├── dataset.py             # PyTorch Dataset with validation splits
│   │   ├── validation_splits.py   # Novel-angle/novel-identity splits (NEW)
│   │   └── test_validation_splits.py  # Validation split verification
│   ├── models/                    # Model architectures
│   │   ├── perceptual.py          # ResNet50 encoder
│   │   ├── cognitive.py           # RNN/GRU/LSTM
│   │   ├── attention.py           # Task-guided attention
│   │   ├── wm_model.py            # Full model
│   │   └── model_factory.py       # Model creation
│   └── utils/                     # Utility functions
├── experiments/                    # Training outputs (gitignored)
│   └── <experiment_name>/
│       ├── best_model.pt          # Best model checkpoint
│       ├── training_log.json      # Per-epoch metrics
│       └── hidden_states/         # Saved activations
├── analysis_results/               # Analysis outputs (gitignored)
│   ├── *.png                      # Publication-ready plots
│   └── *.json                     # Numerical results
├── .env                            # Environment variables (HuggingFace token)
├── .env.example                    # Example environment file
├── .gitignore                      # Git ignore rules
├── LICENSE                         # MIT License
├── README.md                       # This file
├── requirements.txt                # Python dependencies
├── PHASE6_IMPLEMENTATION.md        # Phase 6 technical details (NEW)
├── ANALYSIS_CHECKLIST.md           # Detailed analysis checklist (NEW)
├── COMPREHENSIVE_ANALYSIS_READY.md # Analysis quick start (NEW)
└── ANALYSIS_IMPLEMENTATION_SUMMARY.md  # Implementation summary (NEW)
```

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended for training)
- 8GB+ RAM

### Setup

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/WM-model.git
   cd WM-model
   ```

2. **Activate virtual environment:**

   The project includes a pre-configured `venv` directory. Activate it:

   ```bash
   # Unix/Linux/macOS
   source venv/bin/activate

   # Windows
   venv\Scripts\activate.bat
   ```

   If `venv` doesn't exist, create it first:

   ```bash
   python -m venv venv
   ```

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables:**

   Create a `.env` file in the project root for your HuggingFace token:

   ```bash
   # Copy the example file
   cp .env.example .env
   
   # Edit .env and add your token
   # HUGGINGFACE_TOKEN=your_token_here
   ```

   This allows the download script to automatically load your HuggingFace token without passing it as a command-line argument.

5. **Set up Python path (automatically handled by activation scripts):**

   For convenience, use the provided activation scripts that handle both venv and PYTHONPATH:

   ```bash
   # Unix/Linux/macOS
   source scripts/util/activate_env.sh

   # Windows
   scripts\util\activate_env.bat
   ```

   Or manually:

   ```bash
   export PYTHONPATH="${PWD}/src:${PYTHONPATH}"
   ```

## Quick Start

### 1. Download and Setup Data

#### Option A: Quick Start with Placeholder Data (Recommended for Testing)

```bash
# Generate placeholder ShapeNet data instantly
python -m src.data.download_shapenet --placeholder

# Verify the data
python -m src.data.download_shapenet --verify
```

#### Option B: Download Real ShapeNet via Hugging Face Hub

```bash
# Install required packages
pip install huggingface_hub python-dotenv

# Add your token to .env file (see step 4 in Installation)
# HUGGINGFACE_TOKEN=your_token_here

# Download and organize real ShapeNet (token auto-loaded from .env)
python -m src.data.download_shapenet \
  --download-hf ShapeNetCore.v2.zip

# Or download only specific categories to save space
python -m src.data.download_shapenet \
  --download-hf ShapeNetCore.v2.zip \
  --categories airplane car
```

**Note:** The full ShapeNet archive is ~25GB. The script downloads the complete archive but only organizes and keeps the specified categories (or all 4 by default: airplane, car, chair, table).

#### Option C: Manual Download and Organization

```bash
# After manually downloading ShapeNet from https://shapenet.org/
python -m src.data.download_shapenet --organize /path/to/ShapeNetCore.v2
```

**Test the pipeline:**

```bash
python -m src.data.renderer           # Test 3D renderer
python -m src.data.nback_generator    # Test N-back generator
```

### 2. Generate Stimuli

```bash
# Generate stimuli from downloaded ShapeNet objects
python -m src.data.generate_stimuli

# Or programmatically:
```

```python
from src.data.renderer import StimulusRenderer
from src.data.shapenet_downloader import ShapeNetDownloader

# Use the organized ShapeNet data
downloader = ShapeNetDownloader(data_dir="data/shapenet")

# Render stimuli for all categories
renderer = StimulusRenderer()
for category in ["airplane", "car", "chair", "table"]:
    obj_paths = downloader.get_object_paths(category)
    renderer.render_stimulus_set(
        obj_paths=obj_paths,
        output_dir="data/stimuli"
    )
```

### 3. Train a Model

#### Standard Training

```bash
# Train baseline models (choose one scenario)
python -m src.train --config configs/stsf.yaml  # Single-Task Single-Feature
python -m src.train --config configs/stmf.yaml  # Single-Task Multi-Feature
python -m src.train --config configs/mtmf.yaml  # Multi-Task Multi-Feature

# Train attention-enhanced models
python -m src.train --config configs/attention_mtmf.yaml
```

#### Training with Generalization Validation (Phase 6)

**Recommended for research**: Train with proper validation splits to test generalization.

```bash
# Train with novel-angle and novel-identity validation
python -m src.train_with_generalization --config configs/mtmf.yaml

# This automatically:
# - Splits data: 3 angles for training, 1 angle for novel-angle validation
# - Splits data: 3 identities for training, 2 identities for novel-identity validation
# - Evaluates on BOTH validation sets every epoch
# - Saves hidden states for comprehensive analysis
# - Logs separate accuracies: val_novel_angle_acc & val_novel_identity_acc
```

**Expected output**: Novel-identity accuracy should be "substantially weaker" than novel-angle accuracy, confirming proper generalization testing.

### 4. Run Comprehensive Analysis

#### Verify Setup First

```bash
# Verify all analysis components are ready
python scripts/verify_analysis_setup.py
# Expected: 5/5 tests passed ✅
```

#### Complete Analysis Pipeline (All 5 Analyses)

```bash
# Run all analyses matching paper methodology
python -m src.analysis.comprehensive_analysis \
  --analysis all \
  --hidden_root experiments/wm_mtmf/hidden_states \
  --output_dir analysis_results

# Generates:
# - Analysis 1: Behavioral performance (Figure A1c)
# - Analysis 2: Encoding properties (Figures 2a, 2b, 2c)
# - Analysis 3: Orthogonalization (Figure 3b)
# - Analysis 4: WM dynamics (Figures 4b, 4d, 4g)
# - Analysis 5: Causal perturbation (Figure A7) - placeholder
```

#### Individual Analyses

```bash
# Analysis 1: Model behavioral performance
python -m src.analysis.comprehensive_analysis --analysis 1 \
  --hidden_root experiments/wm_mtmf/hidden_states

# Analysis 2: Encoding of object properties
python -m src.analysis.comprehensive_analysis --analysis 2 \
  --hidden_root experiments/wm_mtmf/hidden_states

# Analysis 4: WM dynamics
python -m src.analysis.comprehensive_analysis --analysis 4 \
  --hidden_root experiments/wm_mtmf/hidden_states \
  --property identity

# Traditional analyses (still available)
python -m src.analysis.decoding \
  --hidden_root experiments/wm_mtmf/hidden_states \
  --property identity

python -m src.analysis.analyze_procrustes_batch \
  --hidden_root experiments/wm_mtmf/hidden_states \
  --property identity --visualize
```

## Architecture

### Model Components

**Two-Stage Architecture:**

```text
Input Images → Perceptual Module → Visual Embeddings → Cognitive Module → Classifier → Predictions
                (ResNet50)           (H-dim)          (RNN/GRU/LSTM)      (3-way)
```

**Components:**

- **PerceptualModule** (`src/models/perceptual.py`): ResNet50 backbone with 1×1 convolution to reduce channels to hidden size, followed by global average pooling
- **CognitiveModule** (`src/models/cognitive.py`): RNN/GRU/LSTM variants that process visual embeddings concatenated with task vectors
- **TaskGuidedAttention** (`src/models/attention.py`): Optional attention mechanism for spatial feature selection
- **WorkingMemoryModel** (`src/models/wm_model.py`): Full model combining perceptual + cognitive + classifier

### Data Pipeline

1. **ShapeNet Download System** (`src/data/`)
   - **Core Module**: `shapenet_downloader.py` - All download/organization logic
   - **CLI**: `download_shapenet.py` - Simple unified interface with `.env` support
   - Supports: placeholder generation, Hugging Face Hub download, manual organization
   - Flexible category selection: download specific categories or all 4 (airplane, car, chair, table)
   - Token management: auto-loads from `.env` file for seamless authentication
   - 2 identities per category

2. **Stimulus Generation** (`src/data/`)
   - **CLI**: `generate_stimuli.py` - Batch stimulus generation
   - **Renderer**: `renderer.py` - 3D→2D rendering (4 locations, multiple angles)
   - **N-back Generator**: `nback_generator.py` - Trial sequence generation
   - **Dataset**: `dataset.py` - PyTorch Dataset/DataLoader wrapper

#### Dataset Download Workflow

The project provides a streamlined three-tier system:

1. **Quick Testing**: `--placeholder` flag creates instant dummy data
2. **Automated Real Data**: `--download-hf` downloads from Hugging Face Hub
3. **Manual Organization**: `--organize` processes manually downloaded ShapeNet

All modes use the same unified CLI: `python -m src.data.download_shapenet [mode]`

### Key Features

- **Flexible N-back Tasks**: Support for 1-back, 2-back, 3-back conditions
- **Multiple Task Types**: Location, Identity, and Category matching
- **3D Stimulus Rendering**: From ShapeNet objects to 2D images
- **PyTorch Integration**: Full DataLoader support with transforms
- **Configurable Parameters**: Sequence length, match probability, batch size

## Training

### Configuration

Training is controlled via YAML configuration files in `configs/`. Key parameters:

**Model Architecture:**

- `hidden_size`: RNN hidden dimension (default: 512)
- `rnn_type`: RNN variant - `rnn`, `gru`, or `lstm`
- `num_layers`: Number of RNN layers
- `dropout`: Dropout probability
- `pretrained_backbone`: Use pretrained ResNet50
- `freeze_backbone`: Freeze perceptual module weights

**Task Configuration:**

- `n_values`: List of N-back levels (e.g., [1, 2, 3])
- `task_features`: Task types - location, identity, category
- `sequence_length`: Trials per sequence
- `batch_size`: Training batch size

**Optimization:**

- `lr`: Learning rate (default: 3e-4)
- `weight_decay`: AdamW weight decay
- `milestones`: Learning rate decay milestones
- `gamma`: LR decay factor
- `grad_clip`: Gradient clipping threshold

### Training Scenarios

**Baseline Models:**

```bash
python -m src.train --config configs/stsf.yaml   # Single-Task Single-Feature
python -m src.train --config configs/stmf.yaml   # Single-Task Multi-Feature  
python -m src.train --config configs/mtmf.yaml   # Multi-Task Multi-Feature
```

**Attention-Enhanced Models:**

```bash
python -m src.train --config configs/attention_stsf.yaml
python -m src.train --config configs/attention_stmf.yaml
python -m src.train --config configs/attention_mtmf.yaml
```

### Output Structure

Training outputs are saved to `runs/<experiment_name>/`:

```text
runs/wm_mtmf/
├── checkpoints/           # Model checkpoints
│   ├── best_epoch_*.pt
│   └── last.pt
├── hidden_states/         # Saved hidden states (if enabled)
│   └── epoch_XXX/
│       ├── batch_000.pt
│       └── ...
└── logs/                  # Training logs
```

## Analysis Tools

### Decoding Analysis

Linear SVM decoding to assess information preservation:

```bash
python -m src.analysis.decoding \
  --hidden_root runs/wm_mtmf/hidden_states \
  --property identity \
  --time_point 3
```

### Orthogonalization Analysis

Measure representational geometry:

```bash
python -m src.analysis.orthogonalization \
  --hidden_root runs/wm_mtmf/hidden_states \
  --property location
```

### Procrustes Analysis

Study temporal dynamics via rotation alignment:

```bash
# Basic Procrustes between time points
python -m src.analysis.procrustes \
  --hidden_root runs/wm_mtmf/hidden_states \
  --property identity \
  --source_time 2 --target_time 3

# Swap hypothesis test (chronological organization)
python -m src.analysis.procrustes \
  --hidden_root runs/wm_mtmf/hidden_states \
  --property identity \
  --swap_test --encoding_time 2 --k_offset 1

# Full batch analysis
python -m src.analysis.analyze_procrustes_batch \
  --hidden_root runs/wm_mtmf/hidden_states \
  --property identity --visualize
```

### Model Comparison

Compare baseline vs attention models:

```bash
python -m src.analysis.compare_models \
  --baseline runs/wm_mtmf/hidden_states \
  --attention runs/wm_attention_mtmf/hidden_states \
  --property identity \
  --output_dir results/comparison
```

### Attention Visualization

Visualize attention patterns:

```bash
python -m src.analysis.visualize_attention \
  --checkpoint runs/wm_attention_mtmf/checkpoints/best_*.pt \
  --num_samples 5 \
  --output_dir results/attention_viz
```

## Dependencies

**Core:**

- Python >= 3.8
- PyTorch >= 2.0.0
- torchvision >= 0.15.0
- NumPy >= 1.24.0

**3D Rendering:**

- trimesh >= 3.21.0
- open3d >= 0.17.0
- PyTorch3D >= 0.7.5 (optional)

**Analysis:**

- scikit-learn >= 1.3.0
- scipy >= 1.10.0
- matplotlib >= 3.7.0
- seaborn >= 0.13.0 (for comprehensive analysis visualizations)

**Configuration:**

- PyYAML >= 6.0
- python-dotenv >= 1.0.0 (for `.env` file support)

**Data Download:**

- huggingface_hub (for HuggingFace dataset downloads)

See `requirements.txt` for complete list.

## Development

### Running Tests

```bash
# Test data pipeline
python test_data_pipeline.py

# Test individual components
python -m src.data.renderer
python -m src.data.nback_generator
```

### Creating Custom Experiments

1. **Create a new config:**

   ```yaml
   # configs/my_experiment.yaml
   experiment_name: my_experiment
   hidden_size: 512
   rnn_type: gru
   n_values: [2, 3]
   task_features: ["location", "identity"]
   # ... other parameters
   ```

2. **Train:**

   ```bash
   python -m src.train --config configs/my_experiment.yaml
   ```

3. **Analyze:**

   ```bash
   python -m src.analysis.decoding --hidden_root runs/my_experiment/hidden_states
   ```

### Code Structure

**Models** (`src/models/`):

- `perceptual.py`: ResNet50-based visual encoder
- `cognitive.py`: RNN/GRU/LSTM variants
- `attention.py`: Task-guided attention mechanism
- `wm_model.py`: Full working memory model
- `model_factory.py`: Unified model creation interface

**Data** (`src/data/`):

- `shapenet_downloader.py`: Dataset management
- `renderer.py`: 3D → 2D rendering
- `nback_generator.py`: Trial sequence generation
- `dataset.py`: PyTorch Dataset/DataLoader

**Analysis** (`src/analysis/`):

- `decoding.py`: Linear SVM decoding
- `orthogonalization.py`: Representational geometry
- `procrustes.py`: Temporal dynamics via rotation
- `activations.py`: Hidden state utilities

## Advanced Analysis

### Procrustes Analysis (Temporal Dynamics)

Orthogonal Procrustes alignment reveals how neural representations transform over time:

**Key Analyses:**

#### 1. Procrustes Alignment

Finds optimal rotation matrices that align decoder weights across time points:

```bash
# Basic Procrustes between two time points
python -m src.analysis.procrustes \
  --hidden_root runs/wm_mtmf/hidden_states \
  --property identity \
  --source_time 2 --target_time 3 \
  --task location --n 2
```

**Output metrics:**

- **Procrustes disparity**: Measures alignment quality (lower = better)
- **Reconstruction accuracy**: How well rotated weights decode new time point
- **Accuracy ratio**: Reconstruction vs. baseline performance

#### 2. Swap Hypothesis Test (Figure 4g Replication)

Tests the core finding about chronological organization:

```bash
# Run swap test
python -m src.analysis.procrustes \
  --hidden_root runs/wm_mtmf/hidden_states \
  --property identity \
  --swap_test \
  --encoding_time 2 \
  --k_offset 1 \
  --task location --n 2
```

**Three rotations compared:**

- **Correct**: R(S=i, T=j) - Actual stimulus-time rotation
- **Swap 1** (wrong time): R(S=i, T=j+1) - Same stimulus, different time
- **Swap 2** (same age): R(S=i+k, T=j+k) - Different stimulus, same relative age

**Key finding**: Swap 2 maintains higher accuracy than Swap 1, showing that temporal structure dominates stimulus identity.

#### 3. Demo Scripts

Interactive demonstrations:

```bash
# Full demo with all analyses
python demo_procrustes.py --hidden_root runs/wm_mtmf/hidden_states

# Specific demo
python demo_procrustes.py --demo swap --property identity --visualize

# Different properties
python demo_procrustes.py --property location
python demo_procrustes.py --property category
```

#### 4. Batch Analysis (Full Figure 4 Replication)

Comprehensive analysis across all time points:

```bash
# Complete Figure 4 analysis
python -m src.analysis.analyze_procrustes_batch \
  --hidden_root runs/wm_mtmf/hidden_states \
  --property identity \
  --n 2 \
  --visualize

# Fast version (skip slow temporal generalization)
python -m src.analysis.analyze_procrustes_batch \
  --hidden_root runs/wm_mtmf/hidden_states \
  --property identity \
  --skip_tg \
  --visualize
```

**Outputs:**

- **Temporal generalization matrix**: Cross-time decoding accuracy
- **Procrustes disparity matrix**: Alignment quality across time pairs
- **Swap test results**: Chronological organization validation
- **Figure 4 visualization**: Publication-ready plots

### Complete Analysis Workflow

1. **Train model** with hidden state saving:

   ```bash
   python -m src.train --config configs/mtmf.yaml
   ```

2. **Generate full analysis**:

   ```bash
   python -m src.analysis.analyze_procrustes_batch \
     --hidden_root runs/wm_mtmf/hidden_states \
     --property identity --n 2 --visualize
   ```

3. **Compare across properties**:

   ```bash
   for prop in location identity category; do
     python -m src.analysis.analyze_procrustes_batch \
       --hidden_root runs/wm_mtmf/hidden_states \
       --property $prop --visualize
   done
   ```

### Interpretation

**Procrustes Disparity:**

- Low values (< 0.1): Strong alignment between time points
- High values (> 0.5): Substantial representational change
- Adjacent times typically have lower disparity

**Swap Test Results:**

- Swap 2 > Swap 1: Confirms chronological organization
- High reconstruction accuracy: Linear transformability over time
- Consistent across properties: Universal temporal dynamics

**Scientific Implications:**

- Representations transform in chronologically-organized subspaces
- Temporal structure preserved across different stimuli
- Memory maintenance follows predictable geometric trajectories

## Task-Guided Attention

### Attention Overview

The attention mechanism enables spatial focus on task-relevant features:

**Mechanism:**

The **TaskGuidedAttention** module allows the model to focus on task-relevant spatial locations:

```text
CNN Feature Map (B, T, C, H', W') + Task Vector (B, 3)
    ↓
Attention Mechanism
    ↓
Context Vector (B, T, C) → RNN → Classifier
```

**Key features:**

- Spatial attention over CNN feature maps
- Task-conditioned (different attention for different tasks)
- Multiplicative attention with learned projections
- Returns attention weights for visualization

### Training Attention Models

```bash
# Train attention-enhanced models
python -m src.train --config configs/attention_stsf.yaml
python -m src.train --config configs/attention_stmf.yaml
python -m src.train --config configs/attention_mtmf.yaml

# Model types: attention_gru, attention_lstm, attention_rnn
```

### Baseline vs Attention Comparison

Compare baseline vs. attention models across all metrics:

```bash
# Comprehensive comparison
python -m src.analysis.compare_models \
  --baseline runs/wm_mtmf/hidden_states \
  --attention runs/wm_attention_mtmf/hidden_states \
  --property identity \
  --output_dir results/comparison
```

**Metrics compared:**

- **Decoding**: Task-irrelevant information preservation
- **Orthogonalization**: Representational geometry
- **Procrustes**: Temporal dynamics
- **Swap Test**: Chronological organization

### Visualizing Attention Heatmaps

Generate and analyze attention heatmaps:

```bash
# Visualize attention patterns
python -m src.analysis.visualize_attention \
  --checkpoint runs/wm_attention_mtmf/checkpoints/best_*.pt \
  --num_samples 5 \
  --output_dir results/attention_viz
```

**Outputs:**

- Attention heatmaps overlaid on input images
- Per-timestep attention weights
- Attention statistics (sparsity, entropy)
- Comparison across tasks

### Model Factory

Unified interface for creating models:

```python
from src.models import create_model

# Baseline models
baseline_gru = create_model('gru', hidden_size=512)
baseline_lstm = create_model('lstm', hidden_size=512)

# Attention models
attention_gru = create_model('attention_gru', hidden_size=512)
attention_lstm = create_model('attention_lstm', hidden_size=512)
```

### Complete Workflow

1. **Train both models:**

   ```bash
   python -m src.train --config configs/mtmf.yaml           # Baseline
   python -m src.train --config configs/attention_mtmf.yaml # Attention
   ```

2. **Compare performance:**

   ```bash
   python -m src.analysis.compare_models --baseline ... --attention ...
   ```

3. **Visualize attention:**

   ```bash
   python -m src.analysis.visualize_attention --checkpoint ...
   ```

4. **Analyze results:**
   - Check `results/comparison/comparison.json` for metrics
   - Review attention heatmaps in `results/attention_viz/`
   - Compare training curves and final accuracy

### Expected Results

**Performance improvements:**

- 5-10% higher validation accuracy
- Faster convergence (fewer epochs needed)
- Better task-specific representations

**Attention patterns:**

- **Location task**: Focus on spatial positions (corners/edges)
- **Identity task**: Focus on object features (center)
- **Category task**: Distributed attention (multiple features)

**Representational changes:**

- Higher orthogonalization index
- Better decoding of task-relevant features
- Maintained temporal dynamics

### Available Tools

| Tool | Purpose | Output |
|------|---------|--------|
| `src.analysis.compare_models` | Full comparison | JSON with all metrics |
| `src.analysis.visualize_attention` | Attention heatmaps | PNG visualizations |
| `src/models.model_factory` | Model creation | Model instances |

## Documentation

### Phase 6: Comprehensive Analysis Pipeline (NEW)

**Quick Start:**
- [COMPREHENSIVE_ANALYSIS_READY.md](COMPREHENSIVE_ANALYSIS_READY.md) - **Start here** - Quick start guide
- [ANALYSIS_CHECKLIST.md](ANALYSIS_CHECKLIST.md) - Detailed task-by-task checklist for all 5 analyses
- [PHASE6_IMPLEMENTATION.md](PHASE6_IMPLEMENTATION.md) - Technical implementation details
- [ANALYSIS_IMPLEMENTATION_SUMMARY.md](ANALYSIS_IMPLEMENTATION_SUMMARY.md) - Implementation status summary

**Previous Documentation:**
- [ANALYSIS_METHODOLOGY.md](docs/ANALYSIS_METHODOLOGY.md) - Comprehensive analysis guide
- [PROCRUSTES_GUIDE.md](docs/PROCRUSTES_GUIDE.md) - Procrustes analysis tutorial
- [PHASE4_SUMMARY.md](docs/PHASE4_SUMMARY.md) - Temporal dynamics analysis
- [PHASE5_SUMMARY.md](docs/PHASE5_SUMMARY.md) - Attention mechanism details
- [PROJECT_COMPLETE.md](docs/PROJECT_COMPLETE.md) - Complete project overview
- [QUICK_REFERENCE.md](docs/QUICK_REFERENCE.md) - Command reference

## Troubleshooting

### HuggingFace token not found

- Create a `.env` file in the project root (copy from `.env.example`)
- Add your token: `HUGGINGFACE_TOKEN=your_token_here`
- Alternatively, use `--hf-token YOUR_TOKEN` argument
- Get a token from [HuggingFace Settings](https://huggingface.co/settings/tokens)
- Ensure you have access to the ShapeNet dataset on HuggingFace Hub

### CUDA out of memory

- Reduce `batch_size` in config file
- Use smaller `hidden_size` (e.g., 256 instead of 512)
- Enable gradient checkpointing

### PyTorch3D installation fails

- PyTorch3D is optional; the project uses Open3D as alternative
- Follow [PyTorch3D installation guide](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md)

### No stimuli found

- Run `python -m src.data.download_shapenet --placeholder` first
- Check `data/stimuli/` directory exists
- Use `--use_real_stimuli false` in config for placeholder data

### Hidden states not saved

- Ensure `save_hidden: true` in config file
- Or use `--save_hidden` flag when running training script
- Check available disk space in `experiments/` or `runs/` directory
- Note: `train_with_generalization.py` saves hidden states by default

### Validation splits test failed

- Ensure stimuli are generated: `python -m src.data.generate_stimuli`
- You need at least 5 identities per category for proper splits
- Run verification: `python -m src.data.test_validation_splits`
- Check that you have 320 stimuli (5 IDs × 4 categories × 4 locations × 4 angles)

### Analysis pipeline errors

- Run setup verification: `python scripts/verify_analysis_setup.py`
- Install missing dependencies: `pip install seaborn`
- Ensure training log exists: `experiments/<name>/training_log.json`
- Check hidden states directory structure is correct

### python-dotenv not installed

- Install with: `pip install python-dotenv`
- Or continue without `.env` support by using `--hf-token` argument
- Included in `requirements.txt` for easy installation

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{wm_model_2024,
  title = {Working Memory Model: A PyTorch Framework for N-back Tasks},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/yourusername/WM-model}
}
```

## Acknowledgments

- Inspired by neuroscience research on working memory and temporal dynamics
- Built with PyTorch, PyTorch3D, and Open3D
- ShapeNet dataset: [https://shapenet.org/](https://shapenet.org/)

## Contact

For questions or issues, please open an issue on GitHub or contact the maintainers.
