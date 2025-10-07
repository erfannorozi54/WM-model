# Working Memory Model

A PyTorch implementation of a neural network model for working memory experiments using N-back tasks with 3D object stimuli.

## Project Structure

```tree
WM-model/
├── src/                    # Source code
│   ├── data/              # Data handling modules
│   │   ├── shapenet_downloader.py  # ShapeNet dataset management
│   │   ├── renderer.py            # 3D object rendering
│   │   ├── nback_generator.py     # N-back trial generation
│   │   └── dataset.py             # PyTorch Dataset/DataLoader
│   ├── models/            # Neural network models
│   └── utils/             # Utility functions
├── data/                  # Data directory
│   ├── shapenet/         # ShapeNet 3D objects
│   ├── stimuli/          # Rendered 2D stimuli
│   └── sample_stimuli/   # Sample/demo stimuli
├── notebooks/            # Jupyter notebooks
├── models/               # Trained model checkpoints
└── runs/                 # Training outputs (checkpoints, hidden states)

```

## Quick Start

### 1. Environment Setup

Activate the environment:

```bash
# Unix/Linux/macOS
source activate_env.sh

# Windows
activate_env.bat

# Or manually:
source venv/bin/activate  # Unix/Linux/macOS
# venv\Scripts\activate.bat  # Windows
export PYTHONPATH="src:$PYTHONPATH"
```

### 2. Download and Setup Data

```python
# Download ShapeNet data (placeholder setup)
python -m src.data.shapenet_downloader

# Test the renderer
python -m src.data.renderer

# Test the N-back generator
python -m src.data.nback_generator

# Test the full data pipeline
python test_data_pipeline.py
```

### 3. Generate Stimuli

```python
from src.data.renderer import StimulusRenderer
from src.data.shapenet_downloader import ShapeNetDownloader

# Setup data
downloader = ShapeNetDownloader()
downloader.download_all_categories()

# Render stimuli
renderer = StimulusRenderer()
# Add rendering code here...
```

### 4. Create N-back Dataset

```python
from src.data.dataset import NBackDataModule
from src.data.nback_generator import TaskFeature

# Create data module
data_module = NBackDataModule(
    stimulus_data=your_stimulus_data,
    n_values=[1, 2, 3],
    task_features=[TaskFeature.LOCATION, TaskFeature.IDENTITY, TaskFeature.CATEGORY],
    batch_size=32
)

# Get data loaders
train_loader = data_module.train_dataloader()
val_loader = data_module.val_dataloader()
```

## Components

### Data Pipeline

1. **ShapeNet Downloader** (`src/data/shapenet_downloader.py`)
   - Downloads and organizes 3D object data
   - 4 categories, 2 identities each (as per paper)

2. **Stimulus Renderer** (`src/data/renderer.py`)
   - Renders 3D objects to 2D images
   - 4 screen locations, black background
   - Multiple viewing angles

3. **N-back Generator** (`src/data/nback_generator.py`)
   - Generates trial sequences for Location, Identity, Category tasks
   - Configurable N (1-back, 2-back, 3-back)
   - Flexible sequence length and match probability

4. **PyTorch Dataset** (`src/data/dataset.py`)
   - Wraps generators in PyTorch Dataset/DataLoader
   - Handles batching, image loading, preprocessing
   - Train/validation/test splits

### Key Features

- **Flexible N-back Tasks**: Support for 1-back, 2-back, 3-back conditions
- **Multiple Task Types**: Location, Identity, and Category matching
- **3D Stimulus Rendering**: From ShapeNet objects to 2D images
- **PyTorch Integration**: Full DataLoader support with transforms
- **Configurable Parameters**: Sequence length, match probability, batch size

## Model and Training (Phase 2)

- **Perceptual module** (`src/models/perceptual.py`): ResNet50 features from the penultimate conv stage (layer4). A 1x1 point-wise conv reduces channels 2048 → H (RNN hidden size), then global average pooling yields per-image embeddings.
- **Cognitive module** (`src/models/cognitive.py`): Base `CognitiveModule` plus variants: `VanillaRNN`, `GRUCog`, `LSTMCog`. Input is `[visual_embedding ; task_one_hot]` passed through `Linear → LayerNorm → ReLU` before the recurrent cell.
- **Full model** (`src/models/wm_model.py`): Combines perceptual + cognitive and adds a `Linear(H, 3)` classifier for three responses.
- **Hidden states for analyses**: During validation, per-timestep RNN hidden states are saved under `runs/<experiment>/hidden_states/epoch_xxx/` as `.pt` files containing `hidden (B,T,H)`, `logits (B,T,3)`, `task_vector (B,3)`, `n (B,)`, `targets (B,T)`.
- **Exact ResNet hook (optional)**: You can capture the exact `layer4[2].relu` activation via a forward hook by setting `capture_exact_layer42_relu=True` in `PerceptualModule` (enabled by default in `train.py`).

### Train

```bash
pip install -r requirements.txt

# Scenarios
python train.py --config configs/stsf.yaml   # STSF: Single-Task Single-Feature
python train.py --config configs/stmf.yaml   # STMF: Single-Task Multi-Feature
python train.py --config configs/mtmf.yaml   # MTMF: Multi-Task Multi-Feature

# Optional flags
python train.py --config configs/stsf.yaml --save_hidden     # force save hidden states
python train.py --config configs/stsf.yaml --no_save_hidden  # disable saving hidden states
```

Key config fields (see YAMLs in `configs/`):

- **hidden_size** (H), **rnn_type** (rnn|gru|lstm), **num_layers**, **dropout**
- **pretrained_backbone**, **freeze_backbone**, **capture_exact_layer42_relu**
- **n_values**, **task_features** (location|identity|category), **sequence_length**, **batch_size**
- **optimizer**: AdamW with MultiStepLR (`milestones`, `gamma`), `lr`, `weight_decay`, `grad_clip`

## Requirements

See `requirements.txt` for full dependency list. Key dependencies:

- PyTorch >= 2.0.0
- PyTorch3D >= 0.7.5 (for 3D rendering)
- scikit-learn >= 1.3.0 (for analyses)
- trimesh, open3d (3D processing; pyrender optional)
- PyYAML >= 6.0 (YAML config parsing)

## Development

### Running Tests

```bash
python test_data_pipeline.py
```

### Creating Custom Experiments

```python
# See notebooks/ directory for examples
# Customize n_values, task_features, and other parameters as needed
```

## Notes

- This implementation creates placeholder ShapeNet files for development
- For real experiments, download actual ShapeNet data from <https://shapenet.org/>
- The renderer can be extended for additional stimulus variations
- N-back parameters can be adjusted for different experimental designs

## Citation

If you use this code, please cite the original working memory paper that inspired this implementation.
