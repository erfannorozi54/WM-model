# Attention-Based Working Memory Model: Architecture Guide

This document provides a comprehensive walkthrough of the attention-based working memory model architecture, with exact tensor dimensions at each stage.

## Overview

The model performs N-back tasks on naturalistic images. Given a sequence of object images and a task instruction (match by location, identity, or category), the model must determine if the current stimulus matches the one seen N steps ago.

**Key Innovation**: The Feature-Channel Attention mechanism filters task-irrelevant features before they enter the RNN, allowing the model to focus on task-relevant information and suppress distractors.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                   FEATURE-CHANNEL ATTENTION WORKING MEMORY MODEL            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Images (B,T,3,224,224)                                                     │
│         │                                                                   │
│         ▼                                                                   │
│  ┌─────────────────┐                                                        │
│  │  PERCEPTUAL     │  ResNet50 (frozen) + 1×1 Conv + GAP                   │
│  │  MODULE         │  Output: Features (B,T,256)                           │
│  └────────┬────────┘  Contains: location + identity + category info        │
│           │                                                                 │
│           ▼                                                                 │
│  ┌─────────────────┐     Task Vector (B,6)                                 │
│  │  FEATURE-CHANNEL│◄────────────────────                                  │
│  │  ATTENTION      │  Channel-wise gating based on task                    │
│  └────────┬────────┘  Output: Gated Features (B,T,256)                     │
│           │           Task-irrelevant channels suppressed                   │
│           ▼                                                                 │
│  ┌─────────────────┐                                                        │
│  │  COGNITIVE      │  Preprocessor + GRU/LSTM processing over time         │
│  │  MODULE (RNN)   │  Output: Hidden States (B,T,256)                      │
│  └────────┬────────┘                                                        │
│           │                                                                 │
│           ▼                                                                 │
│  ┌─────────────────┐                                                        │
│  │  CLASSIFIER     │  Linear layer                                         │
│  │                 │  Output: Logits (B,T,3)                               │
│  └─────────────────┘  Classes: [no_action, non_match, match]               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## The Key Idea: Feature-Channel Attention

### Problem with Baseline Model
The CNN extracts features that encode ALL object properties:
- **Location features**: Where is the object in the image?
- **Identity features**: Which specific object is this?
- **Category features**: What type of object is this?

In the baseline model, ALL these features go to the RNN, even when only one property matters for the task.

### Solution: Task-Guided Channel Gating
The Feature-Channel Attention learns to:
1. **Amplify** channels encoding task-relevant information
2. **Suppress** channels encoding task-irrelevant information

```
Example: Location Task (task_vector = [1, 0, 0, 0, 1, 0] for 2-back)

CNN Features (256 channels):
┌────────────────────────────────────────────────────────────────┐
│ ch 0-85: location info │ ch 86-170: identity │ ch 171-255: category │
│        HIGH            │       LOW           │        LOW           │
└────────────────────────────────────────────────────────────────┘
                              │
                              ▼ Task-guided gates
┌────────────────────────────────────────────────────────────────┐
│ gates ≈ 0.9            │ gates ≈ 0.1         │ gates ≈ 0.1          │
└────────────────────────────────────────────────────────────────┘
                              │
                              ▼ Gated output
┌────────────────────────────────────────────────────────────────┐
│ PRESERVED              │ SUPPRESSED          │ SUPPRESSED           │
└────────────────────────────────────────────────────────────────┘
```

---

## Attention Modes

The `FeatureChannelAttention` module (in `src/models/attention.py`) supports two modes:

### 1. Task-Only Mode (`attention_mode: "task_only"`)

Gates depend **only** on the task vector - same gates applied to all timesteps.

```python
# Architecture
gate_network: Linear(6 → H) → ReLU → Dropout → Linear(H → H) → ReLU → Dropout → Linear(H → 256)
task_bias: Parameter(6, 256)  # Learnable task-specific bias

# Forward pass
gate_logits = gate_network(task_vector)           # (B, 256)
task_bias_contrib = task_vector @ task_bias       # (B, 256)
gates = sigmoid(gate_logits + task_bias_contrib)  # (B, 256)
gated_features = features * gates                 # (B, T, 256)
```

**Pros**: Simpler, more interpretable, consistent gating per task
**Cons**: Cannot adapt to specific input content

### 2. Dual Mode (`attention_mode: "dual"`)

Gates depend on **both** task vector AND input features - adaptive gating.

```python
# Architecture
task_proj: Linear(6 → H) → ReLU
feature_proj: Linear(256 → H) → ReLU
gate_network: Linear(H → H) → ReLU → Dropout → Linear(H → 256)

# Forward pass
task_query = task_proj(task_vector)       # (B, H)
feature_key = feature_proj(features)      # (B, H)
combined = task_query * feature_key       # (B, H) element-wise
gate_logits = gate_network(combined)      # (B, 256)
gates = sigmoid(gate_logits)              # (B, 256)
gated_features = features * gates         # (B, T, 256)
```

**Pros**: Can adapt gating based on current stimulus
**Cons**: More complex, harder to interpret

---

## Model Architecture Details

### AttentionWorkingMemoryModel

Located in `src/models/attention.py`:

```python
class AttentionWorkingMemoryModel(nn.Module):
    def __init__(
        self,
        perceptual: PerceptualModule,      # ResNet50-based
        cognitive: CognitiveModule,         # GRU/LSTM/RNN
        hidden_size: int,                   # 256
        attention_hidden_dim: int = None,   # Default: hidden_size
        attention_dropout: float = 0.1,
        attention_mode: str = 'task_only',  # or 'dual'
        classifier_layers: List[int] = None,
    ):
        self.perceptual = perceptual
        self.attention = FeatureChannelAttention(...)
        self.cognitive = cognitive
        self.classifier = nn.Linear(hidden_size, 3)
```

### Forward Pass

```python
def forward(self, images, task_vector, return_attention=False, return_cnn_activations=False):
    B, T = images.shape[:2]
    
    # 1. Perceptual: CNN feature extraction
    x = images.reshape(B * T, 3, 224, 224)
    cnn_features, _ = self.perceptual(x)          # (B*T, 256)
    cnn_features = cnn_features.view(B, T, -1)    # (B, T, 256)
    
    # 2. Attention: Task-guided channel gating
    gated_features, gates = self.attention(cnn_features, task_vector)  # (B, T, 256)
    
    # 3. Cognitive: RNN processing
    task_rep = task_vector.unsqueeze(1).expand(B, T, 6)
    cog_in = torch.cat([gated_features, task_rep], dim=-1)  # (B, T, 262)
    outputs, final_state, hidden_seq = self.cognitive(cog_in)
    
    # 4. Classifier
    logits = self.classifier(outputs)  # (B, T, 3)
    
    return logits, hidden_seq, final_state
```

---

## Configuration

### YAML Config Parameters

```yaml
# Model type selection
model_type: "attention"  # Enables AttentionWorkingMemoryModel

# Attention-specific parameters
attention_hidden_dim: 256    # Hidden dim for gate computation
attention_dropout: 0.1       # Dropout in gate network
attention_mode: "task_only"  # "task_only" or "dual"

# Standard model parameters
hidden_size: 256
rnn_type: "gru"              # gru|lstm|rnn
num_layers: 1
```

### Available Configs

| Config | Mode | Description |
|--------|------|-------------|
| `attention_stmf.yaml` | task_only | Single-task multi-feature with attention |
| `attention_mtmf.yaml` | task_only | Multi-task multi-feature with attention |
| `dual_attention_stmf.yaml` | dual | STMF with dual (adaptive) attention |
| `dual_attention_mtmf.yaml` | dual | MTMF with dual (adaptive) attention |

---

## Dimension Summary Table

| Stage | Tensor | Shape | Description |
|-------|--------|-------|-------------|
| Input | images | (B, T, 3, 224, 224) | RGB images |
| Input | task_vector | (B, 6) | [feature(3), n(3)] |
| Perceptual | resnet_out | (B*T, 2048, 7, 7) | Raw CNN features |
| Perceptual | reduced | (B*T, 256, 7, 7) | After 1×1 conv |
| Perceptual | cnn_features | (B, T, 256) | After GAP + reshape |
| Attention | gates | (B, T, 256) | Sigmoid gates [0,1] |
| Attention | gated_features | (B, T, 256) | Filtered features |
| Cognitive | cog_input | (B, T, 262) | Gated features + task |
| Cognitive | preprocessed | (B, T, 256) | After Linear+LN+ReLU |
| Cognitive | hidden_seq | (B, T, 256) | RNN hidden states |
| Classifier | logits | (B, T, 3) | Response scores |

---

## Comparison: Baseline vs Attention Model

| Aspect | Baseline (`WorkingMemoryModel`) | Attention (`AttentionWorkingMemoryModel`) |
|--------|--------------------------------|------------------------------------------|
| CNN Output | Features (B, T, 256) | Features (B, T, 256) |
| Task Usage | Concatenated with features | **Gates channels** + concatenated |
| Feature Filtering | None | Task-irrelevant suppressed |
| RNN Input | CNN features + task (262) | **Gated** features + task (262) |
| Task-Irrelevant Info | Preserved in RNN | **Filtered before RNN** |

---

## Experimental Results

| Model | Train Masked | Val Novel Angle | Val Novel Identity |
|-------|--------------|-----------------|-------------------|
| STMF (Baseline) | 82.7% | 80.0% | 57.6% |
| MTMF (Baseline) | 83.0% | 78.8% | 57.0% |
| **STMF + Attention** | **98.7%** | **92.4%** | **72.6%** |
| **STMF + Dual Attention** | **99.7%** | **91.9%** | **72.4%** |
| **MTMF + Attention** | **99.0%** | **90.1%** | **70.6%** |
| **MTMF + Dual Attention** | **98.9%** | **92.3%** | **73.5%** |

**Key Findings**:
- Attention improves novel angle generalization by ~12-13%
- Novel identity generalization improves by ~15-16%
- Dual attention slightly better for MTMF (multi-task scenarios)

---

## Usage

### Training

```bash
# Task-only attention
python -m src.train_with_generalization --config configs/attention_stmf.yaml

# Dual attention
python -m src.train_with_generalization --config configs/dual_attention_mtmf.yaml
```

### Accessing Attention Gates

```python
model = AttentionWorkingMemoryModel(...)
logits, hidden, state, gates = model(images, task_vector, return_attention=True)

# gates shape: (B, T, 256) - gate values per channel per timestep
# Analyze which channels are activated for each task
```

### Visualizing Attention

```bash
python -m src.analysis.visualize_attention \
  --checkpoint experiments/<exp>/best_model.pt \
  --output_dir attention_viz/
```

---

## Key Insight: Why Attention Helps

The Feature-Channel Attention computes **channel-wise** gates:
- It learns which **feature channels** are important for each task
- For **location tasks**: Keeps spatial encoding channels, suppresses object identity channels
- For **identity tasks**: Keeps object-specific channels, suppresses location channels
- For **category tasks**: Keeps semantic channels, suppresses fine-grained identity channels

**This directly addresses the paper's finding**: The baseline model maintains task-irrelevant information because nothing filters it out. The attention model explicitly gates channels based on task relevance, leading to better generalization.
