# Task-Guided Attention

Merged from the former `attention_model_guide.md` and `ATTENTION_MODES_DETAILED.md`,
which covered the same architecture twice and both carried result tables from a
training run that the current deck supersedes.

**This document describes the architecture only. It reports no results.**
Every accuracy figure for these models lives in one place — `docs/RESULTS.md` —
so there is exactly one number to correct when a run is repeated.

---

## Overview

The model performs N-back tasks on naturalistic images. Given a sequence of object images and a task instruction (match by location, identity, or category), the model must determine if the current stimulus matches the one seen N steps ago.

**Key Innovation**: The Feature-Channel Attention mechanism filters task-irrelevant features before they enter the RNN, allowing the model to focus on task-relevant information and suppress distractors.

---

## Architecture Diagram

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

## The Problem: Why Attention is Needed

### Baseline Model Limitation

The CNN extracts features that encode ALL object properties simultaneously:
- **Location features**: Where is the object in the image?
- **Identity features**: Which specific object is this?
- **Category features**: What type of object is this?

In the baseline model, ALL these features pass directly to the RNN, even when only one property matters for the current task. This creates interference and reduces generalization performance.

### Solution: Task-Guided Channel Gating

The Feature-Channel Attention learns to:
1. **Amplify** channels encoding task-relevant information
2. **Suppress** channels encoding task-irrelevant information

This filtering happens BEFORE the RNN, so the memory system only receives task-relevant information.

```
Example: Location Task

CNN Features (256 channels):
┌────────────────────────────────────────────────────────────────┐
│ ch 0-85: location info │ ch 86-170: identity │ ch 171-255: category │
└────────────────────────────────────────────────────────────────┘
                              │
                              ▼ Task-guided gates
┌────────────────────────────────────────────────────────────────┐
│ gates ≈ 0.9 (KEEP)     │ gates ≈ 0.1 (SUPPRESS) │ gates ≈ 0.1 (SUPPRESS) │
└────────────────────────────────────────────────────────────────┘
```

---

## Attention Modes

The model supports two attention modes, controlled by the `attention_mode` configuration parameter.

### 1. Task-Only Mode

In this mode, the gates depend **only** on the task vector. The same gates are applied to all timesteps in a sequence.

**How it works:**
- The task vector (6 dimensions) passes through a 3-layer MLP
- A learnable task-specific bias is added
- Sigmoid activation produces gates in the range [0, 1]
- Gates are multiplied element-wise with CNN features

**Characteristics:**
- Simpler architecture with fewer parameters
- More interpretable - each task learns a fixed gating pattern
- Consistent filtering regardless of input content
- Recommended for most use cases

### 2. Dual Mode

In this mode, the gates depend on **both** the task vector AND the current input features. This allows adaptive gating based on stimulus content.

**How it works:**
- Task vector is projected to a query space
- CNN features are projected to a key space
- Query and key are combined via element-wise multiplication
- Combined representation passes through an MLP to produce gates

**Characteristics:**
- More expressive - can adapt gating to specific inputs
- Higher capacity but more complex
- Harder to interpret
- Slightly better performance on multi-task scenarios (MTMF)

---

## Module Descriptions

### 1. Perceptual Module

The perceptual module extracts visual features from input images using a pre-trained ResNet50 backbone.

**Components:**
- ResNet50 backbone (frozen weights from ImageNet pre-training)
- 1×1 convolution layer reducing 2048 channels to 256
- Global Average Pooling (GAP) to produce a single feature vector per image

**Processing:**
- Input images are flattened across batch and time dimensions
- Each 224×224 RGB image produces a 256-dimensional feature vector
- Features are reshaped back to (Batch, Time, 256) format

The frozen backbone ensures stable visual representations while the 1×1 convolution learns task-relevant channel combinations.

### 2. Feature-Channel Attention Module

This is the key innovation of the attention model. It computes channel-wise gates based on task identity.

**Architecture (Task-Only Mode):**
- Gate network: 3-layer MLP (6 → 256 → 256 → 256) with ReLU and dropout
- Task bias: Learnable parameter matrix (6 × 256)
- Output activation: Sigmoid to constrain gates to [0, 1]

**Architecture (Dual Mode):**
- Task projection: Linear layer (6 → 256) with ReLU
- Feature projection: Linear layer (256 → 256) with ReLU
- Gate network: 2-layer MLP with dropout
- Output activation: Sigmoid

**Gate Application:**
Gates are applied via element-wise multiplication with the CNN features. A gate value of 1.0 preserves the channel completely, while 0.0 suppresses it entirely.

### 3. Cognitive Module

The cognitive module processes the gated features over time using a recurrent neural network.

**Preprocessor:**
- Linear projection from (256 + 6) to 256 dimensions
- Layer normalization for training stability
- ReLU activation

**RNN Options:**
- GRU (default): Gated Recurrent Unit with update and reset gates
- LSTM: Long Short-Term Memory with cell state
- Vanilla RNN: Simple recurrent network with tanh activation

The preprocessor normalizes the concatenated input (gated features + task vector) before RNN processing. This improves training stability and convergence.

### 4. Classifier

A simple linear layer mapping RNN hidden states to response logits.

**Output Classes:**
- Class 0: `no_action` - timesteps before N-back comparison is possible
- Class 1: `non_match` - current stimulus differs from N-back stimulus
- Class 2: `match` - current stimulus matches N-back stimulus

---

## Tensor Dimensions

| Stage | Tensor | Shape | Description |
|-------|--------|-------|-------------|
| Input | images | (B, T, 3, 224, 224) | RGB image sequences |
| Input | task_vector | (B, 6) | One-hot: [feature(3), n_back(3)] |
| Perceptual | cnn_features | (B, T, 256) | Visual embeddings |
| Attention | gates | (B, T, 256) | Channel gates [0,1] |
| Attention | gated_features | (B, T, 256) | Filtered features |
| Cognitive | rnn_input | (B, T, 262) | Gated features + task |
| Cognitive | hidden_seq | (B, T, 256) | RNN hidden states |
| Classifier | logits | (B, T, 3) | Response scores |

---

## Comparison: Baseline vs Attention Model

| Aspect | Baseline Model | Attention Model |
|--------|----------------|-----------------|
| Feature filtering | None | Task-guided channel gating |
| Task information | Concatenated with features | Gates channels + concatenated |
| Task-irrelevant info | Preserved in RNN | Suppressed before RNN |
| Generalization | Lower | Higher (+12-16%) |
| Training convergence | Faster initial learning | Slower start, better final |

---

## Configuration Options

### Model Selection

Set `model_type: "attention"` in the YAML config to use the attention model instead of the baseline.

### Attention Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `attention_mode` | "task_only" or "dual" | "task_only" |
| `attention_hidden_dim` | Hidden dimension for gate MLP | 256 |
| `attention_dropout` | Dropout rate in gate network | 0.1 |

### Available Configurations

| Config File | Attention Mode | Tasks | N-back Values |
|-------------|----------------|-------|---------------|
| `attention_stmf.yaml` | task_only | location, identity, category | 2 |
| `attention_mtmf.yaml` | task_only | location, identity, category | 1, 2, 3 |
| `dual_attention_stmf.yaml` | dual | location, identity, category | 2 |
| `dual_attention_mtmf.yaml` | dual | location, identity, category | 1, 2, 3 |

## Architecture Comparison

### Task-Only Mode

#### How It Works

Task-Only mode computes channel gates based exclusively on the task identity. The same gates are applied to all timesteps in a sequence.

#### Mathematical Formulation

```
Input:
  - task_vector: (B, 6) where 6 = [feature_one_hot(3), n_back_one_hot(3)]
  - features: (B, T, 256) or flattened to (B*T, 256)

Gate Computation:
  1. gate_logits = MLP_3layer(task_vector)
     - Linear(6 → H) + ReLU + Dropout
     - Linear(H → H) + ReLU + Dropout
     - Linear(H → 256)
     Output shape: (B, 256)

  2. task_bias = task_vector @ task_bias_matrix
     - task_bias_matrix: learnable parameter (6, 256)
     - Output shape: (B, 256)

  3. gate_logits = gate_logits + task_bias
     Output shape: (B, 256)

  4. gates = sigmoid(gate_logits)
     Output shape: (B, 256)
     Range: [0, 1] for each channel

Gating Application:
  gated_features = features * gates
  - Element-wise multiplication
  - Each of 256 channels is scaled by its corresponding gate value
  - Output shape: same as input features
```

#### Code Implementation

```python
def _build_task_only_attention(self, dropout: float):
    """Task-only: gates = f(task_vector)"""
    self.gate_network = nn.Sequential(
        nn.Linear(self.task_dim, self.hidden_dim),      # 6 → 256
        nn.ReLU(inplace=True),
        nn.Dropout(dropout),
        nn.Linear(self.hidden_dim, self.hidden_dim),    # 256 → 256
        nn.ReLU(inplace=True),
        nn.Dropout(dropout),
        nn.Linear(self.hidden_dim, self.feature_dim),   # 256 → 256
    )
    self.task_bias = nn.Parameter(torch.zeros(self.task_dim, self.feature_dim))

def _compute_gates(self, features, task_vector):
    # Gates depend only on task
    gate_logits = self.gate_network(task_vector)       # (B, 256)
    task_bias = torch.matmul(task_vector, self.task_bias)  # (B, 256)
    gate_logits = gate_logits + task_bias              # (B, 256)
    gates = torch.sigmoid(gate_logits)                 # (B, 256)
    
    gated_features = features * gates                  # (B, 256) or (B, T, 256)
    return gated_features, gates
```

#### Characteristics

| Aspect | Details |
|--------|---------|
| **Gate Computation** | Depends only on task vector |
| **Temporal Consistency** | Same gates applied to all T timesteps |
| **Adaptability** | Fixed gating pattern per task |
| **Parameters** | Fewer (only task-dependent) |
| **Interpretability** | High - each task learns a fixed gating pattern |
| **Computational Cost** | Lower - gates computed once per batch |
| **Use Case** | When task-relevant features are consistent across inputs |

#### Example: Location Task

```
Task Vector: [1, 0, 0, 0, 1, 0]  (location task, N=2)
                ↓
MLP processes task → produces 256 gate values
                ↓
Gates ≈ [0.9, 0.9, ..., 0.1, 0.1, ..., 0.2, 0.2, ...]
         (location channels)  (identity channels)  (category channels)
                ↓
Applied to ALL timesteps in sequence:
  t=0: features[0] * gates
  t=1: features[1] * gates  (same gates)
  t=2: features[2] * gates  (same gates)
  ...
  t=5: features[5] * gates  (same gates)
```

---

### Dual Mode

#### How It Works

Dual mode computes adaptive gates that depend on both the task vector AND the current input features. This allows the gating to adapt based on the specific stimulus content at each timestep.

#### Mathematical Formulation

```
Input:
  - task_vector: (B, 6)
  - features: (B, T, 256) or flattened to (B*T, 256)

Gate Computation:

  1. Task Projection (Query):
     task_query = MLP_task(task_vector)
     - Linear(6 → H) + ReLU
     Output shape: (B, 256)

  2. Feature Projection (Key):
     feature_key = MLP_feature(features)
     - Linear(256 → H) + ReLU
     Output shape: (B, 256)

  3. Interaction (Element-wise Multiplication):
     combined = task_query * feature_key
     - Element-wise multiplication (Hadamard product)
     - Combines task and feature information
     Output shape: (B, 256)

  4. Gate Network:
     gate_logits = MLP_gate(combined)
     - Linear(256 → 256) + ReLU + Dropout
     - Linear(256 → 256)
     Output shape: (B, 256)

  5. Sigmoid Activation:
     gates = sigmoid(gate_logits)
     Output shape: (B, 256)
     Range: [0, 1] for each channel

Gating Application:
  gated_features = features * gates
  - Element-wise multiplication
  - Output shape: same as input features
```

#### Code Implementation

```python
def _build_dual_attention(self, dropout: float):
    """Dual: gates = f(task_vector, features)"""
    # Project task to query space
    self.task_proj = nn.Sequential(
        nn.Linear(self.task_dim, self.hidden_dim),     # 6 → 256
        nn.ReLU(inplace=True),
    )
    # Project features to key space
    self.feature_proj = nn.Sequential(
        nn.Linear(self.feature_dim, self.hidden_dim),  # 256 → 256
        nn.ReLU(inplace=True),
    )
    # Compute gates from combined representation
    self.gate_network = nn.Sequential(
        nn.Linear(self.hidden_dim, self.hidden_dim),   # 256 → 256
        nn.ReLU(inplace=True),
        nn.Dropout(dropout),
        nn.Linear(self.hidden_dim, self.feature_dim),  # 256 → 256
    )

def _compute_gates(self, features, task_vector):
    # Gates depend on both task and features
    task_query = self.task_proj(task_vector)           # (B, 256)
    feature_key = self.feature_proj(features)          # (B, 256)
    combined = task_query * feature_key                # (B, 256) - element-wise mult
    gate_logits = self.gate_network(combined)          # (B, 256)
    gates = torch.sigmoid(gate_logits)                 # (B, 256)
    
    gated_features = features * gates                  # (B, 256) or (B, T, 256)
    return gated_features, gates
```

#### Characteristics

| Aspect | Details |
|--------|---------|
| **Gate Computation** | Depends on both task AND features |
| **Temporal Consistency** | Different gates for each timestep |
| **Adaptability** | Adaptive - gates change based on input content |
| **Parameters** | More (task + feature projections) |
| **Interpretability** | Lower - gates depend on specific inputs |
| **Computational Cost** | Higher - gates computed per timestep |
| **Use Case** | When task-relevant features vary across inputs |

#### Example: Location Task with Dual Mode

```
Timestep t=0:
  Task Vector: [1, 0, 0, 0, 1, 0]  (location task, N=2)
  Features[0]: [0.5, 0.3, ..., 0.8, 0.2, ...]  (specific image features)
                ↓
  task_query = MLP_task(task_vector)
  feature_key = MLP_feature(features[0])
  combined = task_query * feature_key  (element-wise)
                ↓
  gates[0] ≈ [0.85, 0.88, ..., 0.12, 0.15, ..., 0.18, 0.22, ...]

Timestep t=1:
  Task Vector: [1, 0, 0, 0, 1, 0]  (same task)
  Features[1]: [0.7, 0.4, ..., 0.6, 0.3, ...]  (different image features)
                ↓
  task_query = MLP_task(task_vector)  (same as t=0)
  feature_key = MLP_feature(features[1])  (different!)
  combined = task_query * feature_key  (different interaction)
                ↓
  gates[1] ≈ [0.82, 0.90, ..., 0.14, 0.13, ..., 0.20, 0.19, ...]
             (different gates due to different features)
```

---

## Detailed Comparison

### Processing Flow

#### Task-Only Mode Flow
```
Batch of sequences: (B, T, 256)
         ↓
Flatten time: (B*T, 256)
         ↓
Extract task_vector: (B*T, 6)
         ↓
Compute gates from task only: (B*T, 256)
         ↓
Apply gates: (B*T, 256) * (B*T, 256) = (B*T, 256)
         ↓
Reshape back: (B, T, 256)
```

#### Dual Mode Flow
```
Batch of sequences: (B, T, 256)
         ↓
Flatten time: (B*T, 256)
         ↓
Extract task_vector: (B*T, 6)
         ↓
Project task: (B*T, 6) → (B*T, 256)
Project features: (B*T, 256) → (B*T, 256)
         ↓
Combine (element-wise mult): (B*T, 256)
         ↓
Compute gates from combined: (B*T, 256)
         ↓
Apply gates: (B*T, 256) * (B*T, 256) = (B*T, 256)
         ↓
Reshape back: (B, T, 256)
```

### Parameter Count

**Task-Only Mode:**
- gate_network: 6→256→256→256 = (6×256) + (256×256) + (256×256) ≈ 132K params
- task_bias: 6×256 = 1.5K params
- **Total: ~133.5K parameters**

**Dual Mode:**
- task_proj: 6→256 = 6×256 ≈ 1.5K params
- feature_proj: 256→256 = 256×256 ≈ 65K params
- gate_network: 256→256→256 = (256×256) + (256×256) ≈ 131K params
- **Total: ~197.5K parameters**

Dual mode has ~48% more parameters due to feature projection.

---

## When to Use Each Mode

### Use Task-Only Mode When:
- Task-relevant features are consistent across different stimuli
- You want faster inference (gates computed once per batch)
- You need better interpretability (fixed gating pattern per task)
- You have limited computational resources
- Training data is limited (fewer parameters to learn)

### Use Dual Mode When:
- Task-relevant features vary significantly across stimuli
- You need adaptive gating based on input content
- Computational cost is not a constraint

Note that this is an architectural rationale, not an empirical recommendation.
Measured across both hidden sizes, dual mode does not outperform task-only on any
scenario tested here (`docs/RESULTS.md`).

---

## Mathematical Intuition

### Task-Only: Fixed Filtering
```
Think of it as a fixed filter per task:
- Location task: "Always amplify spatial channels, suppress identity channels"
- Identity task: "Always amplify identity channels, suppress location channels"
- Category task: "Always amplify semantic channels, suppress fine-grained channels"

The filter doesn't change based on what you see.
```

### Dual: Adaptive Filtering
```
Think of it as a content-aware filter:
- Location task + specific image: "This image has clear spatial structure, amplify spatial channels more"
- Location task + ambiguous image: "This image is ambiguous, moderate amplification"
- Identity task + distinctive object: "This object is distinctive, strongly amplify identity channels"
- Identity task + similar objects: "These objects are similar, moderate amplification"

The filter adapts based on what you see.
```

---

## Implementation Details

### Sigmoid Activation

Both modes use sigmoid to constrain gates to [0, 1]:
```
gates = sigmoid(gate_logits) = 1 / (1 + exp(-gate_logits))

- gate_logits = 0 → gates = 0.5 (neutral)
- gate_logits > 0 → gates > 0.5 (amplify)
- gate_logits < 0 → gates < 0.5 (suppress)
```

### Element-wise Multiplication

Gating is applied via element-wise multiplication:
```
gated_features[i] = features[i] * gates[i]

For each of 256 channels:
- If gates[i] ≈ 1.0: channel is preserved
- If gates[i] ≈ 0.5: channel is halved
- If gates[i] ≈ 0.0: channel is suppressed
```

### Dropout in Gate Network

Dropout is applied during training to prevent overfitting:
- Task-Only: Dropout after each hidden layer in MLP
- Dual: Dropout in gate_network (after first ReLU)

---

## Summary Table

| Aspect | Task-Only | Dual |
|--------|-----------|------|
| **Gate Formula** | `sigmoid(MLP(task))` | `sigmoid(MLP(task_proj * feat_proj))` |
| **Temporal Variation** | No (same gates for all T) | Yes (different gates per timestep) |
| **Parameters** | ~133.5K | ~197.5K |
| **Inference Speed** | Faster | Slower |
| **Interpretability** | High | Medium |
| **Complexity** | Simple | Complex |
| **Best for** | The default | No scenario where it consistently wins — see `docs/RESULTS.md` |

---

## Results

Not reproduced here, deliberately. See **`docs/RESULTS.md`**, which records for
each claim the experiment directory, the artifact, and the slide that states it.

Two findings from that document bear directly on the architecture above, because
they bound where it should be used:

- Attention gains ~+11pp on the multi-feature scenarios (STMF, MTMF) on both
  generalization splits.
- Attention **costs** 11pp on STSF (single task, single feature), and 32pp at
  h=128. With one task there is no ambiguity for the gate to resolve, so gating
  contributes only optimization difficulty. This replicates in both hidden sizes.

Dual mode shows no consistent advantage over task-only. Prior versions of these
documents claimed it was better for multi-task scenarios and best for novel
identity; the current data does not support either claim.

---

## Why Attention Helps: Theoretical Explanation

The Feature-Channel Attention addresses a fundamental limitation identified in the original paper: baseline RNNs maintain task-irrelevant information because nothing filters it out.

**Mechanism:**
- The attention learns which feature channels encode which properties
- For location tasks: spatial encoding channels are preserved, object identity channels are suppressed
- For identity tasks: object-specific channels are preserved, location channels are suppressed
- For category tasks: semantic channels are preserved, fine-grained identity channels are suppressed

**Benefits:**
1. **Reduced interference**: Task-irrelevant features don't clutter RNN memory
2. **Cleaner comparisons**: N-back matching operates on relevant features only
3. **Better generalization**: Model learns task-specific feature selection rather than memorizing training examples

This explicit gating mechanism complements the RNN's memory dynamics, creating a more effective working memory system.
