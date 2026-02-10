# Feature-Channel Attention: Task-Only vs Dual Mode

## Overview

The Feature-Channel Attention module implements task-guided gating that filters CNN features before they enter the RNN. It supports two modes that differ in how they compute the gating weights:

1. **Task-Only Mode**: Gates depend solely on the task vector
2. **Dual Mode**: Gates depend on both the task vector AND the input features

---

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
- You have sufficient training data
- Computational cost is not a constraint
- You're working with multi-task scenarios (MTMF) where feature relevance varies

---

## Experimental Results

From the paper's experiments:

| Model | Train Acc | Val Novel Angle | Val Novel Identity |
|-------|-----------|-----------------|-------------------|
| STMF + Task-Only | 98.7% | 92.4% | 72.6% |
| STMF + Dual | 99.7% | 91.9% | 72.4% |
| MTMF + Task-Only | 99.0% | 90.1% | 70.6% |
| MTMF + Dual | 98.9% | 92.3% | 73.5% |

**Key Findings:**
- Task-Only performs better on single-task scenarios (STMF)
- Dual performs better on multi-task scenarios (MTMF)
- Dual achieves best novel identity generalization (73.5%)
- Task-Only is simpler and faster

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
| **Best For** | Single-task (STMF) | Multi-task (MTMF) |
| **Interpretability** | High | Medium |
| **Complexity** | Simple | Complex |
| **Generalization** | Good on novel angles | Better on novel identities |

