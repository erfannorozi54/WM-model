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
│  ┌─────────────────┐     Task Vector (B,3)                                 │
│  │  FEATURE-CHANNEL│◄────────────────────                                  │
│  │  ATTENTION      │  Channel-wise gating based on task                    │
│  └────────┬────────┘  Output: Gated Features (B,T,256)                     │
│           │           Task-irrelevant channels suppressed                   │
│           ▼                                                                 │
│  ┌─────────────────┐                                                        │
│  │  COGNITIVE      │  GRU/LSTM processing over time                        │
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
Example: Location Task (task_vector = [1, 0, 0])

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

## Example Input

Let's trace through a concrete example:

```
Task: 2-back Location matching
Sequence Length: T = 6
Batch Size: B = 1 (for clarity)

Input Sequence:
┌─────┬─────┬─────┬─────┬─────┬─────┐
│ t=0 │ t=1 │ t=2 │ t=3 │ t=4 │ t=5 │
├─────┼─────┼─────┼─────┼─────┼─────┤
│ ✈️   │ 🪑   │ 🚗   │ ✈️   │ 🪑   │ 🚗   │
│loc=0│loc=1│loc=2│loc=0│loc=3│loc=2│
└─────┴─────┴─────┴─────┴─────┴─────┘

Expected Responses (2-back location):
t=0: no_action (no 2-back reference)
t=1: no_action (no 2-back reference)
t=2: non_match (loc=2 vs t=0 loc=0)
t=3: non_match (loc=0 vs t=1 loc=1)
t=4: non_match (loc=3 vs t=2 loc=2)
t=5: MATCH     (loc=2 vs t=3 loc=0? NO → non_match)
     Actually: t=5 loc=2 vs t=3 loc=0 → non_match
```

---

## Stage 1: Input Preparation

### Images
```python
images.shape = (B, T, 3, H, W) = (1, 6, 3, 224, 224)
```

Each image is a 224×224 RGB image of a 3D object rendered at one of 4 locations.

### Task Vector
```python
task_vector.shape = (B, 3) = (1, 3)

# One-hot encoding:
# [1, 0, 0] = Location task
# [0, 1, 0] = Identity task  
# [0, 0, 1] = Category task

task_vector = [[1, 0, 0]]  # Location task
```

---

## Stage 2: Perceptual Module (ResNet50)

The perceptual module extracts visual features from each image.

### Architecture
```
ResNet50 Backbone (frozen):
  conv1 → bn1 → relu → maxpool → layer1 → layer2 → layer3 → layer4
  
Output of layer4: (B*T, 2048, 7, 7)

1×1 Convolution (trainable):
  Reduces 2048 channels → 256 channels
  
Output: (B*T, 256, 7, 7)
```

### Tensor Flow
```python
# Flatten batch and time for CNN processing
x = images.reshape(B*T, 3, 224, 224)  # (6, 3, 224, 224)

# ResNet50 feature extraction
features = resnet_layer4(x)           # (6, 2048, 7, 7)

# Channel reduction
reduced = conv1x1(features)           # (6, 256, 7, 7)

# Reshape back to separate batch and time
feature_maps = reduced.view(B, T, 256, 7, 7)  # (1, 6, 256, 7, 7)
```

### What the 7×7 Grid Represents
```
The 224×224 image is reduced to a 7×7 spatial grid.
Each cell covers approximately 32×32 pixels of the original image.

┌───┬───┬───┬───┬───┬───┬───┐
│0,0│0,1│0,2│0,3│0,4│0,5│0,6│  Each cell has 256 feature channels
├───┼───┼───┼───┼───┼───┼───┤  describing "what" is at that location
│1,0│1,1│...│   │   │   │   │
├───┼───┼───┼───┼───┼───┼───┤
│...│   │   │   │   │   │   │
├───┼───┼───┼───┼───┼───┼───┤
│6,0│   │   │   │   │   │6,6│
└───┴───┴───┴───┴───┴───┴───┘
```

---

## Stage 3: Feature-Channel Attention

The attention module computes channel-wise gates based on the task vector.

### Architecture
```python
FeatureChannelAttention:
  gate_network: Linear(3 → 256) → ReLU → Dropout 
                → Linear(256 → 256) → ReLU → Dropout
                → Linear(256 → 256)
  task_bias: Parameter(3, 256)  # Task-specific bias
  
  Output: gates = sigmoid(gate_network(task) + task_bias[task])
```

### Tensor Flow
```python
# Input
features = cnn_features      # (1, 6, 256) - all timesteps
task_vector = [[1, 0, 0]]    # (1, 3) - location task

# Step 1: Compute gate logits from task
gate_logits = gate_network(task_vector)  # (1, 256)

# Step 2: Add task-specific bias
# task_vector @ task_bias selects the bias row for active task
task_bias_selected = task_vector @ task_bias  # (1, 256)
gate_logits = gate_logits + task_bias_selected  # (1, 256)

# Step 3: Apply sigmoid to get gates in [0, 1]
gates = sigmoid(gate_logits)  # (1, 256)

# Step 4: Expand gates for all timesteps
gates_expanded = gates.unsqueeze(1).expand(1, 6, 256)  # (1, 6, 256)

# Step 5: Apply gates to features (element-wise multiplication)
gated_features = features * gates_expanded  # (1, 6, 256)
```

### What the Gates Do
```
For Location Task [1, 0, 0]:
  - Channels encoding spatial position: gates ≈ 0.8-1.0 (KEEP)
  - Channels encoding object identity: gates ≈ 0.1-0.3 (SUPPRESS)
  - Channels encoding category: gates ≈ 0.1-0.3 (SUPPRESS)

For Identity Task [0, 1, 0]:
  - Channels encoding spatial position: gates ≈ 0.1-0.3 (SUPPRESS)
  - Channels encoding object identity: gates ≈ 0.8-1.0 (KEEP)
  - Channels encoding category: gates ≈ 0.3-0.5 (PARTIAL)

For Category Task [0, 0, 1]:
  - Channels encoding spatial position: gates ≈ 0.1-0.3 (SUPPRESS)
  - Channels encoding object identity: gates ≈ 0.3-0.5 (PARTIAL)
  - Channels encoding category: gates ≈ 0.8-1.0 (KEEP)
```

### Why This Helps
1. **Reduces interference**: Task-irrelevant features don't clutter RNN memory
2. **Cleaner comparisons**: When comparing N-back, only relevant features matter
3. **Better generalization**: Model learns task-specific feature selection

---

## Stage 4: Cognitive Module (RNN)

The RNN processes the attended features over time, maintaining working memory.

### Input Preparation
```python
# Concatenate attended features with task vector
task_expanded = task_vector.unsqueeze(1).expand(1, 6, 3)  # (1, 6, 3)
rnn_input = concat([attended_seq, task_expanded], dim=-1)  # (1, 6, 259)
```

### GRU Architecture
```python
GRU(input_size=259, hidden_size=256, num_layers=1)

# Internal GRU equations (for reference):
# z_t = σ(W_z · [h_{t-1}, x_t])     # Update gate
# r_t = σ(W_r · [h_{t-1}, x_t])     # Reset gate  
# h̃_t = tanh(W · [r_t * h_{t-1}, x_t])  # Candidate
# h_t = (1 - z_t) * h_{t-1} + z_t * h̃_t  # New hidden state
```

### Tensor Flow
```python
# Initial hidden state
h_0 = zeros(1, 1, 256)  # (num_layers, B, hidden_size)

# Process sequence
outputs, h_final = gru(rnn_input, h_0)

# outputs.shape = (1, 6, 256)  - hidden state at each timestep
# h_final.shape = (1, 1, 256)  - final hidden state
```

### Hidden State Evolution
```
t=0: h_0 → GRU(x_0) → h_1  [Encodes airplane at loc=0]
t=1: h_1 → GRU(x_1) → h_2  [Encodes chair at loc=1, remembers t=0]
t=2: h_2 → GRU(x_2) → h_3  [Encodes car at loc=2, can compare with t=0]
t=3: h_3 → GRU(x_3) → h_4  [Encodes airplane at loc=0, can compare with t=1]
t=4: h_4 → GRU(x_4) → h_5  [Encodes chair at loc=3, can compare with t=2]
t=5: h_5 → GRU(x_5) → h_6  [Encodes car at loc=2, can compare with t=3]
```

---

## Stage 5: Classifier

A linear layer maps hidden states to response logits.

### Architecture
```python
classifier = Linear(256, 3)

# 3 output classes:
# Index 0: no_action  (t < N, no comparison possible)
# Index 1: non_match  (current ≠ N-back on task feature)
# Index 2: match      (current = N-back on task feature)
```

### Tensor Flow
```python
logits = classifier(outputs)  # (1, 6, 3)

# Example output (before softmax):
logits = [
    [[ 2.1, -1.0, -0.5],   # t=0: high no_action
     [ 1.8, -0.8, -0.3],   # t=1: high no_action
     [-1.2,  1.5, -0.8],   # t=2: high non_match
     [-1.0,  1.3, -0.6],   # t=3: high non_match
     [-0.9,  1.4, -0.7],   # t=4: high non_match
     [-1.1,  1.6, -0.5]]   # t=5: high non_match
]
```

### Predictions
```python
probs = softmax(logits, dim=-1)  # (1, 6, 3)
preds = argmax(logits, dim=-1)   # (1, 6)

# preds = [[0, 0, 1, 1, 1, 1]]
# Meaning: [no_action, no_action, non_match, non_match, non_match, non_match]
```

---

## Complete Forward Pass Summary

```python
def forward(images, task_vector):
    """
    Args:
        images: (B, T, 3, 224, 224)
        task_vector: (B, 3)
    
    Returns:
        logits: (B, T, 3)
        hidden_seq: (B, T, 256)
        attention_weights: (B, T, 1, 7, 7)
    """
    B, T = images.shape[:2]
    
    # Stage 1: Perceptual (CNN)
    x = images.reshape(B*T, 3, 224, 224)      # (B*T, 3, 224, 224)
    _, feat_maps = perceptual(x, return_feature_map=True)  # (B*T, 256, 7, 7)
    feat_maps = feat_maps.view(B, T, 256, 7, 7)  # (B, T, 256, 7, 7)
    
    # Stage 2: Task-Guided Attention
    attended_seq = []
    for t in range(T):
        context_t, _ = attention(feat_maps[:, t], task_vector)  # (B, 256)
        attended_seq.append(context_t)
    attended_seq = torch.stack(attended_seq, dim=1)  # (B, T, 256)
    
    # Stage 3: Prepare RNN input
    task_rep = task_vector.unsqueeze(1).expand(B, T, 3)  # (B, T, 3)
    rnn_input = torch.cat([attended_seq, task_rep], dim=-1)  # (B, T, 259)
    
    # Stage 4: Cognitive (RNN)
    outputs, final_state, hidden_seq = cognitive(rnn_input)  # (B, T, 256)
    
    # Stage 5: Classifier
    logits = classifier(outputs)  # (B, T, 3)
    
    return logits, hidden_seq, attention_weights
```

---

## Dimension Summary Table

| Stage | Tensor | Shape | Description |
|-------|--------|-------|-------------|
| Input | images | (B, T, 3, 224, 224) | RGB images |
| Input | task_vector | (B, 3) | One-hot task |
| Perceptual | resnet_out | (B*T, 2048, 7, 7) | Raw CNN features |
| Perceptual | reduced | (B*T, 256, 7, 7) | After 1×1 conv |
| Perceptual | pooled | (B*T, 256) | After GAP |
| Perceptual | cnn_features | (B, T, 256) | Reshaped |
| Attention | gate_logits | (B, 256) | Raw gate values |
| Attention | gates | (B, 256) | Sigmoid gates [0,1] |
| Attention | gated_features | (B, T, 256) | Filtered features |
| Cognitive | rnn_input | (B, T, 259) | Features + task |
| Cognitive | hidden_seq | (B, T, 256) | RNN hidden states |
| Classifier | logits | (B, T, 3) | Response scores |
| Output | predictions | (B, T) | Class indices |

---

## Comparison: Baseline vs Feature-Channel Attention Model

| Aspect | Baseline Model | Attention Model |
|--------|---------------|-----------------|
| CNN Output | Features (B, T, 256) | Features (B, T, 256) |
| Task Usage | Concatenated with features | **Gates feature channels** |
| Feature Filtering | None - all features pass | Task-irrelevant suppressed |
| RNN Input | All CNN features + task | **Gated** features + task |
| Task-Irrelevant Info | Preserved in RNN | **Filtered before RNN** |
| Parameters | ~1.7M trainable | ~1.9M trainable |

---

## Expected Behavior After Training

### Baseline Model (from paper)
- Task-relevant features: Decodable at >95%
- Task-irrelevant features: **Also decodable at >85%** (not filtered)

### Attention Model (hypothesis)
- Task-relevant features: Decodable at >95%
- Task-irrelevant features: **Lower decodability** (filtered by gates)

### How to Verify
Run the decoding analysis on both models:
```bash
python -m src.analysis.comprehensive_analysis \
  --analysis 2 \
  --hidden_root experiments/<model>/hidden_states \
  --output_dir analysis_results/<model>
```

Compare the off-diagonal values in the task-relevance heatmap (Figure 2b style).

---

## Key Insight: What the Attention Does

The Feature-Channel Attention computes **channel-wise** gates:
- It learns which **feature channels** are important for each task
- For **location tasks**: Keeps spatial encoding channels, suppresses object identity channels
- For **identity tasks**: Keeps object-specific channels, suppresses location channels
- For **category tasks**: Keeps semantic channels, suppresses fine-grained identity channels

**This directly addresses the paper's finding**: The baseline model maintains task-irrelevant information because nothing filters it out. The attention model explicitly gates channels based on task relevance.
