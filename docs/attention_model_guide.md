# Attention-Based Working Memory Model: Architecture Guide

This document provides a comprehensive explanation of the attention-based working memory model architecture.

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

---

## Experimental Results

| Model | Train Masked Acc | Val Novel Angle | Val Novel Identity |
|-------|------------------|-----------------|-------------------|
| STMF (Baseline) | 82.7% | 80.0% | 57.6% |
| MTMF (Baseline) | 83.0% | 78.8% | 57.0% |
| **STMF + Attention** | **98.7%** | **92.4%** | **72.6%** |
| **STMF + Dual Attention** | **99.7%** | **91.9%** | **72.4%** |
| **MTMF + Attention** | **99.0%** | **90.1%** | **70.6%** |
| **MTMF + Dual Attention** | **98.9%** | **92.3%** | **73.5%** |

### Key Findings

1. **Dramatic improvement on multi-feature tasks**: Attention improves novel angle generalization by 12-13 percentage points.

2. **Better identity generalization**: Novel identity accuracy improves by 15-16 percentage points.

3. **Dual attention benefits MTMF**: The adaptive gating of dual attention provides marginal gains when handling multiple N-back levels.

4. **Training dynamics differ**: Attention models show slower initial learning (flat for ~10 epochs) but achieve much higher final performance.

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
