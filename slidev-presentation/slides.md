---
theme: default
title: Working Memory in RNNs
info: |
  ## Geometry of Naturalistic Object Representations
  Based on paper arXiv:2411.02685
class: text-center
highlighter: shiki
drawings:
  persist: false
transition: slide-left
mdc: true
---

# Geometry of Naturalistic Object Representations in RNN Models of Working Memory

**Based on:** Lei, Ito & Bashivan (NeurIPS 2024)

**Implementation & Extension:** Task-Guided Attention Models

<div class="pt-12">
  <span class="px-2 py-1 rounded cursor-pointer" hover="bg-white bg-opacity-10">
    Press Space for next slide →
  </span>
</div>

---
layout: two-cols
---

# The Problem

<v-clicks>

- **Traditional WM Research**: Uses simple categorical inputs (one-hot vectors, colored dots)

- **The Gap**: How do networks handle *naturalistic*, high-dimensional stimuli?

- **Real World**: Objects have multiple features (location, identity, category, viewpoint)

- **Key Question**: How is this information encoded, maintained, and retrieved?

</v-clicks>

::right::

<div class="ml-4 mt-8">

```
Traditional Input:
[0, 1, 0, 0]  ← One-hot category

Naturalistic Input:
Image → CNN → 2048-dim embedding
  ↓
Location: quadrant 1-4
Identity: object instance
Category: chair/car/plane/table
Viewpoint: 4 angles
```

</div>

---

# Research Goals

<div class="grid grid-cols-2 gap-4">

<div>

### Paper Goals

<v-clicks>

1. **Task Selection**: How do RNNs select task-relevant properties from naturalistic objects?

2. **Memory Maintenance**: What strategies maintain information against distractors?

3. **Architecture Comparison**: How do vanilla RNN vs GRU/LSTM differ?

4. **Memory Mechanism**: Slot-based vs chronological organization?

</v-clicks>

</div>

<div>

### Our Extension

<v-clicks>

5. **Task-Guided Attention**: Can explicit attention improve feature selection?

6. **Generalization**: Does attention help with novel objects?

7. **Multi-Task Learning**: How does attention affect MTMF scenarios?

</v-clicks>

</div>

</div>

---

# N-back Task Design

<div class="grid grid-cols-2 gap-8">

<div>

### Task Structure

- **N ∈ {1, 2, 3}**: Memory depth
- **Features**: Location (L), Identity (I), Category (C)
- **9 Task Variants**: 3 × 3 combinations
- **Sequence Length**: 6 trials

### Stimuli (ShapeNet)

- 4 Categories (chair, car, airplane, table)
- 5 Identities per category
- 4 Locations (quadrants)
- 4 Viewing angles

</div>

<div>

```
Example: 2-back Category Task

Trial 1: 🪑 chair    → no_action
Trial 2: 🚗 car      → no_action  
Trial 3: 🪑 chair    → MATCH! (same as T1)
Trial 4: ✈️ plane    → non_match
Trial 5: 🚗 car      → non_match
Trial 6: ✈️ plane    → MATCH! (same as T4)
```

**Responses**: `no_action` | `non_match` | `match`

</div>

</div>

---

# Model Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Two-Stage Sensory-Cognitive Model            │
├─────────────────────────────────────────────────────────────────┤
│  Images (B,T,3,224,224)                                        │
│       ↓                                                         │
│  ┌─────────────┐                                               │
│  │  ResNet50   │  ← Frozen ImageNet weights                    │
│  │  (CNN)      │  → 2048-dim features                          │
│  └─────────────┘                                               │
│       ↓                                                         │
│  1×1 Conv → GAP → Visual Embedding (B,T,H)                     │
│       ↓                                                         │
│  Concat with Task Vector (B,T,H+3)                             │
│       ↓                                                         │
│  ┌─────────────┐                                               │
│  │ RNN/GRU/LSTM│  ← Trainable                                  │
│  └─────────────┘                                               │
│       ↓                                                         │
│  Linear Classifier → Logits (B,T,3)                            │
│       ↓                                                         │
│  Predictions: no_action | non_match | match                    │
└─────────────────────────────────────────────────────────────────┘
```

---

# Training Scenarios

| Scenario | Description | N-values | Tasks | Complexity |
|----------|-------------|----------|-------|------------|
| **STSF** | Single-Task Single-Feature | [2] | 1 (e.g., category) | Simplest |
| **STMF** | Single-Task Multi-Feature | [2] | 3 (L, I, C) | Medium |
| **MTMF** | Multi-Task Multi-Feature | [1,2,3] | 9 (all combinations) | Hardest |

<v-click>

### Validation Splits

- **Novel Angle**: Same objects, new viewing angle (tests view-invariance)
- **Novel Identity**: New object instances (tests generalization)

</v-click>

---
layout: section
---

# Paper's 5 Analyses

Understanding Working Memory Representations

---

# Analysis 1: Behavioral Performance

### Goal
Validate model achieves expected generalization patterns

### Method
- Track accuracy on training, novel-angle, and novel-identity sets
- Compare generalization gaps

### Expected Pattern (Figure A1c)
```
Training Accuracy:        ~95%
Novel Angle Accuracy:     ~90% (slight drop - view invariance works)
Novel Identity Accuracy:  ~70% (substantial drop - generalization gap)
```

<v-click>

### Key Finding
**Novel identity generalization is substantially weaker** than novel angle - models learn view-invariant but not identity-invariant representations

</v-click>

---

# Analysis 1: Our Results - Baseline MTMF

<div class="grid grid-cols-2 gap-4">

<div>
<img src="/results/wm_mtmf_20260105_182040/analysis1_training_curves.png" class="h-60" />
</div>

<div>
<img src="/results/wm_mtmf_20260105_182040/analysis1_generalization_comparison.png" class="h-60" />
</div>

</div>

**Observations:**
- Training: 88.6% | Novel Angle: 85.9% | Novel Identity: 70.7%
- ✅ **Pattern confirmed**: Novel Identity < Novel Angle (15% gap)
- Model plateaus at ~88% training accuracy

---

# Analysis 1: Our Results - Dual Attention MTMF

<div class="grid grid-cols-2 gap-4">

<div>
<img src="/results/wm_dual_attention_mtmf_20260107_095814/analysis1_training_curves.png" class="h-60" />
</div>

<div>
<img src="/results/wm_dual_attention_mtmf_20260107_095814/analysis1_generalization_comparison.png" class="h-60" />
</div>

</div>

**Observations:**
- Training: 99.3% | Novel Angle: 94.6% | Novel Identity: 81.2%
- ✅ **Attention dramatically improves** all metrics (+10% across the board)
- Sharp improvement at epoch ~12 when attention "kicks in"

---

# Analysis 2: Encoding Properties

<div class="grid grid-cols-2 gap-4">

<div>

### 2A: Task-Relevance (Figure 2b)

**Question**: Do RNNs preserve task-irrelevant info?

**Method**: Decode each property from each task context

**Finding**:
- STSF: Only task-relevant features preserved
- MTMF: **All features preserved** (>85%)

</div>

<div>

### 2B: Cross-Task Generalization (Figure 2a)

**Question**: Are encodings shared across tasks?

**Method**: Train decoder on Task A, test on Task B

**Finding**:
- **Vanilla RNN**: High cross-task generalization
- **GRU/LSTM**: Low cross-task generalization (task-specific)

</div>

</div>

---

# Analysis 2A: Task-Relevance Results

<div class="grid grid-cols-2 gap-4">

<div>

### Baseline MTMF
<img src="/results/wm_mtmf_20260105_182040/analysis2a_task_relevance.png" class="h-55" />

All cells >87% - **full object representation preserved**

</div>

<div>

### Dual Attention MTMF
<img src="/results/wm_dual_attention_mtmf_20260107_095814/analysis2a_task_relevance.png" class="h-55" />

All cells >90% - attention maintains full representation

</div>

</div>

**Key Finding**: Both models preserve task-relevant AND irrelevant features (>85%)

---

# Analysis 2B: Cross-Task Generalization

<div class="grid grid-cols-2 gap-4">

<div>
<img src="/results/wm_mtmf_20260105_182040/analysis2b_cross_task_location.png" class="h-55" />
</div>

<div>
<img src="/results/wm_mtmf_20260105_182040/analysis2b_cross_task_identity.png" class="h-55" />
</div>

</div>

**Results**: 
- **Diagonal (same task)**: 90-100% accuracy
- **Off-diagonal (cross-task)**: 8-35% accuracy
- ✅ Confirms GRU uses **task-specific subspaces** (paper's finding for gated RNNs)

---

# Analysis 3: Orthogonalization

### Goal
Compare representational geometry between CNN (perceptual) and RNN (encoding) spaces

### Method
1. Train one-vs-rest SVM for each feature value
2. Extract hyperplane normal vectors W
3. Compute orthogonalization index:

$$O = E[\text{triu}(\tilde{W})] \quad \text{where} \quad \tilde{W}_{ij} = 1 - |\cos(W_i, W_j)|$$

### Expected Pattern (Figure 3b)
- O = 1: Perfectly orthogonal (excellent separation)
- O = 0: Completely overlapping
- **Points below diagonal** = RNN de-orthogonalizes

---

# Analysis 3: Orthogonalization Results

<div class="grid grid-cols-2 gap-4">

<div>

### Baseline MTMF
<img src="/results/wm_mtmf_20260105_182040/analysis3_orthogonalization.png" class="h-55" />

Location & Category below diagonal ✅

</div>

<div>

### Dual Attention MTMF
<img src="/results/wm_dual_attention_mtmf_20260107_095814/analysis3_orthogonalization.png" class="h-55" />

Similar pattern - attention doesn't change geometry

</div>

</div>

**Finding**: RNN de-orthogonalizes for Location & Category (more efficient representation)

---

# Analysis 4: WM Dynamics - H1 Test

### Hypothesis H1: Slot-Based Memory
If true, decoder trained at t=0 should work at t=1,2,3...

<div class="grid grid-cols-2 gap-4">

<div>

### Baseline MTMF
<img src="/results/wm_mtmf_20260105_182040/analysis4a_cross_time_decoding.png" class="h-50" />

</div>

<div>

### Dual Attention MTMF
<img src="/results/wm_dual_attention_mtmf_20260107_095814/analysis4a_cross_time_decoding.png" class="h-50" />

</div>

</div>

**Result**: Accuracy drops from 100% → ~5% immediately → **H1 DISPROVED**
Memory is NOT stored in fixed slots!

---

# Summary: All Analysis Results

| Analysis | Paper Finding | Our Result | Status |
|----------|---------------|------------|--------|
| **1. Behavioral** | Novel identity < Novel angle | ✅ 70.7% vs 85.9% (baseline) | ✅ |
| **2A. Task-Relevance** | MTMF preserves all features | ✅ All >87% accuracy | ✅ |
| **2B. Cross-Task** | GRU task-specific | ✅ Diagonal 90-100%, Off-diag 8-35% | ✅ |
| **3. Orthogonalization** | RNN de-orthogonalizes | ✅ Location/Category below diagonal | ✅ |
| **4. H1 Test** | Slot-based disproved | ✅ 100%→5% drop over time | ✅ |

**All paper findings replicated!**

---

# Our Innovation: Task-Guided Attention

<div class="grid grid-cols-2 gap-4">

<div>

### Standard Model
```
CNN → RNN → Classifier
```

### Our Model
```
CNN → Task-Guided Attention → RNN → Classifier
```

### Attention Mechanism
- **Query**: Task embedding (+ hidden state for dual)
- **Key/Value**: Visual features from CNN
- **Output**: Task-modulated visual representation

</div>

<div>

### Performance Gains

| Metric | Baseline | + Attention |
|--------|----------|-------------|
| Train Acc | 88.6% | **99.3%** |
| Novel Angle | 85.9% | **94.6%** |
| Novel Identity | 70.7% | **81.2%** |

**+10% improvement across all metrics!**

</div>

</div>

---

# All Models Comparison

| Model | Train | Novel Angle | Novel Identity |
|-------|-------|-------------|----------------|
| **STSF** (baseline) | 99.99% | 99.93% | 93.60% |
| **STMF** (baseline) | 88.44% | 86.31% | 72.54% |
| **MTMF** (baseline) | 88.64% | 85.86% | 70.67% |
| **STMF + Attention** | 99.15% | 93.90% | 81.03% |
| **STMF + Dual Attn** | 99.80% | 94.67% | 81.33% |
| **MTMF + Attention** | 99.33% | 92.49% | 79.69% |
| **MTMF + Dual Attn** | 99.29% | 94.64% | 81.18% |

<v-click>

### Key Insights
- Attention helps most for multi-feature tasks (STMF, MTMF)
- Dual attention slightly better for complex MTMF
- STSF already near-perfect (no room for improvement)

</v-click>

---

# Conclusions

<v-clicks>

### Paper Contributions Validated
1. ✅ Multi-task RNNs preserve full object representations
2. ✅ GRU uses task-specific subspaces (low cross-task generalization)
3. ✅ RNNs de-orthogonalize compared to perceptual space
4. ✅ Slot-based memory hypothesis disproved

### Our Contributions
5. ✅ Task-guided attention improves multi-feature performance by ~10%
6. ✅ Attention doesn't fundamentally change representational geometry
7. ✅ Dual attention provides marginal gains for complex MTMF

### Implications
- Explicit attention mechanism complements RNN memory dynamics
- Supports "resource-based" over "slot-based" WM models

</v-clicks>

---
layout: center
class: text-center
---

# Thank You

<div class="pt-12">

**Paper**: arXiv:2411.02685

**Code**: github.com/erfannorozi54/WM-model

</div>
