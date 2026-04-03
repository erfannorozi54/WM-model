---
theme: academic
title: Working Memory in RNNs
info: |
  ## Geometry of Naturalistic Object Representations
  Based on paper arXiv:2411.02685
coverAuthor: Erfan Norozi
coverDate: "February 2026"
class: text-center
highlighter: shiki
drawings:
  persist: false
transition: slide-left
mdc: true
themeConfig:
  paginationX: r
  paginationY: b
---

# Geometry of Naturalistic Object Representations in RNN Models of Working Memory

<div class="pt-4 text-lg opacity-80">
Lei, Ito & Bashivan — NeurIPS 2024
</div>

<div class="pt-2 text-sm opacity-60">
Implementation & Extension: Task-Guided Attention Models
</div>

<div class="abs-br m-6 flex gap-2">
  <a href="https://arxiv.org/abs/2411.02685" target="_blank" class="text-xl slidev-icon-btn opacity-50 !border-none !hover:text-white">
    📄
  </a>
</div>

---
layout: two-cols-header
transition: fade-out
---

# The Problem

::left::

<v-clicks>

- **Traditional WM Research**: Uses simple categorical inputs (one-hot vectors, colored dots)

- **The Gap**: How do networks handle *naturalistic*, high-dimensional stimuli?

- **Real World**: Objects have multiple features (location, identity, category, viewpoint)

- **Key Question**: How is this information encoded, maintained, and retrieved?

</v-clicks>

::right::

<div class="ml-6 mt-4">

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
transition: fade-out
---

# Research Goals

<div class="grid grid-cols-2 gap-8">

<div>

### 📄 Paper Goals

<v-clicks>

1. **Task Selection**: How do RNNs select task-relevant properties from naturalistic objects?

2. **Memory Maintenance**: What strategies maintain information against distractors?

3. **Architecture Comparison**: How do vanilla RNN vs GRU/LSTM differ?

4. **Memory Mechanism**: Slot-based vs chronological organization?

</v-clicks>

</div>

<div>

### 🔬 Our Extension

<v-clicks>

5. **Task-Guided Attention**: Can explicit attention improve feature selection?

6. **Generalization**: Does attention help with novel objects?

7. **Multi-Task Learning**: How does attention affect MTMF scenarios?

</v-clicks>

</div>

</div>

---
layout: two-cols-header
transition: slide-up
---

# N-back Task Design

::left::

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

::right::

<div class="ml-4">

```
Example: 2-back Category Task

Trial 1: 🪑 chair    → no_action
Trial 2: 🚗 car      → no_action  
Trial 3: 🪑 chair    → MATCH! (= T1)
Trial 4: ✈️ plane    → non_match
Trial 5: 🚗 car      → non_match
Trial 6: ✈️ plane    → MATCH! (= T4)
```

<div class="mt-4 p-3 bg-blue-500/10 rounded-lg">

**Responses**: `no_action` | `non_match` | `match`

</div>

</div>

---
transition: fade-out
---

# Model Architecture

<div class="flex justify-center">

```mermaid {scale: 0.75}
graph TD
    A["🖼️ Images (B,T,3,224,224)"] --> B["ResNet50 (frozen)"]
    B --> C["1×1 Conv → GAP"]
    C --> D["Visual Embedding (B,T,H)"]
    E["Task Vector (one-hot)"] --> F["Concat"]
    D --> F
    F --> G["RNN / GRU / LSTM"]
    G --> H["Linear Classifier"]
    H --> I["no_action | non_match | match"]
    
    style A fill:#4a9eff,color:#fff
    style B fill:#ff6b6b,color:#fff
    style G fill:#51cf66,color:#fff
    style I fill:#ffd43b,color:#333
```

</div>

---
transition: slide-up
---

# Training Scenarios

| Scenario | Description | N-values | Tasks | Complexity |
|----------|-------------|----------|-------|------------|
| **STSF** | Single-Task Single-Feature | [2] | 1 (category) | ⭐ |
| **STMF** | Single-Task Multi-Feature | [2] | 3 (L, I, C) | ⭐⭐ |
| **MTMF** | Multi-Task Multi-Feature | [1,2,3] | 9 (all) | ⭐⭐⭐ |

<v-click>

<div class="mt-6 p-4 bg-green-500/10 rounded-lg">

### Validation Splits

- **Novel Angle**: Same objects, new viewing angle → tests view-invariance
- **Novel Identity**: New object instances → tests generalization

</div>

</v-click>

---
layout: section
transition: fade
---

# Paper's 5 Analyses

Understanding Working Memory Representations

---
layout: quote
transition: fade-out
---

# Analysis 1: Behavioral Performance

<div class="text-base">

> "Novel identity generalization is substantially weaker than novel angle — models learn view-invariant but not identity-invariant representations"

</div>

<v-clicks>

<div class="text-sm">

- Track accuracy on training, novel-angle, and novel-identity sets
- Expected: Training ~95%, Novel Angle ~90%, Novel Identity ~70%
- **Key Finding**: Generalization gap reveals what the model truly learns

</div>

</v-clicks>

---
layout: two-cols-header
transition: fade-out
---

# Analysis 1: Baseline MTMF

::left::

<img src="/results/wm_mtmf_20260105_182040/analysis1_training_curves.png" class="h-72 rounded shadow-lg" />

::right::

<img src="/results/wm_mtmf_20260105_182040/analysis1_generalization_comparison.png" class="h-72 rounded shadow-lg" />

<div class="mt-3 p-3 bg-orange-500/10 rounded-lg text-xs">

**Training**: 88.6% &nbsp;|&nbsp; **Novel Angle**: 85.9% &nbsp;|&nbsp; **Novel Identity**: 70.7%

✅ Pattern confirmed: Novel Identity < Novel Angle (15% gap)

</div>

---
layout: two-cols-header
transition: fade-out
---

# Analysis 1: Dual Attention MTMF

::left::

<img src="/results/wm_dual_attention_mtmf_20260107_095814/analysis1_training_curves.png" class="h-60 rounded shadow-lg" />

::right::

<img src="/results/wm_dual_attention_mtmf_20260107_095814/analysis1_generalization_comparison.png" class="h-60 rounded shadow-lg" />

<div class="mt-4 p-3 bg-green-500/10 rounded-lg text-xs">

**Training**: 99.3% &nbsp;|&nbsp; **Novel Angle**: 94.6% &nbsp;|&nbsp; **Novel Identity**: 81.2%

✅ Attention dramatically improves all metrics (+10% across the board)

</div>

---
transition: fade-out
---

# Analysis 2: Encoding Properties

<div class="grid grid-cols-2 gap-8">

<div class="p-4 bg-blue-500/10 rounded-lg">

### 2A: Task-Relevance Decoding <span class="text-sm opacity-60">(Figure 2b)</span>

**Question**: Does the network only encode task-relevant information, or does it preserve everything?

**Method**: Within each task context, train a linear decoder to predict each property (location, identity, category) from the hidden states. This produces a 3×3 matrix where rows = task context and columns = decoded property.

**Key distinction**:
- **Diagonal** (e.g., decode location from location task) → task-relevant → should be high (>85%)
- **Off-diagonal** (e.g., decode identity from location task) → task-irrelevant

**Finding**:
- **STSF**: Only diagonal is high — irrelevant info is discarded
- **MTMF**: All cells >85% — full object representation preserved across tasks

</div>

<div class="p-4 bg-purple-500/10 rounded-lg">

### 2B: Cross-Task Generalization <span class="text-sm opacity-60">(Figure 2a)</span>

**Question**: Are the neural representations for a property (e.g., identity) the same across different tasks, or does the network use separate subspaces?

**Method**: Train a decoder on Task A (e.g., identity from location task), then test it on Task B (identity from identity task). This produces a 3×3 matrix per property where rows = train task and columns = test task.

**Key distinction**:
- **Diagonal** (train on A, test on A) → baseline decoding accuracy
- **Off-diagonal** (train on A, test on B) → do representations generalize?

**Finding**:
- **Vanilla RNN**: High off-diagonal → shared/overlapping representations
- **GRU/LSTM**: Low off-diagonal (8-35%) → task-specific subspaces

</div>

</div>

<div class="mt-6 p-3 bg-yellow-500/10 rounded-lg text-sm">

**The difference**: 2A asks *"what information is present within one task?"* while 2B asks *"do representations transfer between tasks?"* — they probe different aspects of the encoding geometry.

</div>

---

# Analysis 2A: Task-Relevance Results

<div class="flex gap-6 justify-center">

<div class="text-center">
<p class="text-sm font-bold mb-2">Baseline MTMF</p>
<img src="/results/wm_mtmf_20260105_182040/analysis2a_task_relevance.png" class="h-72 rounded shadow-lg" />
</div>

<div class="text-center">
<p class="text-sm font-bold mb-2">Dual Attention MTMF</p>
<img src="/results/wm_dual_attention_mtmf_20260107_095814/analysis2a_task_relevance.png" class="h-72 rounded shadow-lg" />
</div>

</div>

<div class="mt-4 p-4 bg-blue-500/10 rounded-lg text-sm">

**What this shows**: Each cell = decoding accuracy for one property (columns) within one task context (rows).

- **Baseline MTMF**: All cells >87% — the model preserves the full object representation regardless of task
- **Dual Attention MTMF**: All cells >90% — attention further strengthens this mixed representation
- ✅ Both models encode *all* properties, not just task-relevant ones (unlike STSF which only encodes the diagonal)

</div>

---

# Analysis 2B: Cross-Task Generalization

<div class="flex gap-6 justify-center">

<div>
<img src="/results/wm_mtmf_20260105_182040/analysis2b_cross_task_location.png" class="h-72 rounded shadow-lg" />
</div>

<div>
<img src="/results/wm_mtmf_20260105_182040/analysis2b_cross_task_identity.png" class="h-72 rounded shadow-lg" />
</div>

</div>

<div class="mt-4 p-4 bg-red-500/10 rounded-lg text-sm">

**What this shows**: Each cell = decoder trained on one task (rows), tested on another task (columns).

- **Diagonal (same task)**: 90-100% accuracy → baseline decoding works well
- **Off-diagonal (cross-task)**: 8-35% accuracy → decoder fails when switching tasks
- ✅ Confirms GRU uses **task-specific subspaces** — "location" in the location task vs "location" in the identity task live in different subspaces
- This matches the paper's finding for gated RNNs (GRU/LSTM): they segregate representations by task context

</div>

---
transition: fade-out
---

# Analysis 3: Orthogonalization

<div class="grid grid-cols-2 gap-6">

<div>

### Method

1. Train one-vs-rest SVM for each feature value
2. Extract hyperplane normal vectors **W**
3. Compute orthogonalization index:

$$O = E[\text{triu}(\tilde{W})] \quad \text{where} \quad \tilde{W}_{ij} = 1 - |\cos(W_i, W_j)|$$

<div class="mt-4 text-sm">

- **O = 1**: Perfectly orthogonal (excellent separation)
- **O = 0**: Completely overlapping
- **Points below diagonal** = RNN de-orthogonalizes

</div>

</div>

<div class="flex flex-col items-center gap-2">

<img src="/results/wm_mtmf_20260105_182040/analysis3_orthogonalization.png" class="h-48 rounded shadow-lg" />

<div class="text-sm opacity-80">Location & Category below diagonal ✅</div>

</div>

</div>

---
layout: two-cols-header
transition: fade-out
---

# Analysis 4: WM Dynamics — H1 Test

<div class="mb-2 p-2 bg-yellow-500/10 rounded-lg text-sm">

**Hypothesis H1 (Slot-Based)**: If memory uses fixed slots, a decoder trained at t=0 should work at t=1,2,3...

</div>

::left::

### Baseline MTMF
<img src="/results/wm_mtmf_20260105_182040/analysis4a_cross_time_decoding.png" class="h-48 rounded shadow-lg" />

::right::

### Dual Attention MTMF
<img src="/results/wm_dual_attention_mtmf_20260107_095814/analysis4a_cross_time_decoding.png" class="h-48 rounded shadow-lg" />

<div class="mt-2 p-3 bg-red-500/10 rounded-lg">

**Result**: Accuracy drops 100% → ~5% immediately → **H1 DISPROVED** — Memory is NOT stored in fixed slots!

</div>

---
layout: fact
transition: slide-up
---

# All Paper Findings Replicated ✅

<div class="text-lg mt-4">

| Analysis | Paper Finding | Our Result |
|----------|---------------|------------|
| **1. Behavioral** | Novel identity < Novel angle | 70.7% vs 85.9% ✅ |
| **2A. Task-Relevance** | MTMF preserves all features | All >87% ✅ |
| **2B. Cross-Task** | GRU task-specific | Diag 90-100%, Off 8-35% ✅ |
| **3. Orthogonalization** | RNN de-orthogonalizes | Below diagonal ✅ |
| **4. H1 Test** | Slot-based disproved | 100%→5% drop ✅ |

</div>

---
layout: section
transition: fade
---

# Our Innovation

Task-Guided Attention Models

---
transition: fade-out
---

# Task-Guided Attention

<div class="grid grid-cols-2 gap-8">

<div>

### Standard Model
```
CNN → RNN → Classifier
```

### Our Model
```
CNN → Task-Guided Attention → RNN → Classifier
```

<div class="mt-4">

### Attention Mechanism
- **Query**: Task embedding (+ hidden state for dual)
- **Key/Value**: Visual features from CNN
- **Output**: Task-modulated visual representation

</div>

</div>

<div>

### Performance Gains

| Metric | Baseline | + Attention |
|--------|----------|-------------|
| Train Acc | 88.6% | **99.3%** |
| Novel Angle | 85.9% | **94.6%** |
| Novel Identity | 70.7% | **81.2%** |

<div class="mt-4 p-3 bg-green-500/10 rounded-lg text-center text-xl font-bold">
+10% improvement across all metrics
</div>

</div>

</div>

---
transition: fade-out
---

# All Models Comparison

<div class="flex justify-center">

| Model | Train | Novel Angle | Novel Identity |
|-------|------:|------------:|---------------:|
| **STSF** (baseline) | 99.99% | 99.93% | 93.60% |
| **STMF** (baseline) | 88.44% | 86.31% | 72.54% |
| **MTMF** (baseline) | 88.64% | 85.86% | 70.67% |
| **STMF + Attention** | 99.15% | 93.90% | 81.03% |
| **STMF + Dual Attn** | 99.80% | 94.67% | 81.33% |
| **MTMF + Attention** | 99.33% | 92.49% | 79.69% |
| **MTMF + Dual Attn** | 99.29% | 94.64% | 81.18% |

</div>

<v-click>

<div class="mt-4 grid grid-cols-3 gap-4 text-sm">
<div class="p-3 bg-blue-500/10 rounded-lg text-center">

**Insight 1**: Attention helps most for multi-feature tasks (STMF, MTMF)

</div>
<div class="p-3 bg-purple-500/10 rounded-lg text-center">

**Insight 2**: Dual attention slightly better for complex MTMF

</div>
<div class="p-3 bg-green-500/10 rounded-lg text-center">

**Insight 3**: STSF already near-perfect (no room for improvement)

</div>
</div>

</v-click>

---
transition: fade-out
---

# Conclusions

<div class="grid grid-cols-2 gap-8">

<div>

### 📄 Paper Contributions Validated

<v-clicks>

1. ✅ Multi-task RNNs preserve full object representations
2. ✅ GRU uses task-specific subspaces
3. ✅ RNNs de-orthogonalize compared to perceptual space
4. ✅ Slot-based memory hypothesis disproved

</v-clicks>

</div>

<div>

### 🔬 Our Contributions

<v-clicks>

5. ✅ Task-guided attention improves multi-feature performance by ~10%
6. ✅ Attention doesn't fundamentally change representational geometry
7. ✅ Dual attention provides marginal gains for complex MTMF

</v-clicks>

</div>

</div>

<v-click>

<div class="mt-6 p-4 bg-blue-500/10 rounded-lg text-center">

### Implication
Explicit attention mechanism complements RNN memory dynamics — supports **"resource-based"** over **"slot-based"** WM models

</div>

</v-click>

---
layout: center
class: text-center
transition: fade
---

# Thank You

<div class="pt-8 text-lg opacity-80">

**Paper**: arXiv:2411.02685

**Code**: github.com/erfannorozi54/WM-model

</div>

<div class="pt-8">
  <span class="opacity-50 text-sm">
    Built with Slidev + Academic Theme
  </span>
</div>
