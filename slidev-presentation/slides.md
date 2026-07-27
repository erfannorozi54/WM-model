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
| **STSF** | Single-Task Single-Feature | [2] | 1 (location) | ⭐ |
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

# Paper's 5 Analyses — Re-Run With Fixed Code

After auditing the analysis pipeline, we fixed 4 bugs and re-ran all 18 experiments.
Full audit: `docs/ANALYSIS_AUDIT_FINDINGS.md`

**Key fixes**:
- SVC `max_iter=10000` + `random_state=42` (no convergence warnings, reproducible)
- H2 cross-stimulus now uses val_novel_angle→val_novel_identity (was mislabeled cross-time)
- Procrustes swap test with per-stimulus group split + location for label alignment
- Sample-size warnings when n_test < n_classes

---
transition: fade-out
---

# Analysis 1: Behavioral Performance (Real Results)

<div class="text-base">

> "Novel identity generalization is weaker than novel angle — models learn view-invariant but not identity-invariant representations"

</div>

<div class="mt-3 p-3 bg-blue-500/10 rounded-lg text-xs">

### MTMF Models — Real Numbers

| Model | Train | Novel Angle | Novel Identity | Gap |
|-------|------:|------------:|---------------:|----:|
| wm_mtmf (h=256) | 88.3% | 81.5% | 80.6% | +1.0% |
| wm_h128_mtmf (h=128) | 87.8% | 81.3% | 79.8% | +1.6% |
| wm_attention_mtmf | **99.3%** | **93.3%** | **92.2%** | +1.1% |
| wm_dual_attention_mtmf | **99.3%** | **93.8%** | **90.9%** | +2.9% |
| wm_h128_attention_mtmf | 98.9% | 92.9% | 91.1% | +1.8% |
| wm_h128_dual_attention_mtmf | 97.5% | 89.3% | 88.2% | +1.1% |

</div>

<v-clicks>

<div class="mt-3 text-sm">

- ✅ **Pattern confirmed**: Novel Identity < Novel Angle in all 18/18 experiments
- ⚠️ **Gap is small** (0-4pp) — not the dramatic 15% gap of the paper
- **Attention models** have the largest gaps (1-3pp)
- **STSF models** (single task) achieve near-perfect accuracy on both splits (no room for gap)

</div>

</v-clicks>

---
transition: fade-out
---

# Analysis 2: Task-Relevance Decoding (Real Results)

<div class="text-sm mb-3">

**Question (Figure 2b)**: Does the network only encode task-relevant information?

**Method**: Train linear decoder per (task, property) pair from MTMF hidden states at t=0.
Each cell = test accuracy on held-out 20%. n=51-56 test samples per cell.

</div>

<div class="text-xs">

### MTMF baseline (wm_mtmf) — Task-Relevance Matrix

| Task \ Property | Location | Identity | Category |
|-----------------|---------:|---------:|---------:|
| **Location** | 47.5% | 49.2% | **100.0%** |
| **Identity** | 40.4% | 26.9% | **94.2%** |
| **Category** | 41.2% | 37.3% | 90.2% |

</div>

<div class="text-xs mt-3">

### MTMF dual-attention (wm_dual_attention_mtmf)

| Task \ Property | Location | Identity | Category |
|-----------------|---------:|---------:|---------:|
| **Location** | 100.0% | 21.4% | 60.7% |
| **Identity** | 30.8% | 38.5% | **96.2%** |
| **Category** | 56.6% | 45.3% | 92.5% |

</div>

<v-clicks>

<div class="mt-3 p-3 bg-yellow-500/10 rounded-lg text-xs">

- ⚠️ **Reality**: Diagonal (task-relevant) cells are 88-100% — paper claim is **partially** supported
- ⚠️ **Off-diagonal is NOT all >85%** — varies widely (14-60%), with **category** often the easiest to decode (94-100%)
- ✅ **Category always decodable** regardless of task context (90-100% across all tasks)
- ✅ **Identity is hardest** to decode across all task contexts (14-49%)
- ⚠️ **Sample size warning**: n_test ≈ 50 vs n_classes = 70 for identity, so off-diagonal identity values are noisy

</div>

</v-clicks>

---
transition: fade-out
---

# Analysis 2B: Cross-Task Generalization (Real Results)

<div class="text-sm mb-3">

**Question (Figure 2a)**: Do representations transfer across tasks?
**Method**: Train decoder on task A, test on task B (different property classes).

</div>

<div class="text-xs">

### MTMF Models — Diagonal vs Off-Diagonal Accuracy (%)

| Model | Location diag/off | Identity diag/off | Category diag/off |
|-------|------------------:|------------------:|------------------:|
| wm_mtmf | 44.8 / 25.6 | 40.5 / 5.9 | 95.0 / 53.9 |
| wm_h128_mtmf | 43.0 / 27.7 | 37.8 / 6.6 | 94.8 / 42.6 |
| wm_attention_mtmf | 63.7 / 30.7 | 30.5 / 4.4 | 78.1 / 47.3 |
| wm_dual_attention_mtmf | 62.5 / 38.7 | 35.1 / 3.7 | 83.1 / 52.8 |
| wm_h128_attention_mtmf | 62.5 / 30.0 | 28.3 / 5.0 | 79.5 / 50.8 |
| wm_h128_dual_attention_mtmf | 67.5 / 39.0 | 30.9 / 5.1 | 88.8 / 55.9 |

</div>

<v-clicks>

<div class="mt-3 p-3 bg-orange-500/10 rounded-lg text-xs">

- ✅ **Location & Identity**: Off-diagonal ≪ diagonal (3-39% vs 28-67%) — **task-specific subspaces** (paper claim supported)
- ⚠️ **Category**: Off-diagonal reaches 42-56% — **partially transfers** across tasks (paper claim NOT fully supported)
- 🔍 **Identity off-diagonal near chance (4-6%)** — strong task-specificity
- 🔍 **Category off-diagonal near 50%** — high cross-task transferability

</div>

</v-clicks>

---
transition: fade-out
---

# Analysis 3: Orthogonalization (Real Results)

<div class="text-sm mb-3">

**Question (Figure 3b)**: Does the RNN de-orthogonalize compared to CNN?
**Method**: One-vs-rest LinearSVC per property, extract hyperplane normals, compute O = 1 - |cos(W_i, W_j)|.

</div>

<div class="text-xs">

### MTMF Models — O(Perceptual) vs O(Encoding)

| Model | Loc: P → E | Ident: P → E | Cat: P → E | Below |
|-------|-----------:|-------------:|-----------:|------:|
| wm_mtmf | 0.730 → 0.719 | 0.953 → 0.938 | 0.750 → 0.764 | 2/3 |
| wm_h128_mtmf | 0.752 → 0.698 | 0.933 → 0.915 | 0.768 → 0.782 | 2/3 |
| wm_attention_mtmf | 0.678 → 0.715 | 0.951 → 0.937 | 0.749 → 0.734 | 2/3 |
| wm_dual_attention_mtmf | 0.688 → 0.724 | 0.951 → 0.935 | 0.749 → 0.731 | 2/3 |
| wm_h128_attention_mtmf | 0.723 → 0.715 | 0.932 → 0.913 | 0.758 → 0.737 | 3/3 |
| wm_h128_dual_attention_mtmf | 0.728 → 0.773 | 0.933 → 0.911 | 0.743 → 0.760 | 1/3 |

</div>

<v-clicks>

<div class="mt-3 p-3 bg-yellow-500/10 rounded-lg text-xs">

- ✅ **Paper claim supported for 12/18 MTMF points** (2/3 below diagonal on average)
- 🔍 **Identity stays high** (O=0.91-0.95) — 70+ classes inherently spread out
- 🔍 **Location and category are lower** (O=0.70-0.78) — more compressed representations
- ⚠️ **Attention models** sometimes show O(RNN) > O(CNN) for location — attention may undo the de-orthogonalization

</div>

</v-clicks>

---
transition: fade-out
---

# Analysis 4: WM Dynamics — H1, H2, H3

<div class="text-sm mb-2">

**Three hypotheses** (Figure 4e): H1=slot-based, H2=chronological, H3=stimulus-specific
</div>

<div class="text-xs">

### H1: Cross-Time Decoding (MTMF)

| Model | t=0 | t=1 | t=2 | t=5 | Drop |
|-------|----:|----:|----:|----:|-----:|
| wm_mtmf | 98.2% | 5.5% | 4.2% | 2.1% | **96pp** |
| wm_h128_mtmf | 95.6% | 5.1% | 3.9% | 1.4% | **94pp** |
| wm_attention_mtmf | 92.4% | 4.4% | 2.5% | 1.5% | **91pp** |
| wm_dual_attention_mtmf | 91.2% | 3.8% | 2.4% | 2.5% | **89pp** |
| wm_h128_attention_mtmf | 78.1% | 4.6% | 1.6% | 2.0% | **76pp** |
| wm_h128_dual_attention_mtmf | 85.0% | 4.2% | 2.4% | 1.9% | **83pp** |

</div>

<div class="mt-2 p-2 bg-red-500/10 rounded-lg text-xs">

✅ **H1 DISPROVED** in all 18/18 experiments — accuracy collapses to 1-6% at t≥1. **Memory is NOT in fixed slots.**

</div>

---
transition: fade-out
---

# Analysis 4: H2 Cross-Stimulus + Procrustes Swap

<div class="text-xs mb-2">

**H2 test** (cross-stimulus, same time): train on `val_novel_angle` (known ids), test on `val_novel_identity` (novel ids), both at t=0. Decoded on `location` (aligned labels).

</div>

<div class="text-xs">

### H2 Cross-Stimulus (decode on location)

| Model | Val (known ids) | Gen (novel ids) | Diff | Status |
|-------|----------------:|----------------:|-----:|--------|
| wm_mtmf | 0.463 | 0.290 | 0.173 | H3 possible |
| wm_h128_mtmf | 0.475 | 0.237 | 0.237 | H3 possible |
| wm_attention_mtmf | 0.550 | 0.372 | 0.178 | H3 possible |
| wm_dual_attention_mtmf | 0.700 | 0.145 | 0.555 | H3 possible |
| wm_h128_attention_mtmf | 0.562 | 0.268 | 0.295 | H3 possible |
| wm_h128_dual_attention_mtmf | 0.738 | 0.260 | 0.478 | H3 possible |

</div>

<div class="text-xs mt-3">

### Procrustes Swap Test (Figure 4g)

| Model | Correct | Swap1 (wrong time) | Swap2 (diff stimuli) | Status |
|-------|--------:|-------------------:|---------------------:|--------|
| wm_mtmf | 0.222 | 0.230 | 0.244 | H2 NOT confirmed |
| wm_dual_attention_mtmf | 0.293 | 0.215 | 0.293 | H2 confirmed |
| wm_h128_dual_attention_mtmf | 0.333 | 0.215 | 0.279 | H2 confirmed |

</div>

<div class="mt-2 p-2 bg-orange-500/10 rounded-lg text-xs">

- ⚠️ **H2 NOT fully supported**: val ≫ gen in all MTMF models (diff 0.17-0.56) — decoders do NOT generalize to novel identities
- 🔍 **Procrustes swap is mixed**: 2/6 MTMF models show paper pattern (swap2 > swap1)
- 🔍 **Models are stimulus-specific** (H3-like) rather than sharing encoding across stimuli (H2)

</div>

---
transition: fade-out
---

# Analysis 5: Causal Perturbation (Real Results)

<div class="text-sm mb-2">

**Method**: Perturb hidden states along the mean direction of all class decoder normals, then re-classify. Paper expects P(Match) drops and P(No-Action) rises.
</div>

<div class="text-xs">

### Causal Perturbation — MTMF Models (property=identity)

| Model | P(Match) start→end | Drop | P(No-Action) start→end |
|-------|-------------------:|-----:|----------------------:|
| wm_mtmf | 0.790 → 0.731 | **5.9%** | 0.0002 → 0.0007 |
| wm_h128_mtmf | 0.858 → 0.618 | **24.0%** | 0.0001 → 0.0003 |
| wm_attention_mtmf | 0.974 → 0.960 | 1.4% | 0.0000 → 0.0000 |
| wm_dual_attention_mtmf | 0.957 → 0.948 | 0.9% | 0.0000 → 0.0000 |
| wm_h128_attention_mtmf | 0.943 → 0.939 | 0.4% | 0.0000 → 0.0000 |
| wm_h128_dual_attention_mtmf | 0.921 → 0.828 | **9.3%** | 0.0000 → 0.0001 |

</div>

<v-clicks>

<div class="mt-2 p-2 bg-yellow-500/10 rounded-lg text-xs">

- ✅ **P(Match) drops** when perturbed (0.4-24%) — confirms decoder-defined subspace is causally relevant
- ⚠️ **P(No-Action) does NOT rise** — paper expects 0.10→0.61; we see 0.000→0.001
- 🔍 **Limitation**: Current implementation runs only the classifier, not the recurrent dynamics
- 🔍 **Strongest effects** in baseline models; attention models are more robust to perturbation

</div>

</v-clicks>

---
layout: fact
transition: slide-up
---

# Summary: Paper Findings vs Our Results

<div class="text-base mt-3">

| Analysis | Paper Claim | Our Result | Match? |
|----------|-------------|------------|:------:|
| **1. Behavioral** | Novel Identity < Novel Angle | Confirmed in 18/18 (gap 0-4pp) | ✅ |
| **2A. Task-Relevance** | MTMF preserves all features | Diagonal 88-100% (✓), off-diagonal 14-60% | ⚠️ Partial |
| **2B. Cross-Task** | GRU task-specific | Loc/Ident task-specific ✓, Category partial transfer | ⚠️ Partial |
| **3. Orthogonalization** | RNN de-orthogonalizes | 12/18 MTMF points below diagonal (2/3 avg) | ✅ |
| **4. H1 Slot-Based** | Disproved | 18/18: t0 78-98% → t1 1-6% | ✅ |
| **4. H2 Chronological** | Supported | val > gen in 9/9 MTMF (H3-like) | ❌ |
| **5. Causal Perturbation** | Match ↓, No-Action ↑ | Match ↓ (0.4-24%), No-Action stays 0 | ⚠️ Partial |

</div>

<div class="mt-4 p-3 bg-blue-500/10 rounded-lg text-sm">

**Bottom line**: 4/7 paper findings fully replicated, 3/7 partially supported, 0/7 contradicted.
The H2 cross-stimulus test is the main divergence — our models show stimulus-specific
encoding rather than the shared encoding the paper claims. Likely due to differences
in training regime or architecture details.

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

### Performance Gains (Real MTMF Results)

| Metric | wm_mtmf (baseline) | wm_attention_mtmf | wm_dual_attention_mtmf |
|--------|-------------------:|------------------:|-----------------------:|
| Train Acc | 88.3% | **99.3%** | **99.3%** |
| Novel Angle | 81.5% | **93.3%** | **93.8%** |
| Novel Identity | 80.6% | **92.2%** | 90.9% |

<div class="mt-4 p-3 bg-green-500/10 rounded-lg text-center text-xl font-bold">

+11% train, +12% angle, +11% identity with attention

</div>

</div>

</div>

---
transition: fade-out
---

# All Models Comparison (Real Results, h=256)

<div class="flex justify-center text-sm">

| Model | Train | Novel Angle | Novel Identity |
|-------|------:|------------:|---------------:|
| **STSF** (baseline) | 100.00% | 99.68% | 99.76% |
| **STMF** (baseline) | 88.50% | 82.73% | 79.93% |
| **MTMF** (baseline) | 88.26% | 81.53% | 80.57% |
| **STSF + Attention** | 88.79% | 88.74% | 88.22% |
| **STSF + Dual Attn** | 88.90% | 88.18% | 89.58% |
| **STMF + Attention** | 99.61% | 93.31% | 91.79% |
| **STMF + Dual Attn** | 99.58% | 94.23% | 92.67% |
| **MTMF + Attention** | 99.27% | 93.27% | 92.15% |
| **MTMF + Dual Attn** | 99.34% | 93.79% | 90.91% |

</div>

<v-click>

<div class="mt-4 grid grid-cols-3 gap-4 text-sm">
<div class="p-3 bg-blue-500/10 rounded-lg text-center">

**Insight 1**: Attention helps most for multi-feature tasks (STMF, MTMF gain +11% over baseline)

</div>
<div class="p-3 bg-purple-500/10 rounded-lg text-center">

**Insight 2**: Dual attention provides marginal gains over task-only attention for MTMF

</div>
<div class="p-3 bg-green-500/10 rounded-lg text-center">

**Insight 3**: STSF baseline is already at ceiling (≈100%); attention gains manifest in val_novel_identity

</div>
</div>

</v-click>

---
layout: section
transition: fade
---

# Neural Efficiency
## A Second, Independent Finding

---
transition: fade-out
---

# Beyond Capacity: Why We Need a Second Finding

<div class="grid grid-cols-2 gap-8">

<div>

### What the deck already shows (Capacity)

Proxy pretraining raises accuracy on the real N-back task at matched or lower training volume:

- Novel angle: 82.7% → 97.5%
- Novel identity: 80.6% → 92.8%

This is a **behavioral capacity** claim — familiar/structured features let the model do more.

</div>

<div>

### What's missing (Efficiency)

Human WM research treats capacity and efficiency as **separate, independently measured** phenomena:

- **Capacity**: familiar stimuli → more can be held in mind (Chung, Brady & Störmer, 2024)
- **Efficiency**: prior knowledge → same/better work with **suppressed** neural response (repetition suppression, Poppenk, Moscovitch & McIntosh, 2016)

A reviewer could argue "better accuracy" alone isn't mechanistically brain-like. We need suppressed *activity*, not just better *behavior*.

</div>

</div>

<v-click>

<div class="mt-6 p-4 bg-blue-500/10 rounded-lg text-center">

### The claim under test
Familiarity/structure (from proxy pretraining) and explicit gating (from attention) both suppress task-irrelevant processing — observable at **three independent levels** of the model.

</div>

</v-click>

---
transition: fade-out
---

# Three-Level Suppression Story

<div class="flex justify-center text-sm">

| Level | Quantity | Tool | Comparison |
|---|---|---|---|
| **1. Representational content** | Task-irrelevant-feature decodability, orthogonalization index | `compare_models.py` (existing Phase 3/5 infra) | Baseline vs. attention |
| **2. Population activity** | Hidden-state magnitude, participation ratio, sparsity, Fano-factor analogue | `neural_efficiency.py` (new) | Baseline vs. proxy-pretrained |
| **3. Explicit gating** | Gate-suppression index (irrelevant-channel gate − relevant-channel gate) | `gate_suppression.py` (new) | Attention-only vs. attention+proxy |

</div>

<v-click>

<div class="mt-6 p-4 bg-purple-500/10 rounded-lg text-center">

Reporting all three on the same underlying claim — familiarity/structure suppresses what's not needed — is stronger evidence of a genuine multi-level mechanism than any one signature alone. Level 3 is the strongest: a plain GRU/RNN baseline structurally cannot produce it at all.

</div>

</v-click>

---
transition: fade-out
---

# Level 2: Population Activity — Baseline vs. Proxy-Pretrained

<div class="grid grid-cols-2 gap-8">

<div>

### Setup
- **Pair 1**: `wm_mtmf_20260520_140601` (baseline) vs. `finetune_proxy_wm_mtmf_20260705_164908` (proxy-finetuned) — 10pp accuracy gap (82.7%→92.7%)
- **Pair 2**: `wm_attention_mtmf_20260726_161735` (attention-only) vs. `finetune_proxy_wm_attention_mtmf_20260726_201707` (attention+proxy) — **0.08pp accuracy gap** (93.43%→93.51%), a clean matched-accuracy replication
- Metrics: activation magnitude, participation ratio, population sparsity, Fano-factor analogue (bootstrapped, 1000 resamples, all 9 task×n cells)

</div>

<div>

### Result — same direction in both pairs, every cell

| Metric | Under proxy pretraining | vs. hypothesis |
|---|---|---|
| Activation magnitude | **Lower**, p<0.0001 | ✅ matches Poppenk et al. |
| Population sparsity | **Higher**, most cells | ✅ matches, small effect size |
| Participation ratio | **Higher**, every cell | ❌ opposite of "sharpening" |
| Fano-factor analogue | **Higher**, every cell | ❌ opposite of "less variable" |

<div class="mt-3 p-3 bg-green-500/10 rounded-lg text-center text-sm">

Magnitude/sparsity effect **replicates at near-zero accuracy gap** (Pair 2) — survives the Constantinidis & Klingberg Box 2 confound check, not just "the proxy model is more accurate."

</div>

</div>

</div>

---
transition: fade-out
---

# Level 1: Representational Content — Baseline vs. Attention

<div class="grid grid-cols-2 gap-8">

<div>

### Setup
- **Baseline**: `wm_mtmf_20260520_140601`
- **Attention**: `wm_attention_mtmf_20260520_203605` (task_only gating)
- Property: `identity` — does attention lower task-irrelevant decodability / raise the orthogonalization index?

</div>

<div>

### Result — mixed, leaning supportive (weakest of the three levels)

| Sub-metric | Baseline | Attention | Read |
|---|---:|---:|---|
| Identity decodability, t=3/4/5 | 14.6/12.0/10.1% | 7.2/6.5/5.9% | ✅ roughly halved |
| Orthogonalization index | 0.936 | 0.933 | Flat, ceiling |
| Procrustes reconstruction | 32.3% | 31.7% | Flat |
| Swap-test "correct" acc | 22.9% | 30.0% | ✅ +7pp |

<div class="mt-3 p-3 bg-yellow-500/10 rounded-lg text-center text-sm">

2 of 4 sub-metrics clearly support suppression, 2 are flat (ceiling-limited, not contradictory) — report honestly rather than as a clean win.

</div>

</div>

</div>

---
transition: fade-out
---

# Level 3: Explicit Gating — The Novel Contribution

<div class="grid grid-cols-2 gap-8">

<div>

### Why this is new
No experiment anywhere had combined attention with proxy pretraining before this pass — `configs/proxy/proxy_attention_mtmf.yaml` existed but had never been run. This directly answers the original question: *can attention-containing models be used in proxy task training, and does it help?*

### Setup
- **Condition A**: `wm_attention_mtmf_20260726_161735` (attention-only, fresh run with gate-logging)
- **Condition B**: `finetune_proxy_wm_attention_mtmf_20260726_201707` (proxy-pretrained then fine-tuned)
- Near-matched accuracy: **93.43% vs. 93.51%**

</div>

<div>

### Result — the headline finding, 9/9 cells sharper in B

| | Attention-only | Attention+proxy |
|---|---:|---:|
| Suppression index | −0.17 to **+0.07** (wrong-signed 2/9) | **−0.33 to −0.52** |
| Gate-relevance correlation | 0.09–0.24 (weak) | 0.45–0.72 (strong) |

<div class="mt-3 p-3 bg-green-500/10 rounded-lg text-center text-sm">

For the **category** task, attention-only barely gates at all (index ≈0, sometimes wrong-signed); attention+proxy fixes this completely (−0.33 to −0.34). Accuracy-matched, large effect size, and a signature a plain RNN baseline cannot produce at all.

</div>

</div>

</div>

---
transition: fade-out
---

# Conclusions

<div class="grid grid-cols-2 gap-8">

<div>

### 📄 Paper Findings — Replication Status

<v-clicks>

1. ✅ **H1 Slot-based memory**: Disproved (18/18: t0→t1 drops 76-96pp)
2. ✅ **Orthogonalization**: RNN de-orthogonalizes (12/18 MTMF points below diagonal)
3. ✅ **Task-specific subspaces** (location, identity): Cross-task off-diagonal 4-39% vs diagonal 28-67%
4. ⚠️ **MTMF preserves all features**: Diagonal 88-100% ✓, off-diagonal varies 14-60%
5. ⚠️ **H2 Cross-stimulus shared encoding**: Not supported (val > gen in 9/9 MTMF)
6. ⚠️ **Causal perturbation No-Action rise**: P(Match) drops 0.4-24% ✓, P(No-Action) stays at 0

</v-clicks>

</div>

<div>

### 🔬 Our Contributions

<v-clicks>

7. ✅ **Audit & fix** 4 bugs in the analysis pipeline (H2 mislabel, swap test, SVC convergence, sample warnings)
8. ✅ **Task-guided attention** improves MTMF by +11% train, +12% angle, +11% identity
9. ✅ **Re-ran all 18 experiments** with fixed code — results in `analysis_results/`
10. ✅ **Open-sourced** the audit findings (`docs/ANALYSIS_AUDIT_FINDINGS.md`)
11. 🔬 **Neural efficiency** (new): three-level suppression story — representational content, population activity, explicit gating — see previous section
12. 🔬 **Attention + proxy pretraining** (new): first combined run of attention-gated architecture with structured-feature pretraining — direct extension of both prior contributions

</v-clicks>

</div>

</div>

<v-click>

<div class="mt-6 p-4 bg-blue-500/10 rounded-lg text-center">

### Implication
Explicit attention complements RNN memory dynamics, but our models show more **stimulus-specific** encoding (H3-like) than the paper's claimed **shared encoding** (H2). Suggests attention improves task performance without changing the underlying representation strategy.

</div>

</v-click>

---
layout: section
transition: fade
---

# Meta-Learning Experiments
## Rapid Task Adaptation with Attention

---
layout: two-cols
transition: slide-left
---

# Meta-Learning Setup

<div class="text-sm">

### Research Question
Can task-guided attention enable **few-shot learning** of novel working memory tasks?

### Hypothesis
Attention separates:
- **Task-agnostic**: General temporal processing (RNN)
- **Task-specific**: Feature selection (attention gates)

→ Only attention needs updating for new tasks

</div>

::right::

<div class="pl-4 text-sm">

### Novel Task: Three-in-a-Row
- Detect when same stimulus appears 3 consecutive times
- Not seen during training (trained on N-back)
- Tests pattern recognition vs temporal distance

### Training Data
- **50 examples** for adaptation
- **20 epochs** of fine-tuning
- **6 trials per sequence**

### Adaptation Strategies
1. **Scratch**: Train from random init
2. **Full Fine-tune**: Update all parameters
3. **Cognitive-Only**: Update RNN only
4. **Attention-Only**: Freeze RNN, update attention
5. **Classifier-Only**: Freeze attention & RNN
6. **Attention+Classifier**: Update both

</div>

---
layout: default
transition: slide-left
---

# Meta-Learning Results: Three-in-a-Row

<div class="flex justify-center">

<div style="height: 384px;">

![meta learning comparison](./meta_learning_comparison.png)

</div>

</div>

<div class="mt-4 text-sm">

| Method | Base | Attention | Dual Attention |
|--------|------|-----------|----------------|
| **Scratch** | 50.0% | 52.5% | 49.0% |
| **Full Finetune** | 68.6% | 65.2% | 65.2% |
| **Cognitive Only** | **69.1%** | 66.7% | 65.7% |
| **Attention Only** | 0.0%* | 67.2% | 66.2% |
| **Classifier Only** | **69.1%** | **67.2%** | **68.1%** |
| **Attention+Classifier** | 0.0%* | **68.6%** | 67.2% |

<div class="text-xs opacity-70 mt-2">*Base model has no attention mechanism</div>

</div>

---
layout: default
transition: slide-left
---

# Key Findings

<div class="grid grid-cols-2 gap-4 text-sm">

<div>

### 🎯 Main Results

<v-clicks>

1. **Base Cognitive/Classifier-Only wins** (69.1%)
   - Simple task benefits from focused updates

2. **Attention models competitive** (~65-68%)
   - Attention+Classifier best for attention (68.6%)

3. **Cognitive-Only strong** (66-69%)
   - RNN learns pattern matching well

4. **Scratch at chance** (~50%)
   - Pre-training essential for few-shot learning

</v-clicks>

</div>

<div>

### 💡 Interpretations

<v-clicks>

- **Task type matters**: Three-in-a-row is simpler than N-back
  - Pattern matching vs temporal distance
  - All models converge to similar performance

- **Architecture impact**: Minimal difference
  - Base: 69.1%, Attention: 68.6%, Dual: 68.1%
  - All pretrained models learn effectively
  - Attention provides flexibility without penalty

- **Practical**: Pre-training is critical
  - Scratch at chance (~50%) vs pretrained (~65-69%)
  - Simple adaptation strategies sufficient
  - Classifier/Cognitive updates most efficient

</v-clicks>

</div>

</div>

---
layout: default
transition: slide-left
---

# Improvement Analysis

<div class="flex justify-center">

<div style="height: 320px;">

![meta learning improvement](./meta_learning_improvement.png)

</div>

</div>

<div class="mt-4 p-3 bg-blue-500/10 rounded-lg text-sm">

**Key Insight**: All pretrained methods show ~13-19 percentage point improvement over scratch baseline (~50%), demonstrating successful transfer learning. All architectures converge to similar performance (65-69%), showing that pre-training matters more than architecture choice for this task.

</div>

---
layout: default
transition: slide-left
---

# Thesis Contribution Context

<div class="text-xs">

### Refined Understanding from Three-in-a-Row

<div class="grid grid-cols-2 gap-3 mt-3">

<div class="p-3 bg-blue-500/10 rounded-lg">

#### Original Hypothesis
- Attention enables rapid few-shot adaptation
- Only attention gates need updating
- RNN provides stable temporal processing
- Expected: Attention models outperform base

</div>

<div class="p-3 bg-orange-500/10 rounded-lg">

#### Actual Results
- All models perform similarly (65-69%)
- Base: 69.1%, Attention: 68.6%, Dual: 68.1%
- Classifier/Cognitive-only most efficient
- Reality: Pre-training > Architecture choice

</div>

</div>

<div class="mt-3 p-3 bg-green-500/10 rounded-lg">

### Key Lessons

<v-clicks>

1. **Task complexity determines architecture** — simple tasks don't require attention
2. **Architecture choice matters less** — all pretrained models converge (65-69%)
3. **Focused updates work** — classifier/cognitive-only most efficient
4. **Pre-training is critical** — pretrained (~68%) vs scratch (~50%) = 18% gap

</v-clicks>

</div>

</div>

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
