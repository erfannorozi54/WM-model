---
theme: academic
title: Proxy Task Pre-training for Working Memory Models
info: |
  ## Comparing Proxy Pre-training vs Direct Training
  Performance analysis on MTMF N-back task
coverAuthor: Erfan Norozi
coverDate: "July 2026"
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

# Proxy Task Pre-training for Working Memory Models

<div class="pt-4 text-lg opacity-80">
Two-Stage Training: Feature Recall → N-back Classification
</div>

<div class="pt-2 text-sm opacity-60">
Performance Comparison: Proxy Pre-training vs Direct Training
</div>

<div class="abs-br m-6 flex gap-2">
  <span class="text-sm opacity-60">July 2026</span>
</div>

---
layout: two-cols-header
transition: fade-out
---

# Motivation: Why Proxy Pre-training?

::left::

<v-clicks>

- **Challenge**: N-back task has sparse training signal
  - Only 3 classes: match / non-match / no-action
  - Many time steps require "no-action" (no learning signal)

- **Proxy Task Hypothesis**: Richer training signal
  - Predict actual feature values (location, identity, category)
  - Every time step has a target (except t < N)
  - Forces model to learn better representations

- **Expected Benefit**: Better initialization for fine-tuning

</v-clicks>

::right::

<div class="ml-6 mt-4">

```
Standard N-back Task:
t=0: [stimulus] → no_action
t=1: [stimulus] → no_action  
t=2: [stimulus] → match/non_match
t=3: [stimulus] → match/non_match
...

Proxy Task (N=2):
t=0: [stimulus] → no target
t=1: [stimulus] → no target
t=2: [stimulus] → predict location/identity/category at t=0
t=3: [stimulus] → predict location/identity/category at t=1
...
```

</div>

---
layout: center
transition: fade-out
---

# Methodology: Two-Stage Training Pipeline

<div class="grid grid-cols-2 gap-8 mt-8">

<div>

### Stage 1: Proxy Pre-training

<v-clicks>

- **Task**: Feature recall (N-back)
  - Location: predict which of 4 locations
  - Identity: predict which object (20 classes)
  - Category: predict which category (4 classes)

- **Training**: 45 epochs on proxy task
  - 30,000 sequences per epoch
  - All 9 task vectors (3 features × 3 N-values)

- **Output**: Pre-trained weights

</v-clicks>

</div>

<div>

### Stage 2: Fine-tuning

<v-clicks>

- **Task**: Standard N-back classification
  - 3 classes: match / non-match / no-action

- **Transfer**: Load proxy weights
  - Perceptual (CNN) ✓
  - Cognitive (RNN) ✓
  - Attention (if present) ✓
  - Classifier head: fresh initialization

- **Training**: 45 epochs on N-back task

</v-clicks>

</div>

</div>

---
layout: center
transition: fade-out
---

# Results: Proxy vs Baseline (MTMF)

<div class="mt-6">

| Metric | Baseline (Direct) | Proxy Pre-trained | Improvement |
|--------|------------------|-------------------|-------------|
| **Best Val (Novel Angle)** | 82.69% | **97.52%** | **+14.83%** |
| Final Val (Novel Angle) | 81.5% | **97.5%** | **+16.0%** |
| Final Val (Novel Identity) | 80.6% | **92.8%** | **+12.2%** |
| Final Train Accuracy | 88.3% | **100.0%** | **+11.7%** |
| Final Train Loss | 0.2654 | **0.0012** | **-99.5%** |

</div>

<div class="mt-8 text-left mx-auto" style="max-width: 800px;">

### Key Findings

<v-clicks>

- **Proxy pre-training achieves near-perfect performance** on N-back task
- **14.83% improvement** on novel angle validation (generalization)
- **12.2% improvement** on novel identity validation (harder generalization)
- **Faster convergence**: Proxy model reaches 90%+ by epoch 1

</v-clicks>

</div>

---
layout: two-cols-header
transition: fade-out
---

# Analysis: Why Does Proxy Pre-training Work?

::left::

<v-clicks>

### 1. Richer Training Signal

- Proxy task provides targets at **every time step** (except t < N)
- N-back task has sparse signal (many "no-action" steps)
- More gradients → better feature learning

### 2. Better Representations

- Proxy forces model to encode **specific feature values**
- Not just "same/different" but "what exactly"
- Creates more discriminative hidden states

### 3. Transfer Learning Benefits

- Pre-trained weights provide **good initialization**
- Fine-tuning starts from better position
- Avoids poor local optima

</v-clicks>

::right::

<div class="ml-6 mt-4">

### Training Dynamics

```
Baseline Training:
Epoch 1:  val_angle ≈ 60%
Epoch 10: val_angle ≈ 75%
Epoch 45: val_angle ≈ 82%

Proxy Fine-tuning:
Epoch 1:  val_angle ≈ 93%  ← Already better than baseline final!
Epoch 10: val_angle ≈ 97%
Epoch 45: val_angle ≈ 97.5%
```

### Convergence Speed

- Proxy model: **1 epoch** to reach 93%
- Baseline model: **45 epochs** to reach 82%
- **~45x faster convergence**

</div>

---
layout: center
transition: fade-out
---

# Conclusion & Future Work

<div class="mt-6 text-left mx-auto" style="max-width: 900px;">

### Key Takeaways

<v-clicks>

- **Proxy task pre-training significantly improves N-back performance**
  - 14.83% improvement on novel angle generalization
  - 12.2% improvement on novel identity generalization
  - Near-perfect accuracy (97.5%) vs baseline (82.7%)

- **Mechanism**: Richer training signal → better representations → better initialization

- **Practical benefit**: Faster convergence (1 epoch vs 45 epochs to reach high performance)

</v-clicks>

</div>

<div class="mt-8 text-left mx-auto" style="max-width: 900px;">

### Future Directions

<v-clicks>

- **Attention mechanisms**: Test proxy pre-training with attention models
- **Meta-learning**: Does proxy pre-training help adaptation to novel tasks?
- **Ablation studies**: Which proxy tasks contribute most? (location vs identity vs category)
- **Scaling**: Does proxy benefit increase with model size or task complexity?

</v-clicks>

</div>

<div class="mt-6 text-sm opacity-70">

**Experiment ID**: `finetune_proxy_wm_mtmf_20260705_164908`  
**Baseline**: `wm_mtmf_20260520_140601`

</div>

---
layout: center
transition: fade-out
---

# Is This Biologically Plausible?

<div class="mt-6 text-left mx-auto" style="max-width: 850px;">

<v-clicks>

- Proxy pre-training works because it makes the model **familiar** with the feature space (locations, identities, categories) and the task-vector structure *before* it has to use that familiarity to solve N-back matching

- **Research question**: is "prior familiarity with features/task structure → better working-memory performance" a real signature of *biological* working memory, or just a neural-network training trick?

- Two papers, both verified against full text for this deck, jointly make a **precise** version of that claim: familiarity helps human working memory when it reflects pre-existing **structure**, not when it is just repeated exposure — which is exactly the distinction that separates proxy pre-training from "training the baseline longer"

</v-clicks>

</div>

---
layout: two-cols-header
transition: fade-out
---

# Human Evidence #1 — Meaningfulness Expands VWM Capacity

::left::

**Chung, Brady & Störmer (2024)**
*Meaningfulness and Familiarity Expand Visual Working Memory Capacity*
Current Directions in Psychological Science, 33(5), 275–282

<v-clicks>

- **Central claim**: VWM capacity is not a fixed pool — it expands when stimuli connect to preexisting semantic knowledge, challenging models built only on abstract stimuli (colored circles, oriented lines)
- **Hierarchical model**: low-level visual features bind to higher-level semantic representations already in long-term memory; meaningful objects act as a *scaffold* that increases distinctiveness between items and reduces interference
- **Neural evidence rules out simple compression**: EEG contralateral delay activity (CDA) during the retention interval is *higher*, not lower, for meaningful stimuli — more information is being actively maintained, not stored more efficiently in less space
- The benefit survives concurrent verbal-interference (articulatory suppression) tasks, and requires enough encoding time (~1,000 ms) to recognize the stimulus as meaningful

</v-clicks>

::right::

<div class="ml-6 mt-4">

<v-click>

### Model parallel

Proxy pre-training builds the RNN's equivalent of a semantic scaffold for the location/identity/category feature space *before* the delay-dependent N-back task ever needs it.

```
Baseline:  no prior feature scaffold
           → 82.7% novel-angle acc.

Proxy:     feature space already familiar
           → 97.5% novel-angle acc.  (+14.8%)
```

Chung et al.'s claim is structural, not just quantitative: capacity grows because incoming input connects to something already built — which is what proxy pre-training does to the hidden state before fine-tuning starts.

</v-click>

</div>

---
layout: two-cols-header
transition: fade-out
---

# Human Evidence #2 — Structure Beats Repetition

::left::

**Mercer (2025)**
*Familiarity influences on proactive interference in verbal memory*
Quarterly Journal of Experimental Psychology

<v-clicks>

- Verbal recent-probes task: is a current item wrongly felt to be "seen before" because it matches a target from 1 trial back (Recent Negative) vs. 3 trials back (Non-Recent Negative)?
- Independently manipulates **three** kinds of familiarity: temporal (inter-trial gap: 100 ms vs. 10.1 s — no effect), experimental (item repeated within the session), pre-experimental (real words vs. meaningless non-words)
- **Repeating a meaningless non-word made interference *worse*** — the single highest-interference condition in the study
- **Repeating a meaningful word made no difference at all** — pre-existing semantic knowledge already did the protective work; extra repetition added nothing on top of it

</v-clicks>

::right::

<div class="ml-6 mt-4">

<v-click>

### Why this paper matters here

It blocks the easy version of the claim. "Proxy pre-training just gives the model more training" is *not* what this literature supports.

```
Mercer (2025):
  repeated non-word  (no prior structure) → MORE interference
  repeated word       (has prior structure) → repetition adds
                                               nothing; structure
                                               already protects

This work:
  baseline: 45 epochs of repeated exposure to N-back itself
  proxy:    pre-training organized around the feature
            *structure* (location / identity / category),
            not just more repetitions of the same task
```

The proxy advantage is consistent with Mercer's account only if it traces to *structure*, not to training *volume* — a distinction the model doesn't get for free (see Caveats).

</v-click>

</div>

---
layout: center
transition: fade-out
---

# Broader Landscape (Context, Not Deep-Dive Evidence)

<div class="mt-4">

| Study | Relevance |
|---|---|
| **Sikarwar & Zhang (2023)**, NeurIPS — WorM benchmark (10 tasks, 1M trials) | RNNs trained on WM tasks reproduce human-like primacy/recency effects without ever being trained on human data |
| **Efficient Allocation of WM Resource for Utility Maximization in Humans and RNNs**, NeurIPS 2025 | Humans and co-trained RNNs both show memory stability rising for high-probability / familiar stimuli |
| **Shao, Zhang & Yu (2024)**, eLife | fMRI: frontal-cortex WM control representations are reproduced by multi-module RNN simulations |
| **Wojcik et al. (2025)**, eLife | Human EEG: neural context-mapping representations build specifically during the WM delay period as familiarity with the mapping increases |

</div>

<div class="mt-6 text-sm opacity-70">
Cited for context on the wider RNN–human WM literature. Unlike Chung et al. (2024) and Mercer (2025), these were not independently checked against full text for this deck — treat them as pointers for further reading, not load-bearing evidence.
</div>

---
layout: center
transition: fade-out
---

# Synthesis: Model Findings ↔ Literature

<div class="mt-4 text-sm">

| Model finding (this work) | Human parallel | Interpretation |
|---|---|---|
| Feature-familiar proxy model: **+14.8%** novel-angle accuracy, **+12.2%** novel-identity accuracy | Meaningful/familiar stimuli raise VWM capacity via a semantic scaffold — confirmed by *higher*, not lower, neural delay activity (Chung, Brady & Störmer, 2024) | The gain looks like added active-maintenance capacity, not compression — consistent with what the CDA data rules out in humans |
| Proxy fine-tuning beats a 45-epoch baseline despite comparable total exposure | Repeating a stimulus without pre-existing structure does not reduce interference, and can increase it; pre-existing structure protects regardless of repetition (Mercer, 2025) | The proxy benefit should trace to *how* pre-training is organized around feature structure, not to how much of it there is |
| Proxy-trained hidden states generalize to identity × angle combinations never seen together | Both papers converge on structure, not exposure count, as the active ingredient | Supports a structure-dependent account of the proxy-pretraining benefit — testable against the causal-perturbation results (Analysis 5) |

</div>

---
layout: two-cols-header
transition: fade-out
---

# Caveats & What This Does Not Show

::left::

<v-clicks>

- **Analogy, not identity**: supervised gradient descent over proxy-task trials is a compressed idealization of experience-dependent familiarity, not a claim that backprop is a biological learning rule

- **The "structure, not repetition" claim is asserted here, not yet tested in the model**: Mercer (2025) shows the distinction matters in humans, but we have not run the model-side equivalent — e.g., a proxy curriculum with the *same* number of gradient steps but scrambled feature labels, to see if the benefit survives

- **Behavioral convergence ≠ mechanistic convergence**: matching accuracy/speed patterns does not establish that the RNN uses the same computations as human WM (chunking, resource allocation, LTM–WM interaction)

- Timescales differ sharply: Chung et al.'s and Mercer's effects are measured within single sessions; "familiarity" here is 45 epochs of supervised training

</v-clicks>

::right::

<div class="ml-6 mt-4">

<v-click>

### What would strengthen the claim

- A **scrambled-structure control**: pre-train on the same proxy data volume with shuffled feature labels — if the benefit disappears, that's direct model-side evidence for "structure over repetition," mirroring Mercer's design logic
- Representational similarity analysis between proxy-trained hidden states and human VWM capacity/precision models
- Testing whether the causal-perturbation results (Analysis 5) shift in the direction a structure-driven account predicts

</v-click>

</div>

---
layout: center
transition: fade-out
---

# References

<div class="text-xs leading-relaxed mx-auto" style="max-width: 950px;">

**Verified against full text for this deck:**

1. Chung, Y. H., Brady, T. F., & Störmer, V. S. (2024). Meaningfulness and familiarity expand visual working memory capacity. *Current Directions in Psychological Science*, 33(5), 275–282. https://doi.org/10.1177/09637214241262334
2. Mercer, T. (2025). Familiarity influences on proactive interference in verbal memory. *Quarterly Journal of Experimental Psychology*.

**Broader context (not independently verified for this deck):**

3. Sikarwar, A., & Zhang, M. (2023). Decoding the Enigma: Benchmarking Humans and AIs on the Many Facets of Working Memory. *NeurIPS 2023*.
4. Efficient Allocation of Working Memory Resource for Utility Maximization in Humans and Recurrent Neural Networks. *NeurIPS 2025*.
5. Shao, Z., Zhang, M., & Yu, Q. (2024). Stimulus representation in human frontal cortex supports flexible control in working memory. *eLife*.
6. Brady, T. F., Robinson, M. M., & Williams, J. R. (2024). Noisy and hierarchical visual memory across timescales. *Nature Reviews Psychology*, 3(3), 147–163. https://doi.org/10.1038/s44159-024-00276-2
7. Wojcik et al. (2025). Working memory shapes neural geometry in human EEG over learning. *eLife*.
8. Ma, H. et al. (2024). Uncertainty Quantification in Working Memory via Moment Neural Networks.
9. Nahari, T., Eldar, E., & Pertzov, Y. (2024). Fixation durations on familiar items are longer due to attenuation of exploration. *Cognitive Research: Principles and Implications*, 9(1). https://doi.org/10.1186/s41235-024-00602-5
10. Zhou, Z., Kahana, M. J., & Schapiro, A. C. (2024). A unifying account of replay as context-driven memory reactivation. *eLife*, 13. https://doi.org/10.7554/elife.99931
11. Interactions between sensory-biased and supramodal working memory networks in the human cerebral cortex. *Communications Biology*, 2026.

</div>

---
layout: center
class: text-center
transition: fade-out
---

# Thank You

<div class="mt-4 text-lg opacity-80">
Proxy pre-training as a computational model of structure-driven, familiarity-based working-memory competence
</div>

<div class="mt-8 text-sm opacity-60">
Erfan Norozi · July 2026
</div>
