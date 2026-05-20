---
theme: academic
title: Neural Geometry and Dynamics of Working Memory
info: |
  ## Theoretical Innovations and Future Directions
  Based on naturalistic object representations in RNNs
coverAuthor: Erfan Norozi
coverDate: "May 2026"
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

# Neural Geometry and Dynamics of Naturalistic Object Representations in RNNs

<div class="pt-4 text-lg opacity-80">
Theoretical Innovations and Future Directions
</div>

<div class="pt-2 text-sm opacity-60">
Working Memory Models: From Modularity to Manifolds
</div>

---
layout: default
---

# The Paradigm Shift in Working Memory

<v-clicks>

## From Modular Buffers to Neural Manifolds

- **Traditional View**: Working memory as modality-specific buffers controlled by a central executive<sup>1</sup>
  - Memory as static repository
  - Fixed "slots" for information storage

- **Modern View**: Dynamical systems perspective<sup>2</sup>
  - Working memory as emergent property of neural population activity
  - Neural manifolds: low-dimensional surfaces in high-dimensional activity space
  - Represents internal cognitive states

- **The Challenge**: Bridge artificial models with biological reality using naturalistic, multidimensional stimuli<sup>1</sup>

</v-clicks>

---
layout: default
---

# Evolution of Working Memory Models

<v-clicks>

## Historical Foundations

- **Baddeley's Modular Theory**<sup>1</sup>:
  - Phonological loop
  - Visuospatial sketchpad
  - Central executive
  - Limited-capacity system (Von-Neumann architecture analogy)

## Neuroscience Challenges

- Distributed brain networks don't follow rigid boundaries<sup>1</sup>
- Complex time-varying dynamics in prefrontal cortex
- Flexible, high-dimensional representational space<sup>3</sup>

</v-clicks>

---
layout: default
---

# The Naturalistic Turn

<v-clicks>

## Limitations of Previous Research

- **Categorical/one-hot inputs**<sup>1</sup>:
  - Simplified model training
  - Failed to capture real-world complexity

## Why Naturalistic Stimuli Matter

- Ecologically relevant and multidimensional<sup>1</sup>
- Perceived from different angles, distances, lighting
- Must map to stable identities in memory

## Solution: Sensory-Cognitive Models

- Integrate visual processing with recurrent dynamics<sup>1</sup>
- CNN for sensory encoding + RNN for cognitive processing

</v-clicks>


---
layout: center
class: text-center
---

# Part I: Attention vs. Concentration

Neurobiological foundations and distinct systems

---
layout: default
---

# Executive Summary: Definitions

<v-clicks>

## Attention: The Gateway

- **Foundational, mechanistic** information processing
- Neurobiological network for selecting relevant sensory data
- Filters out noise

## Concentration: Sustained Engagement

- Not an independent neural system
- Operationalized as **sustained attention + working memory capacity**
- Inherently active, continuous, prefrontal-driven (top-down)

## Key Distinction

- **Attention**: Can be passive, brief, externally triggered (bottom-up)
- **Concentration**: Active, continuous, internally driven (top-down)

</v-clicks>

---
layout: default
---

# Two Distinct Neural Systems

<v-clicks>

## 1. Attentional Selection (The Spotlight)

**Posner & Rothbart's Three Networks:**

- **Alerting Network**: Readiness/arousal
  - Right frontal/parietal cortices
  - Norepinephrine from locus coeruleus

- **Orienting Network**: Directs sensory focus
  - Parietal lobe, superior colliculus

- **Executive Network**: Conflict resolution, filtering
  - Anterior cingulate cortex, prefrontal regions

</v-clicks>

---
layout: default
---

# Sustained Concentration (The Holding Force)

<v-clicks>

## Prolonged Executive Orchestration

- Maintains behavioral responses over time
- "Block-level" cognitive state

## Neural Signatures

- **Frontal theta rhythms** (4–7 Hz)
- **Occipital alpha shifts**
- Suppresses default mode network (mind-wandering)

## Metabolic Effort

- Attention: Quick spotlight flick
- Concentration: Massive effort to **lock spotlight in place**

</v-clicks>

---
layout: default
---

# Key Empirical Differences

| Feature | Attention (Selective/Transient) | Concentration (Sustained) |
|---------|--------------------------------|---------------------------|
| **Neural Correlates** | Parietal cortex, superior colliculus, subcortical networks | Prefrontal cortex, frontoparietal control, WM circuits |
| **Temporal Nature** | Transient, event-related, millisecond shifts | Block-level; decays after 20–25 minutes |
| **Control Direction** | Bottom-Up or Top-Down | Almost exclusively Top-Down |
| **Cognitive Dependency** | Initial filter; prerequisite for cognition | Highly dependent on Working Memory |

---
layout: default
---

# Evidence 1: Factor Analysis Separation

<v-clicks>

## Landmark 2025 Study (Sharpe & Smith)

- **Question**: Is sustained focus the same as attention control?
- **Method**: Structural equation modeling on large-scale human data

## Key Findings

- **Two-factor model** (sustained attention vs. attentional control) outperformed single-factor
- Sustained attention factor strongly predicted **long-term memory formation**
- Concentration has distinct cognitive footprint from attention switching

> Sharpe, M., & Smith, T. (2025). Sustained attention is more closely related to long-term memory than to attentional control. *bioRxiv*, 643171.

</v-clicks>

---
layout: default
---

# Evidence 2: Visual Concentration Pathways

<v-clicks>

## Zhang et al. (2023) Review

- Specific neural pathways for **visual sustained attention**
- Selective attention: prioritizes information
- Sustained concentration: persistent metabolic gatekeeper

## Vigilance Decrement

- Prolonged concentration → prefrontal fatigue
- Brain defaults to task-unrelated distractors
- Concentration as depletable resource

> Zhang, L., et al. (2023). A review of visual sustained attention: neural mechanisms and computational models. *Frontiers in Computational Neuroscience*, PMC10274610.

</v-clicks>

---
layout: default
---

# Evidence 3: The 25-Minute Threshold

<v-clicks>

## Classroom Cognitive Fatigue (Corriveau et al., 2025)

- Human concentration is **metabolically capped**
- High-level concentration falters after ~25 minutes

## Neural Mechanism

- Drop in frontal delta and theta wave coherence
- Leads to attention lapses
- **Concentration**: Depletable resource
- **Basic orienting attention**: Remains intact

> Corriveau, R., et al. (2025). Sustaining student concentration: the effectiveness of micro-breaks in a classroom setting. *Frontiers in Psychology*, 1589411.

</v-clicks>

---
layout: default
---

# Implications for Working Memory Models

<v-clicks>

## Attention in N-back Tasks

- Selective attention filters incoming stimuli
- Orienting network directs focus to relevant features

## Concentration Requirements

- Sustained attention maintains task engagement
- Working memory holds representations across trials
- Prefrontal control prevents distractor interference

## Design Considerations

- Task duration and metabolic demands
- Top-down control mechanisms
- Integration with WM capacity limits

</v-clicks>

---
layout: center
class: text-center
---

# Part II: Geometric Principles

Neural representations in hybrid CNN-RNN models

---
layout: default
---

# Simultaneous Representation of Features

<v-clicks>

## Task-Relevant + Task-Irrelevant Information

Multi-task RNNs maintain rich representations<sup>1</sup>:

- **Example**: Network tasked with object identity also encodes viewing angle
- Recurrent dynamics don't filter all irrelevant data
- Information organized to remain accessible while prioritizing goal-relevant dimensions

| Architectural Feature | Effect on Latent Space |
|----------------------|------------------------|
| Multi-task Training | Simultaneous encoding of multiple properties<sup>1</sup> |
| Recurrent Dynamics | Maintains info across distractors<sup>1</sup> |
| CNN Front-end | Maps pixels to perceptual features<sup>1</sup> |
| Linear Readout | Extracts decisions from manifold<sup>6</sup> |

</v-clicks>

---
layout: default
---

# Gating and Subspace Specialization

<v-clicks>

## Critical Architectural Distinction<sup>1</sup>

| RNN Type | Subspace Relationship | Generalization Strategy |
|----------|----------------------|------------------------|
| Vanilla (Gateless) | Largely Shared | Reusable representations<sup>1</sup> |
| Gated (GRU/LSTM) | Highly Task-Specific | Partitioned representations<sup>1</sup> |

## Implications

- **Gated networks**: "Switch" between geometric configurations per task<sup>1</sup>
  - Higher performance on individual tasks
  
- **Vanilla RNNs**: Shared subspaces offer flexibility<sup>1</sup>
  - May mirror multifunctional biological populations

</v-clicks>

---
layout: default
---

# The Orthogonalization Paradox

<v-clicks>

## Expected: Orthogonalization Minimizes Interference<sup>5</sup>

- Different features on orthogonal dimensions
- Independent processing without cross-talk

## Observed: Reduced Orthogonalization<sup>1</sup>

- Object features **less orthogonalized** in RNN hidden states than in CNN perceptual space
- Counter-intuitive finding

## Proposed Explanations

1. **Integration over separation**: Facilitates comparison across time<sup>5</sup>
2. **Dimensionality reduction**: Sensory buffers → constrained cognitive representations<sup>5</sup>

## Open Questions<sup>5</sup>

- Measurement artifacts (PCA limitations)?
- Functionally significant or epiphenomenal?

</v-clicks>


---
layout: center
class: text-center
---

# Part III: Temporal Dynamics

Chronological memory subspaces and dynamic coding

---
layout: default
---

# Chronological Memory Subspaces

<v-clicks>

## The Temporal Binding Problem<sup>1</sup>

- N-back task: differentiate stimulus at t-1 from t-2
- How do RNNs track temporal order?

## Solution: Distinct Memory Subspaces<sup>1</sup>

| Temporal Stage | Geometric Operation | Functional Purpose |
|----------------|--------------------|--------------------|
| Encoding | Stimulus-to-Memory Transform | Stable latent subspace<sup>1</sup> |
| Retention | Time-Dependent Rotation | Chronological "slots"<sup>10</sup> |
| Distraction | Distinct Transformations | Protects from interference<sup>1</sup> |

## Testable Prediction

- Brain should show latency-dependent populations
- Activity patterns correspond to "how long ago"<sup>1</sup>

</v-clicks>

---
layout: default
---

# Dynamic Coding and Manifold Rotations

<v-clicks>

## Core Concept<sup>3</sup>

- **Coding space** for stimulus is stable
- **Activity patterns** representing stimulus change over time

## Evidence from Neuroscience<sup>3</sup>

- fMRI and neural recordings support rotated representations
- Working memory = rotated versions of perceptual representations
- Separates current percept from memory (multiplexing)

## Distractor Interference<sup>3</sup>

- Interference linked to subspace alignment
- Distractor forced into memory subspace → performance decreases
- Importance of stable, non-overlapping subspaces

</v-clicks>

---
layout: center
class: text-center
---

# Part IV: Mathematical Frameworks

Laplace Neural Manifolds and traveling waves

---
layout: default
---

# Laplace Neural Manifolds

<v-clicks>

## The "What × When" Problem<sup>11</sup>

- How does brain represent continuous time as events recede?
- Requires mixed selectivity: conjunctive receptive fields<sup>11</sup>

## Mathematical Framework<sup>11,12</sup>

| Component | Function | Neural Representation |
|-----------|----------|----------------------|
| Laplace Population | Encodes real Laplace transform | Stable subspace; "edge" attractor<sup>11</sup> |
| Inverse Laplace | Approximates original function | Rotational dynamics; "bump"<sup>11</sup> |
| Log-Time Tiling | Scale-invariance | Basis functions tile log time<sup>11</sup> |

## Key Properties

- Temporal memory buffer stores event history<sup>12</sup>
- Bridges cognitive models with circuit-level dynamics<sup>11</sup>

</v-clicks>

---
layout: default
---

# Scale Invariance and Logarithmic Growth

<v-clicks>

## Human/Animal Timing Behavior<sup>11</sup>

- **Scale invariance**: Timing error ∝ interval length
- Explained by logarithmically tiled basis functions

## Neural Evidence<sup>11</sup>

- Covariance matrix rank grows logarithmically with time
- Powerful mathematical constraint on neural dynamics
- Defines what dynamics can support functional working memory

</v-clicks>

---
layout: default
---

# Hidden Traveling Waves

<v-clicks>

## Emerging Theme<sup>15</sup>

- Traveling waves coordinate activity across scales
- Oscillations move across cortex/hippocampus

## Human Hippocampal Theta<sup>15</sup>

- Confirmed as traveling waves (posterior-anterior)
- Different positions → different phases
- Crucial for phase-coding mechanisms

## Binding in Artificial Models<sup>16,17</sup>

- Hidden traveling waves bind WM variables
- Spatial and temporal organization intertwined
- Maintains structured multi-variable representations

</v-clicks>


---
layout: center
class: text-center
---

# Part V: Biological Constraints

Metabolic efficiency and spiking dynamics

---
layout: default
---

# Rotational Dynamics as Efficient Solution

<v-clicks>

## Metabolic Constraints Shape Geometry<sup>16</sup>

- Brains are energy-efficient systems
- Pressure to minimize wiring length and metabolic cost

## Rotational Dynamics<sup>19</sup>

- Networks trained with metabolic constraints develop rotations
- Not just time encoding—energy-efficient information maintenance
- Rotation clears input channel while protecting previous info<sup>3</sup>

## Spatial Organization<sup>16</sup>

- Wiring minimization → spatiotemporal locality
- Topographic organization emerges
- Grid-like representations (hippocampal-entorhinal system)<sup>21</sup>

</v-clicks>

---
layout: default
---

# Spiking Neural Networks

<v-clicks>

## Beyond Continuous RNNs<sup>22</sup>

- Biological brain operates via discrete spikes
- SNNs present unique challenges and insights

## Temporal Stability Mechanisms<sup>23</sup>

- Strong inhibitory signaling
- Specific inhibitory-inhibitory connectivity motifs

## Long Intrinsic Timescales<sup>23</sup>

- "Slow" neurons contribute to WM performance
- Sculpted by training to maintain delay activity
- Organization determines robustness to noise/distractors

</v-clicks>

---
layout: center
class: text-center
---

# Part VI: Comparative Strategies

Species differences and developmental trajectories

---
layout: default
---

# Monkey vs. Human Strategies

<v-clicks>

## Strategy Discrepancies<sup>24</sup>

- **Monkeys**: Recency-based strategy (recent stimuli dominate)
- **Humans**: Target-selective strategy (optimal)

## ANN Developmental Stages<sup>24</sup>

| Training Stage | Strategy | Representational Signature |
|----------------|----------|---------------------------|
| Untrained | Random | No manifold structure<sup>24</sup> |
| Partially Trained | Recency-like | Emerging structure; high interference<sup>24</sup> |
| Fully Trained | Target-Selective | Organized subspaces; low interference<sup>24</sup> |

## Key Insight<sup>24</sup>

- Species differences may reflect developmental progression
- Not fundamental architectural differences
- Neural manifold maturation drives behavioral change

</v-clicks>


---
layout: center
class: text-center
---

# Part VII: Innovative Research Directions

High-potential avenues for master's thesis

---
layout: default
---

# Proposal 1: Causal Intervention in Chronological Subspaces

<v-clicks>

## Problem<sup>5</sup>

- Inconclusive causal relevance tests in current research
- Are chronological subspaces necessary or epiphenomenal?

## Innovation<sup>26</sup>

- Robust causal intervention techniques
- Manifold perturbations or Counterfactual Latent (CL) loss
- Systematically "delete" or rotate subspaces during delay

## Expected Contribution

- Definitive answer on functional vs. epiphenomenal
- Major methodological contribution

</v-clicks>

---
layout: default
---

# Proposal 2: Geometry of View-Invariance Across Time

<v-clicks>

## Research Question<sup>1</sup>

- How does identity manifold evolve as object rotates in WM?
- Static representation or predictable rotation?

## Methodology<sup>3</sup>

- Procrustes analysis comparing manifold structure
- Different time steps and rotation speeds
- Link perceptual invariance with temporal dynamics

## Innovation

- Deeper exploration beyond current view-invariance work
- Novel connection between perception and memory dynamics

</v-clicks>

---
layout: default
---

# Proposal 3: Scaling Effects and Representational Abstraction

<v-clicks>

## Unexplored Territory<sup>9</sup>

- Network size effects remain unclear
- Task complexity influence unknown

## Research Questions<sup>1,5</sup>

- Do larger networks show more orthogonalization?
- More "room" in latent space → better separation?
- Or more specialized subspaces?

## Significance

- Clarify if paradoxes are fundamental or scale-dependent
- Guide architecture design principles

</v-clicks>

---
layout: default
---

# Proposal 4: Integrating Laplace Manifolds with Naturalistic Stimuli

<v-clicks>

## Current Limitation<sup>11</sup>

- Laplace Neural Manifolds use abstract/low-dimensional inputs
- Not tested with naturalistic objects

## Ambitious Innovation<sup>1,11</sup>

- Train CNN-RNN with Laplace-like time representation constraint
- Process naturalistic objects with "what × when" binding

## Evaluation

- Compare with standard goal-driven models
- Test if brain's solution is mathematically optimal
- Handle complex environments with many distractors

</v-clicks>

---
layout: default
---

# Proposal 5: Strategy Trajectories and Metabolic Cost

<v-clicks>

## Research Question<sup>19</sup>

- What is metabolic cost of different strategies?
- Does efficiency pressure favor specific strategies?

## Methodology<sup>16</sup>

- Add metabolic penalty to loss function (L1/L2 norm on activity)
- Compare recency vs. target-selective strategies

## Significance

- Connect behavioral differences to biological constraints
- Explain monkey-human discrepancies through efficiency lens

</v-clicks>

---
layout: default
---

# Synthesis of Methodologies

<v-clicks>

## Required Techniques

| Method | Application | Relevance |
|--------|-------------|-----------|
| Procrustes Analysis | Manifold alignment across views/time | Representational stability<sup>3</sup> |
| Linear Decoding | Track information accessibility | What's represented at each step<sup>1</sup> |
| Manifold Perturbation | Inject noise into subspaces | Causal relevance<sup>26</sup> |
| PCA / SVCCA | Compare latent spaces | Shared vs. unique features<sup>5</sup> |
| L1/L2 Regularization | Simulate metabolic efficiency | Biological constraints<sup>16</sup> |

## Multi-Layered Analysis

- Beyond performance metrics
- Uncover principles of cognitive computation
- Explore "hidden world" of latent space geometry

</v-clicks>


---
layout: center
class: text-center
---

# Theoretical Implications

Understanding the computational principles of cognition

---
layout: default
---

# Fundamental Insights

<v-clicks>

## Temporal Organization is Fundamental<sup>1</sup>

- Chronological memory subspaces not a byproduct
- Core feature of cognitive system
- Enables simultaneous tracking with interference protection

## Rethinking Neural Efficiency<sup>5</sup>

- Orthogonalization Paradox challenges assumptions
- "Optimal" code may reduce feature separation
- Enhances high-level relational inference capability

## Path Forward<sup>30</sup>

- Study simpler relational tasks with modern RNNs
- Identify neural implementations of abstract rules
- Bridge computation and cognition

</v-clicks>

---
layout: default
---

# Future Outlook: 2025-2026

<v-clicks>

## Converging Frameworks<sup>2</sup>

- Neural manifold theory
- Laplace transforms
- Traveling waves

## Unified Understanding

- How brain navigates complex, multidimensional world
- From perception to memory to decision

## Opportunity for Master's Students

- Test theories in naturalistic contexts
- High-dimensional, ecologically valid experiments
- Immense potential impact

</v-clicks>

---
layout: default
---

# Conclusion

<v-clicks>

## Foundation Established

- Working memory as dynamic geometric process<sup>1</sup>
- Simultaneous representation of diverse properties
- Specialized temporal subspaces

## Bridging Artificial and Biological

- Metabolic efficiency principles<sup>16</sup>
- Spiking dynamics<sup>22,23</sup>
- Naturalistic stimuli processing<sup>1</sup>

## The Path Ahead

- Causal interventions<sup>26</sup>
- Laplace manifolds<sup>11</sup>
- Cross-modal integration
- Decode the fundamental algorithms of mind

</v-clicks>

---
layout: center
class: text-center
---

# Key Takeaways

<div class="grid grid-cols-2 gap-6 mt-8">

<div>

## 🧠 Theoretical

- Neural manifolds over memory slots
- Temporal organization is fundamental
- Geometry reveals computation

</div>

<div>

## 🔬 Practical

- Naturalistic stimuli essential
- Multiple methodologies needed
- Rich research opportunities

</div>

</div>

---
layout: default
---

# References (1/2)

<div class="text-xs">

1. Lei et al. (2024). Geometry of naturalistic object representations in recurrent neural network models of working memory. NeurIPS.
2. Neural manifold theory and dynamical systems perspective
3. Rotational dynamics and manifold transformations in working memory
4. arXiv:2411.02685
5. OpenReview discussion and critiques
6. Linear readout and manifold decoding
7. Population-level neural coding
8. IBM NeurIPS 2024
9. Scaling effects discussion
10. Neural signatures of associational cortex

</div>

---
layout: default
---

# References (2/2)

<div class="text-xs">

11. Laplace Neural Manifolds: "What" × "When" representations (arXiv:2409.20484)
12. Learning temporal relationships with Laplace manifolds
15. Traveling theta waves in human hippocampus
16. Spatiotemporal perspective on dynamical computation (arXiv:2409.13669v2)
17. Brain-like slot representation
19. Neural dynamics and geometry for transitive inference
21. Spatiotemporal activity generation
22. Working memory in spiking neural networks
23. Timescales of learning in prefrontal cortex (PMC:12616548)
24. Strategy differences: humans vs. monkeys (PLOS Comp Bio)
26. Schema formation and learning-to-learn (PMC:11559441)
30. Neural dynamics for transitive inference

</div>

---
layout: end
class: text-center
---

# Thank You

<div class="pt-12">
  <span class="text-6xl">🧠</span>
</div>

<div class="pt-4 opacity-70">
Questions and Discussion
</div>
