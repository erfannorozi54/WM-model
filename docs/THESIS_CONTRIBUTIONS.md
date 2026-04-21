# Master's Thesis: Novel Contributions to Working Memory Models

**Author**: Erfan  
**Date**: April 2026  
**Base Paper**: arXiv:2411.02685 - "Geometry of naturalistic object representations in recurrent neural network models of working memory"

---

## Overview

This thesis extends the baseline working memory model with task-guided attention mechanisms and investigates two novel research directions:

1. **Meta-Learning for Rapid Task Adaptation** (Primary Contribution)
2. **Continual Learning & Catastrophic Forgetting** (Secondary Contribution)

**Core Innovation**: Demonstrating that attention mechanisms enable both rapid adaptation to new tasks and robust retention of previously learned knowledge.

---

## Contribution 1: Meta-Learning for Rapid Task Adaptation

### Research Question

Can task-guided attention enable few-shot learning of novel working memory tasks?

### Hypothesis

Attention creates separation between:
- **Task-agnostic representations**: General temporal processing (RNN)
- **Task-specific routing**: Feature selection (attention gates)

This enables rapid adaptation: only attention gates update for new tasks, while RNN remains stable.

### Experimental Design

#### Phase 1: Meta-Training

Train on standard N-back tasks:
- **Tasks**: Location, Identity, Category N-back (N ∈ {1, 2, 3})
- **Data**: ~30,000 sequences
- **Goal**: Learn general WM processing + task-specific attention patterns

#### Phase 2: Meta-Testing (Few-Shot Adaptation)

Test on novel tasks with minimal training data (K ∈ {10, 25, 50, 100, 500} examples):

**Novel Task 1: Pattern Detection - "Three-in-a-Row"**
- Detect when same feature value appears three consecutive times
- Task vector: All three N-elements set to 1 (signals pattern mode)
- Feature elements specify which feature to track (location/identity/category)
- Example: [loc0, loc0, loc0, loc1, ...] → match at position 3
- **Challenge**: Requires sliding window comparison, not fixed N-back offset

**Novel Task 2: Extended N-back (N=4)**
- Standard N-back with N=4 (not seen during training)
- Tests generalization to longer temporal distances
- **Challenge**: Probes capacity limits and whether model learned generalizable computation

**Novel Task 3: Alternating Feature Task** (Optional)
- Alternate between two features every timestep
- Requires dynamic attention switching within sequence
- **Challenge**: Tests flexibility beyond static task-level gating

#### Adaptation Strategies

**Strategy A: Train New Model from Scratch**
- Random initialization, train on novel task
- **Purpose**: Baseline without transfer learning

**Strategy B: Full Fine-Tuning**
- Update all parameters on novel task
- **Purpose**: Standard transfer learning

**Strategy C: Attention-Only Fine-Tuning** (Primary)
- Freeze perceptual (ResNet50) and cognitive (RNN) modules
- Update only attention gates
- **Rationale**: Visual features and temporal processing already learned; only feature selection needs adaptation

**Strategy D: Attention + Classifier Fine-Tuning**
- Freeze perceptual and cognitive, update attention + classifier
- **Purpose**: Middle ground between flexibility and stability

**Strategy E: Progressive Attention**
- Add new parallel attention module for novel task
- Keep original attention frozen
- **Purpose**: Zero catastrophic forgetting, test compositionality

**Strategy F: MAML-Style Meta-Learning**
- Optimize initialization for fast adaptation during meta-training
- **Purpose**: Best few-shot performance

### Key Metrics

**Primary**:
- **Few-Shot Accuracy**: Test accuracy after K examples
- **Sample Efficiency**: Examples needed to reach 80% accuracy
- **Adaptation Speed**: Accuracy improvement per training step

**Secondary**:
- **Catastrophic Interference**: Performance drop on meta-training tasks
- **Transfer Efficiency**: (Acc_adapted - Acc_random) / (Acc_full - Acc_random)

### Key Analyses

**A. Learning Curves**: Compare adaptation trajectories across strategies
- Expected: Attention-only reaches 75-85% with 50 examples; baseline needs 500+

**B. Attention Gate Analysis**: Visualize gate changes during adaptation
- Expected: Only 20-40% of channels change; patterns are task-specific

**C. Representational Similarity**: Measure RNN vs attention-gated representations
- Expected: RNN stable across tasks; attention creates task-specific transformations

**D. Layer-Wise Adaptation**: Track which components change
- Expected: Attention adapts in 1-10 steps; RNN barely changes

### Expected Outcomes

**Quantitative**:
- Attention-only: 75-85% accuracy with 50 examples
- Full fine-tuning: 65-75% with 50 examples
- Train from scratch: 50-60% with 50 examples, needs 500+ for 75%

**Qualitative**:
- Clear task-specific gate signatures
- RNN maintains stable temporal processing
- Minimal catastrophic forgetting (<5% drop)

**Theoretical**: Demonstrates attention enables compositional learning by separating task-agnostic processing from task-specific control.

---

## Contribution 2: Continual Learning & Catastrophic Forgetting

### Research Question

Does task-guided attention mitigate catastrophic forgetting when learning new object categories?

### Hypothesis

Attention reduces forgetting through:
1. **Representational Orthogonalization**: Different categories use different channels
2. **Selective Plasticity**: Category-specific gates enable independent learning
3. **Stable Temporal Processing**: RNN provides general-purpose temporal dynamics

### Experimental Design

#### Continual Learning Protocol

**Phase 1: Initial Training**
- Categories: Airplane, Car
- Tasks: Location, Identity, Category N-back (N ∈ {1, 2, 3})
- Training: 15,000 sequences

**Phase 2: Category Expansion**
- Add: Chair, Table
- Training: 15,000 sequences (50% new, 50% old for replay methods)
- Evaluate: Both old and new categories

**Phase 3: Further Expansion** (Optional)
- Add more categories or novel tasks
- Test cumulative forgetting

#### Training Strategies

**Strategy A: Naive Fine-Tuning** (Baseline)
- Train on new categories only, no forgetting prevention
- Expected: 30-50% forgetting

**Strategy B: Train New Model from Scratch**
- Separate model for new categories
- Expected: 0% forgetting but not continual learning

**Strategy C: Experience Replay**
- Mix 50% old data, 50% new data
- Expected: 10-20% forgetting, requires data storage

**Strategy D: Elastic Weight Consolidation (EWC)**
- Penalize changes to important weights
- Expected: 15-25% forgetting, no data storage

**Strategy E: Attention-Guided Consolidation** (Primary)
- Learn category-specific attention gates
- Freeze old gates, learn new gates for new categories
- Expected: <10% forgetting, no data storage

**Strategy F: Progressive Neural Networks**
- Add new columns for new categories
- Expected: 0% forgetting but model grows

### Key Metrics

**Primary**:
- **Backward Transfer (BWT)**: Performance change on old tasks after learning new ones
- **Forward Transfer (FWT)**: How well old knowledge helps new tasks
- **Forgetting Rate**: Accuracy drop per category

**Secondary**:
- **Average Accuracy**: Overall performance across all tasks/categories
- **Memory Efficiency**: Storage requirements

### Key Analyses

**A. Forgetting Curves**: Track accuracy per category across phases
- Expected: Attention shows flat curves for old categories; baseline shows decline

**B. Representational Drift**: Measure how representations change across phases
- Expected: Attention shows small drift; baseline shows large drift

**C. Subspace Overlap**: Measure angles between category subspaces
- Expected: Attention creates orthogonal subspaces; baseline shows high overlap

**D. Attention Gate Specialization**: Analyze category-specific gate patterns
- Expected: Clear category signatures; old gates remain stable in Phase 2

**E. Interference Analysis**: Measure gradient conflicts between old and new tasks
- Expected: Attention shows low conflict; baseline shows high conflict

### Expected Outcomes

**Quantitative**:
- Naive: -30% to -50% BWT (severe forgetting)
- Replay: -10% to -20% BWT
- EWC: -15% to -25% BWT
- Attention: -5% to -10% BWT (minimal forgetting)

**Qualitative**:
- Attention creates four orthogonal category subspaces
- Category-specific gate signatures are interpretable
- Old representations remain stable

**Theoretical**: Demonstrates that geometric structure (orthogonality) is key to continual learning, not just regularization or replay.

---

## Integration: Meta-Learning + Continual Learning

### Unified Research Question

Can a single attention mechanism support both rapid adaptation and robust retention?

### Unified Protocol

1. **Meta-train** on Location, Identity, Category tasks (4 categories)
2. **Few-shot adapt** to Three-in-a-row task (50 examples)
3. **Add categories** (2 new categories) to all tasks
4. **Few-shot adapt** to N=4 task (50 examples)
5. **Evaluate** all tasks on all categories

### Key Metric

**Compositional Generalization**: Performance on novel task + novel category combinations (e.g., Three-in-a-row on new category)

### Expected Synergy

- **Fast adaptation**: 75%+ accuracy with 50 examples
- **Minimal forgetting**: 90%+ retention of original performance
- **Compositional generalization**: Successfully combine learned tasks and categories

**Theoretical Unification**: Both capabilities emerge from attention-based routing:
- Meta-learning = rapid gate adaptation
- Continual learning = orthogonal gate allocation

---

## Timeline

**Month 1**: Setup & baselines (train standard models)
**Month 2**: Meta-learning experiments (few-shot adaptation)
**Month 3**: Continual learning experiments (sequential category learning)
**Month 4**: Integration & writing (combined experiments, thesis writing)

---

## Expected Contributions

### To Machine Learning
1. First application of meta-learning to working memory models
2. Novel attention-based continual learning approach
3. Single mechanism supporting both adaptation and retention

### To Neuroscience
1. Computational model of PFC gating in working memory
2. Explanation for rapid task switching in humans
3. Account of how brain avoids catastrophic forgetting

### To Thesis
1. Two major experimental contributions
2. Comprehensive analysis pipeline
3. Clear theoretical framework linking attention to cognitive flexibility

---

## Key References

- **Base Paper**: arXiv:2411.02685 - Geometry of naturalistic object representations in RNN models of WM
- **Meta-Learning**: Finn et al. (2017) - MAML; Santoro et al. (2016) - Memory-Augmented Neural Networks
- **Continual Learning**: Kirkpatrick et al. (2017) - EWC; Zenke et al. (2017) - Synaptic Intelligence
- **Neuroscience**: Miller & Cohen (2001) - PFC function; Baddeley (2003) - Working memory

---

**End of Document**
