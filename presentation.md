This markdown outline is structured specifically for a PowerPoint presentation. Each level 1 heading represents a new slide, with bullet points provided for slide content and speaker notes for additional depth.

# Slide 1: Title & Overview

**Title:** Geometry of Naturalistic Object Representations in RNN Models of Working Memory **Authors:** Xiaoxuan Lei, Takuya Ito, Pouya Bashivan **Institutions:** McGill University, Mila, IBM Research 

---

# Slide 2: The Core Problem

* 
**Traditional WM Research:** Primarily uses simple "one-hot" categorical inputs (e.g., colored dots).


* 
**The Gap:** Lack of understanding regarding how high-dimensional, naturalistic object information is maintained in Working Memory (WM).


* 
**The Need:** Exploring how networks handle ecologically relevant, multidimensional stimuli across multiple cognitive tasks.



---

# Slide 3: Research Goals

* 
**Goal 1:** Investigate how task-optimized RNNs select relevant properties from naturalistic objects.


* 
**Goal 2:** Understand the computational strategies used to maintain information against incoming distractors.


* 
**Goal 3:** Compare representational geometry across different neural architectures (Vanilla RNN vs. Gated RNNs).



---

# Slide 4: Methodology: Task Design

* 
**9 N-back Task Variants:** 


* **Memory Depth ():** 1-back, 2-back, and 3-back.
* **Features ():** Location (L), Identity (I), and Category (C).


* 
**Stimuli:** 3D object models from ShapeNet.


* 4 Categories, 2 Identities per category, 4 Locations.


* Requires view-invariant and identity-invariant processing.





---

# Slide 5: Methodology: Model Architecture

* 
**Sensory-Cognitive Model:** 


* 
**Stage 1 (Perception):** Pre-trained ResNet50 to derive visual embeddings.


* 
**Stage 2 (Cognition):** Recurrent Neural Network (Vanilla RNN, GRU, or LSTM).




* 
**Training Scenarios:** 


* Single-Task-Single-Feature (STSF).
* Single-Task-Multi-Feature (STMF).
* Multi-Task-Multi-Feature (MTMF).



---

# Slide 6: Finding 1: Simultaneous Encoding

* 
**Observation:** Multi-task RNNs represent both task-relevant and task-irrelevant information simultaneously.


* 
**Evidence:** Decoders could predict irrelevant features (e.g., location during an identity task) with >85% accuracy in MTMF models.


* 
**Conclusion:** Goal-driven networks maintain a "full representation" of objects rather than pruning irrelevant data immediately.



---

# Slide 7: Finding 2: Architecture-Specific Geometry

* 
**Vanilla RNNs:** Utilize "shared" latent subspaces; the way an object is encoded generalizes across different tasks.


* 
**Gated RNNs (GRU/LSTM):** Utilize "task-specific" subspaces; encoding for one task (e.g., 1-back Location) does not generalize to another.


* 
**Impact:** Gated architectures may learn more specialized, less reusable representations.



---

# Slide 8: Finding 3: De-orthogonalization

* 
**Expectation:** RNNs might orthogonalize (separate) features to improve performance.


* 
**Result:** RNNs actually **de-orthogonalize** features compared to the perceptual space.


* 
**Proposed Reason:** Creating a more efficient, lower-dimensional representation that is easier for the "read-out" layer to decode.



---

# Slide 9: Finding 4: Chronological Memory Subspaces

* 
**Hypotheses Tested:** 


* **H1:** Slot-based (independent slots for each item).
* **H2:** Relative Chronological (subspaces based on the "age" of info).
* **H3:** Stimulus-specific.


* 
**Winner:** Data supports **H2** (Chronological Subspaces).


* 
**Meaning:** Networks track information based on *when* it was encoded rather than assigning it to a static "slot".



---

# Slide 10: Technical Deep Dive: De-orthogonalization Math

* 
**Method:** Calculation of decision hyperplanes using Support Vector Classifiers (SVCs).


* 
**Step 1:** Calculate angles between hyperplane normal vectors () using cosine similarity.


* 
**Step 2:** Define Orthogonalization Index (): 




* 
**Result:** .



---

# Slide 11: Technical Deep Dive: Procrustes Analysis

* 
**Goal:** Characterize how information transforms over time within the RNN.


* 
**Mechanism:** Uses a rotation matrix () to align decision hyperplanes from one time step to another.


* 
**Discovery:** Transformations are consistent across different stimuli (shared logic) but are not stable over time (dynamic process).



---

# Slide 12: Discussion & Conclusion

* 
**Resource Model Support:** Findings align with "resource-based" models where memory is flexibly distributed.


* 
**Challenge to Slot Models:** Findings challenge traditional "slot-based" theories of human Working Memory.


* 
**Summary:** RNNs solve complex WM tasks by using dynamic, chronological rotations to separate incoming information from stored memory.



---

# Slide 13: Our Innovation: Task-Guided Attention

* **Motivation:** Original RNN models lack explicit task-selection mechanism
* **Innovation:** Add attention layer between CNN and RNN that modulates visual features based on task context

* **Two Attention Variants:**
  * **Task-Only Attention:** Task embedding guides feature selection
  * **Dual Attention:** Both task embedding and previous hidden state guide attention

* **Research Question:** Does explicit attention improve task-relevant feature selection and generalization?

---

# Slide 14: Attention Architecture

* **Standard Model:** CNN → RNN → Classifier
* **Our Model:** CNN → **Task-Guided Attention** → RNN → Classifier

* **Attention Mechanism:**
  * Query: Task embedding (+ hidden state for dual)
  * Key/Value: Visual features from CNN
  * Output: Task-modulated visual representation

* **Hypothesis:** Attention will learn to suppress task-irrelevant features, improving generalization

---

# Slide 15: Experimental Results

| Model | Train Masked | Val Novel Angle | Val Novel Identity |
|-------|--------------|-----------------|-------------------|
| STSF (Baseline) | 100.0% | 99.9% | 89.2% |
| STMF | 82.7% | 80.0% | 57.6% |
| MTMF | 83.0% | 78.8% | 57.0% |
| **STMF + Attention** | **98.7%** | **92.4%** | **72.6%** |
| **STMF + Dual Attention** | **99.7%** | **91.9%** | **72.4%** |
| **MTMF + Attention** | **99.0%** | **90.1%** | **70.6%** |
| **MTMF + Dual Attention** | **98.9%** | **92.3%** | **73.5%** |

---

# Slide 16: Key Findings from Our Experiments

* **Attention Dramatically Improves Multi-Feature Performance:**
  * STMF: 80.0% → 92.4% on novel angles (+12.4%)
  * MTMF: 78.8% → 92.3% on novel angles (+13.5%)

* **Generalization to Novel Identities Improved:**
  * STMF: 57.6% → 72.6% (+15.0%)
  * MTMF: 57.0% → 73.5% (+16.5%)

* **Dual Attention Slightly Better for MTMF:**
  * Hidden state context helps with multiple N-back levels

---

# Slide 17: Analysis - Why Attention Helps

* **Training Dynamics:**
  * Baseline models plateau at ~88% training accuracy
  * Attention models reach ~99% training accuracy

* **Interpretation:**
  * Attention enables selective feature gating per task
  * Reduces interference between task-irrelevant features
  * Creates more separable representations for the RNN

* **Trade-off:** Slower initial learning (needs ~10 epochs to "discover" attention patterns)

---

# Slide 18: Conclusion & Future Work

* **Conclusion:**
  * Task-guided attention significantly improves multi-feature WM task performance
  * Explicit attention mechanism complements RNN memory dynamics
  * Dual attention provides marginal gains for complex multi-task scenarios

* **Future Work:**
  * Analyze attention weights to understand learned feature selection
  * Compare de-orthogonalization patterns between attention and baseline models
  * Test TransformerFAM architecture for fully attention-based WM

---

*Note: Citation numbers refer to the specific source lines or pages in the provided paper (2411.02685v1.pdf).*