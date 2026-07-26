# Presenter Transcript
## Working Memory Shapes Neural Geometry in Human EEG Over Learning
### Wojcik et al., 2025

---

## Slide 1 — Title

> Today I'll present the paper "Working Memory Shapes Neural Geometry in Human EEG Over Learning" by Wojcik and colleagues, published as a bioRxiv preprint in 2025. The central question of this paper is: does working memory simply store sensory information, or does it actively transform information into a more efficient representation for future decisions? This connects directly to a growing body of work — including computational models like the NeurIPS 2024 RNN paper — that argues working memory is a computational resource, not just a temporary buffer.

---

## Slide 2 — Why This Paper Matters

> The traditional view of working memory is that it's a passive storage system — you see something, you hold it briefly, and you retrieve it. But the modern view, supported by recent theoretical and empirical work, is that working memory is actively computational. Information is transformed while it's being maintained. This paper draws on four key theoretical frameworks:
>
> **1. Memory as computational resource.** This idea comes from Dasgupta and Gershman (2021) and others. The core insight is that working memory adapts computations to current task demands. Pre-computed information can be stored in working memory, reducing the computation time at the moment of decision. In other words, working memory isn't just holding data — it's holding partially processed answers.
>
> **2. Neural geometry.** This is the idea that we can understand what the brain represents by looking at the geometry of neural population activity. If different conditions are represented as points in a high-dimensional space, the structure of those points — how they cluster, how far apart they are, how many dimensions they occupy — tells us about the underlying computation. Low-dimensional geometry means the brain has compressed the representation to only task-relevant distinctions.
>
> **3. Dynamic coding.** This framework, developed by Stokes and colleagues (Stokes, 2015; Spaak et al., 2017), proposes that working memory representations are not static. Instead, the same neural population can dynamically switch between encoding different variables at different moments in time. This is sometimes called "activity-silent" working memory — the information is stored in synaptic weights, not in ongoing neural firing.
>
> **4. Temporal decomposition of computation.** This comes from computational modeling by Ehrlich and Murray (2022). The idea is that complex computations — like XOR — can be broken down into simpler steps that happen at different times. Instead of computing everything at the moment of decision, the brain pre-computes one variable during the delay, then combines it with the second variable at decision time. This reduces the computational load when it matters most.
>
> What makes this paper novel is that they test all of these ideas in humans using EEG, which gives us millisecond-level temporal resolution to track how representations change over the course of a single trial.

---

## Slide 3 — Main Hypothesis

> The core hypothesis is shown in this flow diagram. The authors predict that sensory input enters working memory, where it's transformed into an abstract context representation. This abstract context then supports a low-dimensional decision code. In other words, instead of remembering exact sensory details — like the specific color of a stimulus — working memory should retain only the information that's useful for future decisions. This is a strong claim: it says the brain is not a camera, it's a compressor.

---

## Slide 4 — Task Design

> Let me walk you through the task. Participants see a color cue, then there's a delay period, then a shape appears, and they must respond. The correct response depends on the XOR of the color and the shape. XOR is a classic nonlinear problem — you can't solve it with a simple linear combination. Participants had to learn this rule by trial and error. The key insight is that there are two possible strategies. Strategy 1 is memorization: you just remember all four combinations explicitly. Blue-square means left, blue-diamond means right, and so on. This is high-dimensional. Strategy 2 is to extract a context: you convert the four colors into two contexts (Blue/Pink → Context 1, Green/Khaki → Context 2), store the context, and then compute the XOR from context plus shape. This is low-dimensional. The authors predict that humans will adopt Strategy 2.

---

## Slide 5 — Context vs Color

> This slide highlights the critical distinction. The first stimulus — the color cue — contains both irrelevant information (the specific color identity: blue, green, pink, khaki) and relevant information (the abstract context: Context 1 or Context 2). The question the paper asks is: what does working memory actually store? Does it store the raw color, or does it store the abstract context? If it stores context, that means the brain has already performed a computation — it has extracted the task-relevant variable and discarded the rest.

---

## Slide 6 — Behavioral Evidence

> First, let's look at behavior. Participants learned rapidly: accuracy went from 75% in Stage 1 to 96% in Stage 4. But the more telling result is what happened with context switches versus shape switches. When the context changed between trials, participants were slower and less accurate than when the shape changed. This is the signature of context being the important variable. If participants were just memorizing color-shape pairs, context switches shouldn't be harder than shape switches. The fact that they are tells us that participants are treating context as the key variable to track.

---

## Slide 7 — EEG Decoding Framework

> Now let me explain the analytical framework. The authors used linear decoders — essentially, they trained classifiers on the EEG signal to decode different task variables: context, color, shape, XOR, and motor response. By training these decoders at each time point, they could track when each type of information is represented in the brain. This is the standard approach in the neural decoding literature, and it allows them to ask: at what moment does the brain represent context versus color versus the XOR rule?

---

## Slide 8 — Result 1: Context Is Maintained

> Here's the first key result. On the left, we see context decoding across time. The x-axis shows time from color onset, with vertical dashed lines marking the color period, the delay period, and the shape period. The black line shows decoding accuracy, and the horizontal bars at the bottom indicate time points where decoding is significantly above chance. Notice that context decoding stays above chance throughout the entire delay period — from color onset all the way through to shape onset. On the right panel, we see that delay-locked context decoding increases with learning stage, with a significant asterisk at Stage 4. This tells us that working memory actively maintains context information throughout the delay.

---

## Slide 9 — Result 2: Color Is Discarded

> Now look at what happens with color decoding. On the left, color decoding is strong immediately after the cue appears — you can see the sharp peak at time zero. But it rapidly drops to chance before the delay period even begins. On the right, delay-locked color decoding shows no significant effect across learning stages — the "ns" means not significant. This is striking. The brain initially encodes the color, but then quickly discards it. It keeps the context — the abstract, task-relevant variable — and throws away the color identity. This is exactly what the selection-and-compression hypothesis predicts.

---

## Slide 10 — Result 3: XOR Representation Emerges

> The third result concerns XOR decoding. On the left, XOR decoding is at chance during the color and delay periods, but then sharply increases after the shape appears. This makes sense: the XOR rule can only be computed once you have both the context (maintained from the color) and the shape. On the right, shape-locked XOR decoding increases significantly with learning. This is important because the XOR signal is not present in the stimulus itself — it must be computed by the brain. The fact that it emerges after shape presentation, and strengthens with learning, shows that the brain is actively constructing the task rule.

---

## Slide 11 — Result 4: Context Becomes Abstract

> The fourth result addresses whether context becomes abstract. The authors used a cross-generalization analysis: they trained a decoder on Blue versus Green (two colors that share Context 1 versus Context 2), and then tested it on Pink versus Khaki — colors the decoder has never seen. If the decoder succeeds, it means the context representation is independent of the specific color. On the left, cross-generalized context decoding is above chance during the delay period. On the right, we see that this cross-generalized decoding increases significantly with learning. This confirms that context becomes an abstract variable — it's not tied to any particular color, but represents the underlying task structure.

---

## Slide 12 — What Is Neural Geometry?

> Let me now introduce the concept of neural geometry. Imagine each task condition — blue-square, blue-diamond, green-square, and so on — as a point in a high-dimensional neural space. In a high-dimensional geometry, all of these conditions are represented separately. But in a low-dimensional geometry, only the task-relevant distinctions remain: Context 1 versus Context 2, XOR-True versus XOR-False. The irrelevant distinctions have been compressed away. The question is: does learning change this geometry? Does the brain move from representing all sensory details to representing only the abstract, task-relevant variables?

---

## Slide 13 — Result 5: XOR-Dominated Representations

> The answer is yes. Looking at neural geometry immediately before the response, the authors found that XOR decoding increases from 0.516 to 0.572, and abstract XOR coding increases from 0.512 to 0.560. The figure on the right shows this clearly: in Stage 1 (grey), context, shape, and XOR are all decoded at similar levels. But by Stage 4 (black), XOR is decoded significantly better than the other variables, with a triple asterisk indicating strong significance. The representation has reorganized itself around the final decision variable — not around the sensory details. This is a geometric transformation driven by learning.

---

## Slide 14 — Dimensionality Does Not Decrease

> Here's the surprising finding. The authors expected that learning would reduce dimensionality — that the neural representation would become lower-dimensional as participants learned the task. But that's not what they found. Looking at shattering dimensionality — a measure of how many linearly separable distinctions the neural population supports — they found that low dimensionality already exists from Stage 1. For correct trials (green line), there's no significant change across learning stages. The compression happens immediately, not gradually. The authors' interpretation is that the working memory delay forces participants to compress color into context from the very start — the delay itself is the mechanism that triggers compression.

---

## Slide 15 — Critical Result: The Correlation

> This is the strongest analysis in the paper. The question is: does stronger context maintenance predict lower-dimensional decision representations? The answer is yes, with a correlation of r = -0.37, p = 0.04. Looking at the left scatter plot (Stage 1), each dot is a participant. The x-axis is shattering dimensionality at the decision, and the y-axis is context maintenance during the delay. Participants who maintained context more strongly had lower dimensionality at decision time. On the right (Stage 4), this correlation disappears — r = 0.08, not significant. This means the relationship between context maintenance and dimensionality is strongest early in learning, when participants are first figuring out the task.

---

## Slide 16 — Computational Interpretation

> Let me put this all together. Working memory performs two operations: selection and compression. When a participant sees Blue, the brain maps it to Context 1. When they see Pink, the brain also maps it to Context 1. Blue and Pink are different colors, but they're the same context. By mapping both to the same context, the brain reduces unnecessary distinctions — it compresses the representation. This creates a more efficient neural geometry where only task-relevant variables remain.

---

## Slide 17 — Relation to NeurIPS 2024 RNN Paper

> This connects directly to the NeurIPS 2024 paper by Lei, Ito, and Bashivan, which showed that RNN models of working memory transform neural geometry to create low-dimensional representations. In that computational work, the RNN learned to compress high-dimensional input into task-relevant features. What Wojcik et al. show is that the same geometric transformation occurs in biological brains — human participants performing an analogous task. Both lines of work converge on the same message: working memory is not storage, it's geometry transformation.

---

## Slide 18 — Main Conclusions

> Let me summarize the six key conclusions. First, working memory stores context, not sensory details. Second, irrelevant color information is actively discarded. Third, context becomes increasingly abstract over learning. Fourth, XOR representations emerge through learning — the brain computes the task rule. Fifth, stronger context maintenance predicts simpler, lower-dimensional neural geometry. And sixth, working memory acts as a computational resource that compresses information for future decisions. The overarching message is that working memory is a geometry transformation engine.

---

## Slide 19 — Thank You

> Thank you for your attention. The paper is Wojcik et al., 2025, "Working Memory Shapes Neural Geometry in Human EEG Over Learning." I'm happy to take questions.
