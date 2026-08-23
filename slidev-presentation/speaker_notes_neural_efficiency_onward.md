# Speaker Notes: Neural Efficiency Through End

These notes cover every slide from the **"Neural Efficiency"** section title through the **"Thank You"** slide. For each slide: what to say, what to emphasize, and what to be careful about.

---

## Slide: Neural Efficiency — Section Title

> *"A Second, Independent Finding"*

**Say:**
> "So far we've covered the paper's five analyses and our attention-mechanism contribution. Now I want to present something entirely separate — a second, independent finding that grew out of a question our professor asked us: beyond just showing better accuracy, can we point to an observable working-memory *phenomenon* in these models? That question led us down a path that connects our work directly to the human neuroscience of neural efficiency."

**Emphasize:** This is a *new finding*, not a restatement of anything earlier in the talk. Signal the transition clearly.

---

## Slide: The Claim We're Testing

**Say:**
> "Let me be precise about what we already showed. Proxy pretraining raises accuracy — novel angle from 83 to 98 percent, novel identity from 81 to 93 percent. But that's a *performance* result at the same N-back levels. We never tested whether the model can hold more items or succeed at higher N. So calling it a 'capacity increase' would be an over-claim."

> "Human working-memory research distinguishes capacity from efficiency. Capacity means holding more items. Efficiency means doing the same or better work with *suppressed* neural response — the same signature seen for stimulus repetition, now also shown for prior knowledge."

> "Our claim under test is: familiarity and structure — from proxy pretraining — and explicit gating — from attention — both suppress task-irrelevant processing. And we test this at three independent levels of the model, using real prior findings from human WM research as the standard to compare against."

**Emphasize:**
- The distinction between performance (accuracy) and efficiency (suppressed activity). This is the conceptual pivot of the entire section.
- That this answers a specific ask — not just "did better," but "show me an observable phenomenon."

**Be careful about:** Do not let the audience conflate the accuracy gain with the efficiency finding. They are separate claims. Say so explicitly.

---

## Slide: Our Method

**Say:**
> "We organized the test into three levels, each measuring something different but all targeting the same underlying idea — that familiarity and structure suppress what's not needed."

> "Level 1 is representational content: can you still decode task-irrelevant features from the hidden state? If attention is suppressing them, decodability should drop. We compare baseline versus attention models."

> "Level 2 is population activity: hidden-state magnitude, participation ratio, sparsity, and a Fano-factor analogue. If proxy pretraining produces familiarity-based suppression, these metrics should shift — and we compare baseline versus proxy-pretrained models across two independent pairs."

> "Level 3 is explicit gating: the attention model has literal per-channel gates between zero and one. A gate near zero on a task-irrelevant channel *is* suppression by construction, not a proxy for it. We compare attention-only versus attention-plus-proxy."

> "Two design principles run through all three. First, we report all three levels rather than cherry-picking the one that looks best — convergent evidence across independent tests is stronger than any single metric. Second, every comparison reports the accuracy gap between conditions, so nobody can dismiss an activity difference as 'the model just got more accurate.'"

**Emphasize:**
- The three-level structure. Say "three independent levels" more than once.
- The matched-accuracy design. This is the methodological contribution that protects against the most obvious confound.

---

## Slide: Reference 1 — The Question They Asked

**Say:**
> "For Level 2, our direct precedent is a 2016 fMRI paper by Poppenk, Moscovitch and McIntosh. Let me start with the background everyone already half-knows: when you see something a second time, the brain region that processes it usually responds *more weakly* — as if it's not working as hard because it already recognizes it. That's called repetition suppression, and it's one of the best-established findings in cognitive neuroscience."

> "But here's the gap nobody had closed: almost every prior study only tested this a few *minutes* after the first exposure. Nobody had asked — if you already know something well, from years of everyday exposure rather than a recent viewing, does just *seeing it* produce that same 'worked less hard' signal?"

> "Their hypothesis was simple: if suppression really reflects 'the brain already has relevant information available,' it shouldn't matter *how* that information got there. A proverb you saw thirty minutes ago and a proverb you've known your whole life should look the same to the brain."

**Emphasize:**
- This is a *question*, not yet a result — resist the urge to jump ahead to the finding. Let the setup land first.
- The gap being closed: prior work only ever tested minutes-old repetition, never lifelong familiarity.

---

## Slide: Reference 1 — What They Actually Did

**Say:**
> "Eighteen people sat in an MRI scanner reading proverbs. There were three kinds, all matched for length and difficulty. First, brand-new Asian proverbs, translated, that nobody had seen before. Second, different Asian proverbs shown three times about thirty minutes earlier in that same session — recent repetition. Third, common English proverbs — things like 'the early bird catches the worm' — known from a lifetime of everyday exposure, but never shown earlier in the experiment itself."

> "While people read and rated each proverb, the scanner measured how much brain activity dropped for the recently-repeated and the known-for-a-lifetime proverbs, each relative to the novel ones. Then the key comparison: are those two drop-off patterns the same regions and the same size, or different?"

> "One important design point: participants obviously couldn't have 'recently repeated' the English proverbs — they'd known them for years. So any similarity between the two suppression patterns can't be explained by recent exposure. It has to come from familiarity itself."

**Emphasize:**
- The three proverb types and why English proverbs specifically rule out a "recent exposure" explanation.
- This is a controlled comparison, not just two unrelated observations — that's what makes it interpretable.

---

## Slide: Reference 1 — What They Found

**Say:**
> "Across a broad network of vision and language brain regions, recently-repeated and known-for-a-lifetime proverbs produced statistically indistinguishable suppression — same regions, same strength, confirmed by a multivariate conjunction analysis with a correlation of 0.65. Only two small regions broke that pattern, and they broke it in a way that makes sense — they showed suppression only for recent repetition, consistent with tracking recent-episode memory specifically, not general familiarity."

> "In one line: knowing something well quiets the brain the same way seeing it twice does."

> "Why this matters for us: if prior knowledge suppresses neural response in humans regardless of how that knowledge was acquired, then our proxy-pretrained model — which acquires knowledge from a *different* task — should show the same signature in its hidden-state activity. That's exactly what we test at Level 2."

**Emphasize:**
- The one-line summary — that's the memorable takeaway to leave with the audience.
- That this paper was read in full, not skimmed from an abstract.
- The conceptual bridge to our work: our proxy model's structured features are the analogue of "prior knowledge," not "repetition."

---

## Slide: Reference 2 — The Pattern Across Studies

**Say:**
> "Our second reference is Constantinidis and Klingberg, 2016, in Nature Reviews Neuroscience. Flag up front that this is a *review*, not a single experiment — they read and synthesised dozens of monkey electrophysiology and human imaging studies of WM training and pulled out the pattern that repeats across all of them. That's a deliberate choice, not a weaker citation: for this claim we wanted the well-replicated general pattern, not one lab's single result."

> "Two things happen together after training. First, more prefrontal neurons get involved — more of them become active during the task, and they fire more overall. Second, and this is the counterintuitive part, each individual neuron gets *less* picky — more broadly tuned, less selective. It's not that neurons become sharper specialists. The job spreads across a wider crew, each doing a less narrowly-defined part."

> "At the same time, the population as a whole gets more reliable. Trial to trial, the same neuron's firing becomes less erratic — that wobble is called the Fano factor — and neurons stop making the same noisy mistakes together, which is a drop in noise correlation."

> "Put together: efficiency after training isn't simply 'less activity.' It's a reorganization — broader per-neuron tuning, more neurons recruited, and a calmer, less noisy population. That's the specific, testable prediction we carry into our own Level 2 results."

**Emphasize:**
- The counterintuitive part — broader tuning, not sharper — is worth slowing down for. It's easy to assume "more efficient" means "more precise," and this paper says otherwise at the single-neuron level.
- This sets up the specific prediction (PR down, Fano down) that our results will be graded against.

---

## Slide: Reference 2 — A Warning We Adopted (Box 2)

**Say:**
> "This paper also gives us a methodological warning — Box 2 — that we took seriously. fMRI's brain-activity signal, the BOLD signal, is a blurry, indirect proxy. It cannot distinguish 'this region is genuinely processing the task more efficiently' from 'the person is simply less engaged' or 'getting more of it wrong.'"

> "The rule this forces: you cannot read 'activity went down' as 'got more efficient' without first checking that performance is genuinely comparable between the two things you're comparing. A quieter signal that also performs worse is not evidence of efficiency."

> "That's exactly why every comparison in our method reports the accuracy gap between conditions — and why we specifically re-ran our Level 2 comparison at a near-zero accuracy gap, instead of trusting only the first pair, where the two models also differed a lot in accuracy."

**Emphasize:**
- Box 2 by name. This is the reason the matched-accuracy design exists — say it plainly.
- That this review licenses exactly **one** directional prediction for us: Fano down. Not magnitude (§4 has firing rate going *up* — that prediction is Ref 1's), and not participation ratio (a population measure, where the review's finding is about single-unit tuning). We report honestly when our Fano result goes the opposite way.

---

## Slide: The Grading Contract

**Say:**
> "Before I show any result, I want to fix how each metric gets graded — because an earlier version of this chapter graded three of four metrics against the wrong reference, and I'd rather show you the corrected contract than quietly fix it in the background."

> "Magnitude is graded against Reference 1, Poppenk. Reference 2 actually predicts the opposite — its single-neuron section has firing rate going *up* after training. The Fano analogue is the one metric graded against Reference 2, and it's the one clean directional prediction that review licenses. Participation ratio is graded against nothing: it's a *population* dimensionality measure, while Reference 2's tuning finding is a single-unit property — and the review reports tuning getting *broader*, not sharper. Sparsity is our own assumption; neither paper predicts it. And the gate-suppression index needs no reference at all, because the gates are a literal suppression signal we read off directly."

> "The reason to state this first: a metric can only confirm or contradict a prediction that was actually made. Deciding the grading up front is what stops a result from being retro-fitted to whichever reference it happens to agree with."

**Emphasize:**
- Only **one** of five metrics is graded against Reference 2 — and it's the one we contradict. Say that plainly; it's the honest structure of the chapter.
- If asked why the grading changed: the old PR prediction was derived from "sharpened tuning," but the review says tuning broadens, and PR is a population measure anyway. Withdrawing an unsound prediction is not the same as hiding a result — the PR result is still on the next slides.

---

## Slide: Results — Level 1: Representational Content

**Say:**
> "Level 1 compares the baseline GRU against the attention GRU on the MTMF config, decoding identity. Before the numbers, one plain-language anchor for the whole table: 'decodability' means — if you trained a simple classifier to guess the object's identity just from the model's hidden state, how often would it succeed? Lower is what we want here, because identity is task-irrelevant — lower decodability means the model is hiding it better, i.e. suppressing it. The other three sub-metrics probe the same idea from different angles — how separated the representations are, how similarly-shaped the space is, how robust the encoding is to swaps."

> "The results are mixed, leaning supportive — and I'll be honest, this is the weakest of the three levels."

> "We run this separately for the two task contexts where identity is genuinely irrelevant, because pooling them — which the first pass did — mixes in the identity task, where identity is the feature the model is *supposed* to keep."

> "Separated, the result is much more interesting than the pooled version. Under the location task, identity decodability drops from 28, 18, 15 percent to 5, 2, 5 percent — chance is about 3 percent, so attention drives it essentially to the floor. Under the category task, it doesn't move at all: 20, 16, 16 becomes 15, 19, 17."

> "So attention suppresses irrelevant identity almost completely in one task context and not at all in another. The pooled run reported a uniform 'roughly halved', which was the average of those two very different things."

> "Orthogonalization is flat in both contexts, at ceiling. Procrustes reconstruction is actually lower under attention."

> "You'll see the swap-test row struck out. We removed it on inspection: that test decodes *location*, not identity — deliberately, because identity labels are unique per trial and can't be aligned across the two stimulus groups the test needs. It was being reported under 'property: identity', which made it look like evidence about identity suppression. It isn't, so we don't count it."

> "Bottom line: a real, strong, but task-dependent suppression effect — sharper than what we reported before filtering, not weaker."

**Emphasize:**
- Anchor the audience with the one-sentence "what is decodability" definition before showing numbers — the table is unreadable without it.
- State the chance level out loud. It's the discipline this project adopted after the earlier audits.
- The struck-out row is a *strength*, not an embarrassment — we checked what the metric actually computed rather than trusting its label.
- That you're reporting this honestly, graded, not as a uniform win.
- The distinction between "flat at ceiling" and "contradictory." These are not failures — they're limitations of the metric in this config.

---

## Slide: Results — Level 2: Population Activity

**Say:**
> "Level 2 is where things get interesting, and also where we have to be the most careful. Quick definitions before the table, because otherwise the four metric names are just jargon: 'magnitude' is how loud the hidden-state signal is on average. 'Sparsity' is what fraction of hidden units are actually doing something for a given input — higher means fewer units firing at once. 'Participation ratio' is roughly how many independent patterns the population is using — higher means information is spread across more directions instead of squeezed into a few. And 'Fano-factor analogue' is the same trial-to-trial wobble idea from Reference 2, just computed on our model's units instead of real neurons."

> "We ran two independent pairs. Pair 1 is baseline versus proxy-finetuned with a 10-percentage-point accuracy gap — that's the unmatched comparison. Pair 2 is attention-only versus attention-plus-proxy with a *0.08-percentage-point* accuracy gap — that's our clean, matched-accuracy replication."

> "Same direction in both pairs, in all eighteen cells. Activation magnitude is lower under proxy pretraining — that matches *Reference 1*, Poppenk. I want to be precise about the attribution: Reference 2 actually reports the opposite for single neurons, more cells recruited and firing rate going up, so it would be wrong to put the magnitude result in a 'versus Reference 2' column. Population sparsity is higher in 17 of 18 cells, but neither paper predicts sparsity — that one is our own assumption, and I'm labelling it as such."

> "Participation ratio is higher in every cell. An earlier version of this deck graded that as contradicting Reference 2. We withdrew that: PR measures the effective dimensionality of the *population*, while the review's tuning finding is about *single neurons* — and the review reports tuning getting *broader*, not sharper. Those are different quantities, so the review licenses no PR prediction at all. The effect is real, it's just ungraded."

> "And it is real — we checked whether it was a sample-size artifact, because PR is biased upward by trial count. It isn't: in 11 of the 18 cells the proxy condition has *fewer* trials and still shows higher PR, and in one cell with exactly equal trial counts, 258 against 258, PR still rises 76 percent."

> "That leaves one genuine contradiction: the Fano-factor analogue is higher in every cell, and Reference 2 cleanly predicts lower. It's actually understated. Var-over-mean scales with activity level, and the proxy condition is *quieter* — so a pure scale effect would have pushed this metric *down*. It went up anyway, which means the scale-invariant version of the same measure moves further still."

> "The critical control: the magnitude effect *replicates at near-zero accuracy gap* in Pair 2. It survives the Box 2 confound check. It's not just 'the proxy model is more accurate.'"

**Emphasize:**
- The two-pair design. Pair 1 is the unmatched comparison, Pair 2 is the confound check. Say this clearly.
- The corrected attribution. Magnitude is graded against Ref 1, sparsity against our own assumption, PR against nothing. Only Fano is graded against Ref 2 — and only Fano contradicts it.
- If asked why the PR grading changed: the prediction was derived from "sharpened tuning," but the review says tuning gets broader, and PR is a population measure anyway. Withdrawing an unsound prediction is not the same as hiding a result — the result stays on the slide.
- That the finding is *nuanced*: lower magnitude and sparser, yes — but higher-dimensional and more variable, not the simple "sharpening" story. A genuine finding in its own right, not a failure.

---

## Slide: Results — Level 3: Explicit Gating (Headline)

**Say:**
> "This slide used to be my headline result, and I'm going to tell you why it isn't any more — because I think how we found the problem matters more than the number we lost."

> "Two quick definitions. The 'gate' is the attention mechanism's literal per-channel dial. The 'suppression index' is how much lower the gate sits on task-irrelevant channels than task-relevant ones — more negative means it mutes irrelevant features more strongly."

> "The original run compared attention-only against attention-plus-proxy and found the proxy model gated more sharply in nine cells out of nine, with attention-only barely gating at all. That's a big, clean-looking result. But that run pooled every saved checkpoint of both models — about 45 each. The attention-only model was trained from scratch, so its pool included checkpoints from near initialisation. The proxy model was fine-tuned from a pretrained network and was already converged at its first epoch. So I was partly comparing a half-trained model against a trained one."

> "When we pin both to the accuracy-matched epoch pair — 43 against 1 — the result changes. It's six cells out of nine, not nine, and the gaps are small. Attention-only already gates strongly on location and category, around minus 0.42 to minus 0.52. The claim that it 'barely gates' was an artifact of averaging in its untrained checkpoints."

> "What does survive: the gate-relevance correlation improves consistently under proxy pretraining — 0.66 to 0.73 becoming 0.84 to 0.85 for location. And a genuinely interesting negative: *neither* model gates on identity. Both sit near zero or wrong-signed there."

**Emphasize:**
- Lead with the retraction. Do not let the audience discover it in questions.
- The mechanism of the confound — training maturity, not the intervention — in one sentence. It's the most transferable lesson in the talk.
- That the gates being a *literal* signal is what made the confound detectable at all.
- If asked why you re-ran: an audit found `epoch_a` and `epoch_b` were null in the output. The check took a minute; the result it overturned was the chapter's headline.

**Be careful about:** do not soften this into "the effect was smaller than we thought." The specific claims — 9/9 cells, attention-only barely gating — are withdrawn.

---

## Slide: Neural-Efficiency Chapter: Conclusion

**Say:**
> "Let me grade each level, in the order the evidence now supports rather than the order I originally expected."

> "Level 2, population activity, is the strongest leg. Proxy pretraining lowers activation magnitude in all eighteen cells, across two independent model pairs, and it replicates at a near-zero accuracy gap — so it survives the Box 2 check. That matches Reference 1. The Fano analogue and its scale-invariant companion, CV-squared, both rise in all eighteen cells, which genuinely contradicts Reference 2's one directional prediction. Participation ratio also rises, but that one is ungraded — the review licenses no PR direction."

> "Level 1, representational content, is task-dependent. Under the location task, attention drives irrelevant identity decodability from 28 percent essentially down to chance. Under the category task, it doesn't move it at all. The earlier pooled run reported a uniform 'roughly halved', which was just the average of those two."

> "Level 3, explicit gating, is the one I have to downgrade. It was my headline: nine cells out of nine. That run pooled all checkpoints, and the from-scratch model contributed near-initialisation ones. With epochs pinned it's six of nine with small gaps, and attention-only turns out to already gate strongly. I'm retracting the strong version of that claim."

> "So what we defend is this: proxy pretraining produces a lower-magnitude, sparser, but higher-dimensional and more variable population code, at matched accuracy, in every cell we measured. That is an observable mechanistic phenomenon distinct from the accuracy gain — which is what was asked for."

**Emphasize:**
- Lead with Level 2 now. It is the result that survived every check.
- State the Level 3 retraction plainly and early. An audience that hears it from you reads it as rigour; an audience that finds it in questions reads it as an error.
- The graded honesty is the point: three levels were run precisely so that one failing would not sink the chapter — and that is exactly what happened.

**If asked "why should we trust the rest?":** because the same audit that found this also confirmed Level 2 twice, at two different accuracy gaps, with a scale-invariant control added specifically to rule out the most likely artifact.

---

## Slide: Conclusions

**Say:**
> "Let me pull everything together. On the left, the paper's six findings and our replication status. On the right, our contributions."

> "The paper's claims: slot-based memory is disproved — information drops from t-zero to t-one by 76 to 96 percentage points. Orthogonalization: the RNN actually *de-orthogonalizes* over time, which is the opposite of what you might expect. Task-specific subspaces are confirmed. MTMF preserves all features on the diagonal but off-diagonal varies. Cross-stimulus shared encoding — Hypothesis 2 — is not supported. And the causal perturbation No-Action rise is confirmed."

> "Our contributions: we audited and fixed four bugs in the analysis pipeline. We showed task-guided attention improves MTMF by 11 to 12 percent. We re-ran all 18 experiments with the fixed code. We open-sourced the audit findings."

> "And the two new contributions at the bottom — items 11 and 12 — are the neural-efficiency story I just walked you through, and the first combined run of attention gating with proxy pretraining. These are genuinely new findings, not restatements of what came before."

> "The overall implication: explicit attention complements RNN memory dynamics, but our models show more stimulus-specific encoding than the paper's claimed shared encoding. Attention improves task performance without changing the underlying representation strategy."

**Emphasize:**
- Items 11 and 12 — the neural-efficiency findings — are the capstone. Give them the most time.
- The implication about stimulus-specific versus shared encoding is the thesis-level takeaway.

---

## Slide: Meta-Learning Experiments — Section Title

> *"Rapid Task Adaptation with Attention"*

**Say:**
> "The last set of experiments asks a different question entirely: can the attention mechanism enable rapid adaptation to a *novel* task?"

**Emphasize:** This is a clean topic shift. Signal it.

---

## Slide: Meta-Learning Setup

**Say:**
> "The research question: can task-guided attention enable few-shot learning of a novel WM task? Our hypothesis was that attention separates task-agnostic temporal processing in the RNN from task-specific feature selection in the attention gates — so only the attention gates need updating for new tasks."

> "The novel task is Three-in-a-Row: detect when the same stimulus appears three consecutive times. This was never seen during training — the model was trained on N-back. It tests pattern recognition versus temporal distance."

> "We gave each model 50 examples and 20 epochs of fine-tuning. We tested six adaptation strategies: training from scratch, full fine-tuning, cognitive-only (RNN only), attention-only (freeze RNN, update gates), classifier-only, and attention-plus-classifier."

**Emphasize:**
- The hypothesis that *only attention needs updating*. This is what we're testing.
- That Three-in-a-Row is genuinely novel — never seen during training.

---

## Slide: Meta-Learning Results: Three-in-a-Row

**Say:**
> "The plot shows learning curves for all six strategies across the three architectures. The table gives the final numbers."

> "The headline: Base Cognitive-Only and Classifier-Only both win at 69.1 percent. Attention models are competitive at 65 to 68 percent. And training from scratch lands at chance — 50 percent."

> "A few things to notice. The Base model has no attention mechanism, so Attention-Only and Attention-Plus-Classifier are zero for it. For the attention and dual-attention models, Attention-Only actually works — 67 and 66 percent — which partially supports our hypothesis that the gates can adapt independently."

**Emphasize:**
- That scratch is at chance. Pre-training is doing the heavy lifting, not architecture.
- The table is the detail; the plot is the story. Point to the plot first.

---

## Slide: Key Findings

**Say:**
> "Four main results. First, Base Cognitive-Only and Classifier-Only win at 69 percent — simple task benefits from focused updates. Second, attention models are competitive — 65 to 68 percent — with Attention-Plus-Classifier best for attention at 69 percent. Third, Cognitive-Only is strong across the board — the RNN learns pattern matching well. Fourth, scratch is at chance — pre-training is essential."

> "Three interpretations. Task type matters: Three-in-a-Row is simpler than N-back — it's pattern matching versus temporal distance — so all models converge to similar performance. Architecture impact is minimal: Base 69, Attention 69, Dual 68 — all pretrained models learn effectively, and attention provides flexibility without penalty. And practically, pre-training is critical: scratch at 50 percent versus pretrained at 65 to 69 percent."

**Emphasize:**
- Pre-training matters more than architecture. This is the practical takeaway.
- That attention doesn't *hurt* — it provides flexibility without a performance penalty.

---

## Slide: Improvement Analysis

**Say:**
> "This plot visualizes the improvement over the scratch baseline. All pretrained methods show 13 to 19 percentage points of improvement over the 50 percent scratch baseline. All architectures converge to similar performance — 65 to 69 percent — showing that pre-training matters more than architecture choice for this task."

**Emphasize:**
- The visual is the point. Let the plot speak. The key insight is in the callout box.

---

## Slide: Thesis Contribution Context

**Say:**
> "Let me close by contrasting what we expected versus what we found."

> "Our original hypothesis: attention enables rapid few-shot adaptation, only the attention gates need updating, the RNN provides stable temporal processing, and attention models should outperform base."

> "What actually happened: all models perform similarly — 65 to 69 percent. Base 69, Attention 69, Dual 68. Classifier-Only and Cognitive-Only are the most efficient strategies. Reality: pre-training matters more than architecture choice."

> "Four key lessons. Task complexity determines architecture — simple tasks don't require attention. Architecture choice matters less — all pretrained models converge. Focused updates work — classifier and cognitive-only are the most efficient. And pre-training is critical — pretrained at 68 percent versus scratch at 50 percent is an 18-point gap."

**Emphasize:**
- The honesty of reporting a hypothesis that didn't pan out as expected. The audience will respect this.
- That the lesson is *positive*: pre-training works, attention doesn't hurt, and simple adaptation strategies are sufficient. This is useful information, not a failure.

---

## Slide: Thank You

**Say:**
> "Thank you for your attention. The paper is on arXiv at 2411.02685, and all the code is open-sourced at the GitHub link on the slide. I'm happy to take questions."

**Emphasize:**
- Pause. Make eye contact. Don't rush off the slide.
- If you expect questions about the neural-efficiency section — and you should — mentally rehearse your answer to "but participation ratio went *up*, how do you explain that?" before you get there.
