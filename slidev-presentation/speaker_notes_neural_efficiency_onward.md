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

## Slide: Reference 2 — What Kind of Paper Is This?

**Say:**
> "Our second reference is Constantinidis and Klingberg's 2016 Nature Reviews Neuroscience paper on working-memory training. I want to flag something important up front: this is a *review*, not a single experiment. The authors didn't collect new data — they read and summarized dozens of separate studies, both monkey brain-cell recordings and human brain-imaging studies of WM training, and pulled out the pattern that repeats across all of them."

> "We're deliberately citing a review here rather than one paper, because for this claim we needed the general, well-replicated pattern of what happens to brain activity when working-memory performance improves through training — not just one lab's single result."

> "The question relevant to us: when training improves WM performance, what changes about *how* brain cells represent the task — not just how much they fire, but how that activity is organized?"

**Emphasize:**
- Correct the natural assumption that this is a single study — say clearly it's a review.
- Frame this as a deliberate, appropriate choice of source, not a weaker citation.

---

## Slide: Reference 2 — The Pattern Across Studies

**Say:**
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
- That the paper predicts specific directions: participation ratio down, Fano down. We will report honestly when our results go the *opposite* direction.

---

## Slide: Results — Level 1: Representational Content

**Say:**
> "Level 1 compares the baseline GRU against the attention GRU on the MTMF config, decoding identity. Before the numbers, one plain-language anchor for the whole table: 'decodability' means — if you trained a simple classifier to guess the object's identity just from the model's hidden state, how often would it succeed? Lower is what we want here, because identity is task-irrelevant — lower decodability means the model is hiding it better, i.e. suppressing it. The other three sub-metrics probe the same idea from different angles — how separated the representations are, how similarly-shaped the space is, how robust the encoding is to swaps."

> "The results are mixed, leaning supportive — and I'll be honest, this is the weakest of the three levels."

> "Identity decodability at timesteps 3, 4, and 5 roughly halves under attention — from 15, 12, 10 percent down to 7, 7, 6 percent. That's clearly supporting suppression. The Procrustes swap test also supports it — 7 percentage points better accuracy when you respect the identity boundary."

> "But the orthogonalization index is flat — 0.936 versus 0.933 — and Procrustes reconstruction is also flat. Both are at ceiling for this config, so this isn't a contradiction, it's a limitation of the MTMF setup for this particular metric."

> "Bottom line: 2 of 4 sub-metrics clearly support suppression, 2 are flat at ceiling. Supporting, not central, evidence."

**Emphasize:**
- Anchor the audience with the one-sentence "what is decodability" definition before showing numbers — the table is unreadable without it.
- That you're reporting this honestly, graded, not as a uniform win.
- The distinction between "flat at ceiling" and "contradictory." These are not failures — they're limitations of the metric in this config.

---

## Slide: Results — Level 2: Population Activity

**Say:**
> "Level 2 is where things get interesting, and also where we have to be the most careful. Quick definitions before the table, because otherwise the four metric names are just jargon: 'magnitude' is how loud the hidden-state signal is on average. 'Sparsity' is what fraction of hidden units are actually doing something for a given input — higher means fewer units firing at once. 'Participation ratio' is roughly how many independent patterns the population is using — higher means information is spread across more directions instead of squeezed into a few. And 'Fano-factor analogue' is the same trial-to-trial wobble idea from Reference 2, just computed on our model's units instead of real neurons."

> "We ran two independent pairs. Pair 1 is baseline versus proxy-finetuned with a 10-percentage-point accuracy gap — that's the unmatched comparison. Pair 2 is attention-only versus attention-plus-proxy with a *0.08-percentage-point* accuracy gap — that's our clean, matched-accuracy replication."

> "Same direction in both pairs, in every cell. Activation magnitude is lower under proxy pretraining — that matches Poppenk et al.'s suppression finding. Population sparsity is higher in most cells — also matches, though the effect is small."

> "But here's where we have to report honestly: participation ratio is *higher* in every cell, and the Fano-factor analogue is also *higher* in every cell. Both go the *opposite* direction from what Constantinidis and Klingberg predict. Their prediction was that training produces a sharper, less variable code — lower PR, lower Fano. Ours says the proxy model produces a lower-magnitude, sparser code, but one that is *higher-dimensional* and *more variable*."

> "The critical point: the magnitude and sparsity effects *replicate at near-zero accuracy gap* in Pair 2. They survive the Box 2 confound check. They're not just 'the proxy model is more accurate.'"

**Emphasize:**
- The two-pair design. Pair 1 is the headline, Pair 2 is the confound check. Say this clearly.
- The honest contradiction with Reference 2. Do not hide this. Say "opposite of prediction" and mean it.
- That the result is *nuanced*: lower magnitude and sparser, yes — but higher-dimensional and more variable, not the simple "sharpening" the literature predicts. This is a genuine, interesting finding in its own right, not a failure.

---

## Slide: Results — Level 3: Explicit Gating (Headline)

**Say:**
> "Level 3 is our headline result. Two quick definitions first: the 'gate' is our attention mechanism's literal on/off dial per feature channel. 'Suppression index' is how much lower the gate sits on task-irrelevant channels versus task-relevant ones — more negative means it mutes the irrelevant stuff more strongly. 'Gate-relevance correlation' is how tightly the gate's setting tracks a channel's actual relevance — higher means the gate is reliably reading relevance, not doing something only loosely related to it."

> "We compared attention-only against attention-plus-proxy on the MTMF config, at near-matched accuracy — 93.43 percent versus 93.51 percent. This experiment had never been run before this analysis pass. It directly answers: can attention-containing models be used in proxy pretraining, and does it help?"

> "The answer is yes, dramatically. Nine out of nine cells are sharper under proxy pretraining."

> "For the attention-only model, the suppression index ranges from negative 0.17 to positive 0.07 — that means in 2 of 9 cells, the index is *wrong-signed*: the model is gating *up* the irrelevant channels relative to the relevant ones. The gate-relevance correlation is weak, 0.09 to 0.24."

> "For attention-plus-proxy, the suppression index ranges from negative 0.33 to negative 0.52 — consistently negative, consistently large. The gate-relevance correlation jumps to 0.45 to 0.72 — strong."

> "Let me put this in concrete terms. For the *category* feature, the attention-only model barely gates at all — the index is near zero, sometimes wrong-signed. Attention-plus-proxy fixes this completely. This is at matched accuracy, the effect is large, and it's a signature a plain RNN baseline structurally cannot produce."

> "One more thing that makes this level different from the other two: it needs no external reference to interpret. Levels 1 and 2 are analogies to a human finding. Here the gates *are* an explicit suppression signal by construction — we're reading it off directly, not inferring it."

**Emphasize:**
- **This is the strongest result in the entire section.** Slow down here. Let the 9-out-of-9 number land.
- That Level 3 is *not* an analogy the way Levels 1 and 2 are. The gates are literally suppression, so no outside citation is doing any work here.
- The word "headline" is on the slide for a reason. This is the most novel and most defensible finding.
- That a plain GRU *cannot* produce this result. It's unique to the attention architecture combined with proxy pretraining.
- The wrong-signed cells in attention-only. This is a striking detail — the model without proxy pretraining isn't just bad at gating, it's sometimes gating in the *wrong direction*.

---

## Slide: Neural-Efficiency Chapter: Conclusion

**Say:**
> "Let me grade each level against its reference prediction."

> "Level 3, explicit gating: strongest support. Accuracy-matched, 9 out of 9 cells, large effect. This is the most novel result."

> "Level 2, population activity: partial support. Magnitude and sparsity match Poppenk et al. and replicate at matched accuracy — that's real. But participation ratio and Fano factor contradict Constantinidis and Klingberg's sharpening prediction. The population code is lower-magnitude and sparser, but higher-dimensional and more variable — not the simple 'quieter and sharper' story the literature predicts."

> "Level 1, representational content: weakest, not contradictory. Two of four sub-metrics support suppression, two are flat at ceiling."

> "The claim we can defend: proxy pretraining produces a lower-magnitude, sparser, but higher-dimensional and more variable population code, and dramatically sharpens explicit gating — both at matched accuracy, so neither is just 'the model got better.' This is a genuine, observable WM phenomenon, distinct from and not reducible to the accuracy gain already in the deck."

> "We report this honestly graded. Level 3 is the strongest and most novel result. Level 2 partially confirms the literature and partially contradicts it. Level 1 is supporting, not central, evidence."

**Emphasize:**
- The graded honesty. You're not claiming a uniform win. You're saying "here's what's strong, here's what's partial, here's what's weak."
- The claim is *defensible* precisely because you're reporting it this way.
- That the finding is *distinct from the accuracy gain*. This is the answer to the professor's ask.

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
