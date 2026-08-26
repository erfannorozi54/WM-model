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

## Slide: Established Findings We Test Against

**Say:**
> "We anchor our measurements to two findings from the human working-memory literature. I'm presenting only the specific result we test against from each — not the studies themselves, because our purpose here is an alignment check on our own model, not a literature review."

> "The first, from Poppenk, Moscovitch and McIntosh: prior knowledge suppresses processing-related activity in visual and language cortex just as strongly as recently repeating the same material. The two whole-brain suppression maps are statistically indistinguishable. What matters for us is that suppression does not depend on *how* the familiarity was acquired — which is exactly our situation, since our proxy-pretrained model acquires its knowledge from a different task. Prediction: hidden-state magnitude should drop."

> "The second, from Constantinidis and Klingberg's review of WM-training studies: after training, prefrontal neurons become less variable trial to trial — the Fano factor drops. Prediction: our Fano-factor analogue should drop too."

> "I want to be explicit about what these findings do *not* license, because getting this wrong is easy. Magnitude is graded against Poppenk alone — the review actually reports firing *rate* increasing after training, which is the opposite direction under a different manipulation. Participation ratio is graded against nothing: it measures population dimensionality, while the review's result is about single-neuron tuning, which it reports getting broader. And sparsity is our own assumption; neither source predicts it."

> "One method rule comes from the review: a drop in activity means nothing unless accuracy is comparable. So every comparison in this chapter is accuracy-matched and reports the residual gap."

**Emphasize:**
- That you are deliberately quoting only the relevant finding from each source. Say this once, plainly — it pre-empts "why didn't you cover the rest of that paper?"
- The grading assignments. Only one metric is graded against the review, and it is the one that will diverge.
- The accuracy-matching rule, because the next three slides all depend on it.

---

## Slide: Results — Level 1: Representational Content

**Say:**
> "Level 1 asks whether attention removes task-irrelevant information from the hidden state. We decode object identity from the baseline GRU and from the attention GRU, each at its own best epoch, on the novel-identity split."

> "One anchor before the numbers: 'decodability' means — if you trained a simple classifier to read the object's identity out of the hidden state, how often would it succeed? Identity is irrelevant in both contexts shown here, so *lower* is the signature we're looking for."

> "We run the two task contexts separately, because in the identity task, identity is the feature the model is supposed to retain — including it would dilute the very thing we're measuring."

> "Under the location task, identity decodability falls from 28, 18 and 15 percent to 5, 2 and 5 percent. Chance is about 3 percent, so attention drives it essentially to the floor — the information is not merely reduced, it is gone."

> "Under the category task, it doesn't move: 20, 16, 16 becomes 15, 19, 17."

> "So the suppression is real but conditional, and the condition is interpretable: it appears where identity competes with a spatial code, not where it competes with a categorical one. Orthogonalization is at ceiling in both contexts and doesn't discriminate; Procrustes reconstruction is lower under attention."

**Emphasize:**
- Quote the chance level out loud. "Down to chance" is the claim, and it is much stronger than "roughly halved."
- The task-dependence is a finding, not a shortfall. Present it as the boundary condition of the framework.
- Why the identity task context is excluded — one sentence, before anyone asks.

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

## Slide: Results — Level 3: Explicit Gating

**Say:**
> "Level 3 is the most direct measurement in the chapter, because the attention model has literal per-channel gates. A gate near zero on a task-irrelevant channel *is* suppression — we read it off, we don't infer it. The suppression index is how much lower the gate sits on irrelevant channels than on relevant ones, so more negative means stronger muting."

> "Both models are read at a single pinned checkpoint, the accuracy-matched pair. That control matters here: the attention-only model is trained from scratch, so averaging across its checkpoints would fold in near-initialisation gates that the fine-tuned model — converged at its first epoch — never contributes. Pinned, we are comparing two trained models."

> "The result: proxy pretraining sharpens gating in six of nine cells, and the sharpening is modest. On location and category, the attention-only model already gates strongly, around minus 0.42 to minus 0.52, and proxy pretraining adds a little. What does improve consistently is the gate-relevance correlation — how tightly the gate tracks a channel's actual relevance — from 0.66–0.73 up to 0.84–0.85 on location."

> "And a clear negative worth stating: neither model gates on identity. Both sit near zero or slightly wrong-signed there. That lines up with Level 1, where identity suppression also failed to appear in the category context — these two levels agree that identity is the feature this architecture handles least well."

**Emphasize:**
- That this level needs no external reference to interpret. The gates are a built-in suppression signal.
- The checkpoint control, stated as a control rather than a caveat.
- The convergence between Levels 1 and 3 on identity. Two independent measurements pointing at the same limitation is a stronger statement than either alone.

---

## Slide: Neural-Efficiency Chapter: Conclusion

**Say:**
> "This slide does two jobs: what our framework survived, and how our model lines up against the human findings."

> "Our framework was that familiarity and explicit gating both suppress task-irrelevant processing, testable at three levels. The verdict is *partial corroboration*, and I'd rather give you the shape of it than a single word."

> "Level 2, population activity, corroborates it outright — every metric moves in the same direction across all eighteen cells, in two independent model pairs, at matched accuracy. Level 1 corroborates it conditionally: near-total suppression under the location task, none under category. Level 3 corroborates it partially: a modest sharpening in six of nine cells, with gate-relevance correlation improving throughout."

> "On alignment: our model reproduces Poppenk's signature cleanly. Prior knowledge acquired from a different task lowers hidden-state magnitude in all eighteen cells, at an accuracy gap of eight hundredths of a percentage point. That is the human result, in our model, with the confound controlled."

> "It diverges from the variability finding. Our Fano analogue rises rather than falls, and so does the scale-invariant CV-squared — so this isn't an artifact of the magnitude difference; it is a real increase in relative variability."

> "I'd argue that divergence is interpretable rather than anomalous. The Constantinidis and Klingberg result comes from weeks of repeated training on the same task. Ours comes from knowledge transferred out of a different task. That is precisely the distinction Poppenk's design was built to isolate — and on the axis where the two manipulations are comparable, magnitude, we align."

**Emphasize:**
- Answer both questions explicitly: does the framework hold, and does the model match known WM behaviour. Those are the two things this chapter was for.
- "Partially corroborated" is the honest verdict and a strong one. Do not inflate it to "confirmed."
- The divergence has a mechanistic explanation, not an excuse. State the manipulation difference in one sentence.

**If asked whether one divergence undermines the chapter:** no — the three levels were measured independently precisely so the claim doesn't rest on any single metric, and the alignment on magnitude is the one tested at a near-zero accuracy gap.

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
