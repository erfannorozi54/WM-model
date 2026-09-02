# Speaker Notes: Proxy Pretraining Through End

*Covers the proxy chapter, the neural-efficiency chapter, the two-modification comparison, limitations, future work and the conclusions. Last aligned with `slides.md` on 2026-09-02.*

These notes cover every slide from the **"Proxy Pre-training"** section title through the **"Thank You"** slide. For each slide: what to say, what to emphasize, and what to be careful about.

---

## Slide: Proxy Pre-training — Section Title

> *"Modification 2 — Training Regimen"*

**Say:**
> "The attention mechanism changed the architecture. The second modification changes the training regimen instead, and leaves the architecture completely untouched."

**Emphasize:** That distinction is doing work later. Because nothing about the network changes, anything we observe downstream is attributable to what the model learned, not to added capacity.

---

## Slide: Two-Stage Training

**Say:**
> "N-back gives a sparse training signal. Three classes, and most timesteps are no-action — so most of the sequence teaches the model nothing. The proxy task fixes that: same stimuli, same task vectors, but at every step the model predicts the actual feature value N steps back. Location, identity, or category. Every step from t equals N onward carries a target."

> "Then we load those weights, swap in the three-class N-back head, and fine-tune on the real task."

**Emphasize:** Same ResNet50, same RNN, same classifier. Only the curriculum changed.

**Be careful about:** Someone will ask whether this is just more training. Say plainly that the volume control has not been run — see the caveats slide — and that we do not claim to have excluded it.

---

## Slide: Results: Proxy vs. Baseline (MTMF)

**Say:**
> "Fourteen point eight points on novel angle, twelve point two on novel identity. And the convergence story is sharper than the endpoint: the proxy model passes the baseline's *final* accuracy at epoch one."

**Emphasize:** The amber box. This is a performance result at unchanged N-back levels. We never tested whether the model can hold more items or succeed at higher N, so "capacity" is not available to us as a word.

**Be careful about:** Do not say "the model got more efficient" here. That is the next section, and it is a separate measurement.

---

## Slide: Alignment With Human Working Memory

**Say:**
> "Two findings from human work bear on this. Chung and colleagues show visual WM capacity expands when stimuli connect to preexisting semantic knowledge — and their EEG rules out the compression explanation, because delay activity goes *up*, not down. Mercer shows that repeating a meaningless non-word makes interference worse, while repeating a meaningful word changes nothing at all — structure had already done the protective work."

> "Mercer is the one that constrains us. It means 'proxy pretraining is just more training' is not the reading this literature supports."

**Emphasize:** We present only the portions of these two papers that bear on our alignment check. This is not a literature review.

**Be careful about:** The three orange boxes are not decoration. Read at least the first one aloud — the structure-versus-volume control has not been run in our model, and saying so before someone asks is much stronger than saying it after.

---

## Slide: Neural Efficiency — Section Title

> *"What the Two Modifications Do to the Population Code"*

**Say:**
> "Both modifications raise accuracy. But accuracy alone doesn't identify a working-memory phenomenon — it says the model got better, not that it got better in the way working memory does. This section measures something else."

**Emphasize:** Signal the transition clearly. This is a different kind of measurement, not a further accuracy number.

---

## Slide: The Claim We're Testing

**Say:**
> "Both modifications land in the same place on accuracy — attention takes novel identity from 81 to 92, proxy pretraining from 81 to 93. Both at unchanged N-back levels. Neither tells us whether the model can hold more items, so 'capacity' is not a word available to us."

> "Human working-memory research separates two things that raw accuracy conflates. Capacity means holding more items. Efficiency means doing the same or better work with a *suppressed* neural response — the signature seen for stimulus repetition, and also for prior knowledge."

> "Our claim under test is: familiarity and structure — from proxy pretraining — and explicit gating — from attention — both suppress task-irrelevant processing. We measure that at three independent levels of the model, graded against findings from human WM research."

**Emphasize:**
- The distinction between performance (accuracy) and efficiency (suppressed activity). This is the conceptual pivot of the entire section.
- Three levels rather than one because each is independently falsifiable.

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

> "We ran two independent pairs. Pair 1 is baseline versus proxy-finetuned — nine points apart on the split we analyse, and no closer pair exists, because the proxy model's first epoch already beats everything the baseline ever reaches. Pair 2 is attention-only versus attention-plus-proxy at *0.84 percentage points* — under one point — and that's our clean, matched-accuracy replication."

> "Same direction in both pairs, in all eighteen cells. Activation magnitude is lower under proxy pretraining — that matches *Reference 1*, Poppenk. I want to be precise about the attribution: Reference 2 actually reports the opposite for single neurons, more cells recruited and firing rate going up, so it would be wrong to put the magnitude result in a 'versus Reference 2' column. Population sparsity is higher in 17 of 18 cells, but neither paper predicts sparsity — that one is our own assumption, and I'm labelling it as such."

> "Participation ratio is higher in every cell. An earlier version of this deck graded that as contradicting Reference 2. We withdrew that: PR measures the effective dimensionality of the *population*, while the review's tuning finding is about *single neurons* — and the review reports tuning getting *broader*, not sharper. Those are different quantities, so the review licenses no PR prediction at all. The effect is real, it's just ungraded."

> "And it is real — we checked whether it was a sample-size artifact, because PR is biased upward by trial count. It isn't: in 11 of the 18 cells the proxy condition has *fewer* trials and still shows higher PR, and in one cell with exactly equal trial counts, 258 against 258, PR still rises 76 percent."

> "That leaves one genuine contradiction: the Fano-factor analogue is higher in every cell, and Reference 2 cleanly predicts lower. It's actually understated. Var-over-mean scales with activity level, and the proxy condition is *quieter* — so a pure scale effect would have pushed this metric *down*. It went up anyway, which means the scale-invariant version of the same measure moves further still."

> "The critical control: the magnitude effect *replicates at a sub-one-point accuracy gap* in Pair 2. It survives the Box 2 confound check. It's not just 'the proxy model is more accurate.'"

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

> "On alignment: our model reproduces Poppenk's signature cleanly. Prior knowledge acquired from a different task lowers hidden-state magnitude in all eighteen cells, at an accuracy gap of eight tenths of a percentage point. That is the human result, in our model, with the confound controlled."

> "It diverges from the variability finding. Our Fano analogue rises rather than falls, and so does the scale-invariant CV-squared — so this isn't an artifact of the magnitude difference; it is a real increase in relative variability."

> "I'd argue that divergence is interpretable rather than anomalous. The Constantinidis and Klingberg result comes from weeks of repeated training on the same task. Ours comes from knowledge transferred out of a different task. That is precisely the distinction Poppenk's design was built to isolate — and on the axis where the two manipulations are comparable, magnitude, we align."

**Emphasize:**
- Answer both questions explicitly: does the framework hold, and does the model match known WM behaviour. Those are the two things this chapter was for.
- "Partially corroborated" is the honest verdict and a strong one. Do not inflate it to "confirmed."
- The divergence has a mechanistic explanation, not an excuse. State the manipulation difference in one sentence.

**If asked whether one divergence undermines the chapter:** no — the three levels were measured independently precisely so the claim doesn't rest on any single metric, and the alignment on magnitude is the one tested at a sub-one-point accuracy gap.

---

## Slide: Telling the Two Modifications Apart — Section Title

**Say:**
> "One more section, and it is the one that ties the previous two together."

> "Up to this point every comparison I have shown you measures one modification against the plain baseline. Attention against baseline. Proxy pretraining against baseline. Even the efficiency chapter runs proxy-versus-no-proxy twice — once inside each architecture. Not one of those contrasts can tell the two modifications apart."

**Emphasize:**
- Say plainly why this section exists. It is the section that keeps the attention chapter and the proxy chapter from being two unrelated successes.

---

## Slide: Why This Section Exists

**Say:**
> "Here is the problem stated precisely. Each modification is confirmed against its own control, so each one looks like an independent success — and either chapter could be deleted without the other noticing."

> "But the four models I have already trained form a two-by-two: baseline or attention, crossed with from-scratch or proxy-pretrained. The chapters above read the rows. This section reads the columns — what does attention do, and does it still do it once proxy pretraining has run?"

> "No new training. The same four models, re-read at pinned, accuracy-matched checkpoints."

**Emphasize:**
- "No new training" is worth saying out loud — it pre-empts "how much compute did this cost."

**If asked why this wasn't the design from the start:** it was not framed as a factorial experiment; the two modifications were developed as separate chapters. Recognising the 2×2 was a re-reading of existing runs, and that is exactly why it was cheap.

---

## Slide: Result 1 — On Accuracy, They Are Redundant

**Say:**
> "Attention alone buys nine and a half points. Proxy pretraining alone buys eleven. Both together buy eleven point one — not twenty."

> "That is an interaction of minus nine point three percentage points. Whichever modification arrives first captures nearly all of the gain that is available, and the second adds a rounding error. On novel angle it is worse: minus twelve point three, and attention stacked on proxy pretraining actually costs about half a point."

> "I want to be careful about how I read that. This is not a null result and it is not a disappointment. It says the two modifications are solving the *same* problem — pulling the task-relevant feature out of a naturalistic embedding — by different routes. An architectural gate and a training regimen arrive at the same ceiling, which locates that ceiling in the task and the representation rather than in either mechanism."

**Emphasize:**
- Give the interpretation immediately after the number. A bare "−9.30pp" invites the reading "so your second contribution was pointless."
- Say "redundant, not additive." That phrasing is precise and it is what the data shows.

**If asked whether this makes the attention chapter unnecessary:** the opposite — before this comparison the two chapters were interchangeable and either could have been dropped. This is the result that gives each of them a distinct role, and the scope boundary on STSF shows they are not interchangeable in general.

**Caveat to volunteer if pressed on statistics:** one seed per cell. The interaction is observed, not yet measured with an error bar. Three seeds × four cells is the fix and it is the cheapest run in the repo.

---

## Slide: Result 2 — The Population Code Says The Same Thing

**Say:**
> "The accuracy result has a counterpart inside the model, and this is the part I find most convincing."

> "Reading the left column — attention with no proxy pretraining — attention lowers activation magnitude in all nine cells and lowers effective dimensionality in all nine, and these are not small moves: participation ratio drops from around seven to around four and a half."

> "Reading the right column — attention on top of proxy pretraining, at an accuracy gap of exactly zero, the tightest match anywhere in this project — dimensionality does not move at all. Four cells of nine, which is a coin flip. And sparsity actually goes the other way."

> "So attention reshapes the population code when it is the only modification, and that effect is absorbed once proxy pretraining has already shaped it. The geometry and the accuracy tell the same story."

**Emphasize:**
- The zero-point-zero-zero accuracy gap is your strongest methodological card in the whole talk. Say it explicitly.
- Mention that excluding the three near-rank-1 location cells leaves the PR result at six of six — you are not leaning on degenerate cells.

**On the guard box — say it, don't skip it:**
> "One thing I deliberately do not claim here. The Fano analogue falls in seven of nine cells under attention, which would be a tidy agreement with Constantinidis and Klingberg. But the scale-invariant CV-squared is five of nine — a coin flip. Attention lowers magnitude, and the Fano analogue is scale-dependent, so that apparent drop is exactly the artifact CV-squared was added to catch. The variability rise stays attributed to proxy pretraining, where both metrics agree in all eighteen cells."

**If asked why you show a result you then withdraw:** because the control worked. A scale-invariant companion metric was added specifically to catch this class of error, and here it caught one. Reporting the catch is the evidence that the other eighteen-of-eighteen results are not artifacts of the same kind.

---

## Slide: Limitations — What These Numbers Cannot Bear

**Say:**
> "I want to put the limits on the record myself rather than have them found for me."

> "Statistically: one seed per cell, so the interaction has no error bar. The h-equals-256 sampling regime gives about fifty-two test samples against seventy identity classes, so only effects above roughly twenty-five points are measurable — and figures are never compared across hidden sizes. The bootstrap floor is a thousand resamples, so I quote no p-value below zero point zero zero two. And the H2 generalization trend is suggestive only: eighty validation trials, standard error five to six points, no pair separates."

> "Structurally: efficiency is not capacity — higher N-back was never tested. The baseline-to-proxy accuracy gap of eight point nine points is irreducible, because proxy epoch one already beats every baseline checkpoint, which is why the attention pair is the primary evidence. Level 1 has not been run across the full square. And this audit found that the epochs flag in the comprehensive analysis script is a no-op, so that particular 2×2 is not yet accuracy-matched."

**Emphasize:**
- Deliver this slide at a steady pace, not apologetically. A limitations slide you wrote yourself is a strength; one the committee writes for you is not.
- Item 8 — the `--epochs` no-op — is a bug you found in your own pipeline and disclosed. Frame it that way.

**If asked "so which of your results survive all this?"** Two. The magnitude result, which replicates in four independent contrasts including one at a zero-point accuracy gap. And the accuracy interaction, which is large enough that a seed effect is unlikely to erase it — though it has not yet been measured.

---

## Slide: Future Work — In Priority Order

**Say:**
> "Five things, in the order I would actually do them."

> "First, seeds on the interaction — the only task that upgrades my headline from observed to measured, and the cheapest run in the repo. Second, the STSF scope-boundary test: attention *costs* eleven points on STSF because a single task gives the gate no ambiguity to resolve, whereas proxy pretraining's mechanism, dense feature recall, has no reason to fail there. If proxy survives STSF while attention collapses, that is the cleanest mechanistic dissociation this thesis can offer — it turns the attention chapter's failure case into a positive result about why the two differ."

> "Then three housekeeping items: fix the epochs flag and re-run the five-analysis 2×2 properly, complete Level 1 across the square against one consistent attention checkpoint, and add an h-equals-128 proxy arm where the sampling actually resolves small effects."

> "Every one of these is a re-run or a re-reading of models that already exist. None needs a new task, a new dataset, or a new architecture."

**Emphasize:**
- Item 2 is the one to sell. It is the best remaining science in the project and it is cheap — STSF is the fastest config in the repo.
- Closing on "no new architecture required" leaves the committee with a tractable project, not an open-ended one.

---

## Slide: Conclusions

**Say:**
> "Let me pull everything together. On the left, the paper's six findings and our replication status. On the right, our contributions."

> "The paper's claims: slot-based memory is disproved — information is readable at encoding and at chance by t-plus-one. Orthogonalization: the RNN actually *de-orthogonalizes* over time, which is the opposite of what you might expect. Task-specific subspaces are confirmed — cross-task off-diagonal decoding is far below the diagonal. Task-relevance is the one claim we do *not* support: the task-relevant cell is often out-decoded by category, which stays readable in every task context. Cross-stimulus shared encoding — Hypothesis 2 — *is* supported, at mean generalization zero point six, once we fixed a class-index bug in our own pipeline. And the causal perturbation No-Action rise is confirmed in twelve of eighteen models."

> "Our contributions: two audits, nine bugs fixed. Task-guided attention, plus eleven points on STMF and MTMF — and a measured scope boundary where it *costs* eleven points, because with a single task there is no ambiguity for the gate to resolve. Proxy pretraining, plus fourteen point eight on novel angle. All eighteen experiments re-run with the fixed code, and the audit findings open-sourced."

> "Items eleven and twelve are the new work. Eleven is the three-level neural-efficiency result. Twelve is the two-by-two I just showed you: the two modifications are redundant, not additive — minus nine point three points of interaction, and attention's effect on the population code is absorbed once proxy pretraining is present."

> "The thesis-level takeaway is in the box. Two independent modifications — one architectural, one in the training regimen — each recover about ten points and each produce the lower-magnitude population code that Poppenk reports for prior knowledge in humans. But they are two routes to one ceiling, not two mechanisms that compose. That locates the ceiling in the task and the representation, not in either modification."

> "And the methodological result is the one I would keep if I could keep only one. Three findings in this project reversed under epoch pinning, accuracy matching, and a scale-invariant control. Every time, the first and more attractive answer was the artifact."

**Emphasize:**
- Item 4 is the honest negative — the paper's task-relevance claim is the one we do not reproduce. State it without hedging; a replication that confirms everything is less credible, not more.
- Item 5: H2 *is* supported. Earlier drafts of these notes said it was not; that was before the class-index fix. Do not say "not supported."
- Items 11 and 12 are the capstone. Give them the most time.
- The closing line about three reversed findings is the strongest sentence in the talk. Slow down for it.

**If asked which contribution is the real one:** the comparison. The replication was the entry price and the two modifications are each a normal engineering result; the 2×2 is the part that says something about the problem rather than about the model.

---

## Slide: Thank You

**Say:**
> "Thank you for your attention. The paper is on arXiv at 2411.02685, and all the code is open-sourced at the GitHub link on the slide. I'm happy to take questions."

**Emphasize:**
- Pause. Make eye contact. Don't rush off the slide.
- If you expect questions about the neural-efficiency section — and you should — mentally rehearse your answer to "but participation ratio went *up*, how do you explain that?" before you get there.
