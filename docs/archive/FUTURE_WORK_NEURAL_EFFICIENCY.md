# Future Research Direction: Neural-Efficiency Signatures of Feature Familiarity

**Status:** the RNN-hidden-state metrics (§4.3) and the gate-suppression index (§4.8.1) are implemented and have now been run on real GPU experiment data (see `docs/EXECUTION_PLAN_NEURAL_EFFICIENCY.md` §7 for the real results).
**Correction:** an earlier version of this document, and `slidev-presentation/proxy_training_comparison.md`, described the existing proxy-pretraining accuracy gain as a **capacity** effect (citing Chung, Brady & Störmer, 2024). That's an over-claim: capacity specifically means holding *more items* (succeeding at higher N, or a higher K in span tasks); our result is accuracy improvement *at the same N-back levels*, i.e. a **performance/accuracy** effect, not a demonstrated capacity increase. `proxy_training_comparison.md` still needs a matching correction (flagged, not yet done). This document's own target — familiarity **reducing neural activity**, plus the gate-suppression signature — is not affected by that correction; if anything it becomes more important, since it's now the *only* genuinely new, literature-grounded WM phenomenon this thesis demonstrates, rather than one of two.

---

## 1. Why this is a distinct finding (not a restatement of the accuracy/performance result)

The existing deck shows the proxy-pretrained model reaches higher accuracy with familiar/structured features (novel-angle: 82.7% → 97.5%; novel-identity: 80.6% → 92.8%), at the same N-back levels — a **performance/accuracy** claim, not a capacity claim (we never tested whether the model can now handle more items or higher N). It says nothing about *how much internal activity* the model spends to get there. Human WM research treats capacity and efficiency as separate, independently measured phenomena, and our accuracy result maps onto neither directly — it's a third, more generic "did better" result that this document's efficiency work is meant to give actual mechanistic content:

- **Capacity effect**: meaningful/familiar stimuli let people hold more items in mind (Chung et al., 2024).
- **Efficiency effect**: prior knowledge lets the brain do the same or better perceptual/mnemonic work with **suppressed** neural response, the same signature classically seen for stimulus repetition (repetition suppression, RS).

If the model only demonstrated the first, a reviewer could argue you've shown "better accuracy" without showing anything mechanistically brain-like. Showing that the **hidden-state activity itself** is more efficient (lower magnitude / lower dimensionality / more stable across trials) in the familiar-feature (proxy) condition — at matched or better accuracy — is a second, independent line of evidence that the model recapitulates a *specific, well-documented neural mechanism*, not just an aggregate performance gain.

---

## 2. Primary paper reference (read in full — verified)

**Poppenk, J., Moscovitch, M., & McIntosh, A. R. (2016). fMRI evidence of equivalent neural suppression by repetition and prior knowledge.** *Neuropsychologia*, 90, 159–169. (Preprint: bioRxiv 10.1101/056069, read in full — 20 pages inspected directly.)

**What it actually shows** (verified from the full text, not an abstract skim):

- Design: participants read (a) novel Asian proverbs, (b) Asian proverbs *repeated* ~30 minutes earlier within the same session, and (c) English proverbs known from a lifetime of prior exposure (**prior knowledge**, no recent repetition at all).
- Core result: recently-repeated items and previously-known items produced **statistically indistinguishable neural suppression** (RS) relative to novel items, across a broad visual-linguistic network (left inferior PFC, posterior superior temporal gyrus, etc.), confirmed by a multivariate conjunction analysis (`r = 0.65, BSR = 8.32, P < 0.001`) and an ROI check that found *no* reliable RS-region difference between the repetition and prior-knowledge conditions even at a liberal threshold.
- Critically, this happened **despite the two conditions relying on different memory systems** (episodic retrieval for recently-repeated items vs. semantic/long-term knowledge for previously-known items) — the "activated" regions did dissociate along expected episodic-vs-semantic lines, but the **suppressed** regions did not. The authors' interpretation: RS is a general signature of *any* retrieved information facilitating perception/comprehension, not a narrow repetition-specific decay process.
- Practical implication for this proposal: the correct experimental logic is not "familiar → less activity" in isolation, but "**information retrieved from memory (however it got there) → suppressed processing-related activity, dissociable from the memory-retrieval activity itself**." That two-part structure (suppression in a stable network + dissociable activation in retrieval-type-specific regions) is exactly the kind of two-part signature we can look for in the model's hidden states (see §4.2 and §4.3).

## 3. Supporting paper (read in full — verified)

**Constantinidis, C., & Klingberg, T. (2016). The neuroscience of working memory capacity and training.** *Nature Reviews Neuroscience*, 17(7), 438–449. (Full text of the review body and both boxes read directly.)

Relevant, verified content used below:

- After WM training, PFC neurons show **decreased mean neuronal selectivity** for trained stimuli (i.e., broader/less-sharp tuning) even as *more* neurons become recruited and *mean firing rate* increases — efficiency here is not simply "less activity," it is a **shift in how activity is distributed and organized** (fewer neurons carrying more stable, less variable signal).
- Training is also associated with **decreased trial-to-trial firing-rate variability** (lower Fano factor) and **decreased noise correlation** between neurons — i.e., a more efficient, less noisy population code, independent of any change in mean activity level.
- **Box 2 of the paper is an explicit methodological warning** worth carrying into our design: BOLD-signal changes are ambiguous between "efficiency" (fewer/sharper units doing more work) and simple changes in reaction time or task-related engagement, and a naive "lower BOLD = more efficient" reading is not licensed without more information. This directly motivates the matched-accuracy design and multiple, converging metrics in §4 below — no single "activity went down" number should be treated as sufficient on its own.

**How the two papers combine to define the phenomenon we're testing:** familiarity (whether from repetition or from prior/structural knowledge) should produce (a) suppressed activity in a stable, generic "facilitated-processing" sense (Poppenk et al.), and (b) a shift toward a less variable, more sharply organized code rather than a naively "quieter" one (Constantinidis & Klingberg). Both properties are directly measurable from the RNN hidden states this codebase already saves.

---

## 4. Step-by-step methodology

### 4.1 Data you already have

Both conditions already exist as trained artifacts (per `slidev-presentation/proxy_training_comparison.md`):

- Baseline: `experiments/wm_mtmf_20260520_140601/hidden_states/epoch_XXX/<split>/batch_*.pt`
- Proxy-finetuned: `experiments/finetune_proxy_wm_mtmf_20260705_164908/hidden_states/epoch_XXX/<split>/batch_*.pt`

Both were trained with `save_hidden: true`, so payloads contain `hidden (B,T,H)`, `task_index`, `n`, `targets`, `locations`, `categories`, `identities`, `sample_index`/`sample_keys`, and `split` ∈ {`val_novel_angle`, `val_novel_identity`} — confirmed directly from `src/analysis/activations.py`. No new training run is required to start; only new *analysis* code.

### 4.2 Matched-accuracy design (the critical control)

Because the proxy model is simply *better* (higher accuracy at every epoch), a raw "proxy hidden-state activity < baseline hidden-state activity" comparison is confounded — better performance could reduce activity through many uninteresting routes (e.g., cleaner gradients, different weight scale) that have nothing to do with familiarity per se, echoing the Constantinidis & Klingberg Box 2 warning.

Design the comparison so accuracy is held (approximately) constant, then ask whether activity still differs:

1. From each experiment's `training_log.json`, pick the **earliest proxy epoch** whose `val_novel_angle` accuracy is ≥ the baseline's *best* `val_novel_angle` accuracy (proxy hits ~93% by epoch 1 vs. baseline's ceiling of 82.7% per the existing deck — if proxy never needs to "catch down" to baseline, instead pick the baseline epoch and proxy epoch pair whose accuracies are closest, and report the residual accuracy gap explicitly rather than hiding it).
2. Also run the *unmatched* comparison (best-epoch vs. best-epoch) as a secondary analysis, and report both — if the effect only appears in the unmatched comparison, that is itself informative (it would say "efficiency effect requires more capable performance," not "familiarity produces suppression independent of ability").
3. Restrict all hidden-state comparisons to the same split (`val_novel_angle` or `val_novel_identity`) and the same `task_index`/`n` cells, since these differ in item structure and are not directly comparable to each other.

### 4.3 Metrics (compute all four; no single metric should carry the claim)

For each condition (baseline @ matched epoch, proxy @ matched epoch), using `src/analysis/activations.py`'s `load_payloads()` + `iterate_records()`/`build_matrix()`:

1. **Activation magnitude.** Per-timestep L2 norm of the hidden vector, `‖h_t‖₂`, averaged across trials within each `(task_index, n, split)` cell. This is the direct RNN analogue of "response magnitude" in Poppenk et al. Report mean ± bootstrap CI per timestep (not just a single scalar), since RS-type effects in humans are often time-locked to stimulus onset.
2. **Effective dimensionality / participation ratio.** Run PCA on the (trials × H) hidden-state matrix per cell; compute the participation ratio `PR = (Σλᵢ)² / Σλᵢ²`. **Report as an exploratory descriptive, with no directional prediction attached.** An earlier version of this document said lower PR would "mirror the sharpened tuning Constantinidis & Klingberg describe" — that was wrong on both counts (they describe tuning getting *broader*, and PR is a population-level quantity that does not track single-unit tuning width in a fixed direction). See §8 of `docs/PAPER_EXPLAINED_CONSTANTINIDIS_KLINGBERG_2016.md`. **PR is also biased upward by trial count** (~14.4 → ~20.7 going from N=50 to N=1600 on identical data), so only compare it at matched trial counts.
3. **Population sparsity.** Fraction of hidden units with near-zero activation (below a small threshold relative to that unit's own max) or a Gini coefficient over unit activations. "A more efficient code should be sparser" is **our own assumption, not a claim from either reference** — Constantinidis & Klingberg in fact report *more* neurons recruited after training, i.e. a less sparse code. Do not grade this row against Reference 2. The threshold-based variant is also biased upward by trial count (the per-unit max that sets the threshold grows with the sample), so compare it only at matched trial counts; the Gini variant is preferable when counts differ.
4. **Trial-to-trial variability (Fano-factor analogue).** For repeated presentations of matched conditions (same `task_index`, `n`, `location`/`identity`/`category` combination across different trials), compute `Var(activation)/Mean(activation)` per unit, averaged over units — this is the direct RNN analogue of the Fano-factor reduction reported after WM training, and **the one Level-2 metric Constantinidis & Klingberg does make a clean directional prediction for** (it should go *down*). Two implementation requirements: use `ddof=1` (with `ddof=0` a group of size *g* underestimates the variance by `(g-1)/g` — a 33% bias at the size-3 floor), and read it alongside the scale-invariant `Var/Mean²` (CV²), because `Var/Mean` on a continuous signal scales linearly with activity magnitude and the magnitude metric above already reports a large scale difference between the two conditions.

### 4.4 Statistical approach

- Use **bootstrap resampling over trials** (resample with replacement within each condition, 1,000–2,000 iterations) to build CIs for each metric's baseline-vs-proxy difference at the matched epoch — consistent with the resampling-based approach already used for the Procrustes/decoding analyses in this repo (`src/analysis/procrustes.py`), so the statistical style stays consistent across the thesis.
- For the per-timestep magnitude curves, treat timestep as a repeated measure: fit a mixed-effects model (`activation ~ condition * timestep + (1 | trial)`) rather than testing each timestep independently, and correct for the number of `(task_index, n)` cells tested with FDR (Benjamini–Hochberg).
- Report effect sizes (Cohen's d or the bootstrap-ratio convention already used in this codebase's Procrustes swap test) alongside p-values — a "significant but tiny" difference in a 256-unit hidden state is not the same claim as the human EEG/fMRI literature's effect sizes, and the thesis should be explicit about magnitude, not just significance.

### 4.5 Visualizations

1. **Magnitude-over-time plot**: mean `‖h_t‖₂` (± CI ribbon) across the trial timeline, baseline vs. proxy, at matched accuracy — the direct visual analogue of an RS time course.
2. **Participation-ratio bar/scatter plot**: PR per `(task_index, n)` cell, baseline vs. proxy, paired by cell — shows whether the "sharpening" pattern generalizes across task conditions or is specific to a subset.
3. **Accuracy-vs-activity scatter across training epochs**: one point per epoch for both baseline and proxy trajectories (accuracy on x, mean activation magnitude on y). This is the single most persuasive figure: it should show the proxy trajectory occupying an "upper-left" region (same/higher accuracy, lower activity) relative to baseline, directly visualizing the efficiency claim rather than asserting it from a table.
4. **Fano-factor distribution histogram**: per-unit variability ratio, baseline vs. proxy, overlaid — analogous to the trial-to-trial variability plots in the animal WM-training literature.

### 4.6 Implementation plan

- Add a new module `src/analysis/neural_efficiency.py` implementing the four metrics in §4.3 as pure functions operating on the `(X, sample_ids)` output of `build_matrix_with_metadata()` (already handles filtering by `task_index`/`n`/`property_name`, so no changes to the saving/loading pipeline are needed).
- Add `--analysis neural_efficiency` as a sixth option in `src/analysis/comprehensive_analysis.py`, following the existing pattern for the other five analyses, so it can be run as `python -m src.analysis.comprehensive_analysis --analysis neural_efficiency --model <ckpt> --hidden_root <exp>/hidden_states ...` alongside the existing pipeline.
- Output: a JSON summary (metric values + CIs per condition/cell) and the four plots in §4.5, written to the standard `analysis_results/<exp>/` directory used by the rest of the pipeline.

### 4.7 Caveats to state explicitly in the thesis

- RNN hidden-state L2 norm is **not** literally a BOLD or spike-rate signal; treat the mapping as an analogy licensed by the *functional* parallel (facilitated processing → suppressed representational magnitude), not a claim of biophysical equivalence. State this plainly, the same way the slide deck already flags "Verified vs. not independently verified" for its citations.
- Guard against the trivial confound that different training runs may simply produce different weight-norm scales (e.g., if the proxy-pretrained encoder converges to smaller overall weight magnitudes for unrelated optimization reasons). Before interpreting a magnitude difference as "efficiency," check whether normalizing hidden states by the model's own per-layer weight norm removes the effect — if it does, the finding is an optimization artifact, not a familiarity effect.
- If the matched-accuracy comparison (§4.2) shows the effect disappearing once accuracy is equated, report that honestly — it would mean the "efficiency" signature is a downstream consequence of better task performance rather than an independent hallmark of familiarity, which is still a usable (if weaker) finding, but a different claim than the one in §1.

---

## 4.8 Extension: attention-gate suppression as a third, independent signature

Everything above operates on RNN hidden-state activity (`hidden`), which is the only quantity the baseline model exposes. The attention-enhanced architecture (`AttentionWorkingMemoryModel` / `FeatureChannelAttention`, `src/models/attention.py`) exposes something strictly stronger: an explicit, literal per-channel gate in `[0, 1]` computed *before* the RNN even sees the CNN features. A gate near 0 on a task-irrelevant channel is not a proxy for suppression inferred from a norm — it **is** suppression, by construction. Unlike §4.1–4.7, this signature needs no external human-neuroscience reference to interpret: it is read off the model directly rather than argued by analogy, and it is a signature the plain GRU/RNN baseline cannot produce at all, which makes it a stronger candidate for "a new finding only our model can show" than the magnitude/PR/sparsity metrics alone.

**Status of the infrastructure (as of this writing):**

- `train_proxy.py` (proxy pretraining) and `finetune_from_proxy.py` (fine-tuning onto the real N-back task) already support `model_type: attention` end-to-end, including transferring the trained `attention` submodule's weights during fine-tuning (`finetune_from_proxy.py`, `transfer_proxy_weights()`). Ready-made configs already exist: `configs/proxy/proxy_attention_mtmf.yaml`, `configs/proxy/proxy_dual_attention_mtmf.yaml`.
- Until this document, gate values were **never saved to disk** — `evaluate_model()` in `train_with_generalization.py` and `finetune_from_proxy.py` never called `forward(..., return_attention=True)`, so saved payloads only ever contained `hidden`, `cnn_activations`, `logits`, and task/split metadata. This has now been fixed (see below), so newly-run experiments will save gates; existing attention-model experiment directories (e.g. `experiments/wm_h128_attention_mtmf_*`) do **not** have this data and would need a re-run to produce it.
- **Implemented in this pass:**
  - `AttentionWorkingMemoryModel.forward()` (`src/models/attention.py`) now supports `return_cnn_activations=True` and `return_attention=True` simultaneously (previously mutually exclusive), returning `(logits, hidden_seq, final_state, cnn_activations, gates)`.
  - `evaluate_model()` / `save_states_and_activations()` in both `train_with_generalization.py` and `finetune_from_proxy.py` now detect attention models via `hasattr(model, "attention")` (consistent with the existing check in `transfer_proxy_weights`), capture gates during validation, and save them under a new `"gates"` key (`(B, T, H)`) in the payload — `None` for non-attention models, so existing analysis code that ignores the key is unaffected.
  - `src/analysis/activations.py` gained two new functions mirroring the existing `build_cnn_matrix` pattern: `build_gate_matrix()` (decode a property from gate values instead of hidden states or CNN activations — plugs directly into the existing `decoding.py`/`orthogonalization.py` machinery) and `gate_channel_means()` (mean gate value per channel, pooled over trials matching a given `task_index`/`n`/`time` filter — the basis for the gate-suppression index below).
- **Not yet done** (deliberately out of scope for this pass): gate-saving in the proxy-*pretraining* loop itself (`train_proxy.py`'s `_save_proxy_states` / `ProxyWorkingMemoryModel.forward()`). The proxy task's per-trial format (single `task_feature`, no `task_index`/`n` metadata in the N-back sense) makes the suppression comparison below less directly interpretable during pretraining; the metric is intended to be computed during evaluation on the real N-back task (post fine-tuning), where `task_index`/`n` are meaningful. Re-running `train_proxy.py` itself is unaffected by this — it will simply keep omitting gates from its own saved payloads.

### 4.8.1 The gate-suppression index — implemented in `src/analysis/gate_suppression.py`

For a given task (`task_index`), channels are labeled task-relevant or task-irrelevant using the same decodability logic as Analysis 2 (`docs/ANALYSIS_METHODOLOGY.md` §3.2), but computed in **CNN-activation space**, not RNN space — the gate is applied to `cnn_activations` (the pre-RNN visual embedding; see the module docstring for why this distinction matters). Concretely, as implemented:

1. For each of the three properties, `channel_relevance_scores()` builds a per-channel relevance vector by training one-vs-rest decoders (`orthogonalization.one_vs_rest_weights`) on `cnn_activations` at each timestep and averaging `|weight|` across classes and timesteps.
2. `gate_suppression_index()` z-scores the task-relevant property's relevance vector and the (averaged) two task-irrelevant properties' relevance vectors, and subtracts them to get a per-channel **relevance contrast**. The top/bottom `top_frac` (default 25%) of channels by this contrast define the relevant/irrelevant channel sets — a continuous, k-free companion statistic (`gate_relevance_correlation`, the correlation between mean gate value and the contrast score) is reported alongside so the headline number doesn't hinge on the `top_frac` choice.
3. The **gate-suppression index** itself: mean gate value on the irrelevant channel set minus mean gate value on the relevant channel set (bootstrap CI via the same `bootstrap_ci` used in §4.3), expected negative — irrelevant channels gated down more than relevant ones.
4. `compare_gate_suppression()` computes this index for two conditions (e.g. attention-only vs. attention+proxy) **independently** — each condition ranks its own channel relevance from its own `cnn_activations`, since proxy pretraining/fine-tuning updates the trainable 1x1-conv projection ahead of the frozen ResNet50 backbone, so channel `c`'s meaning is not guaranteed to be stable across conditions. Only the resulting scalar indices are compared, not raw channel identities, and the JSON output states this explicitly (`"index_sharper_in_b"`, `"index_gap"`).
5. **Verified on synthetic data** (see execution plan): planted a known 0.5-vs-0.9 (condition A) and 0.1-vs-0.9 (condition B) irrelevant/relevant gate gap and recovered indices of −0.40 and −0.80 respectively (i.e. essentially exact recovery of the planted −0.4 and −0.8 gaps), with `index_sharper_in_b=True` and `index_gap≈0.40` matching the designed difference precisely. Not yet run on real model checkpoints.

The key comparison for this document's thesis remains: **does proxy pretraining sharpen this index** compared to an attention model trained from scratch on the real N-back task directly? If so, that is direct evidence that structured/familiar features cause the network to suppress irrelevant channels *more*, independent of and complementary to the RNN-hidden-state-magnitude story in §4.3.

### 4.8.2 Why this connects to (not replaces) your existing decoding/orthogonalization contribution

Analysis 2 in `docs/ANALYSIS_METHODOLOGY.md` already asks "can task-irrelevant properties still be decoded from the hidden state?" — STSF suppresses irrelevant decodability (<85%), MTMF does not (>85%, "mixed representations"). Phase 5's `compare_models.py` was explicitly built to re-run that same decoding/orthogonalization comparison for attention vs. baseline models, on the hypothesis that attention should *lower* task-irrelevant decodability further — but (per the local experiment audit) it has never actually been pointed at a real attention checkpoint's hidden states with matching baseline data to produce a reportable number.

Rather than treating "add attention" as a competing idea to the neural-efficiency proposal above, the strongest write-up combines three convergent signatures of the same underlying claim (familiarity/structure → suppression of what's not needed), each at a different level of the model:

| Level | Quantity | Where it comes from |
|---|---|---|
| Representational content | Task-irrelevant-feature decodability (existing Phase 3 `decoding.py`/`orthogonalization.py`) | Already built — needs to be run on attention vs. baseline hidden states |
| Population activity | RNN hidden-state magnitude / participation ratio / sparsity / Fano factor (§4.3 above) | Already scoped in this document — baseline vs. proxy-pretrained |
| Explicit gating | Gate-suppression index (§4.8.1) | New — needs the gate-logging fix (done) + a fresh attention+proxy training/fine-tuning run |

Reporting these three together — ideally on the *same* trained checkpoints where possible (e.g., the attention model fine-tuned from the proxy encoder) — is a substantially stronger case for "the model exhibits a genuine, multi-level suppression mechanism" than any one of them in isolation, and it directly answers your professor's ask for a finding beyond the accuracy/performance result already in the slide deck.

### 4.8.3 Practical next steps to produce data

1. Run (or check `hamrah-gpu-internal` for existing) `python -m src.train_proxy --config configs/proxy/proxy_attention_mtmf.yaml`, then `python -m src.finetune_from_proxy --proxy_exp_dir <that_dir> --config configs/attention_mtmf.yaml` (or the `configs_128/` equivalent) — this now saves gates automatically at evaluation time thanks to the fix above.
2. Run the equivalent *without* proxy pretraining — `python -m src.train_with_generalization --config configs/attention_mtmf.yaml` — as the "attention, no familiarity" control.
3. Compute the gate-suppression index (§4.8.1) for both, plus re-run `compare_models.py`'s decoding/orthogonalization comparison against the matching baseline (`configs/mtmf.yaml`) hidden states, so all three levels in the table above are available for the same scenario.

---

## 5. Optional follow-up: a causal test of "structure vs. mere exposure"

The existing deck (via Mercer, 2025) already argues informally that the proxy model's benefit should come from *feature structure*, not training volume, but never tests this. A natural, low-cost extension of the current proposal: pretrain a second proxy variant using the same architecture and the same number of gradient steps, but with **scrambled feature labels** (shuffle location/identity/category assignments during proxy pretraining only). If the neural-efficiency signature in §4.3–4.4 appears for the real proxy model but not the scrambled-label control (despite matched training volume), that is direct causal evidence that the effect tracks *meaningful structure*, not repetition count — closing the loop between this document and the caveat already recorded in the slide deck's "Caveats & What This Does Not Show" slide.

---

## 6. References

**Read in full for this document (verified):**
1. Poppenk, J., Moscovitch, M., & McIntosh, A. R. (2016). fMRI evidence of equivalent neural suppression by repetition and prior knowledge. *Neuropsychologia*, 90, 159–169.
2. Constantinidis, C., & Klingberg, T. (2016). The neuroscience of working memory capacity and training. *Nature Reviews Neuroscience*, 17(7), 438–449.

**Verified in the companion slide deck (cross-referenced, not re-verified here):**
3. Chung, Y. H., Brady, T. F., & Störmer, V. S. (2024). Meaningfulness and familiarity expand visual working memory capacity. *Current Directions in Psychological Science*, 33(5), 275–282.
4. Mercer, T. (2025). Familiarity influences on proactive interference in verbal memory. *Quarterly Journal of Experimental Psychology*.

**§4.8 (attention-gate suppression) cites no external reference by design** — the gate-suppression index is a direct read-out of the model's own gates, not an analogy to a human finding.
