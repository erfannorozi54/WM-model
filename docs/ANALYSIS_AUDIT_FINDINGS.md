# Comprehensive Analysis: Code Methodology Audit

> **This is a chronological log, not a statement of current state.** It records four
> audit passes in the order they happened, and later passes correct earlier ones —
> so reading it top-down gives you superseded conclusions first. For the
> neural-efficiency chapter (passes 3 and 4), the current truth is
> [`docs/NEURAL_EFFICIENCY.md`](NEURAL_EFFICIENCY.md). Passes 1 and 2 cover the
> paper-replication analyses and remain the record for those.

**Date**: 2026-06-15
**Scope**: All 18 experiments (9 original + 9 h128) audited against paper methodology.
**Status**: Code fixed, h128 experiments re-run with new code, original experiments re-running in background.

## Files Modified

| File | Changes |
|------|---------|
| `src/analysis/comprehensive_analysis.py` | H2 fix, sample-size warnings, SVC max_iter=10000 |
| `src/analysis/causal_perturbation.py` | Reverted to mean direction (was per-class — wrong direction) |
| `src/analysis/orthogonalization.py` | LinearSVC max_iter=10000, random_state=42 for determinism |
| `src/analysis/decoding.py` | SVC max_iter=10000 |
| `src/analysis/procrustes.py` | Full swap test, location for label alignment |

## Key Methodology Issues Found & Fixed

### 1. H2 Cross-Stimulus Test Was Testing Cross-Time (CRITICAL BUG)

**Before**: `_test_h2_cross_stimulus` was training decoder on t=0, testing on t=1,2,3.
This is the H1 test (slot-based memory), mislabeled as H2. All 18 experiments
showed "H3 POSSIBLE" because validation (t=0) >> generalization (t>0).

**After**: Uses val_novel_angle (known identities) for training/validation, val_novel_identity
(novel identities) for generalization, both at t=0. This is the proper cross-stimulus test.

**Finding**: With the corrected test, all 9 h128 experiments show `val >> gen`:
```
wm_h128_stsf:               val=1.000  gen=0.000  (H3_POSSIBLE)
wm_h128_stmf:               val=0.500  gen=0.228  (H3_POSSIBLE)
wm_h128_mtmf:               val=0.475  gen=0.237  (H3_POSSIBLE)
wm_h128_attention_stsf:     val=0.900  gen=0.290  (H3_POSSIBLE)
wm_h128_attention_stmf:     val=0.850  gen=0.087  (H3_POSSIBLE)
wm_h128_attention_mtmf:     val=0.562  gen=0.268  (H3_POSSIBLE)
wm_h128_dual_attention_stsf: val=0.887  gen=0.135  (H3_POSSIBLE)
wm_h128_dual_attention_stmf: val=0.775  gen=0.370  (H3_POSSIBLE)
wm_h128_dual_attention_mtmf: val=0.738  gen=0.260  (H3_POSSIBLE)
```

**Interpretation**: Our models show H3 (stimulus-specific encoding) rather than the
paper's H2 (shared encoding). This is a meaningful difference — our hidden states
for novel identities are very different from known identities, even at the same
location. The paper's models may generalize better because of different training
or model architecture.

### 2. Procrustes Swap Test (FIXED)

**Before**: Only computed consecutive disparities, no proper swap test.
**After**: Implements correct/swap1/swap2 with per-stimulus group split.

**Results (9 h128 experiments)**:
```
Experiment                       correct  swap1   swap2   Status
wm_h128_stsf                        0.994   0.241   0.223   NOT_CONFIRMED (high correct, both swaps low)
wm_h128_stmf                        0.294   0.235   0.256   CONFIRMED (paper pattern)
wm_h128_mtmf                        0.264   0.262   0.253   NOT_CONFIRMED
wm_h128_attention_stsf              0.263   0.253   0.261   CONFIRMED
wm_h128_attention_stmf              0.277   0.207   0.320   CONFIRMED (paper pattern)
wm_h128_attention_mtmf              0.319   0.303   0.343   NOT_CONFIRMED
wm_h128_dual_attention_stsf         0.098   0.257   0.181   CONFIRMED
wm_h128_dual_attention_stmf         0.345   0.248   0.239   NOT_CONFIRMED
wm_h128_dual_attention_mtmf         0.333   0.215   0.279   CONFIRMED
```

5/9 show paper pattern (swap2 > swap1), 4/9 don't. STSF model shows extreme pattern:
correct=0.99 (rotations work for same stimulus) but both swaps fail (rotations don't
transfer to new stimuli). This is consistent with H3 for STSF.

### 3. Causal Perturbation — Reverted Per-Class Fix

**Initial fix attempt**: Changed from mean direction to per-trial class-specific direction.
**Problem**: Per-class direction points TOWARD the class. Pushing a match trial in this
direction moves it DEEPER into the match class, INCREASING P(Match) — opposite of paper.
**Reverted to**: Mean direction (original approach).

**Results (9 h128 with mean direction)**:
```
Experiment                        P(Match)  drop%   P(No-Action)
wm_h128_stsf                       1.000→0.995   0.5%   0.000→0.000
wm_h128_stmf                       0.951→0.795  16.5%   0.000→0.000
wm_h128_mtmf                       0.858→0.618  27.9%   0.000→0.000
wm_h128_attention_stsf             0.495→0.512  -3.5%   0.000→0.000
wm_h128_attention_stmf             0.960→0.898   6.5%   0.000→0.000
wm_h128_attention_mtmf             0.943→0.939   0.4%   0.000→0.000
wm_h128_dual_attention_stsf        0.785→0.709   9.7%   0.000→0.000
wm_h128_dual_attention_stmf        0.964→0.884   8.3%   0.000→0.000
wm_h128_dual_attention_mtmf        0.921→0.828  10.1%   0.000→0.000
```

**Largest effects**: MTMF models show 10-28% P(Match) drops. STSF models show
minimal effects (model is too confident).

**Remaining limitation**: P(No-Action) does NOT rise. The paper expects 0.10→0.61
but we see 0.000→0.000. This is because the perturbation only runs through the
classifier, not the recurrent dynamics. To match the paper would require storing
the input sequence (cog_in) in the payload and re-running the recurrent module —
this is a training-time change that wasn't done in this audit.

### 4. SVC Convergence Warnings (FIXED)
All SVC/LinearSVC calls now use `max_iter=10000`. The 4-warning clusters seen in
old logs are gone in new logs.

### 5. Sample-Size Warnings (FIXED)
When `n_test < n_classes` or `n_test < 2 × n_classes`, the analysis prints a warning
informing the user that test accuracy is unreliable.

## Code Changes Summary

### `comprehensive_analysis.py` (lines 320, 458, 582, 583, 1024-...)
- `SVC(..., max_iter=10000)` in 3 places
- `train_test_split` with fallback for non-stratifiable data
- Sample-size warnings in `_analyze_task_relevance`
- New H2 cross-stimulus test using `val_novel_angle`/`val_novel_identity` splits
- Calls new `swap_hypothesis_test` in `_test_h2_procrustes_swap`

### `procrustes.py` (new function `swap_hypothesis_test`)
- Splits by `identity` hash for cross-stimulus effect
- Decodes on `location` (4 fixed classes) for label alignment between groups
- Reports correct/swap1/swap2/baseline accuracies
- Tests hypothesis: `|swap2 - correct| < |swap1 - correct|`

### `causal_perturbation.py`
- Reverted to mean direction (was per-trial class-specific — wrong direction)
- Kept `property_values` and `label2idx` parameters for API compatibility
- Added documentation explaining why mean direction is correct

### `orthogonalization.py` and `decoding.py`
- `LinearSVC(..., max_iter=10000, random_state=42)` for reproducibility
- `SVC(..., max_iter=10000, random_state=42)` where applicable

## Inspection Locations (all on local machine)

| What | Where |
|------|-------|
| Analysis outputs (JSON) | `/home/erfan/Projects/WM-model/analysis_results/<exp>/*.json` |
| Analysis plots (PNG) | `/home/erfan/Projects/WM-model/analysis_results/<exp>/*.png` |
| Analysis log (recent) | `/tmp/analysis_rerun.log` |
| Source code | `/home/erfan/Projects/WM-model/src/analysis/*.py` |
| This document | `/home/erfan/Projects/WM-model/docs/ANALYSIS_AUDIT_FINDINGS.md` |

## Reproducing the Audit

```bash
# On GPU server
ssh hamrah-gpu-internal
cd ~/Projects/WM-model
export PYTHONPATH=src:$PYTHONPATH

# Re-run a single experiment
~/.venv/WM-model/bin/python -m src.analysis.comprehensive_analysis \
  --analysis all \
  --hidden_root experiments/wm_h128_stsf_20260602_230425/hidden_states \
  --output_dir /tmp/test_out \
  --property identity \
  --model experiments/wm_h128_stsf_20260602_230425/best_model.pt
```

## Open Questions / Future Work

1. **Recurrent dynamics in causal perturbation**: Need to store `cog_in` in the
   payload during training to re-run the recurrent module. This would let
   P(No-Action) rise as the paper expects.

2. **H2 vs H3 discrepancy**: Our models show H3 (stimulus-specific) rather than
   the paper's H2 (shared encoding). This could be due to:
   - Different model architecture details
   - Different training regime
   - Different stimulus set
   - Different cross-stimulus test methodology in the paper

3. **Sample sizes**: With `num_val=200` and identity property having 70 classes,
   test sets are smaller than 2× n_classes. The new warnings flag this. To
   improve reliability, increase `num_val` in configs.

4. **Original experiments re-run in background**: Original 9 experiments
   (wm_stsf through wm_dual_attention_mtmf) are re-running with the new code
   for direct comparison with the h128 experiments.

---

# Second Audit (2026-08-16): Class-Index Misalignment

**Trigger**: the deck's "results incompatible with the paper" section — chiefly
the H2 cross-stimulus ❌ ("our models are stimulus-specific, the paper's are
not") — was traced to the analysis code, not to the models.

## Root cause

`build_matrix*` in `src/analysis/activations.py` assigned class indices **in
order of first appearance within that call**. Every analysis that trains on one
matrix and scores against another matrix's labels therefore compared indices
that meant different classes in the two matrices. Sorting by raw value fixes the
index space; the affected call sites now also pass an explicit shared
`label2idx`.

### Evidence it was an artifact, not a finding

| Result | Value | Why it cannot be real |
|--------|-------|-----------------------|
| `wm_h128_stsf` H2 | val=1.000, **gen=0.000** on n=2000 | A decoder that is perfect on held-out data cannot score *exactly* zero on a 4-class problem; a failed decoder floors at chance (0.25). Exactly 0.0 is a derangement of the class labels. |
| `wm_h128_stsf` swap1 | **0.000** | Same signature. |
| `wm_stsf` swap1 | 0.041 | Sub-chance. |
| `wm_mtmf` swap | correct=0.222 vs **baseline=0.861** | `baseline` is the one quantity computed from weights and labels built in a *single* call; everything built across calls collapsed to chance. |

Reproduced on synthetic data where both splits are geometrically identical
(i.e. perfect H2 by construction): the pipeline reported `val=1.000,
gen=0.000 → H3 POSSIBLE`. After the fix: `val=1.000, gen=1.000 → H2 SUPPORTED`.
Likewise a Procrustes `swap2` whose true value is 1.0 was reported as 0.0.

## Fixes

| File | Change |
|------|--------|
| `activations.py` | `make_label2idx()` — value-sorted class indices; all `build_*` use it; optional `label2idx=` to force a shared space; new `build_matrix_tracked()` |
| `comprehensive_analysis.py` | H2 test builds the novel-identity matrix in the training decoder's class space; H1 rewritten (below); consecutive-Procrustes weights share one class space; `_ensure_data_loaded()` so analyses 3/4 use the best epoch like analysis 2 |
| `procrustes.py` | `swap_hypothesis_test` + `procrustes_analysis` build every matrix in one shared class space, and warn if group class sets differ; `_stable_hash` replaces builtin `hash()` (PYTHONHASHSEED made group A/B assignment differ per run); `cnn_activations`/`gates` now masked with the rest of the payload |
| `orthogonalization.py` | `one_vs_rest_weights(..., input_space=True)` maps the normal out of StandardScaler space |
| `causal_perturbation.py` | uses `input_space=True` (the perturbation is added to *raw* hidden states); distances are now in SDs of the hidden states' projection onto the direction, so they are comparable across models |

## H1 cross-time was measuring the wrong thing

`_test_h1_cross_time` documented "train on E(S=1,T=1), test on M(S=1,T=2..6)"
but labelled each test state with the stimulus **on screen at that timestep**,
not the tracked item from t=0. It also reported the t=0 point as the decoder's
own training accuracy, so the headline "98% → 5%, 96pp drop" compared an
in-sample number against out-of-sample ones.

Now: an 80/20 trial split scores every timestep including t=0, `accuracies` is
the tracked item (the actual H1 test), and the previous quantity is kept as
`accuracies_current_stimulus`. `chance_level` is reported — with 72 identity
classes chance is 0.014, so the old "collapses to 1-6%" was partly *at* chance.

## Not bugs, but they invalidate comparisons in the deck

1. **`num_val` differs 5× between the two config sets**: `configs/*.yaml` uses
   `num_val: 400`, `configs_128/*.yaml` uses `num_val: 2000`. Decoder sample
   size, not hidden size, dominates the difference between the `wm_*` and
   `wm_h128_*` rows (identity decoding 0.34-0.48 vs 0.80-0.84). Any h256-vs-h128
   table is confounded until they are re-run with matched `num_val`.
2. **Deck numbers are stitched from superseded analysis runs**: e.g. the
   Analysis 2A matrix and the h128 rows of Analysis 2B do not match any JSON
   currently in `analysis_results/`. Every table needs regenerating from one
   analysis pass.
3. **Location is weakly decodable by construction**: `PerceptualModule` global-
   average-pools the reduced feature map, discarding spatial layout before the
   RNN ever sees it. Location decoding ~0.50 (chance 0.25) while category ~0.96
   follows from the architecture, so "off-diagonal is not all >85%" is a
   statement about the encoder, not about task-relevance gating.
4. **Analysis 5 only runs `model.classifier`**, not the recurrent step — already
   noted in the deck, still true.

---

# Third Audit (2026-08-16): Neural-Efficiency Chapter (Levels 1–3)

Scope: the second half of `slidev-presentation/slides.md` — the chapter graded
against **Poppenk, Moscovitch & McIntosh (2016)** and **Constantinidis &
Klingberg (2016)** — plus `neural_efficiency.py`, `gate_suppression.py`,
`compare_models.py`. The trigger was the same question as the first two audits:
*are the places where our results disagree with the papers real, or artifacts?*

**Headline: the two ❌ rows in the Level 2 table are not both real.** One of them
grades our result against a prediction that was derived with the sign flipped
from what the review actually reports; the other survives, and is in fact
understated. Separately, the Level 3 headline result is confounded by an
unfiltered epoch pool, and Level 1 was never run on a trained checkpoint.

## A. Mis-derived reference predictions (comparison errors, not code bugs)

| Level 2 row | Deck said | Actually |
|---|---|---|
| Activation magnitude ↓ | "✅ matches" under a **vs. Reference 2** column | Matches **Reference 1** (Poppenk). Reference 2 §4 reports the *opposite* for single neurons — more PFC neurons recruited, mean firing rate *up*. The column header grades all four rows against Ref 2 while the cell text cites Ref 1. |
| Population sparsity ↑ | "✅ matches" vs. Reference 2 | Neither reference predicts sparsity. "Efficient ⇒ sparser" is our own assumption (`FUTURE_WORK` §4.3 item 3). Ref 2's "more neurons recruited" argues for *less* sparse. |
| Participation ratio ↑ | "❌ opposite of prediction" | **The prediction is unsound; withdraw the ❌.** `PAPER_EXPLAINED_CONSTANTINIDIS_KLINGBERG_2016.md` §8 derived "lower PR" from "sharpened tuning", but §4 of the same document reports tuning gets **broader** after training. And PR is a *population* effective-dimensionality measure, not a *single-unit* tuning-width measure — more neurons recruited and a newly multiplexed rule signal push PR up, redundant broad tuning pushes it down, so the review makes no determinate PR prediction. |
| Fano analogue ↑ | "❌ opposite of prediction" | **Real, and understated.** Ref 2 does cleanly predict lower Fano. Moreover `Var/Mean` on a continuous signal scales *linearly* with activity magnitude, and the proxy condition has *lower* magnitude — so a pure scale effect would have pushed this metric *down*. Observing it go *up* anyway means the scale-invariant CV² moves further still. |

Net: **Level 2 has one genuine contradiction with the literature (Fano), not
two**, and the magnitude/sparsity ✅s should be re-attributed to Reference 1 /
to our own assumption respectively.

To actually test Reference 2's tuning claim you need a **per-unit selectivity**
metric (per-hidden-unit decoding accuracy, or a selectivity index across
stimulus conditions); the prediction there is that selectivity gets *broader*.
Not implemented, not run.

## B. Estimator biases, verified numerically

Same underlying distribution in every row below — all differences are estimator
artifacts (`/home/erfan/.claude/jobs/6028ae81/tmp/metric_bias_check.py`):

| Metric | N=50 | N=400 | N=1600 | Consequence |
|---|---:|---:|---:|---|
| `participation_ratio` | 14.4 | 20.0 | 20.7 | +44% from sample size alone |
| `population_sparsity` | 0.099 | 0.126 | 0.141 | +42% from sample size alone |

`fano_factor_analogue` used `np.var(..., ddof=0)`, which underestimates variance
by `(g-1)/g` — a **33% downward bias** at the `min_group_size=3` floor, and a
bias that differs between conditions whenever their group sizes differ.

**Before quoting the PR or sparsity rows, check `n_trials_a == n_trials_b` in
`neural_efficiency.json`.** The code now emits `trial_count_warning` when they
differ. Magnitude and CV² are unaffected by both biases.

## C. Level 3 (the headline) is not accuracy-matched as the deck claims

The `gate_suppression.py` command in `docs/archive/EXECUTION_PLAN_NEURAL_EFFICIENCY.md` §3/§4
passes **no `--epoch_a`/`--epoch_b`**, so `load_payloads(epochs=None)` pooled
*every saved epoch* of both runs. Condition A (`wm_attention_mtmf_20260726_161735`)
is a from-scratch run contributing many near-initialization checkpoints;
condition B (`finetune_proxy_wm_attention_mtmf_20260726_201707`) is a short
fine-tune contributing only already-converged ones. "Attention+proxy gates more
sharply, 9/9 cells" is therefore partly a statement about **training maturity**,
not about the two trained models.

The "93.43% vs 93.51%" accuracy match quoted on the Level 3 slide was selected
by `select_matched_epoch` for the **Level 2** run and does **not** transfer to
Level 3 as executed.

**Verify in one command:** `epoch_a` / `epoch_b` in
`analysis_results/gate_suppression_mtmf/gate_suppression.json` should be `null`.
Re-run with explicit epochs (ep43 vs ep1) before the claim can stand.

Related: both models are `attention_mode: "task_only"`, so the gate is a pure
function of the task vector — every trial in a `(task, n)` cell from one
checkpoint carries an **identical** gate vector. The trial-level bootstrap CI is
therefore zero-width and carries no information; epoch pooling was the only
thing injecting variability into it, which is the confound itself. The code now
reports `n_distinct_gate_vectors` / `ci_degenerate` / `ci_warning`.

## D. Level 1 was never run on a trained checkpoint

`compare_models.py` called `decoding_evaluate` / `orthogonalization_evaluate` /
`procrustes_analysis` / `swap_hypothesis_test` **without `epochs=` or `split=`**
— the same class of bug as gotcha 1, in the one tool that had no way to filter.
The reported "identity decodability 14.6/12.0/10.1% → 7.2/6.5/5.9%" is averaged
over each run's entire training trajectory and over both validation splits.

Also on Level 1:
- **No chance level reported.** Pooling both splits gives ~72 identity classes,
  so chance ≈ **1.4%**. Both models are above chance, so the direction survives
   — but the deck should state the floor, per the discipline adopted for
  Analysis 4 H1 in the second audit.
- **`--task` was not passed**, so all three task contexts are pooled. In MTMF,
  identity is the *task-relevant* feature on a third of trials, which dilutes an
  "irrelevant-feature suppression" claim. Use `--task location` / `--task
  category` for a clean test.
- `test_times` includes `train_time=2`; that entry is in-sample, not held out.
  The deck correctly omits it; the JSON now labels it.

## Fixes applied

| File | Change |
|---|---|
| `neural_efficiency.py` | `ddof=1` in the Fano analogue; new `coefficient_of_variation_squared()` (scale-invariant); `n_groups_used_*` / `mean_group_size_*`; `trial_count_warning` on unequal N; `DIMENSIONALITY_CAVEAT` for PR |
| `gate_suppression.py` | epoch-pooling warning (stdout + `epochs_pooled` / `epoch_warning` in JSON); `n_distinct_gate_vectors`, `ci_degenerate`, `ci_warning` |
| `compare_models.py` | `--best_epoch` / `--baseline_epochs` / `--attention_epochs` / `--split`; per-model epoch plumbing through all four sub-analyses; chance level + class counts in the decoding output; in-sample `train_time` note |
| `decoding.py`, `orthogonalization.py`, `procrustes.py` | `split=` parameter plumbed to `load_payloads` (was unreachable from these entry points) |
| `PAPER_EXPLAINED_CONSTANTINIDIS_KLINGBERG_2016.md` §8 | rewritten: states which single prediction the review licenses (Fano ↓) and why the magnitude and PR predictions were wrong |
| `docs/archive/FUTURE_WORK_NEURAL_EFFICIENCY.md` §4.3 | items 2/3/4 corrected (PR ungraded + N-biased; sparsity is our assumption; Fano needs ddof=1 + CV²) |

## Re-runs required before the chapter's numbers can be quoted

1. **Level 3** — `gate_suppression.py` with `--epoch_a 43 --epoch_b 1` (the
   accuracy-matched pair). This is the headline result; it is the priority.
2. **Level 1** — `compare_models.py --best_epoch --split val_novel_identity
   --task location` (and `--task category`), for a trained-checkpoint,
   irrelevant-feature-only comparison.
3. **Level 2** — re-run for `cv_squared` and the corrected Fano; check
   `n_trials_a == n_trials_b` before quoting PR/sparsity.

---

# Fourth pass (2026-08-22): the Third Audit checked against the real result JSONs

The Third Audit above reasoned largely from the code. This pass re-derived its
claims from the **actual output files** —
`~/.claude/jobs/7ae0bd7c/tmp/{neural_efficiency_baseline_vs_proxy_mtmf,neural_efficiency_attention_pair,gate_suppression_mtmf,compare_baseline_vs_attention_mtmf}.json`
— plus a numerical bias check at the trial counts and PR values those files
actually contain. Three of its claims were overstated; three new issues surfaced.

## Confirmed as written

- **Fano ↑ is a genuine contradiction of Ref 2, and understated.** Higher in
  18/18 cells. Scale-correcting by the magnitude ratio strengthens it in every
  cell (CV²-equivalent ratio 1.08–7.16), because the quieter proxy condition
  should have driven `Var/Mean` *down*.
- **PR should not be graded against Ref 2** — for the construct reason
  (population dimensionality ≠ single-unit tuning), not because the effect is
  suspect. See correction 2.
- **Level 3 pooled all epochs.** `epoch_a: null, epoch_b: null` confirmed in the
  JSON. ~45 checkpoints per condition (12150 trials / 270 per epoch).
- **Level 1 ran unfiltered.** No epoch field, `"task": null`. Chance for identity
  is 1/72 = **1.39%**, unreported in the deck.
- Magnitude lower in 18/18 cells; every deck number for Level 3 matches the JSON
  exactly (A −0.171→+0.070, B −0.520→−0.333, corr 0.09–0.24 vs 0.45–0.72, 9/9).

## Corrections to the Third Audit

1. **The `ddof=0` bias is ~3% here, not 33%.** The 33% figure is the worst case
   at the `min_group_size=3` floor. Real group sizes are 9–13 trials
   (`n_trials / n_groups`), and the *between-condition differential* — the only
   part that can distort a comparison — is 0.97–1.03 across all 18 cells. The
   `ddof=1` fix is still correct; it just does not change any conclusion.
   *Exception:* identity cells have mean group size 1.7–1.8, so most groups fall
   below the floor and are dropped. Those Fano values rest on a selected minority
   of groups and cannot be assessed without the raw payloads.
2. **The sample-size bias does not threaten the PR result.** Measured on
   synthetic data at the PR values and trial counts actually present
   (PR 1.2–9.8, N 204–324), N-induced drift is **0–2%** for PR and 2–4% for
   sparsity — against observed effects of +2.6% to +605%. The Third Audit's
   "+44%" came from a regime (true PR ≈ 20, N = 50) that does not occur in this
   data. Independently: in **11/18 cells the proxy condition has *fewer* trials
   yet higher PR**, and one cell with exactly equal N (258 vs 258) shows +76%.
   Keep the `trial_count_warning`, but the PR effect is real.
   Sparsity is weaker — its three smallest cells (−5.3%, +11.7%, +14.8%) are
   within a few multiples of the drift.
3. **Level 3's confound is maturity, not checkpoint count.** The audit says
   condition B "contributes only already-converged" checkpoints. Both conditions
   pool ~45 epochs. The confound is that A trains from scratch (its pool includes
   near-initialization checkpoints) while B fine-tunes from a pretrained model and
   is converged by epoch 1. Same conclusion, different mechanism.

## New issues

4. **The swap-test row is not an identity measurement.** `swap_hypothesis_test`
   hardcodes `swap_property = "location"` — deliberately and correctly, since
   identity labels are unique per trial and cannot be aligned across the two
   disjoint stimulus groups. But `compare_models.py::compare_swap_test` reports
   `'property': property_name` (i.e. `"identity"`) and drops the function's own
   `verdict` / `note` fields, so the deck presented a **location**-decoding number
   as identity-suppression evidence. Row struck from the Level 1 slide.
5. **`p<0.0001` is unattainable at `n_boot=1000`.** `bootstrap_difference` computes
   `p = 2·min(prop_below_zero, 1−prop_below_zero)`, so the finest resolvable value
   is 2/1000. A stored `0.0` means "no resample crossed zero" → report **p < 0.002**.
   Deck corrected.
6. **The Level 2 "matched" pair is epoch 43 vs epoch 1** — condition B is one epoch
   into fine-tuning, so the two conditions differ in training maturity even where
   accuracy matches. Separately, `select_matched_epoch` defaults to
   `metric_key="val_novel_angle_acc"` while the runs used `--split val_novel_identity`;
   if the default was used, the accuracy match is on a different split than the
   hidden states analyzed. Worth pinning explicitly on any re-run.

## Not verifiable locally

The accuracy figures themselves (93.43/93.51, 82.7/92.7) live in `training_log.json`
on the GPU server; the relevant `experiments/` directories are not on this machine.
Also note the baseline-vs-proxy pair's location cells have PR_a ≈ 1.19–1.41 (near
rank-1), so that pair's 5–6× ratios come from a degenerate condition-A
representation, not a strong proxy effect — quote the attention pair instead.

---

# Fifth pass (2026-08-26): the Fourth pass's "worth pinning explicitly" is now measured

## Issue 6's second half confirmed: the epoch pairs were matched on the wrong split

The Fourth pass suspected that `select_matched_epoch`'s default
(`metric_key="val_novel_angle_acc"`) meant the accuracy match happened on a
different split than the one analysed. The GPU server became reachable and the
`training_log.json` figures are now measured directly:

| Pair | Checkpoints | Novel-angle gap (what the pins were selected on) | Identity gap (the analysed split) |
|---|---|---|---|
| baseline vs. proxy | ep12 vs ep1 | 10.0pp (82.69 / 92.67) | **8.9pp** (81.17 / 90.10) |
| attention-only vs. attention+proxy | ep43 vs ep1 | 0.08pp (93.43 / 93.51) | **0.84pp** (91.75 / 92.59) |

Every document quoting "10pp" or "0.08pp" as the accuracy match of these runs
was wrong; all have been corrected (`run_neural_efficiency.sh`,
`docs/NEURAL_EFFICIENCY.md`, `docs/RESULTS.md`, the deck, its speaker notes,
and this directory's README).

Two structural facts established in the same pass:

1. **The baseline pair cannot be fixed by re-pinning.** Proxy epoch 1 already
   exceeds every baseline checkpoint on the analysed split (baseline identity
   ceiling: 82.53% at ep17). No accuracy-matched baseline pair exists; the
   8.9pp figure is the best possible. This strengthens the existing guidance to
   quote the attention pair.
2. **A strict identity-matched attention pair exists**: ep18 vs ep8, 0.44pp
   (91.91 / 92.35). Adopting it would invalidate every published Level 2/3
   artifact (all rest on 43/1) and requires a full chapter re-run — a deliberate
   decision, recorded here as open rather than taken silently.

## Disposition

Pins unchanged (12/1, 43/1); the record now states which split each number comes
from. The Box-2 control stands on the attention pair (sub-1pp on the analysed
split); the baseline pair must be presented as structurally unmatched, not as a
10pp match.
