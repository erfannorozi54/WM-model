# Neural Efficiency: the chapter, its evidence, and its limits

**This is the single authoritative document for the neural-efficiency chapter.**
Where it disagrees with `docs/archive/FUTURE_WORK_NEURAL_EFFICIENCY.md`,
`docs/archive/EXECUTION_PLAN_NEURAL_EFFICIENCY.md`, or any earlier pass in
`docs/ANALYSIS_AUDIT_FINDINGS.md`, this document wins — those are historical
records of how the work got here, not statements of what is currently true.

Last verified against the real result artifacts: **2026-08-26**.

---

## 1. The claim

Proxy pretraining and explicit attention gating both **suppress task-irrelevant
processing** in the model, and that suppression is measurable at three
independent levels.

**What this is not.** It is not a capacity claim. The proxy-pretraining accuracy
gain already in the deck (novel angle 82.7% → 97.5%, novel identity 80.6% →
92.8%) is a **performance** result at unchanged N-back levels — we never tested
whether the model can hold more items or succeed at higher N. Capacity and
efficiency are separate, independently measured phenomena in the human
literature, and only the second is what this chapter tests.

---

## 2. The two references, and exactly what each licenses

Both were read in full from source and are summarised in
`docs/PAPER_EXPLAINED_POPPENK_2016.md` and
`docs/PAPER_EXPLAINED_CONSTANTINIDIS_KLINGBERG_2016.md`. Read those before
re-deriving anything here.

| | Reference 1 — Poppenk et al. (2016) | Reference 2 — Constantinidis & Klingberg (2016) |
|---|---|---|
| What it is | Single fMRI experiment, n=17 | Review synthesising many monkey/human studies |
| Manipulation | Prior knowledge vs. recent repetition vs. novel | Weeks of working-memory **training** |
| Licenses for us | **Magnitude ↓** — prior knowledge suppresses processing-related activity regardless of how the knowledge was acquired | **Fano ↓** — trial-to-trial variability drops with training |
| Also supplies | — | **Box 2**: never read "activity went down" as "more efficient" without confirming comparable accuracy |
| Does **not** license | — | Magnitude (its §4 reports firing rate going **up**); participation ratio (no determinate direction) |

**Three attribution rules that the deck previously got wrong:**

1. The magnitude result is graded against **Reference 1**, not Reference 2.
   Reference 2 predicts the opposite for single neurons.
2. **Sparsity is our own assumption**, not a prediction from either paper.
   Reference 2's "more neurons recruited" arguably argues for *less* sparse.
3. **Participation ratio is ungraded.** PR measures *population* effective
   dimensionality; Reference 2's tuning result is a *single-unit* property, and
   the review reports tuning getting **broader**, not sharper. It makes no PR
   prediction in either direction.

### Why these two papers were kept rather than replaced

The Fano divergence (§4) is a real disagreement with Reference 2, so replacing
that reference was considered. It was kept, for three reasons:

- Reference 1 is a clean, direct match to our manipulation — proxy pretraining
  is knowledge acquired elsewhere, which is exactly its "known for a lifetime"
  condition — and our magnitude result confirms it in 18/18 cells.
- Reference 2 supplies Box 2, which is the design discipline the entire chapter
  rests on. Losing it would cost more than the one failed prediction gains.
- A reference that yields one falsifiable prediction we then **fail** is
  stronger evidence of an honest test than one that agrees with everything.

The most likely reason for the divergence is a **manipulation mismatch**, and
the chapter should say so plainly: Reference 2 measures weeks of repeated WM
training on the same task, while we measure prior knowledge transferred from a
*different* task. Those are different interventions, and Reference 1 exists
precisely because the two are known to behave differently.

### A third reference pair was removed (2026-08-22)

Desimone & Duncan (1995) and Treue & Martínez-Trujillo (1999) were cited for
Level 3 but had only been read as summaries, not in full — the standard applied
to References 1–2. They were removed everywhere rather than left as
lower-confidence citations. **Level 3 needs no external reference**: the
attention gates are a literal, built-in suppression signal, read off the model
directly rather than argued by analogy.

---

## 3. Method — three levels

| Level | What it measures | Tool | Comparison |
|---|---|---|---|
| 1. Representational content | Task-irrelevant-feature decodability, orthogonalization | `compare_models.py` | Baseline vs. attention |
| 2. Population activity | Magnitude, participation ratio, sparsity, Fano analogue + CV² | `neural_efficiency.py` | Baseline vs. proxy-pretrained (×2 pairs) |
| 3. Explicit gating | Gate-suppression index, gate-relevance correlation | `gate_suppression.py` | Attention-only vs. attention+proxy |

Three levels rather than one because each is independently falsifiable; a
mechanism that shows up in all three is a stronger claim than any single metric.

**Matched-accuracy design (Box 2).** Every comparison reports the accuracy gap
between conditions, so an activity difference cannot be dismissed as "that model
was simply more accurate." The epoch pair is selected by closest accuracy **on
the same split whose hidden states are analysed** — matching on a different
split would silently break the control.

---

## 4. Results, with confidence levels

Verified directly from `analysis_results/neural_efficiency/2026-08-22_audit-fixed/`,
produced on `hamrah-gpu-internal` after the audit fixes landed (commit `17499de`,
2026-08-22 14:34 UTC; runs at 14:37-14:42 UTC), with Level 1 regenerated on
2026-08-23 07:13 UTC under `820a8f8` for the swap-test labelling fix.

### Level 2 - Population activity - **corroborates the framework**

Two pairs: baseline vs. proxy (epochs 12/1) and attention-only
vs. attention+proxy (epochs 43/1). Same direction in both, all 18 cells. The
epoch pairs were selected by closest *novel-angle* accuracy — the figures the
script comments once quoted as "10pp" and "0.08pp". On the split actually
analysed (`val_novel_identity`) the same checkpoints sit **8.9pp** (baseline:
81.17 vs 90.10) and **0.84pp** (attention: 91.75 vs 92.59) apart.

The baseline gap is **irreducible**: proxy epoch 1 already exceeds every
baseline checkpoint's identity accuracy (ceiling 82.53% at ep17), so no
accuracy-matched baseline pair exists on the analysed split. This is a property
of the manipulation, not a pinning error — and it is one more reason the
attention pair is the primary evidence. A strict identity-matched attention
pair would be 18/8 (0.44pp); re-pinning would invalidate every published
artifact below and is left as a deliberate decision, not taken silently.

| Metric | Result | Graded against | Confidence |
|---|---|---|---|
| Activation magnitude | **Lower**, 18/18 | Ref 1 | Solid - replicates at a sub-1pp accuracy gap |
| Fano analogue | **Higher**, 18/18 | Ref 2 - genuine contradiction | Solid |
| CV-squared (scale-invariant) | **Higher**, 18/18 | confirms Fano | Solid - rules out the magnitude-scaling artifact |
| Participation ratio | **Higher**, 18/18 | ungraded | Solid as a result |
| Population sparsity | **Higher**, 17/18 | our own assumption | Weak - small effects |

The CV-squared row is the important addition over the 2026-07-27 run: `Var/Mean`
scales with activity and the proxy condition is quieter, so a pure scale artifact
would have pushed Fano *down*. Both Fano and its scale-invariant companion rise
in every cell, so the increase in relative variability is real.

Caveats: identity cells have small Fano groups (mean size below the
`min_group_size=3` floor in the earlier run), so those values rest on a subset of
groups. The baseline pair's location cells have PR_a ~ 1.2 (near rank-1), which
inflates that pair's ratios - **quote the attention pair**.

### Level 1 - Representational content - **corroborates conditionally**

Baseline ep17 vs. attention ep25, `val_novel_identity`, run separately per task
context. Chance is ~3.2% (31-32 identity classes survive filtering).

| Sub-metric | Baseline | Attention | Read |
|---|---:|---:|---|
| decodability t=3/4/5, **task=location** | 28.5/18.0/15.3% | 5.3/2.3/5.3% | Collapses to ~chance |
| decodability t=3/4/5, **task=category** | 20.5/15.9/15.9% | 15.4/19.2/16.5% | No suppression |
| Orthogonalization (loc / cat) | 0.939 / 0.944 | 0.946 / 0.944 | Flat, ceiling |
| Procrustes reconstruction (loc / cat) | 92.7% / 100% | 83.0% / 93.5% | Lower under attention |

Filtering made this leg *sharper*, not weaker: the pooled run reported a uniform
"roughly halved", which averaged a near-total suppression in the location context
together with none at all in the category context.

**The swap-test row is withdrawn.** `swap_hypothesis_test` always decodes
**location** - deliberately, since identity labels are unique per trial and cannot
be aligned across the two disjoint stimulus groups it requires. It was reported
under `property: identity`. The artifacts now carry `decoded_property`,
`grouping_property` and an `interpretation_warning` instead, and the per-model
`verdict` (`h2_not_supported` for baseline, `uninformative` for attention) is no
longer dropped.

### Level 3 - Explicit gating - **corroborates partially**

With epochs pinned to the accuracy-matched pair (43 vs. 1):

| Suppression index | Attention-only | Attention+proxy |
|---|---:|---:|
| location n=1/2/3 | -0.48 / -0.43 / -0.52 | -0.51 / -0.49 / -0.49 |
| category n=1/2/3 | -0.48 / -0.45 / -0.42 | -0.48 / -0.53 / -0.48 |
| identity n=1/2/3 | +0.07 / +0.13 / +0.03 | -0.10 / +0.11 / +0.18 |

`index_sharper_in_b` in **6/9 cells**, with small gaps. Gate-relevance correlation
does improve consistently (0.66-0.73 to 0.84-0.85 for location; 0.46-0.50 to
0.54-0.58 for category; ~0 for both on identity).

**Why the checkpoint pin matters.** Reading both models at a single pinned
checkpoint is load-bearing here, not housekeeping. Condition A trains from
scratch, so pooling its checkpoints folds in near-initialisation gates that
condition B - a fine-tune, converged at epoch 1 - never contributes. Pooled, the
same comparison reports 9/9 cells and an attention-only model that appears
barely to gate at all; that figure is an artifact of training maturity and must
not be quoted. Pinned, attention-only already gates strongly on location and
category, and neither model gates on identity.

`ci_degenerate` is `True` in every cell, as predicted: both models are
`attention_mode: "task_only"`, so the gate is a pure function of the task vector
and every trial in a cell from one checkpoint carries an identical gate. Epoch
pooling was the only thing that had been injecting variance into that CI - which
was the confound itself. This is correct behaviour, not a bug.

### Level 2, second reading - **the two modifications are not the same mechanism**

Added 2026-09-02, from `analysis_results/neural_efficiency/2026-08-26_2x2/`.

The two pairs above both vary the *training regimen* inside one architecture, so
each modification is confirmed only against its own control and the chapter
reads them as one framework. The same four models also form a 2x2, and its other
two cells vary the *architecture* inside one regimen:

| Row | A | B | Epochs | Accuracy gap (`val_novel_identity`) |
|---|---|---|---|---|
| scratch | baseline | attention | 17 / 9 | 0.88pp (82.53 vs 81.65) |
| proxy | baseline+proxy | attention+proxy | 20 / 45 | **0.00pp** (93.55 vs 93.55) - the tightest match in the chapter |

| Metric | scratch row (attention alone) | proxy row (attention on top of proxy) |
|---|---|---|
| Activation magnitude | **lower 9/9**, CI excludes 0 in 8/9 | lower 7/9 (6/9); two identity cells null at p=0.73, p=0.61 |
| Participation ratio | **lower 9/9** (8/9); still 6/6 after dropping the three location cells, which are near-rank-1 (PR 1.1-1.4) and inflate ratios. Drops are large: 5.4->4.4, 6.8->5.1, 6.8->4.5, 5.4->4.8, 7.0->5.5, 7.0->4.4 | **4/9 - no effect** |
| Population sparsity | mixed, 3/9 | *higher* in 8/9 (6/9) - the opposite direction |

**The result: attention reshapes the population code only when it is the only
modification, and that effect is absorbed once proxy pretraining is present.**
This is the population-code counterpart of the accuracy interaction - attention
alone and proxy alone each buy roughly +9 to +11pp on novel identity, and
stacking them buys nothing further. Two different mechanisms are not being
composed; the second one is arriving at a code the first has already produced.

Trial-count guard: PR and sparsity are biased upward by sample size and the
cells are unequal, but in the scratch row condition B has *more* trials in 5 of
9 cells and still shows lower PR, so the bias runs **against** the observed
effect. PR also grows with training epoch, so compare directions within a row
only, never PR magnitudes across rows.

**No Fano direction may be read from these two files.** Fano falls 7/9 in the
scratch row, but `cv_squared` is 5/9 - a coin flip. Attention lowers magnitude
and the Fano analogue is scale-dependent (see the caveat carried in the JSON),
so a Fano drop with no CV-squared drop is precisely the scaling artifact
`cv_squared` was added to catch. **The Fano/CV-squared rise, and therefore the
contradiction with Reference 2, remains a property of proxy pretraining** -
18/18 cells on both metrics in the two pairs above. The attention arm is not
evidence for or against that review.

### What this does to the chapter's claim

Section 1 states the claim as "proxy pretraining **and** explicit attention
gating both suppress task-irrelevant processing." The 2x2 sharpens it into
something more specific and better supported:

- **Magnitude suppression is common to both** - it is the one metric that falls
  in all four contrasts, and it is the metric Reference 1 licenses. This is the
  chapter's strongest result.
- **Everything else is regimen-specific.** Higher dimensionality, higher
  sparsity and higher variability belong to proxy pretraining; lower
  dimensionality belongs to attention-alone; and where both are present, proxy
  pretraining sets the geometry.

So the two modifications are **redundant, not complementary**, and the chapter
should say so. That is a result about mechanism, not a failure: it explains why
stacking them yields no further accuracy, and it is what makes the attention
section load-bearing rather than a second route to the same number.

---

## 5. What to say, and what not to

**Defensible:** proxy pretraining produces a lower-magnitude, sparser, but
higher-dimensional and more variable population code, at matched accuracy, in
18/18 cells across two independent pairs. The magnitude half matches Reference 1;
the variability half contradicts Reference 2, most likely because the
manipulations differ. Attention additionally suppresses irrelevant identity to
near chance - but only in the `location` task context. Attention **alone** also
lowers magnitude and effective dimensionality (9/9 cells at a 0.88pp gap);
attention **on top of proxy pretraining** does not (PR 4/9 at a 0.00pp gap).

**Do not say:** that Level 3 shows 9/9 cells or a large gating effect (that was
epoch pooling; pinned it is 6/9 and small); that attention-only "barely gates"
(it gates strongly on location and category); that the PR result contradicts
Reference 2; that the magnitude result confirms Reference 2; that either
reference predicts sparsity; that attention suppresses identity in general
(it does not do so under `task=category`); that any p-value is below 0.002
(the bootstrap floor at `n_boot=1000`); that the pairs were matched at 10pp /
0.08pp or on the analysed split — they were matched on novel-angle, and the
analysed-split gaps are 8.9pp (baseline, irreducible) and 0.84pp (attention);
that the attention arm **agrees** with Reference 2 on variability (the Fano drop
in the scratch row is 7/9 but CV-squared is 5/9, so it is not established — see
§4, second reading); that attention and proxy pretraining are complementary
mechanisms (the 2x2 shows attention's geometric effect is absorbed when proxy
pretraining is present).

---

## 6. Reproducing this

```bash
# GPU server, from ~/Projects/WM-model
source ~/.venv/WM-model/bin/activate
./run_neural_efficiency.sh                 # all three levels
./run_neural_efficiency.sh level1          # one level only
```

Every flag that matters is pinned in that script, including the accuracy-matched
epoch pairs. Do not hand-assemble the CLI: the 2026-07-27 pass did, omitted
`--epoch_a/--epoch_b` on Level 3 and `--best_epoch/--split/--task` on Level 1,
and produced a headline result that reversed on re-run.

Before quoting any number from a new run, check in each JSON:

- `epoch_a` / `epoch_b` are **not** `null`, and `epochs_pooled` is `false`
- `split` is the one you intended, and the accuracy match was made on that same split
- `cv_squared` is present alongside `fano_factor_analogue`
- the swap-test block reports `decoded_property`, not a bare `property`
- any accuracy-gap figure you quote comes from `val_novel_identity` — the split
  analysed — not from the novel-angle figures the epoch pairs were selected on

Logs and provenance land in `logs/neural_efficiency_rerun_<timestamp>/`.

## 7. Where everything lives

| | |
|---|---|
| This document | Authoritative statement of the chapter |
| `docs/PAPER_EXPLAINED_*.md` | Full summaries of both references |
| `src/analysis/{neural_efficiency,gate_suppression,compare_models}.py` | The three tools |
| `run_neural_efficiency.sh` | The one reproducible entry point |
| `analysis_results/neural_efficiency/` | Result artifacts (tracked; small) |
| `slidev-presentation/slides.md` | The chapter's slides |
| `slidev-presentation/speaker_notes_neural_efficiency_onward.md` | Per-slide talking points |
| `docs/ANALYSIS_AUDIT_FINDINGS.md` | Chronological audit log — history, not current state |
| `docs/archive/` | Superseded planning documents |

---

## 8. References

**Read in full, load-bearing:**

1. Poppenk, J., Moscovitch, M., & McIntosh, A. R. (2016). fMRI evidence of
   equivalent neural suppression by repetition and prior knowledge.
   *Neuropsychologia*, 90, 159–169.
2. Constantinidis, C., & Klingberg, T. (2016). The neuroscience of working
   memory capacity and training. *Nature Reviews Neuroscience*, 17(7), 438–449.

**Cited elsewhere in the deck, cross-referenced but not re-verified here:**

3. Chung, Y. H., Brady, T. F., & Störmer, V. S. (2024). Meaningfulness and
   familiarity expand visual working memory capacity. *Current Directions in
   Psychological Science*, 33(5), 275–282.
4. Mercer, T. (2025). Familiarity influences on proactive interference in verbal
   memory. *Quarterly Journal of Experimental Psychology*.

Level 3 cites no external reference by design (§2).
