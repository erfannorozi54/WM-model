# Results: what we claim, and what backs each claim

**One row per claim.** Every number this project states in a slide, a chapter, or
a doc should be findable here, together with the experiment directory that
produced it and the artifact that holds it. If a number is not in this table, it
is not a result of this project.

This document exists because results were previously scattered across a main
deck, a separate proxy deck, two attention guides and five phase summaries, each
with its own numbers, several of them from superseded training runs. Those are
gone; this is the index that replaces them.

Last verified: **2026-09-02**.

---

## The spine

The project makes one argument in four steps:

1. **Replicate** the paper's five analyses on our reimplementation.
2. **Modify** the model — task-guided attention (architecture) and proxy
   pretraining (training regimen).
3. **Show the modified model captures a working-memory feature** — neural
   efficiency, graded against human findings.
4. **Tell the two modifications apart** — the 2×2 that measures each against the
   other rather than each against its own control (§4.5). This is what stops
   steps 2 and 3 from being two independent routes to the same number.
5. **Show it captures further features** — the paper's analyses re-run on the
   modified models. *(Partly done, and currently unusable; see §5.)*

Anything that is not a step in that argument was removed. §6 records what and why.

---

## 1. Replication — the paper's five analyses

Experiments: `wm_{stsf,stmf,mtmf}_*`, `wm_attention_*`, `wm_dual_attention_*` at
h=256, and the `wm_h128_*` counterparts. 18 directories.

| Claim | Status | Artifact |
|---|---|---|
| Behavioural performance reproduces (Fig. A1c) | Supported | `analysis_results/<exp>/analysis1_behavioral.json` |
| Task-relevant information is preferentially encoded (Fig. 2) | **Under review** — recorded as "not supported", but that verdict was computed from *raw* accuracies (see the note below) | `analysis2_encoding.json` |
| Cross-task generalization collapses off-diagonal (Fig. 2c) | Supported, 6/6 models | `analysis2_encoding.json` |
| Orthogonalization drops from perceptual to encoding (Fig. 3b) | Supported | `analysis3_orthogonalization.json` |
| Chronological memory subspaces (H1, Fig. 4b) | Supported | `analysis4_wm_dynamics.json` |
| Shared encoding across stimulus groups (H2, Fig. 4d) | Supported once the class-index bug was fixed | `analysis4_wm_dynamics.json` |
| Causal perturbation shifts behaviour (Fig. A7) | Supported, 12/18 cross the boundary | `analysis5_causal.json` |

**The task-relevance verdict needs recomputing (2026-09-02).** "Not supported"
rested on the observation that category is often the best-decoded property in
every task context. But category and location have **4 classes** (chance 25%)
while identity has **~70** (chance ~1.4%), and the comparison was made on raw
accuracy. A property with 4 classes will out-score one with 70 on that scale
almost regardless of how well either is encoded, so the evidence for the verdict
is confounded with class counts.

`_analyze_task_relevance` now reports `chance_level`, `normalized_accuracy`
= (acc − chance)/(1 − chance), and a per-task `relevance_margin`; the verdict must
be recomputed from `relevance_margin` on a fresh run. **Until then, treat this row
and the matching deck claims (Analysis 2 slide, Conclusions item 4) as
provisional.** Method: `docs/ANALYSIS_METHODOLOGY.md` §3.2.

**Sampling regimes differ and are not comparable across hidden sizes.** h=256 runs
give `n_test ≈ 52` against ~69 identity classes — fewer test samples than classes;
h=128 runs give ≈ 265 (`num_val` 400 vs 2000). The identity diagonal reads
31.8–38.0% in the first regime and 65.6–80.2% in the second. That gap is a
measurement artifact. Compare within a hidden size, never across.

**These artifacts are gitignored** (`analysis_results/<exp>/` is covered by the
repo-wide ignore). They live on `hamrah-gpu-internal`. Only the neural-efficiency
JSONs are tracked — see §4.

Regenerate: `./run_analysis.sh h256` / `./run_analysis.sh h128`.

---

## 2. Modification A — task-guided attention

Architecture change: a feature-channel gate between the CNN and the RNN, driven by
the task vector (`attention_mode: "task_only"`) or by task vector × features
(`"dual"`). See `docs/ATTENTION.md`.

### Where it works

| Scenario | Baseline (train / angle / identity) | + Attention | Δ identity |
|---|---|---|---:|
| STMF, h=256 | 88.50 / 82.73 / 79.93 | 99.61 / 93.31 / 91.79 | **+11.9** |
| MTMF, h=256 | 88.26 / 81.53 / 80.57 | 99.27 / 93.27 / 92.15 | **+11.6** |

### Where it fails — the scope boundary

| Scenario | Baseline identity | + Attention identity | Δ |
|---|---:|---:|---:|
| STSF, h=256 | 99.76% | 88.22% | **−11.5** |
| STSF, h=128 | 99.96% | 67.79% | **−32.2** |

With one task and one feature there is no ambiguity for the gate to resolve, so
gating contributes only optimization difficulty. This replicates in two
independent hidden sizes, which is why it is stated as a boundary of the claim
rather than an anomaly.

h=128 figures verified from
`experiments/wm_h128_stsf_20260602_230425/training_log.json` and
`experiments/wm_h128_attention_stsf_20260603_053139/training_log.json`
(best epoch by `val_novel_identity_acc`).

### Dual vs. task-only

No consistent advantage.

| | Task-only | Dual |
|---|---|---|
| MTMF h=256 | 93.27 angle / 92.15 identity | 93.79 / 90.91 |
| MTMF h=128 | 92.9 / 91.1 | 89.3 / 88.2 |

Dual wins one of four comparisons. Earlier documents claimed it was better for
multi-task scenarios and best for novel identity; the current data does not
support either. It is retained as a co-equal arm of the attention study and
reported with this record intact.

Slide: *All Models Comparison*. Regenerate: `./run_training.sh`.

---

## 3. Modification B — proxy pretraining

Training-regimen change, architecture untouched: pre-train on dense feature recall
(predict the feature value N steps back), then fine-tune on the 3-class N-back
task.

| Metric | Baseline | Proxy pre-trained | Δ |
|---|---:|---:|---:|
| Best val, novel angle | 82.69% | 97.52% | **+14.83** |
| Final val, novel angle | 81.5% | 97.5% | +16.0 |
| Final val, novel identity | 80.6% | 92.8% | **+12.2** |
| Final train accuracy | 88.3% | 100.0% | +11.7 |

Convergence: the proxy model passes the baseline's *final* accuracy at epoch 1.

**This is a performance result, not a capacity result.** N-back level is
unchanged. Whether the model can hold more items or succeed at higher N was never
tested and is not claimed.

Slides: *Two-Stage Training*, *Results: Proxy vs. Baseline*, *Alignment With
Human Working Memory*. Reproduce: `./run_proxy_pipeline.sh baseline`.

---

## 4. WM feature 1 — neural efficiency

Full statement, licensing of each reference, and per-level confidence:
**`docs/NEURAL_EFFICIENCY.md`**. Artifacts are **tracked** in
`analysis_results/neural_efficiency/2026-08-22_audit-fixed/`.

| Level | Result | Verdict |
|---|---|---|
| 1 · Representational content | Irrelevant identity collapses to ~chance under `task=location`; no suppression under `task=category` | Corroborates conditionally |
| 2 · Population activity | Magnitude ↓, PR ↑, sparsity ↑, Fano ↑ and CV² ↑ — 18/18 cells, two independent pairs, at matched accuracy | Corroborates |
| 3 · Explicit gating | Suppression index sharper under proxy in 6/9 cells, small gaps; gate-relevance correlation improves consistently | Corroborates partially |

Graded against Poppenk et al. (2016) for magnitude — confirmed 18/18 — and
Constantinidis & Klingberg (2016) for Fano, which we **contradict**. The
divergence is reported, not smoothed: the manipulations differ (weeks of training
on the same task vs. knowledge transferred from a different task).

Reproduce: `./run_neural_efficiency.sh`.

---

## 4.5 The two modifications compared — the 2×2

Sections 2 and 3 each measure one modification against the plain baseline, and
§4 measures proxy pretraining twice — once inside each architecture. None of
those contrasts can tell the two modifications apart. The same four MTMF models
form a 2×2, and reading it that way is what makes §2 and §3 load-bearing rather
than two routes to the same accuracy.

### Accuracy — they are redundant

Best `val_novel_identity`, h=256 MTMF. *(Figures read from `training_log.json`
on `hamrah-gpu-internal` on 2026-08-26; the epoch anchors below are the ones
pinned in `run_neural_efficiency.sh`.)*

| | from scratch | + proxy pretraining | Δ from proxy |
|---|---:|---:|---:|
| **baseline** | 82.53% | 93.55% | +11.02 |
| **+ attention** | 91.91% | 93.63% | +1.72 |
| **Δ from attention** | +9.38 | +0.08 | |

**Interaction: −9.30pp.** Each modification alone buys +9 to +11pp; together
they buy +11.10pp, not +20. Whichever arrives first captures nearly all of the
available gain. On novel angle the interaction is −12.31pp, and attention on top
of proxy pretraining costs about half a point.

### Population code — attention's effect is absorbed

Artifact: `analysis_results/neural_efficiency/2026-08-26_2x2/`. Epochs pinned
17/9 (0.88pp gap) and 20/45 (**0.00pp** — the tightest accuracy match anywhere
in this project).

| Metric | attention alone | attention on top of proxy |
|---|---|---|
| Activation magnitude | lower **9/9** (CI excl. 0 in 8/9) | lower 7/9 (6/9) |
| Participation ratio | lower **9/9** (8/9); 6/6 excluding near-rank-1 location cells | **4/9 — no effect** |
| Population sparsity | mixed, 3/9 | *higher* 8/9 — opposite direction |

The accuracy interaction has a population-code counterpart: attention reshapes
the code when it is the only modification, and does not when proxy pretraining
already has. **The two modifications are redundant, not complementary.**

**No Fano direction is claimed here.** Fano falls 7/9 under attention-alone but
`cv_squared` is 5/9, and Fano is scale-dependent while attention lowers
magnitude — the exact artifact `cv_squared` exists to catch. The Fano/CV² rise,
and the contradiction with Constantinidis & Klingberg, remains attributed to
proxy pretraining (§4).

Full statement: `docs/NEURAL_EFFICIENCY.md` §4, "Level 2, second reading".
Reproduce: `./run_neural_efficiency.sh level2x`.

**What is still open here:** Level 1 (`compare_models.py`) has not been run
across the square — only baseline vs. attention. And Level 1 currently reads
`wm_attention_mtmf_20260520_203605` while §4's Levels 2/3 and this 2×2 read
`wm_attention_mtmf_20260726_161735`; one attention run should serve throughout.

---

## 5. WM feature 2 — in progress

**Status corrected 2026-09-02.** This section previously said the five analyses
had never been run on the proxy-pretrained models. They were, on 2026-08-26:
`analysis_results/` now holds `finetune_proxy_wm_mtmf_20260705_164908`,
`finetune_proxy_wm_attention_mtmf_20260726_201707`, and an `_ep*` directory for
each.

**The existing pinned runs are not usable, but the cause is fixed.** `--epochs`
was declared in `comprehensive_analysis.py` and never passed to anything — every
analysis resolved its own `_find_best_epoch()`. So each `*_ep*` directory on disk
is a byte-identical duplicate of its unpinned twin, not an accuracy-matched run.
Verified with `cmp` across analyses 1, 2, 3 and 5; analysis 4 differs only in
field names added by `820a8f8`. `wm_mtmf_20260520_140601_ep12` was produced from
that model's *best* epoch, 17, not 12.

Fixed 2026-09-02: `--epochs` now reaches every analysis (including the H2 swap
test and analysis 5, which each re-resolved their own epoch), `--split` was
added, and every analysis JSON carries a `provenance` block recording
`epochs_used` / `epoch_source` / `epochs_pooled` / `split` / `n_trials`. **The
directories on disk predate the fix and must be regenerated** before anything in
them is quoted as pinned.

Until that is fixed, this step is **not** matched-accuracy evidence: the plain
baseline sits at 82.53% while the other three cells sit at 91.9-93.6%, so every
baseline→X decoding difference is confounded with "X is simply the better
model." The three contrasts among the other cells are within 1.8pp and survive.

Closing the gap requires no new task and no cross-model benchmarking — it puts
our own modified model through the battery our own baseline already went
through, once the flag works.

| Condition A | Condition B | Matched epochs |
|---|---|---|
| `wm_mtmf_20260520_140601` | `finetune_proxy_wm_mtmf_20260705_164908` | 12 / 1 — selected on novel-angle; **8.9pp on the analysed split, and irreducible** (proxy ep1 exceeds the baseline's identity ceiling) |
| `wm_attention_mtmf_20260726_161735` | `finetune_proxy_wm_attention_mtmf_20260726_201707` | 43 / 1 — selected on novel-angle; **0.84pp on the analysed split** (a strict match, 18 / 8 at 0.44pp, would require re-running the chapter) |

The intent is that each pair runs twice: with `--epochs` pinned to the
accuracy-matched pair (primary, carrying the matched-accuracy discipline from
§4), and at the auto-selected best epoch (robustness check). **The first of
those two does not currently happen** — see the status note above.

**Known difference to record with the output:** the epoch pairs were matched on
`val_novel_identity`, but `comprehensive_analysis.py` has no `--split` flag and
pools both validation splits. The efficiency tools in §4 do not pool. This is
documented rather than silent.

Run — all four MTMF models, identical settings, with a comparability audit:

```bash
./run_2x2.sh matched      # accuracy-matched epoch pins
./run_2x2.sh ceiling      # each model at its own best epoch
./run_2x2.sh both
```

Settings live in `configs/analysis/2x2.yaml`; results and the audit land in
`analysis_results/2x2/<design>/comparison.md`. Single experiments still go
through `run_analysis.sh` (now with `SPLIT=`), but that script analyses each
experiment on its own terms and does not check comparability.

---

## 6. Removed lines of work

| Line | Why it was removed |
|---|---|
| **Meta-learning** (few-shot adaptation to three-in-a-row) | The hypothesis was refuted — attention gave no adaptation advantage, and every architecture converged to 65–69%. Decisively, no result artifact survived: the reported numbers traced only to two PNGs and prose, so they could not be verified or regenerated. |
| **Continual learning / catastrophic forgetting** | Planned in the former `THESIS_CONTRIBUTIONS.md` as a second contribution. No code was ever written. |
| **Desimone & Duncan (1995); Treue & Martínez-Trujillo (1999)** | Cited for Level 3 but read only as summaries, not in full — below the standard applied to the two references that remain. Level 3 needs no external reference: the gates are a literal built-in suppression signal. |

Nothing above is deleted from history; `git log` retains all of it.

---

## Where the evidence lives

| | |
|---|---|
| This document | Index of every claim → artifact |
| `docs/NEURAL_EFFICIENCY.md` | The efficiency chapter in full |
| `docs/ATTENTION.md` | Attention architecture (no results, by design) |
| `docs/ANALYSIS_METHODOLOGY.md` | How the five analyses are computed |
| `docs/PAPER_EXPLAINED_*.md` | The two human-WM references, read in full |
| `docs/ANALYSIS_AUDIT_FINDINGS.md` | Chronological audit log — history, not current state |
| `analysis_results/neural_efficiency/` | Tracked artifacts (§4) |
| `slidev-presentation/slides.md` | The deck |
| `AGENTS.md` | Commands, config gotchas, pipeline pitfalls |
