# Results: what we claim, and what backs each claim

**One row per claim.** Every number this project states in a slide, a chapter, or
a doc should be findable here, together with the experiment directory that
produced it and the artifact that holds it. If a number is not in this table, it
is not a result of this project.

This document exists because results were previously scattered across a main
deck, a separate proxy deck, two attention guides and five phase summaries, each
with its own numbers, several of them from superseded training runs. Those are
gone; this is the index that replaces them.

Last verified: **2026-08-26**.

---

## The spine

The project makes one argument in four steps:

1. **Replicate** the paper's five analyses on our reimplementation.
2. **Modify** the model — task-guided attention (architecture) and proxy
   pretraining (training regimen).
3. **Show the modified model captures a working-memory feature** — neural
   efficiency, graded against human findings.
4. **Show it captures further features** — the paper's analyses re-run on the
   modified models. *(In progress; see §5.)*

Anything that is not a step in that argument was removed. §6 records what and why.

---

## 1. Replication — the paper's five analyses

Experiments: `wm_{stsf,stmf,mtmf}_*`, `wm_attention_*`, `wm_dual_attention_*` at
h=256, and the `wm_h128_*` counterparts. 18 directories.

| Claim | Status | Artifact |
|---|---|---|
| Behavioural performance reproduces (Fig. A1c) | Supported | `analysis_results/<exp>/analysis1_behavioral.json` |
| Task-relevant information is preferentially encoded (Fig. 2) | **Not supported** — the task-relevant cell is often not the best-decoded one; category decodes well in every task context | `analysis2_encoding.json` |
| Cross-task generalization collapses off-diagonal (Fig. 2c) | Supported, 6/6 models | `analysis2_encoding.json` |
| Orthogonalization drops from perceptual to encoding (Fig. 3b) | Supported | `analysis3_orthogonalization.json` |
| Chronological memory subspaces (H1, Fig. 4b) | Supported | `analysis4_wm_dynamics.json` |
| Shared encoding across stimulus groups (H2, Fig. 4d) | Supported once the class-index bug was fixed | `analysis4_wm_dynamics.json` |
| Causal perturbation shifts behaviour (Fig. A7) | Supported, 12/18 cross the boundary | `analysis5_causal.json` |

**Sampling regimes differ and are not comparable across hidden sizes.** h=256 runs
give `n_test ≈ 52` against ~69 identity classes — fewer test samples than classes;
h=128 runs give ≈ 265 (`num_val` 400 vs 2000). The identity diagonal reads
31.8–38.0% in the first regime and 65.6–80.2% in the second. That gap is a
measurement artifact. Compare within a hidden size, never across.

**These artifacts are gitignored** (`analysis_results/<exp>/` is covered by the
repo-wide ignore). They live on `hamrah-gpu-internal`. Only the neural-efficiency
JSONs are tracked — see §3.

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

## 5. WM feature 2 — in progress

The five paper analyses have **never been run on the proxy-pretrained models**.
All 18 directories in `analysis_results/` are `wm_*`; none is `finetune_proxy_*`.
Closing that gap requires no new task and no cross-model benchmarking — it puts
our own modified model through the battery our own baseline already went through.

| Condition A | Condition B | Matched epochs |
|---|---|---|
| `wm_mtmf_20260520_140601` | `finetune_proxy_wm_mtmf_20260705_164908` | 12 / 1 (10pp gap) |
| `wm_attention_mtmf_20260726_161735` | `finetune_proxy_wm_attention_mtmf_20260726_201707` | 43 / 1 (0.08pp gap) |

Each pair runs twice: with `--epochs` pinned to the accuracy-matched pair
(primary, carrying the matched-accuracy discipline from §4), and at the
auto-selected best epoch (robustness check).

**Known difference to record with the output:** the epoch pairs were matched on
`val_novel_identity`, but `comprehensive_analysis.py` has no `--split` flag and
pools both validation splits. The efficiency tools in §4 do not pool. This is
documented rather than silent.

Run: `EPOCHS=1 ./run_analysis.sh finetune_proxy_wm_mtmf_20260705_164908`

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
