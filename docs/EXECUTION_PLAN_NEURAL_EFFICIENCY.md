# Execution Plan: Attention + Proxy Pretraining as a Neural-Efficiency Finding

**Purpose:** a run-order checklist for producing the new-finding chapter of the thesis. Rationale, literature, and metric definitions live in `docs/FUTURE_WORK_NEURAL_EFFICIENCY.md` — this document only says *what to run, in what order, what to compare, and what to prioritize*.

**Core claim to establish:** familiarity/structure (from proxy pretraining) and explicit gating (from attention) both suppress task-irrelevant processing, observable at three independent levels — representational content, population activity, and explicit gate values. Three converging signatures make a stronger chapter than any one alone.

---

## 0. What already exists — corrected after directly checking disk contents

**Correction to an earlier version of this guide:** I had assumed the local h128 experiment directories included saved hidden states, based on `save_hidden: true` in their configs. Direct inspection (`find experiments/<dir> -name "*.pt"`, checking for a `hidden_states/` subdirectory) shows this is wrong for **every** local experiment directory: none contain `best_model.pt` or a `hidden_states/` folder, only lightweight artifacts (`config.yaml`, `training_log.json`, `training.log`, small PNG visualizations). The training logs even reference `best_model.pt` as an output path that isn't actually present locally. The most likely explanation is that these runs happened on `hamrah-gpu-internal` and only the small files were synced down, leaving the large checkpoint + hidden-state tensors on the server. Practical consequence: **none of the analyses below are runnable on this machine right now without first pulling data from the GPU server, or rerunning training here.**

| Data | Location | Status |
|---|---|---|
| Baseline STMF, h128 (no attention) | `experiments/wm_h128_stmf_20260603_010546` | Config/logs only locally — **no `best_model.pt`, no `hidden_states/`** |
| Attention STMF, h128 | `experiments/wm_h128_attention_stmf_20260603_073349` | Config/logs only locally — **no `best_model.pt`, no `hidden_states/`** |
| Attention MTMF, h128 | `experiments/wm_h128_attention_mtmf_20260603_094851` | Config/logs only locally — **no `best_model.pt`, no `hidden_states/`** |
| Baseline MTMF, h256 (no attention) | referenced as `experiments/wm_mtmf_20260520_140601` | Not present locally at all — check `hamrah-gpu-internal` |
| Proxy-pretrained baseline MTMF, h256 | referenced as `experiments/finetune_proxy_wm_mtmf_20260705_164908` | Not present locally at all — check `hamrah-gpu-internal` |
| `configs/proxy/proxy_attention_mtmf.yaml`, `proxy_dual_attention_mtmf.yaml` | `configs/proxy/` | Configs exist, never run (no matching experiment dir anywhere found yet) |
| Attention model, no proxy, h256 control | `configs/attention_mtmf.yaml` | Config exists, never run |
| Gate-value saving | `src/models/attention.py`, `train_with_generalization.py`, `finetune_from_proxy.py` | Implemented — only takes effect on runs started from now on |
| `src/analysis/neural_efficiency.py` (magnitude/PR/sparsity/Fano metrics) | `src/analysis/neural_efficiency.py` | **Implemented and smoke-tested** on synthetic payloads matching the real schema (see verification note below) — never yet run against real experiment data, since none is currently reachable locally |
| Gate-suppression index (`src/analysis/gate_suppression.py`) | `src/analysis/gate_suppression.py` | **Implemented and smoke-tested** with a planted effect (see below) — never yet run against real experiment data |

**Verification done on `neural_efficiency.py`:** since no real hidden-state data is reachable on this machine, I validated it against synthetic payloads built to match the exact saved-payload schema, with a known injected effect (two conditions differing only by a fixed activation-scale factor). Results matched the closed-form expectation exactly where one exists — e.g. the Fano-factor analogue scaled by the same factor as the injected scale change (0.442 → 0.265 for a 0.6× scale change, i.e. 0.442 × 0.6 ≈ 0.265), and participation ratio stayed invariant under pure rescaling, as it should mathematically. This confirms the code is logically correct; it has not yet been run on real model data.

**Verification done on `gate_suppression.py`:** planted a known relevant/irrelevant gate gap in synthetic `cnn_activations` + `gates` (0.9 vs. 0.5 for one condition, 0.9 vs. 0.1 for a "sharper" second condition) and recovered suppression indices of −0.40 and −0.80 respectively — essentially exact recovery of the planted −0.4/−0.8 gaps — with `index_sharper_in_b=True` and an `index_gap` of ≈0.40 matching the designed difference. Also not yet run on real model data.

**Action before anything else:** check `hamrah-gpu-internal` for the h128 checkpoints/hidden-states and the two h256 baseline/proxy runs. If they exist there, you don't need to retrain anything — rsync `hidden_states/` (and `best_model.pt` if you want matched-epoch selection) down, or run the analysis commands below directly over SSH on the server.

---

## 1. First, once hidden states are reachable: the STMF result to write up

**Update: this is not actually free right now** (see the correction in §0) — the accuracy numbers for this pair are already known from `training_log.json` (baseline 79.3–80.8% vs. attention 91.9% novel-identity accuracy, already cited in this conversation), but the *representational* comparison below needs the saved `hidden_states/`, which do not exist locally for either run. Once you've pulled them from `hamrah-gpu-internal` (or re-run both configs locally — same hidden size, same 45 epochs, ~15-20 min each on GPU per the Phase 5 benchmarks), this is the first analysis to run, since it needs no new training:

```bash
# Behavioral comparison (Analysis 1) — already-trained checkpoints
python -m src.analysis.comprehensive_analysis --analysis 1 \
  --hidden_root experiments/wm_h128_stmf_20260603_010546/hidden_states \
  --output_dir analysis_results/wm_h128_stmf_baseline

python -m src.analysis.comprehensive_analysis --analysis 1 \
  --hidden_root experiments/wm_h128_attention_stmf_20260603_073349/hidden_states \
  --output_dir analysis_results/wm_h128_attention_stmf

# Representational comparison — task-irrelevant decodability, orthogonalization (Phase 5 tool)
python -m src.analysis.compare_models \
  --baseline experiments/wm_h128_stmf_20260603_010546/hidden_states \
  --attention experiments/wm_h128_attention_stmf_20260603_073349/hidden_states \
  --property identity \
  --output_dir analysis_results/compare_stmf_h128
```

**What to look for:** `compare_models.py` was built specifically to test whether attention lowers task-irrelevant-feature decodability and raises the orthogonalization index relative to baseline. This gives you a real number for the "representational content" row of the three-level table without needing any *new* training — just the existing checkpoints' data pulled or reproduced. Run this before the new §2 training — it validates the pipeline still runs correctly before you spend GPU hours.

---

## 2. New training runs, in priority order

All of these use infrastructure that already exists; none require new code.

### 2a. Attention + proxy pretraining (highest priority — this is the new contribution)

```bash
# Pretrain attention model on proxy task
python -m src.train_proxy --config configs/proxy/proxy_attention_mtmf.yaml

# Fine-tune onto the real N-back task (transfers perceptual + cognitive + attention weights)
python -m src.finetune_from_proxy \
  --proxy_exp_dir experiments/proxy_attention_mtmf_<timestamp> \
  --config configs/attention_mtmf.yaml
```

This is the run that didn't exist anywhere before this conversation. It now also saves gate values automatically (the fix applied to `finetune_from_proxy.py`).

### 2b. Attention, no proxy (the control 2a needs)

```bash
python -m src.train_with_generalization --config configs/attention_mtmf.yaml
```

Without this, you cannot tell whether any effect in 2a comes from attention or from proxy pretraining — it isolates the "architecture" variable from the "familiarity" variable.

### 2c. (If time allows) Dual-attention variant of both 2a/2b

```bash
python -m src.train_proxy --config configs/proxy/proxy_dual_attention_mtmf.yaml
python -m src.finetune_from_proxy --proxy_exp_dir experiments/proxy_dual_attention_mtmf_<timestamp> --config configs/dual_attention_mtmf.yaml
python -m src.train_with_generalization --config configs/dual_attention_mtmf.yaml
```

Lower priority than 2a/2b: task-only is simpler to interpret and already has the STMF result above as a sanity check; dual is a nice-to-have that strengthens the story but isn't required for the core claim.

### 2d. Baseline MTMF + proxy MTMF at h256 (only if not found on the GPU server)

```bash
python -m src.train_with_generalization --config configs/mtmf.yaml
python -m src.train_proxy --config configs/proxy/proxy_mtmf.yaml
python -m src.finetune_from_proxy --proxy_exp_dir experiments/proxy_mtmf_<timestamp> --config configs/mtmf.yaml
```

Skip this entirely if `hamrah-gpu-internal` already has `wm_mtmf_20260520_140601` and `finetune_proxy_wm_mtmf_20260705_164908` — re-training wastes GPU time for data you already have.

---

## 3. Analysis to run once 2a/2b (and 2d, if needed) finish

For **every** resulting experiment directory, run the existing full pipeline first — it's zero marginal effort and gives you Figures 1–5 of the base paper replication for free:

```bash
python -m src.analysis.comprehensive_analysis --analysis all \
  --model experiments/<exp>/best_model.pt \
  --hidden_root experiments/<exp>/hidden_states \
  --property identity --output_dir analysis_results/<exp>
```

Then the comparisons specific to this chapter:

| Comparison | Tool | What it tests |
|---|---|---|
| Attention vs. baseline decodability/orthogonalization | `src.analysis.compare_models` (as in §1) | Representational-content suppression |
| Baseline vs. proxy-pretrained baseline hidden-state magnitude/PR/sparsity/Fano | `src.analysis.neural_efficiency` (implemented, smoke-tested on synthetic data — not yet run on real experiments) | Population-activity suppression (§4.2–4.3 of the future-work doc) |
| Attention-only vs. attention+proxy gate-suppression index | `src.analysis.gate_suppression` (implemented, smoke-tested — not yet run on real experiments) | Explicit-gating suppression (§4.8.1 of the future-work doc) |
| Accuracy: baseline / attention / proxy-baseline / attention+proxy | `training_log.json` per experiment, `best_val_novel_identity_acc` | The matched-accuracy control everything else depends on (§4.2) |

Always compute the accuracy row first for whichever pair you're comparing — if accuracies aren't close, any activity/decodability/gate difference is confounded by the model just being better, not more "efficient" (this is the Constantinidis & Klingberg Box 2 caveat already in the future-work doc).

---

## 4. What to focus on, given limited time

If you only do three things before writing the report, do these, in this order:

1. **§1 (STMF compare_models.py run)** — zero cost, already-trained data, gives you a real representational-suppression number today.
2. **§2a + §2b (attention+proxy vs. attention-alone)** — this is the actual new contribution; everything else is supporting evidence for it. Run this before writing any new analysis code.
3. **The gate-suppression index (§3, row 3), now implemented** — it's the metric most clearly *not* obtainable from the baseline model, which is the strongest answer to "what can your model show that a plain RNN can't." Once you have attention-only and attention+proxy hidden states (from §2a/§2b), run:

```bash
python -m src.analysis.gate_suppression \
  --root_a experiments/<attention_only_exp>/hidden_states \
  --root_b experiments/<attention_proxy_exp>/hidden_states \
  --label_a attention_only --label_b attention_proxy \
  --split val_novel_identity \
  --output_dir analysis_results/gate_suppression_mtmf
```

Note this compares each condition against its *own* channel-relevance ranking (see the module docstring) — it does not assume channel `c` means the same thing in both conditions, since proxy pretraining updates the trainable projection ahead of the frozen ResNet50 backbone. Look at `index_sharper_in_b` and `index_gap` in the JSON output for the headline number.

`src/analysis/neural_efficiency.py` (RNN hidden-state magnitude/PR/sparsity/Fano) is now implemented and smoke-tested against synthetic data, but still treat running it as secondary to 1–3 above — it depends on the matched-accuracy baseline/proxy hidden states, which are the hardest data to get right (they need genuinely close accuracies, not just any two checkpoints), and it's the metric furthest from something the baseline model structurally cannot do (unlike the gate-suppression index). Once baseline + proxy hidden states are available (locally or on the GPU server) and you've picked a matched-accuracy epoch pair, run it as:

```bash
python -m src.analysis.neural_efficiency \
  --root_a experiments/<baseline_exp>/hidden_states \
  --root_b experiments/<proxy_finetuned_exp>/hidden_states \
  --label_a baseline --label_b proxy \
  --training_log_a experiments/<baseline_exp>/training_log.json \
  --training_log_b experiments/<proxy_finetuned_exp>/training_log.json \
  --match_metric val_novel_angle_acc \
  --split val_novel_identity \
  --output_dir analysis_results/neural_efficiency_mtmf
```

`--training_log_a`/`--training_log_b` auto-select the closest-accuracy epoch pair instead of requiring you to inspect `training_log.json` by hand and pass `--epoch_a`/`--epoch_b` yourself — it prints the resulting accuracy gap so you can judge whether the match is close enough to trust.

The scrambled-feature-label causal control (§5 of the future-work doc) is a stretch goal for a "future work" paragraph in the report, not something to execute before submission unless everything above is finished early.

---

## 5. Report-writing note

Structure the new chapter around the three-level table (representational content / population activity / explicit gating), not around "attention" and "proxy" as separate sub-sections — the point of this plan is that they converge on one claim (familiarity and structure suppress irrelevant processing), observed three different ways. State plainly which of the three you completed and which remain proposed, the same verified-vs-not-verified discipline already used for citations in `docs/FUTURE_WORK_NEURAL_EFFICIENCY.md`.
