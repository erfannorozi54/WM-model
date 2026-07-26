# Execution Plan: Attention + Proxy Pretraining as a Neural-Efficiency Finding

**Purpose:** a run-order checklist for producing the new-finding chapter of the thesis. Rationale, literature, and metric definitions live in `docs/FUTURE_WORK_NEURAL_EFFICIENCY.md` — this document only says *what to run, in what order, what to compare, and what to prioritize*.

**Core claim to establish:** familiarity/structure (from proxy pretraining) and explicit gating (from attention) both suppress task-irrelevant processing, observable at three independent levels — representational content, population activity, and explicit gate values. Three converging signatures make a stronger chapter than any one alone.

---

## 0. What already exists — confirmed by direct audit of `hamrah-gpu-internal` on 2026-07-26

**This section supersedes the earlier "check the GPU server" placeholder.** The server was reachable, the new commit (`b4d5ff7`, containing the gate-logging fix + `neural_efficiency.py`/`gate_suppression.py`) was pulled there via fast-forward, and every `experiments/` directory was enumerated directly (`best_model.pt` presence, `hidden_states/` file counts, `config.yaml` key fields). Full inventory:

| Experiment dir | Config | Status |
|---|---|---|
| `wm_stsf_20260520_095056`, `wm_stmf_20260520_115231`, `wm_mtmf_20260520_140601` | baseline, h256 | best_model.pt + 1170 hidden_state files each |
| `wm_attention_stsf_20260520_161910`, `wm_attention_stmf_20260520_182203`, `wm_attention_mtmf_20260520_203605` | attention (task_only), h256 | best_model.pt + 1170 hidden_state files each |
| `wm_dual_attention_stsf_20260520_225000`, `wm_dual_attention_stmf_20260521_005245`, `wm_dual_attention_mtmf_20260521_030657` | attention (dual), h256 | best_model.pt + 1170 hidden_state files each |
| `proxy_mtmf_20260704_155609` | proxy pretrain, baseline, h256, mtmf | best_model.pt + 1170 hidden_state files |
| `finetune_proxy_wm_mtmf_20260705_164908` | finetuned from above, h256, mtmf | best_model.pt + 1170 hidden_state files |
| `wm_h128_{stsf,stmf,mtmf}`, `wm_h128_attention_{...}`, `wm_h128_dual_attention_{...}` (9 dirs) | h128 variants, 2026-06-23 | best_model.pt + 5670 hidden_state files each |

**Critical finding — none of the existing attention checkpoints have `gates` in their payloads.** Every attention/dual-attention run above was trained in May/June 2026, before today's gate-logging fix existed in the codebase. `gate_suppression.py` cannot run on any of them as-is. There is also **no attention+proxy experiment anywhere** — `configs/proxy/proxy_attention_mtmf.yaml` and `proxy_dual_attention_mtmf.yaml` exist but have never been run (confirmed: no matching experiment directory). This is exactly the gap your original question ("can I use attention models in proxy pretraining") identified — it genuinely requires new training, not just new analysis code.

**Large accuracy gap in the one existing baseline/proxy pair** (`wm_mtmf_20260520_140601` vs. `finetune_proxy_wm_mtmf_20260705_164908`, from `training_log.json` best epochs): novel-identity accuracy 82.5% → 93.5%, novel-angle accuracy 81.2% → 97.1%. This is a large improvement, not a small one — any population-activity difference `neural_efficiency.py` finds between these two must be reported alongside this gap, not instead of it. This is the Constantinidis & Klingberg Box 2 caveat in practice, not just in theory.

**Server also had ~175MB of stale top-level `.log` files (deleted, never git-tracked) and a stale pre-refactor duplicate code tree at the repo root** (`analysis/`, `meta/`, `models/`, `scripts/`, `utils/`, `train.py`, etc., living alongside `src/` since before the `src/` migration, confirmed via `diff -rq` to be outdated copies missing `gate_suppression.py`/`proxy_model.py`/etc.) — left untouched pending confirmation, since it's outside `src/` and not used by any documented command.

**Verification done on `neural_efficiency.py` and `gate_suppression.py`:** both were smoke-tested against synthetic payloads with known injected effects before ever touching real data (see git history) — the Fano-factor analogue recovered a planted 0.6× scale change almost exactly (0.442 → 0.265), and the gate-suppression index recovered planted gaps of −0.40/−0.80 almost exactly. Both are now also running on real data — see §6 below for live status.

**Action items, in order (superseding the old "check hamrah-gpu-internal" instruction — already done):**
1. ~~Check GPU server~~ Done — see table above.
2. Run the two analyses that need zero new training (§1, revised below) — **in progress, see §6**.
3. Train the missing attention+proxy pair (§2a/§2b) — **in progress, see §6**.
4. Once §2a/§2b finish, run `gate_suppression.py` and `neural_efficiency.py` on the new pair (§3/§4).

---

## 1. Running now, zero new training needed (real experiment data, h256 mtmf)

Both use existing checkpoints already on `hamrah-gpu-internal` — no training required:

```bash
# Population-activity layer: baseline vs. proxy-pretrained-then-finetuned baseline
python -m src.analysis.neural_efficiency \
  --root_a experiments/wm_mtmf_20260520_140601/hidden_states \
  --root_b experiments/finetune_proxy_wm_mtmf_20260705_164908/hidden_states \
  --label_a baseline --label_b baseline_proxy_finetuned \
  --training_log_a experiments/wm_mtmf_20260520_140601/training_log.json \
  --training_log_b experiments/finetune_proxy_wm_mtmf_20260705_164908/training_log.json \
  --match_metric val_novel_angle_acc --split val_novel_identity \
  --output_dir analysis_results/neural_efficiency_baseline_vs_proxy_mtmf

# Representational-content layer: baseline vs. attention (task_only), same mtmf config family
python -m src.analysis.compare_models \
  --baseline experiments/wm_mtmf_20260520_140601/hidden_states \
  --attention experiments/wm_attention_mtmf_20260520_203605/hidden_states \
  --property identity \
  --output_dir analysis_results/compare_baseline_vs_attention_mtmf
```

**Why mtmf, not stmf:** the only real baseline/proxy pair on the server is mtmf-only (the "full paper config," N=1,2,3, all task features), so anchoring every comparison to mtmf keeps all four conditions (baseline / attention / baseline+proxy / attention+proxy) on the same config for a clean 2×2 table. An stmf or stsf version of any of these can be added later only if new stsf/stmf attention+proxy runs are also done — not planned by default since GPU time is better spent completing the mtmf 2×2 first.

**What to look for:** `compare_models.py` tests whether attention lowers task-irrelevant-feature decodability / raises the orthogonalization index relative to baseline — the "representational content" row. `neural_efficiency.py` tests whether proxy pretraining reduces hidden-state magnitude/participation-ratio/sparsity/Fano-factor relative to the no-proxy baseline — the "population activity" row, but see the accuracy-gap caveat in §0 before interpreting the result as pure "efficiency."

---

## 2. New training runs, in priority order

All of these use infrastructure that already exists; none require new code. The server has two idle RTX 3090s, so 2a and 2b run **in parallel** on separate GPUs (`CUDA_VISIBLE_DEVICES=0`/`1`) to cut wall-clock roughly in half. Historical timings from the existing mtmf runs on this same server: attention-only mtmf training ≈1h52m, proxy pretraining ≈3h10m, finetuning ≈2h9m — so the critical path is proxy-pretrain (3a) → finetune (3c) ≈ 5h20m, with 2b (attention-only retrain) finishing well before that and just waiting.

### 2a. Attention + proxy pretraining (highest priority — this is the new contribution)

```bash
CUDA_VISIBLE_DEVICES=0 nohup python -m src.train_proxy --config configs/proxy/proxy_attention_mtmf.yaml \
  > proxy_attention_mtmf.log 2>&1 &

# once it finishes, fine-tune onto the real N-back task
# (transfers perceptual + cognitive + attention weights; classifier reinitialized)
CUDA_VISIBLE_DEVICES=0 nohup python -m src.finetune_from_proxy \
  --proxy_exp_dir experiments/proxy_attention_mtmf_<timestamp> \
  --config configs/attention_mtmf.yaml \
  > finetune_attention_mtmf.log 2>&1 &
```

This is the run that didn't exist anywhere before this conversation. It now also saves gate values automatically (the fix applied to `finetune_from_proxy.py`) — this run is what makes `gate_suppression.py`'s "attention+proxy" side possible at all.

### 2b. Attention, no proxy (the control 2a needs — also backfills gates for the attention-only side)

```bash
CUDA_VISIBLE_DEVICES=1 nohup python -m src.train_with_generalization --config configs/attention_mtmf.yaml \
  > attention_mtmf_regate.log 2>&1 &
```

Without this, you cannot tell whether any effect in 2a comes from attention or from proxy pretraining — it isolates the "architecture" variable from the "familiarity" variable. Note this is a **re-run** of a config that was already trained once (`wm_attention_mtmf_20260520_203605`) — that old run predates the gate-logging fix and has no `gates` in its payloads, so a fresh run is needed anyway to get an attention-only condition with real gate data comparable to 2a's output. This produces a new `wm_attention_mtmf_<new-timestamp>` directory; use it (not the old one) for gate_suppression comparisons.

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
  --root_a experiments/wm_mtmf_20260520_140601/hidden_states \
  --root_b experiments/finetune_proxy_wm_mtmf_20260705_164908/hidden_states \
  --label_a baseline --label_b proxy \
  --training_log_a experiments/wm_mtmf_20260520_140601/training_log.json \
  --training_log_b experiments/finetune_proxy_wm_mtmf_20260705_164908/training_log.json \
  --match_metric val_novel_angle_acc \
  --split val_novel_identity \
  --output_dir analysis_results/neural_efficiency_baseline_vs_proxy_mtmf
```

(This exact command is already running — see §6. The paths above are the real, confirmed directory names, not placeholders.)

`--training_log_a`/`--training_log_b` auto-select the closest-accuracy epoch pair instead of requiring you to inspect `training_log.json` by hand and pass `--epoch_a`/`--epoch_b` yourself — it prints the resulting accuracy gap so you can judge whether the match is close enough to trust.

The scrambled-feature-label causal control (§5 of the future-work doc) is a stretch goal for a "future work" paragraph in the report, not something to execute before submission unless everything above is finished early.

---

## 5. Report-writing note

Structure the new chapter around the three-level table (representational content / population activity / explicit gating), not around "attention" and "proxy" as separate sub-sections — the point of this plan is that they converge on one claim (familiarity and structure suppress irrelevant processing), observed three different ways. State plainly which of the three you completed and which remain proposed, the same verified-vs-not-verified discipline already used for citations in `docs/FUTURE_WORK_NEURAL_EFFICIENCY.md`.

---

## 6. Live status (updated as each step completes, 2026-07-26)

| Step | Status | Output |
|---|---|---|
| §1 `neural_efficiency.py` baseline vs. proxy-finetuned, mtmf | Running (`hamrah-gpu-internal`, nohup) | `analysis_results/neural_efficiency_baseline_vs_proxy_mtmf/` |
| §1 `compare_models.py` baseline vs. attention, mtmf | Running (`hamrah-gpu-internal`, nohup) | `analysis_results/compare_baseline_vs_attention_mtmf/` |
| §2a `train_proxy.py` (proxy_attention_mtmf) | Queued next | `experiments/proxy_attention_mtmf_<timestamp>/` |
| §2a `finetune_from_proxy.py` (from above) | Queued, depends on above | `experiments/finetune_proxy_wm_attention_mtmf_<timestamp>/` |
| §2b `train_with_generalization.py` (attention_mtmf regate) | Queued next, parallel with 2a | `experiments/wm_attention_mtmf_<new-timestamp>/` |
| §3 `gate_suppression.py` attention-only vs. attention+proxy | Blocked on 2a+2b | `analysis_results/gate_suppression_mtmf/` |
| §4 `neural_efficiency.py` attention-only vs. attention+proxy | Blocked on 2a+2b | `analysis_results/neural_efficiency_attention_vs_attention_proxy_mtmf/` |

Update this table in place rather than appending a new one each time a step finishes.
