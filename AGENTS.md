# AGENTS.md

Instructions for agents working in this repository. Only includes what an agent would likely miss without help.

## Environment

`PYTHONPATH` must include `src/` or nothing imports:

```bash
export PYTHONPATH="${PWD}/src:${PYTHONPATH}"
```

| Machine | Venv activate | SSH alias |
|---------|--------------|-----------|
| Local | `source venv/bin/activate` | — |
| GPU server | `source ~/.venv/WM-model/bin/activate` | `hamrah-gpu-internal` |
| Local PC (batch scripts) | `source ~/.virtualenvs/WM-model/bin/activate` | — |

On the GPU server, always `conda deactivate` first if a conda base is active.

## Security

- `.env` contains a real `HUGGINGFACE_TOKEN`. Never commit, print, or expose it.
- `.env` is gitignored but exists locally — do not `git add -f` it.

## Architecture

```
src/
├── train_with_generalization.py   # Main training entry (prefer over train.py)
├── train.py                       # Basic training (no novel-angle/novel-identity splits)
├── train_proxy.py                 # Proxy task pre-training (feature recall N-back)
├── finetune_from_proxy.py         # Fine-tune proxy-pretrained model on real N-back
├── models/
│   ├── model_factory.py           # create_model() + create_proxy_model()
│   ├── wm_model.py                # WorkingMemoryModel (baseline)
│   ├── attention.py               # AttentionWorkingMemoryModel — modes: "task_only" | "dual"
│   ├── proxy_model.py             # ProxyWorkingMemoryModel (proxy pre-training)
│   ├── proxy_heads.py             # Multi-head proxy classifiers + loss functions
│   ├── perceptual.py              # ResNet50 perceptual encoder
│   └── cognitive.py               # VanillaRNN, GRUCog, LSTMCog
├── analysis/
│   ├── comprehensive_analysis.py  # Orchestrates all 5 paper analyses
│   ├── causal_perturbation.py     # Analysis 5 — needs --model flag and best_epoch filtering
│   ├── decoding.py                # Standalone decoding
│   ├── procrustes.py              # Procrustes + swap_hypothesis_test
│   ├── orthogonalization.py       # Analysis 3 — LinearSVC one-vs-rest
│   ├── compare_models.py          # Baseline vs. attention: decoding/orthogonalization comparison
│   ├── neural_efficiency.py       # Baseline vs. proxy-pretrained: hidden-state magnitude/PR/sparsity/Fano comparison
│   ├── gate_suppression.py        # Attention-gate suppression index (ranks channel relevance in CNN-activation space)
│   ├── activations.py             # load_payloads(), build_matrix(), build_matrix_with_values(), build_gate_matrix(), gate_channel_means()
│   └── visualize_attention.py     # Attention weight visualization — stale against the current (B,T,C) channel-gate shape; do not trust without checking
├── data/                          # dataset, renderer, validation_splits, nback_generator
│   ├── proxy_generator.py         # Proxy task sequence generator
│   └── proxy_dataset.py           # Proxy task dataset and data module
├── scripts/
│   ├── plot_experiments.py         # Training metric plots across experiments
│   └── verify_analysis_setup.py    # Pre-flight check (5/5 tests)
├── utils/
│   └── proxy_visualization.py     # Proxy task visualization utilities
└── meta/                          # Novel task definitions for meta-learning
```

`model_type` in config controls architecture via `model_factory.py`:
- `"gru"` / `"rnn"` / `"lstm"` → `WorkingMemoryModel` (baseline)
- `"attention"` → `AttentionWorkingMemoryModel(attention_mode="task_only")`
- For dual attention, set `model_type: "attention"` + `attention_mode: "dual"` (the string `"dual_attention"` only appears in the experiment name, never in `model_type`).

## Training

```bash
# Recommended (novel-angle + novel-identity validation)
python -m src.train_with_generalization --config configs/stsf.yaml

# Basic (no generalization splits)
python -m src.train --config configs/stsf.yaml

# Background (GPU server)
nohup python -m src.train_with_generalization --config configs/mtmf.yaml > train.log 2>&1 &
```

### Configs

Two parallel sets exist (differ in `hidden_size` — **and in `num_val`**):
- `configs/*.yaml` — `hidden_size: 256`, `num_val: 400`, experiments prefixed `wm_*`
- `configs_128/*.yaml` — `hidden_size: 128`, `num_val: 2000`, experiments prefixed `wm_h128_*`

The `num_val` difference means the two sets are **not** a clean hidden-size comparison: every decoding analysis trains on the validation payloads, so the h128 runs get 5× the decoder samples. With 72 identity classes that is the difference between ~50 and ~270 test samples, and it moves identity decoding from ~0.40 to ~0.83. Match `num_val` before reporting any h256-vs-h128 table.

Naming pattern per set: `{stsf,stmf,mtmf}` × `{base, attention_, dual_attention_}` = 9 configs each.

| Config | N-values | task_features | Notes |
|--------|----------|---------------|-------|
| `stsf.yaml` | [2] | `["location"]` | Fastest. README is wrong — task_feature is `location`, not `category`. |
| `stmf.yaml` | [2] | all three | |
| `mtmf.yaml` | [1,2,3] | all three | Full paper config |

Dual attention: `model_type: "attention"` + `attention_mode: "dual"` (not `model_type: "dual_attention"`).

### Key config values
```yaml
hidden_size: 256 | 128       rnn_type: "gru"       # rnn|gru|lstm
epochs: 45                   lr: 0.0001
save_hidden: true            # MUST be true for analysis to work
```

### Outputs (gitignored)
```
experiments/<exp_name>/
├── config.yaml              # Saved config
├── training.log             # Full log
├── training_log.json        # Per-epoch metrics dict (list of dicts)
├── best_model.pt            # Checkpoint: {model_state_dict, config, val_novel_identity_acc, epoch, ...}
└── hidden_states/           # Activation payloads per epoch/split/batch
    └── epoch_XXX/<split>/batch_XXXX.pt
```

Splits under `hidden_states/epoch_XXX/`: `val_novel_angle` (same identities, new angles) and `val_novel_identity` (new identities).

`_find_best_epoch()` reads `training_log.json` and selects the epoch with highest `val_novel_identity_acc`.

## Analysis

### Comprehensive (all 5 paper analyses)
```bash
python -m src.analysis.comprehensive_analysis \
  --analysis all \
  --model experiments/<exp>/best_model.pt \
  --hidden_root experiments/<exp>/hidden_states \
  --property identity \
  --output_dir analysis_results/<exp>
```

- `--analysis 1|2|3|4|5` runs individual analyses. `--analysis 5` (causal perturbation) requires `--model`.
- Analysis 5 auto-detects best epoch and loads only that epoch's data.

### Individual analyses
```bash
python -m src.analysis.decoding --hidden_root experiments/<exp>/hidden_states \
  --property identity --train_time 2 --test_times 3 4 5

python -m src.analysis.procrustes --hidden_root experiments/<exp>/hidden_states \
  --property identity --source_time 2 --target_time 3

python -m src.scripts.verify_analysis_setup             # pre-flight check
python -m src.scripts.plot_experiments --exp_dir experiments --output_dir plots
```

### Batch scripts
- `run_all_experiments.sh` — 9 base experiments (uses `~/.virtualenvs/WM-model`)
- `run_all_experiments_h128.sh` — 9 h128 experiments
- `run_all_analysis.sh` — comprehensive analysis on all `experiments/wm_*/`
- `run_h128_analysis.sh` — comprehensive analysis on all `experiments/wm_h128_*/`

### Known gotchas

0. **Class indices must come from `make_label2idx` (value-sorted), never from order of appearance.** `build_matrix*` numbers classes by sorted raw value precisely so that a decoder/rotation built from one matrix can be scored against another matrix's labels (other split, other timestep, other stimulus group). Any new cross-matrix comparison must either pass `label2idx=` to share one class space explicitly, or map raw values through `_align_test_labels`. This was the root cause of the 2026-08-16 audit: the H2 cross-stimulus test and the Procrustes swap test were reporting label permutations as "our models are stimulus-specific (H3), unlike the paper". Tell-tale sign: an accuracy *below* chance (a failed 4-class decoder floors at 0.25, it does not reach 0.000). See the second-audit section of `docs/ANALYSIS_AUDIT_FINDINGS.md`.

0b. **A "our result contradicts the paper" cell needs the prediction re-derived from the paper summary, not from memory of it.** The 2026-08-16 third audit found the neural-efficiency deck reporting a contradiction with Constantinidis & Klingberg (2016) on participation ratio that does not exist: the prediction had been written down as "training sharpens tuning ⇒ lower PR" when the review actually reports tuning getting **broader**, and PR (a population effective-dimensionality measure) does not track single-unit tuning width in a fixed direction anyway. Two rules that follow: (a) each graded row must name **which** reference it is graded against — Poppenk (familiarity ⇒ *lower* activity) and Constantinidis & Klingberg (WM training ⇒ *more* neurons, *higher* firing rate) make **opposite** magnitude predictions, so a single "vs. the references" verdict column is always wrong; (b) if a metric is our own operationalization rather than something a reference measured (population sparsity is one), mark it ungraded instead of putting a ✅ next to a reference. See the third-audit section of `docs/ANALYSIS_AUDIT_FINDINGS.md`.

0c. **Population metrics in `neural_efficiency.py` are N-sensitive, but check the magnitude before discarding a result.** `participation_ratio` and threshold-based `population_sparsity` both grow with N on identical data. The often-quoted "+44%/+42%" figures come from N=50 vs N=1600 at a true PR of ~20 — a regime this project never operates in. Measured at the real operating range (PR 1–10, N 200–330) the drift is **0–2% for PR and 2–4% for sparsity**, so it cannot explain larger effects. `compare_cell()` emits `trial_count_warning` when `n_trials_a != n_trials_b`: treat it as a prompt to check the effect size and the direction of the imbalance, not as an automatic disqualification. `fano_factor_analogue` is `Var/Mean` on a continuous signal and so scales **linearly with activation magnitude**, which is exactly what the magnitude metric reports differing between conditions; read `cv_squared` (scale-invariant) alongside it. `activation_magnitude` and `cv_squared` are unaffected by both issues.

1. **Causal perturbation loads all epochs by default**: `load_payloads()` without `epochs=` loads every batch from every epoch. **`compare_models.py` and `gate_suppression.py` are the two tools where this bites hardest**, because they compare *two different runs*: pooling epochs silently compares one run's whole training trajectory against another's, and a from-scratch run vs. a short fine-tune-from-proxy run are maximally mismatched that way (this invalidated the neural-efficiency chapter's Level 1 and Level 3 numbers). `compare_models.py` now takes `--best_epoch` / `--baseline_epochs` / `--attention_epochs` / `--split`; `gate_suppression.py` warns and sets `epochs_pooled` in its JSON when `--epoch_a`/`--epoch_b` are omitted. Always pass explicit, accuracy-matched epochs for cross-run comparisons. `comprehensive_analysis.py` passes `epochs=[best_epoch]` for analysis 5. If calling `causal_perturbation.py` directly, filter epochs yourself. Inside `ComprehensiveAnalysis`, always go through `_ensure_data_loaded()` rather than `load_data()` — analyses must see the same best-epoch data whether run individually or under `--analysis all`, and pooling epochs also puts the same trial in both sides of the decoder split (trials are only identified by position in the loaded payload list, since training does not save `sample_index`).

1b. **`attention_mode: "task_only"` gates are a pure function of the task vector.** Every trial in a `(task_index, n)` cell from one checkpoint carries an identical gate vector, so a trial-level bootstrap over gates has nothing to resample and returns a zero-width CI that looks precise and means nothing. `gate_suppression.py` reports `n_distinct_gate_vectors` and `ci_degenerate`; treat a degenerate interval as absent, not as tight. Real uncertainty there lives across channels, checkpoints and seeds. (`"dual"` mode gates do vary per input.)

2. **Dual-attention model loading**: In `causal_perturbation.py`, `model_type="dual_attention"` maps to `attention_{rnn_type}` with `attention_mode="dual"`. This must match how the model was trained (via `dual_attention_*.yaml` configs).

3. **STSF single-task**: STSF experiments have only 1 task, so cross-task analyses (Analysis 2b) skip with "Only 1 task(s) available". This is expected.

4. **Identity decoding with small class count**: When decoding `identity` (70+ classes) in multi-task experiments, many classes have <2 samples after filtering. The pipeline falls back to non-stratified `train_test_split` when `class_counts.min() < 2`. Sample-size warnings are printed when `n_test < n_classes` or `n_test < 2 × n_classes`.

5. **H2 cross-stimulus uses val_novel splits** (not cross-time): the test trains on `val_novel_angle` (known identities) and tests on `val_novel_identity` (novel identities) at the same t=0. Both earlier (H1 cross-time) and H2 cross-stimulus results live in `analysis4_wm_dynamics.json`.

5b. **H1 tracks the item, not the screen**: in `analysis4_wm_dynamics.json`, `accuracies` decodes the property of the stimulus shown at t=0 out of later hidden states (the memory-age test H1 actually needs). `accuracies_current_stimulus` decodes whatever is on screen at t — a stationarity measure, not H1. Both are scored on the same held-out 20% of trials, including at t=0; compare them against the reported `chance_level` (0.014 for 72-class identity), not against zero.

6. **Procrustes swap test label alignment**: `swap_hypothesis_test` in `procrustes.py` splits trials by `identity` hash (for cross-stimulus effect) but decodes on `location` (4 fixed classes) — identity labels are unique per trial and would not align between disjoint identity groups. Results: `correct_accuracy`, `swap1_accuracy` (wrong time), `swap2_accuracy` (different stimuli, same age), `baseline_accuracy`, `hypothesis_confirmed` (true when swap2 is closer to correct than swap1).

7. **Causal perturbation direction**: uses the **mean** of all class decoder normals as the perturbation direction. Per-class direction (pushing toward a specific class) was tested and is weaker — it pushes the state deeper into the class instead of across the boundary. The normals are taken with `one_vs_rest_weights(..., input_space=True)` because the perturbation is added to *raw* hidden states while the SVC is fitted on standardized ones; and `--perturbation_range` is interpreted in SDs of the hidden states' projection onto that direction (`projection_sd` in the output JSON), since raw units are not comparable across models.

7b. **Stimulus-group splits must use `_stable_hash`**, not the builtin `hash()`: string hashing is salted per interpreter run, so group A/B membership in the Procrustes swap test changed between otherwise identical runs.

8. **Determinism**: `LinearSVC` and `SVC` in analysis modules use `max_iter=10000` and `random_state=42` to avoid convergence warnings and ensure reproducibility. `train_test_split` also uses `random_state=42`.

9. **`gates` payload key only exists on runs started after the gate-logging fix**: `AttentionWorkingMemoryModel.forward()` supports `return_cnn_activations=True` and `return_attention=True` simultaneously; `train_with_generalization.py`/`finetune_from_proxy.py` detect attention models via `hasattr(model, "attention")` and save gates accordingly. Older attention experiment directories will have `gates: None` even with `save_hidden: true`. Also, gates are applied to `cnn_activations` (the pre-RNN visual embedding), not RNN `hidden` states — any channel-relevance ranking against gate values must decode properties from `cnn_activations` (`build_cnn_matrix`), never from `hidden`, or it will compare the wrong channel space. See `src/analysis/gate_suppression.py`'s module docstring for the full rationale, including why relevance ranking is computed independently per experiment rather than assuming shared channel identity across differently-trained runs.

10. **BLAS thread contention in `neural_efficiency.py`/`gate_suppression.py`/`compare_models.py`**: these call numpy/sklearn linear algebra thousands of times inside 1000-iteration bootstrap loops on small matrices. `src/analysis/__init__.py` caps `OMP_NUM_THREADS`/`OPENBLAS_NUM_THREADS`/`MKL_NUM_THREADS`/`NUMEXPR_NUM_THREADS` to `1` as the very first thing it does (before any numpy import) — do not remove this. Without it, OpenBLAS spawns a full thread pool per call; two such jobs running concurrently on the GPU server drove load average to ~100 on a 64-core box and turned a ~1-2 minute analysis into 4+ hours with zero output. This only takes effect for code invoked as `python -m src.analysis.*` (package `__init__.py` runs first); a standalone script importing e.g. `neural_efficiency.py` directly without going through the package would not get this protection. Separately, `decoding.py`'s `train_decoder()` uses `SVC(kernel="linear", ...)` (libsvm backend, scales ~O(n²–n³) with sample count) rather than `LinearSVC` (liblinear, ~O(n)) — on `compare_models.py`'s swap-test/Procrustes stages with ~20,000-trial splits this is still slow (hours) even single-threaded; known but intentionally not changed, since other already-reported thesis results depend on this exact solver's numbers.

See `docs/ANALYSIS_AUDIT_FINDINGS.md` for the full audit of the 5 analyses against the paper.

## Proxy Task Pre-training

Two-stage training: (1) pre-train on proxy task (feature recall), (2) fine-tune on real N-back.

### Proxy Task
Instead of 3-class match/non_match/no_action, the proxy task asks the model to **recall the feature value from N steps back**:
- Location: predict which of 4 locations (4-class)
- Identity: predict which identity (N-class)
- Category: predict which category (4-class)

The model uses **separate heads** for each feature type instead of a single 3-class classifier. Training uses all 9 standard task vectors (3 features x 3 N-values), balanced across tasks. Hidden state resets for each new task vector (fresh per sequence).

### Training
```bash
# Step 1: Proxy pre-training
python -m src.train_proxy --config configs/proxy/proxy_mtmf.yaml

# Step 2: Fine-tune on real N-back
python -m src.finetune_from_proxy \
  --proxy_exp_dir experiments/proxy_mtmf_<timestamp> \
  --config configs/mtmf.yaml
```

### Proxy Configs (`configs/proxy/`)
| Config | Architecture | Notes |
|--------|-------------|-------|
| `proxy_mtmf.yaml` | Base GRU | All 9 task vectors, balanced |
| `proxy_attention_mtmf.yaml` | Attention GRU (task_only) | |
| `proxy_dual_attention_mtmf.yaml` | Attention GRU (dual) | |

### Outputs
```
experiments/proxy_<exp>_<timestamp>/
├── best_model.pt            # Proxy model: {model_state_dict, proxy_heads_state_dict, identity_mapping, ...}
├── training_log.json        # Per-epoch proxy accuracy per task
├── visualizations/          # Proxy task visualizations (feature recall)
└── hidden_states/           # Activation payloads

experiments/finetune_proxy_<exp>_<timestamp>/
├── best_model.pt            # Standard model (same format as other experiments)
├── training_log.json        # Standard N-back metrics
└── ...                      # Same structure as regular experiments
```

### Weight Transfer
`finetune_from_proxy.py` transfers perceptual + attention + cognitive weights from the proxy model to a standard model. The classifier is initialized fresh. The resulting model has the same architecture as models trained from scratch, enabling direct comparison.

### Key Details
- **Identity mapping**: Proxy training builds a mapping from identity names to indices. This mapping is saved in the checkpoint and must be consistent.
- **Balanced training**: Proxy data is generated with equal samples per task vector (3 features x 3 N-values = 9 groups).
- **Loss**: Per-sample cross-entropy with the appropriate head selected by task vector. Groups by task feature for batched computation.

## Data Pipeline

Stimuli already exist at `data/stimuli/` (320 images: 4 categories × 5 identities × 4 locations × 4 angles). Regenerate only if needed:

```bash
python -m src.data.download_shapenet --placeholder     # quick test data
python -m src.data.generate_stimuli
```

For real ShapeNet data, set `HUGGINGFACE_TOKEN` in `.env` then `python -m src.data.download_shapenet --download-hf ShapeNetCore.v2.zip` (25 GB).

## Guardrails

- Prefer `src.train_with_generalization` over `src.train` unless explicitly asked.
- Keep config-driven behavior in `configs/*.yaml` and `configs_128/*.yaml`; no hardcoded experiment settings.
- Keep Python entry points under `src/` (including `src/scripts/`).
- Do not change path assumptions (`~/Projects/WM-model`, `PYTHONPATH` pattern) unless the task explicitly asks.
- Never commit `.env` or expose the HuggingFace token.
- `save_hidden: true` must be set in config for analysis to work.
- `experiments/`, `analysis_results/`, and `*.pt` are gitignored — they are not in the repo, only on local/GPU machines.

## Monitoring

```bash
tail -f train.log
ps aux | grep python
nvidia-smi
```
