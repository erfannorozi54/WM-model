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
├── meta_learning.py               # Meta-learning experiments
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
│   ├── activations.py             # load_payloads(), build_matrix(), build_matrix_with_values()
│   └── visualize_attention.py     # Attention weight visualization
├── data/                          # dataset, renderer, validation_splits, nback_generator
│   ├── proxy_generator.py         # Proxy task sequence generator
│   └── proxy_dataset.py           # Proxy task dataset and data module
├── scripts/
│   ├── plot_experiments.py         # Training metric plots across experiments
│   ├── plot_meta_learning.py       # Meta-learning result plots
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

Two parallel sets exist (differ in `hidden_size`):
- `configs/*.yaml` — `hidden_size: 256`, experiments prefixed `wm_*`
- `configs_128/*.yaml` — `hidden_size: 128`, experiments prefixed `wm_h128_*`

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

1. **Causal perturbation loads all epochs by default**: `load_payloads()` without `epochs=` loads every batch from every epoch. `comprehensive_analysis.py` passes `epochs=[best_epoch]` for analysis 5. If calling `causal_perturbation.py` directly, filter epochs yourself.

2. **Dual-attention model loading**: In `causal_perturbation.py`, `model_type="dual_attention"` maps to `attention_{rnn_type}` with `attention_mode="dual"`. This must match how the model was trained (via `dual_attention_*.yaml` configs).

3. **STSF single-task**: STSF experiments have only 1 task, so cross-task analyses (Analysis 2b) skip with "Only 1 task(s) available". This is expected.

4. **Identity decoding with small class count**: When decoding `identity` (70+ classes) in multi-task experiments, many classes have <2 samples after filtering. The pipeline falls back to non-stratified `train_test_split` when `class_counts.min() < 2`. Sample-size warnings are printed when `n_test < n_classes` or `n_test < 2 × n_classes`.

5. **H2 cross-stimulus uses val_novel splits** (not cross-time): the test trains on `val_novel_angle` (known identities) and tests on `val_novel_identity` (novel identities) at the same t=0. Both earlier (H1 cross-time) and H2 cross-stimulus results live in `analysis4_wm_dynamics.json`.

6. **Procrustes swap test label alignment**: `swap_hypothesis_test` in `procrustes.py` splits trials by `identity` hash (for cross-stimulus effect) but decodes on `location` (4 fixed classes) — identity labels are unique per trial and would not align between disjoint identity groups. Results: `correct_accuracy`, `swap1_accuracy` (wrong time), `swap2_accuracy` (different stimuli, same age), `baseline_accuracy`, `hypothesis_confirmed` (true when swap2 is closer to correct than swap1).

7. **Causal perturbation direction**: uses the **mean** of all class decoder normals as the perturbation direction. Per-class direction (pushing toward a specific class) was tested and is weaker — it pushes the state deeper into the class instead of across the boundary.

8. **Determinism**: `LinearSVC` and `SVC` in analysis modules use `max_iter=10000` and `random_state=42` to avoid convergence warnings and ensure reproducibility. `train_test_split` also uses `random_state=42`.

See `docs/ANALYSIS_AUDIT_FINDINGS.md` for the full audit of the 5 analyses against the paper.

## Meta-Learning

```bash
python -m src.meta_learning --help
python -m src.meta_learning --list-models
python -m src.meta_learning --exp-dir experiments/<pretrained> --task <task> \
  --shots 50 --epochs 20 --output-dir experiments/meta_learning_<name>
```

- **Tasks**: `nback_4`, `nback_5`, `three_in_a_row`, `alternating` (defined in `src/meta/tasks.py`)
- **Methods**: `attention_only`, `full_finetune`, etc. (`--help` to list)
- **Plot**: `python -m src.scripts.plot_meta_learning`
- **Naming**: `meta_learning_<arch>_<config>_<task>`

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
- **Novel task vectors**: Proxy generator supports novel tasks (nback_4, nback_5, three_in_a_row, alternating) with appropriate proxy targets.

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
