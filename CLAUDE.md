# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

> **`AGENTS.md` in the repo root is the primary, actively-maintained reference for this codebase** — it has detailed architecture notes, config gotchas, and a running list of known pitfalls in the analysis pipeline. Read it before making non-trivial changes. This file summarizes the essentials and defers to `AGENTS.md` for depth.

## What this is

PyTorch implementation of a two-stage neural network for N-back working memory tasks (paper 2411.02685): a frozen ResNet50 perceptual encoder feeds a task vector + RNN/GRU/LSTM cognitive module, trained to classify each trial as `no_action` / `non_match` / `match`.

## Environment setup

`PYTHONPATH` must include `src/` or nothing imports:

```bash
export PYTHONPATH="${PWD}/src:${PYTHONPATH}"
```

| Machine | Venv activate |
|---------|--------------|
| Local | `source venv/bin/activate` |
| GPU server (`hamrah-gpu-internal`) | `source ~/.venv/WM-model/bin/activate` (run `conda deactivate` first if conda base is active) |
| Local PC batch scripts | `source ~/.virtualenvs/WM-model/bin/activate` |

`.env` holds a real `HUGGINGFACE_TOKEN` — never commit, print, or expose it; it's gitignored.

GPU server access: `ssh hamrah-gpu-internal` (a configured SSH alias resolving to an internal IP — reachable only when the VPN is up), then `cd ~/Projects/WM-model && git pull`. Long-running training/analysis must be launched with `nohup ... > out.log 2>&1 & disown` so it survives the SSH session ending; check two idle RTX 3090s with `nvidia-smi` before launching anything, and check `uptime`'s load average if a job seems to be taking far longer than expected (see the BLAS-threading gotcha in `AGENTS.md`).

## Common commands

```bash
# Train (recommended — includes novel-angle/novel-identity validation splits)
python -m src.train_with_generalization --config configs/stsf.yaml

# Basic training (no generalization splits)
python -m src.train --config configs/stsf.yaml

# Background training on GPU server
nohup python -m src.train_with_generalization --config configs/mtmf.yaml > train.log 2>&1 &

# Proxy pre-training, then fine-tune on real N-back
python -m src.train_proxy --config configs/proxy/proxy_mtmf.yaml
python -m src.finetune_from_proxy --proxy_exp_dir experiments/proxy_mtmf_<timestamp> --config configs/mtmf.yaml

# Comprehensive analysis (all 5 paper analyses) on a finished experiment
python -m src.analysis.comprehensive_analysis --analysis all \
  --model experiments/<exp>/best_model.pt \
  --hidden_root experiments/<exp>/hidden_states \
  --property identity --output_dir analysis_results/<exp>

# Pre-flight check for the analysis pipeline
python -m src.scripts.verify_analysis_setup

# Single manual test (no pytest suite in this repo)
python -m src.data.test_validation_splits
```

Batch scripts: `run_all_experiments.sh` / `run_all_experiments_h128.sh` (9 experiments each), `run_all_analysis.sh` / `run_h128_analysis.sh` (comprehensive analysis over all matching `experiments/` dirs).

## Architecture

```
Input Images (B,T,3,224,224)
    → ResNet50 (frozen) → 1×1 Conv → GAP → Visual Embedding (B,T,H)
    → Concat with Task Vector (B,T,H+3)
    → RNN/GRU/LSTM → Hidden States (B,T,H)
    → Linear Classifier → Logits (B,T,3)  [no_action | non_match | match]
```

```
src/
├── train_with_generalization.py   # Main training entry (prefer over train.py)
├── train.py                       # Basic training, no novel-angle/novel-identity splits
├── train_proxy.py / finetune_from_proxy.py   # Two-stage proxy pre-training pipeline
├── models/
│   ├── model_factory.py           # create_model() + create_proxy_model()
│   ├── wm_model.py                # WorkingMemoryModel (baseline)
│   ├── attention.py               # AttentionWorkingMemoryModel — modes: "task_only" | "dual"
│   ├── proxy_model.py / proxy_heads.py
│   ├── perceptual.py              # ResNet50 encoder
│   └── cognitive.py                # VanillaRNN, GRUCog, LSTMCog
├── analysis/
│   ├── comprehensive_analysis.py  # Orchestrates all 5 paper analyses
│   ├── causal_perturbation.py     # Analysis 5 — needs --model + best_epoch filtering
│   ├── decoding.py / procrustes.py / orthogonalization.py
│   ├── compare_models.py          # Baseline vs. attention: decoding/orthogonalization comparison
│   ├── neural_efficiency.py       # Baseline vs. proxy-pretrained: hidden-state magnitude/participation-ratio/sparsity/Fano comparison (two-hidden_root CLI, not part of comprehensive_analysis.py)
│   ├── gate_suppression.py        # Attention-gate suppression index (ranks relevance in CNN-activation space, not RNN space — see module docstring)
│   └── activations.py             # load_payloads(), build_matrix()
├── data/                          # dataset, renderer, validation_splits, nback_generator, proxy_generator/dataset
└── scripts/                       # plot_experiments.py, verify_analysis_setup.py
```

**`model_type` in config controls architecture** (`model_factory.py`):
- `"gru"` / `"rnn"` / `"lstm"` → baseline `WorkingMemoryModel`
- `"attention"` → `AttentionWorkingMemoryModel(attention_mode="task_only")`
- Dual attention = `model_type: "attention"` + `attention_mode: "dual"`. The string `"dual_attention"` only ever appears in experiment/config *names*, never as a `model_type` value — except inside `causal_perturbation.py`, which maps a literal `model_type="dual_attention"` argument to `attention_{rnn_type}` + `attention_mode="dual"`.

## Configs

Two parallel sets, differing only in `hidden_size`:
- `configs/*.yaml` — `hidden_size: 256`, experiment names prefixed `wm_*`
- `configs_128/*.yaml` — `hidden_size: 128`, experiment names prefixed `wm_h128_*`

Naming pattern per set: `{stsf,stmf,mtmf}` × `{base, attention_, dual_attention_}` = 9 configs.

| Config | N-values | task_features |
|--------|----------|---------------|
| `stsf.yaml` | `[2]` | `["location"]` (fastest) |
| `stmf.yaml` | `[2]` | all three |
| `mtmf.yaml` | `[1,2,3]` | all three (full paper config) |

`save_hidden: true` **must** be set for the analysis pipeline to work.

## Outputs (gitignored — not in the repo, only on local/GPU machines)

```
experiments/<exp_name>/
├── config.yaml
├── training.log
├── training_log.json        # per-epoch metrics; _find_best_epoch() picks highest val_novel_identity_acc
├── best_model.pt            # {model_state_dict, config, val_novel_identity_acc, epoch, ...}
└── hidden_states/epoch_XXX/<split>/batch_XXXX.pt   # splits: val_novel_angle, val_novel_identity
```

Saved payload keys: `hidden`, `cnn_activations`, `logits`, `task_vector`, `task_index`, `n`, `targets`, `locations`, `categories`, `identities`, `split`, plus `gates` (attention models only, `None` otherwise) — `gates` is only populated by `train_with_generalization.py`/`finetune_from_proxy.py` runs started after the gate-logging fix; older attention-model experiment directories won't have it even if `save_hidden: true`.

`experiments/`, `analysis_results/`, and `*.pt` files are all gitignored.

## Data

Stimuli already exist at `data/stimuli/` (320 images: 4 categories × 5 identities × 4 locations × 4 angles); regenerate only if needed via `python -m src.data.generate_stimuli`. Real ShapeNet data requires `HUGGINGFACE_TOKEN` in `.env`.

## Guardrails

- Prefer `src.train_with_generalization` over `src.train` unless explicitly asked otherwise.
- Keep config-driven behavior in `configs/*.yaml` / `configs_128/*.yaml`; no hardcoded experiment settings.
- Keep Python entry points under `src/` (including `src/scripts/`).
- Don't change path assumptions (`~/Projects/WM-model`, the `PYTHONPATH` pattern) unless explicitly asked.
- For the full list of analysis-pipeline gotchas (causal perturbation epoch filtering, dual-attention model loading, Procrustes swap-test label alignment, decoding fallback behavior for small classes, BLAS thread contention, etc.), see the "Known gotchas" section of `AGENTS.md`.
- The **neural-efficiency thesis chapter** (attention-gate suppression + proxy pretraining, distinct from the accuracy/performance result already in `slidev-presentation/`) has one authoritative document: **`docs/NEURAL_EFFICIENCY.md`** — claim, what each reference does and does not license, method, results with confidence levels, and reproduction steps. Read it before touching anything in this area; it supersedes the planning docs now in `docs/archive/`. Note the proxy-pretraining accuracy gain is *not* a capacity claim (never tested at higher N-back or more simultaneous items). The two human-neuroscience references (Poppenk, Moscovitch & McIntosh 2016; Constantinidis & Klingberg 2016) are summarized in full from the source PDFs in `docs/PAPER_EXPLAINED_POPPENK_2016.md` and `docs/PAPER_EXPLAINED_CONSTANTINIDIS_KLINGBERG_2016.md` — read those before re-deriving a summary. Run the analyses with `./run_neural_efficiency.sh`, never by hand-assembling the CLI flags: several results were confounded by omitted `--epoch_*`/`--split`/`--task` filters. `slidev-presentation/speaker_notes_neural_efficiency_onward.md` has the matching per-slide talking points.
