# AGENTS.md

This file defines the preferred workflow for agents working in this repository.

## Scope
- Project: `WM-model` (PyTorch working-memory training + analysis).
- Primary goal for most tasks: run or modify training/analysis workflows without breaking reproducibility.

## Environment

### Local
```bash
source venv/bin/activate
export PYTHONPATH="${PWD}/src:${PYTHONPATH}"
```

### GPU server (`ssh hamrah-gpu`)
```bash
cd ~/Projects/WM-model && git pull
conda deactivate
source ~/.venv/WM-model/bin/activate
export PYTHONPATH="${PWD}/src:${PYTHONPATH}"
```

### Local PC (via `~/.virtualenvs/WM-model/`)
Used by `run_all_experiments.sh`. On this machine the venv is at `~/.virtualenvs/WM-model`; the same `PYTHONPATH` pattern applies.

**Always export `PYTHONPATH`** — imports depend on `src/` being on the path.

## Security
- `.env` contains a real `HUGGINGFACE_TOKEN`. **Never commit or expose `.env` contents.**
- The file is gitignored, but be careful not to print it or include it in patches.

## Data Pipeline
Stimuli already exist at `data/stimuli/` (320 images). Regenerate only if needed:
```bash
python -m src.data.download_shapenet --placeholder     # quick test data
# OR: python -m src.data.download_shapenet --download-hf ShapeNetCore.v2.zip
python -m src.data.generate_stimuli
```

## Training

### Entry points
- **Recommended**: `python -m src.train_with_generalization --config configs/stsf.yaml`  
  Uses novel-angle + novel-identity validation splits. Prefer over `src.train` unless explicitly asked.
- **Basic**: `python -m src.train --config configs/stsf.yaml`
- **Background (GPU)**: `nohup ... > train.log 2>&1 &`

### Config quick reference
| Config | N-values | Tasks | Notes |
|--------|----------|-------|-------|
| `stsf.yaml` | [2] | location | Fastest |
| `stmf.yaml` | [2] | location, identity, category | |
| `mtmf.yaml` | [1,2,3] | location, identity, category | Full paper config |
| `attention_*.yaml` | varies | varies | Sets `model_type: "attention"` |
| `dual_attention_*.yaml` | varies | varies | Dual-attention variant |

### Key config parameters
```yaml
hidden_size: 256       rnn_type: "gru"       # rnn|gru|lstm
epochs: 45             lr: 0.0001
save_hidden: true      # must be true for analysis
model_type: "base"     # "base"|"attention"|"dual_attention" — set by config
```

## Monitoring
```bash
tail -f train.log
ps aux | grep python
nvidia-smi
```
Outputs: `experiments/<exp_name>/`

## Analysis

### Individual analyses
```bash
# Decoding
python -m src.analysis.decoding --hidden_root experiments/<exp>/hidden_states \
  --property identity --train_time 2 --test_times 3 4 5

# Procrustes (temporal dynamics)
python -m src.analysis.procrustes --hidden_root experiments/<exp>/hidden_states \
  --property identity --source_time 2 --target_time 3

# Verify setup
python -m src.scripts.verify_analysis_setup

# Metric plots (across repeated runs)
python -m src.scripts.plot_experiments --exp_dir experiments --output_dir plots
```

### Comprehensive (all 5 paper analyses)
```bash
python -m src.analysis.comprehensive_analysis \
  --analysis all \
  --model experiments/<exp>/best_model.pt \
  --hidden_root experiments/<exp>/hidden_states \
  --property identity \
  --output_dir analysis_results
```

### Batch scripts
- `run_all_experiments.sh` — runs 9 config×architecture combinations
- `run_all_analysis.sh` — runs comprehensive analysis on all `experiments/wm_*/`

## Meta-Learning

### Run
```bash
python -m src.meta_learning --help
python -m src.meta_learning --list-models
python -m src.meta_learning --exp-dir experiments/<pretrained> --task <task> \
  --shots 50 --epochs 20 --output-dir experiments/meta_learning_<name>
```
- **Available tasks**: `nback_4`, `nback_5`, `three_in_a_row`, `alternating`
- **Methods**: `attention_only`, `full_finetune`, etc. (use `--help` to list)
- **Plot**: `python -m src.scripts.plot_meta_learning`

### Naming convention
- Log: `meta_learning_<arch>_<config>_<task>.log`
- Output dir: `experiments/meta_learning_<arch>_<config>_<task>/`
- Result files auto-named: `meta_learning_<task>_<method>_<timestamp>.json`
- Arch types: `base` | `attention` | `dual_attention`
- Config types: `stsf` | `stmf` | `mtmf`

## Expected Outputs
```
experiments/<exp_name>/
├── config.yaml              # Saved config
├── training.log             # Full log
├── training_log.json        # Per-epoch metrics
├── best_model.pt            # Best checkpoint
└── hidden_states/           # Activations for analysis
    └── epoch_XXX/batch_XXXX.pt
```

## Reference Docs
- `docs/ANALYSIS_METHODOLOGY.md` — complete technical methodology
- `docs/PROCRUSTES_GUIDE.md`, `docs/attention_model_guide.md`
- `QUICKSTART.md` — full workflow (data → train → analyze)

## Guardrails
- Prefer `src.train_with_generalization` over `src.train` unless asked.
- Keep config-driven behavior in `configs/*.yaml`; no hardcoded experiment settings.
- Keep Python entry points under `src/` (including `src/scripts/`).
- Do not change path assumptions (`~/Projects/WM-model`, `PYTHONPATH` pattern) unless the task explicitly asks.
- **Never commit `.env` or expose the HuggingFace token.**
