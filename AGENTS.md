# AGENTS.md

This file defines the preferred workflow for agents working in this repository.

## Scope
- Project: `WM-model` (PyTorch working-memory training + analysis).
- Primary goal for most tasks: run or modify training/analysis workflows without breaking reproducibility.

## Environment
- Local setup:
  - `source venv/bin/activate`
  - `export PYTHONPATH="${PWD}/src:${PYTHONPATH}"`
- GPU server setup (from steering):
  - `ssh hamrah-gpu`
  - `cd ~/Projects/WM-model`
  - `git pull`
  - `conda deactivate`
  - `source ~/.venv/WM-model/bin/activate`
  - `export PYTHONPATH="${PWD}/src:${PYTHONPATH}"`

## Canonical Training Commands
- Recommended training path (with generalization validation):
  - `python -m src.train_with_generalization --config configs/stsf.yaml`
- Background run on GPU:
  - `nohup python -m src.train_with_generalization --config configs/stsf.yaml > train.log 2>&1 &`

## Config Selection
- `configs/stsf.yaml`: 2-back, category only (fastest).
- `configs/stmf.yaml`: 2-back, all tasks.
- `configs/mtmf.yaml`: 1/2/3-back, all tasks (full).
- `configs/attention_*.yaml`: attention-enabled variants.

## Monitoring
- Follow logs: `tail -f train.log`
- Check process: `ps aux | grep python`
- Check GPU: `nvidia-smi`
- Outputs location: `experiments/`

## Analysis Entry Points
- Decoding:
  - `python -m src.analysis.decoding --hidden_root experiments/<exp_name>/hidden_states --property identity --train_time 2 --test_times 3 4 5`
- Procrustes:
  - `python -m src.analysis.procrustes --hidden_root experiments/<exp_name>/hidden_states --property identity --source_time 2 --target_time 3`
- Verification:
  - `python -m src.scripts.verify_analysis_setup`
- Metric plots (mean across repeated runs):
  - `python -m src.scripts.plot_experiments --exp_dir experiments --output_dir plots`

## Meta-Learning Entry Points
- Run meta-learning experiment:
  - `python -m src.meta_learning --exp-dir experiments/<pretrained_model> --task <task_name> --shots 50 --epochs 20 --output-dir experiments/meta_learning_<task_name>`
- Background run on GPU:
  - `nohup python -m src.meta_learning --exp-dir experiments/<pretrained_model> --task <task_name> --shots 50 --epochs 20 --output-dir experiments/meta_learning_<task_name> > meta_learning_<model_type>_<task_name>.log 2>&1 &`
- Plot meta-learning results:
  - `python -m src.scripts.plot_meta_learning --exp_dir experiments/meta_learning_<task_name> --output_dir plots/meta_learning_<task_name>`

### Meta-Learning Naming Conventions
- **Log files**: `meta_learning_<arch>_<config>_<task_name>.log`
  - Example: `meta_learning_base_mtmf_three_in_a_row.log`
  - Example: `meta_learning_attention_mtmf_three_in_a_row.log`
- **Results directory**: `experiments/meta_learning_<arch>_<config>_<task_name>/`
  - Example: `experiments/meta_learning_base_mtmf_three_in_a_row/`
  - Example: `experiments/meta_learning_attention_stmf_nback_4/`
- **Result files**: Auto-generated as `meta_learning_<task>_<method>_<timestamp>.json`
- **Available tasks**: `nback_4`, `nback_5`, `three_in_a_row`, `alternating`
- **Architecture types**: `base` (baseline RNN), `attention` (task-guided attention), `dual_attention` (dual attention)
- **Config types**: `mtmf` (multi-task multi-feature), `stmf` (single-task multi-feature), `stsf` (single-task single-feature)

## Expected Outputs
- Training artifacts are stored under `experiments/<exp_name>/`:
  - `config.yaml`, `training.log`, `training_log.json`, `best_model.pt`, `hidden_states/`.

## Agent Guardrails
- Prefer `src.train_with_generalization` over `src.train` unless explicitly requested.
- Keep config-driven behavior in `configs/*.yaml`; avoid hardcoding experiment settings.
- Do not change path assumptions used in steering (`~/Projects/WM-model`, `PYTHONPATH` pattern) unless task explicitly asks for it.
- Keep Python entry points under `src/` (including helper scripts in `src/scripts/`); avoid adding Python files under top-level `scripts/`.
