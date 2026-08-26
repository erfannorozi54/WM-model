#!/usr/bin/env bash
# Train the baseline / attention / dual-attention models.
#
# Replaces run_all_experiments.sh + run_all_experiments_h128.sh, which were the
# same script twice over with a different config directory and a different venv.
#
# Usage (from the repo root):
#   ./run_training.sh                # both hidden sizes, 18 experiments
#   ./run_training.sh h256           # configs/      -> wm_*        (9)
#   ./run_training.sh h128           # configs_128/  -> wm_h128_*   (9)
#   ./run_training.sh h256 mtmf attention_mtmf     # named configs only
#
# Long runs: nohup ./run_training.sh h256 > train.log 2>&1 & disown
# Check nvidia-smi and uptime first -- see AGENTS.md on BLAS thread contention.

source "$(dirname "$0")/run_common.sh"

SCENARIOS=(stsf stmf mtmf attention_stsf attention_stmf attention_mtmf
           dual_attention_stsf dual_attention_stmf dual_attention_mtmf)

SIZES=()
case "${1:-all}" in
  h256) SIZES=(h256); shift ;;
  h128) SIZES=(h128); shift ;;
  all)  SIZES=(h256 h128); shift || true ;;
  *)    SIZES=(h256 h128) ;;
esac

# Any remaining arguments name specific configs (without the .yaml suffix).
if [[ $# -gt 0 ]]; then
  SCENARIOS=("$@")
fi

start_provenance "training"
{
  echo "sizes:      ${SIZES[*]}"
  echo "scenarios:  ${SCENARIOS[*]}"
} | tee -a "$PROV"

for size in "${SIZES[@]}"; do
  if [[ "$size" == "h128" ]]; then CFGDIR=configs_128; else CFGDIR=configs; fi

  for scenario in "${SCENARIOS[@]}"; do
    cfg="${CFGDIR}/${scenario}.yaml"
    if [[ ! -f "$cfg" ]]; then
      echo "SKIP ${size}/${scenario}: no such config ($cfg)" | tee -a "$PROV"
      continue
    fi
    run_step "train_${size}_${scenario}" \
      python -m src.train_with_generalization --config "$cfg"
  done
done

finish_provenance
echo "Experiments in: experiments/"
