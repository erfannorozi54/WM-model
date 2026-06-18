#!/bin/bash
# Run all experiments with hidden_size=128 (halved from 256)
# 3 scenarios × 3 model types = 9 experiments

set -e

cd "$(dirname "$0")"
source ~/.venv/WM-model/bin/activate
export PYTHONPATH="${PWD}/src:${PYTHONPATH}"

mkdir -p logs

declare -a EXPERIMENTS=(
    "stsf.yaml|h128_stsf"
    "stmf.yaml|h128_stmf"
    "mtmf.yaml|h128_mtmf"
    "attention_stsf.yaml|h128_attention_stsf"
    "attention_stmf.yaml|h128_attention_stmf"
    "attention_mtmf.yaml|h128_attention_mtmf"
    "dual_attention_stsf.yaml|h128_dual_attention_stsf"
    "dual_attention_stmf.yaml|h128_dual_attention_stmf"
    "dual_attention_mtmf.yaml|h128_dual_attention_mtmf"
)

echo "========================================"
echo "Running all experiments (hidden_size=128)"
echo "Total: ${#EXPERIMENTS[@]} experiments"
echo "========================================"

for exp in "${EXPERIMENTS[@]}"; do
    IFS='|' read -r config name <<< "$exp"

    echo ""
    echo "========================================"
    echo "Experiment: $name"
    echo "Config: $config"
    echo "========================================"

    log_file="logs/train_${name}.log"

    python -m src.train_with_generalization --config "configs_128/${config}" 2>&1 | tee "$log_file"

    echo "Completed: $name"
    echo "Log saved to: $log_file"
done

echo ""
echo "========================================"
echo "All experiments completed!"
echo "Results in: experiments/"
echo "Logs in: logs/"
echo "========================================"
