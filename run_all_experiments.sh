#!/bin/bash
# Run all experiments: 3 scenarios (stsf, stmf, mtmf) × 3 model types (base, attention, dual_attention)
# Total: 9 experiments

set -e  # Exit on error

# Setup environment
cd "$(dirname "$0")"
source ~/.virtualenvs/WM-model/bin/activate
export PYTHONPATH="${PWD}/src:${PYTHONPATH}"

# Create logs directory
mkdir -p logs

# Define experiments: config_file | experiment_name
declare -a EXPERIMENTS=(
    "stsf.yaml|base_stsf"
    "stmf.yaml|base_stmf"
    "mtmf.yaml|base_mtmf"
    "attention_stsf.yaml|attention_stsf"
    "attention_stmf.yaml|attention_stmf"
    "attention_mtmf.yaml|attention_mtmf"
    "dual_attention_stsf.yaml|dual_attention_stsf"
    "dual_attention_stmf.yaml|dual_attention_stmf"
    "dual_attention_mtmf.yaml|dual_attention_mtmf"
)

echo "========================================"
echo "Running all experiments"
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
    
    python -m src.train_with_generalization --config "configs/${config}" 2>&1 | tee "$log_file"
    
    echo "Completed: $name"
    echo "Log saved to: $log_file"
done

echo ""
echo "========================================"
echo "All experiments completed!"
echo "Results in: experiments/"
echo "Logs in: logs/"
echo "========================================"
