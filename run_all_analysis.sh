#!/bin/bash
# Run comprehensive analysis on all experiments
# Analyzes all trained models in experiments/ directory

set -e  # Exit on error

# Setup environment
cd "$(dirname "$0")"
source ~/.venv/WM-model/bin/activate
export PYTHONPATH="${PWD}/src:${PYTHONPATH}"

# Create analysis output directory
mkdir -p analysis_results
mkdir -p logs/analysis

echo "========================================"
echo "Running Comprehensive Analysis"
echo "========================================"

# Find all experiment directories with trained models
for exp_dir in experiments/wm_*/; do
    exp_name=$(basename "$exp_dir")
    
    # Check if experiment has hidden states and model
    if [ ! -d "${exp_dir}hidden_states" ]; then
        echo "Skipping $exp_name: no hidden_states directory"
        continue
    fi
    
    if [ ! -f "${exp_dir}best_model.pt" ]; then
        echo "Skipping $exp_name: no best_model.pt"
        continue
    fi
    
    echo ""
    echo "========================================"
    echo "Analyzing: $exp_name"
    echo "========================================"
    
    output_dir="analysis_results/${exp_name}"
    log_file="logs/analysis/${exp_name}.log"
    
    # Run comprehensive analysis (all 5 analyses)
    echo "Running all analyses..."
    python -m src.analysis.comprehensive_analysis \
        --analysis all \
        --hidden_root "${exp_dir}hidden_states" \
        --output_dir "$output_dir" \
        --model "${exp_dir}best_model.pt" \
        2>&1 | tee "$log_file"
    
    echo "Completed: $exp_name"
    echo "Results saved to: $output_dir"
    echo "Log saved to: $log_file"
done

echo ""
echo "========================================"
echo "All analyses completed!"
echo "Results in: analysis_results/"
echo "Logs in: logs/analysis/"
echo "========================================"
