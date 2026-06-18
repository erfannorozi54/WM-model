#!/bin/bash
# Batch comprehensive analysis for all wm_h128_* experiments
# Run on GPU server with: nohup bash run_h128_analysis.sh > analysis_batch.log 2>&1 &

PYTHON=~/.venv/WM-model/bin/python
BASE_DIR=~/Projects/WM-model
export PYTHONPATH="${BASE_DIR}/src:${PYTHONPATH}"

EXPERIMENTS=(
    "wm_h128_stsf_20260602_230425"
    "wm_h128_stmf_20260603_010546"
    "wm_h128_mtmf_20260603_031838"
    "wm_h128_attention_stsf_20260603_053139"
    "wm_h128_attention_stmf_20260603_073349"
    "wm_h128_attention_mtmf_20260603_094851"
    "wm_h128_dual_attention_stsf_20260603_120432"
    "wm_h128_dual_attention_stmf_20260603_140709"
    "wm_h128_dual_attention_mtmf_20260603_162319"
)

for EXP in "${EXPERIMENTS[@]}"; do
    echo ""
    echo "========================================"
    echo "STARTING: $EXP at $(date)"
    echo "========================================"

    HIDDEN_ROOT="$BASE_DIR/experiments/$EXP/hidden_states"
    MODEL_PATH="$BASE_DIR/experiments/$EXP/best_model.pt"
    OUTPUT_DIR="$BASE_DIR/analysis_results/$EXP"

    mkdir -p "$OUTPUT_DIR"

    $PYTHON -m src.analysis.comprehensive_analysis \
        --analysis all \
        --hidden_root "$HIDDEN_ROOT" \
        --output_dir "$OUTPUT_DIR" \
        --property identity \
        --model "$MODEL_PATH" 2>&1

    echo "FINISHED: $EXP at $(date)"
    echo "========================================"
done

echo ""
echo "========================================"
echo "ALL ANALYSES COMPLETE at $(date)"
echo "========================================"
