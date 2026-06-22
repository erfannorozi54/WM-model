#!/bin/bash
# Batch comprehensive analysis for all wm_h128_* experiments
# Run on GPU server with: nohup bash run_h128_analysis.sh > analysis_batch.log 2>&1 &
# Auto-discovers latest experiment directory for each prefix.

set -euo pipefail

PYTHON=~/.venv/WM-model/bin/python
BASE_DIR=~/Projects/WM-model
export PYTHONPATH="${BASE_DIR}/src:${PYTHONPATH:-}"
export MPLCONFIGDIR="${BASE_DIR}/.matplotlib"

mkdir -p "${BASE_DIR}/.matplotlib" "${BASE_DIR}/logs"

PREFIXES=(
    "wm_h128_stsf"
    "wm_h128_stmf"
    "wm_h128_mtmf"
    "wm_h128_attention_stsf"
    "wm_h128_attention_stmf"
    "wm_h128_attention_mtmf"
    "wm_h128_dual_attention_stsf"
    "wm_h128_dual_attention_stmf"
    "wm_h128_dual_attention_mtmf"
)

echo "========================================"
echo "H128 Comprehensive Analysis (auto-discover)"
echo "Total: ${#PREFIXES[@]} experiments"
echo "========================================"

for PREFIX in "${PREFIXES[@]}"; do
    EXP_DIR="$(find "${BASE_DIR}/experiments" -maxdepth 1 -type d \
        -name "${PREFIX}_[0-9]*" -printf '%T@ %p\n' | sort -nr | head -1 | cut -d' ' -f2-)"

    if [[ -z "${EXP_DIR}" || ! -f "${EXP_DIR}/best_model.pt" ]]; then
        echo "[$(date --iso-8601=seconds)] SKIP: No completed experiment for ${PREFIX}" >&2
        continue
    fi

    EXP="$(basename "${EXP_DIR}")"
    HIDDEN_ROOT="${EXP_DIR}/hidden_states"
    MODEL_PATH="${EXP_DIR}/best_model.pt"
    OUTPUT_DIR="${BASE_DIR}/analysis_results/${EXP}"

    if [[ ! -d "${HIDDEN_ROOT}" ]]; then
        echo "[$(date --iso-8601=seconds)] SKIP: No hidden_states for ${EXP}" >&2
        continue
    fi

    mkdir -p "${OUTPUT_DIR}"

    echo ""
    echo "========================================"
    echo "[$(date --iso-8601=seconds)] STARTING: ${EXP}"
    echo "========================================"

    "${PYTHON}" -m src.analysis.comprehensive_analysis \
        --analysis all \
        --hidden_root "${HIDDEN_ROOT}" \
        --output_dir "${OUTPUT_DIR}" \
        --property identity \
        --model "${MODEL_PATH}" 2>&1

    echo "[$(date --iso-8601=seconds)] FINISHED: ${EXP}"
    echo "========================================"
done

echo ""
echo "========================================"
echo "ALL H128 ANALYSES COMPLETE at $(date --iso-8601=seconds)"
echo "========================================"
