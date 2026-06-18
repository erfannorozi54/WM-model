#!/bin/bash
# Run shared decoder-state extraction and Analysis 2 for all h128 checkpoints.

set -euo pipefail

PYTHON="${HOME}/.venv/WM-model/bin/python"
BASE_DIR="${HOME}/Projects/WM-model"
export PYTHONPATH="${BASE_DIR}/src:${PYTHONPATH:-}"
export MPLCONFIGDIR="${BASE_DIR}/.matplotlib"

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

mkdir -p "${BASE_DIR}/.matplotlib" "${BASE_DIR}/logs"

for PREFIX in "${PREFIXES[@]}"; do
    EXP_DIR="$(find "${BASE_DIR}/experiments" -maxdepth 1 -type d \
        -name "${PREFIX}_[0-9]*" -printf '%T@ %p\n' | sort -nr | head -1 | cut -d' ' -f2-)"
    if [[ -z "${EXP_DIR}" || ! -f "${EXP_DIR}/best_model.pt" ]]; then
        echo "Missing completed experiment for ${PREFIX}" >&2
        exit 1
    fi
    EXP="$(basename "${EXP_DIR}")"
    DECODER_ROOT="${EXP_DIR}/decoder_hidden_states"
    OUTPUT_DIR="${BASE_DIR}/analysis_results/${EXP}"

    echo "[$(date --iso-8601=seconds)] Extracting shared decoder states: ${EXP}"
    "${PYTHON}" -m src.analysis.extract_decoder_states \
        --model "${EXP_DIR}/best_model.pt" \
        --output_root "${DECODER_ROOT}" \
        --samples_per_task 2100 \
        --seed 42 \
        --overwrite

    rm -rf "${OUTPUT_DIR}"
    echo "[$(date --iso-8601=seconds)] Running decoder analysis: ${EXP}"
    "${PYTHON}" -m src.analysis.comprehensive_analysis \
        --analysis 2 \
        --hidden_root "${DECODER_ROOT}" \
        --output_dir "${OUTPUT_DIR}"
done

"${PYTHON}" -m src.scripts.verify_decoder_results --base_dir "${BASE_DIR}"
echo "[$(date --iso-8601=seconds)] All h128 decoder analyses completed"
