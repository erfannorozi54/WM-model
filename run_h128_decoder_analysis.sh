#!/bin/bash
# Run shared decoder-state extraction and Analysis 2 for all h128 checkpoints.

set -euo pipefail

PYTHON="${HOME}/.venv/WM-model/bin/python"
BASE_DIR="${HOME}/Projects/WM-model"
export PYTHONPATH="${BASE_DIR}/src:${PYTHONPATH:-}"
export MPLCONFIGDIR="${BASE_DIR}/.matplotlib"

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

mkdir -p "${BASE_DIR}/.matplotlib" "${BASE_DIR}/logs"

for EXP in "${EXPERIMENTS[@]}"; do
    EXP_DIR="${BASE_DIR}/experiments/${EXP}"
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

echo "[$(date --iso-8601=seconds)] All h128 decoder analyses completed"
