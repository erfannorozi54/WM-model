#!/usr/bin/env bash
# Neural-efficiency chapter: the complete three-level analysis, in one place.
#
# Every flag here exists for a reason recorded in docs/NEURAL_EFFICIENCY.md.
# The 2026-07-27 run omitted several of them and its numbers were confounded --
# Level 3 in particular reversed once epochs were pinned. Do not drop flags to
# "simplify" a re-run.
#
# Usage (GPU server, from ~/Projects/WM-model):
#   source ~/.venv/WM-model/bin/activate    # conda deactivate first if base is active
#   ./run_neural_efficiency.sh              # all three levels
#   ./run_neural_efficiency.sh level1       # one level only
#   ./run_neural_efficiency.sh level2 level3
#
# Writes results into analysis_results/<per-run dirs> and logs into
# logs/neural_efficiency_rerun_<timestamp>/.

set -u

cd "$(dirname "$0")"
export PYTHONPATH="${PWD}/src:${PYTHONPATH:-}"

LEVELS="${*:-level1 level2 level3}"
want () { [[ " $LEVELS " == *" $1 "* ]]; }

LOGDIR="logs/neural_efficiency_rerun_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOGDIR"
PROV="$LOGDIR/00_provenance.log"
{
  echo "commit:  $(git rev-parse --short HEAD)  branch: $(git rev-parse --abbrev-ref HEAD)"
  echo "levels:  $LEVELS"
  echo "started: $(date -u)"
} | tee "$PROV"

SPLIT=val_novel_identity
EXP=experiments
BASELINE=$EXP/wm_mtmf_20260520_140601
BASELINE_PROXY=$EXP/finetune_proxy_wm_mtmf_20260705_164908
ATTENTION_L1=$EXP/wm_attention_mtmf_20260520_203605
ATTENTION=$EXP/wm_attention_mtmf_20260726_161735
ATTENTION_PROXY=$EXP/finetune_proxy_wm_attention_mtmf_20260726_201707

# Accuracy-matched epoch pairs, resolved once and pinned here rather than
# re-derived per run. Auto-selection via --training_log_a/--training_log_b picks
# whichever pair is closest at the time and would silently change these results.
EP_BASELINE_A=12; EP_BASELINE_B=1     # 82.7% vs 92.7% (10pp gap)
EP_ATTN_A=43;     EP_ATTN_B=1         # 93.43% vs 93.51% (0.08pp gap)

run () {
  local name="$1"; shift
  echo "=== [$(date -u +%H:%M:%S)] START $name ===" | tee -a "$PROV"
  "$@" > "$LOGDIR/${name}.log" 2>&1
  local rc=$?
  echo "=== [$(date -u +%H:%M:%S)] END $name rc=$rc ===" | tee -a "$PROV"
}

# --- Level 3 (headline): epochs MUST be pinned ---------------------------------
# Without --epoch_a/--epoch_b, load_payloads pools every saved checkpoint.
# Condition A trains from scratch and contributes near-initialisation
# checkpoints that condition B (a fine-tune, converged at epoch 1) does not, so
# the gap becomes a statement about training maturity. That confound produced
# the original "9/9 cells, large effect" result; pinned, it is 6/9 and small.
if want level3; then
  run level3_gate_suppression \
    python -m src.analysis.gate_suppression \
      --root_a $ATTENTION/hidden_states \
      --root_b $ATTENTION_PROXY/hidden_states \
      --label_a attention_only --label_b attention_proxy \
      --epoch_a $EP_ATTN_A --epoch_b $EP_ATTN_B \
      --split $SPLIT \
      --output_dir analysis_results/gate_suppression_mtmf
fi

# --- Level 2: population activity, two pairs -----------------------------------
if want level2; then
  run level2_baseline_vs_proxy \
    python -m src.analysis.neural_efficiency \
      --root_a $BASELINE/hidden_states \
      --root_b $BASELINE_PROXY/hidden_states \
      --label_a baseline --label_b baseline_proxy_finetuned \
      --epoch_a $EP_BASELINE_A --epoch_b $EP_BASELINE_B \
      --split $SPLIT \
      --output_dir analysis_results/neural_efficiency_baseline_vs_proxy_mtmf

  run level2_attention_vs_attention_proxy \
    python -m src.analysis.neural_efficiency \
      --root_a $ATTENTION/hidden_states \
      --root_b $ATTENTION_PROXY/hidden_states \
      --label_a attention_only --label_b attention_proxy \
      --epoch_a $EP_ATTN_A --epoch_b $EP_ATTN_B \
      --split $SPLIT \
      --output_dir analysis_results/neural_efficiency_attention_vs_attention_proxy_mtmf
fi

# --- Level 1: representational content -----------------------------------------
# --best_epoch: compare trained checkpoints, not whole training trajectories.
# --split:      one split, not both pooled (class counts differ between them).
# --task:       identity is the task-RELEVANT feature on identity trials, so an
#               irrelevant-feature claim must exclude that context. Run the two
#               task contexts where identity is genuinely irrelevant.
if want level1; then
  for TASK in location category; do
    run "level1_compare_task_${TASK}" \
      python -m src.analysis.compare_models \
        --baseline $BASELINE/hidden_states \
        --attention $ATTENTION_L1/hidden_states \
        --property identity \
        --task "$TASK" \
        --best_epoch \
        --split $SPLIT \
        --output_dir "analysis_results/compare_baseline_vs_attention_mtmf_task_${TASK}"
  done
fi

echo "ALL DONE $(date -u)" | tee -a "$PROV"
echo
echo "Logs: $LOGDIR"
echo "Before quoting any number, check in each JSON:"
echo "  - epoch_a / epoch_b are NOT null, and epochs_pooled is false"
echo "  - split is '$SPLIT'"
echo "  - cv_squared present alongside fano_factor_analogue"
echo "  - swap_test reports decoded_property, not a bare 'property'"
