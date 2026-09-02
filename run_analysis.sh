#!/usr/bin/env bash
# Run the paper's 5 analyses over finished experiments.
#
# Replaces run_all_analysis.sh + run_h128_analysis.sh. It keeps the better half
# of each: the auto-discovery of the newest timestamped directory per prefix
# (from the h128 script) and the ability to sweep whatever is on disk (from the
# other one).
#
# Usage (from the repo root):
#   ./run_analysis.sh                        # every wm_* experiment on disk
#   ./run_analysis.sh h256                   # wm_* but not wm_h128_*
#   ./run_analysis.sh h128                   # wm_h128_* only
#   ./run_analysis.sh proxy                  # finetune_proxy_* only
#   ./run_analysis.sh wm_mtmf_20260520_140601 [more...]   # named dirs
#
# Per-experiment epoch pinning (matched-accuracy comparisons, see
# docs/RESULTS.md) is passed through with EPOCHS, and the validation split with
# SPLIT:
#   EPOCHS=1 OUT_SUFFIX=_ep1 ./run_analysis.sh finetune_proxy_wm_mtmf_20260705_164908
#   SPLIT=val_novel_identity ./run_analysis.sh h256
#
# To compare several models against each other, prefer ./run_2x2.sh: it
# holds epoch, split and property identical across models and emits a
# comparability audit. This script analyses each experiment on its own terms.
#
# Long runs: nohup ./run_analysis.sh > analysis.log 2>&1 & disown

source "$(dirname "$0")/run_common.sh"

PROPERTY="${PROPERTY:-identity}"
ANALYSIS="${ANALYSIS:-all}"
EPOCHS="${EPOCHS:-}"
# One validation split, or empty to pool both. Pooling mixes two generalization
# regimes (novel identities are unseen in one, seen in the other), so any
# cross-model comparison should set this.
SPLIT="${SPLIT:-}"
# Suffix the output directory so a pinned-epoch run does not overwrite the
# best-epoch run of the same experiment.
OUT_SUFFIX="${OUT_SUFFIX:-}"

collect_by_glob () {
  find "${REPO_ROOT}/experiments" -maxdepth 1 -type d -name "$1" -printf '%P\n' 2>/dev/null | sort
}

TARGETS=()
case "${1:-all}" in
  all)   mapfile -t TARGETS < <(collect_by_glob 'wm_*'; collect_by_glob 'finetune_proxy_*') ;;
  h256)  mapfile -t TARGETS < <(collect_by_glob 'wm_*' | grep -v '^wm_h128_') ;;
  h128)  mapfile -t TARGETS < <(collect_by_glob 'wm_h128_*') ;;
  proxy) mapfile -t TARGETS < <(collect_by_glob 'finetune_proxy_*') ;;
  *)     TARGETS=("$@") ;;
esac

if [[ ${#TARGETS[@]} -eq 0 ]]; then
  echo "No matching experiments under ${REPO_ROOT}/experiments" >&2
  exit 1
fi

start_provenance "analysis"
{
  echo "analysis:   $ANALYSIS"
  echo "property:   $PROPERTY"
  echo "epochs:     ${EPOCHS:-<best epoch, auto>}"
  echo "split:      ${SPLIT:-<all splits pooled>}"
  echo "targets:    ${#TARGETS[@]}"
} | tee -a "$PROV"

for exp in "${TARGETS[@]}"; do
  exp_dir="${REPO_ROOT}/experiments/${exp}"

  # A bare prefix is allowed; resolve it to the newest timestamped directory.
  if [[ ! -d "$exp_dir" ]]; then
    resolved="$(latest_exp "$exp")"
    if [[ -z "$resolved" ]]; then
      echo "SKIP $exp: no such experiment directory" | tee -a "$PROV"
      continue
    fi
    exp_dir="$resolved"
    exp="$(basename "$exp_dir")"
  fi

  if [[ ! -d "${exp_dir}/hidden_states" ]]; then
    echo "SKIP $exp: no hidden_states/ (was save_hidden set?)" | tee -a "$PROV"
    continue
  fi
  if [[ ! -f "${exp_dir}/best_model.pt" ]]; then
    echo "SKIP $exp: no best_model.pt (analysis 5 needs it)" | tee -a "$PROV"
    continue
  fi

  cmd=(python -m src.analysis.comprehensive_analysis
       --analysis "$ANALYSIS"
       --hidden_root "${exp_dir}/hidden_states"
       --output_dir "${REPO_ROOT}/analysis_results/${exp}${OUT_SUFFIX}"
       --property "$PROPERTY"
       --model "${exp_dir}/best_model.pt")
  [[ -n "$EPOCHS" ]] && cmd+=(--epochs $EPOCHS)
  [[ -n "$SPLIT" ]] && cmd+=(--split "$SPLIT")

  run_step "${exp}${OUT_SUFFIX}" "${cmd[@]}"
done

finish_provenance
echo "Results in: analysis_results/"
