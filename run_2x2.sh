#!/usr/bin/env bash
# The 2x2: the five paper analyses, run identically on all four models, in each
# scenario.
#
#                  from scratch        + proxy pretraining
#   baseline       wm_<sc>             finetune_proxy_wm_<sc>
#   + attention    wm_attention_<sc>   finetune_proxy_wm_attention_<sc>
#
# Everything that must match across the four cells -- validation split, decoded
# property, epoch policy, decoder partition rule -- is pinned in
# configs/analysis/2x2.yaml, not here, and recorded in each output so it can be
# checked afterwards rather than assumed. Epochs are resolved from each model's
# own training_log.json, so this survives retraining.
#
# Usage (from the repo root):
#   ./run_2x2.sh                       # ceiling design, all three scenarios
#   ./run_2x2.sh matched               # 4-way accuracy-matched
#   ./run_2x2.sh both
#   ./run_2x2.sh ceiling mtmf          # one scenario
#   ./run_2x2.sh ceiling --dry_run
#   ./run_2x2.sh ceiling --report_only # re-aggregate without re-running
#
# Long runs: nohup ./run_2x2.sh both > 2x2.log 2>&1 & disown

source "$(dirname "$0")/run_common.sh"

DESIGNS=()
case "${1:-ceiling}" in
  ceiling) DESIGNS=(ceiling); shift ;;
  matched) DESIGNS=(matched); shift ;;
  both)    DESIGNS=(ceiling matched); shift ;;
  -*)      DESIGNS=(ceiling) ;;
  *)       echo "Unknown design '${1}'. Use: ceiling | matched | both" >&2; exit 1 ;;
esac

# Remaining bare words are scenario names; anything with a dash goes to Python.
SCENARIO_ARGS=()
PASSTHRU=()
for arg in "$@"; do
  case "$arg" in
    stsf|stmf|mtmf) SCENARIO_ARGS+=(--scenario "$arg") ;;
    *)              PASSTHRU+=("$arg") ;;
  esac
done

start_provenance "2x2"
{
  echo "designs:   ${DESIGNS[*]}"
  echo "scenarios: ${SCENARIO_ARGS[*]:-<all>}"
  echo "config:    configs/analysis/2x2.yaml"
} | tee -a "$PROV"

for design in "${DESIGNS[@]}"; do
  run_step "2x2_${design}" \
    python -m src.analysis.run_2x2 --design "$design" \
      "${SCENARIO_ARGS[@]}" "${PASSTHRU[@]}"
done

finish_provenance
echo
echo "Comparison tables:"
for design in "${DESIGNS[@]}"; do
  echo "  analysis_results/2x2/${design}/index.md"
done
echo
echo "Read the comparability audit at the top of each table BEFORE quoting any"
echo "number from it. It reports the accuracy spread across the four cells; a"
echo "large spread means differences are confounded with accuracy."
