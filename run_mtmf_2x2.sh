#!/usr/bin/env bash
# The MTMF 2x2: the five paper analyses, run identically on all four models.
#
#   baseline            wm_mtmf
#   attention           wm_attention_mtmf
#   baseline+proxy      finetune_proxy_wm_mtmf
#   attention+proxy     finetune_proxy_wm_attention_mtmf
#
# Everything that has to match across the four cells -- validation split,
# decoded property, epoch policy, decoder partition rule -- is pinned in
# configs/analysis/mtmf_2x2.yaml, not here, and is recorded in each output so it
# can be checked afterwards rather than assumed.
#
# Usage (from the repo root):
#   ./run_mtmf_2x2.sh                  # accuracy-matched epochs (default)
#   ./run_mtmf_2x2.sh ceiling          # each model at its own best epoch
#   ./run_mtmf_2x2.sh both
#   ./run_mtmf_2x2.sh matched --dry_run
#   ./run_mtmf_2x2.sh matched --report_only   # re-aggregate without re-running
#
# Long runs: nohup ./run_mtmf_2x2.sh both > 2x2.log 2>&1 & disown

source "$(dirname "$0")/run_common.sh"

DESIGNS=()
EXTRA=()
case "${1:-matched}" in
  matched) DESIGNS=(matched); shift ;;
  ceiling) DESIGNS=(ceiling); shift ;;
  both)    DESIGNS=(matched ceiling); shift ;;
  -*)      DESIGNS=(matched) ;;
  *)       echo "Unknown design '${1}'. Use: matched | ceiling | both" >&2; exit 1 ;;
esac
EXTRA=("$@")

start_provenance "mtmf_2x2"
echo "designs: ${DESIGNS[*]}" | tee -a "$PROV"
echo "config:  configs/analysis/mtmf_2x2.yaml" | tee -a "$PROV"

for design in "${DESIGNS[@]}"; do
  run_step "mtmf_2x2_${design}" \
    python -m src.analysis.mtmf_2x2 --design "$design" "${EXTRA[@]}"
done

finish_provenance
echo
echo "Comparison tables:"
for design in "${DESIGNS[@]}"; do
  echo "  analysis_results/mtmf_2x2/${design}/comparison.md"
done
echo
echo "Read the comparability audit at the top of each table BEFORE quoting any"
echo "number from it. A cell whose provenance block is missing was produced by"
echo "older code and is not verifiably comparable."
