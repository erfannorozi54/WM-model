#!/usr/bin/env bash
# Two-stage proxy pipeline: pre-train on feature recall, then fine-tune on N-back.
#
# This entry point did not exist. Proxy pre-training is the project's strongest
# result -- +14.8pp novel angle, +12.2pp novel identity, and the whole Level 2
# neural-efficiency comparison rests on its checkpoints -- yet the only recorded
# way to run it was two hand-typed commands, where the second needs the
# timestamped directory the first creates. That gap is why the proxy runs were
# hard to reproduce; the directory is now captured automatically.
#
# Usage (from the repo root):
#   ./run_proxy_pipeline.sh                       # baseline_mtmf
#   ./run_proxy_pipeline.sh all                   # all 6 (2 arms x 3 scenarios)
#   ./run_proxy_pipeline.sh attention_stsf
#   ./run_proxy_pipeline.sh baseline_mtmf attention_mtmf     # several, in order
#
# Arms are <variant>_<scenario>: variant is baseline|attention, scenario is
# stsf|stmf|mtmf. Dual attention was dropped from the thesis on 2026-09-02.
#
# Skip stage 1 and fine-tune from an existing pre-trained directory:
#   PROXY_EXP_DIR=experiments/proxy_mtmf_20260705_120000 ./run_proxy_pipeline.sh
#
# Long runs: nohup ./run_proxy_pipeline.sh attention > proxy.log 2>&1 & disown

source "$(dirname "$0")/run_common.sh"

ALL_ARMS=(baseline_stsf baseline_stmf baseline_mtmf
          attention_stsf attention_stmf attention_mtmf)

if [[ "${1:-}" == "all" ]]; then
  VARIANTS=("${ALL_ARMS[@]}")
else
  VARIANTS=("${@:-baseline_mtmf}")
fi

start_provenance "proxy_pipeline"
echo "variants: ${VARIANTS[*]}" | tee -a "$PROV"

for variant in "${VARIANTS[@]}"; do
  # Accept the old bare-variant spellings as MTMF, which is what they meant.
  case "$variant" in
    baseline)  variant=baseline_mtmf ;;
    attention) variant=attention_mtmf ;;
  esac

  arm="${variant%_*}"        # baseline | attention
  scenario="${variant##*_}"  # stsf | stmf | mtmf

  case "$scenario" in
    stsf|stmf|mtmf) ;;
    *) echo "Unknown scenario '$scenario' in '$variant' (stsf|stmf|mtmf)" >&2
       continue ;;
  esac

  case "$arm" in
    baseline)  proxy_cfg="configs/proxy/proxy_${scenario}.yaml"
               proxy_name="proxy_${scenario}"
               finetune_cfg="configs/${scenario}.yaml" ;;
    attention) proxy_cfg="configs/proxy/proxy_attention_${scenario}.yaml"
               proxy_name="proxy_attention_${scenario}"
               finetune_cfg="configs/attention_${scenario}.yaml" ;;
    *) echo "Unknown arm '$arm' in '$variant' (baseline|attention)" >&2
       continue ;;
  esac

  if [[ ! -f "$proxy_cfg" || ! -f "$finetune_cfg" ]]; then
    echo "SKIP ${variant}: missing config ($proxy_cfg or $finetune_cfg)" | tee -a "$PROV"
    continue
  fi

  # --- Stage 1: proxy pre-training ---------------------------------------------
  if [[ -n "${PROXY_EXP_DIR:-}" ]]; then
    proxy_dir="$PROXY_EXP_DIR"
    echo "stage 1 skipped, using PROXY_EXP_DIR=$proxy_dir" | tee -a "$PROV"
  else
    run_step "proxy_pretrain_${variant}" \
      python -m src.train_proxy --config "$proxy_cfg"

    # train_proxy.py names its output <experiment_name>_<UTC timestamp>; it does
    # not print a machine-readable path, so recover the newest matching dir.
    proxy_dir="$(latest_exp "$proxy_name")"
  fi

  if [[ -z "$proxy_dir" || ! -f "${proxy_dir}/best_model.pt" ]]; then
    echo "ABORT ${variant}: stage 1 produced no usable checkpoint" \
         "(looked for experiments/${proxy_name}_*/best_model.pt)" | tee -a "$PROV"
    continue
  fi
  echo "stage 1 -> $(basename "$proxy_dir")" | tee -a "$PROV"

  # --- Stage 2: fine-tune on the real N-back task ------------------------------
  run_step "proxy_finetune_${variant}" \
    python -m src.finetune_from_proxy \
      --proxy_exp_dir "$proxy_dir" \
      --config "$finetune_cfg"

  finetuned="$(latest_exp "finetune_proxy_$(grep -m1 '^experiment_name:' "$finetune_cfg" | awk '{print $2}')")"
  echo "stage 2 -> $(basename "${finetuned:-<none>}")" | tee -a "$PROV"
done

finish_provenance
cat <<'EOF'

Next: run the paper analyses on what you just produced.
  ./run_2x2.sh ceiling            # the 4-model comparison, per scenario
  ./run_analysis.sh proxy         # or one experiment at a time
EOF
