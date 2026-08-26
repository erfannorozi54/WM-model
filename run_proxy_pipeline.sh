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
#   ./run_proxy_pipeline.sh                 # baseline variant
#   ./run_proxy_pipeline.sh attention
#   ./run_proxy_pipeline.sh dual_attention
#   ./run_proxy_pipeline.sh baseline attention      # several, in order
#
# Skip stage 1 and fine-tune from an existing pre-trained directory:
#   PROXY_EXP_DIR=experiments/proxy_mtmf_20260705_120000 ./run_proxy_pipeline.sh
#
# Long runs: nohup ./run_proxy_pipeline.sh attention > proxy.log 2>&1 & disown

source "$(dirname "$0")/run_common.sh"

VARIANTS=("${@:-baseline}")

start_provenance "proxy_pipeline"
echo "variants: ${VARIANTS[*]}" | tee -a "$PROV"

for variant in "${VARIANTS[@]}"; do
  case "$variant" in
    baseline)       proxy_cfg=configs/proxy/proxy_mtmf.yaml
                    proxy_name=proxy_mtmf
                    finetune_cfg=configs/mtmf.yaml ;;
    attention)      proxy_cfg=configs/proxy/proxy_attention_mtmf.yaml
                    proxy_name=proxy_attention_mtmf
                    finetune_cfg=configs/attention_mtmf.yaml ;;
    dual_attention) proxy_cfg=configs/proxy/proxy_dual_attention_mtmf.yaml
                    proxy_name=proxy_dual_attention_mtmf
                    finetune_cfg=configs/dual_attention_mtmf.yaml ;;
    *) echo "Unknown variant '$variant' (baseline|attention|dual_attention)" >&2
       continue ;;
  esac

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

Next: run the paper analyses on what you just produced, then the efficiency levels.
  ./run_analysis.sh proxy
  ./run_neural_efficiency.sh
EOF
