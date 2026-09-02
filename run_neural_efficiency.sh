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
#   ./run_neural_efficiency.sh              # all levels
#   ./run_neural_efficiency.sh level1       # one level only
#   ./run_neural_efficiency.sh level2 level3
#   ./run_neural_efficiency.sh level2x      # the architecture contrasts (2x2)
#
# Writes results into analysis_results/<per-run dirs> and logs into
# logs/neural_efficiency_rerun_<timestamp>/.

source "$(dirname "$0")/run_common.sh"

LEVELS="${*:-level1 level2 level2x level3}"
want () { [[ " $LEVELS " == *" $1 "* ]]; }

start_provenance "neural_efficiency"
echo "levels:  $LEVELS" | tee -a "$PROV"

SPLIT=val_novel_identity
EXP=experiments
BASELINE=$EXP/wm_mtmf_20260520_140601
BASELINE_PROXY=$EXP/finetune_proxy_wm_mtmf_20260705_164908
ATTENTION_L1=$EXP/wm_attention_mtmf_20260520_203605
ATTENTION=$EXP/wm_attention_mtmf_20260726_161735
ATTENTION_PROXY=$EXP/finetune_proxy_wm_attention_mtmf_20260726_201707

# Epoch pairs, resolved once and pinned here rather than re-derived per run.
# Auto-selection via --training_log_a/--training_log_b picks whichever pair is
# closest at the time and would silently change these results.
#
# SPLIT CAVEAT (measured from training_log.json, 2026-08-26): these pairs were
# selected by closest val_novel_ANGLE accuracy -- 82.69 vs 92.67 (10.0pp) and
# 93.43 vs 93.51 (0.08pp). The analyses below run --split val_novel_identity,
# where the same checkpoints sit 8.9pp (81.17 vs 90.10) and 0.84pp (91.75 vs
# 92.59) apart. The baseline gap is irreducible on the analysed split: proxy
# epoch 1 already exceeds every baseline checkpoint (identity ceiling 82.53% at
# ep17), so no accuracy-matched baseline pair exists. A strict identity-matched
# attention pair would be 18 vs 8 (0.44pp); re-pinning would invalidate the
# published artifacts and is a deliberate re-run decision, not taken here.
# Quote 8.9pp / 0.84pp, never 10pp / 0.08pp, as the accuracy match.
EP_BASELINE_A=12; EP_BASELINE_B=1
EP_ATTN_A=43;     EP_ATTN_B=1

# The two columns of the 2x2 (level2x below). The pairs above vary the training
# regimen within one architecture; these vary the architecture within one
# regimen, which is what tells attention and proxy pretraining apart rather than
# just measuring each against its own control.
#
# Both are anchored at the *higher-accuracy* model's ceiling on val_novel_identity
# and matched to the nearest checkpoint of the other, so neither is read near
# initialisation:
#   scratch row: baseline@ep17 = 82.53% (its ceiling)  vs attention@ep9  = 81.65%  -> 0.88pp
#   proxy   row: baseline_proxy@ep20 = 93.55% (its ceiling) vs attention_proxy@ep45 = 93.55% -> 0.00pp
EP_SCRATCH_BASE=17; EP_SCRATCH_ATTN=9
EP_PROXY_BASE=20;   EP_PROXY_ATTN=45

# --- Level 3 (headline): epochs MUST be pinned ---------------------------------
# Without --epoch_a/--epoch_b, load_payloads pools every saved checkpoint.
# Condition A trains from scratch and contributes near-initialisation
# checkpoints that condition B (a fine-tune, converged at epoch 1) does not, so
# the gap becomes a statement about training maturity. That confound produced
# the original "9/9 cells, large effect" result; pinned, it is 6/9 and small.
if want level3; then
  run_step level3_gate_suppression \
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
  run_step level2_baseline_vs_proxy \
    python -m src.analysis.neural_efficiency \
      --root_a $BASELINE/hidden_states \
      --root_b $BASELINE_PROXY/hidden_states \
      --label_a baseline --label_b baseline_proxy_finetuned \
      --epoch_a $EP_BASELINE_A --epoch_b $EP_BASELINE_B \
      --split $SPLIT \
      --output_dir analysis_results/neural_efficiency_baseline_vs_proxy_mtmf

  run_step level2_attention_vs_attention_proxy \
    python -m src.analysis.neural_efficiency \
      --root_a $ATTENTION/hidden_states \
      --root_b $ATTENTION_PROXY/hidden_states \
      --label_a attention_only --label_b attention_proxy \
      --epoch_a $EP_ATTN_A --epoch_b $EP_ATTN_B \
      --split $SPLIT \
      --output_dir analysis_results/neural_efficiency_attention_vs_attention_proxy_mtmf
fi

# --- Level 2x: the other two cells of the 2x2 ----------------------------------
# The level2 pairs above answer "what does proxy pretraining do?" twice. These
# two answer "what does attention do?" twice, over the same four models, and so
# separate the two modifications instead of confirming each against its own
# control.
#
# Result (2026-08-26, re-read against the artifacts 2026-09-02). Attention
# reshapes the population code only when it is the ONLY modification:
#   scratch row (no proxy):  magnitude lower 9/9 (CI excludes 0 in 8/9),
#                            participation ratio lower 9/9 (8/9) -- and still
#                            6/6 after dropping the three location cells, which
#                            are near-rank-1 (PR 1.1-1.4) and inflate ratios.
#   proxy row (both proxy):  magnitude lower 7/9 (6/9; the two identity nulls
#                            sit at p=0.73 and p=0.61), PR 4/9 -- no effect --
#                            and sparsity moves the other way, HIGHER in 8/9.
# So attention's effect on the geometry is ABSORBED once proxy pretraining is
# present, which is the population-code counterpart of the accuracy interaction.
#
# Do NOT read a Fano direction out of these two files. Fano falls 7/9 in the
# scratch row, but cv_squared -- its scale-invariant companion, and the only
# reason cv_squared exists -- is 5/9, a coin flip. Attention lowers magnitude,
# Fano is scale-dependent, so that is exactly the artifact cv_squared was added
# to catch. The Fano/CV2 rise stays attributed to PROXY PRETRAINING (18/18 in
# the level2 pairs, both metrics), and these files do not license the claim that
# the attention arm agrees with Constantinidis & Klingberg.
if want level2x; then
  run_step level2x_scratch_baseline_vs_attention \
    python -m src.analysis.neural_efficiency \
      --root_a $BASELINE/hidden_states \
      --root_b $ATTENTION/hidden_states \
      --label_a baseline_scratch --label_b attention_scratch \
      --epoch_a $EP_SCRATCH_BASE --epoch_b $EP_SCRATCH_ATTN \
      --split $SPLIT \
      --output_dir analysis_results/neural_efficiency_2x2/scratch_baseline_vs_attention

  run_step level2x_proxy_baseline_vs_attention \
    python -m src.analysis.neural_efficiency \
      --root_a $BASELINE_PROXY/hidden_states \
      --root_b $ATTENTION_PROXY/hidden_states \
      --label_a baseline_proxy --label_b attention_proxy \
      --epoch_a $EP_PROXY_BASE --epoch_b $EP_PROXY_ATTN \
      --split $SPLIT \
      --output_dir analysis_results/neural_efficiency_2x2/proxy_baseline_vs_attention
fi

# --- Level 1: representational content -----------------------------------------
# --best_epoch: compare trained checkpoints, not whole training trajectories.
# --split:      one split, not both pooled (class counts differ between them).
# --task:       identity is the task-RELEVANT feature on identity trials, so an
#               irrelevant-feature claim must exclude that context. Run the two
#               task contexts where identity is genuinely irrelevant.
if want level1; then
  for TASK in location category; do
    run_step "level1_compare_task_${TASK}" \
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

finish_provenance
echo "Before quoting any number, check in each JSON:"
echo "  - epoch_a / epoch_b are NOT null, and epochs_pooled is false"
echo "  - split is '$SPLIT'"
echo "  - cv_squared present alongside fano_factor_analogue"
echo "  - swap_test reports decoded_property, not a bare 'property'"
echo "  - any accuracy-gap quote names the analysed split (see SPLIT CAVEAT above)"
