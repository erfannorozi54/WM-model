# Comprehensive Analysis: Code Methodology Audit

**Date**: 2026-06-15
**Scope**: All 18 experiments (9 original + 9 h128) audited against paper methodology.
**Status**: Code fixed, h128 experiments re-run with new code, original experiments re-running in background.

## Files Modified

| File | Changes |
|------|---------|
| `src/analysis/comprehensive_analysis.py` | H2 fix, sample-size warnings, SVC max_iter=10000 |
| `src/analysis/causal_perturbation.py` | Reverted to mean direction (was per-class — wrong direction) |
| `src/analysis/orthogonalization.py` | LinearSVC max_iter=10000, random_state=42 for determinism |
| `src/analysis/decoding.py` | SVC max_iter=10000 |
| `src/analysis/procrustes.py` | Full swap test, location for label alignment |

## Key Methodology Issues Found & Fixed

### 1. H2 Cross-Stimulus Test Was Testing Cross-Time (CRITICAL BUG)

**Before**: `_test_h2_cross_stimulus` was training decoder on t=0, testing on t=1,2,3.
This is the H1 test (slot-based memory), mislabeled as H2. All 18 experiments
showed "H3 POSSIBLE" because validation (t=0) >> generalization (t>0).

**After**: Uses val_novel_angle (known identities) for training/validation, val_novel_identity
(novel identities) for generalization, both at t=0. This is the proper cross-stimulus test.

**Finding**: With the corrected test, all 9 h128 experiments show `val >> gen`:
```
wm_h128_stsf:               val=1.000  gen=0.000  (H3_POSSIBLE)
wm_h128_stmf:               val=0.500  gen=0.228  (H3_POSSIBLE)
wm_h128_mtmf:               val=0.475  gen=0.237  (H3_POSSIBLE)
wm_h128_attention_stsf:     val=0.900  gen=0.290  (H3_POSSIBLE)
wm_h128_attention_stmf:     val=0.850  gen=0.087  (H3_POSSIBLE)
wm_h128_attention_mtmf:     val=0.562  gen=0.268  (H3_POSSIBLE)
wm_h128_dual_attention_stsf: val=0.887  gen=0.135  (H3_POSSIBLE)
wm_h128_dual_attention_stmf: val=0.775  gen=0.370  (H3_POSSIBLE)
wm_h128_dual_attention_mtmf: val=0.738  gen=0.260  (H3_POSSIBLE)
```

**Interpretation**: Our models show H3 (stimulus-specific encoding) rather than the
paper's H2 (shared encoding). This is a meaningful difference — our hidden states
for novel identities are very different from known identities, even at the same
location. The paper's models may generalize better because of different training
or model architecture.

### 2. Procrustes Swap Test (FIXED)

**Before**: Only computed consecutive disparities, no proper swap test.
**After**: Implements correct/swap1/swap2 with per-stimulus group split.

**Results (9 h128 experiments)**:
```
Experiment                       correct  swap1   swap2   Status
wm_h128_stsf                        0.994   0.241   0.223   NOT_CONFIRMED (high correct, both swaps low)
wm_h128_stmf                        0.294   0.235   0.256   CONFIRMED (paper pattern)
wm_h128_mtmf                        0.264   0.262   0.253   NOT_CONFIRMED
wm_h128_attention_stsf              0.263   0.253   0.261   CONFIRMED
wm_h128_attention_stmf              0.277   0.207   0.320   CONFIRMED (paper pattern)
wm_h128_attention_mtmf              0.319   0.303   0.343   NOT_CONFIRMED
wm_h128_dual_attention_stsf         0.098   0.257   0.181   CONFIRMED
wm_h128_dual_attention_stmf         0.345   0.248   0.239   NOT_CONFIRMED
wm_h128_dual_attention_mtmf         0.333   0.215   0.279   CONFIRMED
```

5/9 show paper pattern (swap2 > swap1), 4/9 don't. STSF model shows extreme pattern:
correct=0.99 (rotations work for same stimulus) but both swaps fail (rotations don't
transfer to new stimuli). This is consistent with H3 for STSF.

### 3. Causal Perturbation — Reverted Per-Class Fix

**Initial fix attempt**: Changed from mean direction to per-trial class-specific direction.
**Problem**: Per-class direction points TOWARD the class. Pushing a match trial in this
direction moves it DEEPER into the match class, INCREASING P(Match) — opposite of paper.
**Reverted to**: Mean direction (original approach).

**Results (9 h128 with mean direction)**:
```
Experiment                        P(Match)  drop%   P(No-Action)
wm_h128_stsf                       1.000→0.995   0.5%   0.000→0.000
wm_h128_stmf                       0.951→0.795  16.5%   0.000→0.000
wm_h128_mtmf                       0.858→0.618  27.9%   0.000→0.000
wm_h128_attention_stsf             0.495→0.512  -3.5%   0.000→0.000
wm_h128_attention_stmf             0.960→0.898   6.5%   0.000→0.000
wm_h128_attention_mtmf             0.943→0.939   0.4%   0.000→0.000
wm_h128_dual_attention_stsf        0.785→0.709   9.7%   0.000→0.000
wm_h128_dual_attention_stmf        0.964→0.884   8.3%   0.000→0.000
wm_h128_dual_attention_mtmf        0.921→0.828  10.1%   0.000→0.000
```

**Largest effects**: MTMF models show 10-28% P(Match) drops. STSF models show
minimal effects (model is too confident).

**Remaining limitation**: P(No-Action) does NOT rise. The paper expects 0.10→0.61
but we see 0.000→0.000. This is because the perturbation only runs through the
classifier, not the recurrent dynamics. To match the paper would require storing
the input sequence (cog_in) in the payload and re-running the recurrent module —
this is a training-time change that wasn't done in this audit.

### 4. SVC Convergence Warnings (FIXED)
All SVC/LinearSVC calls now use `max_iter=10000`. The 4-warning clusters seen in
old logs are gone in new logs.

### 5. Sample-Size Warnings (FIXED)
When `n_test < n_classes` or `n_test < 2 × n_classes`, the analysis prints a warning
informing the user that test accuracy is unreliable.

## Code Changes Summary

### `comprehensive_analysis.py` (lines 320, 458, 582, 583, 1024-...)
- `SVC(..., max_iter=10000)` in 3 places
- `train_test_split` with fallback for non-stratifiable data
- Sample-size warnings in `_analyze_task_relevance`
- New H2 cross-stimulus test using `val_novel_angle`/`val_novel_identity` splits
- Calls new `swap_hypothesis_test` in `_test_h2_procrustes_swap`

### `procrustes.py` (new function `swap_hypothesis_test`)
- Splits by `identity` hash for cross-stimulus effect
- Decodes on `location` (4 fixed classes) for label alignment between groups
- Reports correct/swap1/swap2/baseline accuracies
- Tests hypothesis: `|swap2 - correct| < |swap1 - correct|`

### `causal_perturbation.py`
- Reverted to mean direction (was per-trial class-specific — wrong direction)
- Kept `property_values` and `label2idx` parameters for API compatibility
- Added documentation explaining why mean direction is correct

### `orthogonalization.py` and `decoding.py`
- `LinearSVC(..., max_iter=10000, random_state=42)` for reproducibility
- `SVC(..., max_iter=10000, random_state=42)` where applicable

## Inspection Locations (all on local machine)

| What | Where |
|------|-------|
| Analysis outputs (JSON) | `/home/erfan/Projects/WM-model/analysis_results/<exp>/*.json` |
| Analysis plots (PNG) | `/home/erfan/Projects/WM-model/analysis_results/<exp>/*.png` |
| Analysis log (recent) | `/tmp/analysis_rerun.log` |
| Source code | `/home/erfan/Projects/WM-model/src/analysis/*.py` |
| This document | `/home/erfan/Projects/WM-model/docs/ANALYSIS_AUDIT_FINDINGS.md` |

## Reproducing the Audit

```bash
# On GPU server
ssh hamrah-gpu-internal
cd ~/Projects/WM-model
export PYTHONPATH=src:$PYTHONPATH

# Re-run a single experiment
~/.venv/WM-model/bin/python -m src.analysis.comprehensive_analysis \
  --analysis all \
  --hidden_root experiments/wm_h128_stsf_20260602_230425/hidden_states \
  --output_dir /tmp/test_out \
  --property identity \
  --model experiments/wm_h128_stsf_20260602_230425/best_model.pt
```

## Open Questions / Future Work

1. **Recurrent dynamics in causal perturbation**: Need to store `cog_in` in the
   payload during training to re-run the recurrent module. This would let
   P(No-Action) rise as the paper expects.

2. **H2 vs H3 discrepancy**: Our models show H3 (stimulus-specific) rather than
   the paper's H2 (shared encoding). This could be due to:
   - Different model architecture details
   - Different training regime
   - Different stimulus set
   - Different cross-stimulus test methodology in the paper

3. **Sample sizes**: With `num_val=200` and identity property having 70 classes,
   test sets are smaller than 2× n_classes. The new warnings flag this. To
   improve reliability, increase `num_val` in configs.

4. **Original experiments re-run in background**: Original 9 experiments
   (wm_stsf through wm_dual_attention_mtmf) are re-running with the new code
   for direct comparison with the h128 experiments.
