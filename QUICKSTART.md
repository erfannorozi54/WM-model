# Quick Start Guide

**Complete workflow from setup to analysis in 4 steps.**

## What's New (Phase 6)

✅ **All 5 analyses** from the paper implemented  
✅ **Validation splits**: Novel-angle and novel-identity generalization testing  
✅ **CNN activations**: Saved alongside RNN hidden states for orthogonalization comparison  
✅ **Causal perturbation**: Full implementation tracking all 3 output actions  
✅ **Comprehensive pipeline**: Single command runs everything

---

## Step 1: Setup Environment

```bash
# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Verify setup
python -m src.scripts.verify_analysis_setup
# Expected: 5/5 tests passed ✅
```

---

## Step 2: Generate Data

```bash
# Option A: Quick test with placeholder data (instant)
python -m src.data.download_shapenet --placeholder

# Option B: Real ShapeNet data (25GB download)
# Add your token to .env file first: HUGGINGFACE_TOKEN=your_token
python -m src.data.download_shapenet --download-hf ShapeNetCore.v2.zip

# Generate rendered stimuli (required for both options)
python -m src.data.generate_stimuli
```

**Expected output**: `data/stimuli/` directory with 320 images (5 IDs × 4 categories × 4 locations × 4 angles)

---

## Step 3: Train Model

### For Complete Analysis (Recommended)

```bash
# Train with novel-angle and novel-identity validation
python -m src.train_with_generalization --config configs/mtmf.yaml
```

**Outputs**:

- `experiments/wm_mtmf/best_model.pt` - Best checkpoint
- `experiments/wm_mtmf/training_log.json` - Metrics per epoch
- `experiments/wm_mtmf/hidden_states/` - Saved activations

### For Simple Training

```bash
# Basic training with single validation set
python -m src.train --config configs/mtmf.yaml
```

---

## Step 4: Run Analysis

### Complete Analysis (All 5 Analyses)

```bash
# Run all 5 analyses from the paper
python -m src.analysis.comprehensive_analysis \
  --analysis all \
  --model experiments/wm_mtmf/best_model.pt \
  --hidden_root experiments/wm_mtmf/hidden_states \
  --property identity \
  --output_dir analysis_results
```

**Outputs** in `analysis_results/`:

- `analysis1_generalization_comparison.png` - Figure A1c (novel angle vs identity)
- `analysis2a_task_relevance.png` - Figure 2b (task-relevant encoding)
- `analysis2b_cross_task_*.png` - Figure 2a (cross-task generalization)
- `analysis3_orthogonalization.png` - Figure 3b (CNN vs RNN geometry)
- `analysis4a_cross_time_decoding.png` - Figure 4b (memory dynamics)
- `causal_perturbation_identity.png` - Figure A7 (causal test)
- `*.json` files - All numerical results

### Individual Analyses

```bash
# Run specific analysis only
python -m src.analysis.comprehensive_analysis --analysis 1 \
  --hidden_root experiments/wm_mtmf/hidden_states

python -m src.analysis.comprehensive_analysis --analysis 2 \
  --hidden_root experiments/wm_mtmf/hidden_states

python -m src.analysis.comprehensive_analysis --analysis 5 \
  --model experiments/wm_mtmf/best_model.pt \
  --hidden_root experiments/wm_mtmf/hidden_states \
  --property location
```

---

## Summary: Complete Workflow

```bash
# 1. Setup
source venv/bin/activate
python -m src.scripts.verify_analysis_setup

# 2. Data
python -m src.data.download_shapenet --placeholder
python -m src.data.generate_stimuli

# 3. Train
python -m src.train_with_generalization --config configs/mtmf.yaml

# 4. Analyze (all 5 analyses)
python -m src.analysis.comprehensive_analysis \
  --analysis all \
  --model experiments/wm_mtmf/best_model.pt \
  --hidden_root experiments/wm_mtmf/hidden_states \
  --property identity \
  --output_dir analysis_results

# 5. Review results
ls analysis_results/
# You should see plots and JSON files for all 5 analyses
```

**Total time**: ~2-3 hours (depending on GPU and number of epochs)

---

## What Each Step Does

| Step | What It Does | Time |
|------|--------------|------|
| **1. Setup** | Install packages, verify environment | 5 min |
| **2. Data** | Download objects, render 320 stimuli | 10-30 min |
| **3. Train** | Train model with 2 validation sets | 1-2 hours |
| **4. Analyze** | Generate all figures and metrics | 10-20 min |

---

## Expected Results

After running all 5 analyses, you should see:

✅ **Analysis 1**: Novel identity accuracy < novel angle accuracy (generalization gap)  
✅ **Analysis 2**: Task-relevant features decoded with >85% accuracy, cross-task matrices generated  
✅ **Analysis 3**: O(RNN) < O(CNN) - RNN de-orthogonalizes compared to CNN perceptual space  
✅ **Analysis 4**: Cross-time decoding accuracy drops over time (H1 disproved, H2 supported)  
✅ **Analysis 5**: P(Match) drops and P(No-Action) rises with perturbation (causal subspaces confirmed)

---

## Troubleshooting

**No stimuli found?**

```bash
python -m src.data.generate_stimuli
```

**Analysis fails?**

```bash
python -m src.scripts.verify_analysis_setup
pip install seaborn  # If missing
```

**Need more details?**

- See `docs/ANALYSIS_METHODOLOGY.md` - **Complete technical documentation (updated)**
- See `COMPREHENSIVE_ANALYSIS_READY.md` - Detailed analysis guide
- See `ANALYSIS_CHECKLIST.md` - Task-by-task checklist
- See `README.md` - Full project reference

---

## Different Training Scenarios

```bash
# Single-task single-feature
python -m src.train_with_generalization --config configs/stsf.yaml

# Single-task multi-feature
python -m src.train_with_generalization --config configs/stmf.yaml

# Multi-task multi-feature (recommended)
python -m src.train_with_generalization --config configs/mtmf.yaml
```

---

**That's it!** You now have a complete working memory model with comprehensive analysis. 🎉
