# Visualization Example

## What You'll See During Training

Every epoch, a visualization image is saved showing:

---

### Layout Structure

```
┌────────────────────────────────────────────────────────────────────┐
│             N-Back Sequence | Task: Location | N=2                 │
├────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐    │
│   │      │  │      │  │      │  │      │  │      │  │      │    │
│   │ IMG  │  │ IMG  │  │ IMG  │  │ IMG  │  │ IMG  │  │ IMG  │    │
│   │  0   │  │  1   │  │  2   │  │  3   │  │  4   │  │  5   │    │
│   │      │  │      │  │      │  │      │  │      │  │      │    │
│   └──────┘  └──────┘  └──────┘  └──────┘  └──────┘  └──────┘    │
│     t=0       t=1       t=2       t=3       t=4       t=5         │
│   ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐    │
│   │Loc: 0│  │Loc: 2│  │Loc: 1│  │Loc: 0│  │Loc: 3│  │Loc: 2│    │
│   │Cat:  │  │Cat:  │  │Cat:  │  │Cat:  │  │Cat:  │  │Cat:  │    │
│   │airplane│ │car   │  │chair │  │airplane│ │lamp  │  │car   │    │
│   │ID:   │  │ID:   │  │ID:   │  │ID:   │  │ID:   │  │ID:   │    │
│   │air_01│  │car_02│  │cha_01│  │air_01│  │lam_03│  │car_02│    │
│   └──────┘  └──────┘  └──────┘  └──────┘  └──────┘  └──────┘    │
│   ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐    │
│   │Target│  │Target│  │Target│  │Target│  │Target│  │Target│    │
│   │ No   │  │ No   │  │ No   │  │Match │  │ No   │  │NonM. │    │
│   │Action│  │Action│  │Action│  │      │  │Action│  │      │    │
│   │      │  │      │  │      │  │      │  │      │  │      │    │
│   │ Pred │  │ Pred │  │ Pred │  │ Pred │  │ Pred │  │ Pred │    │
│   │ No   │  │ No   │  │ No   │  │Match │  │NonM. │  │NonM. │    │
│   │Action│  │Action│  │Action│  │      │  │      │  │      │    │
│   └──────┘  └──────┘  └──────┘  └──────┘  └──────┘  └──────┘    │
│    GREEN     GREEN     GREEN     GREEN      RED      GREEN       │
│   (correct) (correct) (correct) (correct) (wrong)  (correct)     │
│                                                                      │
├────────────────────────────────────────────────────────────────────┤
│                   Sequence Accuracy: 83.3%                          │
│                   Legend: Green=Correct | Red=Incorrect            │
└────────────────────────────────────────────────────────────────────┘
```

---

## Information Displayed

### 1. Header (Top)
- **Task name**: Location, Identity, or Category
- **N-back value**: 1, 2, or 3
- Clear title showing what task the model is performing

### 2. Image Row (Top)
- **6 stimulus images** from the sequence
- **Timestep labels** (t=0 through t=5)
- **Color borders**:
  - 🟢 **Green** = Model predicted correctly
  - 🔴 **Red** = Model predicted incorrectly

### 3. Metadata Row (Middle)
For each timestep:
- **Location** (0-3): Spatial position in 2×2 grid
- **Category**: airplane, car, chair, or lamp
- **Identity**: Specific object ID (e.g., airplane_001)

### 4. Prediction Row (Bottom)
For each timestep:
- **Target**: Expected correct answer
  - No Action (first N trials)
  - Match (current stimulus matches N-back)
  - Non-Match (current stimulus doesn't match N-back)
- **Pred**: Model's prediction
- **Color coding**:
  - 🟢 Green background = Correct prediction
  - 🔴 Red background = Incorrect prediction

### 5. Footer (Bottom)
- **Overall accuracy** for this sequence
- **Legend** for color coding

---

## Example Scenarios

### Scenario 1: Perfect Performance
```
All 6 images have GREEN borders
Sequence Accuracy: 100%
All Target/Pred pairs match
```

### Scenario 2: Learning in Progress
```
First 4 images: GREEN borders (correct)
Last 2 images: RED borders (incorrect)
Sequence Accuracy: 66.7%
Model struggling with later timesteps
```

### Scenario 3: Systematic Errors
```
All "No Action" trials: GREEN (correct)
All "Match" trials: RED (incorrect)
Sequence Accuracy: 50%
Model not learning the match detection
```

---

## How to Use These Visualizations

### During Training
1. **Monitor after each epoch**: Check if predictions improve
2. **Spot patterns**: See which timesteps are hardest
3. **Verify learning**: Ensure model isn't just guessing "No Action"

### After Training
1. **Compare epochs**: See progression from epoch 1 to 10
2. **Error analysis**: Identify systematic mistakes
3. **Model comparison**: Compare STSF vs MTMF performance

### For Presentations
1. **Publication figures**: High-quality 150 DPI images
2. **Demonstrations**: Show what N-back task looks like
3. **Results**: Visual proof of learning

---

## File Locations

```bash
# After training STSF
experiments/wm_stsf/visualizations/
├── epoch_001_sample.png
├── epoch_002_sample.png
├── epoch_003_sample.png
├── ...
└── epoch_010_sample.png

# After training MTMF
experiments/wm_mtmf/visualizations/
├── epoch_001_sample.png
├── epoch_002_sample.png
├── ...
└── epoch_015_sample.png
```

---

## Expected Evolution Across Epochs

### Epoch 1 (Untrained Model)
```
Many RED borders
Accuracy: ~40-50% (mostly random guessing)
Predictions inconsistent
```

### Epoch 3-5 (Learning)
```
More GREEN borders appearing
Accuracy: ~60-70%
Starting to learn "No Action" for early timesteps
```

### Epoch 8-10 (Converged)
```
Mostly GREEN borders
Accuracy: ~80-90%
Consistent correct predictions
May still struggle with edge cases
```

---

## Tips for Interpretation

### Good Signs ✅
- Accuracy increasing over epochs
- Green borders becoming more common
- Correct "Match" detection (hardest task)
- Consistent "No Action" for first N timesteps

### Warning Signs ⚠️
- All predictions are "No Action" (not learning)
- Accuracy stuck at ~67% (baseline guessing)
- Random pattern of red/green (no learning)
- Getting worse over epochs (learning rate issue)

---

## Technical Details

**Image Size**: 18" × 8" figure (high resolution)  
**DPI**: 150 (publication quality)  
**Format**: PNG with transparency support  
**Colors**: Green (#90EE90), Red (#FFB6C1)  
**Font**: Sans-serif, monospace for metadata  

---

## Try It Now!

```bash
# Start training
python -m src.train_with_generalization --config configs/stsf.yaml

# After epoch 1, view the visualization
eog experiments/wm_stsf/visualizations/epoch_001_sample.png

# Or use any image viewer
feh experiments/wm_stsf/visualizations/epoch_001_sample.png
firefox experiments/wm_stsf/visualizations/epoch_001_sample.png
```

---

**This gives you real-time visual feedback on exactly what your model is learning!** 🎨📊
