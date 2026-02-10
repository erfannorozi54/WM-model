# Plotting Code Modifications

## Summary

Modified `scripts/plot_experiments.py` to compute and plot **mean values across multiple runs** of the same experiment configuration, making results more scientifically robust and citable.

## Changes Made

### 1. Multiple Run Detection
- The script now groups experiment runs by their base name (removing timestamp suffixes)
- Example: `wm_stsf_20260101_120000` and `wm_stsf_20260102_140000` are grouped as `wm_stsf`

### 2. Mean Computation
- For each metric at each epoch, computes the **mean** across all runs
- Also computes **standard deviation** when multiple runs exist

### 3. Visualization Improvements
- Plots show mean values as lines
- **Shaded regions** (±1 std) indicate variability across runs
- Legend shows number of runs: `STSF (Baseline) (n=5)`

### 4. Output Information
- Prints the number of runs found for each experiment type
- Reports that plots use mean values
- Explains shaded regions represent ±1 standard deviation

## Usage

```bash
# Generate plots from experiments directory
python scripts/plot_experiments.py

# Specify custom directories
python scripts/plot_experiments.py --exp_dir experiments --output_dir plots

# Plot specific metrics only
python scripts/plot_experiments.py --metrics train_acc val_novel_angle_acc
```

## Expected Directory Structure

```
experiments/
├── wm_stsf_20260101_120000/
│   └── training_log.json
├── wm_stsf_20260102_140000/
│   └── training_log.json
├── wm_stsf_20260103_160000/
│   └── training_log.json
├── wm_mtmf_20260101_120000/
│   └── training_log.json
└── wm_mtmf_20260102_140000/
    └── training_log.json
```

The script will:
- Group `wm_stsf_*` runs together (3 runs)
- Group `wm_mtmf_*` runs together (2 runs)
- Compute mean and std for each group
- Generate plots in `./plots/` directory

## Benefits

1. **Eliminates accidental patterns** - Random fluctuations in single runs are averaged out
2. **Shows reliability** - Standard deviation indicates consistency across runs
3. **Scientifically citable** - Results based on multiple replications are more robust
4. **Transparent reporting** - Number of runs clearly shown in legend and output

## Example Output

```
Found 2 experiment types:
  wm_stsf: 5 run(s)
  wm_mtmf: 3 run(s)

Plotting 8 metrics (using mean across runs)
  Saved train_acc.png
  Saved train_loss.png
  Saved val_novel_angle_acc.png
  Saved val_novel_identity_acc.png
  ...

Plots saved to plots/

Note: Plots show mean values across multiple runs.
      Shaded regions indicate ±1 standard deviation.
```

## Technical Details

- **Grouping logic**: Removes last 2 underscore-separated components (timestamp)
- **Mean calculation**: `np.mean()` across all runs at each epoch
- **Std calculation**: `np.std()` when n_runs > 1, otherwise 0
- **Shading**: `plt.fill_between()` with alpha=0.2 for visibility
