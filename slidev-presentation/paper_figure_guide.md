# Figure Screenshot Guide: Exact Neural Mass Model

This deck intentionally uses only a small set of high-value figures from the paper. Save screenshots into:

```text
slidev-presentation/public/paper_figures/
```

The presentation imports the files directly from `public/paper_figures/`, so a file saved as:

```text
slidev-presentation/public/paper_figures/fig1_validation.png
```

is referenced in the presentation as:

```text
./public/paper_figures/fig1_validation.png
```

## Selected Figures

| Presentation file | Paper figure | Why include it |
|---|---|---|
| `fig1_validation.png` | Fig. 1 | Shows neural mass model agreement with large QIF network simulations. |
| `fig3_wm_modes.png` | Fig. 3 | Shows selective reactivation, spontaneous reactivation, and persistent activity in one result. |
| `fig4_heuristic_comparison.png` | Fig. 4 | Shows what the heuristic firing-rate model misses, especially beta-gamma transients. |
| `fig5_6_competition_juggling.png` | Fig. 5 or Fig. 6 | Shows anti-phase juggling or the stimulus-parameter map for two-item competition. |
| `fig8_9_multi_item_capacity.png` | Fig. 8 or Fig. 9 | Shows splay-state multi-item loading or capacity failure at high load. |
| `fig11_frequency_bands.png` | Fig. 11 | Shows theta, beta, gamma, and alpha power vs memory load. |
| `fig12_voltage_load.png` | Fig. 12 | Shows the ERP-like membrane-potential load signal. |
| `fig13_bifurcation.png` | Fig. 13 | Explains how background current controls the three memory regimes. |

## Recommended Workflow

1. Open `Papers/Exact neural mass model.pdf`.
2. Navigate to the target figure.
3. Use your screenshot tool to crop only the useful panels.
4. Save the screenshot with the exact filename listed above.
5. Refresh the Slidev deck at `http://localhost:3031/`.

Keep crops readable. For dense multi-panel figures, crop only the panels used in the explanation rather than the full figure.

## Likely PDF Pages

These page numbers are approximate because figure captions and references can spill across pages:

| Figure | Likely page area |
|---|---|
| Fig. 1 | PDF pages 6-7 |
| Fig. 3 | PDF pages 9-13 |
| Fig. 4 | PDF pages 12-13 or appendix page 36 |
| Fig. 5 / Fig. 6 | PDF pages 14-16 |
| Fig. 8 / Fig. 9 | PDF pages 18-21 |
| Fig. 11 | PDF page 23 |
| Fig. 12 | PDF pages 24-25 |
| Fig. 13 | PDF pages 33-35 |

## Command-Line Rendering Option

If you prefer to render PDF pages first and crop from images:

```bash
cd /home/erfan/Projects/WM-model
mkdir -p /tmp/exact_neural_mass_pages
pdftoppm -png -r 220 "Papers/Exact neural mass model.pdf" /tmp/exact_neural_mass_pages/page
```

This creates files like:

```text
/tmp/exact_neural_mass_pages/page-006.png
/tmp/exact_neural_mass_pages/page-007.png
```

Then open the rendered page image in an image editor, crop the target figure, and save it into:

```text
slidev-presentation/public/paper_figures/
```

## Suggested Crop Choices

- `fig1_validation.png`: include firing rate plus `x(t)` and `u(t)` panels if readable.
- `fig3_wm_modes.png`: include the three columns or crop one representative row across all three modes.
- `fig4_heuristic_comparison.png`: prioritize the spectrogram or time-series panel that makes beta-gamma absence visible.
- `fig5_6_competition_juggling.png`: use Fig. 5 for intuitive anti-phase bursts, or Fig. 6 for the outcome map.
- `fig8_9_multi_item_capacity.png`: use Fig. 8 for splay-state timing, or Fig. 9 for capacity limits and dropout.
- `fig11_frequency_bands.png`: include all four bands if possible; otherwise gamma and beta are most important.
- `fig12_voltage_load.png`: crop the curve showing increase, saturation, and decrease of `Delta v`.
- `fig13_bifurcation.png`: include the branches and marked background-current regimes.
