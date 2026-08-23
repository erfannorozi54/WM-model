# Neural-efficiency result artifacts

These JSONs are the evidence behind the neural-efficiency chapter. They are
tracked in git (via an explicit negation in `.gitignore`) despite the repo-wide
`*.json` ignore, because they are small and a written chapter depends on them.

Interpretation, confidence levels, and known confounds: **`docs/NEURAL_EFFICIENCY.md`**.
Do not read numbers straight out of these files into a slide — several carry
confounds that the chapter document explains.

## `2026-07-27_run1/` — the run behind the current slides

Produced on `hamrah-gpu-internal`. Recovered into the repo on 2026-08-22; they
had been left in a Claude session temp directory and existed on one machine only.

| File | Level | Status |
|---|---|---|
| `neural_efficiency_baseline_vs_proxy_mtmf.json` | 2 | Usable; 10pp accuracy gap, and its `location` cells have a near-rank-1 condition A (PR ≈ 1.2) that inflates ratios — prefer the attention pair |
| `neural_efficiency_attention_pair.json` | 2 | **Best evidence in the chapter**; 0.08pp accuracy gap, epochs 43 vs 1 |
| `gate_suppression_mtmf.json` | 3 | **Confounded** — `epoch_a`/`epoch_b` are `null`, so ~45 checkpoints per condition were pooled |
| `compare_baseline_vs_attention_mtmf.json` | 1 | **Confounded** — no epoch, split, or task filtering; its swap-test block decodes `location`, not `identity` |

All four predate the estimator fixes, so none contains `cv_squared`,
`n_groups_used_*`, `mean_group_size_*`, or `trial_count_warning`. The `ddof=0`
variance error they carry is worth ~3% at the real group sizes (9–13 trials) and
changes no conclusion.

## Adding a new run

```bash
./run_neural_efficiency.sh <tag>     # writes analysis_results/neural_efficiency/<date>_<tag>/
```

Then update the status table above and `docs/NEURAL_EFFICIENCY.md` §4. Keep old
runs rather than overwriting them — the chapter cites specific numbers, and
being able to point at the file that produced them is the whole reason these are
tracked.
