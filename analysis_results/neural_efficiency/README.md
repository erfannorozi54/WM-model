# Neural-efficiency result artifacts

These JSONs are the evidence behind the neural-efficiency chapter. They are
tracked in git (via an explicit negation in `.gitignore`) despite the repo-wide
`*.json` ignore, because they are small and a written chapter depends on them.

Interpretation, confidence levels, and known limits: **`docs/NEURAL_EFFICIENCY.md`**.

## `2026-08-22_audit-fixed/` — current, and the basis of the slides

Produced on `hamrah-gpu-internal`. The Level 2 and Level 3 files were generated
2026-08-22 14:37–14:42 UTC, three minutes after the audit fixes landed in
`17499de` (14:34:21 UTC). The two Level 1 files were regenerated 2026-08-23
07:13 UTC under `820a8f8` to pick up the swap-test labelling fix; their numbers
are unchanged by that commit, only the reported field names.

| File | Level | Provenance |
|---|---|---|
| `neural_efficiency_attention_pair.json` | 2 | **Best evidence in the chapter** — epochs 43/1, 0.08pp accuracy gap |
| `neural_efficiency_baseline_pair.json` | 2 | Epochs 12/1, 10pp gap; its `location` cells have a near-rank-1 condition A (PR ≈ 1.2) that inflates ratios — prefer the attention pair |
| `gate_suppression.json` | 3 | Epochs 43/1, `epochs_pooled: false` — the run that **overturned** the earlier 9/9 headline |
| `comparison_task_location.json` | 1 | Best epoch, single split, `task=location` — identity decodability collapses to ~chance |
| `comparison_task_category.json` | 1 | Same, `task=category` — no suppression |

All five carry the post-fix fields: `cv_squared`, `n_groups_used_*`,
`mean_group_size_*`, `trial_count_warning`, `epochs_pooled`, `ci_degenerate`,
and (Level 1) `decoded_property` / `interpretation_warning`.

### What was removed

The `2026-07-27_run1/` artifacts were deleted. They were produced by pre-fix
code, existed only in a scratch directory on one machine, and are not merely
older — they are wrong in a way that changed conclusions:

- Level 3 pooled ~45 checkpoints per condition and reported 9/9 cells with a
  large effect. With epochs pinned it is 6/9 and small.
- Level 1 pooled epochs, both validation splits and all three task contexts,
  reporting a uniform "roughly halved" that averaged near-total suppression in
  the location context together with none in the category context.
- Level 2 used `ddof=0` and had no `cv_squared`. Its direction held.

## Adding a new run

```bash
./run_neural_efficiency.sh              # all levels
./run_neural_efficiency.sh level1       # one level
```

Results land in `analysis_results/<per-run dirs>`, logs and provenance in
`logs/neural_efficiency_rerun_<timestamp>/`. Copy the JSONs into a new dated
directory here, update the table above and `docs/NEURAL_EFFICIENCY.md` §4, and
keep old runs rather than overwriting — the chapter cites specific numbers, and
being able to point at the file that produced them is why these are tracked.
