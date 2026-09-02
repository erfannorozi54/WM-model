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
| `neural_efficiency_attention_pair.json` | 2 | **Best evidence in the chapter** — epochs 43/1; 0.84pp accuracy gap on the analysed split (the pair was *selected* on novel-angle, where the gap is 0.08pp) |
| `neural_efficiency_baseline_pair.json` | 2 | Epochs 12/1, 8.9pp gap on the analysed split — **irreducible**: proxy ep1 exceeds every baseline checkpoint's identity accuracy. Its `location` cells also have a near-rank-1 condition A (PR ≈ 1.2) that inflates ratios — prefer the attention pair |
| `gate_suppression.json` | 3 | Epochs 43/1, `epochs_pooled: false` — the run that **overturned** the earlier 9/9 headline |
| `comparison_task_location.json` | 1 | Best epoch, single split, `task=location` — identity decodability collapses to ~chance |
| `comparison_task_category.json` | 1 | Same, `task=category` — no suppression |

All five carry the post-fix fields: `cv_squared`, `n_groups_used_*`,
`mean_group_size_*`, `trial_count_warning`, `epochs_pooled`, `ci_degenerate`,
and (Level 1) `decoded_property` / `interpretation_warning`.

## `2026-08-26_2x2/` — the two architecture contrasts

Produced on `hamrah-gpu-internal` under `dee49ec`, via
`./run_neural_efficiency.sh level2x`. The `2026-08-22` pairs above both vary the
*training regimen* inside one architecture; these two vary the *architecture*
inside one regimen, completing the 2×2 over the same four models.

| File | Pair | Epochs | Accuracy gap (`val_novel_identity`) |
|---|---|---|---|
| `neural_efficiency_scratch_row.json` | baseline vs. attention, both from scratch | 17 / 9 | **0.88pp** — anchored at the baseline's own ceiling (82.53%) |
| `neural_efficiency_proxy_row.json` | baseline+proxy vs. attention+proxy | 20 / 45 | **0.00pp** — the tightest match anywhere in the chapter (both 93.55%) |

Both carry the same post-fix fields as the 2026-08-22 run. Read against the
artifacts on 2026-09-02, the headline is that **attention reshapes the
population code only when it is the only modification**:

| | magnitude lower in B | PR lower in B | sparsity |
|---|---|---|---|
| scratch row (no proxy) | 9/9, CI excl. 0 in 8/9 | 9/9 (8/9) — and 6/6 after dropping the near-rank-1 location cells | mixed, 3/9 |
| proxy row (both proxy) | 7/9 (6/9); identity nulls at p=0.73, p=0.61 | **4/9 — no effect** | *higher* in 8/9 (6/9) |

Attention's effect on the geometry is **absorbed** once proxy pretraining is
present — the population-code counterpart of the accuracy interaction. Written
into `docs/NEURAL_EFFICIENCY.md` §4.5 on 2026-09-02.

**No Fano direction may be read out of these two files.** Fano falls 7/9 in the
scratch row, but `cv_squared` — its scale-invariant companion — is 5/9, a coin
flip. Attention lowers magnitude and Fano is scale-dependent, so that is exactly
the artifact `cv_squared` exists to catch. The Fano/CV² **rise stays attributed
to proxy pretraining** (18/18 in the 2026-08-22 pairs, both metrics). An earlier
note here claimed the attention arm agrees with Constantinidis & Klingberg; it
does not, and the claim is withdrawn.

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
