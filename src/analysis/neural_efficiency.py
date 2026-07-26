#!/usr/bin/env python3
"""
Neural-efficiency analyses on RNN hidden states saved by train_with_generalization.py
/ finetune_from_proxy.py.

Tests whether a "familiar/structured" condition (e.g. proxy-pretrained-then-fine-tuned)
shows a suppressed / more efficient population code relative to a "baseline" condition,
at matched behavioral accuracy. See docs/FUTURE_WORK_NEURAL_EFFICIENCY.md secs 2-4 for
the literature and full rationale (Poppenk et al. 2016; Constantinidis & Klingberg 2016).

Four metrics, computed per (task_index, n) cell:
  1. activation_magnitude    - per-trial L2 norm of the hidden vector
  2. participation_ratio     - PCA-based effective dimensionality
  3. population_sparsity     - fraction of near-zero units (per-unit, relative to its own max)
  4. fano_factor_analogue    - trial-to-trial variability of unit activity, grouped by
                                matched (timestep, task-relevant-property-value) condition

Design notes (deviations from a naive reading of the metric names):
- Hidden states from GRU/LSTM/RNN cells are SIGNED (tanh-bounded), not non-negative firing
  rates. Sparsity and the Fano-factor analogue are therefore computed on |h|, not raw h -
  Var(h)/Mean(h) on a signed quantity is unstable/ill-defined when Mean(h) crosses zero.
  This is a real deviation from the textbook Fano factor (defined for spike counts), and is
  flagged again in the JSON output ("caveat" field) so it isn't silently over-interpreted.
- All four metrics are pure functions of a (trials, H) matrix - I/O and filtering are kept
  separate so the metrics can be unit-tested or reused independently of the payload schema.
- This module is NOT wired into comprehensive_analysis.py's --analysis {1..5} dispatch:
  that CLI is single-hidden_root, single-condition by design (one experiment at a time).
  This analysis is inherently a two-condition comparison (baseline vs. proxy, or
  attention-only vs. attention+proxy), so it follows the existing two-argument
  comparison-tool pattern already used by compare_models.py (--baseline/--attention)
  rather than being force-fit into the single-root pipeline.

Usage example:

python -m src.analysis.neural_efficiency \\
  --root_a experiments/wm_mtmf_20260520_140601/hidden_states \\
  --root_b experiments/finetune_proxy_wm_mtmf_20260705_164908/hidden_states \\
  --label_a baseline --label_b proxy \\
  --epoch_a 45 --epoch_b 12 \\
  --split val_novel_identity \\
  --output_dir analysis_results/neural_efficiency_mtmf
"""

from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any, Callable
import argparse
import json
import numpy as np
import torch

from .activations import load_payloads, _filter_records, iterate_records, TASK_INDEX_TO_NAME

PROPERTY_CHOICES = ["location", "identity", "category"]
TASK_CHOICES = ["location", "identity", "category", "any"]

FANO_CAVEAT = (
    "Computed on |hidden_state| (rectified), not the literal spike-count Fano factor: "
    "GRU/LSTM/RNN hidden states are signed (tanh-bounded), so Var/Mean on the raw signed "
    "value is unstable whenever the mean crosses zero. Treat as a functional analogue only."
)
MAGNITUDE_CAVEAT = (
    "RNN hidden-state L2 norm is not literally a BOLD or spike-rate signal; treat as a "
    "functional analogy (facilitated processing -> suppressed representational magnitude), "
    "not a claim of biophysical equivalence."
)


def _task_name_to_index(name: Optional[str]) -> Optional[int]:
    if name is None or name == "any":
        return None
    for k, v in TASK_INDEX_TO_NAME.items():
        if v == name:
            return k
    raise ValueError(f"Unknown task name: {name}")


# ---------------------------------------------------------------------------
# Data extraction
# ---------------------------------------------------------------------------

def records_for_cell(
    payloads: List[Dict[str, Any]],
    task_index: Optional[int] = None,
    n_value: Optional[int] = None,
    time: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """All (sample, timestep) records for a task/n/time cell, no property required.

    Unlike build_matrix_with_metadata (which requires a decodable property at a
    single fixed timestep), this pulls every matching record regardless of
    property labels - what the efficiency metrics need is raw activity, not
    decodability.
    """
    return _filter_records(
        iterate_records(payloads), time=time, task_index=task_index,
        n_value=n_value, property_name=None,
    )


def hidden_matrix_for_cell(
    payloads: List[Dict[str, Any]],
    task_index: Optional[int] = None,
    n_value: Optional[int] = None,
    time: Optional[int] = None,
) -> np.ndarray:
    """(N, H) matrix of hidden vectors pooled across all matching records."""
    recs = records_for_cell(payloads, task_index=task_index, n_value=n_value, time=time)
    if not recs:
        return np.empty((0, 0))
    return np.stack([r["hidden"] for r in recs], axis=0)


# ---------------------------------------------------------------------------
# Metric 1: activation magnitude
# ---------------------------------------------------------------------------

def activation_magnitude(X: np.ndarray) -> np.ndarray:
    """Per-trial L2 norm, shape (N,). Caller takes mean/CI as needed."""
    if X.size == 0:
        return np.empty((0,))
    return np.linalg.norm(X, axis=1)


# ---------------------------------------------------------------------------
# Metric 2: participation ratio (effective dimensionality)
# ---------------------------------------------------------------------------

def participation_ratio(X: np.ndarray) -> float:
    """PR = (sum(lambda_i))^2 / sum(lambda_i^2), lambda = eigenvalues of the
    trial covariance matrix. PR in [1, min(N, H)]; low PR = activity confined
    to few effective dimensions. Requires N >= 2 trials.
    """
    if X.shape[0] < 2:
        return float("nan")
    Xc = X - X.mean(axis=0, keepdims=True)
    # eigenvalues of the (N,N) Gram matrix == nonzero eigenvalues of the (H,H)
    # covariance matrix, and is cheaper when N < H (the common case here).
    gram = Xc @ Xc.T
    eigvals = np.linalg.eigvalsh(gram)
    eigvals = np.clip(eigvals, 0, None)
    s1 = eigvals.sum()
    s2 = (eigvals ** 2).sum()
    if s2 <= 1e-12:
        return float("nan")
    return float((s1 ** 2) / s2)


# ---------------------------------------------------------------------------
# Metric 3: population sparsity
# ---------------------------------------------------------------------------

def population_sparsity(X: np.ndarray, rel_threshold: float = 0.05) -> float:
    """Fraction of (trial, unit) entries with |activation| below rel_threshold
    times that unit's own max |activation| across trials. Uses |X|, since a
    raw-signed "near zero" comparison would be meaningless for tanh-bounded
    hidden states (values straddle zero by design, not because they're inactive).
    """
    if X.size == 0:
        return float("nan")
    abs_x = np.abs(X)
    unit_max = abs_x.max(axis=0, keepdims=True)
    unit_max = np.where(unit_max < 1e-12, 1.0, unit_max)  # avoid 0/0 -> "sparse" for dead units
    near_zero = abs_x < (rel_threshold * unit_max)
    return float(near_zero.mean())


def population_sparsity_gini(X: np.ndarray) -> float:
    """Gini coefficient of |activation| pooled across all (trial, unit) entries,
    as an alternative to the threshold-based sparsity above (no free parameter)."""
    if X.size == 0:
        return float("nan")
    v = np.sort(np.abs(X).ravel())
    n = v.shape[0]
    if v.sum() <= 1e-12:
        return float("nan")
    cum = np.cumsum(v)
    return float((n + 1 - 2 * (cum.sum() / cum[-1])) / n)


# ---------------------------------------------------------------------------
# Metric 4: Fano-factor analogue
# ---------------------------------------------------------------------------

def fano_factor_analogue(X_groups: List[np.ndarray], min_group_size: int = 3) -> float:
    """Mean, over unit and over condition-groups, of Var(|h|)/Mean(|h|) across
    trials *within* each group of matched-condition trials.

    X_groups: list of (n_i, H) matrices, each holding trials that share the
    same (timestep, task-relevant-property-value) - the closest available
    analogue to "repeated presentations of the same condition" for a dataset
    with no literal stimulus repeats. Groups smaller than min_group_size are
    skipped (variance estimates from <3 trials are not meaningful).
    """
    ratios = []
    for X in X_groups:
        if X.shape[0] < min_group_size:
            continue
        abs_x = np.abs(X)
        mean = abs_x.mean(axis=0)
        var = abs_x.var(axis=0)
        valid = mean > 1e-6
        if not np.any(valid):
            continue
        ratios.append(var[valid] / mean[valid])
    if not ratios:
        return float("nan")
    return float(np.concatenate(ratios).mean())


def fano_groups_for_cell(
    payloads: List[Dict[str, Any]],
    task_index: int,
    n_value: Optional[int] = None,
) -> List[np.ndarray]:
    """Group hidden vectors by (time, task-relevant property value) within a
    task/n cell - the "matched condition" grouping fano_factor_analogue needs.
    """
    property_name = TASK_INDEX_TO_NAME[task_index]
    recs = records_for_cell(payloads, task_index=task_index, n_value=n_value, time=None)
    groups: Dict[Tuple[int, Any], List[np.ndarray]] = {}
    for r in recs:
        val = r.get(property_name)
        if val is None:
            continue
        key = (r["time"], val)
        groups.setdefault(key, []).append(r["hidden"])
    return [np.stack(v, axis=0) for v in groups.values() if len(v) >= 1]


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

def bootstrap_ci(
    X: np.ndarray,
    metric_fn: Callable[[np.ndarray], float],
    n_boot: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Percentile bootstrap over the trial axis (rows of X)."""
    point = metric_fn(X)
    if X.shape[0] < 2:
        return point, float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    vals = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        vals[i] = metric_fn(X[idx])
    lo = np.nanpercentile(vals, (1 - ci) / 2 * 100)
    hi = np.nanpercentile(vals, (1 + ci) / 2 * 100)
    return float(point), float(lo), float(hi)


def bootstrap_difference(
    X_a: np.ndarray,
    X_b: np.ndarray,
    metric_fn: Callable[[np.ndarray], float],
    n_boot: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> Dict[str, float]:
    """Independent percentile bootstrap of metric_fn(X_a) - metric_fn(X_b).

    Returns point difference, CI, and a two-sided empirical p-value (fraction
    of bootstrap differences that cross zero, doubled and clipped to 1).
    """
    point_a, point_b = metric_fn(X_a), metric_fn(X_b)
    point_diff = point_a - point_b
    if X_a.shape[0] < 2 or X_b.shape[0] < 2:
        return {"diff": point_diff, "ci_lo": float("nan"), "ci_hi": float("nan"), "p_value": float("nan")}
    rng = np.random.default_rng(seed)
    na, nb = X_a.shape[0], X_b.shape[0]
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx_a = rng.integers(0, na, size=na)
        idx_b = rng.integers(0, nb, size=nb)
        diffs[i] = metric_fn(X_a[idx_a]) - metric_fn(X_b[idx_b])
    lo = np.nanpercentile(diffs, (1 - ci) / 2 * 100)
    hi = np.nanpercentile(diffs, (1 + ci) / 2 * 100)
    prop_below_zero = float(np.nanmean(diffs < 0))
    p_value = 2 * min(prop_below_zero, 1 - prop_below_zero)
    return {"diff": float(point_diff), "ci_lo": float(lo), "ci_hi": float(hi), "p_value": float(min(p_value, 1.0))}


# ---------------------------------------------------------------------------
# Matched-accuracy epoch selection (sec 4.2 of the future-work doc)
# ---------------------------------------------------------------------------

def select_matched_epoch(
    log_a: List[Dict[str, Any]],
    log_b: List[Dict[str, Any]],
    metric_key: str = "val_novel_angle_acc",
) -> Dict[str, Any]:
    """Find the (epoch_a, epoch_b) pair whose metric_key values are closest.

    log_a / log_b: parsed training_log.json contents (list of per-epoch dicts).
    Reports the residual accuracy gap explicitly rather than hiding it - per
    sec 4.2, a perfectly matched pair is not guaranteed to exist, and picking
    the closest available pair while reporting the gap is preferred over
    silently comparing best-epoch vs. best-epoch (which conflates the
    "familiarity" effect with a "just more accurate" confound).
    """
    best = None
    for ea in log_a:
        if metric_key not in ea:
            continue
        for eb in log_b:
            if metric_key not in eb:
                continue
            gap = abs(ea[metric_key] - eb[metric_key])
            if best is None or gap < best["accuracy_gap"]:
                best = {
                    "epoch_a": ea["epoch"], "epoch_b": eb["epoch"],
                    "acc_a": ea[metric_key], "acc_b": eb[metric_key],
                    "accuracy_gap": gap, "metric_key": metric_key,
                }
    if best is None:
        raise RuntimeError(f"No epochs with '{metric_key}' found in one or both logs")
    return best


# ---------------------------------------------------------------------------
# Per-cell comparison and orchestration
# ---------------------------------------------------------------------------

def compare_cell(
    payloads_a: List[Dict[str, Any]],
    payloads_b: List[Dict[str, Any]],
    task_index: int,
    n_value: Optional[int],
    n_boot: int = 1000,
    seed: int = 42,
) -> Dict[str, Any]:
    X_a = hidden_matrix_for_cell(payloads_a, task_index=task_index, n_value=n_value)
    X_b = hidden_matrix_for_cell(payloads_b, task_index=task_index, n_value=n_value)

    result: Dict[str, Any] = {
        "task": TASK_INDEX_TO_NAME[task_index], "n": n_value,
        "n_trials_a": int(X_a.shape[0]) if X_a.size else 0,
        "n_trials_b": int(X_b.shape[0]) if X_b.size else 0,
    }
    if X_a.size == 0 or X_b.size == 0:
        result["skipped"] = "no trials in one or both conditions for this cell"
        return result

    mag_fn = lambda X: float(activation_magnitude(X).mean())
    result["activation_magnitude"] = {
        "a": bootstrap_ci(X_a, mag_fn, n_boot=n_boot, seed=seed),
        "b": bootstrap_ci(X_b, mag_fn, n_boot=n_boot, seed=seed),
        "difference": bootstrap_difference(X_a, X_b, mag_fn, n_boot=n_boot, seed=seed),
    }
    result["participation_ratio"] = {
        "a": bootstrap_ci(X_a, participation_ratio, n_boot=n_boot, seed=seed),
        "b": bootstrap_ci(X_b, participation_ratio, n_boot=n_boot, seed=seed),
        "difference": bootstrap_difference(X_a, X_b, participation_ratio, n_boot=n_boot, seed=seed),
    }
    result["population_sparsity"] = {
        "a": bootstrap_ci(X_a, population_sparsity, n_boot=n_boot, seed=seed),
        "b": bootstrap_ci(X_b, population_sparsity, n_boot=n_boot, seed=seed),
        "difference": bootstrap_difference(X_a, X_b, population_sparsity, n_boot=n_boot, seed=seed),
    }

    groups_a = fano_groups_for_cell(payloads_a, task_index=task_index, n_value=n_value)
    groups_b = fano_groups_for_cell(payloads_b, task_index=task_index, n_value=n_value)
    result["fano_factor_analogue"] = {
        "a": fano_factor_analogue(groups_a),
        "b": fano_factor_analogue(groups_b),
        "n_groups_a": len(groups_a),
        "n_groups_b": len(groups_b),
    }

    return result


def magnitude_over_time(
    payloads: List[Dict[str, Any]],
    task_index: Optional[int],
    n_value: Optional[int],
    max_time: int = 6,
    n_boot: int = 500,
    seed: int = 42,
) -> List[Dict[str, float]]:
    """Per-timestep mean activation magnitude with bootstrap CI, for plotting."""
    mag_fn = lambda X: float(activation_magnitude(X).mean())
    curve = []
    for t in range(max_time):
        X_t = hidden_matrix_for_cell(payloads, task_index=task_index, n_value=n_value, time=t)
        if X_t.size == 0:
            continue
        point, lo, hi = bootstrap_ci(X_t, mag_fn, n_boot=n_boot, seed=seed)
        curve.append({"time": t, "mean": point, "ci_lo": lo, "ci_hi": hi, "n": int(X_t.shape[0])})
    return curve


def run_neural_efficiency_analysis(
    hidden_root_a: Path,
    hidden_root_b: Path,
    label_a: str,
    label_b: str,
    epoch_a: Optional[int] = None,
    epoch_b: Optional[int] = None,
    split: Optional[str] = "val_novel_identity",
    task_indices: Tuple[int, ...] = (0, 1, 2),
    n_values: Tuple[Optional[int], ...] = (1, 2, 3),
    n_boot: int = 1000,
    seed: int = 42,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    payloads_a = load_payloads(Path(hidden_root_a), epochs=[epoch_a] if epoch_a is not None else None, split=split)
    payloads_b = load_payloads(Path(hidden_root_b), epochs=[epoch_b] if epoch_b is not None else None, split=split)

    if not payloads_a:
        raise RuntimeError(f"No payloads found under {hidden_root_a} (epoch={epoch_a}, split={split})")
    if not payloads_b:
        raise RuntimeError(f"No payloads found under {hidden_root_b} (epoch={epoch_b}, split={split})")

    cells = []
    for ti in task_indices:
        for nv in n_values:
            cells.append(compare_cell(payloads_a, payloads_b, task_index=ti, n_value=nv, n_boot=n_boot, seed=seed))

    curves = {
        "a": magnitude_over_time(payloads_a, task_index=None, n_value=None, n_boot=max(200, n_boot // 2), seed=seed),
        "b": magnitude_over_time(payloads_b, task_index=None, n_value=None, n_boot=max(200, n_boot // 2), seed=seed),
    }

    result = {
        "label_a": label_a, "label_b": label_b,
        "hidden_root_a": str(hidden_root_a), "hidden_root_b": str(hidden_root_b),
        "epoch_a": epoch_a, "epoch_b": epoch_b, "split": split,
        "caveats": {"activation_magnitude": MAGNITUDE_CAVEAT, "fano_factor_analogue": FANO_CAVEAT},
        "cells": cells,
        "magnitude_over_time": curves,
    }

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "neural_efficiency.json", "w") as f:
            json.dump(result, f, indent=2)
        _plot_magnitude_over_time(curves, label_a, label_b, output_dir)
        _plot_participation_ratio(cells, label_a, label_b, output_dir)

    return result


def _plot_magnitude_over_time(curves: Dict[str, List[Dict[str, float]]], label_a: str, label_b: str, output_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    for key, label in [("a", label_a), ("b", label_b)]:
        curve = curves[key]
        if not curve:
            continue
        t = [c["time"] for c in curve]
        mean = [c["mean"] for c in curve]
        lo = [c["ci_lo"] for c in curve]
        hi = [c["ci_hi"] for c in curve]
        ax.plot(t, mean, marker="o", label=label)
        ax.fill_between(t, lo, hi, alpha=0.2)
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Mean activation magnitude ||h_t||_2")
    ax.set_title("Hidden-state activation magnitude over time")
    ax.legend()
    plt.savefig(output_dir / "neural_efficiency_magnitude_over_time.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_participation_ratio(cells: List[Dict[str, Any]], label_a: str, label_b: str, output_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    valid_cells = [c for c in cells if "participation_ratio" in c]
    if not valid_cells:
        return
    labels = [f"{c['task']}/n{c['n']}" for c in valid_cells]
    pr_a = [c["participation_ratio"]["a"][0] for c in valid_cells]
    pr_b = [c["participation_ratio"]["b"][0] for c in valid_cells]

    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.9), 5))
    ax.bar(x - width / 2, pr_a, width, label=label_a)
    ax.bar(x + width / 2, pr_b, width, label=label_b)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Participation ratio")
    ax.set_title("Effective dimensionality per (task, n) cell")
    ax.legend()
    plt.savefig(output_dir / "neural_efficiency_participation_ratio.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Neural-efficiency comparison between two hidden-state conditions")
    p.add_argument("--root_a", type=str, required=True, help="hidden_states dir for condition A (e.g. baseline)")
    p.add_argument("--root_b", type=str, required=True, help="hidden_states dir for condition B (e.g. proxy-pretrained)")
    p.add_argument("--label_a", type=str, default="a")
    p.add_argument("--label_b", type=str, default="b")
    p.add_argument("--epoch_a", type=int, default=None)
    p.add_argument("--epoch_b", type=int, default=None)
    p.add_argument("--split", type=str, default="val_novel_identity",
                   choices=["val_novel_angle", "val_novel_identity"])
    p.add_argument("--n_values", type=int, nargs="*", default=[1, 2, 3])
    p.add_argument("--n_boot", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--training_log_a", type=str, default=None,
                   help="If given with --training_log_b, auto-select the closest-accuracy epoch pair instead of --epoch_a/--epoch_b")
    p.add_argument("--training_log_b", type=str, default=None)
    p.add_argument("--match_metric", type=str, default="val_novel_angle_acc")
    args = p.parse_args()

    epoch_a, epoch_b = args.epoch_a, args.epoch_b
    if args.training_log_a and args.training_log_b:
        with open(args.training_log_a) as f:
            log_a = json.load(f)
        with open(args.training_log_b) as f:
            log_b = json.load(f)
        match = select_matched_epoch(log_a, log_b, metric_key=args.match_metric)
        epoch_a, epoch_b = match["epoch_a"], match["epoch_b"]
        print(f"Matched-accuracy epoch pair: {args.label_a}@epoch{epoch_a} "
              f"(acc={match['acc_a']:.4f}) vs {args.label_b}@epoch{epoch_b} "
              f"(acc={match['acc_b']:.4f}), gap={match['accuracy_gap']:.4f}")

    result = run_neural_efficiency_analysis(
        hidden_root_a=Path(args.root_a),
        hidden_root_b=Path(args.root_b),
        label_a=args.label_a, label_b=args.label_b,
        epoch_a=epoch_a, epoch_b=epoch_b,
        split=args.split,
        n_values=tuple(args.n_values),
        n_boot=args.n_boot, seed=args.seed,
        output_dir=Path(args.output_dir) if args.output_dir else None,
    )

    if args.output_dir is None:
        print(json.dumps(result, indent=2))
    else:
        print(f"Saved results to {args.output_dir}/neural_efficiency.json")


if __name__ == "__main__":
    main()
