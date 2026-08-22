#!/usr/bin/env python3
"""
Gate-suppression index for attention-enhanced working memory models.

Tests whether FeatureChannelAttention's per-channel gate (src/models/attention.py)
suppresses task-irrelevant channels more than task-relevant ones, and whether
that gap sharpens after proxy pretraining. See docs/FUTURE_WORK_NEURAL_EFFICIENCY.md
sec 4.8 / 4.8.1 for the full rationale and literature this operationalizes.

Design notes (read before trusting the numbers):

- "Relevant" / "irrelevant" channels are ranked using CNN-ACTIVATION-space
  decoders (build_cnn_matrix + orthogonalization.one_vs_rest_weights), not
  RNN hidden-state decoders. The gate is applied to `cnn_activations` (the
  pre-RNN visual embedding, see AttentionWorkingMemoryModel.forward: gates
  are computed on and multiplied into `cnn_features`, and `cnn_activations`
  saved in the payload is a clone of that exact same pre-gating tensor). RNN
  hidden-state channels are a different, GRU/LSTM-transformed space with no
  guaranteed correspondence to CNN-feature channel indices, so ranking
  relevance there would compare the wrong channels against the gate values.

- Relevance ranking is computed INDEPENDENTLY per condition (e.g. once for
  "attention-only", once for "attention+proxy"), each from its own
  cnn_activations. The frozen-ResNet50-backbone assumption does not extend to
  the small trainable 1x1-conv projection ahead of it, which proxy pretraining
  and fine-tuning do update - so channel c is not guaranteed to mean the same
  thing across conditions. Each condition's suppression index is computed
  against its own relevance ranking; only the resulting SCALAR indices are
  compared across conditions, not raw channel identities.

- The primary statistic (gate_suppression_index) uses a hard top/bottom
  --top_frac split by a channel "relevance contrast" score (z-scored
  task-relevant decodability minus z-scored decodability of the other two
  properties). A companion, k-free statistic (gate_relevance_correlation) -
  the correlation between per-channel mean gate value and the same contrast
  score - is also reported so the headline number doesn't hinge on the
  arbitrary top_frac choice.

Usage:

python -m src.analysis.gate_suppression \\
  --root_a experiments/wm_attention_mtmf_<ts>/hidden_states \\
  --root_b experiments/finetune_proxy_attention_mtmf_<ts>/hidden_states \\
  --label_a attention_only --label_b attention_proxy \\
  --split val_novel_identity \\
  --output_dir analysis_results/gate_suppression_mtmf
"""

from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
import argparse
import json
import numpy as np

from .activations import load_payloads, build_cnn_matrix, TASK_INDEX_TO_NAME, PROPERTY_NAMES
from .orthogonalization import one_vs_rest_weights
from .neural_efficiency import bootstrap_ci, bootstrap_difference


def _infer_sequence_length(payloads: List[Dict[str, Any]]) -> int:
    for payload in payloads:
        gates = payload.get("gates")
        if gates is not None:
            return gates.shape[1]
        cnn_act = payload.get("cnn_activations")
        if cnn_act is not None:
            return cnn_act.shape[1]
    raise RuntimeError("Could not infer sequence length: no 'gates' or 'cnn_activations' in any payload")


def channel_relevance_scores(
    payloads: List[Dict[str, Any]],
    property_name: str,
    task_index: Optional[int],
    n_value: Optional[int],
    times: List[int],
) -> Optional[np.ndarray]:
    """Per-channel relevance (H,) for decoding `property_name` from CNN-activation
    space, averaged across `times`. Relevance = mean, over one-vs-rest classes,
    of |normalized decision-hyperplane weight| per channel - channels a linear
    decoder relies on to separate that property's classes. Returns None if no
    timestep yields at least 2 classes with data.
    """
    per_time_scores = []
    for t in times:
        X, y, label2idx = build_cnn_matrix(payloads, property_name, time=t, task_index=task_index, n_value=n_value)
        if X.numel() == 0 or len(label2idx) < 2:
            continue
        W = one_vs_rest_weights(X, y)
        if len(W) < 2:
            continue
        stacked = np.stack([np.abs(w) for w in W.values()], axis=0)  # (n_classes, H)
        per_time_scores.append(stacked.mean(axis=0))  # (H,)
    if not per_time_scores:
        return None
    return np.mean(per_time_scores, axis=0)


def gate_matrix_for_cell(
    payloads: List[Dict[str, Any]],
    task_index: Optional[int],
    n_value: Optional[int],
    times: List[int],
) -> np.ndarray:
    """(N, H) matrix of gate vectors pooled across `times` for a task/n cell."""
    xs = []
    for payload in payloads:
        gates = payload.get("gates")
        if gates is None:
            continue
        B, T, H = gates.shape
        task_indices = payload.get("task_index")
        n_vals = payload.get("n")
        for b in range(B):
            if task_index is not None and task_indices is not None and int(task_indices[b]) != task_index:
                continue
            if n_value is not None and n_vals is not None and int(n_vals[b]) != n_value:
                continue
            for t in times:
                if t < T:
                    xs.append(gates[b, t].numpy())
    if not xs:
        return np.empty((0, 0))
    return np.stack(xs, axis=0)


def _zscore(x: np.ndarray) -> np.ndarray:
    std = x.std()
    return (x - x.mean()) / std if std > 1e-12 else np.zeros_like(x)


def gate_suppression_index(
    payloads: List[Dict[str, Any]],
    task_index: int,
    n_value: Optional[int] = None,
    times: Optional[List[int]] = None,
    top_frac: float = 0.25,
    n_boot: int = 1000,
    seed: int = 42,
) -> Dict[str, Any]:
    """Gate-suppression index for one condition, one (task_index, n) cell.

    suppression_index = mean gate on irrelevant-channel set - mean gate on
    relevant-channel set (expected negative: irrelevant channels gated down
    more than relevant ones, if the attention mechanism is doing what it's
    supposed to).
    """
    if times is None:
        times = list(range(_infer_sequence_length(payloads)))

    relevant_property = TASK_INDEX_TO_NAME[task_index]
    irrelevant_properties = [p for p in sorted(PROPERTY_NAMES) if p != relevant_property]

    relevant_scores = channel_relevance_scores(payloads, relevant_property, task_index, n_value, times)
    irrelevant_scores_list = [
        s for s in (channel_relevance_scores(payloads, p, task_index, n_value, times) for p in irrelevant_properties)
        if s is not None
    ]
    if relevant_scores is None or not irrelevant_scores_list:
        return {"task": relevant_property, "n": n_value, "skipped": "insufficient labeled data to rank channel relevance"}

    irrelevant_scores = np.mean(irrelevant_scores_list, axis=0)
    contrast = _zscore(relevant_scores) - _zscore(irrelevant_scores)
    H = contrast.shape[0]
    k = max(1, int(round(top_frac * H)))
    order = np.argsort(contrast)  # ascending: most irrelevant-specific first
    irrelevant_channel_idx = order[:k]
    relevant_channel_idx = order[-k:]

    X_gates = gate_matrix_for_cell(payloads, task_index=task_index, n_value=n_value, times=times)
    if X_gates.size == 0:
        return {"task": relevant_property, "n": n_value, "skipped": "no gate data (baseline / non-attention model, or run predates the gate-saving fix)"}

    def suppression_metric(G: np.ndarray) -> float:
        return float(G[:, irrelevant_channel_idx].mean() - G[:, relevant_channel_idx].mean())

    point, lo, hi = bootstrap_ci(X_gates, suppression_metric, n_boot=n_boot, seed=seed)

    mean_gate_per_channel = X_gates.mean(axis=0)
    corr = float(np.corrcoef(mean_gate_per_channel, contrast)[0, 1]) if H > 1 and contrast.std() > 1e-12 else float("nan")

    # In 'task_only' mode the gate is a pure function of the task vector, so every
    # trial in a (task, n) cell from a single checkpoint carries an IDENTICAL gate
    # vector. Resampling trials then cannot move the statistic, and the bootstrap
    # CI collapses to zero width - which looks like a precise estimate but carries
    # no information. Detect and say so rather than reporting a vacuous interval.
    n_distinct_gates = len({row.tobytes() for row in np.ascontiguousarray(X_gates)})
    ci_degenerate = bool(n_distinct_gates <= 1 or (np.isfinite(lo) and np.isfinite(hi) and hi - lo < 1e-12))

    result = {
        "task": relevant_property, "n": n_value,
        "n_trials": int(X_gates.shape[0]), "n_channels": int(H),
        "n_distinct_gate_vectors": n_distinct_gates,
        "top_frac": top_frac, "k": int(k),
        "suppression_index": {"point": point, "ci_lo": lo, "ci_hi": hi},
        "gate_relevance_correlation": corr,
        "ci_degenerate": ci_degenerate,
        "note": "suppression_index expected negative; gate_relevance_correlation expected positive, if attention suppresses irrelevant channels and amplifies relevant ones",
    }
    if ci_degenerate:
        result["ci_warning"] = (
            "Trial-level bootstrap CI is degenerate: the gate does not vary across the "
            "resampled trials (expected for attention_mode='task_only', where the gate "
            "depends only on the task vector). Treat the interval as uninformative - "
            "uncertainty here lives across channels, checkpoints and seeds, not trials."
        )
    return result


def compare_gate_suppression(
    hidden_root_a: Path,
    hidden_root_b: Path,
    label_a: str,
    label_b: str,
    epoch_a: Optional[int] = None,
    epoch_b: Optional[int] = None,
    split: Optional[str] = "val_novel_identity",
    task_indices: Tuple[int, ...] = (0, 1, 2),
    n_values: Tuple[Optional[int], ...] = (1, 2, 3),
    top_frac: float = 0.25,
    n_boot: int = 1000,
    seed: int = 42,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Compare the gate-suppression index between two conditions (e.g.
    attention-only vs. attention+proxy), each ranked against its OWN channel
    relevance (see module docstring for why cross-condition channel identity
    is not assumed).
    """
    payloads_a = load_payloads(Path(hidden_root_a), epochs=[epoch_a] if epoch_a is not None else None, split=split)
    payloads_b = load_payloads(Path(hidden_root_b), epochs=[epoch_b] if epoch_b is not None else None, split=split)
    if not payloads_a:
        raise RuntimeError(f"No payloads found under {hidden_root_a} (epoch={epoch_a}, split={split})")
    if not payloads_b:
        raise RuntimeError(f"No payloads found under {hidden_root_b} (epoch={epoch_b}, split={split})")

    # Pooling epochs is a decisive confound for this particular comparison: a
    # from-scratch run contributes many early checkpoints whose gates are still
    # near their initialization, while a short fine-tune-from-proxy run
    # contributes only already-converged ones. The resulting "sharper gating
    # after proxy pretraining" would then be a statement about the two training
    # trajectories, not about the two trained models.
    epochs_pooled = epoch_a is None or epoch_b is None
    if epochs_pooled:
        print(
            "  ! WARNING: epoch_a and/or epoch_b is None, so every saved epoch is pooled.\n"
            "    Condition A and condition B will contribute checkpoints of different\n"
            "    maturity, and the accuracy match between them (if any) does not apply.\n"
            "    Pass --epoch_a/--epoch_b to compare specific, accuracy-matched checkpoints."
        )

    cells = []
    for ti in task_indices:
        for nv in n_values:
            cell_a = gate_suppression_index(payloads_a, task_index=ti, n_value=nv, top_frac=top_frac, n_boot=n_boot, seed=seed)
            cell_b = gate_suppression_index(payloads_b, task_index=ti, n_value=nv, top_frac=top_frac, n_boot=n_boot, seed=seed)
            entry = {"task": TASK_INDEX_TO_NAME[ti], "n": nv, label_a: cell_a, label_b: cell_b}
            if "skipped" not in cell_a and "skipped" not in cell_b:
                entry["index_sharper_in_b"] = bool(
                    cell_b["suppression_index"]["point"] < cell_a["suppression_index"]["point"]
                )
                entry["index_gap"] = cell_a["suppression_index"]["point"] - cell_b["suppression_index"]["point"]
            cells.append(entry)

    result = {
        "label_a": label_a, "label_b": label_b,
        "hidden_root_a": str(hidden_root_a), "hidden_root_b": str(hidden_root_b),
        "epoch_a": epoch_a, "epoch_b": epoch_b, "split": split, "top_frac": top_frac,
        "epochs_pooled": epochs_pooled,
        "caveat": (
            "Relevance ranking is computed independently per condition (see module docstring): "
            "only the scalar suppression_index values are compared across a/b, not raw channel indices."
        ),
        **({"epoch_warning": (
            "epoch_a and/or epoch_b was None: results pool every saved checkpoint of each run. "
            "A from-scratch run contributes near-initialization gates that a short fine-tune run "
            "does not, so any a/b difference reported here is confounded with training maturity "
            "and is NOT accuracy-matched."
        )} if epochs_pooled else {}),
        "cells": cells,
    }

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "gate_suppression.json", "w") as f:
            json.dump(result, f, indent=2)
        _plot_suppression_index(cells, label_a, label_b, output_dir)

    return result


def _plot_suppression_index(cells: List[Dict[str, Any]], label_a: str, label_b: str, output_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    valid = [c for c in cells if "skipped" not in c[label_a] and "skipped" not in c[label_b]]
    if not valid:
        return
    labels = [f"{c['task']}/n{c['n']}" for c in valid]
    idx_a = [c[label_a]["suppression_index"]["point"] for c in valid]
    idx_b = [c[label_b]["suppression_index"]["point"] for c in valid]
    err_a = [
        [c[label_a]["suppression_index"]["point"] - c[label_a]["suppression_index"]["ci_lo"] for c in valid],
        [c[label_a]["suppression_index"]["ci_hi"] - c[label_a]["suppression_index"]["point"] for c in valid],
    ]
    err_b = [
        [c[label_b]["suppression_index"]["point"] - c[label_b]["suppression_index"]["ci_lo"] for c in valid],
        [c[label_b]["suppression_index"]["ci_hi"] - c[label_b]["suppression_index"]["point"] for c in valid],
    ]

    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.9), 5))
    ax.bar(x - width / 2, idx_a, width, yerr=err_a, label=label_a, capsize=3)
    ax.bar(x + width / 2, idx_b, width, yerr=err_b, label=label_b, capsize=3)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Gate-suppression index (irrelevant - relevant mean gate)")
    ax.set_title("Gate suppression per (task, n) cell — more negative = stronger suppression")
    ax.legend()
    plt.savefig(output_dir / "gate_suppression_index.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description="Gate-suppression index comparison between two hidden-state conditions")
    p.add_argument("--root_a", type=str, required=True)
    p.add_argument("--root_b", type=str, required=True)
    p.add_argument("--label_a", type=str, default="a")
    p.add_argument("--label_b", type=str, default="b")
    p.add_argument("--epoch_a", type=int, default=None)
    p.add_argument("--epoch_b", type=int, default=None)
    p.add_argument("--split", type=str, default="val_novel_identity",
                   choices=["val_novel_angle", "val_novel_identity"])
    p.add_argument("--n_values", type=int, nargs="*", default=[1, 2, 3])
    p.add_argument("--top_frac", type=float, default=0.25)
    p.add_argument("--n_boot", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_dir", type=str, default=None)
    args = p.parse_args()

    result = compare_gate_suppression(
        hidden_root_a=Path(args.root_a), hidden_root_b=Path(args.root_b),
        label_a=args.label_a, label_b=args.label_b,
        epoch_a=args.epoch_a, epoch_b=args.epoch_b,
        split=args.split, n_values=tuple(args.n_values),
        top_frac=args.top_frac, n_boot=args.n_boot, seed=args.seed,
        output_dir=Path(args.output_dir) if args.output_dir else None,
    )

    if args.output_dir is None:
        print(json.dumps(result, indent=2))
    else:
        print(f"Saved results to {args.output_dir}/gate_suppression.json")


if __name__ == "__main__":
    main()
