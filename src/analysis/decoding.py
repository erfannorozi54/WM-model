#!/usr/bin/env python3
"""
Decoding analyses on RNN hidden states saved by train.py.
- Train decoders (SVC with linear kernel) to predict object properties from hidden states
- Cross-task generalization: train in one task context, test in another
- Cross-time generalization: train at encoding time, test at later times

Usage examples:

python -m src.analysis.decoding \
  --hidden_root runs/wm_stsf/hidden_states \
  --property location \
  --train_time 2 --test_times 3 4 5 \
  --train_task location --test_task identity \
  --train_n 1 --test_n 2

Notes:
- Labels unseen in the training context are dropped from test evaluation.
- Uses SVC(kernel='linear') to enable weight inspection if needed.
"""

from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
import argparse
import numpy as np
import torch
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

# Local imports
from .activations import load_payloads, build_matrix, build_matrix_with_values, TASK_INDEX_TO_NAME

PROPERTY_CHOICES = ["location", "identity", "category"]
TASK_CHOICES = ["location", "identity", "category", "any"]


def _task_name_to_index(name: Optional[str]) -> Optional[int]:
    if name is None or name == "any":
        return None
    for k, v in TASK_INDEX_TO_NAME.items():
        if v == name:
            return k
    raise ValueError(f"Unknown task name: {name}")


def _labels_to_indices(y: torch.Tensor, mapping: Dict[int, int]) -> np.ndarray:
    out = []
    for v in y.tolist():
        if v in mapping:
            out.append(mapping[v])
        else:
            out.append(-1)  # mark unseen
    return np.asarray(out, dtype=np.int64)


def _align_test_labels(y_test: torch.Tensor, train_label2idx: Dict[Any, int]) -> Tuple[np.ndarray, np.ndarray]:
    # Map test labels to training indices; drop unseen labels
    inv_map = train_label2idx
    y_test_idx = _labels_to_indices(y_test, inv_map)
    keep = y_test_idx >= 0
    return y_test_idx[keep], keep


def train_decoder(X: torch.Tensor, y: torch.Tensor) -> Pipeline:
    clf = Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("svc", SVC(kernel="linear", class_weight="balanced")),
    ])
    clf.fit(X.numpy(), y.numpy())
    return clf


def evaluate(
    hidden_root: Path,
    property_name: str,
    train_time: int,
    test_times: List[int],
    train_task: Optional[str],
    test_task: Optional[str],
    train_n: Optional[List[int]],
    test_n: Optional[List[int]],
    epochs: Optional[List[int]] = None,
) -> Dict[str, Any]:
    payloads = load_payloads(Path(hidden_root), epochs=epochs)

    # Build train matrix
    ti = _task_name_to_index(train_task)
    X_tr, y_tr, label2idx = build_matrix(payloads, property_name, time=train_time, task_index=ti,
                                         n_value=None if not train_n or len(train_n) != 1 else train_n[0])
    if X_tr.numel() == 0:
        raise RuntimeError("No training samples found for the specified context")
    clf = train_decoder(X_tr, y_tr)

    results: Dict[str, Any] = {
        "property": property_name,
        "train_time": train_time,
        "train_task": train_task,
        "train_n": train_n,
        "test": {},
        "classes": {int(v): int(i) for v, i in label2idx.items()},
    }

    # Evaluate across test times and contexts
    tti = _task_name_to_index(test_task)
    for tt in test_times:
        X_te, _y_te, _label2idx_te, raw_vals_te = build_matrix_with_values(
            payloads, property_name, time=tt, task_index=tti,
            n_value=None if not test_n or len(test_n) != 1 else test_n[0]
        )
        if X_te.numel() == 0:
            results["test"][str(tt)] = {"acc": None, "n": 0}
            continue
        y_te_idx, keep = _align_test_labels(raw_vals_te, label2idx)
        if keep.sum() == 0:
            results["test"][str(tt)] = {"acc": None, "n": 0}
            continue
        X_te_np = X_te.numpy()[keep]
        y_te_np = y_te_idx
        y_pred = clf.predict(X_te_np)
        acc = accuracy_score(y_te_np, y_pred)
        results["test"][str(tt)] = {"acc": float(acc), "n": int(len(y_te_np))}

    return results


def parse_int_list(vals: Optional[List[str]]) -> Optional[List[int]]:
    if not vals:
        return None
    out = []
    for v in vals:
        out.append(int(v))
    return out


def main():
    p = argparse.ArgumentParser(description="Decoding analyses on hidden states")
    p.add_argument("--hidden_root", type=str, required=True, help="Path to runs/<exp>/hidden_states")
    p.add_argument("--property", type=str, choices=PROPERTY_CHOICES, required=True)
    p.add_argument("--train_time", type=int, required=True)
    p.add_argument("--test_times", type=int, nargs="+", required=True)
    p.add_argument("--train_task", type=str, choices=TASK_CHOICES, default="any")
    p.add_argument("--test_task", type=str, choices=TASK_CHOICES, default="any")
    p.add_argument("--train_n", type=int, nargs="*")
    p.add_argument("--test_n", type=int, nargs="*")
    p.add_argument("--epochs", type=int, nargs="*")
    args = p.parse_args()

    res = evaluate(
        hidden_root=Path(args.hidden_root),
        property_name=args.property,
        train_time=args.train_time,
        test_times=args.test_times,
        train_task=None if args.train_task == "any" else args.train_task,
        test_task=None if args.test_task == "any" else args.test_task,
        train_n=parse_int_list(args.train_n),
        test_n=parse_int_list(args.test_n),
        epochs=parse_int_list(args.epochs),
    )

    import json
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
