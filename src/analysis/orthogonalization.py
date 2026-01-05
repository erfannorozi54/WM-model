#!/usr/bin/env python3
"""
Orthogonalization analysis using one-vs-rest linear SVCs.
- For a chosen property (location|identity|category), train OVR linear SVCs for each class
- Extract hyperplane normal vectors W and compute pairwise cosine similarities
- Compute orthogonalization index O (as in Eq. 1 of the paper):

  O = 1 - mean_{i<j} cos_sim(W_i, W_j)

Usage example:

python -m src.analysis.orthogonalization \
  --hidden_root runs/wm_mtmf/hidden_states \
  --property location \
  --time 2 --task any --n 2

Notes:
- You can restrict to a specific task and/or N, or set task=any.
- Only classes present in the selected context are used.
"""

from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
import argparse
import numpy as np
import torch
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from .activations import load_payloads, build_matrix, TASK_INDEX_TO_NAME

PROPERTY_CHOICES = ["location", "identity", "category"]
TASK_CHOICES = ["location", "identity", "category", "any"]


def _task_name_to_index(name: Optional[str]) -> Optional[int]:
    if name is None or name == "any":
        return None
    for k, v in TASK_INDEX_TO_NAME.items():
        if v == name:
            return k
    raise ValueError(f"Unknown task name: {name}")


def one_vs_rest_weights(X: torch.Tensor, y: torch.Tensor) -> Dict[int, np.ndarray]:
    classes = sorted(set(y.tolist()))
    W: Dict[int, np.ndarray] = {}
    for c in classes:
        y_bin = (y.numpy() == c).astype(np.int32)
        clf = Pipeline([
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("svc", LinearSVC(class_weight="balanced")),
        ])
        clf.fit(X.numpy(), y_bin)
        w = clf.named_steps["svc"].coef_[0]
        W[c] = w / (np.linalg.norm(w) + 1e-12)
    return W


def cosine_similarity_matrix(W: Dict[int, np.ndarray]) -> np.ndarray:
    keys = sorted(W.keys())
    if len(keys) < 2:
        return np.zeros((len(keys), len(keys)))
    M = np.zeros((len(keys), len(keys)))
    for i, ki in enumerate(keys):
        for j, kj in enumerate(keys):
            M[i, j] = float(np.dot(W[ki], W[kj]))
    return M


def orthogonalization_index(W: Dict[int, np.ndarray]) -> float:
    keys = sorted(W.keys())
    if len(keys) < 2:
        return 0.0
    cos_vals = []
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            cos_vals.append(abs(float(np.dot(W[keys[i]], W[keys[j]]))))
    if not cos_vals:
        return 0.0
    return float(1.0 - np.mean(cos_vals))


def evaluate(
    hidden_root: Path,
    property_name: str,
    time: int,
    task: Optional[str],
    n_value: Optional[int],
    epochs: Optional[List[int]] = None,
) -> Dict[str, Any]:
    payloads = load_payloads(Path(hidden_root), epochs=epochs)
    ti = _task_name_to_index(task)
    X, y, label2idx = build_matrix(payloads, property_name, time=time, task_index=ti, n_value=n_value)
    if X.numel() == 0:
        raise RuntimeError("No samples for the specified context")

    W = one_vs_rest_weights(X, y)
    C = cosine_similarity_matrix(W)
    O = orthogonalization_index(W)

    return {
        "property": property_name,
        "time": time,
        "task": task,
        "n": n_value,
        "n_classes": int(len(W)),
        "orthogonalization": float(O),
        "cosine_matrix": C.tolist(),
        "classes": {int(v): int(i) for v, i in label2idx.items()},
    }


def main():
    p = argparse.ArgumentParser(description="Orthogonalization analysis (one-vs-rest SVC)")
    p.add_argument("--hidden_root", type=str, required=True)
    p.add_argument("--property", type=str, choices=PROPERTY_CHOICES, required=True)
    p.add_argument("--time", type=int, required=True)
    p.add_argument("--task", type=str, choices=TASK_CHOICES, default="any")
    p.add_argument("--n", type=int, default=None)
    p.add_argument("--epochs", type=int, nargs="*")
    args = p.parse_args()

    res = evaluate(
        hidden_root=Path(args.hidden_root),
        property_name=args.property,
        time=args.time,
        task=None if args.task == "any" else args.task,
        n_value=args.n,
        epochs=args.epochs,
    )

    import json
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
