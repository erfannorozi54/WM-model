import torch
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Iterable

TASK_INDEX_TO_NAME = {0: "location", 1: "identity", 2: "category"}
PROPERTY_NAMES = {"location", "identity", "category"}


def _list_epoch_dirs(hidden_root: Path, epochs: Optional[List[int]] = None) -> List[Path]:
    if not hidden_root.exists():
        return []
    dirs = sorted([p for p in hidden_root.iterdir() if p.is_dir() and p.name.startswith("epoch_")])
    if epochs is None:
        return dirs
    epoch_set = {f"epoch_{e:03d}" for e in epochs}
    return [d for d in dirs if d.name in epoch_set]


def _list_payload_files(hidden_root: Path, epochs: Optional[List[int]] = None) -> List[Path]:
    files: List[Path] = []
    for ed in _list_epoch_dirs(hidden_root, epochs):
        files += sorted(ed.glob("epoch*_batch*.pt"))
    return files


def load_payloads(hidden_root: Path, epochs: Optional[List[int]] = None, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """Load hidden-state payload dicts saved by train.py.
    hidden_root typically points to runs/<exp>/hidden_states
    """
    hidden_root = Path(hidden_root)
    files = _list_payload_files(hidden_root, epochs)
    if limit is not None:
        files = files[:limit]
    payloads: List[Dict[str, Any]] = []
    for f in files:
        try:
            payloads.append(torch.load(f, map_location="cpu"))
        except Exception as e:
            print(f"Warning: failed to load {f}: {e}")
    return payloads


def iterate_records(payloads: Iterable[Dict[str, Any]]) -> Iterable[Dict[str, Any]]:
    """Yield flat records for each sample and timestep.
    Each record has keys: hidden(H,), time, task_index, n, location, category, identity.
    """
    for payload in payloads:
        hidden = payload["hidden"]  # (B, T, H)
        B, T, H = hidden.shape
        task_index = payload.get("task_index")  # (B,)
        n_vals = payload.get("n")               # (B,)
        targets = payload.get("targets")        # (B, T)
        locations = payload.get("locations")     # (B, T) tensor or None
        categories = payload.get("categories")   # List[List[str]] or None
        identities = payload.get("identities")   # List[List[str]] or None

        # Normalize locations to list-of-lists
        if locations is not None and torch.is_tensor(locations):
            locations_ll = locations.tolist()
        else:
            locations_ll = [[None] * T for _ in range(B)]

        # Categories/identities may be nested lists already
        categories_ll = categories if categories is not None else [[None] * T for _ in range(B)]
        identities_ll = identities if identities is not None else [[None] * T for _ in range(B)]

        for b in range(B):
            for t in range(T):
                rec = {
                    "hidden": hidden[b, t].numpy(),
                    "time": t,
                    "task_index": int(task_index[b]) if task_index is not None else None,
                    "n": int(n_vals[b]) if n_vals is not None else None,
                    "target": int(targets[b, t]) if targets is not None else None,
                    "location": int(locations_ll[b][t]) if locations_ll is not None else None,
                    "category": categories_ll[b][t],
                    "identity": identities_ll[b][t],
                }
                yield rec


def _filter_records(records: Iterable[Dict[str, Any]],
                    time: Optional[int] = None,
                    task_index: Optional[int] = None,
                    n_value: Optional[int] = None,
                    property_name: Optional[str] = None) -> List[Dict[str, Any]]:
    out = []
    for r in records:
        if time is not None and r["time"] != time:
            continue
        if task_index is not None and r.get("task_index") != task_index:
            continue
        if n_value is not None and r.get("n") != n_value:
            continue
        if property_name is not None and r.get(property_name) is None:
            continue
        out.append(r)
    return out


def build_matrix_with_values(payloads: List[Dict[str, Any]],
                             property_name: str,
                             time: int,
                             task_index: Optional[int] = None,
                             n_value: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor, Dict[Any, int], List[Any]]:
    """Construct X (N,H), y (N,), label2idx, and raw property values list for decoding.
    """
    assert property_name in PROPERTY_NAMES, f"Unknown property: {property_name}"
    recs = _filter_records(iterate_records(payloads), time=time, task_index=task_index, n_value=n_value, property_name=property_name)

    label2idx: Dict[Any, int] = {}
    xs: List[torch.Tensor] = []
    ys: List[int] = []
    vals: List[Any] = []
    for r in recs:
        val = r[property_name]
        if val not in label2idx:
            label2idx[val] = len(label2idx)
        xs.append(torch.from_numpy(r["hidden"]))
        ys.append(label2idx[val])
        vals.append(val)
    if not xs:
        return torch.empty(0), torch.empty(0, dtype=torch.long), label2idx, []
    X = torch.stack(xs, dim=0)
    y = torch.tensor(ys, dtype=torch.long)
    return X, y, label2idx, vals


def build_matrix(payloads: List[Dict[str, Any]],
                 property_name: str,
                 time: int,
                 task_index: Optional[int] = None,
                 n_value: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor, Dict[Any, int]]:
    """Backward-compatible wrapper that discards raw values."""
    X, y, label2idx, _vals = build_matrix_with_values(payloads, property_name, time, task_index, n_value)
    return X, y, label2idx
