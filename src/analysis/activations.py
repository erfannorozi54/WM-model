import torch
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Iterable

from ..utils.logger import get_logger
logger = get_logger()

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


def _list_payload_files(hidden_root: Path, epochs: Optional[List[int]] = None, split: Optional[str] = None) -> List[Path]:
    """List payload files from hidden_states directory.
    
    Structure: epoch_XXX/<split>/batch_XXXX.pt
    """
    files: List[Path] = []
    for ed in _list_epoch_dirs(hidden_root, epochs):
        for sd in ed.iterdir():
            if sd.is_dir() and (split is None or sd.name == split):
                files += sorted(sd.glob("batch_*.pt"))
    return files


def load_payloads(hidden_root: Path, epochs: Optional[List[int]] = None, limit: Optional[int] = None, split: Optional[str] = None) -> List[Dict[str, Any]]:
    """Load hidden-state payload dicts saved by train.py.
    
    Args:
        hidden_root: Path to hidden_states directory
        epochs: Optional list of epoch numbers to load
        limit: Optional max number of files to load
        split: Optional split name filter ('val_novel_angle' or 'val_novel_identity')
    """
    hidden_root = Path(hidden_root)
    files = _list_payload_files(hidden_root, epochs, split)
    if limit is not None:
        files = files[:limit]
    payloads: List[Dict[str, Any]] = []
    for f in files:
        try:
            payloads.append(torch.load(f, map_location="cpu"))
        except Exception as e:
            logger.warning(f"Failed to load {f}: {e}")
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


def build_cnn_matrix(payloads: List[Dict[str, Any]],
                     property_name: str,
                     time: int,
                     task_index: Optional[int] = None,
                     n_value: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor, Dict[Any, int]]:
    """Build matrix from CNN activations instead of RNN hidden states.
    
    Args:
        payloads: List of payload dicts containing 'cnn_activations'
        property_name: Property to decode (location, identity, category)
        time: Timestep to extract
        task_index: Optional task filter
        n_value: Optional n-back value filter
    
    Returns:
        X: (N, H) CNN activation matrix
        y: (N,) label indices
        label2idx: Mapping from label values to indices
    """
    assert property_name in PROPERTY_NAMES, f"Unknown property: {property_name}"
    
    label2idx: Dict[Any, int] = {}
    xs: List[torch.Tensor] = []
    ys: List[int] = []
    
    for payload in payloads:
        cnn_act = payload.get("cnn_activations")
        if cnn_act is None:
            continue
            
        B, T, H = cnn_act.shape
        task_indices = payload.get("task_index")
        n_vals = payload.get("n")
        locations = payload.get("locations")
        categories = payload.get("categories")
        identities = payload.get("identities")
        
        # Normalize locations
        if locations is not None and torch.is_tensor(locations):
            locations_ll = locations.tolist()
        else:
            locations_ll = [[None] * T for _ in range(B)]
        
        categories_ll = categories if categories is not None else [[None] * T for _ in range(B)]
        identities_ll = identities if identities is not None else [[None] * T for _ in range(B)]
        
        for b in range(B):
            # Apply filters
            if task_index is not None and task_indices is not None:
                if int(task_indices[b]) != task_index:
                    continue
            if n_value is not None and n_vals is not None:
                if int(n_vals[b]) != n_value:
                    continue
            
            # Get property value at this timestep
            if property_name == "location":
                val = locations_ll[b][time] if locations_ll else None
            elif property_name == "category":
                val = categories_ll[b][time] if categories_ll else None
            elif property_name == "identity":
                val = identities_ll[b][time] if identities_ll else None
            else:
                val = None
            
            if val is None:
                continue
            
            if val not in label2idx:
                label2idx[val] = len(label2idx)
            
            xs.append(cnn_act[b, time])
            ys.append(label2idx[val])
    
    if not xs:
        return torch.empty(0), torch.empty(0, dtype=torch.long), label2idx
    
    X = torch.stack(xs, dim=0)
    y = torch.tensor(ys, dtype=torch.long)
    return X, y, label2idx
