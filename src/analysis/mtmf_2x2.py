#!/usr/bin/env python3
"""Run the five paper analyses identically across the four MTMF models, then
put the outputs side by side.

The five analyses were previously run one experiment at a time, each resolving
its own best epoch and pooling both validation splits. Two outputs produced that
way cannot be compared: different checkpoints, different mixtures of two
generalization regimes, and -- for Analysis 2 -- raw accuracies over properties
with wildly different class counts.

This module fixes the comparison rather than the individual runs. It reads
configs/analysis/mtmf_2x2.yaml, holds every cross-cell setting identical, and
emits a comparison table plus an explicit comparability audit.

Usage:
    python -m src.analysis.mtmf_2x2 --design matched
    python -m src.analysis.mtmf_2x2 --design ceiling
    python -m src.analysis.mtmf_2x2 --design matched --report_only
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "analysis" / "mtmf_2x2.yaml"


# ---------------------------------------------------------------------------
# Config + experiment resolution
# ---------------------------------------------------------------------------

def load_config(path: Path) -> Dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def resolve_experiment(prefix: str, experiments_root: Path) -> Optional[Path]:
    """Newest experiments/<prefix>_<timestamp> directory, or None.

    Matching is anchored on `<prefix>_` so that `wm_mtmf` does not also match
    `wm_mtmf_something_else`, and so `wm_attention_mtmf` is never picked up by a
    `wm_mtmf` lookup.
    """
    if not experiments_root.is_dir():
        return None
    candidates = sorted(
        (d for d in experiments_root.iterdir()
         if d.is_dir() and d.name.startswith(prefix + "_")
         and d.name[len(prefix) + 1:][:1].isdigit()),
        key=lambda d: d.name,
    )
    return candidates[-1] if candidates else None


def epoch_for(cell: Dict[str, Any], design: str) -> Optional[int]:
    key = "epoch_matched" if design == "matched" else "epoch_ceiling"
    value = cell.get(key)
    return int(value) if value is not None else None


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------

def run_cell(cell: Dict[str, Any], cfg: Dict[str, Any], design: str,
             out_root: Path, experiments_root: Path, dry_run: bool) -> Dict[str, Any]:
    label = cell["label"]
    exp_dir = resolve_experiment(cell["experiment"], experiments_root)
    if exp_dir is None:
        return {"label": label, "status": "missing_experiment",
                "experiment_prefix": cell["experiment"]}

    hidden_root = exp_dir / "hidden_states"
    model_path = exp_dir / "best_model.pt"
    if not hidden_root.is_dir():
        return {"label": label, "status": "missing_hidden_states",
                "experiment": exp_dir.name}

    out_dir = out_root / label
    epoch = epoch_for(cell, design)

    cmd = [
        sys.executable, "-m", "src.analysis.comprehensive_analysis",
        "--analysis", str(cfg.get("analysis", "all")),
        "--hidden_root", str(hidden_root),
        "--output_dir", str(out_dir),
        "--property", str(cfg.get("property", "identity")),
        "--split", str(cfg["split"]),
        "--best_epoch_by", str(cfg.get("best_epoch_by", "val_novel_identity_acc")),
    ]
    if epoch is not None:
        cmd += ["--epochs", str(epoch)]
    if model_path.is_file():
        cmd += ["--model", str(model_path)]

    record = {
        "label": label,
        "arm": cell.get("arm"),
        "experiment": exp_dir.name,
        "epoch": epoch,
        "output_dir": str(out_dir),
        "command": " ".join(cmd),
    }

    if dry_run:
        record["status"] = "dry_run"
        return record

    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "run.log"
    print(f"  -> {label}: {exp_dir.name} @ epoch {epoch} (log: {log_path})")
    with open(log_path, "w") as log:
        proc = subprocess.run(cmd, cwd=REPO_ROOT, stdout=log,
                              stderr=subprocess.STDOUT, text=True)
    record["returncode"] = proc.returncode
    record["status"] = "ok" if proc.returncode == 0 else "failed"
    return record


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _read(path: Path) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def collect_cell(label: str, out_dir: Path) -> Dict[str, Any]:
    """Pull the comparable numbers out of one cell's analysis outputs."""
    a2 = _read(out_dir / "analysis2_encoding.json")
    a3 = _read(out_dir / "analysis3_orthogonalization.json")
    a4 = _read(out_dir / "analysis4_wm_dynamics.json")

    cell: Dict[str, Any] = {"label": label, "output_dir": str(out_dir)}

    prov = None
    for block in (a2, a3, a4):
        if block and isinstance(block.get("provenance"), dict):
            prov = block["provenance"]
            break
    cell["provenance"] = prov

    # Analysis 2 -- task relevance, on the normalised scale.
    if a2 and isinstance(a2.get("task_relevance"), dict):
        tr = a2["task_relevance"]
        summary = tr.get("_summary", {})
        cell["task_relevance"] = {
            task: {
                "relevant_normalized": sv.get("relevant_normalized"),
                "best_irrelevant_normalized": sv.get("best_irrelevant_normalized"),
                "relevance_margin": sv.get("relevance_margin"),
                "relevant_is_best_decoded": sv.get("relevant_is_best_decoded"),
            }
            for task, sv in summary.items()
        }
        # Every (task, property) cell, normalised, so irrelevant-feature
        # suppression can be read directly across models.
        grid = {}
        for task, props in tr.items():
            if str(task).startswith("_") or not isinstance(props, dict):
                continue
            for prop, r in props.items():
                if isinstance(r, dict) and r.get("normalized_accuracy") is not None:
                    grid[f"{task}->{prop}"] = {
                        "normalized": r["normalized_accuracy"],
                        "raw": r.get("test_accuracy"),
                        "chance": r.get("chance_level"),
                        "n_test": r.get("n_test"),
                        "n_classes": r.get("n_classes"),
                        "is_relevant": r.get("is_relevant"),
                        "test_sample_hash": r.get("test_sample_hash"),
                    }
        cell["decoding_grid"] = grid

    # Analysis 3 -- orthogonalization.
    if a3:
        cell["orthogonalization"] = {
            k: v for k, v in a3.items()
            if k != "provenance" and not isinstance(v, (list, dict))
        }
        for key in ("perceptual", "encoding", "indices", "summary"):
            if isinstance(a3.get(key), dict):
                cell.setdefault("orthogonalization_detail", {})[key] = a3[key]

    # Analysis 4 -- H1 / H2.
    if a4:
        keep = {}
        for key in ("h1", "h2", "h1_cross_time", "h2_cross_stimulus",
                    "h2_procrustes_swap", "procrustes"):
            if key in a4:
                keep[key] = a4[key]
        cell["wm_dynamics"] = keep or None

    return cell


def audit_comparability(cells: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Check the things that must match before these outputs may be compared."""
    findings: List[str] = []
    provs = [(c["label"], c.get("provenance")) for c in cells]
    present = [(l, p) for l, p in provs if p]

    missing = [l for l, p in provs if not p]
    if missing:
        findings.append(
            f"No provenance block for: {', '.join(missing)} — these were produced "
            f"before provenance was recorded and cannot be verified as comparable."
        )

    splits = {p["split"] for _, p in present}
    if len(splits) > 1:
        findings.append(f"SPLIT MISMATCH: {sorted(splits)} — outputs are not comparable.")
    if "ALL_SPLITS_POOLED" in splits:
        findings.append(
            "At least one cell pooled every split. Identity decodability is not the "
            "same quantity in val_novel_angle and val_novel_identity."
        )

    sources = {p.get("epoch_source") for _, p in present}
    if len(sources) > 1:
        findings.append(
            f"EPOCH SOURCE MISMATCH: {sorted(str(s) for s in sources)} — some cells "
            f"are pinned and others resolved their own best epoch."
        )
    if any(p.get("epochs_pooled") for _, p in present):
        findings.append("At least one cell pooled every epoch on disk.")

    # Trial-count spread: PR/sparsity-style measures and decoder variance both
    # depend on it, so a large spread is worth surfacing even when nothing is wrong.
    counts = {l: p.get("n_trials") for l, p in present if p.get("n_trials")}
    if len(counts) > 1:
        lo, hi = min(counts.values()), max(counts.values())
        if lo and hi / lo > 1.5:
            findings.append(
                f"Trial counts differ by more than 1.5x across cells: {counts}."
            )

    # The decoder partition is derived from stable trial ids, so cells that saw
    # the same trials should agree cell-by-cell on the test-set hash.
    hash_report: Dict[str, Any] = {}
    grids = [(c["label"], c.get("decoding_grid") or {}) for c in cells]
    keys = set().union(*[set(g.keys()) for _, g in grids]) if grids else set()
    mismatched = []
    for key in sorted(keys):
        hashes = {label: g[key].get("test_sample_hash")
                  for label, g in grids if key in g}
        distinct = {h for h in hashes.values() if h}
        if len(distinct) > 1:
            mismatched.append(key)
        hash_report[key] = {"n_distinct": len(distinct), "by_model": hashes}
    if mismatched:
        findings.append(
            f"Different decoder test sets in {len(mismatched)} of {len(keys)} "
            f"(task, property) cells — the models did not see identical trials, so "
            f"per-cell differences include a sampling component. Example: "
            f"{mismatched[0]}."
        )

    return {
        "comparable": not any(
            f.startswith(("SPLIT MISMATCH", "EPOCH SOURCE MISMATCH")) for f in findings
        ),
        "findings": findings or ["No comparability problems detected."],
        "test_set_hashes": hash_report,
    }


def _fmt(value: Any, width: int = 7, places: int = 3) -> str:
    if value is None:
        return "n/a".rjust(width)
    if isinstance(value, bool):
        return ("yes" if value else "no").rjust(width)
    if isinstance(value, (int, float)):
        return f"{value:.{places}f}".rjust(width)
    return str(value).rjust(width)


def render_markdown(cells: List[Dict[str, Any]], audit: Dict[str, Any],
                    design: str, cfg: Dict[str, Any]) -> str:
    labels = [c["label"] for c in cells]
    lines: List[str] = []
    lines.append(f"# MTMF 2x2 — comprehensive analysis, `{design}` design\n")
    lines.append(f"Split: `{cfg['split']}` · property: `{cfg.get('property')}` · "
                 f"analyses: `{cfg.get('analysis')}`\n")

    lines.append("## Cells\n")
    lines.append("| Model | Experiment | Epoch | Split | Trials |")
    lines.append("|---|---|---:|---|---:|")
    for c in cells:
        p = c.get("provenance") or {}
        used = p.get("epochs_used")
        lines.append(
            f"| `{c['label']}` | {Path(p.get('hidden_root', '')).parent.name or '—'} "
            f"| {used[0] if used else '—'} | {p.get('split', '—')} "
            f"| {p.get('n_trials') or '—'} |"
        )
    lines.append("")

    lines.append("## Comparability audit\n")
    lines.append(f"**Comparable: {'YES' if audit['comparable'] else 'NO'}**\n")
    for finding in audit["findings"]:
        lines.append(f"- {finding}")
    lines.append("")

    lines.append("## Analysis 2 — task relevance (normalised)\n")
    lines.append("`normalised = (accuracy - chance) / (1 - chance)`. Raw accuracy is "
                 "not comparable across properties: identity has ~70 classes "
                 "(chance ~1.4%), location and category have 4 (chance 25%).\n")
    tasks = sorted({t for c in cells for t in (c.get("task_relevance") or {})})
    if tasks:
        lines.append("| Task | " + " | ".join(f"`{l}` margin" for l in labels) + " |")
        lines.append("|---|" + "---:|" * len(labels))
        for task in tasks:
            row = [f"| {task} "]
            for c in cells:
                sv = (c.get("task_relevance") or {}).get(task, {})
                row.append(f"| {_fmt(sv.get('relevance_margin'))} ")
            lines.append("".join(row) + "|")
        lines.append("")
        lines.append("Positive margin = the task-relevant feature is the best-decoded "
                     "one, which is the paper's claim.\n")

    grid_keys = sorted({k for c in cells for k in (c.get("decoding_grid") or {})})
    if grid_keys:
        lines.append("### Every (task → property) cell, normalised\n")
        lines.append("| task → property | rel? | " +
                     " | ".join(f"`{l}`" for l in labels) + " |")
        lines.append("|---|---|" + "---:|" * len(labels))
        for key in grid_keys:
            rel = any((c.get("decoding_grid") or {}).get(key, {}).get("is_relevant")
                      for c in cells)
            row = [f"| {key} | {'**R**' if rel else '·'} "]
            for c in cells:
                g = (c.get("decoding_grid") or {}).get(key)
                row.append(f"| {_fmt(g.get('normalized') if g else None)} ")
            lines.append("".join(row) + "|")
        lines.append("")

    lines.append("## Caveats carried from the config\n")
    for caveat in cfg.get("caveats", []):
        lines.append(f"- {' '.join(caveat.split())}")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run and compare the five paper analyses across the MTMF 2x2")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--design", choices=["matched", "ceiling"], default="matched",
                        help="matched: accuracy-matched epoch pins (default). "
                             "ceiling: each model at its own best epoch.")
    parser.add_argument("--output_root", type=Path, default=None,
                        help="Default: analysis_results/mtmf_2x2/<design>")
    parser.add_argument("--experiments_root", type=Path,
                        default=REPO_ROOT / "experiments")
    parser.add_argument("--report_only", action="store_true",
                        help="Skip running; aggregate whatever is already on disk")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print the commands without running them")
    args = parser.parse_args()

    cfg = load_config(args.config)
    out_root = args.output_root or (
        REPO_ROOT / "analysis_results" / "mtmf_2x2" / args.design)
    out_root.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"MTMF 2x2 — design: {args.design}")
    print(f"  split:    {cfg['split']}")
    print(f"  property: {cfg.get('property')}")
    print(f"  output:   {out_root}")
    print("=" * 70)

    runs = []
    if not args.report_only:
        print("\nRunning cells:")
        for cell in cfg["cells"]:
            runs.append(run_cell(cell, cfg, args.design, out_root,
                                 args.experiments_root, args.dry_run))
        for r in runs:
            if r.get("status") not in ("ok", "dry_run"):
                print(f"  ! {r['label']}: {r['status']}")
        if args.dry_run:
            for r in runs:
                print(f"\n{r['label']}:\n  {r.get('command', '(unresolved)')}")
            return

    print("\nAggregating...")
    cells = [collect_cell(c["label"], out_root / c["label"]) for c in cfg["cells"]]
    audit = audit_comparability(cells)

    payload = {
        "design": args.design,
        "config": {k: v for k, v in cfg.items() if k != "cells"},
        "cells": cells,
        "runs": runs,
        "comparability_audit": audit,
    }
    (out_root / "comparison.json").write_text(json.dumps(payload, indent=2))
    markdown = render_markdown(cells, audit, args.design, cfg)
    (out_root / "comparison.md").write_text(markdown)

    print(f"\nComparable: {'YES' if audit['comparable'] else 'NO'}")
    for finding in audit["findings"]:
        print(f"  - {finding}")
    print(f"\nWrote {out_root / 'comparison.json'}")
    print(f"Wrote {out_root / 'comparison.md'}")


if __name__ == "__main__":
    main()
