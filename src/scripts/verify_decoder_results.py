#!/usr/bin/env python3
"""Verify sample counts and provenance for shared h128 decoder analyses."""

import argparse
import json
from collections import defaultdict
from pathlib import Path


PREFIXES = (
    "wm_h128_stsf", "wm_h128_stmf", "wm_h128_mtmf",
    "wm_h128_attention_stsf", "wm_h128_attention_stmf", "wm_h128_attention_mtmf",
    "wm_h128_dual_attention_stsf", "wm_h128_dual_attention_stmf",
    "wm_h128_dual_attention_mtmf",
)


def _latest(experiments: Path, prefix: str) -> Path:
    matches = sorted(experiments.glob(f"{prefix}_[0-9]*"), key=lambda path: path.stat().st_mtime)
    if not matches:
        raise AssertionError(f"No experiment found for {prefix}")
    return matches[-1]


def _scenario(prefix: str) -> str:
    name = prefix.removeprefix("wm_h128_")
    return name.removeprefix("dual_attention_").removeprefix("attention_")


def verify(base_dir: Path):
    comparable = defaultdict(list)
    for prefix in PREFIXES:
        experiment = _latest(base_dir / "experiments", prefix)
        with open(experiment / "decoder_hidden_states" / "manifest.json") as handle:
            manifest = json.load(handle)
        with open(base_dir / "analysis_results" / experiment.name / "analysis2_encoding.json") as handle:
            results = json.load(handle)

        expected_tasks = 1 if _scenario(prefix) == "stsf" else 3
        assert manifest["samples_per_task"] == 2100
        assert manifest["n_total"] == 2100 * expected_tasks
        assert set(manifest["task_counts"].values()) == {2100}

        task_signatures = {}
        for task, properties in results["task_relevance"].items():
            hashes = set()
            for prop, cell in properties.items():
                assert cell["n_train"] == 1680, (experiment.name, task, prop, cell)
                assert cell["n_test"] == 420, (experiment.name, task, prop, cell)
                hashes.add((cell["train_sample_hash"], cell["test_sample_hash"]))
            assert len(hashes) == 1, f"Property decoders do not share a split: {experiment.name}/{task}"
            task_signatures[task] = hashes.pop()

        comparable[_scenario(prefix)].append(
            (experiment.name, manifest["dataset_sha256"], task_signatures)
        )

    for scenario, entries in comparable.items():
        dataset_hashes = {entry[1] for entry in entries}
        split_signatures = {json.dumps(entry[2], sort_keys=True) for entry in entries}
        assert len(dataset_hashes) == 1, f"Dataset mismatch across {scenario} architectures"
        assert len(split_signatures) == 1, f"Decoder split mismatch across {scenario} architectures"

    print("Verified all nine h128 analyses: n_train=1680, n_test=420 per task.")
    print("Dataset and split hashes match across architectures within every scenario.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=Path, default=Path.home() / "Projects/WM-model")
    args = parser.parse_args()
    verify(args.base_dir)


if __name__ == "__main__":
    main()
