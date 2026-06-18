#!/usr/bin/env python3
"""Extract hidden states from a frozen model on a shared decoder-only dataset."""

import argparse
import hashlib
import json
import random
import shutil
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ..data.dataset import NBackDataset, custom_collate_fn
from ..data.nback_generator import NBackGenerator, TaskFeature
from ..data.shapenet_downloader import scan_generated_stimuli
from .causal_perturbation import load_trained_model


TASKS = {
    "location": TaskFeature.LOCATION,
    "identity": TaskFeature.IDENTITY,
    "category": TaskFeature.CATEGORY,
}


def _canonical_stimuli(stimuli_dir: Path):
    raw = scan_generated_stimuli(str(stimuli_dir))
    return {
        category: {
            identity: sorted(raw[category][identity])
            for identity in sorted(raw[category])
        }
        for category in sorted(raw)
    }


def _condition_seed(seed: int, task: str, n_value: int) -> int:
    value = f"{seed}:{task}:{n_value}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(value).digest()[:8], "big")


def _build_sequences(generator, task_names, n_values, samples_per_task, seed):
    if samples_per_task % len(n_values):
        raise ValueError("samples_per_task must be divisible by the number of N values")
    per_condition = samples_per_task // len(n_values)
    sequences = []
    sample_keys = []
    state = random.getstate()
    try:
        for task_name in task_names:
            for n_value in n_values:
                random.seed(_condition_seed(seed, task_name, n_value))
                generated = generator.generate_batch(
                    per_condition, n_value, TASKS[task_name]
                )
                sequences.extend(generated)
                sample_keys.extend(
                    f"{task_name}:n{n_value}:{i:05d}" for i in range(per_condition)
                )
    finally:
        random.setstate(state)
    return sequences, sample_keys


def extract(args):
    device = torch.device(args.device)
    checkpoint = torch.load(args.model, map_location="cpu")
    cfg = checkpoint["config"]
    task_names = list(cfg["task_features"])
    n_values = [int(value) for value in cfg["n_values"]]

    stimuli = _canonical_stimuli(args.stimuli_dir)
    if not stimuli:
        raise RuntimeError(f"No stimuli found in {args.stimuli_dir}")
    generator = NBackGenerator(
        stimuli,
        sequence_length=int(cfg["sequence_length"]),
        match_probability=float(cfg.get("match_probability", 0.3)),
    )
    sequences, sample_keys = _build_sequences(
        generator, task_names, n_values, args.samples_per_task, args.seed
    )

    dataset = NBackDataset(
        generator=generator,
        n_values=n_values,
        task_features=[TASKS[name] for name in task_names],
        num_sequences=len(sequences),
        sequence_length=int(cfg["sequence_length"]),
        cache_sequences=False,
    )
    dataset.cache_sequences = True
    dataset._sequence_cache = dict(enumerate(sequences))
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=custom_collate_fn,
        pin_memory=device.type == "cuda",
    )

    epoch = int(checkpoint.get("epoch", 1))
    output_dir = args.output_root / f"epoch_{epoch:03d}" / "decoder_shared"
    if args.output_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"{args.output_root} exists; pass --overwrite")
        shutil.rmtree(args.output_root)
    output_dir.mkdir(parents=True)

    model = load_trained_model(args.model, device)
    offset = 0
    manifest_rows = []
    task_counts = Counter()
    condition_counts = Counter()
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            images = batch["images"].to(device)
            task_vector = batch["task_vector"].to(device)
            logits, hidden, _ = model(images, task_vector)
            size = images.shape[0]
            batch_sequences = sequences[offset:offset + size]
            batch_keys = sample_keys[offset:offset + size]
            stimulus_paths = [
                [trial.stimulus_path for trial in sequence.trials]
                for sequence in batch_sequences
            ]
            task_index = batch["task_vector"][:, :3].argmax(dim=-1)
            payload = {
                "hidden": hidden.cpu(),
                "cnn_activations": None,
                "logits": logits.cpu(),
                "task_vector": batch["task_vector"],
                "task_index": task_index,
                "n": batch["n"],
                "targets": batch["responses"].argmax(dim=-1),
                "locations": batch["locations"],
                "categories": batch["categories"],
                "identities": batch["identities"],
                "sample_index": torch.arange(offset, offset + size),
                "sample_keys": batch_keys,
                "stimulus_paths": stimulus_paths,
                "split": "decoder_shared",
            }
            torch.save(payload, output_dir / f"batch_{batch_index:04d}.pt")

            for key, sequence, paths in zip(batch_keys, batch_sequences, stimulus_paths):
                task_name = sequence.task_feature.value
                task_counts[task_name] += 1
                condition_counts[f"{task_name}:n{sequence.n}"] += 1
                manifest_rows.append({"id": key, "task": task_name, "n": sequence.n,
                                      "stimulus_paths": paths})
            offset += size

    encoded = json.dumps(manifest_rows, sort_keys=True, separators=(",", ":")).encode()
    manifest = {
        "version": 1,
        "seed": args.seed,
        "samples_per_task": args.samples_per_task,
        "n_total": len(sequences),
        "task_counts": dict(sorted(task_counts.items())),
        "condition_counts": dict(sorted(condition_counts.items())),
        "dataset_sha256": hashlib.sha256(encoded).hexdigest(),
        "checkpoint_epoch": epoch,
    }
    with open(args.output_root / "manifest.json", "w") as handle:
        json.dump(manifest, handle, indent=2)
    print(json.dumps(manifest, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--stimuli_dir", type=Path, default=Path("data/stimuli"))
    parser.add_argument("--samples_per_task", type=int, default=2100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--overwrite", action="store_true")
    extract(parser.parse_args())


if __name__ == "__main__":
    main()
