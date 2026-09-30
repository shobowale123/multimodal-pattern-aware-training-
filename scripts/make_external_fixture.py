"""Write a small external-format dataset for the documented CPU lifecycle.

This fixture is generated independently of the synthetic experiment runner.
It illustrates a file contract, not evidence of real-world retrieval accuracy.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


DIMENSIONS = {"location": 8, "time": 8, "narrative": 16, "suspect": 4, "weapon": 4}


def make_fixture(output: str | Path) -> Path:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(19)
    ids = np.asarray([f"example-{i:03d}" for i in range(24)])
    labels = np.asarray([f"pattern-{i // 4}" for i in range(24)])
    splits = np.asarray(["train"] * 8 + ["validation"] * 8 + ["test"] * 8)
    schema = {
        "dimensions": DIMENSIONS,
        "encoder_versions": {name: f"example-{name}-encoder-v1" for name in DIMENSIONS},
        "quality_names": {name: ["observed_completeness"] for name in DIMENSIONS},
        "normalization": "none",
    }
    arrays = {}
    for name, dimension in DIMENSIONS.items():
        centers = rng.normal(size=(6, dimension)).astype("float32")
        vectors = centers[np.arange(24) // 4] + rng.normal(
            scale=0.12, size=(24, dimension)
        ).astype("float32")
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        available = np.ones(24, dtype=bool)
        available[-1] = False
        if name == "weapon":
            available[[2, 10, 18]] = False
        quality = rng.uniform(0.6, 1.0, size=(24, 1)).astype("float32")
        vectors[~available] = 0
        quality[~available] = 0
        arrays[name] = (vectors, quality, available)

    for directory, indices, labeled in (
        ("dataset", np.arange(24), True),
        ("queries", np.array([16, 20, 23]), False),
        ("gallery", np.arange(16, 24), False),
    ):
        destination = output / directory
        destination.mkdir()
        columns = ["record_id", "pattern_id", "split", "group_id"] if labeled else ["record_id"]
        with (destination / "records.csv").open("x", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            for i in indices:
                row = {"record_id": str(ids[i])}
                if labeled:
                    row.update(pattern_id=str(labels[i]), split=str(splits[i]), group_id=f"source-{i // 4}")
                writer.writerow(row)
        for name, (vectors, quality, available) in arrays.items():
            # Storage order is intentionally different from the CSV row order.
            rows = indices[::-1]
            with (destination / f"{name}.npz").open("xb") as handle:
                np.savez_compressed(
                    handle,
                    record_ids=ids[rows],
                    embeddings=vectors[rows],
                    quality=quality[rows],
                    content_available=available[rows],
                    source_available=available[rows],
                )
        manifest = {
            "schema_version": 1,
            "schema": schema,
            "records": "records.csv",
            "modalities": {name: f"{name}.npz" for name in DIMENSIONS},
        }
        (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    config = {
        "variant": "v2",
        "training": {"epochs": 3, "patience": 2, "seed": 19, "patterns_per_batch": 2, "members_per_pattern": 2},
        "architecture": {"adapter_dimension": 16, "fusion_hidden_dimension": 32,
                         "output_dimension": 8, "attention_feedforward_dimension": 32},
        "selection_k": 5,
        "cpu_threads": 1,
    }
    (output / "train.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New directory; existing output is never overwritten")
    args = parser.parse_args()
    print(make_fixture(args.output))
