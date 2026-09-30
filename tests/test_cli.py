"""Exercise the public lifecycle in separate Python processes."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pattern_aware


def _run(arguments, cwd, *, success=True):
    environment = os.environ.copy()
    # Exercise the same package installation as this test, even from another cwd.
    environment["PYTHONPATH"] = str(Path(pattern_aware.__file__).resolve().parent.parent)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["OMP_NUM_THREADS"] = "1"
    environment["MKL_NUM_THREADS"] = "1"
    result = subprocess.run([sys.executable, *map(str, arguments)], cwd=cwd, env=environment,
                            capture_output=True, text=True, timeout=120)
    if success:
        assert result.returncode == 0, result.stdout + result.stderr
    else:
        assert result.returncode == 2, result.stdout + result.stderr
    return result


def test_external_file_lifecycle_in_fresh_processes(tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts" / "make_external_fixture.py"
    fixture = tmp_path / "fixture"
    _run([script, "--output", fixture], tmp_path)
    command = ["-m", "pattern_aware"]
    dataset = fixture / "dataset" / "manifest.json"
    _run([*command, "validate", "--data", dataset, "--require-labels"], tmp_path)
    run = tmp_path / "run"
    _run([*command, "train", "--data", dataset, "--config", fixture / "train.json", "--output", run], tmp_path)
    checkpoint = run / "best"
    assert (checkpoint / "metadata.json").is_file()
    weights = checkpoint / "weights.pt"
    before = hashlib.sha256(weights.read_bytes()).hexdigest()

    query_file, gallery_file = tmp_path / "queries.npz", tmp_path / "gallery.npz"
    for kind, output in (("queries", query_file), ("gallery", gallery_file)):
        manifest = fixture / kind / "manifest.json"
        records = pd.read_csv(fixture / kind / "records.csv")
        assert list(records.columns) == ["record_id"]
        _run([*command, "encode", "--checkpoint", checkpoint, "--data", manifest,
              "--batch-size", 2, "--output", output], tmp_path)

    from pattern_aware.checkpoints import load_checkpoint
    from pattern_aware.inference import encode_dataset, load_embeddings
    from pattern_aware.io import load_dataset

    saved = load_embeddings(query_file)
    restored = encode_dataset(load_checkpoint(checkpoint), load_dataset(fixture / "queries" / "manifest.json"),
                              batch_size=2)
    np.testing.assert_array_equal(saved.vectors, restored.vectors)
    assert int(saved.available.sum()) == 2
    np.testing.assert_array_equal(saved.vectors[~saved.available], 0)
    np.testing.assert_allclose(np.linalg.norm(saved.vectors[saved.available], axis=1), 1, atol=1e-6)

    ranking_files = [tmp_path / "rankings.csv", tmp_path / "rankings_other_chunks.csv"]
    for output, query_chunk, gallery_chunk in ((ranking_files[0], 1, 2), (ranking_files[1], 8, 16)):
        _run([*command, "retrieve", "--queries", query_file, "--gallery", gallery_file,
              "--top-k", 5, "--query-chunk-size", query_chunk, "--gallery-chunk-size", gallery_chunk,
              "--output", output], tmp_path)
    first, other = map(pd.read_csv, ranking_files)
    pd.testing.assert_frame_equal(first, other, check_exact=False, atol=1e-6, rtol=1e-6)
    assert len(first) == 10
    assert not (first["query_id"] == first["candidate_id"]).any()
    assert "example-023" not in set(first["query_id"]) | set(first["candidate_id"])
    evaluation = tmp_path / "test.json"
    _run([*command, "evaluate", "--checkpoint", checkpoint, "--data", dataset,
          "--split", "test", "--output", evaluation], tmp_path)
    assert json.loads(evaluation.read_text(encoding="utf-8"))

    # Reusing a run or artifact path must not overwrite the selected weights.
    _run([*command, "train", "--data", dataset, "--config", fixture / "train.json", "--output", run],
         tmp_path, success=False)
    assert hashlib.sha256(weights.read_bytes()).hexdigest() == before
    invalid_output = tmp_path / "invalid.csv"
    _run([*command, "retrieve", "--queries", query_file, "--gallery", gallery_file,
          "--top-k", 0, "--output", invalid_output], tmp_path, success=False)
    assert not invalid_output.exists()

    bad_manifest = fixture / "queries" / "incompatible.json"
    specification = json.loads((fixture / "queries" / "manifest.json").read_text(encoding="utf-8"))
    specification["schema"]["encoder_versions"]["narrative"] = "different-encoder"
    bad_manifest.write_text(json.dumps(specification), encoding="utf-8")
    rejected = tmp_path / "incompatible.npz"
    _run([*command, "encode", "--checkpoint", checkpoint, "--data", bad_manifest, "--output", rejected],
         tmp_path, success=False)
    assert not rejected.exists()


def test_help_lists_reusable_commands(tmp_path):
    result = _run(["-m", "pattern_aware", "--help"], tmp_path)
    assert all(command in result.stdout for command in ("validate", "train", "encode", "retrieve", "evaluate", "demo"))
