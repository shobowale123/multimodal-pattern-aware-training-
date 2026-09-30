# Reproducibility and verification

## Environment

The original reference run was generated on Windows, CPU, Python 3.12.14, PyTorch 2.6.0+cpu, NumPy 2.2.4, pandas 2.2.3, and Matplotlib 3.10.1. `requirements.txt` pins direct development/demo dependencies; `environment-tested.txt` records the historical Windows transitive versions without local paths. That snapshot is not a cross-platform lockfile.

The original source notebook's saved output used Python 3.13.5 and PyTorch 2.10.0+cpu. Its separately reported replay in the reference environment is distinguished from historical saved output in [source assessment](source-assessment.md). The new own-data lifecycle does not regenerate or replace those historical notebook results.

## Original synthetic experiment

After installation, run from the repository root:

```bash
python -m pattern_aware --config configs/demo.json --output artifacts/demo
python -m pytest
```

Notebook execution remains a separate local operation:

```bash
python scripts/execute_notebook.py
```

That command executes the clean notebook from a fresh kernel and replaces `notebooks/02_pattern_aware_training_results.ipynb`. It removes execution timing metadata and fails on a cell error. To deliberately refresh the tracked examples as well, run the demo with `--output examples/reference_run`, then regenerate the results notebook and review the resulting diffs. Update [results](results.md) if settings, metrics, or environment change.

Each synthetic comparison records effective configuration, environment, metrics, history, gate/attention diagnostics, rankings, embeddings, and figures. Checkpoints are off by default. Its `--save-checkpoints` option writes bare selected state dictionaries, not the versioned own-data bundle.

## Own-data lifecycle

The [own-data guide](own-data.md) documents the external-file schema and commands. A local synthetic fixture exercises those commands without using the demo experiment runner:

```bash
python scripts/make_external_fixture.py --output artifacts/fixture
python -m pattern_aware validate --data artifacts/fixture/dataset/manifest.json
python -m pattern_aware train --data artifacts/fixture/dataset/manifest.json --config artifacts/fixture/train.json --output artifacts/model
python -m pattern_aware encode --checkpoint artifacts/model/best --data artifacts/fixture/queries/manifest.json --output artifacts/queries.npz
python -m pattern_aware encode --checkpoint artifacts/model/best --data artifacts/fixture/gallery/manifest.json --output artifacts/gallery.npz
python -m pattern_aware retrieve --queries artifacts/queries.npz --gallery artifacts/gallery.npz --top-k 10 --output artifacts/rankings.csv
python -m pattern_aware evaluate --checkpoint artifacts/model/best --data artifacts/fixture/dataset/manifest.json --split test --output artifacts/test.json
```

Use fresh output paths when rerunning; public own-data commands refuse overwrites. A training run saves effective settings, selection/audit summary, history, and the best self-describing weight bundle automatically. Encoding and retrieval occur in separate processes and require no training labels. Test evaluation is a separate command.

The fixture is deliberately small and synthetic. It verifies software behavior without measuring real-data accuracy, serving latency, robustness under distribution shift, or operational readiness.

## Hosted CI

[GitHub Actions run 36621194783](https://github.com/shobowale123/multimodal-pattern-aware-training-/actions/runs/36621194783) passed on 2026-09-29 at commit `3ceccdb3f036eeb63a0a376231baeff9ae86e18b`: 37 tests and one default CPU comparison covering V1, V1.1, and V2. That historical workflow did not execute notebooks, test own-data checkpoint reload, or retain run artifacts.

The updated [workflow](../.github/workflows/ci.yml) is configured to:

1. Run the mathematical and lifecycle tests on Python 3.11 and 3.12 with CPU PyTorch 2.6.0.
2. Run the unchanged default synthetic comparison.
3. Build a wheel and install it in a separate environment.
4. Verify that the imported package comes from that installation, then run validation, training, fresh-process encoding, retrieval, and held-out evaluation from a working directory outside the source tree.
5. Retain only synthetic summaries, effective configurations, histories, rankings, evaluation reports, test results, and an installed-environment listing. Weights and input/embedding NPZ files are not uploaded.

See the [Actions tab](https://github.com/shobowale123/multimodal-pattern-aware-training-/actions) for the result associated with a particular commit. A configured step is not evidence that it passed; check the exact commit and completed job logs. Notebook execution remains separate and is not claimed as CI coverage.

## Determinism and selection

The package seeds Python, NumPy, and PyTorch, enables deterministic PyTorch algorithms, and defaults to one CPU thread. The synthetic comparison shares the same seed and sampler policy across variants. An own-data run selects one variant; its schema and effective settings are saved with the selected model.

Within the same environment, repeated experiments should reproduce histories and metrics. Floating-point values may differ across operating systems, CPU libraries, dependency versions, and thread settings. Different architectures consume initialization randomness differently, so a shared seed does not imply identical initial representations.

The original demo selects validation pattern-macro NDCG@30. Own-data training exposes `selection_k`, still using only validation pattern-macro NDCG. Never use test results to choose preprocessing, hyperparameters, architecture, or epoch. The saved bundle does not contain the optimizer/RNG state required for exact interrupted-training resume.

Installing dependencies needs network access unless wheels are cached. Training, encoding, retrieval, and evaluation do not use external data services. Keep own-data artifacts outside version control; the repository ignores `artifacts/`, `runs/`, and `data/`. Configure version-control exclusions before choosing other local output paths.
