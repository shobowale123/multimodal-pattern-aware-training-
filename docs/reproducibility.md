# Reproduce the experiment

## Environment

The reference run was generated on Windows, CPU, Python 3.12.14, PyTorch 2.6.0+cpu, NumPy 2.2.4, pandas 2.2.3, and Matplotlib 3.10.1. The README's setup uses an isolated virtual environment. `requirements.txt` pins the direct dependencies; `environment-tested.txt` records installed transitive versions without local paths. The latter is an informational Windows snapshot, not a cross-platform lockfile.

The source notebook's saved output used Python 3.13.5 and PyTorch 2.10.0+cpu. We separately replayed its code in the reference environment to compare the public refactor with the source. Historical outputs and freshly generated results are distinguished in [source assessment](source-assessment.md).

## Commands

After installation, run from the repository root:

```bash
python -m pattern_aware --config configs/demo.json --output artifacts/demo
python -m pytest
python scripts/execute_notebook.py
```

The final command executes the clean notebook from a fresh kernel and replaces `notebooks/02_pattern_aware_training_results.ipynb`. It removes execution timing metadata while preserving synthetic tables and figures. It fails on a cell error rather than silently saving partial output.

To deliberately refresh checked-in examples:

```bash
python -m pattern_aware --config configs/demo.json --output examples/reference_run
python scripts/execute_notebook.py
```

Refresh the prose in `docs/results.md` if the metrics, configuration, or environment changes. Review the resulting diff before sharing it.

## Artifacts

Each run records the effective configuration, environment, metrics, training history, gates, ranked candidates, and figures. `summary.json` contains validation and test summaries for each model plus the mask-only control. Per-query and per-pattern CSV files expose the aggregation behind those summaries. `embeddings.npz` stores synthetic embeddings. No trained checkpoint is saved by default; enable `save_checkpoints` explicitly to retain weights in an ignored artifact directory.

Runtime outputs under `artifacts/` are ignored by Git. Only the deliberately generated `examples/reference_run/` results are tracked. Checkpoints, virtual environments, caches, environment secrets, and raw/private data directories are excluded by `.gitignore`.

## Determinism and comparison

The package seeds Python, NumPy and PyTorch, enables deterministic PyTorch algorithms, and uses one CPU thread by default. Each model starts from the same seed and receives the same sampler schedule. The synthetic generator preserves the source random-draw order. Record IDs were renamed without changing their sort order.

Within the same environment, repeated experiments should reproduce histories and metrics. Exact floating-point values may differ across operating systems, CPU libraries, Python/dependency versions, and thread settings. Architectures differ in parameter count and initialization consumption, so the same seed is not a claim of identical initial representations.

Synthetic data creation, model fitting, retrieval, and plot generation use no external data services. Installing dependencies requires network access unless wheels are already cached. GitHub Actions is configured to run tests and the default CPU experiment; local validation is separate from any future hosted CI run.

## Configuration boundaries

Edit `configs/demo.json` to change the seed, optimizer policy, fixed members per pattern, selected variants, retrieval cutoffs, plots, or checkpoint saving. The 12-pattern, four-member cohort and modality dimensions intentionally preserve the source experiment. Generating a larger or harder benchmark requires extending the generator, not merely increasing epochs.

Checkpoint selection always uses validation pattern-macro NDCG@30, even if reporting cutoffs are changed. Test metrics must not guide tuning. The package validates settings and raises an explicit error when the cohort cannot support the requested batch or retrieval protocol.
