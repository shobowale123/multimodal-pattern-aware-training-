# Synthetic reference run

Generated with the public command:

```bash
python -m pattern_aware --config configs/demo.json --output examples/reference_run
```

`summary.json` records the environment, effective settings, split sizes, selected epochs and retrieval metrics. `history.csv` contains all executed epochs. `per_query_metrics.csv` and `per_pattern_metrics.csv` expose the metric aggregation. The remaining CSV files contain synthetic gate, token, attention and ranking diagnostics.

`embeddings.npz` contains only synthetic validation/test embeddings, keyed by model and split. Their row order follows the eligible records from `cohort.csv` for that split; `record_00048` is the deliberately all-missing test record and has no embedding row. Use `numpy.load(..., allow_pickle=False)` to read it.

The six figures were regenerated from these runs. No historical notebook images or real records are included. Scores saturate on this small experiment; see [results and limits](../../docs/results.md).
