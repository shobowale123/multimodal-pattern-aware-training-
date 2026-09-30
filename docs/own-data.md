# Train and retrieve with your own embeddings

This workflow trains the fusion model using five precomputed modalities. You supply embeddings, identities, masks, pattern labels, and leakage-safe splits. No raw-text/image encoder is trained or downloaded. The installed package operates locally on CPU.

## Dataset layout

A dataset contains a UTF-8 JSON manifest, a UTF-8 CSV, and one NPZ archive per role:

```text
my_dataset/
  manifest.json
  records.csv
  location.npz
  time.npz
  narrative.npz
  suspect.npz
  weapon.npz
```

All five files are required, including for roles that are absent for some records. Paths resolve relative to the manifest; absolute paths are also accepted. Use relative paths for portable datasets. Arrays are aligned by exact record ID, never by row position.

The manifest has exactly these top-level keys. The illustrative encoder versions and quality names below must be replaced with your actual encoder/preprocessing versions and quality definitions:

```json
{
  "schema_version": 1,
  "schema": {
    "dimensions": {
      "location": 256,
      "time": 256,
      "narrative": 1024,
      "suspect": 128,
      "weapon": 128
    },
    "encoder_versions": {
      "location": "my-location-encoder@1",
      "time": "my-time-encoder@1",
      "narrative": "my-narrative-encoder@1",
      "suspect": "my-subject-encoder@1",
      "weapon": "my-object-encoder@1"
    },
    "quality_names": {
      "location": ["location_confidence_v1"],
      "time": ["time_coverage_v1"],
      "narrative": ["narrative_coverage_v1"],
      "suspect": ["subject_coverage_v1"],
      "weapon": ["object_coverage_v1"]
    },
    "normalization": "l2"
  },
  "records": "records.csv",
  "modalities": {
    "location": "location.npz",
    "time": "time.npz",
    "narrative": "narrative.npz",
    "suspect": "suspect.npz",
    "weapon": "weapon.npz"
  }
}
```

All role dictionaries must contain exactly `location`, `time`, `narrative`, `suspect`, and `weapon`. Dimensions are positive integers. An encoder version is a nonempty identity string: include the model revision and any fitted preprocessing revision. Equality of these declared identities is checked; the package cannot verify that the upstream encoder actually matches the declaration.

`normalization` is `"none"` or `"l2"` and applies to every modality when loading. Missing-content vectors are zeroed after validation. With `"l2"`, nonzero vectors are normalized per row and zero vectors remain zero. No normalization parameters are learned by this loader.

V1.1 and V2 require one named scalar quality for every role. Quality is finite and in `[0, 1]`; document its definition and scaling, including how missingness is represented. V1 ignores quality and accepts schemas with zero or one quality field per role. For a quality-free role use `[]` and an `N x 0` quality array. The schema saved during training must still match inference exactly.

## Records and split boundaries

A training CSV requires `record_id,pattern_id,split`, with optional `group_id`:

```csv
record_id,pattern_id,split,group_id
record-a,pattern-a,train,source-a
record-b,pattern-a,train,source-a
record-c,pattern-b,validation,source-b
record-d,pattern-b,validation,source-b
record-e,pattern-c,test,source-c
record-f,pattern-c,test,source-c
```

This snippet illustrates the format; training requires at least two eligible training patterns, with at least K records each. Values must be nonempty strings, record IDs unique, and split names exactly `train`, `validation`, or `test`. Unknown columns, duplicate headers/IDs, malformed rows, cross-split patterns, and cross-split groups are rejected. Validation requires at least two patterns with at least two available records each, so its retrieval task includes positives and distractors.

Assign pattern and optional source/group boundaries before preprocessing or tuning. Fit transforms and any upstream trainable encoder using the training split only. Keep near-duplicates and records from the same leakage source in one split. Supply `group_id` when a grouping beyond pattern membership matters. ID/group checks cannot detect unlabelled duplicates, incorrect group assignments, or preprocessing fit on held-out data.

For label-free inference, use a CSV containing only `record_id`. Labels and splits must either both be present or both be absent; `group_id` requires a labeled dataset. Use the same feature schema and preprocessing as training, with new record IDs as needed.

## Modality NPZ arrays

Each archive contains exactly the following five arrays and can be written with `numpy.savez_compressed`. Object arrays and pickle loading are not supported.

| Array | Shape/type | Contract |
| --- | --- | --- |
| `record_ids` | `(N,)`, Unicode strings | Unique, nonempty; exact same ID set as the CSV; any order |
| `embeddings` | `(N, D)`, floating | Finite; D matches the schema; convertible to finite float32 |
| `quality` | `(N, Q)`, floating | Q is 0 or 1 and matches `quality_names`; finite, in `[0,1]` |
| `content_available` | `(N,)`, Boolean | Whether the embedding represents usable content |
| `source_available` | `(N,)`, Boolean | Must be true whenever content is true |

For example, after computing arrays for one role:

```python
np.savez_compressed(
    "location.npz",
    record_ids=np.asarray(record_ids, dtype=str),
    embeddings=np.asarray(location_vectors, dtype=np.float32),
    quality=np.asarray(location_quality, dtype=np.float32).reshape(-1, 1),
    content_available=np.asarray(location_has_content, dtype=bool),
    source_available=np.asarray(location_has_source, dtype=bool),
)
```

All source values must be finite, including masked rows. Content cannot exist without its source. The model consumes five content bits and the two structured source bits (`suspect`, `weapon`); source flags for the first three roles are validated but are not additional fusion inputs. All-missing records are accepted and counted, but cannot supply training anchors or retrieval candidates.

## Validate and train

```bash
python -m pattern_aware validate --data data/my_dataset/manifest.json
python -m pattern_aware train --data data/my_dataset/manifest.json --config configs/train.json --output runs/model
```

Validation reports dimensions, availability, and split counts. Training performs additional preflight checks for usable batches and validation queries. `configs/train.json` exposes:

| Setting | Meaning |
| --- | --- |
| `variant` | `v1`, `v11`, or `v2`; one model per run |
| `training` | Seed, epochs, AdamW settings, temperature, P/K, patience, minimum improvement |
| `architecture` | Token, fusion hidden/output, feed-forward widths, attention heads, dropout |
| `selection_k` | Validation pattern-macro NDCG cutoff, default 30 |
| `cpu_threads` | Positive CPU thread count, default 1 |

The token width must be divisible by the attention head count. Input widths come from the dataset schema. K is fixed throughout the run; patterns with fewer than K available records are excluded and reported. All training comparisons are non-self batch comparisons. There is no adaptive-K policy.

The output directory must not already exist. A successful run contains:

```text
runs/model/
  config.json
  summary.json
  history.csv
  best/
    metadata.json
    weights.pt
```

The selected epoch and metric come only from validation; test metrics are not computed by `train`. History records loss, validation NDCG, improvement, and patience. Bundle metadata contains the feature schema, architecture, training settings, selection/provenance, and weight/schema digests. Keep both bundle files together. Loading uses CPU mapping, weights-only tensor loading, integrity checks, and strict state reconstruction. The bundle is for inference, not exact interrupted-training resume. The old demo's optional bare `.pt` files are not this format.

The same lifecycle is available as a Python API, as an alternative to the CLI:

```python
from pattern_aware import (
    train_from_manifest, load_checkpoint, load_dataset,
    encode_dataset, save_embeddings,
)

train_from_manifest("data/my_dataset/manifest.json", "configs/train.json", "runs/model")
checkpoint = load_checkpoint("runs/model/best")
queries = load_dataset("data/queries/manifest.json")  # Labels are optional.
encoded = encode_dataset(checkpoint, queries, batch_size=32)
save_embeddings(encoded, "runs/queries.npz")
```

## Encode and retrieve

```bash
python -m pattern_aware encode --checkpoint runs/model/best --data data/queries/manifest.json --output runs/queries.npz --batch-size 32
python -m pattern_aware encode --checkpoint runs/model/best --data data/gallery/manifest.json --output runs/gallery.npz --batch-size 32
python -m pattern_aware retrieve --queries runs/queries.npz --gallery runs/gallery.npz --top-k 10 --output runs/rankings.csv
```

Encoding produces a non-pickle NPZ with `record_ids` (Unicode), `vectors` (float32), `available` (Boolean), and `metadata` (scalar JSON string). Metadata carries format version, feature schema, full `fusion_config`, selected-weight SHA-256, and variant. Available vectors have unit norm; unavailable rows have zero vectors. Outputs retain the input CSV's ID order.

Retrieval requires matching schema, architecture, weight identity, variant, and output width. It excludes unavailable queries/candidates and equal-ID self-matches. Each query receives at most K rows, ordered by descending cosine score and ascending candidate ID for ties. The CSV columns are `query_id,candidate_id,rank,score`; a sibling `rankings.report.json` reports availability/exclusion counts. Chunking bounds temporary score memory; embeddings remain in memory, and exact search still computes query-by-gallery comparisons.

## Evaluate a held-out split

```bash
python -m pattern_aware evaluate --checkpoint runs/model/best --data data/my_dataset/manifest.json --split test --output runs/test.json
```

Evaluation uses each available record as a query against the available gallery in that split, removing itself. Relevant candidates share a pattern. Singleton queries are excluded, but singleton records remain gallery distractors. The JSON summary reports exclusions, query averages, and pattern-macro Recall/NDCG/AP; `test.per_query.csv` and `test.per_pattern.csv` expose the aggregates.

Training-split evaluation is rejected. Test records, patterns, and supplied groups are checked against saved training and validation provenance; overlapping held-out identities are rejected. These are declared-identity checks, not proof that a dataset is free of leakage. Use validation for choices and evaluate test only after choices are fixed. Do not treat repeated test-driven tuning as independent evaluation.

The bundled synthetic fixture exercises the same file/CLI contract in CI. It demonstrates successful optimization, saving, loading, and retrieval, not performance on private data or production readiness.
