# Multimodal Pattern-Aware Training

**Train a PyTorch retrieval model on your own five-modality embeddings, save its selected weights, and reload it for inference.**

This package learns a normalized record representation from five incomplete, precomputed vector representations. It provides pattern-balanced contrastive training, three fusion architectures, strict input validation, validation-only model selection, a self-describing weight bundle, label-free encoding, and chunked cosine retrieval. Training and inference use CPU.

An included synthetic experiment compares the architectures without datasets, credentials, or pretrained model downloads. Its small, easy cohort and high scores do not establish real-world retrieval performance or production readiness.

## Install

Use Python **3.11 or 3.12**:

```bash
git clone https://github.com/shobowale123/multimodal-pattern-aware-training-.git
cd multimodal-pattern-aware-training-
python -m venv .venv
```

Activate with `source .venv/bin/activate` on macOS/Linux or `.venv\Scripts\Activate.ps1` in Windows PowerShell. You can also call the environment's Python directly.

On Windows or Linux, install CPU PyTorch and the package:

```bash
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install .
```

On macOS, use `python -m pip install torch==2.6.0` for the first command. The package still uses CPU. The pinned development/demo environment is available with `python -m pip install -r requirements.txt`. See [reproducibility](docs/reproducibility.md) for environment and CI details.

## Train on your own embeddings

Prepare a versioned JSON manifest, record IDs, five modality NPZ files, and training labels/splits using the [own-data contract](docs/own-data.md). The five roles are `location`, `time`, `narrative`, `suspect`, and `weapon`; each role's vector width is configurable. The final two names are retained from the original structured subject/object channels. The package does not generate embeddings from raw text or images.

```bash
python -m pattern_aware validate --data data/my_dataset/manifest.json
python -m pattern_aware train --data data/my_dataset/manifest.json --config configs/train.json --output runs/model
python -m pattern_aware encode --checkpoint runs/model/best --data data/queries/manifest.json --output runs/queries.npz
python -m pattern_aware encode --checkpoint runs/model/best --data data/gallery/manifest.json --output runs/gallery.npz
python -m pattern_aware retrieve --queries runs/queries.npz --gallery runs/gallery.npz --top-k 10 --output runs/rankings.csv
python -m pattern_aware evaluate --checkpoint runs/model/best --data data/my_dataset/manifest.json --split test --output runs/test.json
```

`train` writes the effective configuration, training history, exclusion/selection summary, and `best/metadata.json` plus `best/weights.pt`. Inference reconstructs the architecture from this bundle and checks that feature widths, encoder identities, normalization, and quality definitions match training. Inference manifests need no labels or splits. Commands refuse to overwrite existing outputs; use a new run path for another experiment.

The retrieval output includes query/candidate IDs, rank, and cosine score, with deterministic ID-based tie breaking and self-matches removed. All-missing records remain explicitly unavailable. `evaluate` writes JSON metrics and per-query/per-pattern CSV files. Model selection uses validation only; the test split is evaluated in a separate command. Keep test outcomes out of model and preprocessing choices.

To try the file-based workflow with a small **synthetic external-format fixture**:

```bash
python scripts/make_external_fixture.py --output artifacts/fixture
python -m pattern_aware validate --data artifacts/fixture/dataset/manifest.json
python -m pattern_aware train --data artifacts/fixture/dataset/manifest.json --config artifacts/fixture/train.json --output artifacts/model
python -m pattern_aware encode --checkpoint artifacts/model/best --data artifacts/fixture/queries/manifest.json --output artifacts/queries.npz
python -m pattern_aware encode --checkpoint artifacts/model/best --data artifacts/fixture/gallery/manifest.json --output artifacts/gallery.npz
python -m pattern_aware retrieve --queries artifacts/queries.npz --gallery artifacts/gallery.npz --top-k 10 --output artifacts/rankings.csv
python -m pattern_aware evaluate --checkpoint artifacts/model/best --data artifacts/fixture/dataset/manifest.json --split test --output artifacts/test.json
```

This fixture checks the public lifecycle. It is not a benchmark for your data.

## Algorithms and boundaries

| Variant | Reliability gate | Cross-modality interaction |
| --- | --- | --- |
| V1 | One learned sigmoid bias per modality | Concatenation + MLP |
| V1.1 | Per-record scalar-quality sigmoid gate | Concatenation + MLP |
| V2 | Per-record scalar-quality sigmoid gate | Self-attention over five tokens, then gates and MLP |

V2 uses four attention heads by default. Its gates follow attention, so low-quality content can influence other tokens before gating. Attention weights and gate values are diagnostics, not calibrated reliability or causal importance.

Each selected pattern contributes exactly `K` distinct eligible records. `K` is configurable and fixed within a run; **adaptive K is not implemented**. Patterns with fewer than K available records are excluded and reported. An epoch visits eligible patterns, not every record. When `P=2` leaves an odd tail, the final batch contains three patterns. Training uses all non-self batch comparisons; the lower-level loss also supports an optional comparison mask.

This release supports five fixed modality roles with configurable vector widths, CPU training/inference, selected-weight persistence, and exact chunked retrieval. It does not include raw-data encoders, arbitrary modality counts, GPU/distributed training, exact interrupted-training resume, a vector database, or a serving API. Validate preprocessing, leakage boundaries, retrieval quality, and operational constraints on representative data before deployment.

## Reproduce the original experiment

Install the pinned development/demo requirements, then run:

```bash
python -m pip install -r requirements.txt
python -m pattern_aware --config configs/demo.json --output artifacts/demo
python -m pytest
python -m jupyter lab notebooks/01_pattern_aware_training_demo.ipynb
```

The demo keeps the original 48 records, 12 patterns, three-variant comparison, diagnostics, and validation NDCG@30 selection. Its optional bare `.pt` state dictionaries are distinct from own-data model bundles. The [executed results notebook](notebooks/02_pattern_aware_training_results.ipynb) provides a preview; notebook execution is a separate local check, not part of CI.

![Training curves from the reproducible synthetic experiment](examples/reference_run/figures/training.png)

All three variants saturate NDCG on the default task. Recall@30 is mechanically perfect because the galleries have fewer than 30 candidates. These results do not establish that attention improves retrieval. Read [results and limitations](docs/results.md) before interpreting the numbers.

## Repository guide

| Path | Purpose |
| --- | --- |
| `src/pattern_aware/config.py`, `data.py`, `io.py` | Settings, feature schemas, aligned inputs, external-file validation |
| `models.py`, `sampling.py`, `losses.py`, `training.py` | Fusion architectures, fixed-K batches, objective, model selection |
| `pipeline.py`, `checkpoints.py` | Own-data training and self-describing selected weights |
| `inference.py`, `retrieval.py`, `evaluation.py` | Label-free encoding, chunked ranking, held-out metrics |
| `experiment.py`, `configs/demo.json`, `notebooks/` | Original reproducible synthetic comparison |
| `configs/train.json`, `scripts/make_external_fixture.py` | Own-data training configuration and synthetic format example |
| `tests/`, `.github/workflows/ci.yml` | Mathematical invariants, lifecycle tests, installed-wheel acceptance |

See [architecture](docs/architecture.md), [methodology](docs/methodology.md), [own-data setup](docs/own-data.md), and [source assessment](docs/source-assessment.md). The code and included synthetic examples use the [MIT license](LICENSE); dependency and user-data rights remain separate. See [privacy and licensing](docs/privacy-and-licensing.md).
