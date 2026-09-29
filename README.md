# Multimodal Pattern-Aware Training

**A reproducible PyTorch simulation of multimodal retrieval, pattern-balanced contrastive learning, and quality-aware fusion.**

How should a retrieval model combine five incomplete representations of the same record? This project compares three fusion architectures under a shared training and evaluation protocol. It includes synthetic data generation, a custom batch sampler, a masked multi-positive contrastive loss, validation-only checkpoint selection, and interpretable gate and attention diagnostics.

The demo runs on CPU with no datasets, credentials, or pretrained model downloads. It is a small educational experiment: its high retrieval scores are not evidence of production performance.

![Training curves from the reproducible synthetic experiment](examples/reference_run/figures/training.png)

## Project at a glance

| Component | Implementation |
| --- | --- |
| Inputs | Five frozen synthetic vector spaces: location, time, narrative, subject and object attributes |
| Representation | Modality adapters → gated fusion → normalized 256-dimensional embedding |
| Variants | V1 global gates; V1.1 per-record quality gates; V2 five-token self-attention and quality gates |
| Training | Pattern-balanced batches and masked multi-positive supervised contrastive loss |
| Evaluation | Pattern-disjoint splits, cosine retrieval, Recall/NDCG/AP, pattern-macro and query-average summaries |
| Diagnostics | Gate responses, modality availability, attention matrices, mask-only negative control |

**Engineering demonstrated:** modular PyTorch models, custom sampling, explicit data contracts, missing-modality handling, leakage-aware evaluation, deterministic experiments, and tests of mathematical invariants.

## Quick start

Use Python **3.11 or 3.12**. Clone the repository and create an environment:

```bash
git clone https://github.com/shobowale123/multimodal-pattern-aware-training-.git
cd multimodal-pattern-aware-training-
python -m venv .venv
```

Activate it with `source .venv/bin/activate` on macOS/Linux or `.venv\Scripts\Activate.ps1` in Windows PowerShell. You can also call the environment's Python directly if activation is restricted.

On Windows or Linux, install the pinned CPU-only PyTorch wheel, then the project:

```bash
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -m pattern_aware --config configs/demo.json --output artifacts/demo
python -m pytest
```

On macOS, use `python -m pip install torch==2.6.0` for the first install command. The demo still uses CPU. See the official [PyTorch installation guide](https://pytorch.org/get-started/locally/) for platform details and [reproducibility notes](docs/reproducibility.md) for the tested environment.

To explore interactively:

```bash
python -m jupyter lab notebooks/01_pattern_aware_training_demo.ipynb
```

Select the environment's Python kernel and run all cells. The clean notebook imports the same package used by the command line. [The executed results notebook](notebooks/02_pattern_aware_training_results.ipynb) provides a readable preview without running anything.

## What the experiment shows

The source experiment uses 48 generated records in 12 patterns. A pattern is a group of related records. Training sees seven patterns, validation sees three different patterns, and testing sees two more. One test record has all modalities missing and is excluded from retrieval.

Ordinary random batches may contain no same-pattern positives. Here, each sampled pattern contributes two distinct records, guaranteeing positives for the contrastive loss. `K` is configurable but fixed within a run; adaptive K selection is not implemented.

The three models share the same synthetic cohort and optimization policy:

| Variant | Reliability gate | Cross-modality interaction |
| --- | --- | --- |
| V1 | One learned scalar per modality | Concatenation + MLP |
| V1.1 | Per-record sigmoid gate from synthetic quality | Concatenation + MLP |
| V2 | Per-record quality gate | Four-head attention over five tokens, then gates and MLP |

All three achieve saturated NDCG on the small default task. The useful evidence is the working training/evaluation pipeline and its diagnostics. The task does **not** establish that attention improves retrieval. Recall@30 is mechanically perfect because the galleries have fewer than 30 candidates. Read the [results and limitations](docs/results.md) before interpreting the numbers.

## Repository guide

```text
├── README.md
├── LICENSE
├── pyproject.toml
├── requirements.txt
├── configs/                       # Reproducible experiment settings
├── src/pattern_aware/
│   ├── config.py                  # Validated settings and modality contract
│   ├── data.py                    # Synthetic cohort and aligned modality inputs
│   ├── models.py                  # Global gates, quality gates, self-attention
│   ├── sampling.py                # Pattern-balanced batch sampler
│   ├── losses.py                  # Masked multi-positive contrastive loss
│   ├── evaluation.py              # Retrieval metrics and aggregation
│   ├── training.py                # Optimization and checkpoint selection
│   └── experiment.py              # Shared experiment runner and artifacts
├── notebooks/                     # Clean demonstration + executed results
├── examples/reference_run/         # Regenerated synthetic metrics and figures
├── docs/                          # Architecture, methods, results and provenance
├── tests/                         # Data, sampler, loss, model and metric invariants
└── .github/workflows/ci.yml        # Tests and a CPU experiment
```

Start with [architecture](docs/architecture.md) for the data flow, [methodology](docs/methodology.md) for the objective and evaluation protocol, or [project summary](docs/project-summary.md) for a concise overview. The [source assessment](docs/source-assessment.md) explains the canonical-notebook decision and the missing original version.

## Scope and license

This repository simulates frozen input embeddings; it does not include upstream text encoders, a production data pipeline, a vector database, or a serving API. No real records or operational systems are included. The two structured modality keys from the source (`suspect`, `weapon`) are retained in code for technical continuity; their contents are entirely synthetic.

Released under the MIT license. See [LICENSE](LICENSE) and the [privacy and licensing notes](docs/privacy-and-licensing.md). Dependency licenses remain separate.
