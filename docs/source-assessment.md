# Source assessment and notebook policy

## What was inspected

Two of the three named source notebooks were available for direct inspection:

| Source notebook | Assessment |
|---|---|
| `multimodal_pattern_aware_training_simulation_v2.ipynb` | Clean source: 40 cells, including 22 code cells and 18 Markdown cells; no saved outputs. |
| `multimodal_pattern_aware_training_simulation_v2_executed.ipynb` | Exactly the same cell sources, cell types and cell IDs; 52 saved output objects, including 11 figures, and execution metadata. |
| `multimodal_pattern_aware_training_simulation.ipynb` | Unavailable during this review. No comparison or claim of feature parity with this original version is made. |

Source equality was checked across all 40 corresponding cells. The executed copy records Python 3.13.5 and PyTorch 2.10.0+cpu, with no saved error outputs. Those observations describe the supplied artifact; they are separate from the environment recorded for fresh runs of this repository.

## Canonical implementation

The V2 source is the basis of this repository because it contains all three comparison variants: global gates, per-record quality gates, and five-token self-attention. A single walkthrough imports the reusable implementation from `src/pattern_aware/`; code in the package is the implementation to maintain and test.

The clean and executed source files are duplicate implementations, so neither is copied wholesale into this repository. The notebook policy is:

1. Maintain one clean, runnable walkthrough as the editable notebook source.
2. Generate a separate result notebook from that walkthrough when preparing a release or refreshing example results.
3. Retain only that newly executed public result notebook, together with selected figures and structured metrics. Do not edit its outputs by hand.
4. Record the experiment configuration and environment with results, and regenerate them when the implementation or defaults change.

Historical numbers below are clearly labeled and are not presented as a fresh run.

## Where the notebook responsibilities moved

| Location | Responsibility |
|---|---|
| `src/pattern_aware/data.py` | Synthetic cohort and frozen vectors, artifact alignment, eligibility and missingness masks. |
| `src/pattern_aware/models.py` | Modality adapters, reliability gates, concatenation fusion, five-token attention, diagnostics. |
| `src/pattern_aware/sampling.py` | Pattern-balanced batches with distinct members and positive/negative coverage. |
| `src/pattern_aware/losses.py` | Supervised contrastive objective and label encoding. |
| `src/pattern_aware/evaluation.py` | Self-excluding cosine retrieval, Recall/NDCG/AP, query and pattern aggregation. |
| `src/pattern_aware/training.py` | Seeding, optimization, validation-only checkpoint selection and embedding extraction. |
| `src/pattern_aware/config.py` and `configs/` | Validated settings and a readable default experiment configuration. |
| `docs/` | Architecture, methodology, limitations, provenance and sharing policy. |
| `tests/` | Contract checks, missing-modality behavior, sampler invariants, loss behavior and hand-computed retrieval cases. |
| `notebooks/` | The guided comparison and its separately generated result artifact. |

## What the saved source results establish

The supplied executed notebook used 48 synthetic records from 12 patterns. Pattern-disjoint splits contained 28 training, 12 validation and eight test records; a deliberately all-missing test record was excluded, leaving seven eligible test records.

| Variant | Trainable parameters | Selected epoch | Epochs run | Saved validation NDCG@30 | Saved test NDCG@30 |
|---|---:|---:|---:|---:|---:|
| V1 global | 1,025,413 | 9 | 14 | 1.0000 | 1.0000 |
| V1.1 per-record | 1,025,418 | 9 | 14 | 1.0000 | 1.0000 |
| V2 attention | 1,158,538 | 2 | 7 | 1.0000 | 1.0000 |

These are pattern-macro metrics from the supplied notebook. Its mask-only validation control has NDCG@30 of approximately 0.5421 and NDCG@5 of approximately 0.2633. They demonstrate a working comparison on an easy synthetic task; they do not establish a superior fusion architecture or operational retrieval performance.

The evaluator clips K to the non-self gallery size: 11 validation candidates and six test candidates in this example. Recall@30 is therefore automatically 1.0 even for an unfavorable ranking. NDCG remains sensitive to ordering. See [methodology](methodology.md) for the metric conventions and experimental limits.

## Sharing changes

The public version defines record IDs without relying on internal domain abbreviations and removes references to unspecified production designs or approval contracts. All records, labels, vectors and quality fields are generated. No raw incident records, private pretrained models, credentials or external data services are needed.

The inspected source text, textual outputs and metadata contained no credentials, email addresses or private filesystem paths. The 11 saved figures were also visually inspected and contained synthetic charts only. Clean notebook metadata excludes the source execution timestamps and environment-specific execution state. See [privacy and licensing](privacy-and-licensing.md) for the scope of the public artifact.

## Source edge cases identified during extraction

The source's default P=3, K=2 sampler works for its seven training patterns, but its generic tail repair has an edge case: with P=2 and an odd pattern count, moving a pattern from the preceding batch leaves that preceding batch with one pattern and no negative pattern. The reusable sampler merges that tail into a three-pattern final batch and updates its reported batch count. Every emitted batch retains at least two patterns, and the sampler tests explicitly cover this case.

The notebook also assumes valid shapes, finite inputs, positive retrieval cutoffs and a nonempty paired query set. These assumptions become explicit contracts in reusable functions, with focused tests for invalid inputs and hand-computed metric examples. This hardening preserves the intended default experiment rather than changing its research objective.

For an all-missing record, V2 temporarily unmasks a sentinel key to prevent an undefined attention softmax. Its final embedding was already zero in the source. The public implementation also zeros the sentinel attention diagnostics so they cannot be mistaken for observed modality interactions.

A one-time extraction check ran the inspected notebook definitions and the public modules in the same local environment. The synthetic arrays and masks matched exactly; default sampled indices matched for all 18 possible epochs. For all three models, initial parameters, all-record embeddings, retrieval metrics, training histories and selected checkpoint weights matched exactly. The contrastive loss and its embedding gradients also matched. This verifies preservation of the default numerical experiment within that environment, not bitwise equivalence across dependency versions or hardware.

Several important behaviors are preserved and documented rather than silently changed:

- K is fixed within a run; the source contains no adaptive K=2–4 policy. With K=2, every anchor has one positive, although the loss supports multiple positives.
- Quality-conditioned gates may decrease with quality. Time and suspect gates have negative slopes in the saved example; the gates are not calibrated reliability probabilities.
- V2 applies quality gates after attention, so a token can influence other tokens before its own contribution is gated.
- Attention weights are diagnostics, not causal explanations. The saved example is close to uniform across modalities.
- The study uses one seed, tiny galleries and simulated frozen embeddings. It does not train raw text, image or geospatial encoders or evaluate a deployed recommendation system.
