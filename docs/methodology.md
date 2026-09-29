# Methodology and evaluation

## Controlled synthetic experiment

The seed-42 cohort contains 48 synthetic records in 12 patterns, with four records per pattern. Seven patterns belong to training, three to validation, and two to test. Splits are assigned at the pattern level before training. No pattern appears in more than one split.

A 16-dimensional pattern latent produces shared signal. Each modality uses a separate fixed random projection, incident-level perturbation, and quality-dependent noise. Quality is sampled from `[0.35, 1.0]` for available modalities; missing modalities receive zero quality and zero vectors. Missingness probabilities differ by modality. The final test record is deliberately all-missing, leaving 28 training, 12 validation, and seven test records eligible at seed 42.

Quality has a designed relationship to noise in this generator. It is not a measured reliability label and does not establish that a similar quality gate will help with real data. Sharing the generator across splits also makes generalization much easier than a real distribution shift.

## Pattern-aware batches

The sampler selects distinct records from each eligible pattern. The default is `P=3` patterns per batch and `K=2` records per pattern. Each training anchor therefore has a positive and records from other patterns as negatives. Tail batches may contain fewer patterns; the reusable implementation handles an odd tail while preserving at least two patterns per emitted batch.

`K` is configurable and fixed within a run. The provided notebooks do not implement an adaptive K=2–4 policy. An epoch visits patterns, but only samples K members from each; it does not promise a complete pass over all records.

When `P=2` leaves one pattern at the end, the reusable sampler merges it into the preceding batch, producing a final three-pattern batch. `P` is a target size rather than a strict maximum in that case.

## Objective

The supervised contrastive objective averages the negative log probability of all same-pattern positives for each anchor. Self-comparisons are removed. An optional approved-comparison mask restricts other candidates while preserving every positive. Temperature defaults to `0.07`.

For unit embedding `z_i`, positive set `P(i)`, and allowed non-self comparisons `A(i)`:

```text
L_i = -(1 / |P(i)|) * sum[j in P(i)] log(
    exp(dot(z_i, z_j) / temperature)
    / sum[a in A(i)] exp(dot(z_i, z_a) / temperature)
)
```

The package validates normalized finite embeddings and positive coverage. The formulation follows the supervised contrastive learning family described by [Khosla et al.](https://arxiv.org/abs/2004.11362); the synthetic task and fusion models here are separate from that paper's image-classification experiments.

## Training and selection

All variants use AdamW, learning rate `0.001`, weight decay `0.0001`, at most 18 epochs, patience five, and minimum improvement `0.00001`. Training is deterministic on CPU within the recorded environment. The same seed and sampler policy are used for each variant. Architectures have different parameter counts; this is not a parameter-matched comparison.

Checkpoint selection uses validation pattern-macro NDCG@30 only. The best weights are restored before test evaluation. The held-out test split is neither used for gradient updates nor for selecting an epoch.

## Retrieval protocol

Each eligible record queries the eligible gallery in the same held-out split. Relevant candidates share its pattern. Self-matches are excluded, and ties are broken by record identifier. A query must have another eligible member of its pattern; singleton records can still act as gallery negatives.

Metrics are Recall@K, NDCG@K, and average precision at K, for K in `{1, 5, 10, 30}`. Average precision uses denominator `min(total relevant, effective K)`. The package reports both the mean across queries and the mean of each pattern's query mean; the latter gives each pattern equal weight. This is a within-split retrieval experiment, not test-to-training-gallery retrieval.

Requested cutoffs are capped at gallery size minus one. At seed 42, validation has only 11 candidates per query and test has six. Consequently Recall@30 is necessarily 1.0 for every eligible query regardless of ordering. NDCG@30 still measures ordering, but this tiny gallery cannot support conclusions about large-scale retrieval.

## Controls and limits

The mask-only control uses just the five content bits and two source bits. It helps detect whether availability patterns carry retrieval signal. Identifier-based tie breaking can influence this control because IDs are generated in pattern order. It is not a statistically calibrated random baseline.

The saved notebook has saturated validation and test NDCG across all three trained variants. This supports execution of the shared pipeline, not an advantage for attention or learned reliability. Stronger experiments would use multiple seeds, larger galleries, an untrained/frozen-feature baseline, randomized tie handling analysis, noise/missingness sweeps, and uncertainty intervals.
