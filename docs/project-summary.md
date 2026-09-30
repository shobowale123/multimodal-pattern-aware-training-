# Project summary

This project implements CPU multimodal retrieval training in PyTorch. It combines five precomputed representations using global gates, per-record quality gates, or five-token self-attention followed by quality gates. The default output is a normalized 256-dimensional embedding; own-data widths and architecture dimensions are configurable.

Training constructs pattern-balanced batches with a fixed K distinct records per selected pattern. A multi-positive supervised contrastive objective learns from all non-self batch comparisons; the lower-level loss also offers an optional comparison mask. Content masks and structured source-availability bits represent missing modalities explicitly. Adaptive K is not implemented.

The public own-data workflow validates a versioned manifest and ID-indexed modality files, enforces declared split/group separation, selects weights using validation pattern-macro NDCG, and saves a self-describing model bundle. A fresh process can reconstruct the model, encode unlabeled records, retrieve from a separate gallery in bounded score chunks, and evaluate a held-out split. IDs, availability, feature identities, and selected-weight identity travel with encoded outputs.

The original synthetic comparison remains reproducible through its existing command and notebooks, including gate/attention diagnostics and a mask-only control. CI tests mathematical invariants, persistence/reload behavior, retrieval, and an installed-wheel lifecycle on external-format synthetic files.

The 48-record original cohort and small acceptance fixture are deliberately limited. They demonstrate working optimization and software contracts, not an advantage for attention, performance on private data, or production readiness. Raw encoders, adaptive sampling, arbitrary modality counts, GPU/distributed training, exact resume, and a serving/vector-database layer remain outside scope.
