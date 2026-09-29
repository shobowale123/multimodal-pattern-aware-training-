# Project summary

This project implements a multimodal retrieval-training simulation in PyTorch. It combines five frozen synthetic representations into a normalized 256-dimensional embedding and compares global modality gates, per-record quality gates, and self-attention across modality tokens.

The training system constructs pattern-balanced batches so that every anchor has a related positive. A masked multi-positive supervised contrastive objective learns to cluster related records while separating allowed negative comparisons. Missing-modality masks and separate source-availability bits make the input contract explicit.

Evaluation separates patterns across training, validation, and test. Checkpoints are selected using validation pattern-macro NDCG@30, then evaluated on the held-out test patterns. Recall, NDCG, and average precision are reported alongside per-query and per-pattern results. Gate diagnostics, attention matrices, and a mask-only negative control help inspect model behavior.

The public package replaces notebook-only execution with reusable modules, validated JSON settings, a command-line entry point, clean demonstration and executed-results notebooks, regenerated example artifacts, and focused tests. It runs on CPU without private data or external services.

The 48-record synthetic cohort is intentionally small and the default retrieval task saturates. Results demonstrate the implementation and evaluation protocol; they do not establish real-world performance or a superior fusion architecture.

**Portfolio summary:** Built a reproducible PyTorch multimodal retrieval simulation with custom pattern-balanced sampling, quality-aware fusion, masked supervised contrastive learning, and pattern-disjoint evaluation. Refactored exploratory notebooks into a tested package with CPU execution and reproducible diagnostics.
