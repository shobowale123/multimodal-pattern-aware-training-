# Contributing

Create a Python 3.11 or 3.12 environment using the README, then run `python -m pytest` before proposing changes. Keep reusable implementation in `src/pattern_aware`, experiment choices in `configs`, and explanations in `docs`.

Use only synthetic examples. Do not commit credentials, private data, raw records, local paths, or downloaded model checkpoints. Keep the demonstration notebook unexecuted; regenerate the separate results notebook and example metrics together when changing the default experiment.

Changes to sampling, masks, selection, or ranking need tests for their mathematical invariants. Explain any resulting metric change and include the effective configuration and environment. Never select a model using the test split.
