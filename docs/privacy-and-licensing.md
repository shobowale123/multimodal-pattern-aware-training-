# Public release scope and licensing

The public demo generates synthetic vectors locally. It needs no credentials, private datasets, raw reports, databases, external model downloads, or network calls during training.

The public refactor replaces domain-specific record identifiers with generic synthetic record IDs and removes notebook wording that refers to internal implementations or approved production contracts. Original raw notebook outputs, execution metadata, local paths, embedded HTML, and historical images are not copied into this repository. Retained result artifacts are regenerated from the public package.

The modality concepts and vector dimensions are retained to preserve the technical experiment. The original `suspect` and `weapon` modality keys describe simulated structured channels, not real individuals or incidents. See the [source assessment](source-assessment.md) for the exact migration scope and missing historical source.

## License

The repository includes the [MIT license](../LICENSE) for the code and included generated examples. It does not relicense Python, PyTorch, NumPy, pandas, Matplotlib, Jupyter, or other dependencies; each retains its own license. The standard template is documented by [Choose a License](https://choosealicense.com/licenses/mit/).

The code was supplied for this portfolio adaptation without an existing license header in the inspected notebooks. A filename or access to a notebook does not establish ownership. Before public release, the owner must confirm that they may distribute the source-derived implementation and apply MIT; no employer rights or third-party permissions have been inferred.

The result is an educational retrieval simulation. The source domain should not be interpreted as validation for decisions about real people. No real-world operational performance, fairness, or deployment suitability was evaluated.
