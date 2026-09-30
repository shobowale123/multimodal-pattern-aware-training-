# Public release scope and licensing

The public examples and acceptance fixture generate synthetic vectors locally. They need no credentials, private datasets, raw reports, operational databases, pretrained model downloads, or network calls during training and inference. The own-data workflow reads files supplied by the user; the package does not upload them.

The original public refactor replaced domain-specific record identifiers with generic synthetic IDs and omitted original raw notebook outputs, execution metadata, local paths, embedded HTML, and historical images. Retained reference artifacts were regenerated from the public package. The `suspect` and `weapon` keys preserve the two original structured channel roles; included values describe no real individuals or incidents. See [source assessment](source-assessment.md).

## User-supplied data and weights

Use only data and upstream embeddings you are entitled to process. Encoder revisions, feature schemas, labels, IDs, and group boundaries are supplied by the user. The contract validates their format and declared consistency; it cannot establish source rights, eliminate semantic duplication, or prove that upstream preprocessing avoided held-out information.

Keep private source files, learned weights, and derived embeddings outside public commits and logs. Model metadata includes schemas, encoder identities, training settings, and hashed split identities used to check evaluation overlap. Those hashes support identity comparisons; they are not a guarantee of anonymization. Run summaries contain aggregate counts, while retrieval/evaluation outputs contain record identifiers.

The public CI upload allowlist contains only generated synthetic summaries/configurations, histories, rankings, reports, test output, and an environment listing. It excludes raw fixture arrays, encoded arrays, and trained weights. The repository ignores `artifacts/`, `runs/`, and `data/`; configure your own version-control exclusions before using other paths for private data.

## License

The [MIT license](../LICENSE) covers the repository's code and included generated examples. It does not relicense user data, upstream encoders, trained models obtained elsewhere, or dependencies. PyTorch, NumPy, pandas, Matplotlib, Jupyter, and other dependencies retain their own licenses.
