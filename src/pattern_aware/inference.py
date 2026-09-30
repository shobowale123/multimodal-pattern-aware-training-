"""Saved-model encoding and held-out evaluation for precomputed features."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, TYPE_CHECKING

import numpy as np

from .config import ContractError, FeatureSchema, FusionConfig
from .retrieval import _identifiers, _positive_integer, _write_exclusive_outputs

if TYPE_CHECKING:
    from .checkpoints import LoadedCheckpoint
    from .data import FeatureInputs
    from .evaluation import RetrievalEvaluation


@dataclass(frozen=True)
class EmbeddingArtifact:
    record_ids: tuple[str, ...]
    vectors: np.ndarray
    available: np.ndarray
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        matrix = self.vectors
        if (not isinstance(matrix, np.ndarray) or matrix.ndim != 2
                or matrix.dtype != np.dtype("float32") or matrix.shape[1] < 1
                or not np.isfinite(matrix).all()):
            raise ContractError("Saved embeddings must be a finite float32 matrix with positive width")
        _identifiers(self.record_ids, len(matrix), "record_ids")
        if (not isinstance(self.available, np.ndarray) or self.available.shape != (len(matrix),)
                or self.available.dtype != np.bool_):
            raise ContractError("Saved embedding availability must be a matching Boolean vector")
        norms = np.linalg.norm(matrix[self.available], axis=1)
        if not np.allclose(norms, 1.0, rtol=1e-5, atol=1e-6):
            raise ContractError("Available saved embeddings must have unit L2 norm")
        if np.any(matrix[~self.available] != 0):
            raise ContractError("Unavailable saved embeddings must be zero")
        required = {"format_version", "feature_schema", "fusion_config", "weights_sha256", "variant"}
        if not isinstance(self.metadata, Mapping) or set(self.metadata) != required:
            raise ContractError(f"Embedding metadata requires exactly {sorted(required)}")
        if type(self.metadata["format_version"]) is not int or self.metadata["format_version"] != 1:
            raise ContractError("Unsupported embedding format_version")
        schema = FeatureSchema.from_dict(self.metadata["feature_schema"])
        if self.metadata["variant"] not in {"v1", "v11", "v2"}:
            raise ContractError("Invalid embedding model variant")
        try:
            config = FusionConfig(**self.metadata["fusion_config"])
        except (ValueError, TypeError, KeyError) as exc:
            raise ContractError(f"Invalid embedding fusion configuration: {exc}") from exc
        if (dict(config.input_dimensions) != dict(schema.dimensions)
                or config.output_dimension != matrix.shape[1]
                or config.quality_dimension != (0 if self.metadata["variant"] == "v1" else 1)):
            raise ContractError("Embedding architecture, schema, variant, and vector width must agree")
        digest = self.metadata["weights_sha256"]
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise ContractError("Embedding weights_sha256 must be a lowercase SHA256 digest")


def save_embeddings(artifact: EmbeddingArtifact, path: str | Path) -> None:
    """Write version 1 NPZ without pickle/object arrays; never replace an existing file."""
    artifact.__post_init__()
    metadata = json.dumps(dict(artifact.metadata), sort_keys=True, allow_nan=False)
    path = Path(path)
    if path.exists():
        raise ContractError("Embedding output already exists; choose a new output path")
    path.parent.mkdir(parents=True, exist_ok=True)
    created = False
    try:
        with path.open("xb") as handle:
            created = True
            np.savez_compressed(
                handle, record_ids=np.asarray(artifact.record_ids, dtype=str),
                vectors=artifact.vectors, available=artifact.available,
                metadata=np.asarray(metadata, dtype=str),
            )
    except Exception:
        if created:
            path.unlink(missing_ok=True)
        raise


def load_embeddings(path: str | Path) -> EmbeddingArtifact:
    """Read and fully validate an embedding NPZ; pickle is always disabled."""
    try:
        with np.load(Path(path), allow_pickle=False) as archive:
            expected = {"record_ids", "vectors", "available", "metadata"}
            if set(archive.files) != expected or len(archive.files) != len(expected):
                raise ContractError(f"Embedding NPZ requires exactly {sorted(expected)}")
            ids = archive["record_ids"]
            metadata_array = archive["metadata"]
            if ids.ndim != 1 or ids.dtype.kind != "U":
                raise ContractError("Saved record_ids must be a one-dimensional Unicode array")
            if metadata_array.shape != () or metadata_array.dtype.kind != "U":
                raise ContractError("Embedding metadata must be a scalar Unicode JSON string")
            metadata = json.loads(str(metadata_array.item()))
            return EmbeddingArtifact(
                tuple(ids.tolist()), archive["vectors"], archive["available"], metadata
            )
    except ContractError:
        raise
    except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
        raise ContractError(f"Cannot read embedding artifact: {exc}") from exc


def encode_dataset(
    checkpoint: LoadedCheckpoint,
    inputs: FeatureInputs,
    *,
    batch_size: int = 32,
) -> EmbeddingArtifact:
    """Encode labels-optional inputs only when every feature schema field matches."""
    from .training import encode_all

    batch_size = _positive_integer("batch_size", batch_size)
    # The dataclass is frozen, but its numpy arrays and mappings remain mutable.
    inputs.__post_init__()
    if inputs.schema != checkpoint.schema:
        raise ContractError("Inference feature schema must exactly match the checkpoint schema")
    variant = checkpoint.metadata["variant"]
    if variant != "v1" and any(values.shape[1] != 1 for values in inputs.qualities.values()):
        raise ContractError("Quality-gated inference requires one quality field per modality")
    model_inputs = inputs.without_quality() if variant == "v1" else inputs
    if len(inputs.record_ids):
        vectors = np.asarray(encode_all(checkpoint.model, model_inputs, batch_size), dtype=np.float32)
    else:
        vectors = np.empty((0, checkpoint.model.config.output_dimension), dtype=np.float32)
    available = inputs.eligible.copy()
    vectors[~available] = 0.0
    return EmbeddingArtifact(
        tuple(inputs.record_ids), vectors, available,
        {"format_version": 1, "feature_schema": checkpoint.schema.to_dict(),
         "fusion_config": dict(checkpoint.metadata["fusion_config"]),
         "weights_sha256": checkpoint.metadata["weights_sha256"], "variant": variant},
    )


def encode_from_manifest(
    checkpoint: str | Path,
    data: str | Path,
    output: str | Path,
    *,
    batch_size: int = 32,
) -> EmbeddingArtifact:
    from .checkpoints import load_checkpoint
    from .io import load_dataset

    if Path(output).exists():
        raise ContractError("Embedding output already exists; choose a new output path")
    artifact = encode_dataset(
        load_checkpoint(checkpoint), load_dataset(data, require_labels=False), batch_size=batch_size
    )
    save_embeddings(artifact, output)
    return artifact


def _check_evaluation_provenance(checkpoint: LoadedCheckpoint, inputs: Any, split: str) -> None:
    details = checkpoint.metadata.get("provenance", {})
    provenance = details.get("split_hashes")
    if not isinstance(provenance, Mapping):
        raise ContractError("Checkpoint lacks split provenance required for evaluation")
    if details.get("group_ids_provided") and getattr(inputs, "group_ids", None) is None:
        raise ContractError("Evaluation group_ids are required by the checkpoint provenance")
    protected_splits = ("train", "validation") if split == "test" else ("train",)
    for protected in protected_splits:
        saved = provenance.get(protected)
        if not isinstance(saved, Mapping):
            raise ContractError(f"Checkpoint lacks {protected} provenance required for evaluation")
        for field in ("record_ids", "pattern_ids", "group_ids"):
            known = saved.get(field)
            if field != "group_ids" and not isinstance(known, list):
                raise ContractError(f"Checkpoint lacks {protected} {field} hashes")
            current = getattr(inputs, field, None)
            if field == "group_ids" and known and current is None:
                raise ContractError("Evaluation group_ids are required by the checkpoint provenance")
            if current is None or not known:
                continue
            current_hashes = {hashlib.sha256(value.encode("utf-8")).hexdigest() for value in current}
            if current_hashes.intersection(known):
                raise ContractError(f"Evaluation {field} overlap checkpoint {protected} data")


def evaluate_checkpoint(
    checkpoint: str | Path,
    manifest: str | Path,
    split: str,
    output: str | Path,
    *,
    k_values: Iterable[int] = (1, 5, 10, 30),
    batch_size: int = 32,
    query_chunk_size: int = 128,
    gallery_chunk_size: int = 1024,
) -> RetrievalEvaluation:
    """Evaluate a fixed checkpoint without training, rejecting fitting/test overlap.

    Validation results are diagnostics on the selection split. Test evaluation
    additionally excludes identities/patterns/groups seen during validation.
    """
    from .checkpoints import load_checkpoint
    from .evaluation import RetrievalEvaluation, evaluate_retrieval
    from .io import load_dataset

    if split not in {"validation", "test"}:
        raise ContractError("Evaluation split must be validation or test")
    output = Path(output)
    per_query_path = output.with_suffix(".per_query.csv")
    per_pattern_path = output.with_suffix(".per_pattern.csv")
    paths = (output, per_query_path, per_pattern_path)
    if len({path.resolve() for path in paths}) != len(paths) or any(path.exists() for path in paths):
        raise ContractError("Evaluation outputs must be distinct new paths")
    loaded = load_checkpoint(checkpoint)
    all_inputs = load_dataset(manifest, require_labels=True)
    selected = np.flatnonzero(np.asarray(all_inputs.splits) == split)
    if not len(selected):
        raise ContractError(f"Dataset has no {split} records")
    inputs = all_inputs.subset(selected)
    _check_evaluation_provenance(loaded, inputs, split)
    artifact = encode_dataset(loaded, inputs, batch_size=batch_size)
    result = evaluate_retrieval(
        artifact.vectors, inputs.pattern_ids, record_ids=artifact.record_ids,
        available=artifact.available, k_values=k_values,
        query_chunk_size=query_chunk_size, gallery_chunk_size=gallery_chunk_size,
    )
    summary = {
        **result.summary, "split": split,
        "evaluation_role": "held_out_test" if split == "test" else "selection_split_diagnostic",
        "total_records": len(artifact.record_ids),
        "unavailable_records_excluded": int((~artifact.available).sum()),
        "singleton_query_records_excluded": int(artifact.available.sum()) - len(result.per_query),
        "weights_sha256": loaded.metadata["weights_sha256"],
    }
    result = RetrievalEvaluation(summary, result.per_query, result.per_pattern)
    _write_exclusive_outputs({
        output: json.dumps(summary, indent=2, allow_nan=False) + "\n",
        per_query_path: result.per_query.to_csv(index=False),
        per_pattern_path: result.per_pattern.to_csv(index=False),
    })
    return result
