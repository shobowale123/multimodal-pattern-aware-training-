"""Load the versioned, non-pickle contract for external precomputed features."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from .config import ContractError, FeatureSchema, MODALITY_ORDER
from .data import FeatureInputs, ModalityArtifact, ModalityIdentity, UnifiedInputs


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ContractError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _referenced_path(root: Path, value: Any, name: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ContractError(f"{name} must be a nonempty file path")
    path = Path(value)
    return path if path.is_absolute() else root / path


def _read_records(path: Path, require_labels: bool) -> tuple[list[dict[str, str]], bool]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames or []
        allowed = {"record_id", "pattern_id", "split", "group_id"}
        if len(columns) != len(set(columns)) or "record_id" not in columns or set(columns) - allowed:
            raise ContractError("CSV requires record_id and only optional pattern_id, split, group_id columns")
        labeled = "pattern_id" in columns and "split" in columns
        if ("pattern_id" in columns) != ("split" in columns):
            raise ContractError("CSV pattern_id and split must be provided together")
        if require_labels and not labeled:
            raise ContractError("Training/evaluation requires pattern_id and split columns")
        if "group_id" in columns and not labeled:
            raise ContractError("group_id requires labeled records with split assignments")
        rows = list(reader)
    if not rows:
        raise ContractError("Dataset must contain at least one record")
    if any(set(row) != set(columns) or any(not isinstance(v, str) or not v.strip() for v in row.values())
           for row in rows):
        raise ContractError("CSV rows must have exactly the named columns and nonempty string values")
    record_ids = [row["record_id"] for row in rows]
    if len(set(record_ids)) != len(record_ids):
        raise ContractError("CSV record IDs must be unique")
    return rows, labeled


def _float32(values: np.ndarray, name: str) -> np.ndarray:
    if values.dtype.kind != "f" or not np.isfinite(values).all():
        raise ContractError(f"{name} must contain finite floating-point values")
    with np.errstate(over="ignore", invalid="ignore"):
        converted = values.astype(np.float32)
    if not np.isfinite(converted).all():
        raise ContractError(f"{name} overflows the float32 model input range")
    return converted


def _read_artifact(path: Path, role: str, schema: FeatureSchema) -> ModalityArtifact:
    expected = {"record_ids", "embeddings", "quality", "content_available", "source_available"}
    with np.load(path, allow_pickle=False) as archive:
        if len(archive.files) != len(expected) or set(archive.files) != expected:
            raise ContractError(f"{role} NPZ must contain exactly {sorted(expected)}")
        arrays = {name: archive[name] for name in expected}
    record_ids = arrays["record_ids"]
    if record_ids.ndim != 1 or record_ids.dtype.kind != "U":
        raise ContractError(f"{role} record_ids must be a one-dimensional Unicode array")
    matrix = _float32(arrays["embeddings"], f"{role} embeddings")
    quality = _float32(arrays["quality"], f"{role} quality")
    artifact = ModalityArtifact(
        name=role,
        record_ids=tuple(record_ids.tolist()),
        matrix=matrix,
        quality=quality,
        quality_names=schema.quality_names[role],
        content_available=arrays["content_available"],
        source_available=arrays["source_available"],
        identity=ModalityIdentity(role, schema.encoder_versions[role], schema.dimensions[role]),
    )
    return artifact


def load_dataset(manifest_path: str | Path, *, require_labels: bool = False) -> FeatureInputs:
    """Read and validate one dataset; modality arrays are joined by exact record IDs.

    Normalization is applied consistently on load. Absent content is zeroed after
    all source values have been validated. No files are modified by this function.
    """
    path = Path(manifest_path)
    try:
        with path.open(encoding="utf-8") as handle:
            manifest = json.load(handle, object_pairs_hook=_unique_object)
        expected = {"schema_version", "schema", "records", "modalities"}
        if not isinstance(manifest, dict) or set(manifest) != expected:
            raise ContractError(f"Manifest requires exactly: {sorted(expected)}")
        if type(manifest["schema_version"]) is not int or manifest["schema_version"] != 1:
            raise ContractError("Unsupported manifest schema_version; expected 1")
        schema = FeatureSchema.from_dict(manifest["schema"])
        paths = manifest["modalities"]
        if not isinstance(paths, dict) or set(paths) != set(MODALITY_ORDER):
            raise ContractError("Manifest must name exactly the five modality files")
        records, labeled = _read_records(
            _referenced_path(path.parent, manifest["records"], "records"), require_labels,
        )
        record_ids = tuple(row["record_id"] for row in records)
        matrices: dict[str, np.ndarray] = {}
        qualities: dict[str, np.ndarray] = {}
        identities: dict[str, ModalityIdentity] = {}
        content_masks = np.zeros((len(record_ids), 5), dtype=bool)
        source_masks = np.zeros((len(record_ids), 2), dtype=bool)
        for position, role in enumerate(MODALITY_ORDER):
            artifact = _read_artifact(_referenced_path(path.parent, paths[role], role), role, schema)
            if set(artifact.record_ids) != set(record_ids):
                raise ContractError(f"{role} IDs must exactly match CSV IDs (no missing or extra rows)")
            index = artifact.index
            order = np.asarray([index[record_id] for record_id in record_ids])
            matrix = artifact.matrix[order].copy()
            available = artifact.content_available[order]
            matrix[~available] = 0
            if schema.normalization == "l2":
                # Float64 norms avoid overflow for large but finite float32 values.
                norms = np.linalg.norm(matrix.astype(np.float64), axis=1, keepdims=True)
                matrix = np.divide(matrix, norms, out=np.zeros_like(matrix), where=norms > 0)
            matrices[role] = matrix
            qualities[role] = artifact.quality[order]
            identities[role] = artifact.identity
            content_masks[:, position] = available
            if role in {"suspect", "weapon"}:
                source_masks[:, position - 3] = artifact.source_available[order]
        arguments: dict[str, Any] = dict(
            record_ids=record_ids, matrices=matrices, qualities=qualities,
            content_masks=content_masks, source_masks=source_masks,
            identities=identities, quality_names=schema.quality_names, schema_metadata=schema,
        )
        if labeled:
            return UnifiedInputs(
                **arguments,
                pattern_ids=tuple(row["pattern_id"] for row in records),
                splits=tuple(row["split"] for row in records),
                group_ids=tuple(row["group_id"] for row in records) if "group_id" in records[0] else None,
            )
        return FeatureInputs(**arguments)
    except ContractError:
        raise
    except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
        raise ContractError(f"Invalid dataset at {path}: {exc}") from exc
