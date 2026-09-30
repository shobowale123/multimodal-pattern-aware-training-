"""Versioned, self-describing inference bundles (not interrupted-training snapshots)."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import io
import json
import math
import os
from pathlib import Path
import pickle
import shutil
import tempfile
from typing import Any, Iterable, Mapping

import torch
from torch import nn

from .config import ContractError, FeatureSchema, FusionConfig, TrainingConfig, _positive_integer
from .models import FiveModalityConcatFusionEncoder, FiveTokenSelfAttentionFusionEncoder

FORMAT_VERSION = 1


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode("utf-8")).hexdigest()


def hash_identifiers(values: Iterable[str]) -> list[str]:
    """Comparable identifiers for split-overlap checks; these are not anonymization."""
    return sorted({hashlib.sha256(value.encode("utf-8")).hexdigest() for value in values})


def create_model(variant: str, config: FusionConfig) -> nn.Module:
    if variant not in {"v1", "v11", "v2"}:
        raise ContractError("variant must be v1, v11, or v2")
    if config.quality_dimension != (0 if variant == "v1" else 1):
        raise ContractError("Architecture quality dimension does not match variant")
    model_type = FiveTokenSelfAttentionFusionEncoder if variant == "v2" else FiveModalityConcatFusionEncoder
    return model_type(config).cpu()


@dataclass(frozen=True)
class LoadedCheckpoint:
    model: nn.Module
    schema: FeatureSchema
    metadata: Mapping[str, Any]


def _validate_state(state: Any, expected: Mapping[str, torch.Tensor]) -> None:
    if not isinstance(state, dict) or set(state) != set(expected):
        raise ContractError("Checkpoint state keys do not match the architecture")
    for key, reference in expected.items():
        value = state[key]
        if (type(value) is not torch.Tensor or value.layout != torch.strided
                or value.shape != reference.shape or value.dtype != reference.dtype):
            raise ContractError(f"Invalid checkpoint tensor shape or dtype: {key}")
        if not torch.isfinite(value).all():
            raise ContractError(f"Checkpoint tensor must be finite: {key}")


def _parse_metadata(metadata: Any) -> tuple[FeatureSchema, FusionConfig]:
    required = {"format_version", "variant", "fusion_config", "feature_schema",
                "training_config", "selection", "provenance", "weights_sha256", "schema_sha256"}
    if not isinstance(metadata, dict) or set(metadata) != required:
        raise ContractError("Checkpoint metadata has missing or unsupported fields")
    if type(metadata["format_version"]) is not int or metadata["format_version"] != FORMAT_VERSION:
        raise ContractError("Unsupported checkpoint format_version")
    for name in ("weights_sha256", "schema_sha256"):
        value = metadata[name]
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ContractError(f"Invalid checkpoint {name}")
    try:
        schema = FeatureSchema.from_dict(metadata["feature_schema"])
        config = FusionConfig(**metadata["fusion_config"])
        training_config = TrainingConfig(**metadata["training_config"])
    except (TypeError, ValueError, KeyError, AttributeError) as exc:
        raise ContractError(f"Invalid checkpoint configuration: {exc}") from exc
    if canonical_sha256(schema.to_dict()) != metadata["schema_sha256"]:
        raise ContractError("Checkpoint feature schema digest mismatch")
    if dict(config.input_dimensions) != dict(schema.dimensions):
        raise ContractError("Checkpoint architecture input dimensions do not match feature schema")
    variant = metadata["variant"]
    if variant not in {"v1", "v11", "v2"} or config.quality_dimension != (0 if variant == "v1" else 1):
        raise ContractError("Checkpoint variant and architecture are incompatible")
    if variant != "v1" and any(len(names) != 1 for names in schema.quality_names.values()):
        raise ContractError("Quality-gated checkpoints require one quality field per modality")
    selection = metadata["selection"]
    if not isinstance(selection, dict) or set(selection) != {"epoch", "metric_name", "metric_value", "k"}:
        raise ContractError("Invalid checkpoint selection metadata")
    _positive_integer("selected epoch", selection["epoch"])
    if selection["epoch"] > training_config.epochs:
        raise ContractError("Selected checkpoint epoch exceeds configured training epochs")
    _positive_integer("selection_k", selection["k"])
    metric = selection["metric_value"]
    if (isinstance(metric, bool) or not isinstance(metric, (float, int))
            or not math.isfinite(metric) or not 0 <= metric <= 1):
        raise ContractError("Invalid checkpoint validation metric")
    if selection["metric_name"] != f"validation_pattern_macro_ndcg_at_{selection['k']}":
        raise ContractError("Checkpoint selection metric name does not match cutoff")
    provenance = metadata["provenance"]
    if not isinstance(provenance, dict) or not isinstance(provenance.get("split_hashes"), dict):
        raise ContractError("Checkpoint requires split provenance")
    if set(provenance["split_hashes"]) != {"train", "validation", "test"}:
        raise ContractError("Checkpoint requires all three split provenance entries")
    if type(provenance.get("group_ids_provided")) is not bool:
        raise ContractError("Checkpoint group provenance must be Boolean")
    for split in provenance["split_hashes"].values():
        if not isinstance(split, dict) or set(split) != {"record_ids", "pattern_ids", "group_ids"}:
            raise ContractError("Invalid split provenance")
        for values in split.values():
            if (not isinstance(values, list) or any(not isinstance(v, str) or len(v) != 64
                    or any(c not in "0123456789abcdef" for c in v) for v in values)
                    or len(set(values)) != len(values)):
                raise ContractError("Invalid identifier hashes in split provenance")
    for kind in ("record_ids", "pattern_ids", "group_ids"):
        seen: set[str] = set()
        for split in ("train", "validation", "test"):
            hashes = set(provenance["split_hashes"][split][kind])
            if hashes & seen:
                raise ContractError(f"Checkpoint {kind} provenance overlaps between splits")
            seen.update(hashes)
        if kind != "group_ids":
            if any(not provenance["split_hashes"][split][kind] for split in ("train", "validation")):
                raise ContractError("Checkpoint train/validation provenance cannot be empty")
        elif not provenance["group_ids_provided"] and seen:
            raise ContractError("Checkpoint group hashes require group_ids_provided")
    return schema, config


def save_checkpoint(
    path: str | Path, model: nn.Module, *, variant: str, schema: FeatureSchema,
    training_config: TrainingConfig, selected_epoch: int, validation_metric: float,
    selection_k: int, provenance: Mapping[str, Any],
) -> Path:
    """Publish a complete bundle; an existing destination is always an error.

    A sibling exclusive lock coordinates writers using this API. The final directory
    appears only after both files are complete. No optimizer or RNG state is saved.
    """
    destination = Path(path)
    if destination.exists():
        raise ContractError(f"Checkpoint output already exists: {destination}")
    if not destination.parent.is_dir():
        raise ContractError("Checkpoint parent directory must already exist")
    lock = destination.with_name(destination.name + ".lock")
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise ContractError(f"Checkpoint output is already being written: {destination}") from exc
    os.close(descriptor)
    staging: Path | None = None
    try:
        if destination.exists():
            raise ContractError(f"Checkpoint output already exists: {destination}")
        staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
        state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        _validate_state(state, model.state_dict())
        weights = staging / "weights.pt"
        torch.save(state, weights)
        metadata = {
            "format_version": FORMAT_VERSION, "variant": variant,
            "fusion_config": asdict(model.config), "feature_schema": schema.to_dict(),
            "training_config": asdict(training_config),
            "selection": {"epoch": selected_epoch, "metric_name": f"validation_pattern_macro_ndcg_at_{selection_k}",
                          "metric_value": validation_metric, "k": selection_k},
            "provenance": dict(provenance),
            "weights_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
            "schema_sha256": canonical_sha256(schema.to_dict()),
        }
        _parse_metadata(metadata)
        (staging / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True,
                                                           allow_nan=False) + "\n", encoding="utf-8")
        # Verify architecture reconstruction, not just the in-memory model's own keys.
        load_checkpoint(staging)
        if destination.exists():
            raise ContractError(f"Checkpoint output already exists: {destination}")
        staging.rename(destination)
        staging = None
        return destination
    finally:
        if staging is not None:
            shutil.rmtree(staging)
        lock.unlink()


def load_checkpoint(path: str | Path) -> LoadedCheckpoint:
    """Reconstruct only known model classes with CPU, weights-only deserialization.

    Hashes detect accidental corruption; they do not authenticate a bundle's author.
    Load bundles only from trusted sources; tensor sizes can still exhaust memory.
    """
    directory = Path(path)
    try:
        metadata_path = directory / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        schema, config = _parse_metadata(metadata)
        payload = (directory / "weights.pt").read_bytes()
        if hashlib.sha256(payload).hexdigest() != metadata["weights_sha256"]:
            raise ContractError("Checkpoint weights digest mismatch")
        state = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)
        # Loading must not perturb the caller's subsequent model initialization.
        with torch.random.fork_rng(devices=[]):
            model = create_model(metadata["variant"], config)
        _validate_state(state, model.state_dict())
        model.load_state_dict(state, strict=True)
        model.eval()
        return LoadedCheckpoint(model, schema, metadata)
    except ContractError:
        raise
    except (OSError, ValueError, TypeError, KeyError, RuntimeError, EOFError, pickle.UnpicklingError) as exc:
        raise ContractError(f"Cannot load checkpoint: {exc}") from exc
