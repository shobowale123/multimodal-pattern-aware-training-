from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Mapping, Sequence, Any
import math
import numpy as np
import pandas as pd
from .config import ContractError, MODALITY_ORDER, MODALITY_DIMENSIONS, QUALITY_FIELD_NAMES

@dataclass(frozen=True)
class ModalityIdentity:
    name: str
    version: str
    dimension: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ModalityArtifact:
    name: str
    record_ids: tuple[str, ...]
    matrix: np.ndarray
    quality: np.ndarray
    quality_names: tuple[str, ...]
    content_available: np.ndarray
    source_available: np.ndarray
    identity: ModalityIdentity

    def __post_init__(self) -> None:
        n = len(self.record_ids)
        if self.name not in MODALITY_ORDER:
            raise ContractError(f"Unknown modality: {self.name}")
        if len(set(self.record_ids)) != n or any(not isinstance(v, str) or not v for v in self.record_ids):
            raise ContractError("Artifact record identifiers must be unique nonempty strings")
        if self.matrix.shape != (n, MODALITY_DIMENSIONS[self.name]):
            raise ContractError(f"Invalid {self.name} embedding matrix shape")
        if self.quality.shape != (n, len(self.quality_names)):
            raise ContractError(f"Invalid {self.name} quality matrix shape")
        if len(self.quality_names) != 1:
            raise ContractError("Synthetic artifacts require one quality field")
        for mask in (self.content_available, self.source_available):
            if mask.shape != (n,) or mask.dtype != np.bool_:
                raise ContractError("Artifact availability masks must be Boolean vectors")
        if not np.isfinite(self.matrix).all() or not np.isfinite(self.quality).all():
            raise ContractError("Artifacts must contain only finite numbers")
        if np.any((self.quality < 0) | (self.quality > 1)):
            raise ContractError("Synthetic quality must be in [0, 1]")
        if np.any(self.content_available & ~self.source_available):
            raise ContractError("Content cannot be available without a source")
        if self.identity.name != self.name or self.identity.dimension != self.matrix.shape[1]:
            raise ContractError("Artifact identity does not match its modality or dimension")

    @property
    def index(self) -> Mapping[str, int]:
        return {record_id: i for i, record_id in enumerate(self.record_ids)}


def unit_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype="float32")
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ContractError("Expected a finite two-dimensional matrix")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return np.divide(values, norms, out=np.zeros_like(values), where=norms > 0)


@dataclass(frozen=True)
class UnifiedInputs:
    record_ids: tuple[str, ...]
    matrices: Mapping[str, np.ndarray]
    content_masks: np.ndarray
    source_masks: np.ndarray
    qualities: Mapping[str, np.ndarray]
    quality_names: Mapping[str, tuple[str, ...]]
    identities: Mapping[str, ModalityIdentity]
    pattern_ids: tuple[str, ...]
    splits: tuple[str, ...]

    def __post_init__(self) -> None:
        n = len(self.record_ids)
        if len(set(self.record_ids)) != n:
            raise ContractError("Aligned record identifiers must be unique")
        if len(self.pattern_ids) != n or len(self.splits) != n:
            raise ContractError("Aligned labels and splits must match the record count")
        for masks, width in ((self.content_masks, 5), (self.source_masks, 2)):
            if masks.shape != (n, width) or masks.dtype != np.bool_:
                raise ContractError("Aligned masks have invalid shape or non-Boolean dtype")
        if np.any(self.content_masks[:, 3:] & ~self.source_masks):
            raise ContractError("Structured content cannot exist without a source")
        for mapping in (self.matrices, self.qualities, self.quality_names, self.identities):
            if set(mapping) != set(MODALITY_ORDER):
                raise ContractError("Exactly five named modalities are required")
        for name in MODALITY_ORDER:
            if self.matrices[name].shape != (n, MODALITY_DIMENSIONS[name]):
                raise ContractError(f"Invalid aligned {name} matrix shape")
            if self.qualities[name].shape != (n, len(self.quality_names[name])):
                raise ContractError(f"Invalid aligned {name} quality shape")
            if not np.isfinite(self.matrices[name]).all() or not np.isfinite(self.qualities[name]).all():
                raise ContractError("Aligned values must be finite")
        memberships: dict[str, str] = {}
        for label, split in zip(self.pattern_ids, self.splits, strict=True):
            if split not in {"train", "validation", "test"}:
                raise ContractError(f"Unknown split: {split}")
            if label in memberships and memberships[label] != split:
                raise ContractError("A pattern cannot appear in more than one split")
            memberships[label] = split

    @property
    def eligible(self) -> np.ndarray:
        return self.content_masks.any(axis=1)

    @property
    def modality_mask_code(self) -> np.ndarray:
        weights = (1 << np.arange(len(MODALITY_ORDER), dtype=np.uint8)).reshape(1, -1)
        return (self.content_masks.astype(np.uint8) * weights).sum(axis=1).astype(np.uint8)

    def subset(self, indices: Sequence[int] | np.ndarray) -> "UnifiedInputs":
        selected = np.asarray(indices, dtype=np.int64)
        if selected.ndim != 1 or np.any(selected < 0) or np.any(selected >= len(self.record_ids)):
            raise ContractError("Subset indices must be a vector of valid row positions")
        return UnifiedInputs(
            record_ids=tuple(self.record_ids[i] for i in selected),
            matrices={name: values[selected] for name, values in self.matrices.items()},
            content_masks=self.content_masks[selected],
            source_masks=self.source_masks[selected],
            qualities={name: values[selected] for name, values in self.qualities.items()},
            quality_names=self.quality_names,
            identities=self.identities,
            pattern_ids=tuple(self.pattern_ids[i] for i in selected),
            splits=tuple(self.splits[i] for i in selected),
        )

    def without_quality(self) -> "UnifiedInputs":
        rows = len(self.record_ids)
        return UnifiedInputs(
            record_ids=self.record_ids,
            matrices=self.matrices,
            content_masks=self.content_masks,
            source_masks=self.source_masks,
            qualities={
                name: np.empty((rows, 0), dtype="float32") for name in MODALITY_ORDER
            },
            quality_names={name: () for name in MODALITY_ORDER},
            identities=self.identities,
            pattern_ids=self.pattern_ids,
            splits=self.splits,
        )

    def audit(self) -> Mapping[str, Any]:
        codes, counts = np.unique(self.modality_mask_code, return_counts=True)
        return {
            "rows": len(self.record_ids),
            "eligible_rows": int(self.eligible.sum()),
            "all_missing_rows": int((~self.eligible).sum()),
            "available_by_modality": {
                name: int(self.content_masks[:, pos].sum())
                for pos, name in enumerate(MODALITY_ORDER)
            },
            "source_available": {
                "suspect": int(self.source_masks[:, 0].sum()),
                "weapon": int(self.source_masks[:, 1].sum()),
            },
            "mask_code_counts": {
                str(int(code)): int(count)
                for code, count in zip(codes, counts, strict=True)
            },
        }


def align_modalities(
    master_cohort: pd.DataFrame,
    modality_artifacts: Mapping[str, ModalityArtifact],
) -> UnifiedInputs:
    if set(modality_artifacts) != set(MODALITY_ORDER):
        raise ContractError("Exactly five named artifacts are required")
    required = {"record_id", "pattern_id", "split"}
    if not required <= set(master_cohort.columns):
        raise ContractError(f"Cohort requires columns: {sorted(required)}")
    if master_cohort[list(required)].isna().any().any():
        raise ContractError("Cohort identifiers and splits cannot be missing")

    record_ids = tuple(master_cohort["record_id"])
    row_count = len(record_ids)
    matrices: dict[str, np.ndarray] = {}
    qualities: dict[str, np.ndarray] = {}
    quality_names: dict[str, tuple[str, ...]] = {}
    identities: dict[str, ModalityIdentity] = {}
    content_masks = np.zeros((row_count, 5), dtype=bool)
    source_masks = np.zeros((row_count, 2), dtype=bool)

    for modality_position, name in enumerate(MODALITY_ORDER):
        artifact = modality_artifacts[name]
        if artifact.name != name or set(artifact.record_ids) != set(record_ids):
            raise ContractError(f"{name} artifact does not match cohort identifiers")
        index = artifact.index
        matrix = np.zeros((row_count, MODALITY_DIMENSIONS[name]), dtype="float32")
        quality = np.zeros((row_count, 1), dtype="float32")

        for target_row, record_id in enumerate(record_ids):
            source_row = index[record_id]
            content_masks[target_row, modality_position] = artifact.content_available[source_row]
            quality[target_row] = artifact.quality[source_row]
            if artifact.content_available[source_row]:
                matrix[target_row] = artifact.matrix[source_row]
            if name == "suspect":
                source_masks[target_row, 0] = artifact.source_available[source_row]
            elif name == "weapon":
                source_masks[target_row, 1] = artifact.source_available[source_row]

        matrices[name] = matrix
        qualities[name] = quality
        quality_names[name] = artifact.quality_names
        identities[name] = artifact.identity

    aligned = UnifiedInputs(
        record_ids=record_ids,
        matrices=matrices,
        content_masks=content_masks,
        source_masks=source_masks,
        qualities=qualities,
        quality_names=quality_names,
        identities=identities,
        pattern_ids=tuple(master_cohort["pattern_id"]),
        splits=tuple(master_cohort["split"]),
    )

    if np.any(aligned.content_masks[:, 3:] & ~aligned.source_masks):
        raise ContractError("Suspect/weapon content cannot exist without source")
    return aligned


@dataclass(frozen=True)
class SyntheticDataset:
    """Generated vectors only; this contains no real incident data."""

    cohort: pd.DataFrame
    artifacts: Mapping[str, ModalityArtifact]
    inputs: UnifiedInputs
    all_missing_record_id: str


def make_synthetic_dataset(seed: int = 42) -> SyntheticDataset:
    """Generate the original 12 disjoint patterns and four records per pattern.

    A shared 16-dimensional latent distribution produces all five modalities.
    Pattern-specific latents are independent across the train/validation/test split.
    Random draw order is retained from the source v2 notebook.
    """
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 2**32 - 1:
        raise ContractError("seed must be an integer in [0, 2**32 - 1]")
    pattern_splits = {
        **{f"P-{i:03d}": "train" for i in range(1, 8)},
        **{f"P-{i:03d}": "validation" for i in range(8, 11)},
        **{f"P-{i:03d}": "test" for i in range(11, 13)},
    }
    cohort = pd.DataFrame([
        {"pattern_id": pattern, "record_id": f"record_{4 * pos + member + 1:05d}",
         "split": split, "member_within_pattern": member + 1}
        for pos, (pattern, split) in enumerate(pattern_splits.items())
        for member in range(4)
    ])
    all_missing = str(cohort.loc[cohort["split"].eq("test"), "record_id"].iloc[-1])
    rng = np.random.default_rng(seed)
    latent_dimension = 16
    pattern_latents = {
        pattern: rng.normal(size=latent_dimension).astype("float32")
        for pattern in pattern_splits
    }
    projections = {
        name: rng.normal(scale=1.0 / math.sqrt(latent_dimension),
                         size=(latent_dimension, MODALITY_DIMENSIONS[name])).astype("float32")
        for name in MODALITY_ORDER
    }
    missing_rates = {"location": .04, "time": .06, "narrative": .10,
                     "suspect": .28, "weapon": .34}
    artifacts: dict[str, ModalityArtifact] = {}
    for name in MODALITY_ORDER:
        vectors = np.zeros((len(cohort), MODALITY_DIMENSIONS[name]), dtype="float32")
        quality = np.zeros((len(cohort), 1), dtype="float32")
        content_available = np.zeros(len(cohort), dtype=bool)
        source_available = np.zeros(len(cohort), dtype=bool)
        for row_index, row in cohort.iterrows():
            available = bool(rng.random() >= missing_rates[name])
            if row["record_id"] == all_missing:
                available = False
            if name in {"suspect", "weapon"}:
                source_available[row_index] = available or bool(rng.random() < .45)
            else:
                source_available[row_index] = available
            content_available[row_index] = available
            reliability = float(rng.uniform(.35, 1.00)) if available else 0.0
            quality[row_index, 0] = reliability
            if available:
                latent = pattern_latents[row["pattern_id"]] + rng.normal(
                    scale=.30, size=latent_dimension).astype("float32")
                base = latent @ projections[name]
                noisy = base + rng.normal(scale=.80 * (1.08 - reliability),
                                          size=MODALITY_DIMENSIONS[name]).astype("float32")
                vectors[row_index] = unit_rows(noisy.reshape(1, -1))[0]
        artifacts[name] = ModalityArtifact(
            name=name, record_ids=tuple(cohort["record_id"]), matrix=vectors,
            quality=quality, quality_names=(QUALITY_FIELD_NAMES[name],),
            content_available=content_available, source_available=source_available,
            identity=ModalityIdentity(name, f"synthetic-{name}-v2-demo", MODALITY_DIMENSIONS[name]),
        )
    return SyntheticDataset(cohort, artifacts, align_modalities(cohort, artifacts), all_missing)


def split_inputs(inputs: UnifiedInputs, split: str) -> UnifiedInputs:
    """Select eligible records from one split without mixing pattern memberships."""
    if split not in {"train", "validation", "test"}:
        raise ContractError(f"Unknown split: {split}")
    indices = np.flatnonzero((np.asarray(inputs.splits) == split) & inputs.eligible)
    return inputs.subset(indices)
