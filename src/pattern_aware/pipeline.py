"""Own-data CPU fitting and validation-selected inference-bundle lifecycle."""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field, replace
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import platform
import random
from typing import Any, Mapping

import numpy as np
import torch

from .checkpoints import create_model, hash_identifiers, save_checkpoint
from .config import (ContractError, FusionConfig, TrainingConfig, V1_CONFIG, V11_CONFIG,
                     V2_CONFIG, _positive_integer)
from .data import UnifiedInputs
from .io import load_dataset
from .sampling import PatternBalancedBatchSampler
from .training import TrainingRun, train_and_select

_ARCHITECTURE_FIELDS = {"adapter_dimension", "fusion_hidden_dimension", "output_dimension",
                        "dropout", "attention_heads", "attention_feedforward_dimension"}


@dataclass(frozen=True)
class RunConfig:
    training: TrainingConfig = field(default_factory=TrainingConfig)
    variant: str = "v2"
    architecture: Mapping[str, Any] = field(default_factory=dict)
    selection_k: int = 30
    cpu_threads: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.training, TrainingConfig):
            raise ContractError("training must be a TrainingConfig")
        if not isinstance(self.variant, str) or self.variant not in {"v1", "v11", "v2"}:
            raise ContractError("variant must be v1, v11, or v2")
        if not isinstance(self.architecture, Mapping) or set(self.architecture) - _ARCHITECTURE_FIELDS:
            raise ContractError(f"architecture accepts only {sorted(_ARCHITECTURE_FIELDS)}")
        _positive_integer("selection_k", self.selection_k)
        _positive_integer("cpu_threads", self.cpu_threads)
        self.fusion_config()

    def fusion_config(self, inputs: UnifiedInputs | None = None) -> FusionConfig:
        base = {"v1": V1_CONFIG, "v11": V11_CONFIG, "v2": V2_CONFIG}[self.variant]
        overrides = dict(self.architecture)
        if inputs is not None:
            overrides["input_dimensions"] = inputs.schema.dimensions
        return replace(base, **overrides)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "RunConfig":
        if not isinstance(values, Mapping):
            raise ContractError("Training run configuration must be a JSON object")
        try:
            arguments = dict(values)
            arguments["training"] = TrainingConfig(**arguments.get("training", {}))
            return cls(**arguments)
        except (TypeError, ValueError) as exc:
            raise ContractError(f"Invalid training run configuration: {exc}") from exc

    @classmethod
    def from_json(cls, path: str | Path) -> "RunConfig":
        try:
            return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
        except (OSError, ValueError) as exc:
            raise ContractError(f"Cannot read training run configuration: {exc}") from exc


def split_provenance(inputs: UnifiedInputs) -> dict[str, Any]:
    """Record comparable split identities without saving input vectors or raw labels."""
    split_hashes: dict[str, Any] = {}
    for split in ("train", "validation", "test"):
        positions = [i for i, value in enumerate(inputs.splits) if value == split]
        split_hashes[split] = {
            "record_ids": hash_identifiers(inputs.record_ids[i] for i in positions),
            "pattern_ids": hash_identifiers(inputs.pattern_ids[i] for i in positions),
            "group_ids": hash_identifiers(inputs.group_ids[i] for i in positions)
            if inputs.group_ids is not None else [],
        }
    try:
        package_version = version("multimodal-pattern-aware-training")
    except PackageNotFoundError:
        package_version = "uninstalled-source"
    return {"split_hashes": split_hashes, "group_ids_provided": inputs.group_ids is not None,
            "package_version": package_version, "python_version": platform.python_version(),
            "torch_version": str(torch.__version__), "numpy_version": np.__version__,
            "device": "cpu", "checkpoint_purpose": "inference; not interrupted-training resume"}


def _preflight(inputs: UnifiedInputs, config: RunConfig) -> tuple[UnifiedInputs, UnifiedInputs, dict[str, Any]]:
    if not isinstance(inputs, UnifiedInputs):
        raise ContractError("Training requires labeled UnifiedInputs")
    # Revalidate even for programmatic callers because NumPy arrays are mutable.
    inputs.__post_init__()
    if config.variant != "v1" and any(len(names) != 1 for names in inputs.quality_names.values()):
        raise ContractError("V1.1 and V2 require one declared quality field per modality")
    split_sets = {split: {value for value, partition in zip(inputs.record_ids, inputs.splits, strict=True)
                          if partition == split} for split in ("train", "validation", "test")}
    if not split_sets["train"] or not split_sets["validation"]:
        raise ContractError("Training requires nonempty train and validation splits")
    prepared = inputs.without_quality() if config.variant == "v1" else inputs
    # Keep unavailable rows in the audit; sampler and retrieval exclude them explicitly.
    train = prepared.subset(np.flatnonzero(np.asarray(prepared.splits) == "train"))
    validation = prepared.subset(np.flatnonzero(np.asarray(prepared.splits) == "validation"))
    sampler = PatternBalancedBatchSampler(train.pattern_ids, train.eligible,
                                         patterns_per_batch=config.training.patterns_per_batch,
                                         members_per_pattern=config.training.members_per_pattern,
                                         seed=config.training.seed)
    counts = Counter(pattern for pattern, eligible in zip(validation.pattern_ids, validation.eligible, strict=True)
                     if eligible)
    paired_patterns = sum(count >= 2 for count in counts.values())
    if paired_patterns < 2:
        raise ContractError("Validation requires at least two patterns with two available records each")
    excluded_patterns = sorted(set(train.pattern_ids) - {
        label for label, count in Counter(label for label, available in
            zip(train.pattern_ids, train.eligible, strict=True) if available).items()
        if count >= config.training.members_per_pattern})
    audit = {**asdict(sampler.audit), "all_missing_rows": int((~train.eligible).sum()),
             "excluded_pattern_hashes": hash_identifiers(excluded_patterns)}
    return train, validation, audit


def train_dataset(inputs: UnifiedInputs, config: RunConfig, output: str | Path) -> TrainingRun:
    """Fit train only, select with validation only, and save the selected CPU model.

    Test rows are validated for split integrity and recorded in provenance, but never
    encoded, ranked, or used to choose the model. Evaluate them in a separate call.
    """
    if not isinstance(config, RunConfig):
        raise ContractError("config must be a RunConfig")
    train, validation, training_audit = _preflight(inputs, config)
    architecture = config.fusion_config(inputs)
    provenance = split_provenance(inputs)
    destination = Path(output)
    try:
        destination.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise ContractError(f"Training output already exists: {destination}") from exc
    (destination / "config.json").write_text(json.dumps(config.to_dict(), indent=2, sort_keys=True,
                                                        allow_nan=False) + "\n", encoding="utf-8")
    python_state, numpy_state = random.getstate(), np.random.get_state()
    torch_state = torch.random.get_rng_state()
    previous_threads = torch.get_num_threads()
    previous_determinism = torch.are_deterministic_algorithms_enabled()
    previous_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.set_num_threads(config.cpu_threads)
        run = train_and_select(config.variant, lambda: create_model(config.variant, architecture),
                               train, validation, config.training, selection_k=config.selection_k)
        save_checkpoint(destination / "best", run.model, variant=config.variant, schema=inputs.schema,
                        training_config=config.training, selected_epoch=run.selected_epoch,
                        validation_metric=run.best_validation_metric, selection_k=config.selection_k,
                        provenance=provenance)
        run.history.to_csv(destination / "history.csv", index=False)
        summary = {"variant": config.variant, "selected_epoch": run.selected_epoch,
                   "validation_metric": run.best_validation_metric, "selection_k": config.selection_k,
                   "parameter_count": run.parameter_count, "training_audit": training_audit,
                   "validation_audit": dict(validation.audit()),
                   "validation_metrics": run.validation_result.summary,
                   "test_evaluated": False}
        (destination / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True,
                                                             allow_nan=False) + "\n", encoding="utf-8")
        return run
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(torch_state)
        torch.use_deterministic_algorithms(previous_determinism, warn_only=previous_warn_only)
        torch.set_num_threads(previous_threads)


def train_from_manifest(data_path: str | Path, config_path: str | Path, output: str | Path) -> TrainingRun:
    return train_dataset(load_dataset(data_path, require_labels=True), RunConfig.from_json(config_path), output)
