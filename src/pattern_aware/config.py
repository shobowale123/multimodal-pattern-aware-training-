from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Mapping, Any
from pathlib import Path
import json
import math

class ContractError(ValueError):
    """Raised when public inputs violate a documented invariant."""


MODALITY_ORDER: tuple[str, ...] = (
    "location",
    "time",
    "narrative",
    "suspect",
    "weapon",
)

MODALITY_DIMENSIONS: Mapping[str, int] = {
    "location": 256,
    "time": 256,
    "narrative": 1024,
    "suspect": 128,
    "weapon": 128,
}

ADAPTER_HIDDEN_DIMENSIONS: Mapping[str, int] = {
    "location": 256,
    "time": 256,
    "narrative": 256,
    "suspect": 128,
    "weapon": 128,
}

QUALITY_FIELD_NAMES: Mapping[str, str] = {
    "location": "synthetic_geocode_reliability",
    "time": "synthetic_time_specificity",
    "narrative": "synthetic_narrative_completeness",
    "suspect": "synthetic_suspect_completeness",
    "weapon": "synthetic_weapon_specificity",
}


@dataclass(frozen=True)
class FusionConfig:
    name: str
    quality_dimension: int
    adapter_dimension: int = 128
    fusion_hidden_dimension: int = 512
    output_dimension: int = 256
    dropout: float = 0.0
    attention_heads: int = 4
    attention_feedforward_dimension: int = 256

    def __post_init__(self) -> None:
        if self.quality_dimension not in (0, 1):
            raise ContractError("Synthetic quality width must be 0 or 1")
        for name in ("adapter_dimension", "fusion_hidden_dimension", "output_dimension",
                     "attention_heads", "attention_feedforward_dimension"):
            _positive_integer(name, getattr(self, name))
        if self.adapter_dimension % self.attention_heads:
            raise ContractError("adapter_dimension must be divisible by attention_heads")
        if not isinstance(self.dropout, (int, float)) or not 0 <= self.dropout < 1:
            raise ContractError("dropout must be in [0, 1)")

    @property
    def quality_dimension_map(self) -> Mapping[str, int]:
        return {name: self.quality_dimension for name in MODALITY_ORDER}

    @property
    def quality_name_map(self) -> Mapping[str, tuple[str, ...]]:
        if self.quality_dimension == 0:
            return {name: () for name in MODALITY_ORDER}
        return {name: (QUALITY_FIELD_NAMES[name],) for name in MODALITY_ORDER}

    @property
    def mask_dimension(self) -> int:
        return 7  # five content masks + suspect/weapon source masks

    @property
    def fusion_input_dimension(self) -> int:
        return len(MODALITY_ORDER) * self.adapter_dimension + self.mask_dimension


def _positive_integer(name: str, value: int, minimum: int = 1) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ContractError(f"{name} must be an integer >= {minimum}")


def _finite_number(name: str, value: float, *, positive: bool = False) -> None:
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value < 0 or (positive and value == 0)):
        raise ContractError(f"{name} must be finite and {'positive' if positive else 'nonnegative'}")


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 18
    learning_rate: float = 1.0e-3
    weight_decay: float = 1.0e-4
    temperature: float = 0.07
    patterns_per_batch: int = 3
    members_per_pattern: int = 2
    patience: int = 5
    minimum_delta: float = 1.0e-5
    seed: int = 42

    def __post_init__(self) -> None:
        for name in ("epochs", "patience"):
            _positive_integer(name, getattr(self, name))
        for name in ("patterns_per_batch", "members_per_pattern"):
            _positive_integer(name, getattr(self, name), minimum=2)
        _positive_integer("seed", self.seed, minimum=0)
        if self.seed > 2**32 - 1:
            raise ContractError("seed must be <= 2**32 - 1")
        for name in ("learning_rate", "temperature"):
            _finite_number(name, getattr(self, name), positive=True)
        for name in ("weight_decay", "minimum_delta"):
            _finite_number(name, getattr(self, name))


@dataclass(frozen=True)
class ExperimentConfig:
    """Small CPU experiment; architecture and synthetic cohort match the notebooks."""

    training: TrainingConfig = field(default_factory=TrainingConfig)
    variants: tuple[str, ...] = ("v1", "v11", "v2")
    k_values: tuple[int, ...] = (1, 5, 10, 30)
    save_checkpoints: bool = False
    plots: bool = True
    cpu_threads: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.training, TrainingConfig):
            raise ContractError("training must be a TrainingConfig")
        if not self.variants or len(set(self.variants)) != len(self.variants):
            raise ContractError("variants must be a nonempty list without duplicates")
        if set(self.variants) - {"v1", "v11", "v2"}:
            raise ContractError("variants must contain only v1, v11, v2")
        if not self.k_values:
            raise ContractError("k_values cannot be empty")
        for k in self.k_values:
            _positive_integer("retrieval k", k)
        for name in ("save_checkpoints", "plots"):
            if not isinstance(getattr(self, name), bool):
                raise ContractError(f"{name} must be Boolean")
        _positive_integer("cpu_threads", self.cpu_threads)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "ExperimentConfig":
        if not isinstance(values, Mapping):
            raise ContractError("Experiment configuration must be a JSON object")
        values = dict(values)
        try:
            values["training"] = TrainingConfig(**values.get("training", {}))
            for name in ("variants", "k_values"):
                if name in values:
                    values[name] = tuple(values[name])
            return cls(**values)
        except (TypeError, ValueError) as exc:
            raise ContractError(f"Invalid experiment configuration: {exc}") from exc

    @classmethod
    def from_json(cls, path: str | Path) -> "ExperimentConfig":
        with Path(path).open(encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))


V1_CONFIG = FusionConfig(name="V1 global-gated concatenation", quality_dimension=0)
V11_CONFIG = FusionConfig(name="V1.1 per-record quality-gated concatenation", quality_dimension=1)
V2_CONFIG = FusionConfig(name="V2 five-token self-attention", quality_dimension=1)
