"""CPU multimodal training and inference from precomputed embeddings."""
from .config import ContractError, ExperimentConfig, FeatureSchema, TrainingConfig
from .data import FeatureInputs, UnifiedInputs
from .experiment import ExperimentResult, run_experiment
from .io import load_dataset
from .checkpoints import LoadedCheckpoint, load_checkpoint
from .pipeline import RunConfig, train_dataset, train_from_manifest
from .inference import (EmbeddingArtifact, encode_dataset, encode_from_manifest,
                        evaluate_checkpoint, load_embeddings, save_embeddings)
from .retrieval import retrieve_from_files, top_k_indices

__all__ = [
    "ContractError", "ExperimentConfig", "TrainingConfig", "ExperimentResult", "run_experiment",
    "FeatureSchema", "FeatureInputs", "UnifiedInputs", "load_dataset", "RunConfig",
    "train_dataset", "train_from_manifest", "LoadedCheckpoint", "load_checkpoint",
    "EmbeddingArtifact", "encode_dataset", "encode_from_manifest", "evaluate_checkpoint",
    "load_embeddings", "save_embeddings", "retrieve_from_files", "top_k_indices",
]
__version__ = "0.2.0"
