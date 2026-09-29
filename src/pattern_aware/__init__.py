"""Multimodal pattern-aware training on fully synthetic data."""
from .config import ExperimentConfig, TrainingConfig
from .experiment import ExperimentResult, run_experiment

__all__ = ["ExperimentConfig", "TrainingConfig", "ExperimentResult", "run_experiment"]
__version__ = "0.1.0"
