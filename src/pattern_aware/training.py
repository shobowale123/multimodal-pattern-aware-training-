from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Any
import random
import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn
from .config import ContractError, TrainingConfig, _positive_integer
from .data import UnifiedInputs
from .models import tensor_batch, count_trainable_parameters, FusionDiagnostics
from .sampling import PatternBalancedBatchSampler
from .losses import pattern_codes, supervised_contrastive_loss
from .evaluation import RetrievalEvaluation, evaluate_retrieval
DEVICE = torch.device('cpu')

def seed_everything(seed: int = 42) -> None:
    """Seed CPU training; exact numeric reproducibility requires matching dependencies."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def encode_all(model: nn.Module, aligned_inputs: UnifiedInputs, batch_size: int = 32) -> np.ndarray:
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ContractError("batch_size must be a positive integer")
    if not len(aligned_inputs.record_ids):
        raise ContractError("Cannot encode an empty dataset")
    model.eval()
    vectors = []
    device = next(model.parameters()).device
    with torch.inference_mode():
        for start in range(0, len(aligned_inputs.record_ids), batch_size):
            rows = np.arange(start, min(start + batch_size, len(aligned_inputs.record_ids)))
            batch = tensor_batch(aligned_inputs, rows, device=device)
            vectors.append(model(*batch).cpu().numpy())
    return np.vstack(vectors)


def encode_all_with_diagnostics(
    model: nn.Module, aligned_inputs: UnifiedInputs,
) -> tuple[np.ndarray, FusionDiagnostics]:
    if not len(aligned_inputs.record_ids):
        raise ContractError("Cannot encode an empty dataset")
    device = next(model.parameters()).device
    batch = tensor_batch(aligned_inputs, np.arange(len(aligned_inputs.record_ids)), device=device)
    model.eval()
    with torch.inference_mode():
        embeddings, diagnostics = model(*batch, return_diagnostics=True)
    return embeddings.cpu().numpy(), diagnostics


@dataclass
class TrainingRun:
    name: str
    model: nn.Module
    history: pd.DataFrame
    selected_epoch: int
    best_validation_metric: float
    validation_result: RetrievalEvaluation
    parameter_count: int


TRAINING_CONFIG = TrainingConfig()


def epoch_loss(
    model: nn.Module,
    aligned_inputs: UnifiedInputs,
    labels: np.ndarray,
    sampler: PatternBalancedBatchSampler,
    *,
    optimizer: torch.optim.Optimizer,
    temperature: float,
) -> float:
    model.train()
    total_loss = 0.0
    anchors = 0

    for indices in sampler:
        matrices, content_masks, source_masks, qualities = tensor_batch(aligned_inputs, indices)
        pattern_labels = torch.as_tensor(labels[indices], device=DEVICE)

        optimizer.zero_grad(set_to_none=True)
        embeddings = model(matrices, content_masks, source_masks, qualities)
        loss = supervised_contrastive_loss(
            embeddings,
            pattern_labels,
            temperature=temperature,
        )
        loss.backward()
        optimizer.step()

        anchor_count = len(indices)
        total_loss += float(loss.detach()) * anchor_count
        anchors += anchor_count

    if anchors == 0:
        raise ContractError("Sampler emitted no anchors")
    return total_loss / anchors


def validation_selection_metric(
    model: nn.Module,
    aligned_inputs: UnifiedInputs,
    selection_k: int = 30,
) -> tuple[float, RetrievalEvaluation]:
    _positive_integer("selection_k", selection_k)
    vectors = encode_all(model, aligned_inputs)
    result = evaluate_retrieval(
        vectors,
        aligned_inputs.pattern_ids,
        record_ids=aligned_inputs.record_ids,
        available=aligned_inputs.eligible,
        k_values=(selection_k,),
    )
    return float(result.summary["pattern_macro"][f"NDCG@{selection_k}"]), result


def train_and_select(
    name: str,
    model_factory: Callable[[], nn.Module],
    train_inputs: UnifiedInputs,
    validation_inputs: UnifiedInputs,
    config: TrainingConfig = TRAINING_CONFIG,
    *,
    selection_k: int = 30,
) -> TrainingRun:
    _positive_integer("selection_k", selection_k)
    if not isinstance(config, TrainingConfig):
        raise ContractError("config must be a TrainingConfig")
    if set(train_inputs.splits) != {"train"} or set(validation_inputs.splits) != {"validation"}:
        raise ContractError("Training and selection require the train and validation splits respectively")
    if set(train_inputs.pattern_ids) & set(validation_inputs.pattern_ids):
        raise ContractError("Training and validation patterns must be disjoint")
    if set(train_inputs.record_ids) & set(validation_inputs.record_ids):
        raise ContractError("Training and validation record identifiers must be disjoint")
    seed_everything(config.seed)
    model = model_factory().to(DEVICE)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    sampler = PatternBalancedBatchSampler(
        train_inputs.pattern_ids,
        train_inputs.eligible,
        patterns_per_batch=config.patterns_per_batch,
        members_per_pattern=config.members_per_pattern,
        seed=config.seed,
    )
    label_codes = pattern_codes(train_inputs.pattern_ids)

    best_metric = float("-inf")
    best_epoch = -1
    best_state: dict[str, Tensor] | None = None
    best_validation_result: RetrievalEvaluation | None = None
    epochs_without_improvement = 0
    records: list[dict[str, Any]] = []

    for epoch in range(config.epochs):
        sampler.set_epoch(epoch)
        train_loss = epoch_loss(
            model,
            train_inputs,
            label_codes,
            sampler,
            optimizer=optimizer,
            temperature=config.temperature,
        )
        validation_metric, validation_result = validation_selection_metric(
            model,
            validation_inputs,
            selection_k,
        )
        if not np.isfinite(train_loss) or not np.isfinite(validation_metric):
            raise ContractError("Training loss and validation selection metric must stay finite")

        improved = validation_metric > best_metric + config.minimum_delta
        if improved:
            best_metric = validation_metric
            best_epoch = epoch + 1
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            best_validation_result = validation_result
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        records.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                f"validation_pattern_macro_ndcg_at_{selection_k}": validation_metric,
                "new_best": improved,
                "patience_used": epochs_without_improvement,
            }
        )

        if not improved and epochs_without_improvement >= config.patience:
            break

    if best_state is None or best_validation_result is None:
        raise RuntimeError(f"No checkpoint selected for {name}")

    model.load_state_dict(best_state, strict=True)
    model.eval()
    return TrainingRun(
        name=name,
        model=model,
        history=pd.DataFrame(records),
        selected_epoch=best_epoch,
        best_validation_metric=best_metric,
        validation_result=best_validation_result,
        parameter_count=count_trainable_parameters(model),
    )
