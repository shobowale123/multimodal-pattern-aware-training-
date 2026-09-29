from __future__ import annotations
from typing import Sequence
import math
import numpy as np
import torch
from torch import Tensor
from .config import ContractError

def pattern_codes(pattern_ids: Sequence[str]) -> np.ndarray:
    mapping = {label: pos for pos, label in enumerate(sorted(set(pattern_ids)))}
    return np.asarray([mapping[label] for label in pattern_ids], dtype=np.int64)


def supervised_contrastive_loss(
    embeddings: Tensor,
    pattern_labels: Tensor,
    *,
    temperature: float = 0.07,
    approved_comparisons: Tensor | None = None,
) -> Tensor:
    if embeddings.ndim != 2:
        raise ContractError("embeddings must be two-dimensional")
    if not isinstance(temperature, (int, float)) or not math.isfinite(temperature) or temperature <= 0:
        raise ContractError("temperature must be finite and positive")
    if len(embeddings) < 2 or not torch.isfinite(embeddings).all():
        raise ContractError("Need at least two finite embeddings")
    if pattern_labels.ndim != 1 or pattern_labels.shape[0] != embeddings.shape[0]:
        raise ContractError("pattern_labels must contain one label per embedding")
    if pattern_labels.device != embeddings.device:
        raise ContractError("Labels and embeddings must be on the same device")
    norms = torch.linalg.vector_norm(embeddings, ord=2, dim=1)
    if not torch.allclose(norms, torch.ones_like(norms), atol=1e-4, rtol=0.0):
        raise ContractError("embeddings must be unit-normalized")

    labels = pattern_labels.reshape(-1, 1)
    self_mask = torch.eye(len(embeddings), dtype=torch.bool, device=embeddings.device)
    positive_mask = labels.eq(labels.T) & ~self_mask
    positives_per_anchor = positive_mask.sum(dim=1)
    if torch.any(positives_per_anchor == 0):
        raise ContractError("Every anchor needs a positive")

    if approved_comparisons is None:
        comparison_mask = ~self_mask
    else:
        if (approved_comparisons.shape != self_mask.shape or approved_comparisons.dtype != torch.bool
                or approved_comparisons.device != embeddings.device):
            raise ContractError("approved_comparisons must be a Boolean square mask on the embedding device")
        comparison_mask = (approved_comparisons | positive_mask) & ~self_mask

    logits = embeddings @ embeddings.T / float(temperature)
    logits = logits.masked_fill(~comparison_mask, -torch.inf)
    log_probabilities = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    positive_log_probability = torch.where(
        positive_mask,
        log_probabilities,
        torch.zeros_like(log_probabilities),
    ).sum(dim=1) / positives_per_anchor
    loss = -positive_log_probability.mean()
    if not torch.isfinite(loss):
        raise ContractError("Supervised contrastive loss became non-finite")
    return loss
