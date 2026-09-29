import math

import pytest
import torch
import torch.nn.functional as F

from pattern_aware.config import ContractError
from pattern_aware.losses import supervised_contrastive_loss


def test_contrastive_loss_excludes_self_and_matches_hand_calculation():
    embeddings = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
    labels = torch.tensor([0, 0, 1, 1])
    loss = supervised_contrastive_loss(embeddings, labels, temperature=1.0)
    # Each denominator has one positive exp(1) and two negatives exp(0).
    assert loss.item() == pytest.approx(math.log(math.e + 2) - 1, abs=1e-6)


def test_comparison_mask_always_retains_positives_and_never_self():
    embeddings = torch.tensor([[1.0, 0.0]] * 3 + [[0.0, 1.0]] * 3)
    labels = torch.tensor([0, 0, 0, 1, 1, 1])
    # No approved comparisons: the two non-self positives must still remain.
    approved = torch.zeros((6, 6), dtype=torch.bool)
    loss = supervised_contrastive_loss(
        embeddings, labels, temperature=1.0, approved_comparisons=approved
    )
    assert loss.item() == pytest.approx(math.log(2), abs=1e-6)
    approved.fill_diagonal_(True)
    diagonal_approved_loss = supervised_contrastive_loss(
        embeddings, labels, temperature=1.0, approved_comparisons=approved
    )
    assert diagonal_approved_loss.item() == pytest.approx(loss.item(), abs=1e-6)


def test_loss_has_finite_nonzero_gradients():
    raw = torch.tensor(
        [[1.0, 0.2, 0.1], [0.7, 0.6, 0.1], [0.1, 1.0, 0.4], [0.4, 0.8, 0.3]],
        requires_grad=True,
    )
    loss = supervised_contrastive_loss(F.normalize(raw, dim=1), torch.tensor([0, 0, 1, 1]))
    loss.backward()
    assert torch.isfinite(loss)
    assert raw.grad is not None
    assert torch.isfinite(raw.grad).all()
    assert torch.count_nonzero(raw.grad) > 0


def test_every_anchor_requires_a_distinct_positive():
    with pytest.raises(ContractError, match="positive"):
        supervised_contrastive_loss(torch.eye(3), torch.arange(3))


def test_loss_rejects_non_normalized_vectors():
    with pytest.raises(ContractError, match="normalized"):
        supervised_contrastive_loss(torch.ones((4, 3)), torch.tensor([0, 0, 1, 1]))
