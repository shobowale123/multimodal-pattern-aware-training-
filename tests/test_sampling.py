from collections import Counter

import numpy as np
import pytest

from pattern_aware.config import ContractError
from pattern_aware.sampling import PatternBalancedBatchSampler


@pytest.mark.parametrize("patterns_per_batch, pattern_count", [(2, 3), (2, 5), (3, 7), (4, 9)])
def test_batches_keep_distinct_positives_and_negatives_with_odd_tail(
    patterns_per_batch, pattern_count
):
    labels = [f"pattern-{p}" for p in range(pattern_count) for _ in range(3)]
    # A singleton and an unavailable group must never be sampled.
    labels += ["singleton", "unavailable", "unavailable"]
    available = np.ones(len(labels), dtype=bool)
    available[-2:] = False
    sampler = PatternBalancedBatchSampler(
        labels,
        available,
        patterns_per_batch=patterns_per_batch,
        members_per_pattern=2,
        seed=7,
    )

    batches = list(sampler)
    assert len(batches) == len(sampler)
    seen_patterns = []
    for batch in batches:
        assert len(set(batch)) == len(batch)
        counts = Counter(labels[index] for index in batch)
        assert len(counts) >= 2, "Each anchor needs both positives and negatives."
        assert set(counts.values()) == {2}
        assert all(available[index] for index in batch)
        seen_patterns.extend(counts)
    assert Counter(seen_patterns) == Counter(f"pattern-{p}" for p in range(pattern_count))


def test_sampling_is_repeatable_for_seed_and_epoch():
    labels = [f"p{p}" for p in range(12) for _ in range(5)]
    available = np.ones(len(labels), dtype=bool)
    first = PatternBalancedBatchSampler(labels, available, seed=71)
    second = PatternBalancedBatchSampler(labels, available, seed=71)
    assert list(first) == list(second)
    original = list(first)
    first.set_epoch(1)
    second.set_epoch(1)
    assert list(first) == list(second)
    assert list(first) != original


@pytest.mark.parametrize("patterns, members", [(1, 2), (2, 1)])
def test_sampler_rejects_batches_without_positive_or_negative_groups(patterns, members):
    with pytest.raises(ContractError):
        PatternBalancedBatchSampler(
            ["a", "a", "b", "b"],
            np.ones(4, dtype=bool),
            patterns_per_batch=patterns,
            members_per_pattern=members,
        )


def test_sampler_rejects_implicit_boolean_masks_and_insufficient_patterns():
    with pytest.raises(ContractError):
        PatternBalancedBatchSampler(["a", "a", "b", "b"], [1, 1, 1, 1])
    with pytest.raises(ContractError):
        PatternBalancedBatchSampler(["a", "a", "b"], np.ones(3, dtype=bool))
