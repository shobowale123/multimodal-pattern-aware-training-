from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from pattern_aware.config import ContractError, MODALITY_DIMENSIONS, MODALITY_ORDER
from pattern_aware.data import align_modalities, make_synthetic_dataset, unit_rows


def test_synthetic_data_has_disjoint_patterns_and_consistent_masks():
    dataset = make_synthetic_dataset(seed=42)
    inputs = dataset.inputs
    assert len(inputs.record_ids) == len(set(inputs.record_ids))
    assert dataset.cohort.groupby("pattern_id")["split"].nunique().max() == 1
    assert set(inputs.splits) == {"train", "validation", "test"}
    assert inputs.content_masks.dtype == np.bool_
    assert inputs.source_masks.dtype == np.bool_
    np.testing.assert_array_equal(inputs.eligible, inputs.content_masks.any(axis=1))
    assert not np.any(inputs.content_masks[:, 3:] & ~inputs.source_masks)

    for position, name in enumerate(MODALITY_ORDER):
        matrix = inputs.matrices[name]
        assert matrix.shape == (len(inputs.record_ids), MODALITY_DIMENSIONS[name])
        assert np.isfinite(matrix).all()
        np.testing.assert_array_equal(matrix[~inputs.content_masks[:, position]], 0)
        assert inputs.qualities[name].shape == (len(inputs.record_ids), 1)
        assert np.isfinite(inputs.qualities[name]).all()


def test_seed_reproduces_data_exactly_and_changes_generated_vectors():
    first = make_synthetic_dataset(seed=42)
    same = make_synthetic_dataset(seed=42)
    different = make_synthetic_dataset(seed=43)
    pd.testing.assert_frame_equal(first.cohort, same.cohort)
    np.testing.assert_array_equal(first.inputs.content_masks, same.inputs.content_masks)
    for name in MODALITY_ORDER:
        np.testing.assert_array_equal(first.inputs.matrices[name], same.inputs.matrices[name])
        np.testing.assert_array_equal(first.inputs.qualities[name], same.inputs.qualities[name])
    assert not np.array_equal(first.inputs.matrices["narrative"], different.inputs.matrices["narrative"])


def test_subset_and_quality_ablation_preserve_record_alignment():
    inputs = make_synthetic_dataset(seed=7).inputs
    indices = np.array([7, 1, 5])
    subset = inputs.subset(indices)
    assert subset.record_ids == tuple(inputs.record_ids[index] for index in indices)
    assert subset.pattern_ids == tuple(inputs.pattern_ids[index] for index in indices)
    without_quality = subset.without_quality()
    assert without_quality.record_ids == subset.record_ids
    for name in MODALITY_ORDER:
        np.testing.assert_array_equal(subset.matrices[name], inputs.matrices[name][indices])
        np.testing.assert_array_equal(without_quality.matrices[name], subset.matrices[name])
        assert without_quality.qualities[name].shape == (3, 0)
    np.testing.assert_array_equal(without_quality.content_masks, subset.content_masks)


def test_row_normalization_preserves_zero_vectors():
    result = unit_rows(np.array([[3, 4], [0, 0]], dtype=np.float32))
    np.testing.assert_allclose(result, [[0.6, 0.8], [0, 0]], atol=1e-7)
    assert np.isfinite(result).all()


def test_artifact_alignment_uses_record_ids_not_storage_position():
    dataset = make_synthetic_dataset(seed=19)
    shuffled = {}
    for name, artifact in dataset.artifacts.items():
        shuffled[name] = replace(
            artifact,
            record_ids=artifact.record_ids[::-1],
            matrix=artifact.matrix[::-1],
            quality=artifact.quality[::-1],
            content_available=artifact.content_available[::-1],
            source_available=artifact.source_available[::-1],
        )
    aligned = align_modalities(dataset.cohort, shuffled)
    assert aligned.record_ids == dataset.inputs.record_ids
    np.testing.assert_array_equal(aligned.content_masks, dataset.inputs.content_masks)
    for name in MODALITY_ORDER:
        np.testing.assert_array_equal(aligned.matrices[name], dataset.inputs.matrices[name])
        np.testing.assert_array_equal(aligned.qualities[name], dataset.inputs.qualities[name])


def test_cross_split_pattern_leakage_is_rejected():
    inputs = make_synthetic_dataset(seed=42).inputs
    leaked_splits = ("validation",) + inputs.splits[1:]
    with pytest.raises(ContractError, match="pattern"):
        replace(inputs, splits=leaked_splits)
