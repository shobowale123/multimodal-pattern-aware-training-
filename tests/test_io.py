"""Own-data contract tests use small files, independent of the demo generator."""
import csv
import json
from dataclasses import replace

import numpy as np
import pytest
import torch

from pattern_aware.config import ContractError, FeatureSchema, MODALITY_ORDER, V1_CONFIG, V11_CONFIG, V2_CONFIG
from pattern_aware.data import FeatureInputs, UnifiedInputs
from pattern_aware.io import load_dataset
from pattern_aware.models import FiveModalityConcatFusionEncoder, FiveTokenSelfAttentionFusionEncoder, tensor_batch


@pytest.fixture
def dataset_files(tmp_path):
    schema = FeatureSchema(
        dimensions={role: pos + 2 for pos, role in enumerate(MODALITY_ORDER)},
        encoder_versions={role: f"example-{role}-1" for role in MODALITY_ORDER},
        quality_names={role: ("observed_confidence",) for role in MODALITY_ORDER},
    )
    rows = [
        {"record_id": "record-b", "pattern_id": "pattern-1", "split": "train", "group_id": "group-1"},
        {"record_id": "record-a", "pattern_id": "pattern-2", "split": "validation", "group_id": "group-2"},
        {"record_id": "record-c", "pattern_id": "pattern-3", "split": "test", "group_id": "group-3"},
    ]
    write_csv(tmp_path / "records.csv", rows)
    for role in MODALITY_ORDER:
        matrix = np.arange(3 * schema.dimensions[role], dtype=np.float32).reshape(3, -1) + 1
        np.savez(
            tmp_path / f"{role}.npz",
            record_ids=np.asarray(["record-c", "record-b", "record-a"]),
            embeddings=matrix,
            quality=np.asarray([[0.9], [0.4], [0.2]], dtype=np.float32),
            content_available=np.asarray([False, True, True]),
            source_available=np.asarray([False, True, True]),
        )
    manifest = dict(
        schema_version=1, schema=schema.to_dict(), records="records.csv",
        modalities={role: f"{role}.npz" for role in MODALITY_ORDER},
    )
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path, manifest, rows


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def rewrite_npz(path, **updates):
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    arrays.update(updates)
    np.savez(path, **arrays)


def test_external_rows_align_by_id_and_preserve_schema(dataset_files):
    path, manifest, _ = dataset_files
    inputs = load_dataset(path, require_labels=True)
    assert isinstance(inputs, UnifiedInputs)
    assert inputs.record_ids == ("record-b", "record-a", "record-c")
    np.testing.assert_array_equal(inputs.matrices["location"], [[3, 4], [5, 6], [0, 0]])
    np.testing.assert_array_equal(inputs.eligible, [True, True, False])
    np.testing.assert_allclose(inputs.qualities["location"].ravel(), [.4, .2, .9])
    assert inputs.schema.to_dict() == manifest["schema"]
    assert inputs.audit()["all_missing_rows"] == 1
    subset = inputs.subset([1, 0]).without_quality()
    assert subset.schema == inputs.schema
    assert subset.group_ids == ("group-2", "group-1")
    assert all(values.shape == (2, 0) for values in subset.qualities.values())


def test_unlabeled_encoding_inputs_do_not_require_fake_labels(dataset_files):
    path, _, rows = dataset_files
    write_csv(path.parent / "records.csv", [{"record_id": row["record_id"]} for row in rows])
    inputs = load_dataset(path)
    assert type(inputs) is FeatureInputs
    assert not hasattr(inputs, "pattern_ids")
    assert type(inputs.subset([0]).without_quality()) is FeatureInputs
    with pytest.raises(ContractError, match="requires pattern_id and split"):
        load_dataset(path, require_labels=True)


@pytest.mark.parametrize("column,value", [
    ("record_id", "record-b"), ("record_id", " "),
    ("pattern_id", "pattern-1"), ("pattern_id", ""),
    ("group_id", "group-1"), ("split", "testing"),
])
def test_invalid_ids_and_cross_split_memberships_rejected(dataset_files, column, value):
    path, _, rows = dataset_files
    rows[1][column] = value
    write_csv(path.parent / "records.csv", rows)
    with pytest.raises(ContractError):
        load_dataset(path)


@pytest.mark.parametrize("updates", [
    {"record_ids": np.asarray(["record-a", "record-b", "extra"])},
    {"record_ids": np.asarray(["record-a", "record-b", "record-b"])},
    {"record_ids": np.asarray([b"record-a", b"record-b", b"record-c"])},
    {"record_ids": np.asarray(["record-a", "record-b", "record-c"], dtype=object)},
    {"embeddings": np.zeros((3, 2), dtype=np.int64)},
    {"embeddings": np.zeros((3, 3), dtype=np.float32)},
    {"embeddings": np.full((3, 2), np.nan)},
    {"embeddings": np.full((3, 2), np.inf)},
    {"embeddings": np.full((3, 2), 1e100)},
    {"content_available": np.asarray([0, 1, 1])},
    {"source_available": np.asarray([False, False, True])},
    {"quality": np.full((3, 1), 1.1, dtype=np.float32)},
    {"quality": np.full((3, 1), -0.1, dtype=np.float32)},
    {"quality": np.zeros((3, 2), dtype=np.float32)},
    {"quality": np.full((3, 1), np.nan)},
])
def test_bad_npz_arrays_rejected_without_pickle(dataset_files, updates):
    path, _, _ = dataset_files
    rewrite_npz(path.parent / "location.npz", **updates)
    with pytest.raises(ContractError):
        load_dataset(path)


@pytest.mark.parametrize("field,value", [("schema_version", 2), ("schema_version", True), ("unexpected", 1)])
def test_unknown_manifest_fields_or_versions_rejected(dataset_files, field, value):
    path, manifest, _ = dataset_files
    manifest[field] = value
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ContractError):
        load_dataset(path)


def test_schema_is_strict_and_tracks_encoder_quality_and_normalization(dataset_files):
    path, manifest, _ = dataset_files
    schema = load_dataset(path).schema
    assert FeatureSchema.from_dict(schema.to_dict()) == schema
    for field, value in [
        ("encoder_versions", {**schema.encoder_versions, "location": "changed-2"}),
        ("quality_names", {**schema.quality_names, "location": ("other_definition",)}),
        ("dimensions", {**schema.dimensions, "location": 3}),
        ("normalization", "l2"),
    ]:
        with pytest.raises(ContractError, match="schema mismatch"):
            schema.assert_compatible(replace(schema, **{field: value}))
    with pytest.raises(ContractError):
        FeatureSchema.from_dict({**manifest["schema"], "unknown": True})
    with pytest.raises(ContractError):
        replace(schema, dimensions={**schema.dimensions, "location": True})
    with pytest.raises(ContractError):
        replace(schema, quality_names={**schema.quality_names, "location": ("a", "b")})


def test_declared_l2_normalization_handles_large_values_and_missing_rows(dataset_files):
    path, manifest, _ = dataset_files
    manifest["schema"]["normalization"] = "l2"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    rewrite_npz(path.parent / "location.npz", embeddings=np.full((3, 2), 1e30, dtype=np.float32))
    inputs = load_dataset(path)
    for matrix in inputs.matrices.values():
        np.testing.assert_allclose(np.linalg.norm(matrix[:2], axis=1), 1, atol=1e-6)
        np.testing.assert_array_equal(matrix[2], 0)
    assert inputs.schema.normalization == "l2"


@pytest.mark.parametrize("base", [V1_CONFIG, V11_CONFIG, V2_CONFIG], ids=["v1", "v11", "v2"])
def test_all_architectures_accept_configurable_input_widths(dataset_files, base):
    path, _, _ = dataset_files
    inputs = load_dataset(path)
    config = replace(base, input_dimensions=inputs.schema.dimensions, adapter_dimension=8,
                     fusion_hidden_dimension=16, output_dimension=4,
                     attention_feedforward_dimension=16)
    model_type = FiveTokenSelfAttentionFusionEncoder if base == V2_CONFIG else FiveModalityConcatFusionEncoder
    model = model_type(config).eval()
    features = inputs.without_quality() if base.quality_dimension == 0 else inputs
    with torch.inference_mode():
        embeddings = model(*tensor_batch(features, np.arange(3)))
    assert embeddings.shape == (3, 4)
    assert torch.isfinite(embeddings).all()
    torch.testing.assert_close(embeddings.norm(dim=1), torch.tensor([1., 1., 0.]))


def test_zero_width_quality_schema_is_valid(dataset_files):
    path, manifest, _ = dataset_files
    for role in MODALITY_ORDER:
        manifest["schema"]["quality_names"][role] = []
        rewrite_npz(path.parent / f"{role}.npz", quality=np.empty((3, 0), dtype=np.float32))
    path.write_text(json.dumps(manifest), encoding="utf-8")
    assert all(value.shape == (3, 0) for value in load_dataset(path).qualities.values())


def test_programmatic_inputs_reject_float32_overflow(dataset_files):
    path, _, _ = dataset_files
    inputs = load_dataset(path)
    with pytest.raises(ContractError, match="float32"):
        replace(inputs, matrices={**inputs.matrices, "location": np.full((3, 2), 1e100)})


def test_duplicate_json_keys_are_rejected(dataset_files):
    path, manifest, _ = dataset_files
    raw = json.dumps(manifest).replace('"schema_version": 1', '"schema_version": 1, "schema_version": 1')
    path.write_text(raw, encoding="utf-8")
    with pytest.raises(ContractError, match="Duplicate JSON key"):
        load_dataset(path)


@pytest.mark.parametrize("changes", [{"quality_dimension": True}, {"dropout": False}])
def test_model_config_rejects_booleans_as_numbers(changes):
    with pytest.raises(ContractError):
        replace(V1_CONFIG, **changes)
