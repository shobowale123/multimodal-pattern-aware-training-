from dataclasses import asdict, replace
from types import SimpleNamespace

import numpy as np
import pytest

from pattern_aware.checkpoints import LoadedCheckpoint, create_model, hash_identifiers
from pattern_aware.config import ContractError, FusionConfig
from pattern_aware.data import make_synthetic_dataset
from pattern_aware.inference import (
    EmbeddingArtifact, _check_evaluation_provenance, encode_dataset,
    load_embeddings, save_embeddings,
)


@pytest.fixture
def inputs():
    return make_synthetic_dataset().inputs


def _checkpoint(inputs, variant):
    config = FusionConfig(
        name=variant, quality_dimension=0 if variant == "v1" else 1,
        adapter_dimension=8, fusion_hidden_dimension=16, output_dimension=7,
        attention_heads=2, attention_feedforward_dimension=16,
        input_dimensions=inputs.schema.dimensions,
    )
    return LoadedCheckpoint(create_model(variant, config), inputs.schema,
                            {"variant": variant, "weights_sha256": "a" * 64,
                             "fusion_config": asdict(config)})


@pytest.mark.parametrize("variant", ["v1", "v11", "v2"])
def test_encoding_batch_parity_allmissing_and_npz_roundtrip(inputs, variant, tmp_path):
    checkpoint = _checkpoint(inputs, variant)
    encoded = encode_dataset(checkpoint, inputs, batch_size=7)
    one_batch = encode_dataset(checkpoint, inputs, batch_size=len(inputs.record_ids))
    np.testing.assert_allclose(encoded.vectors, one_batch.vectors, rtol=1e-5, atol=1e-6)
    assert encoded.record_ids == inputs.record_ids
    assert not encoded.available[-1]
    np.testing.assert_array_equal(encoded.vectors[-1], 0)
    np.testing.assert_allclose(np.linalg.norm(encoded.vectors[encoded.available], axis=1), 1, atol=1e-6)
    path = tmp_path / "vectors.npz"
    save_embeddings(encoded, path)
    loaded = load_embeddings(path)
    assert loaded.metadata == encoded.metadata
    assert loaded.record_ids == encoded.record_ids
    np.testing.assert_array_equal(loaded.vectors, encoded.vectors)
    np.testing.assert_array_equal(loaded.available, encoded.available)
    with np.load(path, allow_pickle=False) as archive:
        assert all(archive[name].dtype.kind != "O" for name in archive.files)
    with pytest.raises(ContractError, match="exists"):
        save_embeddings(encoded, path)


@pytest.mark.parametrize("variant", ["v1", "v11", "v2"])
def test_inference_rejects_encoder_identity_and_quality_schema_drift(inputs, variant):
    checkpoint = _checkpoint(inputs, variant)
    versions = dict(checkpoint.schema.encoder_versions)
    versions["location"] = "different-encoder"
    changed = replace(checkpoint, schema=replace(checkpoint.schema, encoder_versions=versions))
    with pytest.raises(ContractError, match="schema"):
        encode_dataset(changed, inputs)
    names = dict(checkpoint.schema.quality_names)
    names["location"] = ("different-quality-definition",)
    changed = replace(checkpoint, schema=replace(checkpoint.schema, quality_names=names))
    with pytest.raises(ContractError, match="schema"):
        encode_dataset(changed, inputs)


@pytest.mark.parametrize("batch_size", [0, True, 2.5])
def test_encoding_rejects_invalid_batch_sizes(inputs, batch_size):
    with pytest.raises(ContractError, match="batch_size"):
        encode_dataset(_checkpoint(inputs, "v1"), inputs, batch_size=batch_size)


def test_empty_input_encoding_retains_output_dimension(inputs):
    checkpoint = _checkpoint(inputs, "v1")
    artifact = encode_dataset(checkpoint, inputs.subset([]))
    assert artifact.vectors.shape == (0, 7)
    assert artifact.available.shape == (0,)


def test_encoding_revalidates_mutated_programmatic_inputs(inputs):
    checkpoint = _checkpoint(inputs, "v1")
    inputs.matrices["location"][0, 0] = np.nan
    with pytest.raises(ContractError, match="finite"):
        encode_dataset(checkpoint, inputs)


def test_quality_gated_inference_rejects_quality_free_view(inputs):
    with pytest.raises(ContractError, match="one quality field"):
        encode_dataset(_checkpoint(inputs, "v2"), inputs.without_quality())


@pytest.mark.parametrize("field,value", [
    ("vectors", np.array([[2, 0]], dtype="float32")),
    ("vectors", np.array([[1, 0]], dtype="float64")),
    ("vectors", np.array([[np.nan, 0]], dtype="float32")),
    ("available", np.array([1])), ("available", np.array([False])),
    ("record_ids", ("",)), ("record_ids", (1,)),
])
def test_embedding_artifact_rejects_invalid_arrays(inputs, field, value):
    arguments = dict(record_ids=("q",), vectors=np.array([[1, 0]], dtype="float32"),
                     available=np.array([True]), metadata={"format_version": 1,
                     "fusion_config": asdict(FusionConfig(name="v1", quality_dimension=0, output_dimension=2,
                                                           input_dimensions=inputs.schema.dimensions)),
                     "feature_schema": inputs.schema.to_dict(), "weights_sha256": "a" * 64, "variant": "v1"})
    arguments[field] = value
    with pytest.raises(ContractError):
        EmbeddingArtifact(**arguments)


def test_npz_object_ids_are_rejected_without_pickle(inputs, tmp_path):
    artifact = encode_dataset(_checkpoint(inputs, "v1"), inputs)
    path = tmp_path / "unsafe.npz"
    import json
    np.savez(path, record_ids=np.asarray(artifact.record_ids, dtype=object), vectors=artifact.vectors,
             available=artifact.available, metadata=np.asarray(json.dumps(dict(artifact.metadata))))
    with pytest.raises(ContractError):
        load_embeddings(path)


@pytest.mark.parametrize("field,protected", [
    ("record_ids", "train"), ("record_ids", "validation"),
    ("pattern_ids", "train"), ("pattern_ids", "validation"),
    ("group_ids", "train"), ("group_ids", "validation"),
])
def test_held_out_evaluation_rejects_fitting_identity_pattern_group_overlap(field, protected):
    split_hashes = {
        split: {key: hash_identifiers([f"{split}-{key}"])
                for key in ("record_ids", "pattern_ids", "group_ids")}
        for split in ("train", "validation", "test")
    }
    checkpoint = SimpleNamespace(metadata={"provenance": {"split_hashes": split_hashes}})
    values = {key: (f"new-{key}",) for key in ("record_ids", "pattern_ids", "group_ids")}
    _check_evaluation_provenance(checkpoint, SimpleNamespace(**values), "test")
    values[field] = (f"{protected}-{field}",)
    with pytest.raises(ContractError, match=f"{field} overlap"):
        _check_evaluation_provenance(checkpoint, SimpleNamespace(**values), "test")


def test_group_provenance_requires_evaluation_group_ids():
    saved = {"record_ids": hash_identifiers(["r"]), "pattern_ids": hash_identifiers(["p"]),
             "group_ids": hash_identifiers(["g"])}
    checkpoint = SimpleNamespace(metadata={"provenance": {"split_hashes": {"train": saved, "validation": saved}}})
    with pytest.raises(ContractError, match="group_ids are required"):
        _check_evaluation_provenance(checkpoint, SimpleNamespace(record_ids=("new",), pattern_ids=("new",)), "test")
