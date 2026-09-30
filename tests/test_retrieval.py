from dataclasses import asdict, replace
import json

import numpy as np
import pandas as pd
import pytest

from pattern_aware.config import ContractError, FeatureSchema, FusionConfig, MODALITY_ORDER
from pattern_aware.data import unit_rows
from pattern_aware.inference import EmbeddingArtifact, save_embeddings
from pattern_aware.retrieval import retrieve_from_files, top_k_indices


def _metadata():
    schema = FeatureSchema(
        {name: 2 for name in MODALITY_ORDER},
        {name: "example-v1" for name in MODALITY_ORDER},
        {name: () for name in MODALITY_ORDER},
    )
    config = FusionConfig(name="v1", quality_dimension=0, output_dimension=2,
                          input_dimensions=schema.dimensions)
    return {"format_version": 1, "feature_schema": schema.to_dict(), "fusion_config": asdict(config),
            "weights_sha256": "a" * 64, "variant": "v1"}


@pytest.mark.parametrize("query_chunk,gallery_chunk", [(1, 1), (2, 3), (100, 100)])
def test_chunked_ranks_match_brute_force_with_ties_and_self_ids(query_chunk, gallery_chunk):
    rng = np.random.default_rng(17)
    queries = unit_rows(rng.normal(size=(7, 9)).astype("float32"))
    gallery = unit_rows(rng.normal(size=(11, 9)).astype("float32"))
    gallery[2] = gallery[0]
    query_ids = ["shared", *[f"query-{i}" for i in range(1, 7)]]
    gallery_ids = ["z-tied", "shared", "a-tied", *[f"gallery-{i}" for i in range(3, 11)]]
    q_available = np.array([True] * 6 + [False])
    g_available = np.array([True] * 10 + [False])
    result = top_k_indices(
        queries, gallery, query_ids=query_ids, gallery_ids=gallery_ids,
        query_available=q_available, gallery_available=g_available, top_k=5,
        query_chunk_size=query_chunk, gallery_chunk_size=gallery_chunk,
    )
    brute_scores = np.einsum("id,jd->ij", unit_rows(queries), unit_rows(gallery), optimize=False)
    np.clip(brute_scores, -1, 1, out=brute_scores)
    for row in range(len(queries)):
        candidates = np.array([i for i in range(len(gallery))
                               if g_available[i] and gallery_ids[i] != query_ids[row]], dtype=int)
        order = np.lexsort((np.asarray(gallery_ids)[candidates], -brute_scores[row, candidates]))
        expected = candidates[order[:5]] if q_available[row] else np.array([], dtype=int)
        np.testing.assert_array_equal(result.indices[row], expected)
        np.testing.assert_array_equal(result.scores[row], brute_scores[row, expected])
    assert result.summary["self_matches_excluded"] == 1
    assert result.summary["unavailable_query_records"] == 1
    assert result.summary["unavailable_gallery_records"] == 1


def test_ties_are_id_ordered_across_chunks_and_score_blocks_are_bounded(monkeypatch):
    original = np.einsum
    shapes = []

    def observe(expression, queries, gallery, **kwargs):
        shapes.append((len(queries), len(gallery)))
        return original(expression, queries, gallery, **kwargs)

    monkeypatch.setattr(np, "einsum", observe)
    result = top_k_indices(
        np.tile(np.array([1, 0], dtype="float32"), (5, 1)),
        np.tile(np.array([1, 0], dtype="float32"), (6, 1)),
        query_ids=["a", "w", "x", "y", "z"], gallery_ids=["f", "d", "c", "a", "b", "e"],
        query_available=np.ones(5, dtype=bool), gallery_available=np.ones(6, dtype=bool),
        top_k=20, query_chunk_size=2, gallery_chunk_size=2,
    )
    assert shapes and all(q <= 2 and g <= 2 for q, g in shapes)
    np.testing.assert_array_equal(result.indices[0], [4, 2, 1, 5, 0])
    np.testing.assert_array_equal(result.indices[1], [3, 4, 2, 1, 5, 0])


@pytest.mark.parametrize("count,available", [(0, []), (2, [False, False]), (1, [True])])
def test_empty_unavailable_and_self_only_queries_are_reported(count, available):
    result = top_k_indices(
        np.tile(np.array([1, 0], dtype="float32"), (count, 1)),
        np.array([[1, 0]], dtype="float32"),
        query_ids=["same", "other"][:count], gallery_ids=["same"],
        query_available=np.array(available, dtype=bool), gallery_available=np.array([True]),
    )
    assert all(not len(values) for values in result.indices)
    assert result.summary["ranking_rows"] == 0
    assert result.summary["queries_without_candidates"] == int(count == 1)


@pytest.mark.parametrize("field,value", [
    ("top_k", True), ("top_k", 0), ("top_k", 1.5),
    ("query_chunk_size", -1), ("gallery_chunk_size", 0),
    ("query_ids", [""]), ("query_ids", [3]), ("gallery_ids", ["g", "g"]),
    ("query_available", [1]), ("gallery_available", [False, False]),
    ("query_vectors", np.array([[0, 0]], dtype="float32")),
    ("query_vectors", np.array([[1, 0]], dtype="int64")),
    ("query_vectors", np.array([[np.nan, 0]], dtype="float32")),
])
def test_invalid_retrieval_contracts(field, value):
    arguments = dict(
        query_vectors=np.array([[1, 0]], dtype="float32"),
        gallery_vectors=np.eye(2, dtype="float32"), query_ids=["q"], gallery_ids=["g", "h"],
        query_available=np.array([True]), gallery_available=np.array([True, True]),
    )
    arguments[field] = value
    with pytest.raises(ContractError):
        top_k_indices(**arguments)


def test_file_retrieval_roundtrip_exclusive_outputs_and_model_compatibility(tmp_path):
    query = EmbeddingArtifact(("q",), np.array([[1, 0]], dtype="float32"), np.array([True]), _metadata())
    gallery = EmbeddingArtifact(("b", "a"), np.array([[1, 0], [1, 0]], dtype="float32"),
                                np.array([True, True]), _metadata())
    save_embeddings(query, tmp_path / "queries.npz")
    save_embeddings(gallery, tmp_path / "gallery.npz")
    result = retrieve_from_files(tmp_path / "queries.npz", tmp_path / "gallery.npz", tmp_path / "ranks.csv")
    assert list(result.rankings.candidate_id) == ["a", "b"]
    pd.testing.assert_frame_equal(pd.read_csv(tmp_path / "ranks.csv"), result.rankings)
    assert json.loads((tmp_path / "ranks.report.json").read_text())["ranking_rows"] == 2
    before = (tmp_path / "ranks.csv").read_bytes()
    with pytest.raises(ContractError, match="exists"):
        retrieve_from_files(tmp_path / "queries.npz", tmp_path / "gallery.npz", tmp_path / "ranks.csv")
    assert (tmp_path / "ranks.csv").read_bytes() == before
    incompatible = replace(gallery, metadata={**gallery.metadata, "weights_sha256": "b" * 64})
    save_embeddings(incompatible, tmp_path / "other.npz")
    with pytest.raises(ContractError, match="weights_sha256"):
        retrieve_from_files(tmp_path / "queries.npz", tmp_path / "other.npz", tmp_path / "bad.csv")
    assert not (tmp_path / "bad.csv").exists()


def test_same_weight_digest_with_different_attention_heads_is_incompatible(tmp_path):
    metadata = _metadata()
    metadata["variant"] = "v2"
    metadata["fusion_config"]["quality_dimension"] = 1
    metadata["feature_schema"]["quality_names"] = {name: ["quality-v1"] for name in MODALITY_ORDER}
    query = EmbeddingArtifact(("q",), np.array([[1, 0]], dtype="float32"), np.array([True]), metadata)
    changed = {**metadata, "fusion_config": {**metadata["fusion_config"], "attention_heads": 8}}
    gallery = EmbeddingArtifact(("g",), np.array([[1, 0]], dtype="float32"), np.array([True]), changed)
    save_embeddings(query, tmp_path / "query.npz")
    save_embeddings(gallery, tmp_path / "gallery.npz")
    with pytest.raises(ContractError, match="fusion_config"):
        retrieve_from_files(tmp_path / "query.npz", tmp_path / "gallery.npz", tmp_path / "ranking.csv")
    assert not (tmp_path / "ranking.csv").exists()
