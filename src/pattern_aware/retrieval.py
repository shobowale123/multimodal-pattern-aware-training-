"""Bounded-memory cosine retrieval with deterministic ID tie breaking."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .config import ContractError
from .data import unit_rows


def _positive_integer(name: str, value: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ContractError(f"{name} must be a positive integer")
    return int(value)


def _identifiers(values: Sequence[str], count: int, name: str) -> np.ndarray:
    values = tuple(values)
    if len(values) != count or any(not isinstance(value, str) or not value.strip() for value in values):
        raise ContractError(f"{name} must contain one nonempty string per embedding")
    if len(set(values)) != count:
        raise ContractError(f"{name} must be unique")
    return np.asarray(values, dtype=str)


def _vectors(values: np.ndarray, available: np.ndarray, name: str) -> np.ndarray:
    values = np.asarray(values)
    if values.ndim != 2 or not values.shape[1] or values.dtype.kind != "f" or not np.isfinite(values).all():
        raise ContractError(f"{name} must be a finite floating-point matrix with positive width")
    if available.shape != (len(values),) or available.dtype != np.bool_:
        raise ContractError(f"{name} availability must be a matching Boolean vector")
    # Check after conversion as very large float64 values may overflow float32.
    with np.errstate(over="ignore", invalid="ignore"):
        normalized = unit_rows(values)
    if np.any(np.linalg.norm(normalized[available], axis=1) == 0):
        raise ContractError(f"{name} available embeddings must be nonzero")
    return normalized


@dataclass(frozen=True)
class TopKResult:
    """Original gallery row indices and scores, including empty unavailable queries."""

    indices: tuple[np.ndarray, ...]
    scores: tuple[np.ndarray, ...]
    summary: Mapping[str, Any]


def top_k_indices(
    query_vectors: np.ndarray,
    gallery_vectors: np.ndarray,
    *,
    query_ids: Sequence[str],
    gallery_ids: Sequence[str],
    query_available: Sequence[bool] | np.ndarray,
    gallery_available: Sequence[bool] | np.ndarray,
    top_k: int = 10,
    query_chunk_size: int = 128,
    gallery_chunk_size: int = 1024,
) -> TopKResult:
    """Rank by cosine descending, then candidate ID ascending, excluding identical IDs.

    Score memory is O(query_chunk_size * gallery_chunk_size); retained ranks are
    O(number_of_queries * min(top_k, number_of_available_gallery_records)).
    Input vectors are normalized without modifying the caller's arrays.
    """
    top_k = _positive_integer("top_k", top_k)
    query_chunk_size = _positive_integer("query_chunk_size", query_chunk_size)
    gallery_chunk_size = _positive_integer("gallery_chunk_size", gallery_chunk_size)
    query_available = np.asarray(query_available)
    gallery_available = np.asarray(gallery_available)
    queries = _vectors(query_vectors, query_available, "query_vectors")
    gallery = _vectors(gallery_vectors, gallery_available, "gallery_vectors")
    if queries.shape[1] != gallery.shape[1]:
        raise ContractError("Query and gallery embedding dimensions must match")
    query_ids = _identifiers(query_ids, len(queries), "query_ids")
    gallery_ids = _identifiers(gallery_ids, len(gallery), "gallery_ids")
    query_rows = np.flatnonzero(query_available)
    gallery_rows = np.flatnonzero(gallery_available)
    if not len(gallery_rows):
        raise ContractError("Retrieval needs at least one available gallery embedding")
    retained_k = min(top_k, len(gallery_rows))
    indices = [np.empty(0, dtype=np.int64) for _ in range(len(queries))]
    scores = [np.empty(0, dtype=np.float32) for _ in range(len(queries))]
    self_excluded = 0

    for start in range(0, len(query_rows), query_chunk_size):
        selected_queries = query_rows[start:start + query_chunk_size]
        for gallery_start in range(0, len(gallery_rows), gallery_chunk_size):
            selected_gallery = gallery_rows[gallery_start:gallery_start + gallery_chunk_size]
            # einsum's fixed reduction order makes a pair's score independent of
            # chunk sizes, including exact ties. No full query/gallery matrix.
            block = np.einsum(
                "id,jd->ij", queries[selected_queries], gallery[selected_gallery], optimize=False
            )
            np.clip(block, -1.0, 1.0, out=block)
            self_matches = query_ids[selected_queries, None] == gallery_ids[selected_gallery][None, :]
            self_excluded += int(self_matches.sum())
            for local_query, query_row in enumerate(selected_queries):
                valid = ~self_matches[local_query]
                candidate_indices = np.concatenate((indices[query_row], selected_gallery[valid]))
                candidate_scores = np.concatenate((scores[query_row], block[local_query, valid]))
                order = np.lexsort((gallery_ids[candidate_indices], -candidate_scores))[:retained_k]
                indices[query_row] = candidate_indices[order]
                scores[query_row] = candidate_scores[order]

    summary = {
        "query_records": len(queries),
        "gallery_records": len(gallery),
        "available_query_records": int(query_available.sum()),
        "available_gallery_records": int(gallery_available.sum()),
        "unavailable_query_records": int((~query_available).sum()),
        "unavailable_gallery_records": int((~gallery_available).sum()),
        "queries_without_candidates": sum(not len(indices[row]) for row in query_rows),
        "queries_with_candidates": sum(bool(len(indices[row])) for row in query_rows),
        "self_matches_excluded": self_excluded,
        "self_match_removed": True,
        "requested_top_k": top_k,
        "ranking_rows": sum(len(values) for values in indices),
        "tie_break": "score_descending_then_candidate_id_ascending",
    }
    return TopKResult(tuple(indices), tuple(scores), summary)


@dataclass(frozen=True)
class RetrievalResult:
    rankings: pd.DataFrame
    summary: Mapping[str, Any]


def _write_exclusive_outputs(outputs: Mapping[Path, str]) -> None:
    """Create only new files, cleaning up this call's partial outputs on error."""
    paths = tuple(outputs)
    if len({path.resolve() for path in paths}) != len(paths):
        raise ContractError("Output paths must be distinct")
    if any(path.exists() for path in paths):
        raise ContractError("Output already exists; choose new output paths")
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []
    try:
        for path, content in outputs.items():
            with path.open("x", encoding="utf-8", newline="") as handle:
                created.append(path)
                handle.write(content)
    except Exception:
        for path in created:
            path.unlink(missing_ok=True)
        raise


def retrieve_from_files(
    queries: str | Path,
    gallery: str | Path,
    output: str | Path,
    *,
    top_k: int = 10,
    query_chunk_size: int = 128,
    gallery_chunk_size: int = 1024,
) -> RetrievalResult:
    """Retrieve compatible saved embeddings into a new CSV and sibling report JSON."""
    from .inference import load_embeddings

    query_artifact = load_embeddings(queries)
    gallery_artifact = load_embeddings(gallery)
    for key in ("feature_schema", "fusion_config", "weights_sha256", "variant"):
        if query_artifact.metadata[key] != gallery_artifact.metadata[key]:
            raise ContractError(f"Query and gallery {key} must match")
    result = top_k_indices(
        query_artifact.vectors, gallery_artifact.vectors,
        query_ids=query_artifact.record_ids, gallery_ids=gallery_artifact.record_ids,
        query_available=query_artifact.available, gallery_available=gallery_artifact.available,
        top_k=top_k, query_chunk_size=query_chunk_size, gallery_chunk_size=gallery_chunk_size,
    )
    rows = [
        {"query_id": query_artifact.record_ids[query_row],
         "candidate_id": gallery_artifact.record_ids[int(gallery_row)],
         "rank": rank, "score": float(score)}
        for query_row, (ranked, scores) in enumerate(zip(result.indices, result.scores, strict=True))
        for rank, (gallery_row, score) in enumerate(zip(ranked, scores, strict=True), start=1)
    ]
    rankings = pd.DataFrame(rows, columns=["query_id", "candidate_id", "rank", "score"])
    summary = {**result.summary, "weights_sha256": query_artifact.metadata["weights_sha256"]}
    output = Path(output)
    report = output.with_suffix(".report.json")
    if output == report:
        raise ContractError("Ranking output must differ from its .report.json sibling")
    _write_exclusive_outputs({
        output: rankings.to_csv(index=False),
        report: json.dumps(summary, indent=2, allow_nan=False) + "\n",
    })
    return RetrievalResult(rankings, summary)
