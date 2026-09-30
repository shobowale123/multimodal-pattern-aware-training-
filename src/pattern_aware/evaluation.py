from __future__ import annotations
from dataclasses import dataclass
from typing import Mapping, Any, Iterable, Sequence
import numpy as np
import pandas as pd
from .config import ContractError
from .data import unit_rows
from .retrieval import top_k_indices

@dataclass(frozen=True)
class RetrievalEvaluation:
    summary: Mapping[str, Any]
    per_query: pd.DataFrame
    per_pattern: pd.DataFrame


def evaluate_retrieval(
    embeddings: np.ndarray,
    pattern_ids: Sequence[object],
    *,
    record_ids: Sequence[object],
    available: Sequence[bool] | np.ndarray,
    k_values: Iterable[int] = (1, 5, 10, 30),
    query_chunk_size: int = 128,
    gallery_chunk_size: int = 1024,
) -> RetrievalEvaluation:
    matrix = np.asarray(embeddings, dtype="float32")
    labels = np.asarray([str(v) for v in pattern_ids], dtype=object)
    identifiers = np.asarray([str(v) for v in record_ids], dtype=object)
    usable = np.asarray(available)
    if matrix.ndim != 2 or not np.isfinite(matrix).all():
        raise ContractError("embeddings must be a finite two-dimensional matrix")
    if labels.shape != (len(matrix),) or identifiers.shape != (len(matrix),):
        raise ContractError("Labels and record identifiers must match the embedding count")
    if usable.shape != (len(matrix),) or usable.dtype != np.bool_:
        raise ContractError("available must be a Boolean vector matching embeddings")
    if len(set(identifiers)) != len(identifiers):
        raise ContractError("Record identifiers must be unique")
    ks_input = tuple(k_values)
    if not ks_input or any(isinstance(k, (bool, np.bool_)) or not isinstance(k, (int, np.integer))
                           or k < 1 for k in ks_input):
        raise ContractError("k_values must contain positive integers")

    selected = np.flatnonzero(usable)
    if len(selected) < 2 or np.any(np.linalg.norm(matrix[selected], axis=1) == 0):
        raise ContractError("Retrieval needs at least two available nonzero embeddings")
    matrix = unit_rows(matrix[selected])
    labels = labels[selected]
    identifiers = identifiers[selected]

    group_counts = pd.Series(labels).value_counts()
    query_indices = np.flatnonzero(
        np.asarray([group_counts[label] >= 2 for label in labels], dtype=bool)
    )
    if not len(query_indices):
        raise ContractError("Retrieval needs at least one pattern with two available members")

    ks = tuple(sorted({int(k) for k in ks_input}))
    maximum_k = min(max(ks), matrix.shape[0] - 1)
    records: list[dict[str, Any]] = []

    ranking = top_k_indices(
        matrix[query_indices], matrix,
        query_ids=identifiers[query_indices], gallery_ids=identifiers,
        query_available=np.ones(len(query_indices), dtype=bool),
        gallery_available=np.ones(len(matrix), dtype=bool),
        top_k=maximum_k, query_chunk_size=query_chunk_size,
        gallery_chunk_size=gallery_chunk_size,
    )

    for local_position, query_index in enumerate(query_indices):
        ranked = ranking.indices[local_position]
        query_label = labels[query_index]
        relevance = labels[ranked] == query_label
        total_relevant = int(group_counts[query_label] - 1)
        record: dict[str, Any] = {
            "record_id": identifiers[query_index],
            "pattern_id": query_label,
            "relevant_gallery_records": total_relevant,
        }

        for requested_k in ks:
            effective_k = min(requested_k, maximum_k)
            hits = relevance[:effective_k].astype("float64")
            hit_count = int(hits.sum())
            record[f"RECALL@{requested_k}"] = hit_count / total_relevant

            discounts = 1.0 / np.log2(np.arange(2, effective_k + 2))
            dcg = float(np.sum(hits * discounts))
            ideal_count = min(total_relevant, effective_k)
            idcg = float(np.sum(discounts[:ideal_count]))
            record[f"NDCG@{requested_k}"] = dcg / idcg if idcg else 0.0

            precisions = np.cumsum(hits) / np.arange(1, effective_k + 1)
            denominator = min(total_relevant, effective_k)
            record[f"AP@{requested_k}"] = (
                float(np.sum(precisions * hits)) / denominator if denominator else 0.0
            )
        records.append(record)

    per_query = pd.DataFrame(records)
    metric_columns = [
        c for c in per_query.columns if c.startswith(("RECALL@", "NDCG@", "AP@"))
    ]
    per_pattern = per_query.groupby("pattern_id", sort=True)[metric_columns].mean().reset_index()
    summary = {
        "available_gallery_records": int(len(matrix)),
        "eligible_query_records": int(len(per_query)),
        "eligible_patterns": int(len(per_pattern)),
        "self_match_removed": True,
        "requested_k_values": list(ks),
        "effective_k_values": {str(k): min(k, len(matrix) - 1) for k in ks},
        "ap_convention": "truncated_denominator_min_relevant_k",
        "pattern_macro": {c: float(per_pattern[c].mean()) for c in metric_columns},
        "query_micro": {c: float(per_query[c].mean()) for c in metric_columns},
    }
    return RetrievalEvaluation(summary, per_query, per_pattern)
