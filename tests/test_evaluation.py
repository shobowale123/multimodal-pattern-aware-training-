import numpy as np
import pandas as pd
import pytest

from pattern_aware.config import ContractError
from pattern_aware.evaluation import evaluate_retrieval


def test_retrieval_matches_hand_computed_ranks_and_removes_self():
    # Relevant ranks are 1, 1, 3, 1. ID ordering puts a1 before b2 for b1's tie.
    embeddings = np.array([[1, 0], [0.8, 0.6], [0, 1], [-1, 0]], dtype=np.float32)
    result = evaluate_retrieval(
        embeddings,
        ["a", "a", "b", "b"],
        record_ids=["a1", "a2", "b1", "b2"],
        available=np.ones(4, dtype=bool),
        k_values=(1, 2, 3, 30),
    )
    macro = result.summary["pattern_macro"]
    assert macro["RECALL@1"] == pytest.approx(0.75)
    assert macro["RECALL@2"] == pytest.approx(0.75)
    assert macro["RECALL@3"] == pytest.approx(1.0)
    assert macro["NDCG@3"] == pytest.approx(0.875)
    assert macro["AP@3"] == pytest.approx(5 / 6)
    assert macro["RECALL@30"] == pytest.approx(1.0)
    assert macro["NDCG@30"] == pytest.approx(macro["NDCG@3"])
    assert result.summary["self_match_removed"] is True


def test_ties_are_resolved_by_record_id_not_input_order():
    embeddings = np.tile([1.0, 0.0], (4, 1))
    labels = np.array(["x", "x", "y", "y"])
    ids = np.array(["a", "b", "c", "d"])
    original = evaluate_retrieval(
        embeddings, labels, record_ids=ids, available=np.ones(4, dtype=bool), k_values=(1, 3)
    )
    permutation = np.array([3, 1, 0, 2])
    shuffled = evaluate_retrieval(
        embeddings[permutation],
        labels[permutation],
        record_ids=ids[permutation],
        available=np.ones(4, dtype=bool),
        k_values=(1, 3),
    )
    assert original.summary["pattern_macro"]["RECALL@1"] == pytest.approx(0.5)
    pd.testing.assert_frame_equal(original.per_pattern, shuffled.per_pattern)
    pd.testing.assert_frame_equal(
        original.per_query.sort_values("record_id").reset_index(drop=True),
        shuffled.per_query.sort_values("record_id").reset_index(drop=True),
    )
    assert original.summary["query_micro"] == pytest.approx(shuffled.summary["query_micro"])


def test_singleton_patterns_remain_distractors_but_are_not_queries():
    result = evaluate_retrieval(
        np.array([[1, 0], [0.8, 0.6], [0, 1], [-1, 0]], dtype=np.float32),
        ["a", "a", "b", "b"],
        record_ids=["a1", "a2", "b1", "b2"],
        available=np.array([True, True, True, False]),
        k_values=(1,),
    )
    assert len(result.per_query) == 2
    assert len(result.per_pattern) == 1
    assert result.summary["pattern_macro"]["RECALL@1"] == pytest.approx(1.0)


@pytest.mark.parametrize("available, labels", [([False, False], ["a", "a"]), ([True, True], ["a", "b"])])
def test_evaluation_rejects_no_usable_paired_queries(available, labels):
    with pytest.raises(ContractError):
        evaluate_retrieval(
            np.eye(2, dtype=np.float32), labels,
            record_ids=["a", "b"], available=np.array(available), k_values=(1,)
        )


@pytest.mark.parametrize("ks", [(), (0,), (-1,)])
def test_evaluation_rejects_empty_or_nonpositive_cutoffs(ks):
    with pytest.raises(ContractError):
        evaluate_retrieval(
            np.eye(2, dtype=np.float32), ["a", "a"],
            record_ids=["a", "b"], available=np.ones(2, dtype=bool), k_values=ks
        )
