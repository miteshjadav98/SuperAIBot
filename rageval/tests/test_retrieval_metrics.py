"""Hand-checked fixtures for the retrieval metrics.

Every expected value below is worked out by hand in the comments so the metric
implementation is pinned to arithmetic a reviewer can verify without running anything.
"""

from __future__ import annotations

from math import log2

import pytest

from rageval.metrics.retrieval import (
    _dedup_top_k,
    hit_rate_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)

# Canonical fixture: 5 retrieved, 2 relevant (d2 at rank 2, d4 at rank 4).
RETRIEVED = ["d1", "d2", "d3", "d4", "d5"]
RELEVANT = {"d2", "d4"}


def test_hit_rate() -> None:
    # top-3 = d1,d2,d3 → contains d2 → 1.0 ; top-1 = d1 → no relevant → 0.0
    assert hit_rate_at_k(RETRIEVED, RELEVANT, 3) == 1.0
    assert hit_rate_at_k(RETRIEVED, RELEVANT, 1) == 0.0


def test_recall() -> None:
    # top-3 ∩ relevant = {d2} → 1/2 = 0.5 ; top-5 finds both → 2/2 = 1.0
    assert recall_at_k(RETRIEVED, RELEVANT, 3) == 0.5
    assert recall_at_k(RETRIEVED, RELEVANT, 5) == 1.0


def test_precision() -> None:
    # top-3: 1 hit / 3 = 0.333... ; top-5: 2 hits / 5 = 0.4
    assert precision_at_k(RETRIEVED, RELEVANT, 3) == pytest.approx(1 / 3)
    assert precision_at_k(RETRIEVED, RELEVANT, 5) == pytest.approx(0.4)


def test_reciprocal_rank() -> None:
    # first relevant is d2 at rank 2 → 1/2 = 0.5 ; within top-1 → none → 0.0
    assert reciprocal_rank(RETRIEVED, RELEVANT, 5) == 0.5
    assert reciprocal_rank(RETRIEVED, RELEVANT, 1) == 0.0


def test_ndcg_binary() -> None:
    # top-3 = d1(0), d2(1), d3(0). gain=(2^rel-1).
    # DCG = 1/log2(3) at rank 2 = 0.63093
    # IDCG (ideal top-3 of grades [1,1]) = 1/log2(2) + 1/log2(3) = 1 + 0.63093 = 1.63093
    dcg = 1 / log2(3)
    idcg = 1 / log2(2) + 1 / log2(3)
    assert ndcg_at_k(RETRIEVED, RELEVANT, 3) == pytest.approx(dcg / idcg)


def test_ndcg_graded_perfect_order_is_one() -> None:
    grades = {"a": 3.0, "b": 2.0, "c": 0.0}
    assert ndcg_at_k(["a", "b", "c"], grades, 3) == pytest.approx(1.0)


def test_ndcg_graded_reversed() -> None:
    grades = {"a": 3.0, "b": 2.0, "c": 0.0}
    # retrieved reversed: c(0),b(2),a(3)
    # DCG = 0 + (2^2-1)/log2(3) + (2^3-1)/log2(4) = 3/1.58496 + 7/2 = 1.89279 + 3.5 = 5.39279
    # IDCG = 7/log2(2) + 3/log2(3) + 0 = 7 + 1.89279 = 8.89279
    dcg = 3 / log2(3) + 7 / log2(4)
    idcg = 7 / log2(2) + 3 / log2(3)
    assert ndcg_at_k(["c", "b", "a"], grades, 3) == pytest.approx(dcg / idcg)


def test_perfect_retrieval_scores_one() -> None:
    retrieved = ["r1", "r2", "x", "y"]
    relevant = {"r1", "r2"}
    assert recall_at_k(retrieved, relevant, 2) == 1.0
    assert precision_at_k(retrieved, relevant, 2) == 1.0
    assert reciprocal_rank(retrieved, relevant, 2) == 1.0
    assert hit_rate_at_k(retrieved, relevant, 2) == 1.0
    assert ndcg_at_k(retrieved, relevant, 2) == pytest.approx(1.0)


def test_no_relevant_labels_scores_zero() -> None:
    assert recall_at_k(RETRIEVED, set(), 3) == 0.0
    assert hit_rate_at_k(RETRIEVED, set(), 3) == 0.0
    assert ndcg_at_k(RETRIEVED, set(), 3) == 0.0


def test_duplicates_do_not_inflate() -> None:
    # A target repeating the same hit shouldn't beat one honest hit.
    retrieved = ["d2", "d2", "d2"]
    assert recall_at_k(retrieved, RELEVANT, 3) == pytest.approx(0.5)  # only d2 of {d2,d4}
    assert precision_at_k(retrieved, RELEVANT, 3) == pytest.approx(1 / 3)


def test_dedup_top_k_helper() -> None:
    assert _dedup_top_k(["a", "a", "b", "c"], 2) == ["a", "b"]


def test_k_must_be_positive() -> None:
    with pytest.raises(ValueError):
        recall_at_k(RETRIEVED, RELEVANT, 0)
