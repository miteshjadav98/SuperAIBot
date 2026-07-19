"""Prove-a-lift experiment — CI guards the claim that hybrid retrieval beats BM25.

The experiment is deterministic and keyless, so these assertions pin the measured lift:
a regression in the fusion (or the metrics) would flip them red.
"""

from __future__ import annotations

from rageval.experiments.retrieval_lift import (
    CORPUS,
    GOLDEN,
    compare,
    evaluate,
    make_retrievers,
)


def test_hybrid_beats_bm25_on_headline_metrics() -> None:
    table = compare()
    # Every headline metric improves (or at least never regresses) under hybrid fusion.
    for metric, row in table.items():
        assert row["hybrid"] >= row["baseline"], f"{metric} regressed: {row}"
    # And the headline retrieval metric improves by a clear margin, not just noise.
    assert table["recall@3"]["delta"] >= 0.2
    assert table["mrr@3"]["delta"] > 0.0
    assert table["ndcg@3"]["delta"] > 0.0


def test_experiment_is_deterministic() -> None:
    # Same inputs → identical aggregates on a re-run (no hidden randomness).
    assert compare() == compare()


def test_baseline_actually_misses_some_top3() -> None:
    # The demo is only meaningful if BM25 genuinely fails on the hard queries — guard that
    # so a future corpus edit can't make the baseline trivially perfect and hide a real lift.
    baseline_fn, _hybrid_fn = make_retrievers(CORPUS)
    base = evaluate("baseline", baseline_fn, GOLDEN).retrieval_aggregate
    assert base["recall@3"] < 1.0
