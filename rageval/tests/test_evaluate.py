"""Aggregation layer: metrics applied to a RunResult, averaged over labelled records."""

from __future__ import annotations

import pytest

from rageval.core.adapter import RetrievedChunk
from rageval.core.results import QueryRecord, RunResult
from rageval.metrics.evaluate import format_retrieval_table, score_record, score_run


def _record(retrieved_ids: list[str], relevant: list[str]) -> QueryRecord:
    return QueryRecord(
        question="q",
        answer="a",
        retrieved=[RetrievedChunk(id=i, text=i) for i in retrieved_ids],
        relevant_doc_ids=relevant,
    )


def test_score_record_populates_and_returns() -> None:
    rec = _record(["d1", "d2", "d3"], ["d2"])
    scores = score_record(rec, ks=(1, 3))
    assert rec.retrieval == scores
    assert scores["recall@3"] == 1.0
    assert scores["recall@1"] == 0.0
    assert scores["hitrate@3"] == 1.0


def test_score_record_skips_unlabelled() -> None:
    rec = _record(["d1", "d2"], [])
    assert score_record(rec) == {}
    assert rec.retrieval == {}


def test_score_run_averages_over_labelled_only() -> None:
    run = RunResult(
        run_id="t",
        target_name="mock",
        tier="answer+retrieved+labelled",
        records=[
            _record(["d1", "d2"], ["d1"]),  # recall@1 = 1.0 (d1 at rank 1)
            _record(["x", "d3"], ["d3"]),   # recall@1 = 0.0 (d3 at rank 2)
            _record(["a", "b"], []),        # unlabelled → excluded
        ],
    )
    agg = score_run(run, ks=(1,))
    # mean recall@1 over the 2 labelled records = (1.0 + 0.0)/2 = 0.5
    assert agg["recall@1"] == pytest.approx(0.5)
    assert run.retrieval_aggregate == agg
    # Aggregate lands in the serialized run for reports/compare downstream.
    assert run.to_dict()["retrieval_aggregate"]["recall@1"] == pytest.approx(0.5)


def test_score_run_no_labels_is_empty() -> None:
    run = RunResult(run_id="t", target_name="m", tier="answer+retrieved", records=[
        _record(["a", "b"], []),
    ])
    assert score_run(run) == {}
    assert run.retrieval_aggregate == {}


def test_format_table_shapes_output() -> None:
    table = format_retrieval_table({"recall@1": 0.5, "precision@1": 0.5}, ks=(1,))
    assert "recall" in table and "@1" in table
    assert format_retrieval_table({}) == "(no labelled queries — retrieval metrics skipped)"
