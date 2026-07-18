"""Answer aggregation over a RunResult: per-metric means over real observations only."""

from __future__ import annotations

import pytest

from rageval.core.results import QueryRecord, RunResult
from rageval.metrics.evaluate import format_answer_table, score_answers


class StubEmbedder:
    provider = "stub"
    model = "stub"

    def __init__(self, mapping: dict[str, list[float]]) -> None:
        self.mapping = mapping

    def embed(self, text: str) -> list[float]:
        return self.mapping[text]


def _run(records: list[QueryRecord]) -> RunResult:
    return RunResult(run_id="t", target_name="mock", tier="answer", records=records)


def test_lexical_aggregate_without_providers() -> None:
    run = _run(
        [
            QueryRecord(question="q1", answer="Paris", reference_answer="Paris"),
            QueryRecord(question="q2", answer="London", reference_answer="Berlin"),
        ]
    )
    agg = score_answers(run)
    # q1 token_f1 = 1.0, q2 token_f1 = 0.0 → mean 0.5
    assert agg["token_f1"] == pytest.approx(0.5)
    assert run.answer_aggregate == agg
    assert run.to_dict()["answer_aggregate"]["token_f1"] == pytest.approx(0.5)


def test_metric_averaged_only_over_records_that_produced_it() -> None:
    # Only the second record has a reference answer, so token_f1 has a sample of one.
    run = _run(
        [
            QueryRecord(question="q1", answer="Paris", reference_answer=None),
            QueryRecord(question="q2", answer="Paris", reference_answer="Paris"),
        ]
    )
    agg = score_answers(run)
    # Mean must be over the single record that had a reference (1.0), not diluted to 0.5.
    assert agg["token_f1"] == pytest.approx(1.0)


def test_embedding_tier_included_when_embedder_present() -> None:
    run = _run([QueryRecord(question="q", answer="a", reference_answer="b")])
    emb = StubEmbedder({"a": [1.0, 0.0], "b": [1.0, 0.0]})
    agg = score_answers(run, embedder=emb)
    assert agg["semantic_similarity"] == pytest.approx(1.0)


def test_empty_when_nothing_scorable() -> None:
    run = _run([QueryRecord(question="q", answer="a", reference_answer=None)])
    assert score_answers(run) == {}
    assert run.answer_aggregate == {}


def test_format_answer_table_orders_and_labels() -> None:
    table = format_answer_table({"token_f1": 0.5, "faithfulness": 0.9})
    lines = table.splitlines()
    # Judge tier (faithfulness) ranks above the lexical floor (token_f1) in display order.
    body = [ln for ln in lines if "faithfulness" in ln or "token_f1" in ln]
    assert body[0].startswith("faithfulness")
    assert body[1].startswith("token_f1")


def test_format_answer_table_empty_notice() -> None:
    assert "no answer metrics" in format_answer_table({})
