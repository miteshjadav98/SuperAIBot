"""Answer metrics with a stubbed judge/embedder — no network, no keys.

The stubs return canned replies so we can pin the exact contract that matters most: a
judge reply that doesn't parse (or lacks the expected field) *skips* the metric — it is
never defaulted to a fake neutral score.
"""

from __future__ import annotations

import pytest

from rageval.core.adapter import RetrievedChunk
from rageval.core.results import QueryRecord
from rageval.metrics.answer import (
    answer_relevance,
    context_precision,
    context_recall,
    faithfulness,
    score_answer_record,
    semantic_similarity,
)


class StubJudge:
    """Returns a fixed ``complete`` reply regardless of prompt."""

    provider = "stub"
    model = "stub"

    def __init__(self, reply: str) -> None:
        self.reply = reply

    def complete(self, prompt: str, *, temperature: float = 0.0) -> str:
        return self.reply


class StubEmbedder:
    """Maps known texts to fixed vectors; raises on anything unexpected."""

    provider = "stub"
    model = "stub"

    def __init__(self, mapping: dict[str, list[float]]) -> None:
        self.mapping = mapping

    def embed(self, text: str) -> list[float]:
        return self.mapping[text]


def _record(
    *,
    answer: str = "The capital is Paris.",
    reference: str | None = None,
    chunks: list[str] | None = None,
) -> QueryRecord:
    return QueryRecord(
        question="What is the capital of France?",
        answer=answer,
        retrieved=[RetrievedChunk(id=str(i), text=t) for i, t in enumerate(chunks or [])],
        reference_answer=reference,
    )


class TestFaithfulness:
    def test_parses_score(self) -> None:
        rec = _record(chunks=["Paris is the capital of France."])
        assert faithfulness(rec, StubJudge('{"score": 0.8}')) == pytest.approx(0.8)

    def test_clamps_above_one(self) -> None:
        rec = _record(chunks=["ctx"])
        assert faithfulness(rec, StubJudge('{"score": 1.5}')) == 1.0

    def test_tolerates_prose_around_json(self) -> None:
        rec = _record(chunks=["ctx"])
        reply = 'Sure!\n```json\n{"score": 0.5, "reason": "ok"}\n```'
        assert faithfulness(rec, StubJudge(reply)) == pytest.approx(0.5)

    def test_unparsable_reply_skips(self) -> None:
        rec = _record(chunks=["ctx"])
        assert faithfulness(rec, StubJudge("the answer looks fine to me")) is None

    def test_missing_score_field_skips(self) -> None:
        rec = _record(chunks=["ctx"])
        assert faithfulness(rec, StubJudge('{"reason": "no score here"}')) is None

    def test_non_numeric_score_skips(self) -> None:
        rec = _record(chunks=["ctx"])
        assert faithfulness(rec, StubJudge('{"score": "high"}')) is None

    def test_no_context_skips(self) -> None:
        rec = _record(chunks=[])
        assert faithfulness(rec, StubJudge('{"score": 1.0}')) is None


class TestAnswerRelevance:
    def test_parses_score(self) -> None:
        rec = _record()
        assert answer_relevance(rec, StubJudge('{"score": 0.9}')) == pytest.approx(0.9)

    def test_empty_answer_skips(self) -> None:
        rec = _record(answer="")
        assert answer_relevance(rec, StubJudge('{"score": 0.9}')) is None


class TestContextPrecision:
    def test_fraction_relevant(self) -> None:
        rec = _record(chunks=["a", "b", "c"])
        # 2 of 3 chunks judged relevant.
        assert context_precision(rec, StubJudge('{"relevant_indices": [0, 2]}')) == pytest.approx(
            2 / 3
        )

    def test_out_of_range_indices_ignored(self) -> None:
        rec = _record(chunks=["a", "b"])
        # index 5 doesn't exist; only index 0 counts → 1/2.
        assert context_precision(rec, StubJudge('{"relevant_indices": [0, 5]}')) == pytest.approx(
            0.5
        )

    def test_empty_relevant_is_zero(self) -> None:
        rec = _record(chunks=["a", "b"])
        assert context_precision(rec, StubJudge('{"relevant_indices": []}')) == 0.0

    def test_malformed_skips(self) -> None:
        rec = _record(chunks=["a", "b"])
        assert context_precision(rec, StubJudge('{"relevant_indices": "0,1"}')) is None

    def test_no_chunks_skips(self) -> None:
        rec = _record(chunks=[])
        assert context_precision(rec, StubJudge('{"relevant_indices": [0]}')) is None


class TestContextRecall:
    def test_supported_over_total(self) -> None:
        rec = _record(reference="Paris is the capital.", chunks=["Paris is the capital."])
        assert context_recall(rec, StubJudge('{"supported": 2, "total": 4}')) == pytest.approx(
            0.5
        )

    def test_zero_total_skips(self) -> None:
        rec = _record(reference="ref", chunks=["ctx"])
        assert context_recall(rec, StubJudge('{"supported": 0, "total": 0}')) is None

    def test_no_reference_skips(self) -> None:
        rec = _record(reference=None, chunks=["ctx"])
        assert context_recall(rec, StubJudge('{"supported": 1, "total": 1}')) is None

    def test_malformed_skips(self) -> None:
        rec = _record(reference="ref", chunks=["ctx"])
        assert context_recall(rec, StubJudge('{"supported": "two"}')) is None


class TestSemanticSimilarity:
    def test_identical_direction(self) -> None:
        rec = _record(answer="a", reference="b")
        emb = StubEmbedder({"a": [1.0, 0.0], "b": [1.0, 0.0]})
        assert semantic_similarity(rec, emb) == pytest.approx(1.0)

    def test_orthogonal_is_zero(self) -> None:
        rec = _record(answer="a", reference="b")
        emb = StubEmbedder({"a": [1.0, 0.0], "b": [0.0, 1.0]})
        assert semantic_similarity(rec, emb) == 0.0

    def test_negative_cosine_clamped(self) -> None:
        rec = _record(answer="a", reference="b")
        emb = StubEmbedder({"a": [1.0, 0.0], "b": [-1.0, 0.0]})
        assert semantic_similarity(rec, emb) == 0.0

    def test_no_reference_skips(self) -> None:
        rec = _record(answer="a", reference=None)
        emb = StubEmbedder({"a": [1.0, 0.0]})
        assert semantic_similarity(rec, emb) is None


class TestScoreAnswerRecord:
    def test_lexical_only_without_providers(self) -> None:
        rec = _record(answer="Paris", reference="Paris")
        scores = score_answer_record(rec)
        assert set(scores) == {"token_f1", "rouge_l"}
        assert scores["token_f1"] == 1.0
        assert rec.answer_metrics == scores

    def test_empty_when_no_reference_and_no_providers(self) -> None:
        rec = _record(answer="Paris", reference=None)
        assert score_answer_record(rec) == {}

    def test_adds_embedding_tier(self) -> None:
        rec = _record(answer="a", reference="b")
        emb = StubEmbedder({"a": [1.0, 0.0], "b": [1.0, 0.0]})
        scores = score_answer_record(rec, embedder=emb)
        assert scores["semantic_similarity"] == pytest.approx(1.0)
        assert "token_f1" in scores

    def test_adds_judge_tier_only_where_inputs_allow(self) -> None:
        # Reference + chunks present, so all four judge metrics can run.
        rec = _record(answer="Paris", reference="Paris", chunks=["Paris is the capital."])
        judge = StubJudge('{"score": 0.7, "relevant_indices": [0], "supported": 1, "total": 1}')
        scores = score_answer_record(rec, judge=judge)
        assert scores["faithfulness"] == pytest.approx(0.7)
        assert scores["answer_relevance"] == pytest.approx(0.7)
        assert scores["context_precision"] == pytest.approx(1.0)
        assert scores["context_recall"] == pytest.approx(1.0)

    def test_judge_skipped_metrics_absent_not_zero(self) -> None:
        # No chunks and no reference: faithfulness/context_* cannot run and must be absent.
        rec = _record(answer="Paris", reference=None, chunks=[])
        judge = StubJudge('{"score": 0.9}')
        scores = score_answer_record(rec, judge=judge)
        assert scores == {"answer_relevance": pytest.approx(0.9)}
        assert "faithfulness" not in scores
