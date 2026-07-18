"""MockAdapter: determinism + contract shape. These run with no keys and no network."""

from __future__ import annotations

from rageval.adapters.mock import MockAdapter
from rageval.core.adapter import RAGResult, RetrievedChunk, supports_retrieve


def test_query_is_deterministic() -> None:
    a, b = MockAdapter(), MockAdapter()
    r1 = a.query("What is the capital of France?")
    r2 = b.query("What is the capital of France?")
    assert r1.answer == r2.answer
    assert [c.id for c in r1.retrieved] == [c.id for c in r2.retrieved]
    assert [c.score for c in r1.retrieved] == [c.score for c in r2.retrieved]


def test_different_questions_differ() -> None:
    m = MockAdapter()
    assert m.query("question one").answer != m.query("question two").answer


def test_result_shape() -> None:
    r = MockAdapter(k=3).query("hello")
    assert isinstance(r, RAGResult)
    assert r.answer
    assert len(r.retrieved) == 3
    assert all(isinstance(c, RetrievedChunk) and c.id and c.text for c in r.retrieved)
    assert r.latency_ms > 0
    assert r.cost_usd == 0.0


def test_corpus_alignment() -> None:
    corpus = {"geo-fr-01": "Paris is the capital of France.", "x": "other", "y": "more"}
    m = MockAdapter(corpus=corpus, k=2)
    chunks = m.retrieve("q", 2)
    assert len(chunks) == 2
    assert all(c.id in corpus and c.text == corpus[c.id] for c in chunks)


def test_mock_satisfies_retrieve_capability() -> None:
    assert supports_retrieve(MockAdapter())
