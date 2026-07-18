"""PythonCallableAdapter: wrapping a query callable and a retriever callable."""

from __future__ import annotations

import pytest

from rageval.adapters.python_callable import PythonCallableAdapter
from rageval.core.adapter import RAGResult, RetrievedChunk


def _pipeline(question: str, **_ctx: object) -> RAGResult:
    return RAGResult(
        answer=f"answer to {question}",
        retrieved=[RetrievedChunk(id="d1", text="ctx", score=0.5)],
    )


def _retriever(question: str, **_ctx: object) -> list[RetrievedChunk]:
    return [RetrievedChunk(id="d1", text="a"), RetrievedChunk(id="d2", text="b")]


def test_query_kind_returns_result_and_fills_latency() -> None:
    adapter = PythonCallableAdapter(_pipeline, name="pipe")
    result = adapter.query("q")
    assert result.answer == "answer to q"
    assert result.latency_ms > 0  # filled in because the callable didn't set it
    assert adapter.retrieve("q", 1) == [RetrievedChunk(id="d1", text="ctx", score=0.5)]


def test_retrieve_kind_wraps_chunks_answer_empty() -> None:
    adapter = PythonCallableAdapter(_retriever, name="retr", kind="retrieve")
    result = adapter.query("q")
    assert result.answer == ""
    assert [c.id for c in result.retrieved] == ["d1", "d2"]
    assert [c.id for c in adapter.retrieve("q", 1)] == ["d1"]


def test_from_import_resolves_callable() -> None:
    adapter = PythonCallableAdapter.from_import(
        "tests.test_python_callable:_pipeline", name="imported"
    )
    assert adapter.query("x").answer == "answer to x"


def test_from_import_rejects_bad_path() -> None:
    with pytest.raises(ValueError, match="module:callable"):
        PythonCallableAdapter.from_import("no_colon_here")


def test_query_kind_rejects_non_result() -> None:
    bad = PythonCallableAdapter(lambda q, **_: "not a result", name="bad")  # type: ignore[arg-type,return-value]
    with pytest.raises(TypeError):
        bad.query("q")
