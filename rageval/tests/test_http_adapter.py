"""HTTPAdapter: the three response shapes, templating, env interpolation, and tiering.

All deterministic — no network. We inject an ``httpx.MockTransport`` so the adapter's
request-building and JSONPath response-mapping are exercised against canned responses,
which is exactly how we "prove answer-tier eval on a bare API" without a live server.
"""

from __future__ import annotations

import httpx
import pytest

from rageval.adapters.http import HTTPAdapter
from rageval.config import HTTPResponseMap, HTTPTargetConfig
from rageval.core.store import ResultsStore
from rageval.datasets.golden import GoldenRecord
from rageval.runner import (
    TIER_ANSWER_ONLY,
    TIER_LABELLED,
    run,
)


def _client(handler: object) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))  # type: ignore[arg-type]


# --- /ask shape: {answer, sources:[{id,text,score}]} ---------------------------------


def test_ask_shape_maps_answer_and_chunks() -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "answer": "Paris is the capital of France.",
                "sources": [
                    {"id": "geo-fr-01", "text": "Paris ...", "score": 0.91},
                    {"id": "geo-fr-02", "text": "France ...", "score": 0.42},
                ],
            },
        )

    cfg = HTTPTargetConfig(
        url="http://x/ask",
        method="POST",
        body={"query": "{question}", "thread_id": "rageval"},
        response=HTTPResponseMap(
            answer="$.answer",
            chunks="$.sources[*]",
            chunk_id="$.id",
            chunk_text="$.text",
            chunk_score="$.score",
        ),
    )
    adapter = HTTPAdapter(cfg, name="ask", client=_client(handler))
    result = adapter.query("What is the capital of France?")

    assert seen["body"] == {"query": "What is the capital of France?", "thread_id": "rageval"}
    assert result.answer.startswith("Paris")
    assert [c.id for c in result.retrieved] == ["geo-fr-01", "geo-fr-02"]
    assert result.retrieved[0].score == 0.91
    assert result.latency_ms >= 0


# --- bare /chat shape: {reply} → answer-only tier ------------------------------------


def test_chat_shape_is_answer_only_tier() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"reply": "Hello there."})

    cfg = HTTPTargetConfig(
        url="http://x/chat",
        body={"message": "{question}"},
        response=HTTPResponseMap(answer="$.reply"),
    )
    adapter = HTTPAdapter(cfg, name="chat", client=_client(handler))

    result = adapter.query("hi")
    assert result.answer == "Hello there."
    assert result.retrieved == []

    # End-to-end: the runner detects the ANSWER tier — retrieval metrics simply skipped.
    run_result = run(
        adapter,
        [GoldenRecord("hi", ["whatever"], "Hello there.")],
        store=ResultsStore(),
        run_id="chat-test",
    )
    assert run_result.tier == TIER_ANSWER_ONLY


# --- issue-search shape: {results:[{key,title,body}]} → retrieval-only ----------------


def test_issue_search_get_shape() -> None:
    seen: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["q"] = request.url.params["q"]
        return httpx.Response(
            200,
            json={
                "results": [
                    {"key": "PROJ-1", "title": "Login bug", "body": "cannot log in"},
                    {"key": "PROJ-2", "title": "Other", "body": "unrelated"},
                ]
            },
        )

    cfg = HTTPTargetConfig(
        url="http://x/search",
        method="GET",
        params={"q": "{question}", "top": "10"},
        response=HTTPResponseMap(chunks="$.results[*]", chunk_id="$.key", chunk_text="$.body"),
    )
    adapter = HTTPAdapter(cfg, name="issues", client=_client(handler))
    result = adapter.query("login broken")

    assert seen["q"] == "login broken"
    assert result.answer == ""  # retrieval-only: no answer mapped
    assert [c.id for c in result.retrieved] == ["PROJ-1", "PROJ-2"]

    run_result = run(
        adapter,
        [GoldenRecord("login broken", ["PROJ-1"], None)],
        store=ResultsStore(),
        run_id="issue-test",
    )
    assert run_result.tier == TIER_LABELLED


# --- env interpolation + fail-loud ---------------------------------------------------


def test_env_interpolation_in_headers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ASK_API_TOKEN", "s3cr3t")
    captured: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["auth"] = request.headers.get("authorization", "")
        return httpx.Response(200, json={"reply": "ok"})

    cfg = HTTPTargetConfig(
        url="http://x/chat",
        headers={"Authorization": "Bearer ${ASK_API_TOKEN}"},
        body={"message": "{question}"},
        response=HTTPResponseMap(answer="$.reply"),
    )
    HTTPAdapter(cfg, client=_client(handler)).query("hi")
    assert captured["auth"] == "Bearer s3cr3t"


def test_missing_env_fails_loudly(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DEFINITELY_MISSING", raising=False)

    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover - never hit
        return httpx.Response(200, json={"reply": "ok"})

    cfg = HTTPTargetConfig(
        url="http://x/chat",
        headers={"Authorization": "Bearer ${DEFINITELY_MISSING}"},
        body={"message": "{question}"},
        response=HTTPResponseMap(answer="$.reply"),
    )
    with pytest.raises(RuntimeError, match="DEFINITELY_MISSING"):
        HTTPAdapter(cfg, client=_client(handler)).query("hi")


def test_target_reported_latency_and_cost_win() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"reply": "ok", "latency_ms": 123.0, "cost_usd": 0.007})

    cfg = HTTPTargetConfig(
        url="http://x/chat",
        body={"message": "{question}"},
        response=HTTPResponseMap(
            answer="$.reply", latency_ms="$.latency_ms", cost_usd="$.cost_usd"
        ),
    )
    result = HTTPAdapter(cfg, client=_client(handler)).query("hi")
    assert result.latency_ms == 123.0
    assert result.cost_usd == 0.007
