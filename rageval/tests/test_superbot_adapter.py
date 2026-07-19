"""SuperBotAdapter — black-box /ask, white-box retrieval, auth, and end-to-end tiering.

Fully offline: an ``httpx.MockTransport`` stands in for the gateway (so login + /ask request
building is exercised against canned responses), and a fake retriever stands in for the
app's ``_hybrid_retrieve`` (so the white-box path runs without importing the backend).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import httpx
import pytest

from rageval.adapters.superbot import SuperBotAdapter
from rageval.config import SuperBotConfig, TargetConfig, build_target
from rageval.core.store import ResultsStore
from rageval.datasets.golden import GoldenRecord
from rageval.runner import TIER_ANSWER_ONLY, TIER_LABELLED, run


@dataclass
class _Doc:
    """Stand-in for the app's langchain Document (has page_content + metadata)."""

    page_content: str
    metadata: dict[str, Any] = field(default_factory=dict)


def _gateway() -> httpx.Client:
    """A MockTransport that answers /auth/login and /ask like the real gateway."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/auth/login":
            body = json.loads(request.content)
            assert body == {"email": "e@x.io", "password": "pw"}
            return httpx.Response(
                200, json={"token": "tok-123", "user": {"id": "user-1", "email": "e@x.io"}}
            )
        if request.url.path == "/ask":
            assert request.headers["authorization"] == "Bearer tok-123"
            body = json.loads(request.content)
            return httpx.Response(200, json={"answer": f"Answer to: {body['query']}"})
        return httpx.Response(404)  # pragma: no cover

    return httpx.Client(transport=httpx.MockTransport(handler))  # type: ignore[arg-type]


def _creds(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SUPERBOT_TOKEN", raising=False)
    monkeypatch.setenv("SUPERBOT_EMAIL", "e@x.io")
    monkeypatch.setenv("SUPERBOT_PASSWORD", "pw")


def test_blackbox_ask_is_answer_only(monkeypatch: pytest.MonkeyPatch) -> None:
    _creds(monkeypatch)
    adapter = SuperBotAdapter(client=_gateway())
    result = adapter.query("What is RAG?")
    assert result.answer == "Answer to: What is RAG?"
    assert result.retrieved == []  # /ask returns no chunks → answer-tier
    assert result.latency_ms >= 0

    run_result = run(
        adapter,
        [GoldenRecord("What is RAG?", ["doc-a"], "Retrieval-augmented generation.")],
        store=ResultsStore(),
        run_id="sb-blackbox",
    )
    assert run_result.tier == TIER_ANSWER_ONLY


def test_token_env_skips_login(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SUPERBOT_TOKEN", "tok-123")
    monkeypatch.delenv("SUPERBOT_EMAIL", raising=False)
    monkeypatch.delenv("SUPERBOT_PASSWORD", raising=False)

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/ask"  # never hits /auth/login
        assert request.headers["authorization"] == "Bearer tok-123"
        return httpx.Response(200, json={"answer": "ok"})

    client = httpx.Client(transport=httpx.MockTransport(handler))  # type: ignore[arg-type]
    adapter = SuperBotAdapter(client=client)
    assert adapter.query("hi").answer == "ok"


def test_missing_auth_fails_loudly(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SUPERBOT_TOKEN", raising=False)
    monkeypatch.delenv("SUPERBOT_EMAIL", raising=False)
    monkeypatch.delenv("SUPERBOT_PASSWORD", raising=False)
    adapter = SuperBotAdapter(client=_gateway())
    with pytest.raises(RuntimeError, match="SuperBot auth is unset"):
        adapter.query("hi")


def test_whitebox_populates_retrieval_and_unlocks_labelled_tier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _creds(monkeypatch)

    def fake_hybrid(query: str, owner: str) -> list[_Doc]:
        assert owner == "user-1"  # resolved from the login response
        return [
            _Doc("Paris is the capital.", {"source": "geo.pdf"}),
            _Doc("Unrelated text.", {"source": "misc.pdf"}),
        ]

    adapter = SuperBotAdapter(
        client=_gateway(),
        white_box=True,
        rerank=False,
        hybrid_retrieve=fake_hybrid,
    )
    result = adapter.query("Capital of France?")
    assert result.answer.startswith("Answer to:")
    assert [c.id for c in result.retrieved] == ["geo.pdf", "misc.pdf"]

    # Golden labelled by source document → full retrieval metrics unlock.
    run_result = run(
        adapter,
        [GoldenRecord("Capital of France?", ["geo.pdf"], "Paris.")],
        store=ResultsStore(),
        run_id="sb-whitebox",
    )
    assert run_result.tier == TIER_LABELLED


def test_whitebox_applies_rerank_and_final_k(monkeypatch: pytest.MonkeyPatch) -> None:
    _creds(monkeypatch)
    docs = [_Doc(f"chunk {i}", {"source": f"d{i}.pdf"}) for i in range(6)]

    def fake_hybrid(query: str, owner: str) -> list[_Doc]:
        return docs

    def fake_rerank(query: str, candidates: list[_Doc], top_n: int) -> list[_Doc]:
        # Reverse to prove the reranked order (not the raw order) is what we keep.
        return list(reversed(candidates))[:top_n]

    adapter = SuperBotAdapter(
        client=_gateway(),
        white_box=True,
        rerank=True,
        final_k=2,
        hybrid_retrieve=fake_hybrid,
        rerank_fn=fake_rerank,
    )
    chunks = adapter.retrieve("q")
    assert [c.id for c in chunks] == ["d5.pdf", "d4.pdf"]  # reranked + truncated to final_k


def test_whitebox_needs_owner(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SUPERBOT_TOKEN", "tok-123")  # token path resolves no user id

    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover
        return httpx.Response(200, json={"answer": "ok"})

    client = httpx.Client(transport=httpx.MockTransport(handler))  # type: ignore[arg-type]
    adapter = SuperBotAdapter(
        client=client, white_box=True, hybrid_retrieve=lambda q, o: []
    )
    with pytest.raises(RuntimeError, match="needs an owner"):
        adapter.retrieve("q")


def test_build_target_from_config() -> None:
    cfg = TargetConfig(
        name="superbot",
        adapter="superbot",
        superbot=SuperBotConfig(base_url="http://host:8000/", white_box=True, final_k=3),
    )
    adapter = build_target(cfg)
    assert adapter.name == "superbot"
    # Config fingerprint carries no secrets (creds are env-only).
    fp = cfg.config_fingerprint()
    assert "superbot" in fp and "password" not in json.dumps(fp)
