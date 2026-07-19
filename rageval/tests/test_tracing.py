"""Tracing — keyless local JSON by default, remote only when configured, never fatal."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rageval.core.adapter import RetrievedChunk
from rageval.core.results import QueryRecord, RunResult
from rageval.core.tracing import (
    TRACE_FILE,
    LocalJSONTracer,
    get_tracer,
    langfuse_configured,
)

_LANGFUSE_ENV = ["LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY"]


@pytest.fixture(autouse=True)
def _clear_langfuse_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in _LANGFUSE_ENV:
        monkeypatch.delenv(key, raising=False)


def _run() -> RunResult:
    rec = QueryRecord(
        question="What is the capital of France?",
        answer="Paris.",
        retrieved=[RetrievedChunk(id="geo-1", text="Paris is the capital.", score=0.9)],
        latency_ms=12.5,
        retrieval={"recall@1": 1.0},
        answer_metrics={"token_f1": 0.5},
    )
    return RunResult(run_id="run-xyz", target_name="mock", tier="answer+retrieved", records=[rec])


class TestLocalJSONTracer:
    def test_writes_trace_with_spans(self, tmp_path: Path) -> None:
        tracer = LocalJSONTracer(tmp_path)
        path = tracer.trace_run(_run(), {"git_sha": "abc", "model": "gpt-x", "provider": "openai"})
        assert path is not None and path.name == TRACE_FILE
        trace = json.loads(path.read_text(encoding="utf-8"))
        assert trace["run_id"] == "run-xyz"
        assert trace["git_sha"] == "abc"
        assert len(trace["spans"]) == 1
        span = trace["spans"][0]
        assert span["question"].startswith("What is the capital")
        assert span["retrieved_ids"] == ["geo-1"]
        assert span["retrieval"] == {"recall@1": 1.0}
        assert span["answer_metrics"] == {"token_f1": 0.5}

    def test_lands_in_run_directory(self, tmp_path: Path) -> None:
        tracer = LocalJSONTracer(tmp_path)
        tracer.trace_run(_run(), {})
        assert (tmp_path / "run-xyz" / TRACE_FILE).exists()


class TestGetTracer:
    def test_defaults_to_local_without_keys(self, tmp_path: Path) -> None:
        assert not langfuse_configured()
        tracer = get_tracer(tmp_path)
        assert tracer.name == "local-json"

    def test_partial_keys_stay_local(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        # Only one of the two required keys → not configured, stay local.
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk")
        assert not langfuse_configured()
        assert get_tracer(tmp_path).name == "local-json"

    def test_missing_sdk_falls_back_to_local(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # Both keys set but the langfuse SDK isn't installed in the test env → graceful
        # fallback to local JSON with a notice, never a crash.
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk")
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk")
        try:
            import langfuse  # noqa: F401
        except ImportError:
            tracer = get_tracer(tmp_path)
            assert tracer.name == "local-json"
            assert "falling back to local JSON" in capsys.readouterr().out
        else:  # pragma: no cover - only when the optional SDK is installed
            pytest.skip("langfuse installed; fallback path not exercised")
