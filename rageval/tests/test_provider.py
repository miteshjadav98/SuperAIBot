"""Provider shim env detection — key-free paths only (no SDK calls, no network).

We test detection and the None-without-key contract that keeps Mock/CI runs keyless.
Constructing a live judge would need an installed SDK and is left to integration use.
"""

from __future__ import annotations

import pytest

from rageval.core.provider import detect_provider, get_embedding_model, get_judge_model

_KEYS = [
    "OPENAI_API_KEY",
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_ENDPOINT",
    "ANTHROPIC_API_KEY",
    "RAGEVAL_JUDGE_MODEL",
    "RAGEVAL_EMBED_MODEL",
]


@pytest.fixture(autouse=True)
def _clear_provider_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in _KEYS:
        monkeypatch.delenv(key, raising=False)


def test_no_keys_detects_nothing() -> None:
    assert detect_provider() is None
    assert get_judge_model() is None
    assert get_embedding_model() is None


def test_openai_preferred_first(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    choice = detect_provider()
    assert choice is not None
    assert choice.provider == "openai"
    assert choice.model == "gpt-4o-mini"


def test_judge_model_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("RAGEVAL_JUDGE_MODEL", "gpt-4.1")
    choice = detect_provider()
    assert choice is not None
    assert choice.model == "gpt-4.1"


def test_azure_needs_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "az-test")
    # Endpoint missing → not usable, falls through to None.
    assert detect_provider() is None
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://x.openai.azure.com")
    choice = detect_provider()
    assert choice is not None
    assert choice.provider == "azure"


def test_anthropic_last(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "ant-test")
    choice = detect_provider()
    assert choice is not None
    assert choice.provider == "anthropic"
