"""A tiny, portable LLM/embedding provider shim for the judge-based answer metrics.

The harness core must stay portable, so it deliberately does *not* import SuperBot's
``llm/factory.py``. Instead it ships this small shim that auto-selects a provider from
whatever API key is present in the environment. That mirrors the repo's own env
(OpenAI / Azure / Anthropic) without coupling to it.

Contract for callers (the judge metrics in ``metrics/answer.py``):
  * ``get_judge_model()`` / ``get_embedding_model()`` return a model or **None** when no
    key is configured. None means "skip those metrics", never "crash" — which is exactly
    why Mock/CI runs need no keys at all.
  * Provider SDKs are imported lazily inside each branch, so ``pip install rageval``
    (core only) never needs any of them. A missing SDK with a present key *is* an error
    (the user clearly intended that provider) and says how to fix it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Protocol


class JudgeModel(Protocol):
    """The minimal surface the judge metrics need: name it, and complete a prompt."""

    provider: str
    model: str

    def complete(self, prompt: str, *, temperature: float = 0.0) -> str: ...


class EmbeddingModel(Protocol):
    """Embeds one text into a vector; used for the judge-free similarity metrics."""

    provider: str
    model: str

    def embed(self, text: str) -> list[float]: ...


@dataclass(slots=True)
class ProviderChoice:
    """Which provider/model the shim resolved from the environment."""

    provider: str
    model: str


def detect_provider() -> ProviderChoice | None:
    """Pick a chat provider from env keys, preferring the repo's default order.

    Order: OpenAI → Azure OpenAI → Anthropic. Returns None if no key is present, which
    the caller treats as "no judge available" (metrics degrade, run still succeeds).
    """
    if os.getenv("OPENAI_API_KEY"):
        return ProviderChoice("openai", os.getenv("RAGEVAL_JUDGE_MODEL", "gpt-4o-mini"))
    if os.getenv("AZURE_OPENAI_API_KEY") and os.getenv("AZURE_OPENAI_ENDPOINT"):
        deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")
        return ProviderChoice("azure", os.getenv("RAGEVAL_JUDGE_MODEL", deployment))
    if os.getenv("ANTHROPIC_API_KEY"):
        return ProviderChoice(
            "anthropic", os.getenv("RAGEVAL_JUDGE_MODEL", "claude-opus-4-8")
        )
    return None


class _OpenAIJudge:
    """OpenAI / Azure OpenAI chat-completions judge (both use the ``openai`` SDK)."""

    def __init__(self, provider: str, model: str) -> None:
        self.provider = provider
        self.model = model
        try:
            import openai
        except ImportError as exc:  # pragma: no cover - env-specific
            raise RuntimeError(
                f"{provider!r} judge selected (key found in env) but the 'openai' package "
                "is not installed. Fix: pip install openai"
            ) from exc
        if provider == "azure":
            self._client: Any = openai.AzureOpenAI(
                api_key=os.environ["AZURE_OPENAI_API_KEY"],
                azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
                api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-06-01"),
            )
        else:
            self._client = openai.OpenAI()

    def complete(self, prompt: str, *, temperature: float = 0.0) -> str:
        response = self._client.chat.completions.create(
            model=self.model,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        return str(response.choices[0].message.content or "")


class _AnthropicJudge:
    """Anthropic messages-API judge."""

    def __init__(self, model: str) -> None:
        self.provider = "anthropic"
        self.model = model
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover - env-specific
            raise RuntimeError(
                "'anthropic' judge selected (key found in env) but the 'anthropic' package "
                "is not installed. Fix: pip install anthropic"
            ) from exc
        self._client = anthropic.Anthropic()

    def complete(self, prompt: str, *, temperature: float = 0.0) -> str:
        response = self._client.messages.create(
            model=self.model,
            max_tokens=1024,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        parts = [block.text for block in response.content if hasattr(block, "text")]
        return "".join(parts)


def get_judge_model() -> JudgeModel | None:
    """Return a ready judge model, or None when no provider key is configured."""
    choice = detect_provider()
    if choice is None:
        return None
    if choice.provider in ("openai", "azure"):
        return _OpenAIJudge(choice.provider, choice.model)
    return _AnthropicJudge(choice.model)


class _OpenAIEmbedder:
    """OpenAI / Azure OpenAI embeddings (both via the ``openai`` SDK)."""

    def __init__(self, provider: str, model: str) -> None:
        self.provider = provider
        self.model = model
        try:
            import openai
        except ImportError as exc:  # pragma: no cover - env-specific
            raise RuntimeError(
                f"{provider!r} embeddings selected (key found in env) but the 'openai' "
                "package is not installed. Fix: pip install openai"
            ) from exc
        if provider == "azure":
            self._client: Any = openai.AzureOpenAI(
                api_key=os.environ["AZURE_OPENAI_API_KEY"],
                azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
                api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-06-01"),
            )
        else:
            self._client = openai.OpenAI()

    def embed(self, text: str) -> list[float]:
        response = self._client.embeddings.create(model=self.model, input=text)
        return list(response.data[0].embedding)


def get_embedding_model() -> EmbeddingModel | None:
    """Return an embedding model, or None when no supported key is configured.

    Only OpenAI/Azure are wired (Anthropic has no embeddings API); embedding-based
    metrics simply skip when this returns None.
    """
    model = os.getenv("RAGEVAL_EMBED_MODEL", "text-embedding-3-small")
    if os.getenv("OPENAI_API_KEY"):
        return _OpenAIEmbedder("openai", model)
    if os.getenv("AZURE_OPENAI_API_KEY") and os.getenv("AZURE_OPENAI_ENDPOINT"):
        deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", model)
        return _OpenAIEmbedder("azure", deployment)
    return None
