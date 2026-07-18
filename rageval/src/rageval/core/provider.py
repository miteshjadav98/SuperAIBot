"""A tiny, portable LLM provider shim for the LLM-as-judge answer metrics.

The harness core must stay portable, so it deliberately does *not* import SuperBot's
``llm/factory.py``. Instead it ships this ~zero-dependency shim that auto-selects a
provider from whatever API key is present in the environment. That mirrors the repo's
own env (OpenAI / Azure / Anthropic) without coupling to it.

Contract for callers (the judge metrics land in M4):
  * ``get_judge_model()`` returns a ``JudgeModel`` or **None** when no key is configured.
  * When it returns None, the harness *skips* judge-based metrics rather than crashing —
    which is exactly why Mock/CI runs need no keys at all.
  * Provider SDKs are imported lazily inside each branch so ``pip install rageval`` (core
    only) never needs any of them.

M1 ships only the selection skeleton and the typed interface; the actual ``complete``
implementations are thin and covered when M4 wires in the judge prompts.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Protocol


class JudgeModel(Protocol):
    """The minimal surface the judge metrics need: name it, and complete a prompt."""

    provider: str
    model: str

    def complete(self, prompt: str, *, temperature: float = 0.0) -> str: ...


@dataclass(slots=True)
class ProviderChoice:
    """Which provider/model the shim resolved from the environment."""

    provider: str
    model: str


def detect_provider() -> ProviderChoice | None:
    """Pick a provider from env keys, preferring the repo's default order.

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


def get_judge_model() -> JudgeModel | None:
    """Return a ready judge model, or None when no provider key is configured.

    M1: selection + graceful-None only. The concrete ``JudgeModel`` implementations are
    added in M4 alongside the versioned judge prompts; keeping the seam here means that
    milestone slots in without touching the core or any adapter.
    """
    choice = detect_provider()
    if choice is None:
        return None
    # M4 will lazily construct and return a provider-backed JudgeModel here. Until then
    # we signal "judge unavailable" so answer metrics are skipped rather than faked.
    return None
