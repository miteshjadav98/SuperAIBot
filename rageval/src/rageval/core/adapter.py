"""The adapter contract — the "USB port" every RAG target plugs into.

This module is the single most important design surface in the harness. It defines
the *minimal* interface any target must satisfy, and it deliberately imports nothing
target-specific. Adapters depend on this module; this module never depends on an
adapter. That inversion is what lets the same harness core evaluate SuperBot, a bare
`/chat` endpoint, or a local Python retriever without a single line changing here.

Design rules (see spec §2):
  * Keep the interface tiny so anything can satisfy it.
  * Missing capabilities degrade gracefully — a target that returns no ``retrieved``
    chunks simply can't be scored on retrieval metrics; the harness skips those and
    still reports answer/operational metrics. Nothing here should ever force a target
    to fabricate data it doesn't have.
  * ``**context`` carries per-target needs (auth token, user_id, thread_id, filters)
    without polluting the interface with target-specific parameters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass(slots=True)
class RetrievedChunk:
    """A single piece of context a target retrieved for a question.

    ``score`` is optional because not every target exposes ranking scores; retrieval
    metrics that need scores (e.g. NDCG) degrade to rank-based behavior when it's None.
    """

    id: str
    text: str
    score: float | None = None


@dataclass(slots=True)
class RAGResult:
    """The normalized result of asking a target one question.

    ``retrieved`` may be empty when the target is answer-only (e.g. a bare ``/chat``).
    ``cost_usd`` and ``raw`` are optional passthroughs — ``raw`` is where an adapter
    stashes trace ids or the untouched response body for debugging.
    """

    answer: str
    retrieved: list[RetrievedChunk] = field(default_factory=list)
    latency_ms: float = 0.0
    cost_usd: float | None = None
    raw: dict[str, Any] | None = None


@runtime_checkable
class RAGTarget(Protocol):
    """Every target implements this. ``retrieve`` is optional (white-box eval only).

    ``@runtime_checkable`` lets the runner do a light ``isinstance`` capability check
    for the optional ``retrieve`` method without importing any concrete adapter.
    """

    name: str

    def query(self, question: str, **context: Any) -> RAGResult:
        """Ask the target a question and return a normalized result."""
        ...

    # Optional, for white-box retrieval-only eval. Adapters that can expose the raw
    # retrieval step implement this; those that can't simply omit it and the harness
    # falls back to whatever ``query`` returns.
    def retrieve(self, question: str, k: int, **context: Any) -> list[RetrievedChunk]:
        """Return the top-``k`` retrieved chunks for a question (white-box)."""
        ...


def supports_retrieve(target: RAGTarget) -> bool:
    """True if ``target`` exposes the optional white-box ``retrieve`` method.

    Kept here (next to the contract) so callers never need to know how the capability
    check is implemented — a small guard against the Protocol growing brittle.
    """
    retrieve = getattr(type(target), "retrieve", None)
    base_retrieve = getattr(RAGTarget, "retrieve", None)
    return callable(retrieve) and retrieve is not base_retrieve
