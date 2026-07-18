"""MockAdapter — a deterministic, offline target used by every test and by CI.

Its whole reason to exist: prove the harness runs end-to-end with **zero API keys and
zero network** (spec §3, §10). Answers and retrieved chunks are derived from a hash of
the question, so the same question always yields the same result — which is what lets the
regression gate diff two runs meaningfully on a fixture instead of a live service.

It intentionally satisfies the *full* contract (both ``query`` and the optional
``retrieve``) so it can stand in for a white-box target in tiering tests.
"""

from __future__ import annotations

import hashlib
from typing import Any

from rageval.core.adapter import RAGResult, RetrievedChunk


class MockAdapter:
    """A fake RAG target with reproducible outputs.

    ``corpus`` optionally maps a chunk id -> text so tests can align mock retrieval with a
    golden set's ``relevant_doc_ids``. When omitted, synthetic chunk ids/text are minted
    deterministically from the question.
    """

    def __init__(
        self,
        name: str = "mock",
        *,
        corpus: dict[str, str] | None = None,
        k: int = 3,
        base_latency_ms: float = 12.0,
    ) -> None:
        self.name = name
        self._corpus = corpus or {}
        self._k = k
        self._base_latency_ms = base_latency_ms

    def _seed(self, question: str) -> int:
        digest = hashlib.sha256(question.encode("utf-8")).hexdigest()
        return int(digest[:8], 16)

    def retrieve(self, question: str, k: int, **context: Any) -> list[RetrievedChunk]:
        """Deterministically pick ``k`` chunks for a question."""
        seed = self._seed(question)
        if self._corpus:
            ids = sorted(self._corpus)
            picks = [ids[(seed + i) % len(ids)] for i in range(min(k, len(ids)))]
            return [
                RetrievedChunk(id=cid, text=self._corpus[cid], score=1.0 - i * 0.1)
                for i, cid in enumerate(picks)
            ]
        # No corpus supplied: mint synthetic-but-stable chunks.
        return [
            RetrievedChunk(
                id=f"mock-{(seed + i) % 1000}",
                text=f"Deterministic context #{i} for: {question}",
                score=1.0 - i * 0.1,
            )
            for i in range(k)
        ]

    def query(self, question: str, **context: Any) -> RAGResult:
        """Return a grounded-looking answer plus the retrieved chunks."""
        chunks = self.retrieve(question, self._k, **context)
        seed = self._seed(question)
        # Latency varies deterministically so p50/p95 rollups aren't degenerate.
        latency = self._base_latency_ms + (seed % 7)
        answer = f"Mock answer to '{question}' grounded in {len(chunks)} chunk(s)."
        return RAGResult(
            answer=answer,
            retrieved=chunks,
            latency_ms=float(latency),
            cost_usd=0.0,  # Mock is free — makes the cost table exercise a real number.
            raw={"adapter": self.name, "seed": seed},
        )
