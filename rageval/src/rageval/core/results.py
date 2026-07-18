"""Result types — the harness's own record of a run, decoupled from any adapter.

A ``RunResult`` is what every report, comparison, and regression gate reads. It holds
one ``QueryRecord`` per question plus run-level operational rollups (latency, cost)
that are *always* available because they come straight from ``RAGResult`` — no golden
labels or LLM judge required. Metric slots (``retrieval``/``answer``) are left empty in
M1 and filled by the metric layers in later milestones without changing this schema.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from statistics import median
from typing import Any

from rageval.core.adapter import RAGResult, RetrievedChunk


@dataclass(slots=True)
class QueryRecord:
    """Everything observed for a single evaluated question.

    ``retrieved`` keeps ids/scores (not just text) so retrieval metrics in M2 can score
    against golden ``relevant_doc_ids`` without re-querying the target. ``retrieval`` and
    ``answer`` are open metric maps populated by later milestones.
    """

    question: str
    answer: str
    retrieved: list[RetrievedChunk] = field(default_factory=list)
    latency_ms: float = 0.0
    cost_usd: float | None = None
    reference_answer: str | None = None
    relevant_doc_ids: list[str] = field(default_factory=list)
    retrieval: dict[str, float] = field(default_factory=dict)
    answer_metrics: dict[str, float] = field(default_factory=dict)
    raw: dict[str, Any] | None = None

    @classmethod
    def from_result(
        cls,
        question: str,
        result: RAGResult,
        *,
        reference_answer: str | None = None,
        relevant_doc_ids: list[str] | None = None,
    ) -> QueryRecord:
        """Build a record from a target's ``RAGResult`` plus any golden labels."""
        return cls(
            question=question,
            answer=result.answer,
            retrieved=list(result.retrieved),
            latency_ms=result.latency_ms,
            cost_usd=result.cost_usd,
            reference_answer=reference_answer,
            relevant_doc_ids=list(relevant_doc_ids or []),
            raw=result.raw,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _percentile(values: list[float], pct: float) -> float:
    """Nearest-rank percentile on a small sample. Stdlib-only, deterministic.

    We roll our own instead of pulling numpy: the harness core must stay tiny, and for
    the handful of queries in a run the nearest-rank method is exact and dependency-free.
    """
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    # Nearest-rank: rank = ceil(pct/100 * N), 1-indexed.
    rank = max(1, min(len(ordered), -(-int(pct) * len(ordered) // 100)))
    return ordered[rank - 1]


@dataclass(slots=True)
class RunResult:
    """A whole evaluation run: per-query records + operational rollups + the tier used."""

    run_id: str
    target_name: str
    tier: str
    records: list[QueryRecord] = field(default_factory=list)
    # Run-mean retrieval metrics over labelled records; populated by metrics.score_run
    # (M2). Empty when the run has no golden labels (answer-only / unlabelled tiers).
    retrieval_aggregate: dict[str, float] = field(default_factory=dict)
    # Run-mean answer metrics; populated by metrics.score_answers (M4). Each metric is
    # averaged only over records that actually produced it, so a judge outage or a missing
    # reference answer shrinks a metric's sample rather than dragging its mean toward zero.
    answer_aggregate: dict[str, float] = field(default_factory=dict)

    # --- operational rollups (always available, no labels/judge needed) ---
    @property
    def latencies(self) -> list[float]:
        return [r.latency_ms for r in self.records]

    @property
    def latency_p50_ms(self) -> float:
        return median(self.latencies) if self.latencies else 0.0

    @property
    def latency_p95_ms(self) -> float:
        return _percentile(self.latencies, 95)

    @property
    def total_cost_usd(self) -> float | None:
        costs = [r.cost_usd for r in self.records if r.cost_usd is not None]
        return sum(costs) if costs else None

    def operational(self) -> dict[str, float | None]:
        """The cost/latency table required as a run output (spec §4)."""
        return {
            "queries": len(self.records),
            "latency_p50_ms": round(self.latency_p50_ms, 2),
            "latency_p95_ms": round(self.latency_p95_ms, 2),
            "total_cost_usd": self.total_cost_usd,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "target_name": self.target_name,
            "tier": self.tier,
            "operational": self.operational(),
            "retrieval_aggregate": self.retrieval_aggregate,
            "answer_aggregate": self.answer_aggregate,
            "records": [r.to_dict() for r in self.records],
        }
