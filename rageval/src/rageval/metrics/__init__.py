"""Metrics — implemented from first principles, unit-tested on hand-checked fixtures.

Retrieval metrics land in M2 (this package); answer-quality metrics (faithfulness,
relevance, context precision/recall) arrive in M4. Operational metrics (latency/cost)
are captured for free by the runner from every ``RAGResult``.
"""

from rageval.metrics.evaluate import (
    DEFAULT_KS,
    format_retrieval_table,
    score_record,
    score_run,
)
from rageval.metrics.retrieval import (
    hit_rate_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)

__all__ = [
    "DEFAULT_KS",
    "format_retrieval_table",
    "hit_rate_at_k",
    "ndcg_at_k",
    "precision_at_k",
    "recall_at_k",
    "reciprocal_rank",
    "score_record",
    "score_run",
]
