"""Metrics — implemented from first principles, unit-tested on hand-checked fixtures.

Retrieval metrics (M2) and answer-quality metrics (M4: lexical token-F1/ROUGE-L,
embedding similarity, and the LLM-judge faithfulness / relevance / context precision &
recall) both live here. Operational metrics (latency/cost) are captured for free by the
runner from every ``RAGResult``.
"""

from rageval.metrics.answer import score_answer_record
from rageval.metrics.evaluate import (
    DEFAULT_KS,
    format_answer_table,
    format_retrieval_table,
    score_answers,
    score_record,
    score_run,
)
from rageval.metrics.lexical import cosine_similarity, rouge_l, token_f1
from rageval.metrics.retrieval import (
    hit_rate_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)

__all__ = [
    "DEFAULT_KS",
    "cosine_similarity",
    "format_answer_table",
    "format_retrieval_table",
    "hit_rate_at_k",
    "ndcg_at_k",
    "precision_at_k",
    "recall_at_k",
    "reciprocal_rank",
    "rouge_l",
    "score_answer_record",
    "score_answers",
    "score_record",
    "score_run",
    "token_f1",
]
