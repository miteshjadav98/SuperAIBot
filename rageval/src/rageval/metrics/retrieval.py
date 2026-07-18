"""Retrieval metrics, implemented from first principles.

Every function here answers one question: *given the ranked list of chunk/doc ids a
target retrieved and the golden set of ids that are actually relevant, how good was the
retrieval?* We own these outright (rather than leaning on a library) because they are the
substance of the harness and because owning them lets us document each formula, its
intuition, and its failure mode right where it lives — and unit-test each on tiny,
hand-checkable fixtures (see ``tests/test_retrieval_metrics.py``).

Conventions shared by all functions:
  * ``retrieved`` is an ordered list of ids, best-ranked first. Duplicates are collapsed
    to their first occurrence so a target can't inflate a score by repeating a hit.
  * ``relevant`` is the set of ids that *should* be retrieved (binary relevance), except
    for NDCG which also accepts graded relevance.
  * ``k`` is the cutoff (top-k). ``k <= 0`` is a programming error and raises.
  * A query with no relevant ids can't be scored on recall-style metrics; those return
    ``0.0`` and callers are expected to exclude unlabelled queries from aggregation.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from math import log2


def _dedup_top_k(retrieved: Iterable[str], k: int) -> list[str]:
    """First-occurrence-dedup, then take the top ``k`` by rank. Shared by all metrics."""
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    seen: set[str] = set()
    ordered: list[str] = []
    for rid in retrieved:
        if rid not in seen:
            seen.add(rid)
            ordered.append(rid)
    return ordered[:k]


def hit_rate_at_k(retrieved: Iterable[str], relevant: set[str], k: int) -> float:
    """HitRate@k — did *any* relevant doc make the top-k? (1.0 / 0.0)

    Formula: ``1 if (top_k ∩ relevant) else 0``.
    Intuition: the coarsest "did we get anything useful in front of the model" signal;
    the yes/no floor beneath Recall.
    Failure mode: blind to how many relevant docs you found and where — a single lucky
    hit at rank k scores identically to a perfect ranking. Never use it alone.
    """
    if not relevant:
        return 0.0
    top_k = _dedup_top_k(retrieved, k)
    return 1.0 if any(rid in relevant for rid in top_k) else 0.0


def recall_at_k(retrieved: Iterable[str], relevant: set[str], k: int) -> float:
    """Recall@k — what fraction of the relevant docs did we retrieve in the top-k?

    Formula: ``|top_k ∩ relevant| / |relevant|``.
    Intuition: coverage. For RAG this is often the metric that matters most — if the
    right chunk never enters the context window, no amount of good generation recovers it.
    Failure mode: ignores rank and precision — retrieving everything gives recall 1.0
    while burying the model in noise. Read it alongside Precision/NDCG.
    """
    if not relevant:
        return 0.0
    top_k = _dedup_top_k(retrieved, k)
    hits = sum(1 for rid in top_k if rid in relevant)
    return hits / len(relevant)


def precision_at_k(retrieved: Iterable[str], relevant: set[str], k: int) -> float:
    """Precision@k — what fraction of the top-k we returned were actually relevant?

    Formula: ``|top_k ∩ relevant| / k``.
    Intuition: signal-to-noise of the context you hand the generator; low precision means
    padding the prompt with distractors (cost + a chance to mislead the model).
    Failure mode: denominator is a fixed ``k`` — if there are fewer than ``k`` relevant
    docs in the whole corpus, perfect retrieval still can't reach 1.0. Pair with Recall.
    """
    top_k = _dedup_top_k(retrieved, k)
    if not top_k:
        return 0.0
    hits = sum(1 for rid in top_k if rid in relevant)
    return hits / k


def reciprocal_rank(retrieved: Iterable[str], relevant: set[str], k: int | None = None) -> float:
    """Reciprocal rank — ``1 / rank`` of the *first* relevant doc (0 if none in cutoff).

    Formula: ``1 / rank_of_first_relevant`` (rank is 1-indexed); ``0`` if no relevant doc
    appears within ``k`` (``k=None`` means no cutoff).
    Intuition: rewards putting a relevant doc high — the mean over queries is **MRR**, the
    classic "how far must the model read before it finds something useful" measure.
    Failure mode: only the first hit counts; a query with many relevant docs is scored the
    same as one with a single hit at that rank. Use NDCG when multiple hits should matter.
    """
    cutoff = k if k is not None else len(list(retrieved))
    top = _dedup_top_k(retrieved, cutoff) if cutoff > 0 else []
    for idx, rid in enumerate(top, start=1):
        if rid in relevant:
            return 1.0 / idx
    return 0.0


def _grades(relevant: set[str] | Mapping[str, float]) -> Mapping[str, float]:
    """Normalize binary or graded relevance into an id->grade mapping (grade>=0)."""
    if isinstance(relevant, Mapping):
        return relevant
    return {rid: 1.0 for rid in relevant}


def ndcg_at_k(
    retrieved: Iterable[str],
    relevant: set[str] | Mapping[str, float],
    k: int,
) -> float:
    """NDCG@k — rank-weighted, graded gain normalized by the ideal ordering.

    Formula: ``DCG@k / IDCG@k`` where ``DCG@k = Σ_{i=1..k} (2^rel_i − 1) / log2(i + 1)``
    and IDCG is the DCG of the best possible ordering of the graded relevances.
    Intuition: the most complete retrieval metric — it credits *graded* relevance (a
    "perfect" chunk beats a "somewhat relevant" one) and discounts by position, so moving
    a good result up the list raises the score. Accepts a plain set (binary → grade 1) or
    an id->grade mapping.
    Failure mode: needs graded labels to show its full value; with binary labels it still
    works but collapses toward a position-discounted recall. IDCG=0 (no relevant docs)
    returns 0.0 by convention.
    """
    grades = _grades(relevant)
    top_k = _dedup_top_k(retrieved, k)
    dcg = sum(
        (2 ** grades.get(rid, 0.0) - 1) / log2(i + 1)
        for i, rid in enumerate(top_k, start=1)
    )
    ideal = sorted(grades.values(), reverse=True)[:k]
    idcg = sum((2**g - 1) / log2(i + 1) for i, g in enumerate(ideal, start=1))
    return dcg / idcg if idcg > 0 else 0.0
