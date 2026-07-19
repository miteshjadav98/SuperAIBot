"""Apply the retrieval metrics to a run and aggregate them.

The pure functions in ``retrieval.py`` score one query; this module bridges them to the
harness's ``RunResult``: it fills each ``QueryRecord.retrieval`` map and returns the run
mean over the *labelled* records (queries with golden ``relevant_doc_ids``). Keeping this
separate from ``runner.py`` preserves the design rule that the runner stays metric-agnostic
— metrics read a completed run, they don't change how it's produced.
"""

from __future__ import annotations

from collections.abc import Sequence
from statistics import mean

from rageval.core.provider import EmbeddingModel, JudgeModel
from rageval.core.results import QueryRecord, RunResult
from rageval.metrics.answer import score_answer_record
from rageval.metrics.retrieval import (
    hit_rate_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)

# Default cutoffs. Small values suit small golden sets; callers can override per run.
DEFAULT_KS: tuple[int, ...] = (1, 3, 5)


def score_record(record: QueryRecord, ks: Sequence[int] = DEFAULT_KS) -> dict[str, float]:
    """Compute retrieval metrics for one record and store them on it.

    Returns ``{}`` (and stores nothing) for records without golden labels — those queries
    are simply outside the retrieval tier and must not drag an aggregate toward zero.
    """
    relevant = set(record.relevant_doc_ids)
    if not relevant:
        return {}
    retrieved_ids = [c.id for c in record.retrieved]
    scores: dict[str, float] = {}
    for k in ks:
        scores[f"recall@{k}"] = recall_at_k(retrieved_ids, relevant, k)
        scores[f"precision@{k}"] = precision_at_k(retrieved_ids, relevant, k)
        scores[f"mrr@{k}"] = reciprocal_rank(retrieved_ids, relevant, k)
        scores[f"ndcg@{k}"] = ndcg_at_k(retrieved_ids, relevant, k)
        scores[f"hitrate@{k}"] = hit_rate_at_k(retrieved_ids, relevant, k)
    record.retrieval = scores
    return scores


def score_run(run: RunResult, ks: Sequence[int] = DEFAULT_KS) -> dict[str, float]:
    """Score every labelled record and set ``run.retrieval_aggregate`` to the run means.

    The aggregate averages only over records that carry labels, so mixing labelled and
    unlabelled questions in one run doesn't distort the retrieval numbers.
    """
    per_record = [score_record(r, ks) for r in run.records]
    labelled = [s for s in per_record if s]
    if not labelled:
        run.retrieval_aggregate = {}
        return {}
    keys = labelled[0].keys()
    aggregate = {key: mean(s[key] for s in labelled) for key in keys}
    run.retrieval_aggregate = aggregate
    return aggregate


def format_retrieval_table(aggregate: dict[str, float], ks: Sequence[int] = DEFAULT_KS) -> str:
    """Render the run-mean retrieval metrics as a compact fixed-width table."""
    if not aggregate:
        return "(no labelled queries - retrieval metrics skipped)"
    metrics = ["recall", "precision", "mrr", "ndcg", "hitrate"]
    header = "metric".ljust(11) + "".join(f"@{k}".rjust(8) for k in ks)
    lines = [header, "-" * len(header)]
    for m in metrics:
        row = m.ljust(11) + "".join(f"{aggregate.get(f'{m}@{k}', 0.0):8.3f}" for k in ks)
        lines.append(row)
    return "\n".join(lines)


# Display order: judge tier first (headline quality), then embedding, then lexical floor.
_ANSWER_METRIC_ORDER = (
    "faithfulness",
    "answer_relevance",
    "context_precision",
    "context_recall",
    "semantic_similarity",
    "token_f1",
    "rouge_l",
)


def score_answers(
    run: RunResult,
    *,
    judge: JudgeModel | None = None,
    embedder: EmbeddingModel | None = None,
) -> dict[str, float]:
    """Score answer metrics on every record and set ``run.answer_aggregate`` to the means.

    Each metric is averaged only over the records that actually produced it: the lexical
    floor runs whenever a ``reference_answer`` exists, the embedding and judge tiers only
    when their provider was supplied and the record's inputs allow. So a per-metric sample
    can be smaller than the record count, and a skipped metric never counts as a zero —
    the aggregate reports the mean of real observations, not a diluted average.
    """
    per_record = [
        score_answer_record(r, judge=judge, embedder=embedder) for r in run.records
    ]
    sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    for scores in per_record:
        for name, value in scores.items():
            sums[name] = sums.get(name, 0.0) + value
            counts[name] = counts.get(name, 0) + 1
    aggregate = {name: sums[name] / counts[name] for name in sums}
    run.answer_aggregate = aggregate
    return aggregate


def format_answer_table(aggregate: dict[str, float]) -> str:
    """Render the run-mean answer metrics as a two-column table (metric → mean score)."""
    if not aggregate:
        return "(no answer metrics - no reference answers and no judge/embedder available)"
    # Known metrics in a stable order, then any unexpected ones appended alphabetically.
    ordered = [m for m in _ANSWER_METRIC_ORDER if m in aggregate]
    ordered += sorted(m for m in aggregate if m not in _ANSWER_METRIC_ORDER)
    header = "answer metric".ljust(20) + "score".rjust(8)
    lines = [header, "-" * len(header)]
    for m in ordered:
        lines.append(m.ljust(20) + f"{aggregate[m]:8.3f}")
    return "\n".join(lines)
