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

from rageval.core.results import QueryRecord, RunResult
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
        return "(no labelled queries — retrieval metrics skipped)"
    metrics = ["recall", "precision", "mrr", "ndcg", "hitrate"]
    header = "metric".ljust(11) + "".join(f"@{k}".rjust(8) for k in ks)
    lines = [header, "-" * len(header)]
    for m in metrics:
        row = m.ljust(11) + "".join(f"{aggregate.get(f'{m}@{k}', 0.0):8.3f}" for k in ks)
        lines.append(row)
    return "\n".join(lines)
