"""The runner — drives a target over a set of questions and records a run.

This is the harness's engine. It stays deliberately thin and metric-agnostic: iterate
questions, ask the target, normalize into ``QueryRecord``s, detect which evaluation tier
the run qualifies for, capture a manifest, and persist. Metric computation (M2+) reads the
stored ``RunResult`` afterward, so adding metrics never touches this file.
"""

from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import Any

from rageval.core.adapter import RAGResult, RAGTarget
from rageval.core.manifest import RunManifest
from rageval.core.results import QueryRecord, RunResult
from rageval.core.store import ResultsStore
from rageval.datasets.golden import GoldenRecord

# Tier names — the headline "plug into anything" feature (spec §5). Detected from what the
# target actually returned plus whether golden labels are present, never assumed.
TIER_ANSWER_ONLY = "answer"
TIER_ANSWER_RETRIEVED = "answer+retrieved"
TIER_LABELLED = "answer+retrieved+labelled"


def detect_tier(records: list[QueryRecord], *, has_golden_labels: bool) -> str:
    """Pick the richest tier the run's data can support.

    Answer-only targets can't be scored on retrieval; targets that return chunks can be
    self-scored; add golden ``relevant_doc_ids`` and full retrieval metrics unlock. We
    take the *minimum* capability across records so the reported tier is never optimistic.
    """
    any_retrieved = any(r.retrieved for r in records)
    if not any_retrieved:
        return TIER_ANSWER_ONLY
    if has_golden_labels and any(r.relevant_doc_ids for r in records):
        return TIER_LABELLED
    return TIER_ANSWER_RETRIEVED


def _timed_query(target: RAGTarget, question: str, **context: Any) -> RAGResult:
    """Call the target, filling in latency if the adapter didn't measure it itself."""
    start = time.perf_counter()
    result = target.query(question, **context)
    if not result.latency_ms:
        result.latency_ms = (time.perf_counter() - start) * 1000.0
    return result


def run(
    target: RAGTarget,
    golden: list[GoldenRecord],
    *,
    store: ResultsStore | None = None,
    config: dict[str, Any] | None = None,
    golden_path: Path | str | None = None,
    run_id: str | None = None,
    context: dict[str, Any] | None = None,
    provider: str | None = None,
    model: str | None = None,
    extra: dict[str, Any] | None = None,
) -> RunResult:
    """Evaluate ``target`` over ``golden`` and persist the run.

    ``golden`` may carry labels or just questions (``relevant_doc_ids``/``reference_answer``
    optional) — the tier is detected from what's present. Returns the ``RunResult``; also
    writes ``result.json`` + ``manifest.json`` when a store is provided.

    ``provider``/``model``/``extra`` are provenance the caller records into the manifest —
    e.g. the judge model and its prompt versions — so a scored run remains reproducible.
    The runner itself stays metric-agnostic: it records this provenance but computes no
    metrics (those read the stored ``RunResult`` afterward).
    """
    store = store or ResultsStore()
    run_id = run_id or f"run-{uuid.uuid4().hex[:12]}"
    ctx = context or {}

    records: list[QueryRecord] = []
    for g in golden:
        result = _timed_query(target, g.question, **ctx)
        records.append(
            QueryRecord.from_result(
                g.question,
                result,
                reference_answer=g.reference_answer,
                relevant_doc_ids=g.relevant_doc_ids,
            )
        )

    has_labels = any(g.relevant_doc_ids for g in golden)
    tier = detect_tier(records, has_golden_labels=has_labels)

    run_result = RunResult(
        run_id=run_id,
        target_name=target.name,
        tier=tier,
        records=records,
    )
    manifest = RunManifest.capture(
        run_id=run_id,
        tier=tier,
        target_name=target.name,
        config=config,
        golden_path=golden_path,
        provider=provider,
        model=model,
        extra=extra,
    )
    store.save(run_result, manifest)
    return run_result
