"""Answer-quality metrics: LLM-judge, embedding, and lexical — with graceful tiers.

Three cost/capability levels, all optional per record, all skipped (never faked) when
their inputs or providers are missing:

  * **Lexical** (free, always on when a ``reference_answer`` exists): ``token_f1``,
    ``rouge_l`` — see ``lexical.py``.
  * **Embedding** (needs an embedding key): ``semantic_similarity`` (answer vs reference),
    plus an embedding component inside relevance/groundedness hybrids.
  * **LLM-judge** (needs a chat key): ``faithfulness``, ``answer_relevance``,
    ``context_precision``, ``context_recall`` — prompts versioned in ``judge_prompts.py``.

Judge replies must be strict JSON. A reply that doesn't parse means the metric is
*omitted* for that record — a deliberate contrast to systems that default parse failures
to 0.5 and quietly turn judge outages into plausible-looking scores.

The hybrid design (judge score averaged with embedding similarity, both components also
reported raw) is adapted from a production ITSM bot's online evaluation; here the
components stay visible so a drifting embedding model can't silently move a judged score.
"""

from __future__ import annotations

import json
import re
from typing import Any

from rageval.core.provider import EmbeddingModel, JudgeModel
from rageval.core.results import QueryRecord
from rageval.metrics.judge_prompts import (
    ANSWER_RELEVANCE_V1,
    CONTEXT_PRECISION_V1,
    CONTEXT_RECALL_V1,
    FAITHFULNESS_V1,
)
from rageval.metrics.lexical import cosine_similarity, rouge_l, token_f1

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _parse_judge_json(reply: str) -> dict[str, Any] | None:
    """Extract the first JSON object from a judge reply; None if unparsable.

    Tolerates surrounding prose/code fences (models add them despite instructions) but
    never invents a value: no parsable object → the caller skips the metric.
    """
    match = _JSON_RE.search(reply)
    if not match:
        return None
    try:
        obj = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def _judge_score(judge: JudgeModel, prompt: str) -> float | None:
    """Run a 0-1 score prompt; clamp to [0,1]; None on any parse failure."""
    parsed = _parse_judge_json(judge.complete(prompt))
    if parsed is None or "score" not in parsed:
        return None
    try:
        score = float(parsed["score"])
    except (TypeError, ValueError):
        return None
    return max(0.0, min(1.0, score))


def _context_text(record: QueryRecord) -> str:
    return "\n\n".join(c.text for c in record.retrieved if c.text)


def faithfulness(record: QueryRecord, judge: JudgeModel) -> float | None:
    """Is the answer grounded in the retrieved context? (judge; needs chunks + answer)."""
    context = _context_text(record)
    if not context or not record.answer:
        return None
    return _judge_score(
        judge, FAITHFULNESS_V1.render(context=context, answer=record.answer)
    )


def answer_relevance(record: QueryRecord, judge: JudgeModel) -> float | None:
    """Does the answer address the question? (judge; needs answer only)."""
    if not record.answer:
        return None
    return _judge_score(
        judge, ANSWER_RELEVANCE_V1.render(question=record.question, answer=record.answer)
    )


def context_precision(record: QueryRecord, judge: JudgeModel) -> float | None:
    """Fraction of retrieved chunks the judge deems useful for the question.

    Label-free self-eval: unlike Precision@k (which needs golden ids), the judge decides
    relevance — so this works on any target that returns chunks. Own implementation; the
    optional RAGAS wrapper offers a second opinion under a distinct metric name.
    """
    if not record.retrieved:
        return None
    chunks = "\n".join(f"[{i}] {c.text}" for i, c in enumerate(record.retrieved))
    parsed = _parse_judge_json(
        judge.complete(CONTEXT_PRECISION_V1.render(question=record.question, chunks=chunks))
    )
    if parsed is None or not isinstance(parsed.get("relevant_indices"), list):
        return None
    valid = {i for i in parsed["relevant_indices"] if isinstance(i, int)}
    relevant = {i for i in valid if 0 <= i < len(record.retrieved)}
    return len(relevant) / len(record.retrieved)


def context_recall(record: QueryRecord, judge: JudgeModel) -> float | None:
    """Fraction of the reference answer's statements supported by the retrieved context.

    Needs a ``reference_answer`` — it asks "did retrieval bring back what a correct answer
    requires?", which is undefined without knowing what correct looks like.
    """
    context = _context_text(record)
    if not context or not record.reference_answer:
        return None
    parsed = _parse_judge_json(
        judge.complete(
            CONTEXT_RECALL_V1.render(reference=record.reference_answer, context=context)
        )
    )
    if parsed is None:
        return None
    try:
        supported, total = int(parsed["supported"]), int(parsed["total"])
    except (KeyError, TypeError, ValueError):
        return None
    if total <= 0:
        return None
    return max(0.0, min(1.0, supported / total))


def semantic_similarity(record: QueryRecord, embedder: EmbeddingModel) -> float | None:
    """Embedding cosine similarity between answer and reference answer (judge-free).

    The cheap middle tier between lexical overlap and a judge: catches paraphrase that
    token-F1 misses, at one embedding call per side instead of a judged completion.
    """
    if not record.answer or not record.reference_answer:
        return None
    sim = cosine_similarity(
        embedder.embed(record.answer), embedder.embed(record.reference_answer)
    )
    # Cosine can go negative; clamp to [0,1] so all answer metrics share one scale.
    return max(0.0, min(1.0, sim))


def score_answer_record(
    record: QueryRecord,
    *,
    judge: JudgeModel | None = None,
    embedder: EmbeddingModel | None = None,
) -> dict[str, float]:
    """Compute every answer metric this record's inputs and providers allow.

    Missing provider or missing input → that metric is absent from the result (skipped,
    not zeroed). Stores the outcome on ``record.answer_metrics`` and returns it.
    """
    scores: dict[str, float] = {}

    # Lexical tier — free, needs only a reference answer.
    if record.answer and record.reference_answer:
        scores["token_f1"] = token_f1(record.answer, record.reference_answer)
        scores["rouge_l"] = rouge_l(record.answer, record.reference_answer)

    # Embedding tier.
    if embedder is not None:
        sim = semantic_similarity(record, embedder)
        if sim is not None:
            scores["semantic_similarity"] = sim

    # Judge tier.
    if judge is not None:
        for name, fn in (
            ("faithfulness", faithfulness),
            ("answer_relevance", answer_relevance),
            ("context_precision", context_precision),
            ("context_recall", context_recall),
        ):
            value = fn(record, judge)
            if value is not None:
                scores[name] = value

    record.answer_metrics = scores
    return scores
