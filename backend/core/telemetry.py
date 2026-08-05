"""Per-run token, latency and cost accounting.

You cannot build a cost dashboard over data you never recorded, and "what does a
request cost us" is unanswerable from traces alone once one request fans out
across several agents and a dozen model calls. This records one row per run.

Token counts come from :func:`get_usage_metadata_callback`, LangChain's
first-party aggregator: it collects ``AIMessage.usage_metadata`` from *every*
chat model call inside the block, keyed by model name. Because it propagates
through contextvars it also captures calls made inside ``Send`` workers and
sub-agents, which is exactly the case a per-call counter would miss.

**Cost is opt-in and ships with no prices.** Rates vary by provider, region and
contract — an enterprise Azure agreement is not the public list price — so
inventing defaults would produce confidently wrong numbers. Tokens are always
recorded; cost is computed only for models present in ``MODEL_PRICING``.
Backfilling cost later is a multiply over stored token counts.

Set rates in ``.env`` as USD per **million** tokens::

    MODEL_PRICING='{"gpt-4.1": {"input": 2.0, "output": 8.0}}'
"""

from __future__ import annotations

import json
import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterator

from langchain_core.callbacks import get_usage_metadata_callback

from core import db
from core.settings import settings

logger = logging.getLogger(__name__)

_TOKENS_PER_PRICE_UNIT = 1_000_000


@dataclass
class RunMetrics:
    """What one graph invocation cost."""

    run_type: str  # "chat", "ask", ...
    owner: str | None
    thread_id: str | None
    agent_id: str | None = None
    latency_ms: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float | None = None  # None when no rate is configured
    by_model: dict[str, Any] = field(default_factory=dict)
    ok: bool = True
    error: str | None = None

    def as_document(self) -> dict:
        return {
            "run_type": self.run_type,
            "owner": self.owner,
            "thread_id": self.thread_id,
            "agent_id": self.agent_id,
            "latency_ms": self.latency_ms,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "cost_usd": self.cost_usd,
            "by_model": self.by_model,
            "ok": self.ok,
            "error": self.error,
            "at": datetime.now(timezone.utc),
        }


def pricing() -> dict[str, dict[str, float]]:
    """Configured USD-per-million-token rates, keyed by model name."""
    raw = settings.model_pricing
    if not raw:
        return {}
    try:
        return json.loads(raw) if isinstance(raw, str) else dict(raw)
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        logger.warning("MODEL_PRICING is not valid JSON (%s); cost will not be computed", exc)
        return {}


def _match_rate(model_name: str, rates: dict[str, dict[str, float]]) -> dict | None:
    """Find the rate for a model, tolerating deployment-name drift.

    Providers report names like ``gpt-4.1-2025-04-14`` while the configured key
    is ``gpt-4.1``, so fall back to the longest configured key that prefixes the
    reported name. Longest wins, so ``gpt-4.1-mini`` never matches ``gpt-4.1``.
    """
    if model_name in rates:
        return rates[model_name]
    candidates = [key for key in rates if model_name.startswith(key)]
    return rates[max(candidates, key=len)] if candidates else None


def compute_cost(by_model: dict[str, Any]) -> float | None:
    """Total USD for a run, or ``None`` if no model in it has a configured rate."""
    rates = pricing()
    if not rates:
        return None

    total = 0.0
    priced_anything = False
    for model_name, usage in by_model.items():
        rate = _match_rate(model_name, rates)
        if rate is None:
            continue
        priced_anything = True
        total += (
            usage.get("input_tokens", 0) * rate.get("input", 0.0)
            + usage.get("output_tokens", 0) * rate.get("output", 0.0)
        ) / _TOKENS_PER_PRICE_UNIT

    return round(total, 6) if priced_anything else None


@contextmanager
def measure(
    run_type: str,
    *,
    owner: str | None = None,
    thread_id: str | None = None,
) -> Iterator[RunMetrics]:
    """Measure everything inside the block and persist one metrics row.

    The :class:`RunMetrics` is yielded so callers can enrich it (``agent_id``,
    for instance) before it is written on exit. Failures are recorded too — a
    run that burned tokens and then errored is exactly the one worth costing.

    Recording never raises: telemetry that can fail a request is worse than no
    telemetry.
    """
    metrics = RunMetrics(run_type=run_type, owner=owner, thread_id=thread_id)
    started = time.perf_counter()

    try:
        with get_usage_metadata_callback() as usage:
            try:
                yield metrics
            except Exception as exc:  # noqa: BLE001 — record, then re-raise
                metrics.ok = False
                metrics.error = f"{type(exc).__name__}: {exc}"
                raise
            finally:
                metrics.latency_ms = int((time.perf_counter() - started) * 1000)
                _fill(metrics, dict(usage.usage_metadata))
    finally:
        _persist(metrics)


def _fill(metrics: RunMetrics, by_model: dict[str, Any]) -> None:
    metrics.by_model = by_model
    metrics.input_tokens = sum(u.get("input_tokens", 0) for u in by_model.values())
    metrics.output_tokens = sum(u.get("output_tokens", 0) for u in by_model.values())
    metrics.total_tokens = sum(u.get("total_tokens", 0) for u in by_model.values())
    metrics.cost_usd = compute_cost(by_model)


def _persist(metrics: RunMetrics) -> None:
    collection = db.run_metrics_collection()
    if collection is None:
        return
    try:
        collection.insert_one(metrics.as_document())
    except Exception as exc:  # noqa: BLE001 — never fail a request over metrics
        logger.warning("failed to record run metrics: %s", exc)


def summary(owner: str | None = None, limit: int = 200) -> dict:
    """Aggregate recent runs. Backs ``GET /metrics``.

    Deliberately a small aggregation over the last ``limit`` runs rather than a
    full time-series: enough to answer "what is a request costing us and how
    slow is p95", without pretending to be a metrics backend.
    """
    collection = db.run_metrics_collection()
    if collection is None:
        return {"runs": 0, "note": "MongoDB is not configured, so nothing is recorded."}

    query = {"owner": owner} if owner else {}
    try:
        rows = list(collection.find(query).sort("at", -1).limit(limit))
    except Exception as exc:  # noqa: BLE001
        logger.warning("metrics summary failed: %s", exc)
        return {"runs": 0, "error": str(exc)}

    if not rows:
        return {"runs": 0}

    latencies = sorted(row.get("latency_ms", 0) for row in rows)
    costed = [row["cost_usd"] for row in rows if row.get("cost_usd") is not None]

    return {
        "runs": len(rows),
        "failed": sum(1 for row in rows if not row.get("ok", True)),
        "tokens": {
            "input": sum(row.get("input_tokens", 0) for row in rows),
            "output": sum(row.get("output_tokens", 0) for row in rows),
            "total": sum(row.get("total_tokens", 0) for row in rows),
        },
        "latency_ms": {
            "median": latencies[len(latencies) // 2],
            # p95 is the number that matters: the average hides the tail that
            # users actually complain about.
            "p95": latencies[min(len(latencies) - 1, int(len(latencies) * 0.95))],
            "max": latencies[-1],
        },
        "cost_usd": {
            "total": round(sum(costed), 6) if costed else None,
            "mean_per_run": round(sum(costed) / len(costed), 6) if costed else None,
            "priced_runs": len(costed),
            "note": None if costed else "Set MODEL_PRICING in .env to compute cost.",
        },
    }
