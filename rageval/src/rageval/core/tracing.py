"""Run tracing behind a tiny interface — Langfuse when configured, local JSON otherwise.

Tracing is *observability*, never a dependency of correctness: an eval must produce the
same numbers whether or not a tracer is attached, and a tracer that fails (network down,
SDK missing) must never fail the run. So the contract here is deliberately small and every
remote path is wrapped to degrade to a no-op with a notice.

Two implementations sit behind the ``Tracer`` protocol:
  * ``LocalJSONTracer`` — the keyless default. Writes ``trace.json`` next to the run with
    one span per query (input, output, latency, retrieved ids, and the scored metrics), so
    a run is inspectable offline with no service running.
  * ``LangfuseTracer`` — used when ``LANGFUSE_PUBLIC_KEY``/``LANGFUSE_SECRET_KEY`` are set
    (extra: ``pip install 'rageval[tracing]'``). Lazily imported so core needs no SDK.

``get_tracer`` picks one from the environment and falls back to local JSON if the remote
SDK is selected but unavailable — the same auto-from-env, never-crash stance as the judge
provider shim in ``core/provider.py``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Protocol

from rageval.core.results import RunResult

TRACE_FILE = "trace.json"


class Tracer(Protocol):
    """Records a scored run. Implementations must never raise into the caller."""

    name: str

    def trace_run(self, result: RunResult, manifest: dict[str, Any]) -> Path | None: ...


def _spans(result: RunResult) -> list[dict[str, Any]]:
    """One span per evaluated query — the shape both backends serialize."""
    spans: list[dict[str, Any]] = []
    for order, record in enumerate(result.records):
        spans.append(
            {
                "order": order,
                "question": record.question,
                "answer": record.answer,
                "latency_ms": round(record.latency_ms, 2),
                "cost_usd": record.cost_usd,
                "retrieved_ids": [c.id for c in record.retrieved],
                "retrieval": record.retrieval,
                "answer_metrics": record.answer_metrics,
            }
        )
    return spans


class LocalJSONTracer:
    """Writes a self-contained ``trace.json`` into the run directory. Keyless, offline."""

    name = "local-json"

    def __init__(self, root: Path | str = "runs") -> None:
        self.root = Path(root)

    def trace_run(self, result: RunResult, manifest: dict[str, Any]) -> Path | None:
        trace = {
            "run_id": result.run_id,
            "target_name": result.target_name,
            "tier": result.tier,
            "provider": manifest.get("provider"),
            "model": manifest.get("model"),
            "git_sha": manifest.get("git_sha"),
            "spans": _spans(result),
        }
        target = self.root / result.run_id
        target.mkdir(parents=True, exist_ok=True)
        path = target / TRACE_FILE
        path.write_text(
            json.dumps(trace, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return path


class LangfuseTracer:
    """Emits each query as a Langfuse trace + generation. Failures degrade to a no-op.

    Observability must not gate the eval, so a network error or SDK problem is swallowed
    (with a printed notice) rather than propagated — the run and its numbers stand alone.
    """

    name = "langfuse"

    def __init__(self) -> None:
        try:
            from langfuse import Langfuse
        except ImportError as exc:  # pragma: no cover - env-specific
            raise RuntimeError(
                "LANGFUSE_* keys are set but the 'langfuse' package is not installed. "
                "Fix: pip install 'rageval[tracing]'"
            ) from exc
        self._client: Any = Langfuse()

    def trace_run(self, result: RunResult, manifest: dict[str, Any]) -> Path | None:
        try:
            for span in _spans(result):
                trace = self._client.trace(
                    name=f"rageval:{result.target_name}",
                    input={"question": span["question"]},
                    output={"answer": span["answer"]},
                    metadata={
                        "run_id": result.run_id,
                        "tier": result.tier,
                        "git_sha": manifest.get("git_sha"),
                        "retrieval": span["retrieval"],
                        "answer_metrics": span["answer_metrics"],
                    },
                )
                trace.generation(
                    name="answer",
                    model=manifest.get("model"),
                    input=span["question"],
                    output=span["answer"],
                )
            self._client.flush()
        except Exception as exc:  # noqa: BLE001 - observability must never break a run
            print(f"note: Langfuse tracing failed ({exc}); run and metrics are unaffected.")
        return None


def langfuse_configured() -> bool:
    """True when both Langfuse keys are present in the environment."""
    return bool(os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"))


def get_tracer(root: Path | str = "runs", *, prefer_remote: bool = True) -> Tracer:
    """Pick a tracer from the environment; fall back to local JSON, never crash.

    Langfuse is chosen only when both keys are set and its SDK imports cleanly; anything
    else (no keys, missing SDK) yields the keyless ``LocalJSONTracer`` so CI and offline
    runs always have a trace artifact.
    """
    if prefer_remote and langfuse_configured():
        try:
            return LangfuseTracer()
        except RuntimeError as exc:
            print(f"note: {exc} — falling back to local JSON tracing.")
    return LocalJSONTracer(root)
