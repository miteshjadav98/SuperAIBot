"""Baseline snapshot + regression gate — the rigorous replacement for a guessed threshold.

The point of the harness is to *catch a quality regression before it ships*. That needs a
fixed reference to compare against, so we snapshot a run's aggregate metrics (plus the
provenance that makes a comparison fair) into a ``baseline.json`` and diff later runs
against it. A metric that drops more than ``tolerance`` below baseline fails the gate; CI
exits non-zero.

Two design choices worth stating:

  * **Only quality metrics gate.** Retrieval + answer metrics are bounded in [0, 1] and
    reproducible, so an absolute tolerance is meaningful and the same across environments.
    Latency and cost are reported as deltas but never fail the gate — wall-clock time is
    environment-dependent and would make CI flaky.
  * **A metric missing from the new run is a note, not a regression.** A keyless CI run
    can't reproduce the LLM-judge metrics a locally-keyed baseline recorded; failing on
    their absence would make the gate unusable. We gate what the run could measure and say
    so about the rest.

Comparability is guarded: comparing across a different dataset is refused outright (the
numbers aren't about the same thing), while a tier or judge-prompt-version change is
surfaced as a note so a shift in methodology isn't silently read as a quality change.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def quality_metrics(result: dict[str, Any]) -> dict[str, float]:
    """Flatten the gate-able (higher-is-better, [0,1]) metrics from a result dict."""
    metrics: dict[str, float] = {}
    metrics.update(result.get("retrieval_aggregate", {}))
    metrics.update(result.get("answer_aggregate", {}))
    return metrics


def operational_metrics(result: dict[str, Any]) -> dict[str, float]:
    """Flatten the informational (lower-is-better) latency/cost metrics from a result."""
    op = result.get("operational", {})
    out: dict[str, float] = {}
    for key in ("latency_p50_ms", "latency_p95_ms", "total_cost_usd"):
        value = op.get(key)
        if value is not None:
            out[key] = float(value)
    return out


@dataclass(slots=True)
class Baseline:
    """A frozen reference point: metrics plus the provenance that makes a diff fair."""

    created_at: str
    run_id: str
    tier: str
    git_sha: str | None
    dataset_hash: str | None
    config_hash: str | None
    prompt_versions: dict[str, str]
    metrics: dict[str, float]
    operational: dict[str, float]

    @classmethod
    def from_run(cls, result: dict[str, Any], manifest: dict[str, Any]) -> Baseline:
        """Build a baseline from a stored ``result.json`` + ``manifest.json`` pair."""
        return cls(
            created_at=datetime.now(UTC).isoformat(),
            run_id=result.get("run_id", "unknown"),
            tier=result.get("tier", manifest.get("tier", "unknown")),
            git_sha=manifest.get("git_sha"),
            dataset_hash=manifest.get("dataset_hash"),
            config_hash=manifest.get("config_hash"),
            prompt_versions=dict(manifest.get("extra", {}).get("judge_prompt_versions", {})),
            metrics=quality_metrics(result),
            operational=operational_metrics(result),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: Path | str) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return p

    @classmethod
    def load(cls, path: Path | str) -> Baseline:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(**data)


@dataclass(slots=True)
class MetricDelta:
    """One metric compared against its baseline value."""

    name: str
    baseline: float
    current: float | None  # None → the new run didn't measure this metric
    delta: float | None
    regressed: bool


@dataclass(slots=True)
class RegressionReport:
    """The outcome of diffing a run against a baseline."""

    tolerance: float
    comparable: bool
    deltas: list[MetricDelta] = field(default_factory=list)  # gated quality metrics
    operational: list[MetricDelta] = field(default_factory=list)  # informational only
    notes: list[str] = field(default_factory=list)

    @property
    def regressions(self) -> list[MetricDelta]:
        return [d for d in self.deltas if d.regressed]

    @property
    def passed(self) -> bool:
        """Pass iff the run is comparable and no gated metric regressed."""
        return self.comparable and not self.regressions

    def format_table(self) -> str:
        lines: list[str] = []
        if self.notes:
            lines.extend(f"note: {n}" for n in self.notes)
            lines.append("")
        header = "metric".ljust(22) + "base".rjust(9) + "curr".rjust(9) + "delta".rjust(9)
        lines.append(header)
        lines.append("-" * len(header))
        for d in self.deltas + self.operational:
            curr = f"{'n/a':>9}" if d.current is None else f"{d.current:9.3f}"
            delta = f"{'n/a':>9}" if d.delta is None else f"{d.delta:+9.3f}"
            flag = "  REGRESSED" if d.regressed else ""
            lines.append(f"{d.name.ljust(22)}{d.baseline:9.3f}{curr}{delta}{flag}")
        verdict = "PASS" if self.passed else "FAIL"
        lines.append("")
        lines.append(f"{verdict} (tolerance={self.tolerance:.3f})")
        return "\n".join(lines)


# Latency/cost: lower is better, and reported but never gated (environment-dependent).
_OPERATIONAL_KEYS = ("latency_p50_ms", "latency_p95_ms", "total_cost_usd")


def compare(
    baseline: Baseline,
    result: dict[str, Any],
    manifest: dict[str, Any],
    *,
    tolerance: float = 0.01,
) -> RegressionReport:
    """Diff ``result`` against ``baseline``; regressions are quality drops beyond tolerance.

    ``tolerance`` is an absolute allowance on the [0,1] quality metrics (e.g. 0.01 permits
    noise but catches a real drop). Operational metrics are diffed for visibility only.
    """
    notes: list[str] = []
    comparable = True

    current_dataset = manifest.get("dataset_hash")
    if baseline.dataset_hash != current_dataset:
        comparable = False
        notes.append(
            "dataset changed since baseline "
            f"({baseline.dataset_hash} -> {current_dataset}); metrics are not comparable."
        )
    current_tier = result.get("tier", manifest.get("tier"))
    if baseline.tier != current_tier:
        notes.append(f"tier changed since baseline ({baseline.tier} -> {current_tier}).")
    current_prompts = dict(manifest.get("extra", {}).get("judge_prompt_versions", {}))
    if baseline.prompt_versions and baseline.prompt_versions != current_prompts:
        notes.append(
            "judge prompt versions changed since baseline; judged metrics may shift."
        )

    current_quality = quality_metrics(result)
    deltas: list[MetricDelta] = []
    for name, base_value in sorted(baseline.metrics.items()):
        current_value = current_quality.get(name)
        if current_value is None:
            notes.append(f"{name} not measured in this run - skipped (not a regression).")
            deltas.append(MetricDelta(name, base_value, None, None, regressed=False))
            continue
        delta = current_value - base_value
        # Only a genuine comparison can flag a regression.
        regressed = comparable and delta < -tolerance
        deltas.append(MetricDelta(name, base_value, current_value, delta, regressed))

    current_op = operational_metrics(result)
    op_deltas: list[MetricDelta] = []
    for name in _OPERATIONAL_KEYS:
        if name not in baseline.operational:
            continue
        op_base = baseline.operational[name]
        op_current = current_op.get(name)
        op_delta = None if op_current is None else op_current - op_base
        # Informational: never marked as a regression, never affects pass/fail.
        op_deltas.append(MetricDelta(name, op_base, op_current, op_delta, regressed=False))

    return RegressionReport(
        tolerance=tolerance,
        comparable=comparable,
        deltas=deltas,
        operational=op_deltas,
        notes=notes,
    )
