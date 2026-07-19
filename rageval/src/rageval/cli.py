"""``rageval`` CLI. Runs any configured target, scores it, traces it, and gates regressions.

Commands:
  * ``run``      — evaluate a target and print the retrieval + answer + operational tables.
  * ``baseline`` — snapshot a stored run's metrics as ``baseline.json`` (the reference).
  * ``check``    — evaluate, then fail (non-zero exit) if quality regressed vs a baseline.

``run`` and ``check`` share one pipeline (``_execute_run``) so the gate measures exactly
what a plain run measures. The remaining commands (``compare``, ``report``, ``dashboard``,
``golden``) arrive in later milestones.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import typer

from rageval.adapters.mock import MockAdapter
from rageval.config import build_target, load_target_config
from rageval.core.adapter import RAGTarget
from rageval.core.baseline import Baseline, compare
from rageval.core.provider import get_embedding_model, get_judge_model
from rageval.core.results import RunResult
from rageval.core.store import ResultsStore
from rageval.core.tracing import get_tracer
from rageval.datasets.golden import GoldenRecord, load_golden
from rageval.metrics import (
    format_answer_table,
    format_retrieval_table,
    score_answers,
    score_run,
)
from rageval.metrics.judge_prompts import PROMPT_VERSIONS
from rageval.runner import run as run_eval

app = typer.Typer(add_completion=False, help="Portable RAG evaluation harness.")


def _mock_corpus(golden: list[GoldenRecord], distractors: int = 5) -> dict[str, str]:
    """Build a corpus for the offline demo: the golden relevant ids + a few distractors.

    Without this, the MockAdapter mints synthetic ids that never match golden labels and
    every retrieval metric reads 0.0. Seeding the corpus with the real relevant ids lets
    the (deterministic) mock actually hit some of them, so the retrieval table shows
    meaningful, reproducible numbers — the metric machinery, not the mock, is the point.
    """
    corpus = {
        rid: f"Relevant passage {rid}."
        for g in golden
        for rid in g.relevant_doc_ids
    }
    for i in range(distractors):
        corpus[f"distractor-{i}"] = f"Unrelated passage {i}."
    return corpus


def _execute_run(
    *,
    golden: Path,
    target: Path | None,
    runs_dir: Path,
    k: int,
    judge: bool,
) -> tuple[RunResult, dict[str, Any], Path]:
    """The shared pipeline behind ``run`` and ``check``: evaluate, score, trace, persist.

    Returns the scored ``RunResult``, its persisted manifest dict (for comparability), and
    the run directory. Keeping this single-sourced guarantees the regression gate diffs
    exactly the numbers a plain ``run`` reports.
    """
    records = load_golden(golden)
    store = ResultsStore(runs_dir)

    rag_target: RAGTarget
    config_fingerprint: dict[str, Any]
    if target is None:
        rag_target = MockAdapter(corpus=_mock_corpus(records), k=k)
        config_fingerprint = {"adapter": "mock", "k": k}
    else:
        cfg = load_target_config(target)
        rag_target = build_target(cfg)
        config_fingerprint = cfg.config_fingerprint()

    # Resolve the judge/embedder up front so the manifest records the provenance (provider,
    # model, prompt versions) of whatever scored this run — even though scoring runs after.
    judge_model = get_judge_model() if judge else None
    embedder = get_embedding_model() if judge else None
    if judge and judge_model is None:
        typer.echo(
            "note: no LLM judge key in env (OPENAI_API_KEY / AZURE_OPENAI_* / "
            "ANTHROPIC_API_KEY) - judge metrics skipped."
        )
    if judge and embedder is None:
        typer.echo("note: no embeddings key in env - semantic_similarity skipped.")

    provider = judge_model.provider if judge_model is not None else None
    model = judge_model.model if judge_model is not None else None
    extra: dict[str, Any] = {}
    if judge_model is not None:
        extra["judge_prompt_versions"] = PROMPT_VERSIONS

    result = run_eval(
        rag_target,
        records,
        store=store,
        golden_path=golden,
        config=config_fingerprint,
        provider=provider,
        model=model,
        extra=extra,
    )

    # Score retrieval + answer metrics, then re-persist (manifest stays fixed at run time).
    score_run(result)
    score_answers(result, judge=judge_model, embedder=embedder)
    store.save_result(result)

    # Reload the manifest the runner wrote — the source of truth for comparability + trace.
    manifest = store.load(result.run_id).manifest

    tracer = get_tracer(runs_dir)
    trace_path = tracer.trace_run(result, manifest)
    if trace_path is not None:
        typer.echo(f"trace: {tracer.name} -> {trace_path}")
    else:
        typer.echo(f"trace: {tracer.name} (remote)")

    return result, manifest, store.run_dir(result.run_id)


def _print_summary(result: RunResult) -> None:
    op = result.operational()
    typer.echo(f"run_id={result.run_id} target={result.target_name} tier={result.tier}")
    typer.echo(
        f"queries={op['queries']} latency p50={op['latency_p50_ms']}ms "
        f"p95={op['latency_p95_ms']}ms cost=${op['total_cost_usd']}"
    )
    typer.echo("")
    typer.echo(format_retrieval_table(result.retrieval_aggregate))
    typer.echo("")
    typer.echo(format_answer_table(result.answer_aggregate))


@app.command()
def version() -> None:
    """Print the installed rageval version."""
    from importlib.metadata import version as _v

    typer.echo(_v("rageval"))


@app.command()
def run(
    golden: Path = typer.Option(..., "--golden", help="Path to a golden JSONL file."),
    target: Path | None = typer.Option(
        None, "--target", help="Target config YAML. Omit to use the keyless MockAdapter."
    ),
    runs_dir: Path = typer.Option(Path("runs"), "--runs-dir", help="Where to write runs."),
    k: int = typer.Option(5, "--k", help="Chunks the MockAdapter retrieves (mock only)."),
    judge: bool = typer.Option(
        True,
        "--judge/--no-judge",
        help="Use the auto-from-env LLM judge + embeddings for answer metrics. "
        "Absent keys skip those tiers with a notice; --no-judge forces the free "
        "lexical-only floor.",
    ),
) -> None:
    """Run an evaluation and score retrieval + answer quality.

    With no ``--target`` the offline MockAdapter runs (zero keys). With ``--target`` any
    configured adapter runs — e.g. the universal HTTP adapter against a live ``/ask`` or
    bare ``/chat`` API. The evaluation *tier* is detected from what the target returns.

    Answer metrics degrade gracefully: the lexical floor (token-F1/ROUGE-L) always runs
    when reference answers are present; the embedding and LLM-judge tiers run only when a
    key is found in the environment (or are silenced with ``--no-judge``).
    """
    result, _manifest, run_dir = _execute_run(
        golden=golden, target=target, runs_dir=runs_dir, k=k, judge=judge
    )
    _print_summary(result)
    typer.echo("")
    typer.echo(f"written to {run_dir}")


@app.command()
def baseline(
    run_id: str = typer.Argument(..., help="Run id (or path) of a stored run to bless."),
    runs_dir: Path = typer.Option(Path("runs"), "--runs-dir", help="Where runs live."),
    out: Path = typer.Option(Path("baseline.json"), "--out", help="Baseline file to write."),
) -> None:
    """Snapshot a stored run's metrics as the regression baseline.

    Blessing an inspected run (rather than a freshly-guessed threshold) is the whole point:
    later runs are gated against numbers a human actually looked at and accepted.
    """
    store = ResultsStore(runs_dir)
    stored = store.load(run_id)
    base = Baseline.from_run(stored.result, stored.manifest)
    path = base.save(out)
    typer.echo(f"baseline written to {path} (run={base.run_id}, tier={base.tier})")
    typer.echo(f"gated metrics: {', '.join(sorted(base.metrics)) or '(none)'}")


@app.command()
def check(
    golden: Path = typer.Option(..., "--golden", help="Path to a golden JSONL file."),
    baseline_path: Path = typer.Option(
        Path("baseline.json"), "--baseline", help="Baseline file to gate against."
    ),
    target: Path | None = typer.Option(
        None, "--target", help="Target config YAML. Omit to use the keyless MockAdapter."
    ),
    runs_dir: Path = typer.Option(Path("runs"), "--runs-dir", help="Where to write runs."),
    k: int = typer.Option(5, "--k", help="Chunks the MockAdapter retrieves (mock only)."),
    judge: bool = typer.Option(
        True, "--judge/--no-judge", help="Use the auto-from-env judge/embeddings (see `run`)."
    ),
    tolerance: float = typer.Option(
        0.01, "--tolerance", help="Absolute allowance before a quality drop fails the gate."
    ),
) -> None:
    """Evaluate a target and fail (exit 1) if quality regressed against the baseline.

    This is the CI gate: it runs the same pipeline as ``run``, then diffs the quality
    metrics against ``baseline.json``. A metric that drops more than ``--tolerance`` below
    baseline fails; latency/cost are shown but never fail the gate.
    """
    if not baseline_path.exists():
        typer.echo(
            f"error: baseline not found at {baseline_path}. "
            "Create one with `rageval baseline`."
        )
        raise typer.Exit(code=2)

    result, manifest, _run_dir = _execute_run(
        golden=golden, target=target, runs_dir=runs_dir, k=k, judge=judge
    )
    _print_summary(result)

    base = Baseline.load(baseline_path)
    report = compare(base, result.to_dict(), manifest, tolerance=tolerance)
    typer.echo("")
    typer.echo(report.format_table())

    if not report.passed:
        if not report.comparable:
            typer.echo("\ngate: FAILED - run is not comparable to the baseline (see notes).")
        else:
            regressed = ", ".join(d.name for d in report.regressions)
            typer.echo(f"\ngate: FAILED - regressions in: {regressed}")
        raise typer.Exit(code=1)
    typer.echo("\ngate: PASSED - no quality regression against baseline.")


if __name__ == "__main__":
    app()
