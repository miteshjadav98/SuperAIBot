"""``rageval`` CLI. M3 runs any configured target (mock or HTTP) and scores retrieval.

The full command surface (``compare``, ``baseline``, ``report``, ``dashboard``,
``golden``) arrives in later milestones; declaring the entrypoint now keeps the package
``rageval ...`` invocable from a fresh install without churn later.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import typer

from rageval.adapters.mock import MockAdapter
from rageval.config import build_target, load_target_config
from rageval.core.adapter import RAGTarget
from rageval.core.provider import get_embedding_model, get_judge_model
from rageval.core.store import ResultsStore
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
            "ANTHROPIC_API_KEY) — judge metrics skipped."
        )
    if judge and embedder is None:
        typer.echo("note: no embeddings key in env — semantic_similarity skipped.")

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
    retrieval_aggregate = score_run(result)
    answer_aggregate = score_answers(result, judge=judge_model, embedder=embedder)
    store.save_result(result)

    op = result.operational()
    typer.echo(f"run_id={result.run_id} target={result.target_name} tier={result.tier}")
    typer.echo(
        f"queries={op['queries']} latency p50={op['latency_p50_ms']}ms "
        f"p95={op['latency_p95_ms']}ms cost=${op['total_cost_usd']}"
    )
    typer.echo("")
    typer.echo(format_retrieval_table(retrieval_aggregate))
    typer.echo("")
    typer.echo(format_answer_table(answer_aggregate))
    typer.echo("")
    typer.echo(f"written to {runs_dir / result.run_id}")


if __name__ == "__main__":
    app()
