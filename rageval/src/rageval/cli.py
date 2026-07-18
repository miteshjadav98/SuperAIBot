"""``rageval`` CLI. M2 runs the Mock target end-to-end and scores retrieval.

The full command surface (``compare``, ``baseline``, ``report``, ``dashboard``,
``golden``) arrives in later milestones; declaring the entrypoint now keeps the package
``rageval ...`` invocable from a fresh install without churn later.
"""

from __future__ import annotations

from pathlib import Path

import typer

from rageval.adapters.mock import MockAdapter
from rageval.core.store import ResultsStore
from rageval.datasets.golden import GoldenRecord, load_golden
from rageval.metrics import format_retrieval_table, score_run
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
    runs_dir: Path = typer.Option(Path("runs"), "--runs-dir", help="Where to write runs."),
    name: str = typer.Option("mock", "--name", help="Target name (M2: MockAdapter only)."),
    k: int = typer.Option(5, "--k", help="How many chunks the mock retrieves per query."),
) -> None:
    """Run an evaluation and score retrieval. M2 supports MockAdapter, so it needs no keys."""
    records = load_golden(golden)
    store = ResultsStore(runs_dir)
    result = run_eval(
        MockAdapter(name=name, corpus=_mock_corpus(records), k=k),
        records,
        store=store,
        golden_path=golden,
        config={"target": name, "adapter": "mock", "k": k},
    )

    # Score retrieval on labelled records and re-persist (manifest stays fixed at run time).
    aggregate = score_run(result)
    store.save_result(result)

    op = result.operational()
    typer.echo(f"run_id={result.run_id} tier={result.tier} queries={op['queries']}")
    typer.echo(
        f"latency p50={op['latency_p50_ms']}ms p95={op['latency_p95_ms']}ms "
        f"cost=${op['total_cost_usd']}"
    )
    typer.echo("")
    typer.echo(format_retrieval_table(aggregate))
    typer.echo("")
    typer.echo(f"written to {runs_dir / result.run_id}")


if __name__ == "__main__":
    app()
