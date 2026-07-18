"""``rageval`` CLI. M1 wires just enough to run the Mock target end-to-end.

The full command surface (``compare``, ``baseline``, ``report``, ``dashboard``,
``golden``) arrives in later milestones; declaring the entrypoint now keeps the package
``rageval ...`` invocable from a fresh install without churn later.
"""

from __future__ import annotations

from pathlib import Path

import typer

from rageval.adapters.mock import MockAdapter
from rageval.core.store import ResultsStore
from rageval.datasets.golden import load_golden
from rageval.runner import run as run_eval

app = typer.Typer(add_completion=False, help="Portable RAG evaluation harness.")


@app.command()
def version() -> None:
    """Print the installed rageval version."""
    from importlib.metadata import version as _v

    typer.echo(_v("rageval"))


@app.command()
def run(
    golden: Path = typer.Option(..., "--golden", help="Path to a golden JSONL file."),
    runs_dir: Path = typer.Option(Path("runs"), "--runs-dir", help="Where to write runs."),
    name: str = typer.Option("mock", "--name", help="Target name (M1: MockAdapter only)."),
) -> None:
    """Run an evaluation. M1 supports the MockAdapter so it works with zero keys."""
    records = load_golden(golden)
    result = run_eval(
        MockAdapter(name=name),
        records,
        store=ResultsStore(runs_dir),
        golden_path=golden,
        config={"target": name, "adapter": "mock"},
    )
    op = result.operational()
    typer.echo(f"run_id={result.run_id} tier={result.tier} queries={op['queries']}")
    typer.echo(
        f"latency p50={op['latency_p50_ms']}ms p95={op['latency_p95_ms']}ms "
        f"cost=${op['total_cost_usd']}"
    )
    typer.echo(f"written to {runs_dir / result.run_id}")


if __name__ == "__main__":
    app()
