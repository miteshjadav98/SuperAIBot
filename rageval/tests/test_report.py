"""The static HTML report: self-contained, escapes untrusted text, folds in the gate."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from rageval.cli import app
from rageval.core.baseline import Baseline, compare
from rageval.report import render_report

runner = CliRunner()


def _result() -> dict[str, object]:
    return {
        "run_id": "run-abc",
        "target_name": "mock",
        "tier": "answer+retrieved+labelled",
        "operational": {
            "queries": 2,
            "latency_p50_ms": 12.0,
            "latency_p95_ms": 20.0,
            "total_cost_usd": None,
        },
        "retrieval_aggregate": {"recall@1": 0.5, "recall@3": 1.0, "precision@1": 0.5},
        "answer_aggregate": {"token_f1": 0.42, "rouge_l": 0.30},
        "records": [
            {
                "question": "What is the capital of France?",
                "answer": "Paris.",
                "reference_answer": "Paris is the capital of France.",
                "retrieved": [{"id": "geo-fr-01", "text": "..."}, {"id": "d-2", "text": "..."}],
                "relevant_doc_ids": ["geo-fr-01"],
                "retrieval": {"recall@1": 1.0},
                "answer_metrics": {"token_f1": 0.4},
            }
        ],
    }


def _manifest() -> dict[str, object]:
    return {
        "git_sha": "abcdef1234567890",
        "provider": None,
        "model": None,
        "created_at": "2026-07-19T10:00:00+00:00",
        "dataset_hash": "0123456789abcdef",
        "tier": "answer+retrieved+labelled",
    }


def test_report_is_self_contained_and_has_metrics() -> None:
    html = render_report(_result(), _manifest())
    assert html.startswith("<!doctype html>")
    # No network: no external resources or scripts anywhere in the document.
    assert "http://" not in html and "https://" not in html
    assert "<script" not in html.lower()
    # Headline metrics are present.
    assert "recall" in html and "token_f1" in html
    assert "geo-fr-01" in html  # a retrieved id chip


def test_report_escapes_untrusted_text() -> None:
    result = _result()
    result["records"][0]["question"] = "<script>alert('x')</script>"  # type: ignore[index]
    html = render_report(result, _manifest())
    # The raw tag must not survive into the document; it is HTML-escaped.
    assert "<script>alert" not in html
    assert "&lt;script&gt;" in html


def test_report_includes_regression_verdict() -> None:
    result = _result()
    manifest = _manifest()
    base = Baseline.from_run(result, manifest)
    report = compare(base, result, manifest)
    html = render_report(result, manifest, regression=report)
    assert "GATE: PASSED" in html
    assert "Regression gate" in html


def test_report_cli_writes_html(tmp_path: Path) -> None:
    golden = tmp_path / "golden.jsonl"
    golden.write_text(
        '{"question": "Who wrote Hamlet?", "relevant_doc_ids": ["lit-01"], '
        '"reference_answer": "William Shakespeare."}\n',
        encoding="utf-8",
    )
    runs = tmp_path / "runs"
    runner.invoke(
        app, ["run", "--golden", str(golden), "--runs-dir", str(runs), "--no-judge"],
        catch_exceptions=False,
    )
    run_id = next(runs.iterdir()).name

    result = runner.invoke(
        app, ["report", run_id, "--runs-dir", str(runs)], catch_exceptions=False
    )
    assert result.exit_code == 0, result.output
    report_file = runs / run_id / "report.html"
    assert report_file.exists()
    assert report_file.read_text(encoding="utf-8").startswith("<!doctype html>")
