"""End-to-end CLI gate: run → baseline → check, keyless on the MockAdapter.

Exercises the exit codes CI depends on: 0 when quality holds, 1 when it regresses, 2 when
the baseline is missing. Runs entirely offline (no target, no keys).
"""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from rageval.cli import app

runner = CliRunner()

_GOLDEN = (
    '{"question": "What is the capital of France?", "relevant_doc_ids": ["geo-fr-01"], '
    '"reference_answer": "Paris is the capital of France."}\n'
    '{"question": "Who wrote Hamlet?", "relevant_doc_ids": ["lit-shak-07"], '
    '"reference_answer": "William Shakespeare wrote Hamlet."}\n'
)


def _write_golden(tmp_path: Path) -> Path:
    p = tmp_path / "golden.jsonl"
    p.write_text(_GOLDEN, encoding="utf-8")
    return p


def _run(tmp_path: Path, *args: str):
    return runner.invoke(app, [*args], catch_exceptions=False)


def test_run_writes_result_and_trace(tmp_path: Path) -> None:
    golden = _write_golden(tmp_path)
    runs = tmp_path / "runs"
    result = _run(tmp_path, "run", "--golden", str(golden), "--runs-dir", str(runs), "--no-judge")
    assert result.exit_code == 0, result.output
    # A run dir with result.json + manifest.json + trace.json exists.
    run_dirs = list(runs.iterdir())
    assert len(run_dirs) == 1
    for name in ("result.json", "manifest.json", "trace.json"):
        assert (run_dirs[0] / name).exists()


def test_baseline_then_check_passes(tmp_path: Path) -> None:
    golden = _write_golden(tmp_path)
    runs = tmp_path / "runs"
    baseline_file = tmp_path / "baseline.json"

    # 1. Produce a run.
    _run(tmp_path, "run", "--golden", str(golden), "--runs-dir", str(runs), "--no-judge")
    run_id = next(runs.iterdir()).name

    # 2. Bless it as the baseline.
    saved = _run(tmp_path, "baseline", run_id, "--runs-dir", str(runs), "--out", str(baseline_file))
    assert saved.exit_code == 0, saved.output
    assert baseline_file.exists()

    # 3. Re-check against the baseline — the mock is deterministic, so quality holds.
    checked = _run(
        tmp_path,
        "check",
        "--golden", str(golden),
        "--baseline", str(baseline_file),
        "--runs-dir", str(runs),
        "--no-judge",
    )
    assert checked.exit_code == 0, checked.output
    assert "gate: PASSED" in checked.output


def test_check_fails_on_regression(tmp_path: Path) -> None:
    golden = _write_golden(tmp_path)
    runs = tmp_path / "runs"
    baseline_file = tmp_path / "baseline.json"

    _run(tmp_path, "run", "--golden", str(golden), "--runs-dir", str(runs), "--no-judge")
    run_id = next(runs.iterdir()).name
    _run(tmp_path, "baseline", run_id, "--runs-dir", str(runs), "--out", str(baseline_file))

    # Hand-edit the baseline so its recorded quality is far above what the run can produce,
    # forcing a regression on the next check.
    data = json.loads(baseline_file.read_text(encoding="utf-8"))
    data["metrics"] = {k: 0.999 for k in data["metrics"]}
    baseline_file.write_text(json.dumps(data), encoding="utf-8")

    checked = _run(
        tmp_path,
        "check",
        "--golden", str(golden),
        "--baseline", str(baseline_file),
        "--runs-dir", str(runs),
        "--no-judge",
    )
    assert checked.exit_code == 1, checked.output
    assert "gate: FAILED" in checked.output
    assert "REGRESSED" in checked.output


def test_check_missing_baseline_exits_2(tmp_path: Path) -> None:
    golden = _write_golden(tmp_path)
    checked = _run(
        tmp_path,
        "check",
        "--golden", str(golden),
        "--baseline", str(tmp_path / "nope.json"),
        "--runs-dir", str(tmp_path / "runs"),
        "--no-judge",
    )
    assert checked.exit_code == 2, checked.output
    assert "baseline not found" in checked.output
