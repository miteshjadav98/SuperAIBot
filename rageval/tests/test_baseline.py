"""Baseline snapshot + regression gate — the CI-facing contract, tested keyless."""

from __future__ import annotations

from pathlib import Path

import pytest

from rageval.core.baseline import Baseline, compare

# A minimal stored-run shape (what result.json / manifest.json hold on disk).
_RESULT = {
    "run_id": "run-abc",
    "tier": "answer+retrieved+labelled",
    "retrieval_aggregate": {"recall@1": 0.5, "ndcg@5": 0.8},
    "answer_aggregate": {"token_f1": 0.4, "faithfulness": 0.9},
    "operational": {"latency_p50_ms": 12.0, "latency_p95_ms": 18.0, "total_cost_usd": 0.0},
}
_MANIFEST = {
    "git_sha": "deadbeef",
    "dataset_hash": "hash-v1",
    "config_hash": "cfg-1",
    "tier": "answer+retrieved+labelled",
    "extra": {"judge_prompt_versions": {"faithfulness": "1.0"}},
}


def _baseline() -> Baseline:
    return Baseline.from_run(_RESULT, _MANIFEST)


class TestBaselineSnapshot:
    def test_captures_quality_and_operational(self) -> None:
        base = _baseline()
        assert base.metrics == {
            "recall@1": 0.5,
            "ndcg@5": 0.8,
            "token_f1": 0.4,
            "faithfulness": 0.9,
        }
        assert base.operational["latency_p50_ms"] == 12.0
        assert base.dataset_hash == "hash-v1"
        assert base.prompt_versions == {"faithfulness": "1.0"}

    def test_save_load_roundtrip(self, tmp_path: Path) -> None:
        base = _baseline()
        path = base.save(tmp_path / "baseline.json")
        assert path.exists()
        reloaded = Baseline.load(path)
        assert reloaded.metrics == base.metrics
        assert reloaded.dataset_hash == base.dataset_hash


class TestCompare:
    def test_identical_run_passes(self) -> None:
        report = compare(_baseline(), _RESULT, _MANIFEST)
        assert report.comparable
        assert report.passed
        assert report.regressions == []

    def test_improvement_passes(self) -> None:
        better = {**_RESULT, "answer_aggregate": {"token_f1": 0.6, "faithfulness": 0.95}}
        report = compare(_baseline(), better, _MANIFEST)
        assert report.passed
        # A positive delta is recorded but never a regression.
        token = next(d for d in report.deltas if d.name == "token_f1")
        assert token.delta == pytest.approx(0.2)
        assert not token.regressed

    def test_quality_drop_beyond_tolerance_fails(self) -> None:
        worse = {**_RESULT, "answer_aggregate": {"token_f1": 0.30, "faithfulness": 0.9}}
        report = compare(_baseline(), worse, _MANIFEST, tolerance=0.01)
        assert not report.passed
        assert [d.name for d in report.regressions] == ["token_f1"]

    def test_drop_within_tolerance_passes(self) -> None:
        # token_f1 0.4 → 0.395, a 0.005 dip under a 0.01 tolerance.
        noisy = {**_RESULT, "answer_aggregate": {"token_f1": 0.395, "faithfulness": 0.9}}
        report = compare(_baseline(), noisy, _MANIFEST, tolerance=0.01)
        assert report.passed

    def test_missing_metric_is_note_not_regression(self) -> None:
        # A keyless CI run can't reproduce the judged faithfulness the baseline recorded.
        keyless = {**_RESULT, "answer_aggregate": {"token_f1": 0.4}}
        report = compare(_baseline(), keyless, _MANIFEST)
        assert report.passed
        faith = next(d for d in report.deltas if d.name == "faithfulness")
        assert faith.current is None
        assert not faith.regressed
        assert any("faithfulness" in n for n in report.notes)

    def test_dataset_change_is_not_comparable(self) -> None:
        moved = {**_MANIFEST, "dataset_hash": "hash-v2"}
        # Even with identical metrics, a different dataset can't be compared → fail-safe.
        report = compare(_baseline(), _RESULT, moved)
        assert not report.comparable
        assert not report.passed
        assert any("dataset changed" in n for n in report.notes)

    def test_tier_change_is_noted_but_still_gates(self) -> None:
        shifted_result = {**_RESULT, "tier": "answer+retrieved"}
        shifted_manifest = {**_MANIFEST, "tier": "answer+retrieved"}
        report = compare(_baseline(), shifted_result, shifted_manifest)
        assert report.comparable  # same dataset
        assert any("tier changed" in n for n in report.notes)

    def test_operational_delta_shown_never_gates(self) -> None:
        slow = {
            **_RESULT,
            "operational": {
                "latency_p50_ms": 999.0,
                "latency_p95_ms": 999.0,
                "total_cost_usd": 0.0,
            },
        }
        report = compare(_baseline(), slow, _MANIFEST)
        # Latency ballooned but the gate still passes (informational only).
        assert report.passed
        latency = next(d for d in report.operational if d.name == "latency_p50_ms")
        assert latency.delta == pytest.approx(987.0)
        assert not latency.regressed
