"""End-to-end runner smoke: 3 dummy queries through Mock, with a written manifest."""

from __future__ import annotations

import json
from pathlib import Path

from rageval.adapters.mock import MockAdapter
from rageval.core.store import MANIFEST_FILE, RESULT_FILE, ResultsStore
from rageval.datasets.golden import GoldenRecord, load_golden
from rageval.runner import TIER_ANSWER_ONLY, TIER_LABELLED, run

REPO_GOLDEN = Path(__file__).resolve().parents[1] / "data" / "golden" / "tiny.jsonl"


def _golden() -> list[GoldenRecord]:
    return [
        GoldenRecord("What is the capital of France?", ["geo-fr-01"], "Paris."),
        GoldenRecord("Who wrote Hamlet?", ["lit-shak-07"], "Shakespeare."),
        GoldenRecord("What do plants absorb?", ["bio-photo-03"], "Carbon dioxide."),
    ]


def test_run_end_to_end_writes_manifest(tmp_path: Path) -> None:
    store = ResultsStore(tmp_path / "runs")
    corpus = {"geo-fr-01": "Paris", "lit-shak-07": "Shakespeare", "bio-photo-03": "CO2"}
    result = run(
        MockAdapter(corpus=corpus),
        _golden(),
        store=store,
        golden_path=REPO_GOLDEN,
        config={"adapter": "mock"},
    )

    # Records + operational rollups.
    assert len(result.records) == 3
    op = result.operational()
    assert op["queries"] == 3
    assert op["latency_p50_ms"] > 0
    assert op["latency_p95_ms"] >= op["latency_p50_ms"]
    assert op["total_cost_usd"] == 0.0

    # Tier detected from chunks + golden labels.
    assert result.tier == TIER_LABELLED

    # Both files persisted with the reproducibility fields.
    run_dir = store.run_dir(result.run_id)
    assert (run_dir / RESULT_FILE).exists()
    manifest = json.loads((run_dir / MANIFEST_FILE).read_text(encoding="utf-8"))
    assert manifest["config_hash"]
    assert manifest["dataset_hash"]
    assert manifest["tier"] == TIER_LABELLED
    assert "git_sha" in manifest  # present (value may be None outside a repo)


def test_load_roundtrip(tmp_path: Path) -> None:
    store = ResultsStore(tmp_path / "runs")
    result = run(MockAdapter(), _golden(), store=store)
    loaded = store.load(result.run_id)
    assert loaded.result["run_id"] == result.run_id
    assert len(loaded.result["records"]) == 3


def test_tier_answer_only_when_no_chunks() -> None:
    from rageval.core.results import QueryRecord
    from rageval.runner import detect_tier

    # A record with no retrieved chunks can't reach a retrieval tier, even with labels.
    recs = [QueryRecord(question="q", answer="a")]
    assert detect_tier(recs, has_golden_labels=True) == TIER_ANSWER_ONLY


def test_repo_golden_loads() -> None:
    records = load_golden(REPO_GOLDEN)
    assert len(records) == 3
    assert records[0].relevant_doc_ids == ["geo-fr-01"]
