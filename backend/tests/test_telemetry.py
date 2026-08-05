"""Per-run token, latency and cost accounting."""

from __future__ import annotations

import pytest

from core import telemetry


@pytest.fixture
def priced(monkeypatch):
    monkeypatch.setattr(
        telemetry.settings,
        "model_pricing",
        '{"gpt-4.1": {"input": 2.0, "output": 8.0}, "gpt-4.1-mini": {"input": 0.4, "output": 1.6}}',
    )


USAGE = {"gpt-4.1": {"input_tokens": 1_000_000, "output_tokens": 500_000}}


def test_no_pricing_configured_means_no_cost_not_a_wrong_cost(monkeypatch):
    """Inventing default rates would produce confidently wrong numbers."""
    monkeypatch.setattr(telemetry.settings, "model_pricing", None)

    assert telemetry.compute_cost(USAGE) is None


def test_cost_is_computed_per_million_tokens(priced):
    # 1M input @ $2 + 0.5M output @ $8 = 2 + 4
    assert telemetry.compute_cost(USAGE) == 6.0


def test_cost_sums_across_models(priced):
    usage = {
        "gpt-4.1": {"input_tokens": 1_000_000, "output_tokens": 0},
        "gpt-4.1-mini": {"input_tokens": 1_000_000, "output_tokens": 0},
    }

    assert telemetry.compute_cost(usage) == 2.4


def test_versioned_model_names_match_their_configured_rate(priced):
    """Providers report gpt-4.1-2025-04-14; the config key is gpt-4.1."""
    usage = {"gpt-4.1-2025-04-14": {"input_tokens": 1_000_000, "output_tokens": 0}}

    assert telemetry.compute_cost(usage) == 2.0


def test_longest_prefix_wins_so_mini_is_not_priced_as_full(priced):
    """gpt-4.1-mini starts with gpt-4.1 — matching the shorter key would
    overcharge it fourfold."""
    usage = {"gpt-4.1-mini-2024": {"input_tokens": 1_000_000, "output_tokens": 0}}

    assert telemetry.compute_cost(usage) == 0.4


def test_unknown_models_are_skipped_not_guessed(priced):
    assert telemetry.compute_cost({"some-other-model": {"input_tokens": 999}}) is None


def test_malformed_pricing_config_degrades_quietly(monkeypatch):
    monkeypatch.setattr(telemetry.settings, "model_pricing", "not json {{{")

    assert telemetry.pricing() == {}
    assert telemetry.compute_cost(USAGE) is None


# --- The measure() context manager -------------------------------------------


def test_measure_records_latency_and_is_enrichable():
    with telemetry.measure("chat", owner="alice", thread_id="t1") as metrics:
        metrics.agent_id = "personal_chef"

    assert metrics.ok is True
    assert metrics.agent_id == "personal_chef"
    assert metrics.latency_ms >= 0


def test_measure_records_a_failure_and_re_raises():
    """A run that burned tokens and then failed is exactly the one worth costing."""
    with pytest.raises(ValueError):
        with telemetry.measure("chat", owner="alice") as metrics:
            raise ValueError("boom")

    assert metrics.ok is False
    assert "ValueError: boom" in metrics.error


def test_persistence_failure_never_breaks_the_request(monkeypatch):
    """Telemetry that can fail a request is worse than no telemetry."""

    class Exploding:
        def insert_one(self, _doc):
            raise RuntimeError("mongo is down")

    monkeypatch.setattr(telemetry.db, "run_metrics_collection", lambda: Exploding())

    with telemetry.measure("chat", owner="alice"):
        pass  # must not raise


def test_summary_reports_nothing_when_mongo_is_unconfigured(monkeypatch):
    monkeypatch.setattr(telemetry.db, "run_metrics_collection", lambda: None)

    assert telemetry.summary()["runs"] == 0
