"""Config loading + build_target: YAML → validated models → concrete adapters."""

from __future__ import annotations

from pathlib import Path

import pytest

from rageval.adapters.http import HTTPAdapter
from rageval.adapters.mock import MockAdapter
from rageval.config import TargetConfig, build_target, load_target_config

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


@pytest.mark.parametrize("fname", ["http_ask.yaml", "http_chat.yaml", "http_issue_search.yaml"])
def test_example_configs_load_and_build(fname: str) -> None:
    cfg = load_target_config(CONFIGS / fname)
    assert cfg.adapter == "http"
    assert cfg.http is not None
    target = build_target(cfg)
    assert isinstance(target, HTTPAdapter)
    assert target.name == cfg.name


def test_mock_config_builds_mock() -> None:
    cfg = load_target_config(CONFIGS / "mock.yaml")
    assert isinstance(build_target(cfg), MockAdapter)


def test_config_fingerprint_is_jsonable_and_omits_none() -> None:
    cfg = TargetConfig.model_validate(
        {"name": "t", "adapter": "http", "http": {"url": "http://x", "response": {"answer": "$.a"}}}
    )
    fp = cfg.config_fingerprint()
    assert fp["name"] == "t"
    assert "python_callable" not in fp  # None fields excluded


def test_http_without_block_raises() -> None:
    cfg = TargetConfig.model_validate({"name": "t", "adapter": "http"})
    with pytest.raises(ValueError, match="requires a 'http:' config block"):
        build_target(cfg)


def test_superbot_builds_with_defaults() -> None:
    # superbot is implemented (M7): with no `superbot:` block it builds on defaults.
    cfg = TargetConfig.model_validate({"name": "t", "adapter": "superbot"})
    adapter = build_target(cfg)
    assert adapter.name == "t"


def test_unknown_adapter_raises() -> None:
    # Bypass Literal validation to exercise the defensive fallthrough for an adapter
    # name the builder doesn't handle.
    cfg = TargetConfig.model_construct(name="t", adapter="nope")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Unknown or not-yet-implemented"):
        build_target(cfg)
