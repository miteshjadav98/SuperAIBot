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


def test_unknown_adapter_raises() -> None:
    cfg = TargetConfig.model_validate({"name": "t", "adapter": "superbot"})
    with pytest.raises(ValueError, match="not-yet-implemented"):
        build_target(cfg)
