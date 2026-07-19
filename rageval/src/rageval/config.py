"""Target configuration — declare a target in YAML, get a ``RAGTarget`` back.

This is what turns the harness from "write a Python adapter" into "point a config at
any API." The models here are intentionally small and validated (pydantic), and
``build_target`` is the one place that maps an adapter name to a concrete adapter —
importing the networked ones *lazily* so the base ``pip install rageval`` (core + mock)
never needs httpx or jsonpath.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field

from rageval.core.adapter import RAGTarget

AdapterName = Literal["mock", "http", "python_callable", "superbot"]


class HTTPResponseMap(BaseModel):
    """JSONPath expressions telling the HTTP adapter where to find each field.

    Only ``answer`` (or ``chunks``) is really required — everything else degrades
    gracefully. ``chunk_*`` paths are evaluated *relative to each element* selected by
    ``chunks``. This mapping is the whole reason one adapter fits ``/ask``, bare ``/chat``,
    and issue-search shapes without new code.
    """

    answer: str | None = None
    chunks: str | None = None
    chunk_id: str | None = None
    chunk_text: str | None = None
    chunk_score: str | None = None
    latency_ms: str | None = None
    cost_usd: str | None = None


class HTTPTargetConfig(BaseModel):
    """Everything needed to call one REST endpoint and read its response."""

    url: str
    method: str = "POST"
    headers: dict[str, str] = Field(default_factory=dict)
    # Request body template for JSON methods; string values may contain "{question}"
    # (and any "{context_key}") which are substituted per query.
    body: dict[str, Any] | None = None
    # Query params for GET-style endpoints; same "{question}" templating applies.
    params: dict[str, str] | None = None
    timeout_s: float = 30.0
    response: HTTPResponseMap


class PythonCallableConfig(BaseModel):
    """Point at an importable callable, e.g. ``mypkg.pipeline:answer``."""

    import_path: str
    # "query" → callable returns a RAGResult; "retrieve" → returns list[RetrievedChunk].
    kind: Literal["query", "retrieve"] = "query"


class SuperBotConfig(BaseModel):
    """Talk to this repo's SuperBot gateway. Secrets stay in env, never here.

    ``white_box`` opts into scoring retrieval by calling the app's own retriever directly;
    it needs the backend importable and an ``owner`` (or credentials whose login resolves
    one). Everything security-sensitive (token, email, password) is read from the
    environment at call time, so this block is safe to commit and to hash into the manifest.
    """

    base_url: str = "http://localhost:8000"
    thread_id: str = "rageval"
    timeout_s: float = 60.0
    white_box: bool = False
    owner: str | None = None
    rerank: bool = True
    final_k: int = 4
    retrieve_id_field: str = "source"


class TargetConfig(BaseModel):
    """A full target declaration (usually the ``target:`` block of a YAML file)."""

    name: str
    adapter: AdapterName
    http: HTTPTargetConfig | None = None
    python_callable: PythonCallableConfig | None = None
    superbot: SuperBotConfig | None = None
    golden: str | None = None
    runs_dir: str = "runs"

    def config_fingerprint(self) -> dict[str, Any]:
        """A JSON-able view for the run manifest's config hash (excludes live secrets)."""
        return self.model_dump(exclude_none=True)


def load_target_config(path: Path | str) -> TargetConfig:
    """Read a YAML file whose top-level key is ``target:`` (or is the target itself)."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if isinstance(raw, dict) and "target" in raw:
        raw = raw["target"]
    return TargetConfig.model_validate(raw)


def build_target(config: TargetConfig, **context: Any) -> RAGTarget:
    """Instantiate the adapter named by ``config``. Networked adapters import lazily."""
    if config.adapter == "mock":
        from rageval.adapters.mock import MockAdapter

        return MockAdapter(name=config.name)
    if config.adapter == "http":
        if config.http is None:
            raise ValueError("adapter 'http' requires a 'http:' config block")
        from rageval.adapters.http import HTTPAdapter

        return HTTPAdapter(config.http, name=config.name, context=context)
    if config.adapter == "python_callable":
        if config.python_callable is None:
            raise ValueError("adapter 'python_callable' requires a 'python_callable:' block")
        from rageval.adapters.python_callable import PythonCallableAdapter

        return PythonCallableAdapter.from_import(
            config.python_callable.import_path,
            name=config.name,
            kind=config.python_callable.kind,
        )
    if config.adapter == "superbot":
        sb = config.superbot or SuperBotConfig()
        from rageval.adapters.superbot import SuperBotAdapter

        return SuperBotAdapter(
            base_url=sb.base_url,
            thread_id=sb.thread_id,
            timeout_s=sb.timeout_s,
            white_box=sb.white_box,
            owner=sb.owner,
            rerank=sb.rerank,
            final_k=sb.final_k,
            retrieve_id_field=sb.retrieve_id_field,
            name=config.name,
        )
    raise ValueError(f"Unknown or not-yet-implemented adapter: {config.adapter!r}")
