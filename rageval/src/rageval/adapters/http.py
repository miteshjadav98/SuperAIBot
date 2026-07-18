"""HTTPAdapter — the universal "USB connector" for any REST ask/chat/query API.

Point it at a URL, describe where the answer and (optionally) the retrieved chunks live
in the JSON response via JSONPath, and the harness can evaluate the target with **zero
new code**. The same adapter covers:
  * an ``/ask``-style ``{answer, sources:[{id,text,score}]}`` response,
  * a bare ``/chat`` ``{reply}`` response (answer-only → answer metrics only),
  * an "issue search" ``{results:[{key,title,body}]}`` response (retrieval-only).

Which metrics run is then decided by the tier logic from what this adapter returns — an
answer-only response simply produces no ``retrieved`` chunks and the harness skips
retrieval metrics rather than failing.

Requires the ``[http]`` extra (``httpx`` + ``jsonpath-ng``); it is imported lazily via
``config.build_target`` so the core install stays dependency-light.
"""

from __future__ import annotations

import os
import re
import time
from typing import Any

import httpx
from jsonpath_ng import parse as jsonpath_parse

from rageval.config import HTTPResponseMap, HTTPTargetConfig
from rageval.core.adapter import RAGResult, RetrievedChunk

_ENV_PATTERN = re.compile(r"\$\{([A-Z0-9_]+)\}")


class _SafeDict(dict[str, Any]):
    """format_map helper: leave unknown ``{placeholders}`` untouched instead of raising."""

    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def _interpolate_env(value: str) -> str:
    """Replace ``${VAR}`` with env values; fail loudly if a referenced var is unset.

    Auth headers routinely reference secrets from the environment. Failing loudly (rather
    than silently sending an empty token) is the spec's rule — a missing key is a config
    error, not something to paper over.
    """

    def repl(match: re.Match[str]) -> str:
        name = match.group(1)
        val = os.getenv(name)
        if val is None:
            raise RuntimeError(f"Environment variable {name!r} referenced in config is not set")
        return val

    return _ENV_PATTERN.sub(repl, value)


def _template(obj: Any, values: dict[str, Any]) -> Any:
    """Recursively substitute ``{question}``/``{context}`` placeholders in strings."""
    if isinstance(obj, str):
        return obj.format_map(_SafeDict(values))
    if isinstance(obj, dict):
        return {k: _template(v, values) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_template(v, values) for v in obj]
    return obj


def _first(data: Any, expr: str | None) -> Any:
    """First JSONPath match value, or None. Used for scalar fields."""
    if not expr:
        return None
    matches = jsonpath_parse(expr).find(data)
    return matches[0].value if matches else None


def _all(data: Any, expr: str | None) -> list[Any]:
    """All JSONPath match values (e.g. every element of a chunks array)."""
    if not expr:
        return []
    return [m.value for m in jsonpath_parse(expr).find(data)]


def _to_float(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


class HTTPAdapter:
    """Config-driven REST target. Inject a client for tests (e.g. httpx.MockTransport)."""

    def __init__(
        self,
        config: HTTPTargetConfig,
        *,
        name: str = "http",
        client: httpx.Client | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        self.name = name
        self._cfg = config
        self._client = client or httpx.Client()
        # Static context (e.g. an auth token) merged into every request's templating.
        self._context = context or {}

    def _headers(self) -> dict[str, str]:
        return {k: _interpolate_env(v) for k, v in self._cfg.headers.items()}

    def _send(self, question: str, **context: Any) -> httpx.Response:
        values = {"question": question, **self._context, **context}
        url = _interpolate_env(_template(self._cfg.url, values))
        headers = self._headers()
        if self._cfg.method.upper() == "GET":
            params = _template(self._cfg.params or {}, values)
            return self._client.get(
                url, params=params, headers=headers, timeout=self._cfg.timeout_s
            )
        body = _template(self._cfg.body or {}, values)
        return self._client.post(
            url, json=body, headers=headers, timeout=self._cfg.timeout_s
        )

    def _parse(self, data: Any, elapsed_ms: float) -> RAGResult:
        m: HTTPResponseMap = self._cfg.response
        answer = _first(data, m.answer)
        chunks: list[RetrievedChunk] = []
        for i, element in enumerate(_all(data, m.chunks)):
            cid = _first(element, m.chunk_id) if m.chunk_id else None
            text = _first(element, m.chunk_text) if m.chunk_text else None
            chunks.append(
                RetrievedChunk(
                    id=str(cid) if cid is not None else f"{self.name}-{i}",
                    text=str(text) if text is not None else "",
                    score=_to_float(_first(element, m.chunk_score)) if m.chunk_score else None,
                )
            )
        # Prefer target-reported latency/cost when mapped; else use our measured latency.
        latency = _to_float(_first(data, m.latency_ms))
        return RAGResult(
            answer=str(answer) if answer is not None else "",
            retrieved=chunks,
            latency_ms=latency if latency is not None else elapsed_ms,
            cost_usd=_to_float(_first(data, m.cost_usd)),
            raw=data if isinstance(data, dict) else {"response": data},
        )

    def query(self, question: str, **context: Any) -> RAGResult:
        start = time.perf_counter()
        response = self._send(question, **context)
        response.raise_for_status()
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return self._parse(response.json(), elapsed_ms)
