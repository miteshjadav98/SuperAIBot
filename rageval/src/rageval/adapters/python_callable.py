"""PythonCallableAdapter — wrap any local callable as a target (white-box eval).

When you can import the pipeline directly (rather than going over HTTP), this is the
thinnest possible bridge: give it a ``callable(question, **context) -> RAGResult`` and
you're done. It also accepts a bare *retriever* (``-> list[RetrievedChunk]``) for
retrieval-only white-box evaluation, where the answer is left empty and only retrieval
metrics apply.
"""

from __future__ import annotations

import importlib
import time
from collections.abc import Callable
from typing import Any, Literal

from rageval.core.adapter import RAGResult, RetrievedChunk

QueryFn = Callable[..., RAGResult]
RetrieveFn = Callable[..., list[RetrievedChunk]]


class PythonCallableAdapter:
    """Adapt a Python callable to the ``RAGTarget`` contract."""

    def __init__(
        self,
        fn: QueryFn | RetrieveFn,
        *,
        name: str = "python_callable",
        kind: Literal["query", "retrieve"] = "query",
    ) -> None:
        self.name = name
        self._fn = fn
        self._kind = kind

    @classmethod
    def from_import(
        cls,
        import_path: str,
        *,
        name: str = "python_callable",
        kind: Literal["query", "retrieve"] = "query",
    ) -> PythonCallableAdapter:
        """Build from a ``module.submodule:callable`` string (used by config loading)."""
        module_name, _, attr = import_path.partition(":")
        if not attr:
            raise ValueError(f"import_path must be 'module:callable', got {import_path!r}")
        module = importlib.import_module(module_name)
        fn = getattr(module, attr)
        if not callable(fn):
            raise TypeError(f"{import_path!r} is not callable")
        return cls(fn, name=name, kind=kind)

    def _call_query(self, question: str, **context: Any) -> RAGResult:
        result = self._fn(question, **context)
        if not isinstance(result, RAGResult):
            raise TypeError(f"callable {self.name!r} must return a RAGResult")
        return result

    def _call_retrieve(self, question: str, **context: Any) -> list[RetrievedChunk]:
        chunks = self._fn(question, **context)
        if not isinstance(chunks, list):
            raise TypeError(f"callable {self.name!r} must return a list of RetrievedChunk")
        return chunks

    def query(self, question: str, **context: Any) -> RAGResult:
        start = time.perf_counter()
        if self._kind == "query":
            result = self._call_query(question, **context)
            if not result.latency_ms:
                result.latency_ms = (time.perf_counter() - start) * 1000.0
            return result
        # retrieve-kind: the callable returns chunks; wrap them in an answer-empty result.
        chunks = self._call_retrieve(question, **context)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return RAGResult(answer="", retrieved=chunks, latency_ms=elapsed_ms)

    def retrieve(self, question: str, k: int, **context: Any) -> list[RetrievedChunk]:
        if self._kind == "retrieve":
            return self._call_retrieve(question, **context)[:k]
        return self._call_query(question, **context).retrieved[:k]
