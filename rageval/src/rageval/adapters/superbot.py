"""SuperBotAdapter — the first adapter for a *real* app: this repo's SuperBot platform.

Everything before this proved the harness on the MockAdapter and canned HTTP responses;
this is the payoff — plugging the exact same core into a live RAG service and getting real
numbers. It speaks the platform's two contracts:

  * **Black-box** (always available): ``POST /auth/login`` for a bearer token, then
    ``POST /ask {query, thread_id}`` which returns only ``{"answer": ...}``. That's an
    *answer-tier* target — no chunks come back, so retrieval metrics are skipped, exactly
    as the tier logic intends.
  * **White-box** (opt-in): to score *retrieval* we reach past the API and call the app's
    own ``_hybrid_retrieve(query, owner)`` (RRF + optional FlashRank rerank) directly, map
    its chunks into the harness's ``RetrievedChunk``s, and unlock Recall/Precision/MRR/NDCG.

Two hard rules this file honors:

  * **The harness core never imports SuperBot.** This adapter lives behind the ``[superbot]``
    extra and is only reached lazily via ``config.build_target``; the white-box retriever is
    imported *inside a method*, so a plain ``pip install rageval`` never needs the backend.
  * **No secrets in the repo.** Credentials come from the environment (``SUPERBOT_TOKEN`` or
    ``SUPERBOT_EMAIL`` + ``SUPERBOT_PASSWORD``), never from the config file, so nothing
    sensitive lands in the manifest's config hash.

Retrieval ids: SuperBot chunks don't carry a stable per-chunk id, but their metadata records
the ``source`` document they came from. So white-box retrieval is scored at *document* (PDF)
granularity — a golden ``relevant_doc_ids`` entry is a source filename, and Recall@k answers
"did a chunk from the right document reach the top k?". The id field is configurable.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from typing import Any

import httpx

from rageval.core.adapter import RAGResult, RetrievedChunk

# White-box retrieval reaches into the host app; these are the functions it imports lazily.
_RETRIEVE_MODULE = "agents.pdf_chatbot"
_RETRIEVE_FUNC = "_hybrid_retrieve"
_RERANK_FUNC = "_rerank"

# (query, owner) -> list of the app's Document objects (typed Any: it's the host app's type).
HybridRetrieveFn = Callable[[str, str], list[Any]]
# (query, docs, top_n) -> reranked docs.
RerankFn = Callable[[str, list[Any], int], list[Any]]


class SuperBotAdapter:
    """Talks to a running SuperBot gateway; optionally scores retrieval white-box.

    Inject ``client`` (e.g. an ``httpx.Client`` on a ``MockTransport``) to test request
    building offline, and inject ``hybrid_retrieve``/``rerank_fn`` to exercise the white-box
    path without importing the backend. In production both default to the real thing.
    """

    def __init__(
        self,
        *,
        base_url: str = "http://localhost:8000",
        thread_id: str = "rageval",
        timeout_s: float = 60.0,
        white_box: bool = False,
        owner: str | None = None,
        rerank: bool = True,
        final_k: int = 4,
        retrieve_id_field: str = "source",
        name: str = "superbot",
        client: httpx.Client | None = None,
        hybrid_retrieve: HybridRetrieveFn | None = None,
        rerank_fn: RerankFn | None = None,
    ) -> None:
        self.name = name
        self._base_url = base_url.rstrip("/")
        self._thread_id = thread_id
        self._timeout_s = timeout_s
        self._white_box = white_box
        self._owner = owner
        self._rerank = rerank
        self._final_k = final_k
        self._id_field = retrieve_id_field
        self._client = client or httpx.Client()
        self._hybrid_retrieve = hybrid_retrieve
        self._rerank_fn = rerank_fn
        self._token: str | None = None

    # --- auth ------------------------------------------------------------------------
    def _ensure_auth(self) -> None:
        """Resolve a bearer token once (env token, else login), caching it on the adapter.

        A pre-issued ``SUPERBOT_TOKEN`` wins (handy in CI); otherwise we log in with
        ``SUPERBOT_EMAIL``/``SUPERBOT_PASSWORD``. Login also tells us the user id, which is
        the ``owner`` white-box retrieval must scope to — so black- and white-box paths
        agree on *whose* documents are being evaluated.
        """
        if self._token is not None:
            return
        token = os.getenv("SUPERBOT_TOKEN")
        if token:
            self._token = token
            return
        email = os.getenv("SUPERBOT_EMAIL")
        password = os.getenv("SUPERBOT_PASSWORD")
        if not (email and password):
            raise RuntimeError(
                "SuperBot auth is unset. Provide SUPERBOT_TOKEN, or SUPERBOT_EMAIL and "
                "SUPERBOT_PASSWORD, in the environment (never in the config file)."
            )
        response = self._client.post(
            f"{self._base_url}/auth/login",
            json={"email": email, "password": password},
            timeout=self._timeout_s,
        )
        response.raise_for_status()
        data = response.json()
        self._token = data["token"]
        if self._owner is None:
            self._owner = data.get("user", {}).get("id")

    def _auth_header(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._token}"}

    # --- black-box /ask --------------------------------------------------------------
    def query(self, question: str, **context: Any) -> RAGResult:
        """Ask ``/ask`` (answer-tier); if ``white_box``, also attach retrieved chunks.

        The API returns only an answer, so black-box runs are answer-tier. With white-box
        enabled the same call additionally pulls the retrieval step so one run can report
        both answer *and* retrieval metrics against the live system.
        """
        self._ensure_auth()
        start = time.perf_counter()
        response = self._client.post(
            f"{self._base_url}/ask",
            json={"query": question, "thread_id": self._thread_id},
            headers=self._auth_header(),
            timeout=self._timeout_s,
        )
        response.raise_for_status()
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        data = response.json()

        retrieved = self._white_box_retrieve(question) if self._white_box else []
        return RAGResult(
            answer=str(data.get("answer", "")),
            retrieved=retrieved,
            latency_ms=elapsed_ms,
            raw=data if isinstance(data, dict) else {"response": data},
        )

    # --- white-box retrieval ---------------------------------------------------------
    def retrieve(self, question: str, k: int = 0, **context: Any) -> list[RetrievedChunk]:
        """Expose the app's retrieval step directly (opt-in white-box capability)."""
        return self._white_box_retrieve(question, k=k or self._final_k)

    def _resolve_retriever(self) -> tuple[HybridRetrieveFn, RerankFn | None]:
        """Return the retrieve (+ optional rerank) callables, importing the app lazily.

        Injected callables win (tests, custom pipelines). Otherwise we import the host
        app's functions *here*, not at module load, so the core install never needs the
        backend and a missing backend fails with an actionable message, not an ImportError
        at startup.
        """
        if self._hybrid_retrieve is not None:
            return self._hybrid_retrieve, self._rerank_fn
        try:
            import importlib

            module = importlib.import_module(_RETRIEVE_MODULE)
        except ImportError as exc:  # pragma: no cover - env-specific
            raise RuntimeError(
                "White-box retrieval needs SuperBot's backend importable on the "
                f"PYTHONPATH (module {_RETRIEVE_MODULE!r}). Run the harness from an "
                "environment where the app is installed, or inject hybrid_retrieve."
            ) from exc
        hybrid: HybridRetrieveFn = getattr(module, _RETRIEVE_FUNC)
        rerank: RerankFn | None = getattr(module, _RERANK_FUNC, None)
        return hybrid, rerank

    def _white_box_retrieve(self, question: str, k: int | None = None) -> list[RetrievedChunk]:
        self._ensure_auth()  # resolves owner from the login response if not set explicitly
        if not self._owner:
            raise RuntimeError(
                "White-box retrieval needs an owner to scope documents. Set config.owner "
                "or use credentials whose login returns a user id."
            )
        top_n = k if k is not None else self._final_k
        hybrid, rerank = self._resolve_retriever()
        docs = hybrid(question, self._owner)
        if self._rerank and rerank is not None:
            docs = rerank(question, docs, top_n)
        else:
            docs = docs[:top_n]
        return [self._to_chunk(doc, i) for i, doc in enumerate(docs)]

    def _to_chunk(self, doc: Any, index: int) -> RetrievedChunk:
        """Map one of the app's Document objects to a harness ``RetrievedChunk``.

        Id comes from the configured metadata field (``source`` by default → document-level
        scoring); text from ``page_content``. SuperBot exposes no numeric retrieval score,
        so ``score`` stays None and rank-sensitive metrics use the returned order.
        """
        metadata = getattr(doc, "metadata", {}) or {}
        raw_id = metadata.get(self._id_field)
        chunk_id = str(raw_id) if raw_id is not None else f"{self.name}-{index}"
        text = getattr(doc, "page_content", "")
        return RetrievedChunk(id=chunk_id, text=str(text), score=None)
