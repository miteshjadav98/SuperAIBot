"""rageval — a portable RAG evaluation harness.

Plug any RAG/ask/chat API in through a tiny adapter and get retrieval + answer + cost
metrics and a CI regression gate. The core imports nothing target-specific.
"""

from __future__ import annotations

__version__ = "0.1.0"

from rageval.core.adapter import RAGResult, RAGTarget, RetrievedChunk

__all__ = ["RAGResult", "RAGTarget", "RetrievedChunk", "__version__"]
