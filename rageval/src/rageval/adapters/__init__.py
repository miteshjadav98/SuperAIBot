"""Adapters — concrete targets that satisfy the ``RAGTarget`` contract.

MockAdapter (offline, keyless) ships in the core install. Networked/optional adapters
(HTTP, SuperBot) live behind extras so the base package stays tiny and portable.
"""

from rageval.adapters.mock import MockAdapter

__all__ = ["MockAdapter"]
