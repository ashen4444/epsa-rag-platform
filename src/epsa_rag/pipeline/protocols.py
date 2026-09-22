"""Replaceable boundaries used by system-level orchestration."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from epsa_rag.core.models import RetrievalQuery
from epsa_rag.retrieval.models import RetrievalResult


@runtime_checkable
class HybridRetrieverProtocol(Protocol):
    """Canonical retriever boundary shared by every evaluated system."""

    def retrieve(
        self,
        query: RetrievalQuery,
        *,
        top_k: int | None = None,
    ) -> RetrievalResult:
        """Return backend-independent ranked paragraph chunks."""
