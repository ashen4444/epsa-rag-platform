"""Replaceable retrieval and embedding boundaries."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from epsa_rag.core.models import ParagraphChunk, RetrievalQuery
from epsa_rag.retrieval.models import BackendHit

FloatMatrix = NDArray[np.float32]
FloatVector = NDArray[np.float32]


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Text-to-vector boundary independent of a dense index implementation."""

    @property
    def model(self) -> str:
        """Return the exact embedding model identifier."""

    @property
    def dimensions(self) -> int:
        """Return the expected vector dimension."""

    def embed_documents(self, chunks: Sequence[ParagraphChunk]) -> FloatMatrix:
        """Embed corpus chunks in the supplied order."""

    def embed_query(self, query: RetrievalQuery) -> FloatVector:
        """Embed the exact retrieval query text."""


@runtime_checkable
class RetrievalBackend(Protocol):
    """Ranked-hit boundary consumed by the fusion layer."""

    @property
    def chunk_ids(self) -> tuple[str, ...]:
        """Return indexed chunk IDs in stable index order."""

    def search(self, query: RetrievalQuery, *, top_k: int) -> tuple[BackendHit, ...]:
        """Return deterministic ranked hits for one query."""
